"""Embed training chips → one pooled vector per obs — the durable training artifact.

UniverSat center-token at the chip's native 10 m resolution: the head trained on
these per-location vectors applies token-for-token at wall-to-wall inference. The
output is a single **CRS-less** Zarr keyed by obs_id (pooled vectors have no
geometry, so one store holds them all regardless of source UTM zone).

Throughput: a ``DataLoader`` with ``num_workers`` prefetches the next batch's chips
off the main thread while the current batch is in the encoder forward — so disk
reads overlap compute instead of alternating with it. The exact same loop runs on
CPU or GPU (``device``); on GPU the prefetch is what keeps the device fed.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import rasterio
import torch
import xarray as xr
from loguru import logger
from pyproj import Transformer
from torch.utils.data import DataLoader, Dataset

from cmrv.embeddings.base import MONTH_DOY, Embedder

# CRS the chips were extracted in (must match cmrv.ingest.chips epsg); point
# coords are reprojected from this to EPSG:4326 for a zone-agnostic point index.
CHIP_CRS = "EPSG:32734"


def _load_stack(uris: list[str], scale: float) -> np.ndarray:
    """Read a ``(T, C, H, W)`` chip stack — DN→reflectance, cloud-NaN filled with 0."""
    frames = []
    for uri in uris:
        with rasterio.open(uri) as src:
            frames.append(np.nan_to_num(src.read().astype("float32") * scale, nan=0.0))
    return np.stack(frames)


class _ChipDataset(Dataset):
    """One ``(T, C, H, W)`` chip stack per obs — workers load these off the main thread."""

    def __init__(self, recs: list, scale: float) -> None:
        self.recs = recs
        self.scale = scale

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int) -> np.ndarray:
        return _load_stack(self.recs[i][-1], self.scale)


def embed_chips(
    manifest_uri: str,
    out_uri: str,
    encoder: Embedder,
    *,
    months: tuple[str, ...] = ("feb", "may", "sep"),
    scale: float = 1.0 / 10000,
    min_valid_frac: float = 0.5,
    batch: int = 8,
    num_workers: int = 4,
) -> str:
    """Embed every obs with all ``months`` present → single Zarr (emb + obs_id/block_id).

    Class-scheme-agnostic (no class_id/fold baked in — those come from make-split and
    vary per experiment); the loader joins them by obs_id at train time.
    """
    man = pd.read_parquet(manifest_uri)
    if min_valid_frac > 0 and "valid_frac" in man.columns:
        man = man[man.groupby("obs_id")["valid_frac"].transform("min") >= min_valid_frac]

    recs = []
    for obs_id, g in man.groupby("obs_id"):
        by_month = dict(zip(g["month_label"], g["chip_uri"], strict=False))
        if all(mo in by_month for mo in months):
            r = g.iloc[0]
            recs.append(
                (
                    obs_id,
                    int(r["block_id"]),
                    float(r["x_utm"]),
                    float(r["y_utm"]),
                    [by_month[mo] for mo in months],
                )
            )
    if not recs:
        raise ValueError("no obs with all configured months present")

    torch.set_num_threads(os.cpu_count() or 1)  # all cores for the forward
    loader = DataLoader(
        _ChipDataset(recs, scale),
        batch_size=batch,
        shuffle=False,  # batches arrive in recs order → aligns with the metadata below
        num_workers=num_workers,
        pin_memory=getattr(encoder, "device", "cpu").startswith("cuda"),
    )
    dvec = np.array([MONTH_DOY[mo] for mo in months])
    embs, done = [], 0
    for stacks in loader:  # (B, T, C, H, W) float32, prefetched by the workers
        s = stacks.numpy()
        embs.append(encoder.embed(s, np.tile(dvec, (s.shape[0], 1))))
        done += s.shape[0]
        logger.info("embedded {}/{}", done, len(recs))
    emb = np.concatenate(embs).astype("float32")

    obs_ids = np.array([r[0] for r in recs])
    block_ids = np.array([r[1] for r in recs])
    xs = np.array([r[2] for r in recs])
    ys = np.array([r[3] for r in recs])
    # Point location → EPSG:4326 (lon/lat): one global CRS so the cube stays a valid
    # point layer even as labels span multiple UTM zones. Chips are extracted in UTM
    # 34S → reproject once; wall-to-wall maps get their CRS from the inference tile.
    lon, lat = Transformer.from_crs(CHIP_CRS, "EPSG:4326", always_xy=True).transform(xs, ys)
    ds = xr.Dataset(
        {"emb": (("obs", "feat"), emb)},
        coords={
            "obs_id": ("obs", obs_ids),
            "block_id": ("obs", block_ids),
            "lon": ("obs", np.asarray(lon, dtype="float64")),
            "lat": ("obs", np.asarray(lat, dtype="float64")),
        },
        attrs={"crs": "EPSG:4326"},
    )
    ds.to_zarr(out_uri, mode="w")
    logger.success("wrote {} embeddings ({}-d) → {}", len(recs), emb.shape[1], out_uri)
    return out_uri
