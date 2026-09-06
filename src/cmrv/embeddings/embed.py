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
import pandas as pd  # type: ignore
import rasterio  # type: ignore
import torch  # type: ignore
import xarray as xr  # type: ignore
from loguru import logger  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore

from cmrv.embeddings.constants import MONTH_DOY
from cmrv.embeddings.universat import UniverSatEmbedder


def _load_stack(uri: str, n_months: int, scale: float) -> np.ndarray:
    """Read one chip file as ``(T, C, H, W)`` — DN to reflectance, no-data to 0.

    ``ingest-chips`` writes a single uint16 GeoTIFF per (obs, year) whose bands run
    month-major (``feb_B02 ... feb_B12, may_B02 ...``), with 0 as the L2A product's
    own no-data. A masked cloud pixel is therefore already 0, which is exactly the
    fill UniverSat needs — it NaN-poisons on anything else.
    """
    with rasterio.open(uri) as src:
        arr = src.read().astype("float32") * scale  # (T*C, H, W)
    t, c = n_months, arr.shape[0] // n_months
    return arr.reshape(t, c, arr.shape[1], arr.shape[2])


class _ChipDataset(Dataset):
    """One ``(T, C, H, W)`` chip stack per obs — workers load these off the main thread."""

    def __init__(self, recs: list, scale: float) -> None:
        self.recs = recs
        self.scale = scale

    def __len__(self) -> int:
        return len(self.recs)

    def __getitem__(self, i: int) -> np.ndarray:
        _oid, _bid, _lon, _lat, dvec, uri = self.recs[i]
        return _load_stack(uri, len(dvec), self.scale)


def embed_chips(
    manifest_uri: str,
    out_uri: str,
    encoder: UniverSatEmbedder,
    *,
    min_months: int = 3,
    scale: float = 1.0 / 10000,
    min_valid_frac: float = 0.5,
    batch: int = 8,
    num_workers: int = 4,
) -> str:
    """Embed every obs with >= ``min_months`` present → single Zarr (emb + obs_id/block_id).

    Each obs's months come from its own manifest rows, so winter-rainfall (feb/may/sep)
    and summer-rainfall (feb/jun/sep) obs each embed with their own day-of-year vector.
    Class-scheme-agnostic (no class_id/fold baked in — those come from make-split and
    vary per experiment); the loader joins them by obs_id at train time.
    """
    man = pd.read_parquet(manifest_uri)
    if min_valid_frac > 0 and "valid_frac" in man.columns:
        man = man[man["valid_frac"] >= min_valid_frac]

    recs = []  # (obs_id, block_id, lon, lat, dvec, chip_uri)
    for r in man.drop_duplicates("obs_id").itertuples():
        # `months` is written in the same order as the file's band groups, so the
        # day-of-year vector lines up with the stack without re-sorting anything.
        months = str(r.months).split(",")
        if len(months) < min_months:
            continue
        recs.append(
            (
                r.obs_id,
                int(r.block_id),
                float(r.lon),
                float(r.lat),
                [MONTH_DOY[mo] for mo in months],
                r.chip_uri,
            )
        )
    if not recs:
        raise ValueError(f"no obs with >= {min_months} months present")

    torch.set_num_threads(os.cpu_count() or 1)  # all cores for the forward
    loader = DataLoader(
        _ChipDataset(recs, scale),
        batch_size=batch,
        shuffle=False,  # batches arrive in recs order → aligns with dvecs + metadata below
        num_workers=num_workers,
        pin_memory=getattr(encoder, "device", "cpu").startswith("cuda"),
    )
    dvecs = np.array([r[4] for r in recs])  # (N, T) per-obs day-of-year (zone-dependent)
    embs, done = [], 0
    for stacks in loader:  # (B, T, C, H, W) float32, prefetched by the workers
        s = stacks.numpy()
        b = s.shape[0]
        embs.append(encoder.embed(s, dvecs[done : done + b]))
        done += b
        logger.info("embedded {}/{}", done, len(recs))
    emb = np.concatenate(embs).astype("float32")

    obs_ids = np.array([r[0] for r in recs])
    block_ids = np.array([r[1] for r in recs])
    # Point location is stored in the manifest as EPSG:4326 lon/lat (chips are extracted
    # in each group's own native S2 UTM zone), so the cube is one global CRS directly.
    lon = np.array([r[2] for r in recs], dtype="float64")
    lat = np.array([r[3] for r in recs], dtype="float64")
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
