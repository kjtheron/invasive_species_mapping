"""Embed training chips → one pooled vector per obs — the durable training artifact.

UniverSat center-token at the chip's native 10 m resolution: the head trained on
these per-location vectors applies token-for-token at wall-to-wall inference. The
output is a single **CRS-less** Zarr keyed by obs_id (pooled vectors have no
geometry, so one store holds them all regardless of source UTM zone). The manifest
is streamed in batches so memory stays flat regardless of label count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from loguru import logger

from cmrv.embeddings.base import MONTH_DOY, Embedder


def _load_stack(uris: list[str], scale: float) -> np.ndarray:
    """Read a ``(T, C, H, W)`` chip stack — DN→reflectance, cloud-NaN filled with 0."""
    frames = []
    for uri in uris:
        with rasterio.open(uri) as src:
            frames.append(np.nan_to_num(src.read().astype("float32") * scale, nan=0.0))
    return np.stack(frames)


def embed_chips(
    manifest_uri: str,
    out_uri: str,
    encoder: Embedder,
    *,
    months: tuple[str, ...] = ("feb", "may", "sep"),
    scale: float = 1.0 / 10000,
    min_valid_frac: float = 0.5,
    batch: int = 8,
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

    dvec = np.array([MONTH_DOY[mo] for mo in months])
    obs_ids, block_ids, xs, ys, embs = [], [], [], [], []
    for i in range(0, len(recs), batch):
        chunk = recs[i : i + batch]
        stacks = np.stack([_load_stack(uris, scale) for *_, uris in chunk])
        embs.append(encoder.embed(stacks, np.tile(dvec, (len(chunk), 1))))
        obs_ids += [c[0] for c in chunk]
        block_ids += [c[1] for c in chunk]
        xs += [c[2] for c in chunk]
        ys += [c[3] for c in chunk]
        logger.info("embedded {}/{}", min(i + batch, len(recs)), len(recs))

    emb = np.concatenate(embs).astype("float32")
    # All training chips are extracted in one CRS (UTM 34S), so x_utm/y_utm + the
    # crs attr make this CRS-less feature cube a valid point layer for mapping
    # point predictions. Wall-to-wall maps get their CRS from the inference tile.
    ds = xr.Dataset(
        {"emb": (("obs", "feat"), emb)},
        coords={
            "obs_id": ("obs", np.array(obs_ids)),
            "block_id": ("obs", np.array(block_ids)),
            "x_utm": ("obs", np.array(xs, dtype="float64")),
            "y_utm": ("obs", np.array(ys, dtype="float64")),
        },
        attrs={"crs": "EPSG:32734"},
    )
    ds.to_zarr(out_uri, mode="w")
    logger.success("wrote {} embeddings ({}-d) → {}", len(obs_ids), emb.shape[1], out_uri)
    return out_uri
