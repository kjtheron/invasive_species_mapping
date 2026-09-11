"""Embed training chips → one pooled vector per obs — the durable training artifact.

UniverSat center-token at the chip's native 10 m resolution: the head trained on
these per-location vectors applies token-for-token at wall-to-wall inference. The
output is a single **CRS-less** Zarr keyed by obs_id (pooled vectors have no
geometry, so one store holds them all regardless of source UTM zone).

Throughput: a ``DataLoader`` with ``num_workers`` prefetches the next batch's chips
off the main thread while the current batch is in the encoder forward — so disk
reads overlap compute instead of alternating with it. The exact same loop runs on
CPU or GPU (``device``); on GPU the prefetch is what keeps the device fed.

Resumable: vectors are checkpointed to ``<out>.parts/`` every ``parts_every`` obs,
written atomically, so a crash, reboot, Ctrl+C or ``pkill`` loses at most one part.
A re-run embeds only the obs no part holds, then consolidates the single Zarr. Parts
stay on disk after success, which also makes the verb incremental for new chips.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore
import rasterio  # type: ignore
import torch  # type: ignore
import xarray as xr  # type: ignore
from loguru import logger  # type: ignore
from torch.utils.data import DataLoader, Dataset  # type: ignore

from cmrv.embeddings.constants import MONTH_DOY
from cmrv.embeddings.universat import UniverSatEmbedder
from cmrv.ingest.chips import _term_saves_work

PARTS_EVERY = 500  # obs per checkpoint part: the most a hard kill can lose (~13 min on CPU)
LOG_SECONDS = 60.0


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


def _signature(encoder, scale: float) -> dict:
    """What must match before two runs' vectors may share one store.

    A part written by another model, grid or DN scale would train the head on a
    mixture with no error anywhere. ``amp`` is deliberately absent: bf16 and fp32
    vectors agree to cosine 0.99998 (measured on real chips), so resuming a run in
    the other precision is safe.
    """
    keys = ("name", "repo", "revision", "patch_size", "output_grid")
    return {**{k: getattr(encoder, k, None) for k in keys}, "scale": scale}


def _part_files(parts: Path) -> list[Path]:
    return sorted(p for p in parts.glob("part_*.npz") if not p.name.endswith(".tmp.npz"))


def _read_part(path: Path, keys: tuple[str, ...]) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in keys}


def _save_part(parts: Path, index: int, pending: list) -> None:
    """Write one part via temp + rename: a kill mid-write leaves a .tmp, never a torn part."""
    recs = [r for chunk, _ in pending for r in chunk]
    tmp, final = parts / f"part_{index:05d}.tmp.npz", parts / f"part_{index:05d}.npz"
    np.savez(
        tmp,
        emb=np.concatenate([v for _, v in pending]).astype("float32"),
        obs_id=np.array([r[0] for r in recs]),
        block_id=np.array([r[1] for r in recs]),
        lon=np.array([r[2] for r in recs], dtype="float64"),
        lat=np.array([r[3] for r in recs], dtype="float64"),
    )
    os.replace(tmp, final)


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
    parts_every: int = PARTS_EVERY,
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

    parts = Path(f"{out_uri}.parts")
    parts.mkdir(parents=True, exist_ok=True)
    sig, meta = _signature(encoder, scale), parts / "meta.json"
    if meta.exists():
        old = json.loads(meta.read_text())
        if old != sig:
            raise ValueError(
                f"{parts} holds vectors from {old}, but this run is {sig}. "
                f"Delete {parts} to re-embed from scratch."
            )
    else:
        meta.write_text(json.dumps(sig))

    files = _part_files(parts)
    done = {str(o) for f in files for o in _read_part(f, ("obs_id",))["obs_id"]}
    todo = [r for r in recs if r[0] not in done]
    logger.info(
        "{} obs: {} already embedded in {} parts, {} to go",
        len(recs),
        len(recs) - len(todo),
        len(files),
        len(todo),
    )
    if todo:
        _embed_todo(todo, encoder, parts, len(files), scale, batch, num_workers, parts_every)
    return _consolidate(parts, recs, out_uri)


def _embed_todo(todo, encoder, parts, next_part, scale, batch, num_workers, parts_every) -> None:
    """Embed ``todo`` in order, checkpointing a part every ``parts_every`` obs."""
    torch.set_num_threads(os.cpu_count() or 1)  # all cores for the forward
    loader = DataLoader(
        _ChipDataset(todo, scale),
        batch_size=batch,
        shuffle=False,  # batches arrive in todo order → slice the metadata alongside
        num_workers=num_workers,
        pin_memory=getattr(encoder, "device", "cpu").startswith("cuda"),
    )
    pending, n_pending, seen = [], 0, 0
    t0 = last_log = time.time()
    try:
        with _term_saves_work():  # pkill takes the same finally as Ctrl+C
            for stacks in loader:
                chunk = todo[seen : seen + len(stacks)]
                vec = encoder.embed(stacks.numpy(), np.array([r[4] for r in chunk]))
                pending.append((chunk, vec))  # one append: ids and vectors cannot drift apart
                n_pending += len(chunk)
                seen += len(chunk)
                if n_pending >= parts_every:
                    _save_part(parts, next_part, pending)
                    next_part, pending, n_pending = next_part + 1, [], 0
                if time.time() - last_log >= LOG_SECONDS:
                    last_log = time.time()
                    spc = (last_log - t0) / seen
                    logger.info(
                        "embedded {}/{} this run — {:.2f} s/chip, eta {:.1f} h",
                        seen,
                        len(todo),
                        spc,
                        spc * (len(todo) - seen) / 3600,
                    )
    finally:
        if pending:  # finished vectors are never thrown away, whatever stopped the loop
            _save_part(parts, next_part, pending)


def _consolidate(parts: Path, recs: list, out_uri: str) -> str:
    """Gather every part into the single obs-keyed Zarr the head reads."""
    keys = ("emb", "obs_id", "block_id", "lon", "lat")
    got = [_read_part(f, keys) for f in _part_files(parts)]
    col = {k: np.concatenate([g[k] for g in got]) for k in keys}
    if len(np.unique(col["obs_id"])) != len(col["obs_id"]):
        raise ValueError(f"{parts} holds duplicate obs_ids — did two embed runs write at once?")
    keep = np.isin(col["obs_id"], [r[0] for r in recs])  # drop obs the manifest no longer has
    if keep.sum() != len(recs):
        raise ValueError(f"{parts} covers {keep.sum()} of {len(recs)} obs — re-run to finish")

    ds = xr.Dataset(
        {"emb": (("obs", "feat"), col["emb"][keep])},
        coords={
            "obs_id": ("obs", col["obs_id"][keep]),
            "block_id": ("obs", col["block_id"][keep]),
            # Point location is stored in the manifest as EPSG:4326 lon/lat (chips are
            # extracted in each group's own native S2 UTM zone), so the cube is one
            # global CRS directly.
            "lon": ("obs", col["lon"][keep]),
            "lat": ("obs", col["lat"][keep]),
        },
        attrs={"crs": "EPSG:4326"},
    )
    ds.to_zarr(out_uri, mode="w")
    logger.success("wrote {} embeddings ({}-d) → {}", int(keep.sum()), col["emb"].shape[1], out_uri)
    return out_uri
