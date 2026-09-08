"""Sparse training-chip extraction — temporally aligned to label dates.

For each label point, writes **one** GeoTIFF per observation year holding every
configured season month as bands::

    {out_prefix}/{obs_id}/{year}.tif   uint16, nodata 0, (T*C, chip_px, chip_px)
                                       band names: feb_B02, feb_B03, ... sep_B12
    {out_prefix}/manifest.parquet      one row per (obs_id, year)

uint16 with nodata 0 is the L2A product's own convention — surface reflectance
scaled by 10000, 0 reserved as no-data — so nothing is rescaled or reinterpreted
on the way to disk. Divide by 10000 for reflectance.

How a chip is built, and why in that order:

1. **Search** the label's coarse cell once per month window, memoised across every
   chip in the cell (:func:`cmrv.ingest.s2.search_cell`).
2. **Screen on geometry** — drop the dates whose real footprints do not cover the
   chip. Free, so it runs before any pixel is read.
3. **Screen on SCL** — read one uint8 band per candidate date and keep the date
   that is clearest over *this chip*. ``eo:cloud_cover`` describes a 110 km MGRS
   tile and cannot answer that question.
4. **Read the bands** for the winning date only.

Steps 2 and 3 are the whole speed story. A Planetary-Computer S2 COG has 512x512
internal blocks, so a 128 px chip costs one block read per asset per scene
whatever the chunking is. The old code medianed up to 20 scenes x 11 assets
(~110 MB of source reads) to write a 660 KB chip, and chose those 20 scenes by a
tile-level cloud number. Screening costs one uint8 asset per candidate date and
then spends the bands once.

A chip is written only when **every** configured month succeeds, so ``T`` is
uniform across the training set and the embedding stage needs no masking. An
observation that fails is retried once at ``year - 1``.

CRS handling:
    Labels arrive in EPSG:4326. Each chip is built on a grid snapped to the 10 m
    Sentinel-2 pixel grid of its **own** native UTM zone (``utm_epsg`` of the
    label), so nothing is resampled across zones. The manifest stores lon/lat in
    EPSG:4326, keeping it CRS-agnostic. Spatial-block grids stay in SA Albers
    (``cmrv.aoi``).
"""

from __future__ import annotations

import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path

import geopandas as gpd  # type: ignore
import numpy as np
import pandas as pd  # type: ignore
import pyproj  # type: ignore
import rasterio  # type: ignore
from loguru import logger  # type: ignore
from rasterio.crs import CRS  # type: ignore
from shapely.geometry import box  # type: ignore
from shapely.ops import unary_union  # type: ignore

from cmrv.aoi import SA_ALBERS, utm_epsg
from cmrv.ingest import s2
from cmrv.io import read_parquet_df, write_parquet_df

CHIP_PX = 128
RESOLUTION_M = 10
BLOCK_KM = 10

# One acquisition date per month. There is no median to hide behind, which is
# why the SCL screen picks the date and why MIN_VALID_FRAC is high.
MAX_DATES = 1
# Fraction of the chip a date's footprints must cover before anything is read.
MIN_COVERAGE = 0.98
# Fraction of finite pixels a finished chip-month must reach. 0.85 matches the
# inspiration scripts. The old 0.5 was tolerable only because a 20-scene median
# repaired most holes; with one date it would ship half-cloud chips.
MIN_VALID_FRAC = 0.85
# Ceiling on candidate dates put through the SCL screen. A padded month over SA
# yields ~10-25 dates; the cap only bounds a pathological window.
SCREEN_MAX_DATES = 24
# Side of the square within which chips share a "clearest date" answer.
# Screening is roughly half the bytes a chip costs, and it asks the same
# question of every label. Measured on the real 93,348 thinned labels: at 2 km,
# 69% of labels have a neighbour that already answered it (47% at 1 km, 87% at
# 5 km). 2 km is the conservative pick — cloud edges inside a square make the
# remembered date wrong, and the wider the square the more often that happens.
# A wrong answer is cheap, not dangerous: the finished chip is still checked,
# and a failure falls back to a full screen.
DATE_CELL_M = 2000.0
# Concurrent asset reads inside one chip. The reads are pure network latency, so
# concurrency here is nearly free. Measured cold on Planetary Computer over 8
# scattered SA sites: screening 48 dates took 71.8 s serial against 19.2 s at
# pool=4; a band load took 19.7 s serial against 7.3 s. Returns flatten past ~8.
# Total in-flight sockets is max_workers x this. 20 workers x 8 put 160 streams
# on a 39 Mbit/s link — 30 KB/s each, slow enough that the stall watchdog killed
# healthy reads. With ~270 ms latency and a 4.9 MB/s ceiling, a few dozen streams
# already saturate it; past that, extra concurrency buys only contention.
READ_POOL = 4
# Default +/-days around each calendar month. Widening roughly doubles the
# candidate-date pool, which is what gives the SCL screen something to choose.
WINDOW_PADDING_DAYS = 15
# The manifest is rewritten after this many new rows, OR after this many seconds,
# whichever comes first. The time bound is the one that matters: at ~0.15 chips/s
# a count of 500 alone leaves a 55-minute window in which a hard stop loses every
# row since the last flush. Those chips are still on disk and _adopt_orphan_chips
# reads them back, but that is a repair, not a plan. One small Parquet write a
# minute is nothing against 29 MB per chip.
FLUSH_EVERY = 500
FLUSH_SECONDS = 60.0
# After a stop is requested, how long to let the chips already in flight finish
# and bank their rows. Cancelling the queue does not stop a worker mid-chip, and
# a chip takes ~15 s, so without this those rows land on disk with no manifest
# row and have to be adopted on the next start.
DRAIN_SECONDS = 30.0


def _raise_interrupt(_signum, _frame):
    raise KeyboardInterrupt


@contextmanager
def _term_saves_work():
    """Make SIGTERM behave like Ctrl+C for the duration of the block.

    Python's default SIGTERM handler ends the process outright: no ``finally``,
    so every manifest row since the last flush is lost and its chips have to be
    re-adopted from disk on the next start. Ctrl+C already ran the save path, so
    ``pkill`` and Ctrl+C behaved completely differently with nothing to say so.

    Restores the previous handler on exit. No-ops off the main thread, where
    ``signal.signal`` raises, so tests and subprocesses can use it safely.
    """
    try:
        prev = signal.signal(signal.SIGTERM, _raise_interrupt)
    except ValueError:  # not the main thread
        yield
        return
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, prev)


# ---------------------------------------------------------------------------
# Spatial blocks
# ---------------------------------------------------------------------------


def build_spatial_blocks(
    aoi: gpd.GeoDataFrame,
    block_km: float = BLOCK_KM,
    crs: str = SA_ALBERS,
    min_overlap_frac: float = 0.001,
) -> gpd.GeoDataFrame:
    """Create a coarse grid of spatial blocks over the AOI.

    Blocks are the atomic unit for stratified train/val/test splitting. They no
    longer batch STAC queries — the work unit is one chip, and searches are
    memoised per coarse geographic cell in :mod:`cmrv.ingest.s2`.
    """
    aoi_m = aoi.to_crs(crs)
    minx, miny, maxx, maxy = aoi_m.total_bounds
    step = block_km * 1000.0
    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)
    cells = [box(x, y, x + step, y + step) for x in xs for y in ys]  # type: ignore
    grid = gpd.GeoDataFrame({"block_id": range(len(cells))}, geometry=cells, crs=crs)
    aoi_union = unary_union(aoi_m.geometry)
    overlap = grid.intersection(aoi_union).area
    keep = grid[overlap > min_overlap_frac * step * step].reset_index(drop=True)
    keep["block_id"] = range(len(keep))
    logger.info("built {} spatial blocks ({} km) over AOI", len(keep), block_km)
    return keep


# ---------------------------------------------------------------------------
# Stratified spatial split
# ---------------------------------------------------------------------------


def stratified_spatial_split(
    labels: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    species_col: str = "species_normalized",
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
    existing_block_folds: dict[int, str] | None = None,
) -> tuple[gpd.GeoDataFrame, dict[int, str]]:
    """Assign each label a ``fold`` (train/val/test) via spatial blocks.

    **Iterative stratification** (Sechidis et al. 2011): each block is a
    multi-label item (its per-class label counts); blocks are assigned rarest-class
    first, each to the fold most short of that class. Whole blocks stay intact (no
    spatial leakage), yet every class with enough blocks is spread ~train/val/test
    instead of piling into train. Stratify on ``species_col`` — pass ``class_id``
    to balance the actual training target.

    ``existing_block_folds`` locks those blocks (debiting their share from the
    quotas); only new blocks are assigned. Returns ``(labels, block_to_fold)``.
    """
    folds = {"train": train_frac, "val": val_frac, "test": 1 - train_frac - val_frac}

    if "block_id" in labels.columns:
        labels = labels.drop(columns=["block_id"])
    blocks_wgs = blocks[["block_id", "geometry"]].to_crs("EPSG:4326")
    labels = gpd.sjoin(labels, blocks_wgs, how="inner", predicate="within")
    if "index_right" in labels.columns:
        labels = labels.drop(columns=["index_right"])

    # block x class count matrix; desired[fold] = each fold's remaining per-class quota
    mat = labels.groupby(["block_id", species_col]).size().unstack(fill_value=0)
    desired = {f: mat.sum(axis=0) * fr for f, fr in folds.items()}
    desired_total = {f: float(mat.values.sum()) * fr for f, fr in folds.items()}
    block_to_fold: dict[int, str] = {}

    # Lock existing blocks first, debiting their labels from the quotas.
    existing_block_folds = existing_block_folds or {}
    locked = set(existing_block_folds) & set(mat.index)
    for bid in locked:
        f = existing_block_folds[bid]
        block_to_fold[bid] = f
        desired[f] = desired[f] - mat.loc[bid]  # type: ignore
        desired_total[f] -= float(mat.loc[bid].sum())  # type: ignore
    if locked:
        logger.info(
            "spatial split: {} blocks locked, {} to assign", len(locked), len(mat) - len(locked)
        )

    # Iterative stratification: take the rarest remaining class, and send each of
    # its blocks to the fold most short of that class (tie -> most short overall ->
    # random). Rare classes placed first can't get starved by the common ones.
    rng = np.random.default_rng(seed)
    remaining = mat.drop(index=list(locked))
    while len(remaining):
        active = remaining.sum(axis=0).pipe(lambda s: s[s > 0])
        if active.empty:  # label-less blocks (shouldn't survive thinning) -> emptiest fold
            for bid in remaining.index:
                block_to_fold[bid] = max(desired_total, key=desired_total.get)  # type: ignore
            break
        c = active.idxmin()
        bids = remaining.index[remaining[c] > 0].tolist()
        rng.shuffle(bids)
        for bid in bids:
            row = remaining.loc[bid]
            best = max(folds, key=lambda f: (desired[f][c], desired_total[f], rng.random()))
            block_to_fold[bid] = best
            desired[best] = desired[best] - row
            desired_total[best] -= float(row.sum())
        remaining = remaining.drop(index=bids)

    labels["fold"] = labels["block_id"].map(block_to_fold)

    for f in folds:
        n = (labels["fold"] == f).sum()
        n_sp = labels.loc[labels["fold"] == f, species_col].nunique()
        logger.info("fold {}: {} labels ({:.1f}%), {} species", f, n, 100 * n / len(labels), n_sp)

    return labels, block_to_fold


# ---------------------------------------------------------------------------
# Temporal windowing
# ---------------------------------------------------------------------------


def temporal_windows(
    year: int,
    months_cfg: list[dict],
    padding_days: int = WINDOW_PADDING_DAYS,
) -> list[dict]:
    """Map month templates onto a label's observation year.

    Every configured season month is taken from the label year itself — no
    cross-year offset. Each window is widened by ``padding_days`` on each side, so
    the SCL screen has more dates to choose the clearest from.
    """
    windows = []
    for m in months_cfg:
        start = pd.to_datetime(f"{year}{m['start'][4:]}") - pd.Timedelta(days=padding_days)
        end = pd.to_datetime(f"{year}{m['end'][4:]}") + pd.Timedelta(days=padding_days)
        windows.append(
            {
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "label": m["label"],
            }
        )
    return windows


# ---------------------------------------------------------------------------
# One chip
# ---------------------------------------------------------------------------


_TRANSFORMERS: dict[int, pyproj.Transformer] = {}
_TF_LOCK = threading.Lock()


def _to_utm(lon: float, lat: float, epsg: int) -> tuple[float, float]:
    """Project one lon/lat into a UTM zone, reusing the Transformer per zone."""
    with _TF_LOCK:
        tf = _TRANSFORMERS.get(epsg)
        if tf is None:
            tf = _TRANSFORMERS[epsg] = pyproj.Transformer.from_crs(4326, epsg, always_xy=True)
    return tf.transform(lon, lat)


# ponytail: plain dict, never evicted. One date per (square, month, year) —
# about 90k entries at ~200 bytes, so ~18 MB for a national run. Add eviction
# only if a run ever holds more than that.
_DATE_CACHE: dict[tuple, object] = {}
_DATE_LOCK = threading.Lock()


def _date_key(gbox, month_label: str, year: int, cell_m: float) -> tuple | None:
    """Key a remembered date by square, month and year, or ``None`` if disabled.

    The square is measured in the chip's own UTM zone, and the zone is part of
    the key, so two chips in different zones can never share an answer by
    accident. ``cell_m <= 0`` turns the sharing off — that is the documented
    switch, so it must not divide by zero.
    """
    if cell_m <= 0:
        return None
    bb = gbox.boundingbox
    return (gbox.crs.epsg, int(bb.left // cell_m), int(bb.bottom // cell_m), month_label, year)


def _recall_date(key: tuple | None):
    if key is None:
        return None
    with _DATE_LOCK:
        return _DATE_CACHE.get(key)


def _remember_date(key: tuple | None, day) -> None:
    if key is None:
        return
    with _DATE_LOCK:
        _DATE_CACHE[key] = day


def _month_chip(
    lon: float,
    lat: float,
    gbox,
    win: dict,
    bands: list[str],
    year: int,
    *,
    max_dates: int,
    min_coverage: float,
    min_valid_frac: float,
    screen_max_dates: int,
    read_pool: int,
    date_cell_m: float = DATE_CELL_M,
) -> tuple[np.ndarray, float] | str:
    """One month's chip on *gbox*, or a reason string saying why there is none.

    Three passes, in order, each falling through to the next:

    0. **Remembered date.** If a neighbour in the same ``date_cell_m`` square
       already worked out the clearest date for this month, reuse it and read the
       bands straight away. This skips the SCL screen, which is roughly half of
       what a chip costs to download.
    1. **Full screen.** Read SCL for every candidate date and pick the clearest,
       then remember it for the neighbours.
    2. **Full screen after a re-sign**, for a read that failed.

    A remembered date can be wrong where a cloud edge cuts through the square.
    That is caught, not trusted: the finished chip still has to pass
    ``min_valid_frac`` and the centre-pixel check, and a failure drops through to
    pass 1. So a wrong answer costs one wasted read, never a bad chip.

    Reason strings are counted by the caller, so a run reports *why* it lost
    chips rather than only how many.
    """
    cell = s2.cell_of(lon, lat)
    chip_wgs84 = gbox.extent.to_crs("EPSG:4326").geom
    key = _date_key(gbox, win["label"], year, date_cell_m)
    reason = "no_items"

    for attempt in (0, 1, 2):
        try:
            items = s2.search_cell(cell, win["start"], win["end"])
            if not items:
                return "no_items"
            cov = s2.covering_items(items, chip_wgs84, min_coverage)
            if not cov:
                return "no_coverage"

            best = None
            if attempt == 0:
                day = _recall_date(key)
                if day is not None:
                    best = [i for i in cov if i.datetime.date() == day]
                if not best:
                    continue  # nothing remembered, or it does not cover this chip

            if best is None:
                # Cap the screening pool. Ordering by the scene-level cloud
                # property is not a filter — it only decides which dates get
                # screened first when a window is unusually crowded. The
                # per-chip SCL fraction still decides.
                groups = s2.by_solar_day(cov)
                if len(groups) > screen_max_dates:
                    groups.sort(
                        key=lambda g: min(i.properties.get("eo:cloud_cover", 100.0) for i in g)
                    )
                    groups = groups[:screen_max_dates]
                scored = s2.clear_fraction_per_date(
                    [i for g in groups for i in g], gbox, pool=read_pool
                )
                if not scored:
                    return "screen_empty"
                scored.sort(key=lambda p: -p[0])
                if scored[0][0] < min_valid_frac:
                    return f"screen_clear={scored[0][0]:.2f}"
                best = [i for _f, grp in scored[:max_dates] for i in grp]

            arr = s2.load_composite(best, gbox, bands, pool=read_pool).values

        except Exception as exc:
            # The likeliest cause is an expired SAS token. Dropping the cached
            # search forces a fresh, freshly-signed one — which is the only thing
            # that helps, and is what the old full-tile download fallback was
            # flailing at.
            if attempt < 2:
                s2.drop_cell(cell, win["start"], win["end"])
                reason = "read_error"
                continue
            logger.warning(
                "read failed twice for {} after a re-sign: {}: {}",
                win["label"], type(exc).__name__, str(exc)[:120],
            )
            return "read_error"

        frac = s2.valid_fraction(arr)
        if frac < min_valid_frac:
            reason = f"low_valid={frac:.2f}"
        elif not s2.center_is_valid(arr):
            reason = "center_cloudy"
        else:
            if attempt > 0:  # a date this chip screened itself is worth sharing
                _remember_date(key, best[0].datetime.date())
            return arr, frac

        if attempt == 0:
            continue  # the remembered date does not work here — screen properly
        return reason

    return reason


def _write_chip(
    arr: np.ndarray,
    months: list[str],
    bands: list[str],
    gbox,
    path: str | Path,
    year: int,
) -> None:
    """Write a ``(T, C, H, W)`` float32 stack as one uint16 GeoTIFF.

    NaN becomes 0, which is L2A's own no-data value, so the embedding stage's
    existing ``nan_to_num(..., 0.0)`` fill is unchanged in effect.
    """
    t, c, h, w = arr.shape
    flat = np.nan_to_num(arr.reshape(t * c, h, w), nan=0.0)
    flat = np.clip(np.rint(flat), 0, 65535).astype("uint16")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".{threading.get_ident()}.part.tif")
    try:
        with rasterio.open(
            str(tmp),
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=t * c,
            dtype="uint16",
            crs=CRS.from_epsg(gbox.crs.epsg),
            transform=gbox.transform,
            nodata=0,
            compress="deflate",
            predictor=2,
            tiled=False,
        ) as dst:
            dst.write(flat)
            dst.update_tags(MONTHS=",".join(months), YEAR=str(year), SCALE_FACTOR="10000")
            for i, name in enumerate([f"{m}_{b}" for m in months for b in bands], start=1):
                dst.set_band_description(i, name)
        tmp.replace(p)  # atomic: a crash mid-write leaves no half file to resume from
    finally:
        tmp.unlink(missing_ok=True)


def _process_obs(row, months_cfg: list[dict], bands: list[str], out_prefix: str, opt: dict):
    """Build and write one observation's chip. Returns a manifest row or a reason.

    ``row`` comes from ``itertuples`` after :func:`_run_pool` has renamed ``_year``
    and ``_zone`` — a leading underscore is not addressable on a namedtuple.
    """
    year = int(row.obs_year)
    epsg = utm_epsg(row.lon, row.lat)
    x, y = _to_utm(row.lon, row.lat, epsg)
    gbox = s2.point_geobox(x, y, epsg, opt["chip_px"], opt["resolution_m"])

    frames, fracs, months = [], [], []
    for win in temporal_windows(year, months_cfg, opt["padding_days"]):
        res = _month_chip(row.lon, row.lat, gbox, win, bands, year, **opt["month"])
        if isinstance(res, str):
            return f"{win['label']}:{res}"
        frames.append(res[0])
        fracs.append(res[1])
        months.append(win["label"])

    rel = f"{row.obs_id}/{year}.tif"
    _write_chip(np.stack(frames), months, bands, gbox, f"{out_prefix}/{rel}", year)
    return {
        "obs_id": row.obs_id,
        "species": row.species_normalized,
        "year": year,
        "block_id": int(row.block_id),
        "lon": float(row.lon),
        "lat": float(row.lat),
        "chip_uri": f"{out_prefix}/{rel}",
        "months": ",".join(months),
        "n_months": len(months),
        "valid_frac": float(min(fracs)),
        "utm_epsg": int(epsg),
    }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _load_existing_manifest(out_prefix: str) -> pd.DataFrame:
    """Load ``manifest.parquet``, dropping rows whose chip file has gone.

    A crash between the chip write and the manifest flush leaves the file on disk
    and no row; the next run simply re-chips it. The reverse — a row with no file
    — is what this prunes, because the incremental skip must never claim a chip
    that is not there.
    """
    p = Path(f"{out_prefix}/manifest.parquet")
    if not p.exists():
        return pd.DataFrame()
    m = pd.read_parquet(p)
    if m.empty:
        return m
    alive = m["chip_uri"].map(lambda u: Path(u).exists())
    if not alive.all():
        logger.warning("manifest: {} rows point at a missing chip — dropped", int((~alive).sum()))
    return m[alive].reset_index(drop=True)


def _adopt_orphan_chips(
    labels: gpd.GeoDataFrame,
    out_prefix: str,
    expected_months: dict[str, set[str]],
) -> list[dict]:
    """Re-adopt chips that are on disk but absent from the manifest.

    A kill between a chip write and the next manifest flush leaves the file on
    disk with no row. The incremental skip reads the *manifest*, so those chips
    would be fetched a second time — up to ``FLUSH_EVERY`` of them per stop.

    Downloading is the expensive thing here, not reading a local file: a chip
    costs ~29 MB over the wire against a ~700 KB read from disk. So check the
    file instead. Its ``MONTHS`` tag says which calendar it was cut on, and
    ``valid_frac`` is recomputed from the pixels rather than trusted from a tag,
    so a chip written by any earlier version can still be adopted.

    A chip whose months do not match the label's current zone is left alone —
    it is stale, and ``_reconcile_manifest`` deletes it.
    """
    rows: list[dict] = []
    cols = zip(
        labels["obs_id"], labels["species_normalized"], labels["block_id"],
        labels["lon"], labels["lat"], strict=True,
    )
    for obs_id, species, block_id, lon, lat in cols:
        d = Path(out_prefix) / str(obs_id)
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.tif")):  # glob, so a Y-1 fallback chip is found too
            try:
                with rasterio.open(f) as src:
                    months = [m for m in src.tags().get("MONTHS", "").split(",") if m]
                    if not months or set(months) != expected_months.get(obs_id, set()):
                        break
                    arr = src.read()
                    epsg = src.crs.to_epsg() if src.crs else None
            except Exception:
                logger.debug("unreadable orphan chip {} — will re-chip", f)
                break
            c = arr.shape[0] // len(months)
            # 0 is L2A's own no-data and the value masked pixels were filled
            # with, so this reproduces s2.valid_fraction exactly.
            valid = min(float((arr[i * c : (i + 1) * c] > 0).all(axis=0).mean())
                        for i in range(len(months)))
            rows.append(
                {
                    "obs_id": obs_id, "species": species, "year": int(f.stem),
                    "block_id": int(block_id), "lon": float(lon), "lat": float(lat),
                    "chip_uri": str(f), "months": ",".join(months),
                    "n_months": len(months), "valid_frac": valid,
                    "utm_epsg": int(epsg) if epsg else 0,
                }
            )
            break
    if rows:
        logger.info(
            "adopted {} chips found on disk but missing from the manifest "
            "(a prior run was stopped between a chip write and a flush)", len(rows)
        )
    return rows


def _load_block_folds(uri: str) -> dict[int, str] | None:
    """Load persisted block->fold mapping if it exists."""
    try:
        p = Path(uri)
        if not p.exists():
            raise FileNotFoundError
        df = pd.read_parquet(p)
        mapping = dict(zip(df["block_id"], df["fold"], strict=True))
        logger.info("loaded block folds: {} blocks from {}", len(mapping), uri)
        return mapping
    except Exception:
        return None


def _save_block_folds(block_to_fold: dict[int, str], uri: str) -> None:
    """Persist block->fold mapping to parquet."""
    df = pd.DataFrame(sorted(block_to_fold.items()), columns=["block_id", "fold"])
    write_parquet_df(df, uri)
    logger.info("saved block folds: {} blocks -> {}", len(df), uri)


def _reconcile_manifest(
    manifest: pd.DataFrame,
    keep_obs: set[str],
    manifest_uri: str,
    expected_months: dict[str, set[str]] | None = None,
) -> pd.DataFrame:
    """Delete chips that are no longer wanted, then rewrite the manifest.

    ``ingest-chips`` is additive, so two kinds of cruft accumulate:

    * **Dropped obs** — a re-thinning picked a different one-per-cell
      representative, leaving the old obs's chip behind.
    * **Regrouped obs** — a label moved to a different rainfall zone, so its chip
      holds the old calendar's months. One file per obs makes this a plain set
      comparison; there is no longer a per-month sub-case.

    Disk-only — nothing is re-downloaded here.
    """
    if manifest.empty:
        return manifest
    stale_mask = ~manifest["obs_id"].isin(keep_obs)
    if expected_months:
        # An obs absent from expected_months is not managed by this call (e.g. a
        # --species subset) — leave it alone and let keep_obs decide.
        wrong = pd.Series(
            [
                o in expected_months and set(str(m).split(",")) != expected_months[o]
                for o, m in zip(manifest["obs_id"], manifest["months"], strict=True)
            ],
            index=manifest.index,
        )
        stale_mask = stale_mask | wrong
    stale = manifest[stale_mask]
    if stale.empty:
        return manifest
    for uri in stale["chip_uri"]:
        Path(uri).unlink(missing_ok=True)
    for d in {Path(u).parent for u in stale["chip_uri"]}:
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    kept = manifest[~stale_mask].reset_index(drop=True)
    write_parquet_df(kept, manifest_uri)
    logger.success(
        "reconcile: pruned {} stale chips -> {} chips remain", len(stale), len(kept)
    )
    return kept


# ---------------------------------------------------------------------------
# Thinning
# ---------------------------------------------------------------------------


def thin_labels(
    labels: gpd.GeoDataFrame,
    thin_m: float,
    epsg: int = 32734,
    species_col: str = "species_normalized",
) -> gpd.GeoDataFrame:
    """Keep one label per species per ``thin_m`` grid cell (run before extraction).

    Snaps each label's UTM coordinate to a ``thin_m`` grid and keeps a single
    label per ``(species, cell)`` — removing near-duplicates the embedding cannot
    distinguish (20 m = 2 native S2 pixels) *before* any imagery is fetched.

    The survivor is the smallest ``obs_id`` in the cell, so thinning is
    **deterministic and stable**: independent of input row order and of which
    other species are loaded. A re-run with a different ``--species`` set keeps
    the same points for a given species, so chips are never orphaned.
    """
    if thin_m <= 0 or labels.empty:
        return labels
    utm = labels.to_crs(f"EPSG:{epsg}")
    cell_x = (utm.geometry.x // thin_m).astype(int).to_numpy()
    cell_y = (utm.geometry.y // thin_m).astype(int).to_numpy()
    keep = (
        labels.assign(_cx=cell_x, _cy=cell_y)
        .sort_values("obs_id", kind="stable")
        .drop_duplicates(subset=[species_col, "_cx", "_cy"], keep="first")
        .drop(columns=["_cx", "_cy"])
    )
    logger.info("spatial thin ({}m): {} -> {} labels", int(thin_m), len(labels), len(keep))
    return keep


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def extract_training_chips(
    labels: gpd.GeoDataFrame,
    months_by_zone: dict[str, list[dict]],
    bands: list[str],
    out_prefix: str,
    *,
    default_zone: str = "winter_rainfall",
    chip_px: int = CHIP_PX,
    resolution_m: int = RESOLUTION_M,
    max_dates: int = MAX_DATES,
    min_coverage: float = MIN_COVERAGE,
    min_valid_frac: float = MIN_VALID_FRAC,
    screen_max_dates: int = SCREEN_MAX_DATES,
    padding_days: int = WINDOW_PADDING_DAYS,
    read_pool: int = READ_POOL,
    date_cell_m: float = DATE_CELL_M,
    default_year: int = 2023,
    max_workers: int = 8,
    year_fallback: bool = True,
    reconcile: bool = True,
) -> pd.DataFrame:
    """Extract temporally-aligned training chips for all labels.

    One work unit is **one observation**, not a block of them. The old code
    grouped labels into 10 km blocks so many chips could share one composite; the
    measured median group held 2 labels, so the sharing rarely paid, while the
    machinery that bounded its memory (sub-cell batching, a compute-size cap) was
    permanent. Uniform work units also make a failure cheap: a lost unit is one
    chip, not up to 361 of them.

    *labels* must already carry ``block_id`` (spatial-join with
    :func:`build_spatial_blocks`). No fold assignment happens here — that is
    ``make_split``, at training time.

    Incremental: an obs whose chip file exists **and** whose months match its
    current zone is skipped. A zone change therefore re-chips, because the month
    set differs.

    Year alignment: a pre-existing ``_year`` column is honoured; otherwise
    ``_year = event_date.year`` (NaT -> ``default_year``). Observations that
    produce no chip are retried once at ``_year - 1``, which recovers labels whose
    target year had a persistently cloudy season.
    """
    if "block_id" not in labels.columns:
        raise ValueError("labels must have a 'block_id' column — spatial-join with blocks first")

    labels = labels.copy()
    if "_year" not in labels.columns:
        ed = pd.to_datetime(labels["event_date"], errors="coerce")
        labels["_year"] = ed.dt.year.fillna(default_year).astype(int)
    if "_zone" not in labels.columns:
        labels["_zone"] = default_zone
    # Every zone must have a month set. `cmrv ingest-chips` already raises on a
    # province with no zone; this catches the other half — a zone with no calendar
    # — before any imagery is fetched. Falling back to another zone's months is
    # never right.
    unknown = sorted(set(labels["_zone"]) - set(months_by_zone))
    if unknown:
        raise ValueError(
            f"no month set for zone(s) {unknown}; "
            f"add them to `months_by_zone` (have: {sorted(months_by_zone)})"
        )
    if "lon" not in labels.columns:
        labels["lon"] = labels.geometry.x
        labels["lat"] = labels.geometry.y

    # Canonical set for this (top-level) call, captured before the incremental
    # filter mutates `labels`. Used at the end to prune chips for obs a
    # re-thinning has since dropped.
    canonical_obs = set(labels["obs_id"])
    zone_months = {z: {m["label"] for m in ms} for z, ms in months_by_zone.items()}
    expected_months: dict[str, set[str]] = {
        oid: zone_months[z] for oid, z in zip(labels["obs_id"], labels["_zone"], strict=True)
    }

    manifest_uri = f"{out_prefix}/manifest.parquet"
    existing = _load_existing_manifest(out_prefix)

    if not existing.empty:
        have = {
            oid: set(str(ms).split(","))
            for oid, ms in zip(existing["obs_id"], existing["months"], strict=True)
        }
        done = {o for o, ms in have.items() if expected_months.get(o, set()) == ms}
        regrouped = sum(1 for o, ms in have.items() if o in expected_months and ms != expected_months[o])
        n_before = len(labels)
        labels = labels[~labels["obs_id"].isin(done)]
        logger.info(
            "incremental: {} obs already chipped, {} to process", n_before - len(labels), len(labels)
        )
        if regrouped:
            logger.info("regrouped: {} obs have a chip on the wrong month calendar", regrouped)

    # Chips whose manifest row was lost to a kill: adopt them from disk rather
    # than paying for the download again.
    orphans = _adopt_orphan_chips(labels, out_prefix, expected_months)
    if orphans:
        existing = _flush(existing, orphans, manifest_uri)
        labels = labels[~labels["obs_id"].isin({r["obs_id"] for r in orphans})]

    if labels.empty:
        logger.success("all labels already chipped — nothing to do")
        if year_fallback and reconcile:
            existing = _reconcile_manifest(existing, canonical_obs, manifest_uri, expected_months)
        return existing

    opt = {
        "chip_px": chip_px,
        "resolution_m": resolution_m,
        "padding_days": padding_days,
        "month": {
            "max_dates": max_dates,
            "min_coverage": min_coverage,
            "min_valid_frac": min_valid_frac,
            "screen_max_dates": screen_max_dates,
            "read_pool": read_pool,
            "date_cell_m": date_cell_m,
        },
    }

    rows = _run_pool(labels, months_by_zone, bands, out_prefix, opt, max_workers, existing,
                     manifest_uri)
    manifest = _flush(existing, rows, manifest_uri)

    # Year fallback: obs that produced no chip get one retry at _year - 1. Single
    # step (no Y-2), and disabled on the recursive call so retries cannot run away.
    if year_fallback:
        failed = set(labels["obs_id"]) - set(manifest["obs_id"] if not manifest.empty else [])
        if failed:
            fb = labels[labels["obs_id"].isin(failed)].copy()
            fb["_year"] = fb["_year"] - 1
            logger.info("year fallback: retrying {} obs at Y-1", len(failed))
            manifest = extract_training_chips(
                labels=fb,
                months_by_zone=months_by_zone,
                bands=bands,
                out_prefix=out_prefix,
                default_zone=default_zone,
                chip_px=chip_px,
                resolution_m=resolution_m,
                max_dates=max_dates,
                min_coverage=min_coverage,
                min_valid_frac=min_valid_frac,
                screen_max_dates=screen_max_dates,
                padding_days=padding_days,
                read_pool=read_pool,
                date_cell_m=date_cell_m,
                default_year=default_year,
                max_workers=max_workers,
                year_fallback=False,
                reconcile=False,
            )

    if year_fallback and reconcile:
        manifest = _reconcile_manifest(manifest, canonical_obs, manifest_uri, expected_months)
    return manifest


def _run_pool(labels, months_by_zone, bands, out_prefix, opt, max_workers, existing, manifest_uri):
    """Run every label through the thread pool, flushing the manifest periodically.

    **One work unit is one square, not one chip.** A square's chips run one after
    another inside a single worker, which is what makes the remembered-date cache
    pay: the first chip in the square works out the clearest month, and its
    neighbours read the answer instead of downloading the cloud maps again.

    Handing each chip to its own worker looks more parallel and is worse. With 8
    workers, all of a square's chips start before any of them has an answer to
    share, so every one screens and the cache never gets a hit — measured at
    0.99x, i.e. nothing. Sequencing within a square also keeps GDAL's block cache
    warm, because neighbours read the same 512-block.

    Squares are visited in geographic order, so consecutive workers stay near
    each other and the per-cell STAC search cache is near-pure hits.
    """
    # itertuples() renames any column whose name starts with "_" to a positional
    # placeholder, so `row._zone` silently resolves to nothing. Rename first; the
    # public column names stay `_year` / `_zone` for callers.
    work = labels.rename(columns={"_year": "obs_year", "_zone": "zone"}).copy()
    cell_m = opt["month"]["date_cell_m"] or opt["chip_px"] * opt["resolution_m"]
    sq, geo = [], []
    for lo, la in zip(work["lon"], work["lat"], strict=True):
        e = utm_epsg(lo, la)
        x, y = _to_utm(lo, la, e)
        sq.append((e, int(x // cell_m), int(y // cell_m)))
        geo.append(s2.cell_of(lo, la))
    work["cell"], work["sq"] = geo, sq

    squares: dict[tuple, list] = {}
    for row in work.sort_values(["cell", "obs_year"], kind="stable").itertuples():
        squares.setdefault((row.cell, row.sq), []).append(row)
    units = list(squares.values())
    n_todo = len(work)
    logger.info(
        "extracting {} chips in {} squares, {} workers x {} read threads",
        n_todo, len(units), max_workers, opt["month"]["read_pool"],
    )

    rows: list[dict] = []
    drops: dict[str, int] = {}
    lock = threading.Lock()
    # Cancelling queued futures does not stop a RUNNING one, and a square holds
    # many chips. Without this flag the workers kept starting new chips for ~50 s
    # after Ctrl+C, failing with "cannot schedule new futures after interpreter
    # shutdown" and writing chips nobody banked. Observed on a live stop.
    stop = threading.Event()
    t0 = last = last_flush = time.perf_counter()
    seen = 0

    def publish(res) -> None:
        """Bank one chip's result the moment it exists.

        Workers call this per chip rather than returning a list per square.
        Returning per square held every row hostage until the whole square
        finished — measured: a SIGTERM mid-square left 10 chips on disk with 1
        manifest row. A dense square holds hundreds of labels, so that window
        scales with the survey, which is exactly what a work unit must never do.
        """
        nonlocal seen, last, last_flush
        with lock:
            seen += 1
            if isinstance(res, dict):
                rows.append(res)
                # Whichever bound trips first. The clock is the one that matters:
                # it keeps the exposure window at a minute even when chips are slow.
                if (
                    len(rows) % FLUSH_EVERY == 0
                    or time.perf_counter() - last_flush >= FLUSH_SECONDS
                ):
                    _flush(existing, rows, manifest_uri)
                    last_flush = time.perf_counter()
            else:
                drops[res.split("=")[0]] = drops.get(res.split("=")[0], 0) + 1

            now = time.perf_counter()
            if now - last >= 60.0:
                rate = seen / max(now - t0, 1e-6)
                top = ", ".join(
                    f"{k}={v}" for k, v in sorted(drops.items(), key=lambda kv: -kv[1])[:4]
                )
                logger.info(
                    "progress: {}/{} ({:.1f}%), {} chips, {:.2f} obs/s, eta {:.0f} min{}",
                    seen, n_todo, 100 * seen / n_todo, len(rows), rate,
                    (n_todo - seen) / max(rate, 1e-6) / 60,
                    f" | drops: {top}" if top else "",
                )
                last = now

    def run_square(unit):
        """Chips of one square, in order — that sequencing is what lets the
        remembered-date cache hit. Each result is published as it lands."""
        for row in unit:
            if stop.is_set():  # a stop was requested — start no new chip
                return
            try:
                publish(_process_obs(row, months_by_zone[row.zone], bands, out_prefix, opt))
            except Exception as exc:  # one bad label must never end the run
                logger.debug("obs {} failed: {}: {}", row.obs_id, type(exc).__name__, exc)
                publish(f"error:{type(exc).__name__}")

    # NOT `with ThreadPoolExecutor(...)`. Its __exit__ calls shutdown(wait=True),
    # which drains every future already queued — and every square is queued up
    # front — so a single Ctrl+C looked like a hang and kept chipping to the end
    # of the run. Measured on 2000 queued tasks: 25.1 s to stop that way against
    # 0.3 s with cancel_futures.
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        # SIGTERM raises KeyboardInterrupt in here, so `pkill` reaches the same
        # save path Ctrl+C does instead of ending the process where it stands.
        with _term_saves_work():
            futures = [pool.submit(run_square, u) for u in units]
            for fut in as_completed(futures):
                # A KeyboardInterrupt raised inside a worker is a BaseException,
                # so `run_square`'s `except Exception` lets it through and it
                # re-raises here. Rows are already banked by publish().
                fut.result()
    except KeyboardInterrupt:
        stop.set()
        logger.warning("interrupted — cancelling queued squares, letting in-flight chips finish")
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        # Bank whatever finished before re-raising. Without this the rows since
        # the last flush are lost, and their chips have to be re-adopted from
        # disk on the next start.
        stop.set()
        pool.shutdown(wait=False)
        # Give the chips already in flight a moment to publish. `stop` means no
        # NEW chip starts, so this waits out at most one chip per worker — and it
        # turns rows that would have been orphaned into banked ones. Bounded, so
        # a wedged read cannot hold the stop open.
        deadline = time.perf_counter() + DRAIN_SECONDS
        while (
            any(not f.done() for f in locals().get("futures", []))
            and time.perf_counter() < deadline
        ):
            time.sleep(0.2)
        with lock:
            _flush(existing, rows, manifest_uri)
        logger.info("saved {} rows to the manifest", len(rows))

    if drops:
        logger.info("drops: {}", dict(sorted(drops.items(), key=lambda kv: -kv[1])))
    logger.info("date cache: {} answers remembered across {} squares", len(_DATE_CACHE), len(units))
    return rows


def _flush(existing: pd.DataFrame, rows: list[dict], manifest_uri: str) -> pd.DataFrame:
    """Merge new rows into the manifest and write it. Returns the merged frame."""
    if not rows:
        return existing
    merged = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(
        subset=["obs_id"], keep="last"
    )
    write_parquet_df(merged, manifest_uri)
    return merged


# ---------------------------------------------------------------------------
# Split (training time — decoupled from extraction)
# ---------------------------------------------------------------------------


def _labels_to_gdf(label_pts: pd.DataFrame, manifest: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert manifest label rows to a GeoDataFrame for spatial splitting."""
    coords = manifest.drop_duplicates(subset=["obs_id"])[["obs_id", "lon", "lat"]]
    merged = label_pts.merge(coords, on="obs_id", how="left")
    return gpd.GeoDataFrame(
        merged, geometry=gpd.points_from_xy(merged["lon"], merged["lat"]), crs="EPSG:4326"
    )


def make_split(
    manifest_uri: str,
    aoi_uri: str,
    *,
    species: list[str] | None = None,
    class_map_name: str | None = None,
    schema_path: str = "configs/labels_schema.yaml",
    seed: int = 42,
    block_km: float = 10.0,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    min_class_obs: int = 0,
    out_prefix: str | None = None,
    lock_folds: bool = True,
) -> pd.DataFrame:
    """Build a reproducible spatial split from a chip manifest.

    Pipeline: load manifest -> filter species -> resolve ``class_id`` -> spatial
    block split. Thinning happened earlier, before extraction, in
    :func:`thin_labels`.

    Called at training time — decoupled from chip extraction so the same chips can
    be split differently across experiments.

    Parameters
    ----------
    manifest_uri : str
        Path to ``manifest.parquet``.
    aoi_uri : str
        AOI polygon for building the spatial block grid.
    species : list[str] | None
        Species names to include (exact match, case-insensitive). ``None`` = all.
    class_map_name : str | None
        Name of a ``class_maps`` entry in ``schema_path`` (e.g. ``"sa_landcover"``).
        When set, a ``class_id`` column is added by matching ``manifest.species``
        against the map, so species sharing a class collapse to one training class.
        Unmapped species get ``NaN`` and are dropped with a warning.
    schema_path : str
        Labels schema YAML (used only when ``class_map_name`` is set).
    seed : int
        Random seed for reproducible splits.
    block_km : float
        Spatial block size in km.
    train_frac, val_frac : float
        Target proportions for train and validation folds.
    min_class_obs : int
        Drop classes with fewer than this many obs before splitting.
    out_prefix : str | None
        If set, writes ``block_folds.parquet`` and ``split.parquet``.
    lock_folds : bool
        If True and ``out_prefix`` has an existing ``block_folds.parquet``, lock
        those block assignments and only assign new blocks.
    """
    from cmrv.io import read_gdf

    manifest = _load_existing_manifest(manifest_uri.rsplit("/manifest.parquet", 1)[0])
    if manifest.empty:
        raise ValueError(f"no manifest at {manifest_uri}")

    if "n_months" in manifest.columns:
        for n_months, count in manifest["n_months"].value_counts().sort_index().items():
            logger.info("  chips with {} month(s): {}", n_months, count)

    # --- species filter (exact match, case-insensitive) ---
    if species:
        species_lower = {s.lower() for s in species}
        n_before = manifest["obs_id"].nunique()
        manifest = manifest[manifest["species"].str.lower().isin(species_lower)]
        unmatched = species_lower - set(manifest["species"].str.lower().unique())
        if unmatched:
            logger.warning("species not found in manifest: {}", sorted(unmatched))
        logger.info("species filter: {} -> {} obs_ids", n_before, manifest["obs_id"].nunique())
        if manifest.empty:
            raise ValueError(f"no chips match species filter: {species}")

    blocks = build_spatial_blocks(read_gdf(aoi_uri), block_km=block_km)

    label_pts = manifest[["obs_id", "species", "block_id"]].drop_duplicates(subset=["obs_id"]).copy()
    label_pts.rename(columns={"species": "species_normalized"}, inplace=True)

    # Resolve class_id BEFORE the split so we stratify on the actual training
    # target, drop unmapped species, and drop tiny classes — otherwise a spatially
    # clustered class can pile entirely into one fold.
    strat_col = "species_normalized"
    if class_map_name:
        from cmrv.labels.classmap import build_lookup

        cm = build_lookup(schema_path, class_map_name)
        sp = label_pts["species_normalized"].str.lower().str.strip()
        cid = sp.map(cm.binomial_to_class)
        if cm.genus_to_class:
            miss = cid.isna()
            cid.loc[miss] = sp[miss].str.split().str[0].map(cm.genus_to_class)
        label_pts["class_id"] = cid

        unmapped = label_pts["class_id"].isna()
        if unmapped.any():
            names = sorted(label_pts.loc[unmapped, "species_normalized"].dropna().unique())
            if species:  # explicit species set -> class_map is a labelling shim, keep them
                logger.warning(
                    "class_map '{}': {} unmapped obs KEPT (match --species): {}",
                    class_map_name, int(unmapped.sum()), names[:10],
                )
                label_pts["class_id"] = label_pts["class_id"].fillna(-1)
            else:
                logger.warning(
                    "class_map '{}': dropping {} unmapped obs ({} species): {}",
                    class_map_name, int(unmapped.sum()), len(names), names[:10],
                )
                label_pts = label_pts[~unmapped]

        if min_class_obs > 0:
            counts = label_pts["class_id"].value_counts()
            tiny = sorted(counts.index[counts < min_class_obs])
            if tiny:
                logger.warning("dropping {} classes with <{} obs: {}", len(tiny), min_class_obs, tiny)
                label_pts = label_pts[~label_pts["class_id"].isin(tiny)]
        label_pts["class_id"] = label_pts["class_id"].astype(int)
        strat_col = "class_id"

    existing_folds = None
    if lock_folds and out_prefix:
        existing_folds = _load_block_folds(f"{out_prefix}/block_folds.parquet")

    label_gdf = _labels_to_gdf(label_pts, manifest)
    if class_map_name:
        # Class-aware de-dup: collapse same-class near-dups whose source species
        # strings differ (e.g. "Pinus" vs "Pinus pinaster") — the species-level
        # thin at ingest time keeps those separate. thin_m matches ingest (20 m).
        label_gdf = thin_labels(label_gdf, thin_m=20.0, species_col="class_id")
    label_pts, block_to_fold = stratified_spatial_split(
        label_gdf, blocks, species_col=strat_col, train_frac=train_frac,
        val_frac=val_frac, seed=seed, existing_block_folds=existing_folds,
    )

    # Inner-merge restricts the manifest to surviving obs (drops unmapped / tiny /
    # out-of-AOI) and attaches fold + class_id in one step.
    keep_cols = ["obs_id", "fold"] + (["class_id"] if class_map_name else [])
    manifest = manifest.merge(
        label_pts[keep_cols].drop_duplicates("obs_id"), on="obs_id", how="inner"
    )

    if class_map_name:
        logger.info(
            "class_map '{}': {} classes, {} obs_ids retained",
            class_map_name, manifest["class_id"].nunique(), manifest["obs_id"].nunique(),
        )

    if out_prefix:
        _save_block_folds(block_to_fold, f"{out_prefix}/block_folds.parquet")
        # One-row-per-obs split artifact (obs_id -> fold, class_id) — the head
        # reads this directly so it never re-derives class assignment from the
        # schema. `source` rides along so train-head can score each source
        # separately: a distilled source (NIAPS) scored against itself only
        # measures agreement with the rule that generated it.
        one = manifest.drop_duplicates("obs_id").copy()
        one["source"] = one["obs_id"].astype(str).str.split(":").str[0]
        cols = ["obs_id", "fold", "source"] + (["class_id"] if "class_id" in manifest.columns else [])
        write_parquet_df(one[cols], f"{out_prefix}/split.parquet")

    for fold in ["train", "val", "test"]:
        sub = manifest[manifest["fold"] == fold]
        logger.info(
            "fold {}: {} obs_ids ({:.1f}%), {} species",
            fold, sub["obs_id"].nunique(),
            100 * sub["obs_id"].nunique() / manifest["obs_id"].nunique(),
            sub["species"].nunique(),
        )

    return manifest
