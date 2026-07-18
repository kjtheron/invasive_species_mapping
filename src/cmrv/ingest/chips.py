"""Sparse training-chip extraction — temporally aligned to label dates.

For each label point, extracts a 64x64 px (10 m) chip for each configured
season month from the label's observation year. The multi-month temporal stack
feeds the UniverSat embedding stage directly (no super-resolution).

Spatial blocks (not inference tiles) serve dual purpose:
    1. Query batching — one STAC composite per (block, year, month)
    2. Stratified splitting — blocks are the atomic unit for train/val/test
       (splitting happens at training time via ``make_split``, not during
       extraction)

Output layout::

    {out_prefix}/{obs_id}/{year}/{month_label}.tif  — 10-band float32 GeoTIFF
    {out_prefix}/manifest.parquet                   — chip index (no fold column)

CRS handling:
    Labels arrive in EPSG:4326.  Each (block, year, zone) group is composited in
    its **own native S2 UTM zone** (``utm_epsg`` of the group centroid), so a KZN/EC
    chip is not resampled onto a distant zone.  Label points are transformed to that
    zone for pixel-index math and chips are sliced directly from the composite — no
    reprojection of the chip itself, so no rotation-induced missing corners.  The
    manifest stores each point's lon/lat (EPSG:4326), keeping it CRS-agnostic across
    zones.  Spatial-block / tile grids use SA Albers equal-area (``cmrv.aoi``).

Performance:
    - Tight bounding box around label cluster (not full 20 km block)
    - Concurrent group processing via ThreadPoolExecutor
    - Per-label window compute (only the kept pixels are materialised)
    - Manifest-based incremental resume (per-group shards; crash-safe)
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse, urlunparse

# GDAL HTTP timeouts — prevent /vsicurl/ stalls on bad tiles (e.g. S2C).
# Direct assignment (not setdefault) so we always override anything inherited.
os.environ["GDAL_HTTP_TIMEOUT"] = "60"  # connect + read timeout (s)
os.environ["CPL_VSIL_CURL_TIMEOUT"] = "60"  # /vsicurl/ read timeout (s)
os.environ["GDAL_HTTP_MAX_RETRY"] = "2"  # GDAL-level retries (on top of ours)
os.environ["GDAL_HTTP_RETRY_DELAY"] = "5"

# COG-streaming efficiency for windowed /vsicurl reads (Planetary-Computer recipe):
# fewer HTTP requests + less redundant bytes when reading 64px windows from many
# COG scenes per monthly median. Network-bound chipping, so this is the main lever.
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"  # no dir-listing req per open
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"  # don't probe for sidecars
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"  # HTTP/2 request multiplexing
os.environ["GDAL_HTTP_VERSION"] = "2"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"  # coalesce nearby byte ranges
os.environ["VSI_CACHE"] = "TRUE"  # cache COG blocks across the ~30-scene median
os.environ["GDAL_CACHEMAX"] = "512"  # MB block cache

import dask
import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer as pc
import pystac_client
import rasterio
import stackstac
import xarray as xr
from loguru import logger
from rasterio.crs import CRS
from rasterio.transform import rowcol
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
from shapely.geometry import box
from shapely.ops import unary_union

from cmrv.aoi import SA_ALBERS, utm_epsg
from cmrv.ingest.cloud_mask import apply_scl_mask
from cmrv.ingest.composite import _transform_from_da, monthly_median
from cmrv.io import ensure_parent, read_parquet_df, write_parquet_df

CHIP_PX = 64
RESOLUTION_M = 10
BUFFER_M = (CHIP_PX * RESOLUTION_M) / 2
# Min fraction of chip pixels where ALL bands are finite. The cloud-NaN that
# remains is filled before embedding (UniverSat NaN-poisons otherwise); 0.5 drops
# chips too cloudy to be worth filling. valid_frac is recorded per chip so the
# loader can filter/weight further.
MIN_VALID_FRAC = 0.5
BLOCK_KM = 10
# Default ±days padding around each calendar-month window (chips only).
# Widening to ±15d roughly doubles the candidate-scene pool per window so
# the median can recover cloud-free pixels in cloudy months.
WINDOW_PADDING_DAYS = 15

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"


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

    Blocks serve as STAC query batches and as the atomic unit for
    stratified train/val/test splitting.  Larger than inference tiles
    (default 20 km) to keep the number of STAC queries manageable
    over a province-sized AOI.
    """
    aoi_m = aoi.to_crs(crs)
    minx, miny, maxx, maxy = aoi_m.total_bounds
    step = block_km * 1000.0
    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)
    cells = [box(x, y, x + step, y + step) for x in xs for y in ys]
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

    # block × class count matrix; desired[fold] = each fold's remaining per-class quota
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
        desired[f] = desired[f] - mat.loc[bid]
        desired_total[f] -= float(mat.loc[bid].sum())
    if locked:
        logger.info(
            "spatial split: {} blocks locked, {} to assign", len(locked), len(mat) - len(locked)
        )

    # Iterative stratification: take the rarest remaining class, and send each of
    # its blocks to the fold most short of that class (tie → most short overall →
    # random). Rare classes placed first can't get starved by the common ones.
    rng = np.random.default_rng(seed)
    remaining = mat.drop(index=list(locked))
    while len(remaining):
        active = remaining.sum(axis=0).pipe(lambda s: s[s > 0])
        if active.empty:  # label-less blocks (shouldn't survive thinning) → emptiest fold
            for bid in remaining.index:
                block_to_fold[bid] = max(desired_total, key=desired_total.get)
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
        logger.info(
            "fold {}: {} labels ({:.1f}%), {} species",
            f,
            n,
            100 * n / len(labels),
            n_sp,
        )

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

    All configured season months (Feb/May/Sep) are taken from the label year
    itself — no cross-year offset. Each window is widened by
    ``padding_days`` on each side to give the median composite more
    candidate scenes (helps in cloudy months).
    """
    windows = []
    for m in months_cfg:
        base_start = pd.to_datetime(f"{year}{m['start'][4:]}")
        base_end = pd.to_datetime(f"{year}{m['end'][4:]}")
        start = base_start - pd.Timedelta(days=padding_days)
        end = base_end + pd.Timedelta(days=padding_days)
        windows.append(
            {
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "label": m["label"],
            }
        )
    return windows


# ---------------------------------------------------------------------------
# STAC helpers (reusable client)
# ---------------------------------------------------------------------------


def _stac_client() -> pystac_client.Client:
    # No modifier — _query_items signs once explicitly. Avoids signing items
    # twice (modifier on load + explicit) on every search.
    return pystac_client.Client.open(STAC_URL)


def _strip_sas_inplace(item) -> None:
    # PC SDK 1.0.0 sas.py:142-145 short-circuits sign_url when the href already
    # has st/se/sp query params, so re-signing an already-signed item is a no-op
    # and the cached token never gets a chance to refresh. Strip the query first.
    for asset in item.assets.values():
        p = urlparse(asset.href)
        if p.netloc.endswith(".blob.core.windows.net"):
            asset.href = urlunparse(p._replace(query=""))


def _query_items(
    client: pystac_client.Client,
    geom_wgs84,
    date_start: str,
    date_end: str,
    *,
    cloud_cover_max: int = 40,
) -> list:
    """Search STAC and return freshly-signed items (empty list if none).

    Each item is signed individually so the call works on plain lists too.
    """
    items = list(
        client.search(
            collections=[COLLECTION],
            intersects=geom_wgs84.__geo_interface__,
            datetime=f"{date_start}/{date_end}",
            query={"eo:cloud_cover": {"lt": cloud_cover_max}},
        ).item_collection()
    )
    for item in items:
        pc.sign_inplace(item)
    return items


def _stack_items(
    items: list,
    geom_wgs84,
    bands: list[str],
    *,
    resolution_m: int = RESOLUTION_M,
    epsg: int = 32734,
) -> xr.DataArray:
    """Build lazy masked composite from a list of signed STAC items."""
    da = stackstac.stack(
        items,
        assets=bands + ["SCL"],
        resolution=resolution_m,
        epsg=epsg,
        bounds_latlon=geom_wgs84.bounds,
        rescale=False,
        chunksize=1024,
    )
    return apply_scl_mask(da).astype("float32")


def _download_item_assets(item, bands: list[str], tmp_dir: Path) -> dict[str, str]:
    """Download all band + SCL assets for one STAC item to tmp_dir.

    Returns a mapping of asset key → local file path (str).
    Used as fallback when streaming via VSICURL fails (e.g. S2C tiles that
    return a non-TIFF response over the signed URL).

    Uses requests (not urllib) to avoid system credential handlers that cause
    Azure to reject SAS-token URLs with a 403 auth conflict.
    """
    import requests

    # JIT re-sign: strip stale SAS, then sign so get_token() refreshes if expired.
    _strip_sas_inplace(item)
    pc.sign_inplace(item)

    local_paths: dict[str, str] = {}
    session = requests.Session()
    session.headers.clear()  # no auth headers — SAS token is in the URL

    for key in bands + ["SCL"]:
        asset = item.assets.get(key)
        if asset is None:
            continue
        href = asset.href
        ext = Path(href.split("?")[0]).suffix or ".tif"
        dest = tmp_dir / f"{item.id}_{key}{ext}"
        if not dest.exists():
            try:
                resp = session.get(href, stream=True, timeout=120)
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            except Exception as exc:
                logger.warning("download failed for {} {}: {}", item.id, key, exc)
                continue
        local_paths[key] = str(dest)
    return local_paths


ChipResult = tuple[np.ndarray, rasterio.transform.Affine, float]


def _window_medians(
    stack: xr.DataArray,
    points_utm: list[tuple[float, float]],
    chip_px: int = CHIP_PX,
) -> list[ChipResult | str]:
    """Per-point 64×64 window median from a lazy (time, band, y, x) masked stack.

    Slices each point's window lazily, medians over time, and computes all
    windows in **one** ``dask.compute`` — so only the pixels inside the windows
    are materialised, never the gaps between sparse labels.

    Returns, aligned to ``points_utm``: ``(arr, chip_transform, valid_frac)`` on
    success, or a reason string ``"oob"`` / ``"low_valid_frac=<f>"``.
    ``valid_frac`` = fraction of pixels finite across **all** bands.
    """
    transform = _transform_from_da(stack)
    ny, nx = stack.sizes["y"], stack.sizes["x"]
    half = chip_px // 2

    results: list[ChipResult | str | None] = [None] * len(points_utm)
    lazy: list[xr.DataArray] = []
    pending: list[tuple[int, int, int]] = []  # (result_idx, r0, c0)

    for i, (x_utm, y_utm) in enumerate(points_utm):
        row, col = rowcol(transform, x_utm, y_utm)
        r0, c0 = int(row) - half, int(col) - half
        if r0 < 0 or c0 < 0 or r0 + chip_px > ny or c0 + chip_px > nx:
            results[i] = "oob"
            continue
        win = stack.isel(y=slice(r0, r0 + chip_px), x=slice(c0, c0 + chip_px))
        lazy.append(monthly_median(win))
        pending.append((i, r0, c0))

    if lazy:
        for (i, r0, c0), med in zip(pending, dask.compute(*lazy), strict=True):
            arr = np.asarray(med.values, dtype="float32")
            valid_frac = float(np.isfinite(arr).all(axis=0).mean())
            if valid_frac < MIN_VALID_FRAC:
                results[i] = f"low_valid_frac={valid_frac:.2f}"
            else:
                chip_tf = window_transform(Window(c0, r0, chip_px, chip_px), transform)
                results[i] = (arr, chip_tf, valid_frac)

    return results  # type: ignore[return-value]  # every slot is filled above


def _compute_month(
    client: pystac_client.Client,
    geom_wgs84,
    date_start: str,
    date_end: str,
    bands: list[str],
    points_utm: list[tuple[float, float]],
    *,
    chip_px: int = CHIP_PX,
    cloud_cover_max: int = 40,
    resolution_m: int = RESOLUTION_M,
    epsg: int = 32734,
    max_retries: int = 3,
    retry_delay_s: float = 10.0,
) -> list[ChipResult | str] | None:
    """Query + stack + per-label window median for one month, with retries.

    On transient IO failure the items are re-signed and the windowed compute
    retried. If retries are exhausted, falls back to downloading each scene's
    assets locally. Returns the per-point results from :func:`_window_medians`
    (aligned to ``points_utm``), or None if no scenes / all attempts fail.
    """
    items: list = []
    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 2):  # +1 for the download-fallback attempt
        try:
            if not items:
                items = _query_items(
                    client, geom_wgs84, date_start, date_end, cloud_cover_max=cloud_cover_max
                )
                if not items:
                    logger.info(
                        "  {}/{}: no STAC items (cloud_cover<{}) — month skipped",
                        date_start,
                        date_end,
                        cloud_cover_max,
                    )
                    return None
                logger.debug("STAC {}/{}: {} items", date_start, date_end, len(items))

            if attempt <= max_retries:
                if attempt > 1:
                    # Re-sign individually — the items list isn't an ItemCollection.
                    for item in items:
                        _strip_sas_inplace(item)
                        pc.sign_inplace(item)
                lazy = _stack_items(items, geom_wgs84, bands, resolution_m=resolution_m, epsg=epsg)
                return _window_medians(lazy, points_utm, chip_px)

            # Final fallback: re-sign + download each scene's assets locally.
            logger.info(
                "  {}/{}: streaming failed {} times — downloading assets locally",
                date_start,
                date_end,
                max_retries,
            )
            for item in items:
                _strip_sas_inplace(item)
                pc.sign_inplace(item)
            dl_tmp = Path(tempfile.mkdtemp(prefix="s2dl_"))
            try:
                import copy

                local_items = []
                for item in items:
                    local_paths = _download_item_assets(item, bands, dl_tmp)
                    if not local_paths:
                        continue
                    patched = copy.deepcopy(item)
                    for key, path in local_paths.items():
                        if key in patched.assets:
                            patched.assets[key].href = path
                    local_items.append(patched)
                if not local_items:
                    return None
                lazy = _stack_items(
                    local_items, geom_wgs84, bands, resolution_m=resolution_m, epsg=epsg
                )
                return _window_medians(lazy, points_utm, chip_px)
            finally:
                shutil.rmtree(dl_tmp, ignore_errors=True)

        except Exception as exc:
            last_exc = exc
            if attempt <= max_retries:
                logger.warning(
                    "  {}/{} attempt {}/{} failed: {}: {} — retrying in {:.0f}s",
                    date_start,
                    date_end,
                    attempt,
                    max_retries,
                    type(exc).__name__,
                    exc,
                    retry_delay_s * attempt,
                )
                time.sleep(retry_delay_s * attempt)

    logger.error(
        "  {}/{}: all attempts failed — skipping month. Last error: {}: {}",
        date_start,
        date_end,
        type(last_exc).__name__,
        last_exc,
    )
    return None


# ---------------------------------------------------------------------------
# Chip IO
# ---------------------------------------------------------------------------


def _write_chip_local(
    arr: np.ndarray,
    transform: rasterio.transform.Affine,
    crs: CRS,
    path: str | Path,
) -> None:
    """Write a small (bands, H, W) array as a plain GeoTIFF to a local path."""
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        str(path),
        "w",
        driver="GTiff",
        height=arr.shape[1],
        width=arr.shape[2],
        count=arr.shape[0],
        dtype=arr.dtype.name,
        crs=crs,
        transform=transform,
        nodata=float("nan"),
    ) as dst:
        dst.write(arr)


# ---------------------------------------------------------------------------
# Tight bounding box for label clusters
# ---------------------------------------------------------------------------


def _label_cluster_bbox_wgs84(
    grp: gpd.GeoDataFrame,
    buffer_m: float = BUFFER_M,
    epsg: int = 32734,
) -> object:
    """Compute a tight WGS84 bbox around a group of UTM label points.

    Adds *buffer_m* padding (half-chip) so edge labels get full chips.
    Much smaller than the full 20 km block for sparse label clusters.
    """
    xs = grp.geometry.x.values
    ys = grp.geometry.y.values
    bbox_utm = box(
        xs.min() - buffer_m,
        ys.min() - buffer_m,
        xs.max() + buffer_m,
        ys.max() + buffer_m,
    )
    return gpd.GeoSeries([bbox_utm], crs=f"EPSG:{epsg}").to_crs("EPSG:4326").iloc[0]


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------


def _process_group(
    bid: int,
    year: int,
    grp: gpd.GeoDataFrame,
    months_cfg: list[dict],
    bands: list[str],
    out_prefix: str,
    crs: CRS,
    chip_px: int,
    resolution_m: int,
    cloud_cover_max: int,
    epsg: int,
    g_idx: int,
    n_groups: int,
    chipped_months: set[tuple[str, str]] | None = None,
) -> list[dict]:
    """Process a single (block_id, year) group — self-contained for threading.

    ``chipped_months`` is a set of (obs_id, month_label) pairs already written
    in a prior run.  For each month window, obs_ids that already have that month
    chipped are skipped — the composite is still computed if at least one obs_id
    in the group needs it.
    """
    client = _stac_client()
    cluster_wgs84 = _label_cluster_bbox_wgs84(grp, buffer_m=BUFFER_M, epsg=epsg)
    windows = temporal_windows(year, months_cfg)
    chipped_months = chipped_months or set()

    logger.info(
        "[{}/{}] block {} / year {}: {} labels",
        g_idx,
        n_groups,
        bid,
        year,
        len(grp),
    )

    # Compute each month independently so a single month's IO failure does not
    # kill the other months. Each month is retried (with re-sign) before falling
    # back to a local download of the scene assets.
    rows: list[dict] = []

    for win in windows:
        month_label = win["label"]

        # Filter to obs_ids that still need this month — vectorised set lookup.
        done_this_month = {oid for oid, m in chipped_months if m == month_label}
        pending = grp[~grp["obs_id"].isin(done_this_month)] if done_this_month else grp
        if pending.empty:
            logger.debug(
                "  block {} / {} {}: all obs_ids already chipped — skipping",
                bid,
                year,
                month_label,
            )
            continue

        pending_rows = list(pending.itertuples())
        results = _compute_month(
            client,
            cluster_wgs84,
            win["start"],
            win["end"],
            bands,
            points_utm=[(r.geometry.x, r.geometry.y) for r in pending_rows],
            chip_px=chip_px,
            cloud_cover_max=cloud_cover_max,
            resolution_m=resolution_m,
            epsg=epsg,
        )
        if results is None:
            logger.warning(
                "  block {} / {} {}: skipped (no scenes or all retries failed)",
                bid,
                year,
                month_label,
            )
            continue

        n_written = 0
        drops = {"oob": 0, "low_valid_frac": 0, "write_error": 0}

        for row, result in zip(pending_rows, results, strict=True):
            if isinstance(result, str):
                drops["oob" if result == "oob" else "low_valid_frac"] += 1
                continue
            arr, chip_tf, valid_frac = result

            rel_path = f"{row.obs_id}/{year}/{month_label}.tif"
            try:
                _write_chip_local(arr, chip_tf, crs, f"{out_prefix}/{rel_path}")
            except Exception as exc:
                drops["write_error"] += 1
                logger.warning(
                    "  chip write failed for obs_id={} {}/{}: {}",
                    row.obs_id,
                    month_label,
                    year,
                    exc,
                )
                continue

            n_written += 1
            rows.append(
                {
                    "obs_id": row.obs_id,
                    "species": row.species_normalized,
                    "month_label": month_label,
                    "chip_uri": f"{out_prefix}/{rel_path}",
                    "year": year,
                    "block_id": bid,
                    "lon": float(row.lon),  # EPSG:4326 — chip extracted in the group's native zone
                    "lat": float(row.lat),
                    "valid_frac": valid_frac,
                }
            )

        logger.info(
            "  {} {}: {}/{} chips (drops: cloud={} oob={} write_err={})",
            month_label,
            year,
            n_written,
            len(pending_rows),
            drops["low_valid_frac"],
            drops["oob"],
            drops["write_error"],
        )

    # Persist this group's rows *after* chips are at their final location, so
    # a mid-run crash still leaves a consistent shard → manifest recovery path.
    _write_manifest_shard(rows, out_prefix, bid, year)

    return rows


def _load_existing_manifest(out_prefix: str) -> pd.DataFrame | None:
    """Load existing manifest from GCS/local if it exists.

    Returns ``None`` only when the manifest genuinely does not exist. Any other
    error (corruption, partial write, permissions) is raised — silently treating
    a corrupt manifest as missing would re-chip the entire dataset from scratch.
    """
    p = Path(f"{out_prefix}/manifest.parquet")
    if not p.exists():
        return None
    return pd.read_parquet(p)


# ---------------------------------------------------------------------------
# Per-group manifest shards (crash-safe incremental)
# ---------------------------------------------------------------------------
#
# Each worker writes a small parquet shard to ``{out_prefix}/_manifest_shards/
# {bid}_{year}.parquet`` immediately after its chips are written.  If the run
# crashes, the chips on disk and the shards agree, so the next startup folds the
# shards into ``manifest.parquet`` and incremental skip sees them.  Shard paths
# are unique per group, so workers never collide.


def _shards_dir(out_prefix: str) -> str:
    return f"{out_prefix}/_manifest_shards"


def _write_manifest_shard(rows: list[dict], out_prefix: str, bid: int, year: int) -> None:
    """Persist one group's manifest rows to a per-group shard."""
    if not rows:
        return
    shard_uri = f"{_shards_dir(out_prefix)}/{bid}_{year}.parquet"
    try:
        write_parquet_df(pd.DataFrame(rows), shard_uri)
    except Exception:
        logger.exception("failed to write manifest shard {}", shard_uri)


def _list_shards(out_prefix: str) -> list[str]:
    p = Path(_shards_dir(out_prefix))
    return [str(x) for x in p.glob("*.parquet")] if p.exists() else []


def _read_shards(out_prefix: str, shards: list[str] | None = None) -> pd.DataFrame:
    """Load all shard parquets into a single DataFrame (empty if none).

    Pass ``shards`` to skip re-listing if the caller already has the file list.
    """
    if shards is None:
        shards = _list_shards(out_prefix)
    if not shards:
        return pd.DataFrame()
    frames = []
    for shard in shards:
        try:
            frames.append(read_parquet_df(shard))
        except Exception:
            logger.warning("could not read shard {}", shard)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _delete_shards(out_prefix: str) -> None:
    p = Path(_shards_dir(out_prefix))
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def _consolidate_shards(out_prefix: str, existing: pd.DataFrame | None) -> tuple[pd.DataFrame, int]:
    """Fold any leftover shards into *existing*, return (merged, n_shards_consumed).

    Caller is responsible for rewriting ``manifest.parquet`` and calling
    ``_delete_shards`` once the merged frame is safely persisted.
    """
    shards = _list_shards(out_prefix)
    if not shards:
        return (existing if existing is not None else pd.DataFrame()), 0

    shard_df = _read_shards(out_prefix, shards=shards)
    if shard_df.empty:
        return (existing if existing is not None else pd.DataFrame()), 0

    base = existing if existing is not None and not existing.empty else pd.DataFrame()
    merged = pd.concat([base, shard_df], ignore_index=True).drop_duplicates(
        subset=["obs_id", "month_label", "year"], keep="last"
    )
    return merged, len(shards)


def _load_block_folds(uri: str) -> dict[int, str] | None:
    """Load persisted block→fold mapping if it exists."""
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
    """Persist block→fold mapping to parquet."""
    df = pd.DataFrame(
        sorted(block_to_fold.items()),
        columns=["block_id", "fold"],
    )
    write_parquet_df(df, uri)
    logger.info("saved block folds: {} blocks → {}", len(df), uri)


def _write_split_files(manifest: pd.DataFrame, out_prefix: str) -> None:
    """Write per-fold obs_id lists as text files."""
    for fold in manifest["fold"].unique():
        ids = manifest.loc[manifest["fold"] == fold, "obs_id"].unique()
        split_uri = f"{out_prefix}/{fold}.txt"
        split_content = "\n".join(sorted(ids)) + "\n"
        ensure_parent(split_uri)
        Path(split_uri).write_text(split_content)
        logger.info("split file: {} obs_ids → {}", len(ids), split_uri)


def thin_labels(
    labels: gpd.GeoDataFrame,
    thin_m: float,
    epsg: int = 32734,
    species_col: str = "species_normalized",
) -> gpd.GeoDataFrame:
    """Keep one label per species per ``thin_m`` grid cell (run before extraction).

    Snaps each label's UTM coordinate to a ``thin_m`` grid and keeps a single
    label per ``(species, cell)`` — removing near-duplicates the embedding can't
    distinguish (20 m ≈ 2 native S2 pixels) *before* any imagery is
    fetched, so we never download chips we'd discard.

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
    logger.info("spatial thin ({}m): {} → {} labels", int(thin_m), len(labels), len(keep))
    return keep


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

    Pipeline: load manifest → filter species → spatial block split → assign
    class_id. (Thinning happens earlier, before extraction, in ``thin_labels``;
    obs with only 1–2 of the configured months are kept — the temporal head
    masks missing timesteps.)

    Called at training time — decoupled from chip extraction so the same
    chips can be split differently across experiments.

    Parameters
    ----------
    manifest_uri : str
        Path to ``manifest.parquet`` (local or ``gs://``).
    aoi_uri : str
        AOI polygon for building the spatial block grid.
    species : list[str] | None
        Species names to include (exact match, case-insensitive).
        ``None`` = all species in the manifest.
    class_map_name : str | None
        Name of a ``class_maps`` entry in ``schema_path`` (e.g.
        ``"western_cape_iap"``).  When set, a ``class_id`` column is added by
        matching ``manifest.species`` against the ``species_map`` in the
        schema.  Multiple species that share a class_id are thereby collapsed
        to a single training class (e.g. all *Eucalyptus* spp → class 5).
        Rows whose species are absent from the map get ``class_id = NaN``
        and are dropped with a warning.
    schema_path : str
        Path to the labels schema YAML (used only when ``class_map_name``
        is set).
    seed : int
        Random seed for reproducible splits and thinning tie-breaks.
    block_km : float
        Spatial block size in km (must match extraction).
    train_frac, val_frac : float
        Target proportions for train and validation folds.
    out_prefix : str | None
        If set, writes ``block_folds.parquet`` and ``{fold}.txt`` split files.
    lock_folds : bool
        If True and ``out_prefix`` has an existing ``block_folds.parquet``,
        lock those block assignments and only assign new blocks.

    Returns
    -------
    pd.DataFrame
        Manifest with ``fold`` and (if ``class_map_name`` set) ``class_id``
        columns added.
    """
    from cmrv.io import read_gdf

    manifest = _load_existing_manifest(manifest_uri.rsplit("/manifest.parquet", 1)[0])
    if manifest is None or manifest.empty:
        raise ValueError(f"no manifest at {manifest_uri}")

    # Month-completeness is informational only — we keep obs with 1–3 months
    # and let the temporal head mask the gaps.
    months_per_obs = manifest.groupby("obs_id")["month_label"].nunique()
    for n_months, count in months_per_obs.value_counts().sort_index().items():
        logger.info("  labels with {} month(s): {}", n_months, count)

    # --- species filter (exact match, case-insensitive) ---
    if species:
        species_lower = {s.lower() for s in species}
        mask = manifest["species"].str.lower().isin(species_lower)
        n_before = manifest["obs_id"].nunique()
        manifest = manifest[mask]
        n_after = manifest["obs_id"].nunique()
        matched = manifest["species"].str.lower().unique()
        unmatched = species_lower - set(matched)
        if unmatched:
            logger.warning("species not found in manifest: {}", sorted(unmatched))
        logger.info("species filter: {} → {} obs_ids", n_before, n_after)
        if manifest.empty:
            raise ValueError(f"no chips match species filter: {species}")

    aoi_gdf = read_gdf(aoi_uri)
    blocks = build_spatial_blocks(aoi_gdf, block_km=block_km)

    label_pts = (
        manifest[["obs_id", "species", "block_id"]].drop_duplicates(subset=["obs_id"]).copy()
    )
    label_pts.rename(columns={"species": "species_normalized"}, inplace=True)

    # Resolve class_id BEFORE the split so we stratify on the actual training
    # target (not raw species), drop unmapped species, and drop tiny classes —
    # otherwise a spatially-clustered class can pile entirely into one fold.
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
            if species:  # explicit species set → class_map is a labelling shim, keep them
                logger.warning(
                    "class_map '{}': {} unmapped obs KEPT (match --species): {}",
                    class_map_name,
                    int(unmapped.sum()),
                    names[:10],
                )
                label_pts["class_id"] = label_pts["class_id"].fillna(-1)
            else:
                logger.warning(
                    "class_map '{}': dropping {} unmapped obs ({} species): {}",
                    class_map_name,
                    int(unmapped.sum()),
                    len(names),
                    names[:10],
                )
                label_pts = label_pts[~unmapped]

        if min_class_obs > 0:
            counts = label_pts["class_id"].value_counts()
            tiny = sorted(counts.index[counts < min_class_obs])
            if tiny:
                logger.warning(
                    "dropping {} classes with <{} obs: {}", len(tiny), min_class_obs, tiny
                )
                label_pts = label_pts[~label_pts["class_id"].isin(tiny)]
        label_pts["class_id"] = label_pts["class_id"].astype(int)
        strat_col = "class_id"

    existing_folds = None
    if lock_folds and out_prefix:
        existing_folds = _load_block_folds(f"{out_prefix}/block_folds.parquet")

    label_gdf = _labels_to_gdf(label_pts, manifest)
    if class_map_name:
        # Class-aware de-dup: collapse same-class near-dups whose source species
        # strings differ (e.g. "Pinus" vs "Pinus pinaster") — the species-level thin
        # at ingest time keeps those separate. thin_m matches ingest (20 m).
        label_gdf = thin_labels(label_gdf, thin_m=20.0, species_col="class_id")
    label_pts, block_to_fold = stratified_spatial_split(
        label_gdf,
        blocks,
        species_col=strat_col,
        train_frac=train_frac,
        val_frac=val_frac,
        seed=seed,
        existing_block_folds=existing_folds,
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
            class_map_name,
            manifest["class_id"].nunique(),
            manifest["obs_id"].nunique(),
        )

    if out_prefix:
        _save_block_folds(block_to_fold, f"{out_prefix}/block_folds.parquet")
        _write_split_files(manifest, out_prefix)
        # one-row-per-obs split artifact (obs_id → fold, class_id) — the head reads
        # this directly so it never re-derives class assignment from the schema.
        cols = ["obs_id", "fold"] + (["class_id"] if "class_id" in manifest.columns else [])
        write_parquet_df(manifest.drop_duplicates("obs_id")[cols], f"{out_prefix}/split.parquet")

    for fold in ["train", "val", "test"]:
        sub = manifest[manifest["fold"] == fold]
        logger.info(
            "fold {}: {} obs_ids ({:.1f}%), {} species",
            fold,
            sub["obs_id"].nunique(),
            100 * sub["obs_id"].nunique() / manifest["obs_id"].nunique(),
            sub["species"].nunique(),
        )

    return manifest


def _labels_to_gdf(label_pts: pd.DataFrame, manifest: pd.DataFrame) -> gpd.GeoDataFrame:
    """Convert manifest label rows to a GeoDataFrame for spatial splitting."""
    coords = manifest.drop_duplicates(subset=["obs_id"])[["obs_id", "lon", "lat"]]
    merged = label_pts.merge(coords, on="obs_id", how="left")
    return gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["lon"], merged["lat"]),
        crs="EPSG:4326",
    )


def _reconcile_manifest(
    manifest: pd.DataFrame,
    keep_obs: set[str],
    manifest_uri: str,
) -> pd.DataFrame:
    """Prune chips for obs outside the current thinned set, then rewrite manifest.

    ``ingest-chips`` is additive: a re-thinning that drops an obs (because a
    label-store change picked a different one-per-cell representative) leaves the
    old obs's chips behind, so the manifest becomes a superset of the canonical
    one-rep-per-cell set. This deletes those stale chip files + their now-empty
    obs dirs and rewrites the manifest to equal the canonical set. Disk-only — no
    re-download. ponytail: O(n) manifest scan, fine at chip-set scale.
    """
    if manifest.empty:
        return manifest
    stale = manifest[~manifest["obs_id"].isin(keep_obs)]
    if stale.empty:
        return manifest
    for uri in stale["chip_uri"]:
        Path(uri).unlink(missing_ok=True)
    for d in {Path(u).parent for u in stale["chip_uri"]}:
        if d.exists() and not any(d.iterdir()):
            d.rmdir()
    kept = manifest[manifest["obs_id"].isin(keep_obs)].reset_index(drop=True)
    write_parquet_df(kept, manifest_uri)
    logger.success(
        "reconcile: pruned {} stale chips ({} obs) → {} chips ({} obs) in the thinned set",
        len(stale),
        stale["obs_id"].nunique(),
        len(kept),
        kept["obs_id"].nunique(),
    )
    return kept


def extract_training_chips(
    labels: gpd.GeoDataFrame,
    blocks: gpd.GeoDataFrame,
    months_cfg: list[dict],
    bands: list[str],
    out_prefix: str,
    *,
    months_by_zone: dict[str, list[dict]] | None = None,
    default_zone: str = "winter_rainfall",
    chip_px: int = CHIP_PX,
    resolution_m: int = RESOLUTION_M,
    cloud_cover_max: int = 40,
    default_year: int = 2023,
    max_workers: int = 6,
    year_fallback: bool = True,
) -> pd.DataFrame:
    """Extract temporally-aligned training chips for all labels.

    Groups labels by (block_id, observation_year), builds transient monthly
    composites via MPC STAC, and extracts per-label chips concurrently.

    *labels* must already have a ``block_id`` column (from spatial-join with
    ``build_spatial_blocks``).  No fold assignment is needed — splitting is
    done at training time via ``make_split``.

    Incremental: skips already-chipped (obs_id, month_label) pairs so that
    obs_ids with partial month coverage (e.g. 3/4 months due to a prior
    failure) have their missing months retried on the next run.

    Year alignment: if ``labels`` has a pre-existing ``_year`` column it is
    honoured; otherwise ``_year = event_date.year`` (NaT → ``default_year``).
    For obs_ids that already have prior chips at a non-Y year (typically
    Y-1 from a previous fallback), ``_year`` is realigned to the manifest's
    year so missing months are retried at the same year as the existing chips.

    Year fallback (``year_fallback=True``, default): obs_ids that produce
    zero chips after the main pass are retried once more with ``_year - 1``,
    which often recovers labels whose target year had a cloudy season.
    """
    if "block_id" not in labels.columns:
        raise ValueError("labels must have a 'block_id' column — spatial-join with blocks first")

    labels = labels.copy()
    if "_year" not in labels.columns:
        ed = pd.to_datetime(labels["event_date"], errors="coerce")
        labels["_year"] = ed.dt.year.fillna(default_year).astype(int)
    if "_zone" not in labels.columns:
        labels["_zone"] = default_zone
    # Keep each label's lon/lat (labels arrive in EPSG:4326) so the manifest is
    # CRS-agnostic — chips are extracted per group in the group's native S2 UTM zone.
    if "lon" not in labels.columns:
        labels["lon"] = labels.geometry.x
        labels["lat"] = labels.geometry.y

    # Canonical set for this (top-level) call = the thinned labels as passed in,
    # captured before the incremental filter below mutates `labels`. Used at the
    # end to prune chips for obs a re-thinning has since dropped (additive cruft).
    canonical_obs = set(labels["obs_id"])

    # Recover any leftover shards from a prior crashed run before building the
    # incremental skip set — that way partial-month obs_ids from a prior run
    # are correctly seen as partially-done rather than fully-skipped.
    manifest_uri = f"{out_prefix}/manifest.parquet"
    existing_manifest = _load_existing_manifest(out_prefix)
    recovered, n_shards = _consolidate_shards(out_prefix, existing_manifest)
    if n_shards:
        write_parquet_df(recovered, manifest_uri)
        _delete_shards(out_prefix)
        logger.info(
            "recovered {} manifest shards from prior run → {} rows in manifest",
            n_shards,
            len(recovered),
        )
        existing_manifest = recovered
    elif existing_manifest is None:
        existing_manifest = pd.DataFrame()

    # Build per-(obs_id, month_label) skip set from the manifest.
    # This allows re-runs to retry months that failed previously while skipping
    # months that already have a chip — no bucket listing needed.
    if not existing_manifest.empty:
        # Year override: realign labels._year to the year already present in
        # the manifest for partial obs_ids (e.g. ones chipped at Y-1 from a
        # prior fallback). Without this, a partial Y-1 obs_id would have its
        # missing months retried at Y (where they'll fail again) instead of
        # at Y-1 (which is known to work for that obs_id).
        obs_year_map = (
            existing_manifest.groupby("obs_id")["year"]
            .agg(lambda s: int(s.mode().iloc[0]))
            .to_dict()
        )
        overrides = labels["obs_id"].map(obs_year_map)
        mask = overrides.notna() & (overrides != labels["_year"])
        n_overridden = int(mask.sum())
        if n_overridden:
            labels.loc[mask, "_year"] = overrides[mask].astype(int)
            logger.info(
                "year override: {} obs_ids realigned to manifest's prior year",
                n_overridden,
            )

        chipped_months: set[tuple[str, str]] = set(
            zip(existing_manifest["obs_id"], existing_manifest["month_label"], strict=True)
        )
        fully_chipped = {
            oid
            for oid, cnt in existing_manifest.groupby("obs_id")["month_label"].nunique().items()
            if cnt >= len(months_cfg)
        }
        n_before = len(labels)
        labels = labels[~labels["obs_id"].isin(fully_chipped)]
        partial_ids = {r[0] for r in chipped_months} - fully_chipped
        n_partial = labels["obs_id"].isin(partial_ids).sum()
        logger.info(
            "incremental: {} labels fully chipped, {} labels to process "
            "({} have partial months to retry)",
            n_before - len(labels),
            len(labels),
            n_partial,
        )
    else:
        chipped_months = set()

    if labels.empty:
        logger.success("all labels already fully chipped — nothing to do")
        if year_fallback:  # top-level call: still reconcile away thinned-out cruft
            existing_manifest = _reconcile_manifest(existing_manifest, canonical_obs, manifest_uri)
        return existing_manifest

    # Track obs_ids attempted in this pass — used for the Y-1 fallback below.
    attempted_obs_ids = set(labels["obs_id"])

    groups = list(labels.groupby(["block_id", "_year", "_zone"]))  # labels in EPSG:4326
    n_groups = len(groups)
    logger.info("extracting chips: {} groups with {} workers", n_groups, max_workers)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for g_idx, ((bid, year, zone), grp) in enumerate(groups, 1):
            # each label's month set is chosen by its rainfall zone (winter vs summer)
            grp_months = months_by_zone.get(zone, months_cfg) if months_by_zone else months_cfg
            # extract this group in its OWN native S2 UTM zone (no cross-zone resampling)
            g_epsg = utm_epsg(grp["lon"].mean(), grp["lat"].mean())
            grp_utm = grp.to_crs(f"EPSG:{g_epsg}")
            fut = pool.submit(
                _process_group,
                int(bid),
                int(year),
                grp_utm,
                grp_months,
                bands,
                out_prefix,
                CRS.from_epsg(g_epsg),
                chip_px,
                resolution_m,
                cloud_cover_max,
                g_epsg,
                g_idx,
                n_groups,
                chipped_months,
            )
            futures[fut] = (int(bid), int(year))

        # Time-based progress: log every 60s OR every 50 groups, whichever is
        # sooner. With slow blocks (retries + downloads) the count-only cadence
        # could go ~10min silent.
        progress_interval_s = 60.0
        last_progress_t = time.perf_counter()
        for done, fut in enumerate(as_completed(futures), 1):
            bid, year = futures[fut]
            try:
                fut.result()  # rows are already persisted via per-group shard
            except Exception:
                logger.exception("failed: block {} / year {}", bid, year)
            now = time.perf_counter()
            if done % 50 == 0 or (now - last_progress_t) >= progress_interval_s:
                logger.info(
                    "progress: {}/{} groups done ({:.1f}%)",
                    done,
                    n_groups,
                    100 * done / n_groups,
                )
                last_progress_t = now

    # End-of-run consolidation: fold this run's shards into the manifest.
    manifest_df, n_shards_final = _consolidate_shards(out_prefix, existing_manifest)
    n_new = len(manifest_df) - len(existing_manifest)

    if not manifest_df.empty:
        write_parquet_df(manifest_df, manifest_uri)
        if n_shards_final:
            _delete_shards(out_prefix)

        logger.success(
            "manifest: {} total chips ({} labels), {} new this run → {}",
            len(manifest_df),
            manifest_df["obs_id"].nunique(),
            n_new,
            manifest_uri,
        )

    # Year fallback: obs_ids that produced 0 chips this pass get one retry at
    # _year - 1. Common case: a label-year with persistent winter cloud cover
    # where the prior year was clearer. Single-step (no Y-2), and disabled on
    # the recursive call to prevent runaway retries.
    if year_fallback:
        chipped_after = set(manifest_df["obs_id"]) if not manifest_df.empty else set()
        failed_obs = attempted_obs_ids - chipped_after
        if failed_obs:
            fallback_labels = labels[labels["obs_id"].isin(failed_obs)].copy()
            fallback_labels["_year"] = fallback_labels["_year"] - 1
            logger.info(
                "year fallback: {} obs_ids produced 0 chips at original year — "
                "retrying at Y-1 ({} groups)",
                len(failed_obs),
                fallback_labels.groupby(["block_id", "_year"]).ngroups,
            )
            manifest_df = extract_training_chips(
                labels=fallback_labels,
                blocks=blocks,
                months_cfg=months_cfg,
                months_by_zone=months_by_zone,
                default_zone=default_zone,
                bands=bands,
                out_prefix=out_prefix,
                chip_px=chip_px,
                resolution_m=resolution_m,
                cloud_cover_max=cloud_cover_max,
                default_year=default_year,
                max_workers=max_workers,
                year_fallback=False,
            )

    # Top-level call only (year_fallback=True): after this run + any Y-1 fallback,
    # prune chips for obs no longer in the thinned set so the manifest == canonical.
    if year_fallback:
        manifest_df = _reconcile_manifest(manifest_df, canonical_obs, manifest_uri)

    return manifest_df
