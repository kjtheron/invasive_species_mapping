"""Sentinel-2 L2A access — search, screen, load. The only STAC + raster layer.

Replaces two things at once:

* **stackstac → odc-stac.** stackstac's last release is 0.5.1 (Nov 2023) and it is
  unmaintained. odc-stac also gives us ``geobox=`` (ask for the exact chip grid
  instead of stacking a bbox and slicing windows out of it), ``groupby="solar_day"``
  (two frames of one pass mosaic into ONE timestep instead of two half-NaN ones)
  and per-band resampling (SCL must be nearest; the 20 m red-edge bands want
  bilinear).
* **"median 20 scenes and hope" → "read SCL, pick the clearest date, read that".**

Why the screening pass pays for itself. A Planetary-Computer S2 L2A COG has
512x512 internal blocks (verified: B02 10980², B05/SCL 5490², all ``blocks
[(512,512)]``). A 128 px chip at 10 m is 1280 m; one 512-block at 10 m is 5120 m.
So a chip costs ONE block read per asset per scene whatever the dask chunking is
— the chunk size is not the lever the old code thought it was. Ten bands x 20
scenes is ~110 MB of source reads for a 660 KB chip. SCL alone (one uint8 asset)
over 20 candidate dates is ~5 MB, and it names the single date worth spending the
bands on. Same chip, an order of magnitude less traffic.

Threading: every function here is blocking and dask-free at chip scale
(``chunks=None`` loads eagerly). Parallelism belongs in the caller's thread pool,
where one work unit is one chip. Large-box callers (``cmrv.infer``) pass
``chunks`` and get a lazy dask array instead.
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections import defaultdict
from functools import lru_cache

# GDAL HTTP tuning for windowed /vsicurl/ COG reads. Must precede the first
# rasterio/GDAL use. Direct assignment (not setdefault) so we override anything
# inherited from the parent environment.
os.environ["GDAL_HTTP_TIMEOUT"] = "60"
os.environ["CPL_VSIL_CURL_TIMEOUT"] = "60"
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
os.environ["GDAL_HTTP_RETRY_DELAY"] = "3"
# Abort a stalled connection so GDAL's own retry can act. Without these a read
# that trickles bytes holds its thread for the full 60 s timeout and then fails
# anyway. Mirrors download_tile.py.
os.environ["GDAL_HTTP_LOW_SPEED_TIME"] = "30"  # seconds under the limit
os.environ["GDAL_HTTP_LOW_SPEED_LIMIT"] = "1"  # bytes/second
# Fewer requests, fewer redundant bytes.
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = ".tif"
os.environ["GDAL_HTTP_MULTIPLEX"] = "YES"
os.environ["GDAL_HTTP_VERSION"] = "2"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"
# The block cache is what makes spatially-sorted chips cheap: labels thinned at
# 20 m sit inside the same 512-block, so a sorted run re-reads it from RAM.
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "100000000"
os.environ["GDAL_CACHEMAX"] = "512"  # MB

import warnings  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # type: ignore  # noqa: E402
import planetary_computer as pc  # type: ignore  # noqa: E402
import pystac_client  # type: ignore  # noqa: E402
import xarray as xr  # type: ignore  # noqa: E402
from loguru import logger  # type: ignore  # noqa: E402
from odc.geo.geobox import GeoBox  # type: ignore  # noqa: E402
from odc.geo.geom import BoundingBox  # type: ignore  # noqa: E402
from odc.stac import load as odc_load  # type: ignore  # noqa: E402
from shapely.geometry import shape  # type: ignore  # noqa: E402
from rasterio.errors import NotGeoreferencedWarning  # type: ignore  # noqa: E402
from shapely.ops import unary_union  # type: ignore  # noqa: E402

# odc-loader warps through an in-memory dataset that has no geotransform of its
# own; the warning is noise on every single chip read.
warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"

# SCL codes that mark an unusable pixel.
# 0 no-data, 1 saturated/defective, 3 cloud shadow, 8 cloud medium,
# 9 cloud high, 10 thin cirrus. Left valid: 2 (dark/topographic shadow — common
# on fynbos slopes, over-masking risk) and 11 (snow/ice — rare in SA chips).
BAD_SCL: frozenset[int] = frozenset({0, 1, 3, 8, 9, 10})

# Searches are cached per coarse geographic cell, not per chip. Labels thinned at
# 20 m put hundreds of chips in one cell, and they all want the same item list.
CELL_DEG = 0.5

# How long a cached search stays usable. Planetary Computer signs asset hrefs
# with a SAS token that lives about an hour, so a cache that never expired would
# hand a worker a URL that dies mid-read. 20 minutes leaves a wide margin.
SEARCH_TTL_S = 1200

# One STAC call, retried. Planetary Computer answers "You have exceeded a rate
# limit" under load and pystac-client does not retry.
STAC_ATTEMPTS = 4
STAC_BACKOFF_S = 5.0


# ---------------------------------------------------------------------------
# Client, search, retry
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def stac_client() -> pystac_client.Client:
    """One shared, signing client. ``sign_inplace`` on the modifier means every
    item comes back signed, so no caller re-signs by hand."""
    return pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)


def stac_retry(call, what: str, attempts: int = STAC_ATTEMPTS):
    """Run a STAC call, retrying with a growing, **jittered** backoff.

    The jitter is the part that matters. Every worker hits a rate limit at the
    same moment, so without it they all come back at the same moment and trip it
    again. The old code slept ``delay * attempt`` with no jitter, so 6 workers
    retried in lockstep.
    """
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt == attempts:
                raise
            wait = min(STAC_BACKOFF_S * 2 ** (attempt - 1), 60.0) + random.uniform(0, 5)
            logger.warning(
                "{}: STAC call failed ({}: {}) — retry {}/{} in {:.0f}s",
                what,
                type(exc).__name__,
                str(exc)[:80],
                attempt,
                attempts - 1,
                wait,
            )
            time.sleep(wait)


def cell_of(lon: float, lat: float) -> tuple[int, int]:
    """Coarse cell key for a lon/lat — the search-cache and work-sort key."""
    return (int(np.floor(lat / CELL_DEG)), int(np.floor(lon / CELL_DEG)))


def _cell_bbox(cell: tuple[int, int]) -> tuple[float, float, float, float]:
    """WGS84 bbox of a cell, padded one cell each side so a chip near a cell
    edge still sees every scene that covers it."""
    cy, cx = cell[0] * CELL_DEG, cell[1] * CELL_DEG
    return (cx - CELL_DEG, cy - CELL_DEG, cx + 2 * CELL_DEG, cy + 2 * CELL_DEG)


_SEARCH_CACHE: dict[tuple, tuple[float, tuple]] = {}
_SEARCH_LOCK = threading.Lock()


def search_cell(cell: tuple[int, int], start: str, end: str) -> tuple:
    """Signed items covering one coarse cell over one date window, memoised.

    The cache carries a TTL because the hrefs are signed: an entry older than
    :data:`SEARCH_TTL_S` is re-searched, which re-signs it. Two threads can race
    and search the same cell twice — harmless, and cheaper than holding the lock
    across a network call.
    """
    key = (cell, start, end)
    now = time.monotonic()
    with _SEARCH_LOCK:
        hit = _SEARCH_CACHE.get(key)
        if hit is not None and now - hit[0] <= SEARCH_TTL_S:
            return hit[1]

    items = tuple(
        stac_retry(
            lambda: list(
                stac_client()
                .search(
                    collections=[COLLECTION],
                    bbox=list(_cell_bbox(cell)),
                    datetime=f"{start}/{end}",
                )
                .item_collection()
            ),
            f"S2 cell search {cell} {start}",
        )
    )
    with _SEARCH_LOCK:
        _SEARCH_CACHE[key] = (time.monotonic(), items)
    return items


def drop_cell(cell: tuple[int, int], start: str, end: str) -> None:
    """Forget one cached search, so the next call re-searches and re-signs.

    Call this when a read fails: the most likely cause is an expired signature,
    and only a fresh search can fix it. This is what the old code tried to solve
    by downloading whole tiles to disk.
    """
    with _SEARCH_LOCK:
        _SEARCH_CACHE.pop((cell, start, end), None)


def search_bbox(bbox: tuple[float, float, float, float], start: str, end: str) -> list:
    """Un-cached search over an arbitrary WGS84 bbox — for wall-to-wall boxes,
    which are too big and too few to share a cell cache."""
    return stac_retry(
        lambda: list(
            stac_client()
            .search(collections=[COLLECTION], bbox=list(bbox), datetime=f"{start}/{end}")
            .item_collection()
        ),
        f"S2 bbox search {start}",
    )


# ---------------------------------------------------------------------------
# Geometry screening — free, so it runs before anything is read
# ---------------------------------------------------------------------------


def covering_items(items, geom_wgs84, min_coverage: float = 0.98) -> list:
    """Keep only the acquisition dates whose footprints actually cover *geom_wgs84*.

    STAC matches on bounding boxes; a bbox is not a footprint. An edge granule's
    real data polygon can miss the chip entirely, and the old code only found
    that out after building a stack and computing a median.

    Coverage is measured per **date** over the UNION of that date's scenes,
    because a chip on an MGRS tile boundary is legitimately covered by two frames
    of the same pass. Every scene of a passing date is kept, so ``groupby=
    "solar_day"`` can mosaic them.
    """
    by_day: dict = defaultdict(list)
    for it in items:
        by_day[it.datetime.date()].append(it)

    kept: list = []
    area = geom_wgs84.area
    for _day, day_items in sorted(by_day.items()):
        footprint = unary_union([shape(i.geometry) for i in day_items])
        if geom_wgs84.intersection(footprint).area >= min_coverage * area:
            kept.extend(day_items)
    kept.sort(key=lambda i: i.datetime)
    return kept


def by_solar_day(items) -> list[list]:
    """Group items into per-date lists, chronologically."""
    by_day: dict = defaultdict(list)
    for it in items:
        by_day[it.datetime.date()].append(it)
    return [by_day[d] for d in sorted(by_day)]


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------


def point_geobox(x: float, y: float, epsg: int, chip_px: int, res_m: float) -> GeoBox:
    """Chip grid centred on a projected point, snapped to the ``res_m`` grid.

    Snapping makes the grid deterministic: the same label always yields the same
    pixel edges, so a re-chip is byte-comparable and the chip lands on the source
    S2 pixel grid (whose UTM origins are multiples of 100 km, hence of 10 m).
    That alignment is what makes the resampling a no-op in the common case.
    """
    half = chip_px * res_m / 2.0
    xmin = np.floor((x - half) / res_m) * res_m
    ymin = np.floor((y - half) / res_m) * res_m
    bbox = BoundingBox(xmin, ymin, xmin + chip_px * res_m, ymin + chip_px * res_m,
                       crs=f"EPSG:{epsg}")
    return GeoBox.from_bbox(bbox, resolution=res_m, tight=True)


def bbox_geobox(bounds, epsg: int, res_m: float) -> GeoBox:
    """Snapped grid covering a projected bounds tuple — the wall-to-wall case."""
    minx, miny, maxx, maxy = bounds
    xmin = np.floor(minx / res_m) * res_m
    ymin = np.floor(miny / res_m) * res_m
    xmax = np.ceil(maxx / res_m) * res_m
    ymax = np.ceil(maxy / res_m) * res_m
    return GeoBox.from_bbox(
        BoundingBox(xmin, ymin, xmax, ymax, crs=f"EPSG:{epsg}"), resolution=res_m, tight=True
    )


def _buffered(gbox: GeoBox, buffer_px: int) -> GeoBox:
    """Grow a geobox by whole pixels, keeping CRS, resolution and alignment.

    Load onto this, then slice the buffer off. It gives the resampler a full
    source neighbourhood at every pixel we keep, so a same-day mosaic of two
    frames cannot leave a NaN seam down the chip.
    """
    if buffer_px <= 0:
        return gbox
    return gbox.buffered(
        xbuff=buffer_px * abs(gbox.resolution.x), ybuff=buffer_px * abs(gbox.resolution.y)
    )


# ---------------------------------------------------------------------------
# Reads
# ---------------------------------------------------------------------------


def clear_fraction_per_date(items, gbox: GeoBox, *, pool: int = 8) -> list[tuple[float, list]]:
    """``[(clear_fraction, items_of_that_date), ...]`` — SCL only, one asset.

    This is the screening pass. It reads a single uint8 band per date instead of
    the ten reflectance bands, and the answer it gives — which date is clearest
    **over this exact chip** — is the one ``eo:cloud_cover`` cannot give, because
    that property describes a 110 km MGRS tile.

    ``pool`` is odc-stac's own read concurrency. These reads are pure network
    latency, so concurrency inside one chip is nearly free. Measured cold (8
    scattered SA sites, so no block-cache reuse): 48 dates screened in 71.8 s at
    pool=None against 19.2 s at pool=4, and band loads 19.7 s against 7.3 s each.
    Beyond ~8 the returns flatten. Total in-flight sockets is
    ``max_workers x pool`` — keep the product in the low hundreds.
    """
    groups = by_solar_day(items)
    if not groups:
        return []
    by_date = {g[0].datetime.date(): g for g in groups}
    flat = [i for g in groups for i in g]
    ds = odc_load(
        flat,
        bands=["SCL"],
        geobox=gbox,
        resampling="nearest",  # SCL is a class code — bilinear would invent classes
        groupby="solar_day",
        chunks=None,
        pool=pool,
        fail_on_error=False,
    )
    scl = ds["SCL"].values  # (time, y, x) uint8
    if scl.ndim == 2:
        scl = scl[None]
    clear = ~np.isin(scl, list(BAD_SCL))
    fracs = clear.reshape(clear.shape[0], -1).mean(axis=1)

    # Pair each fraction to its date by the returned `time` coordinate, never by
    # position. ``fail_on_error=False`` lets odc-stac drop a date whose asset will
    # not read, and a positional zip would then shift every later pairing by one —
    # so the chip would be built from a different scene than the one that scored.
    # That failure is invisible: the chip looks fine and is simply the wrong day.
    times = pd.to_datetime(np.atleast_1d(ds["time"].values))
    out: list[tuple[float, list]] = []
    for frac, ts in zip(fracs.tolist(), times, strict=True):
        grp = by_date.get(ts.date())
        if grp is not None:
            out.append((float(frac), grp))
    return out


def load_composite(
    items,
    gbox: GeoBox,
    bands: list[str],
    *,
    buffer_px: int = 8,
    chunks: dict | None = None,
    pool: int = 8,
) -> xr.DataArray:
    """SCL-masked median composite on *gbox* → ``(band, y, x)`` float32, NaN where masked.

    ``groupby="solar_day"`` mosaics the frames of one pass into one timestep, so a
    chip on a tile boundary is filled from both frames instead of producing two
    half-empty timesteps. With a single date the median is a no-op.

    Pass ``chunks`` for a large box to get a lazy dask array; leave it ``None``
    for a chip, which is one block read and wants no scheduler at all.
    """
    load_gbox = _buffered(gbox, buffer_px)
    ds = odc_load(
        items,
        bands=list(bands) + ["SCL"],
        geobox=load_gbox,
        resampling={"SCL": "nearest", "*": "bilinear"},
        groupby="solar_day",
        chunks=chunks,
        pool=None if chunks is not None else pool,
        fail_on_error=False,
    )
    clear = ~ds["SCL"].isin(list(BAD_SCL))
    arr = ds[list(bands)].where(clear).to_array(dim="band").astype("float32")
    # 0 is L2A's own no-data, so a masked pixel must not come back as a real
    # reflectance of zero. Drop it before the median, not after.
    arr = arr.where(arr > 0)
    if "time" in arr.dims:
        if chunks is not None:
            arr = arr.chunk({"time": -1})  # nanmedian needs time in one chunk
        arr = arr.median(dim="time", skipna=True)
    if buffer_px > 0:
        arr = arr.isel(y=slice(buffer_px, -buffer_px), x=slice(buffer_px, -buffer_px))
    return arr


def valid_fraction(arr: np.ndarray) -> float:
    """Fraction of pixels finite in EVERY band of a ``(band, y, x)`` array."""
    return float(np.isfinite(arr).all(axis=0).mean())


def center_is_valid(arr: np.ndarray) -> bool:
    """Is the centre pixel finite in every band?

    The head pools the **centre token**, so a chip whose centre sits under cloud
    trains on a filled zero however clean the rest of it is. With one date per
    month there is no median to repair it, so this is checked, not assumed.
    """
    c = arr.shape[-1] // 2
    return bool(np.isfinite(arr[:, c, c]).all())
