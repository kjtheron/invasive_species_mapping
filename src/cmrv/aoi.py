"""AOI utilities: SA province boundaries (WC or national) + equal-area tile grid.

Boundary source: **GeoBoundaries gbOpen ADM1** (provinces), CC-BY 4.0
(Runfola et al. 2020). Downloaded + cached under ``data/aoi/raw/``; pass a local
``source`` file to override. ``fetch_provinces`` dissolves any subset (or all, for
national SA); tile/block grids build in :data:`SA_ALBERS` equal-area.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely import get_coordinates, make_valid
from shapely.geometry import box
from shapely.ops import unary_union

GEOBOUNDARIES_API = "https://www.geoboundaries.org/api/current/gbOpen/{iso3}/{adm}/"
GEOBOUNDARIES_RAW = (
    "https://raw.githubusercontent.com/wmgeolab/geoBoundaries/main/releaseData/"
    "gbOpen/{iso3}/{adm}/geoBoundaries-{iso3}-{adm}.geojson"
)
ZAF_ADM1_CACHE = Path("data/aoi/raw/geoBoundaries-ZAF-ADM1.geojson")

WC_PROVINCE_NAME = "Western Cape"
WC_BUFFER_M = 1_000.0
# Prince Edward Islands are administratively WC but ~2000 km offshore (~-46.9°S);
# drop anything south of this so the tile grid doesn't span the ocean.
MAINLAND_MIN_LAT = -35.0

# National equal-area CRS for tile/block grids over all of South Africa: Albers Equal
# Area Conic, standard parallels by the 1/6 rule over SA's ~-22..-35° lat range, central
# meridian 25°E. Equal-area everywhere in SA — unlike UTM 34S, which distorts badly for
# the KZN/EC/Limpopo/Mpumalanga labels (zones 35–36). The official SA-BSU Albers
# (EPSG:9219) isn't in the bundled PROJ db, so use a PROJ4 string (ref: OSGeo SA-Albers).
SA_ALBERS = (
    "+proj=aea +lat_1=-24 +lat_2=-33 +lat_0=0 +lon_0=25 +x_0=0 +y_0=0 "
    "+datum=WGS84 +units=m +no_defs"
)


def utm_epsg(lon: float, lat: float) -> int:
    """UTM EPSG for a lon/lat — matches Sentinel-2's native MGRS UTM zone.

    Used to extract each chip / inference box in its **own** S2 zone (no cross-zone
    resampling), rather than forcing everything onto one national UTM zone.
    """
    return (32600 if lat >= 0 else 32700) + int((lon + 180) // 6) + 1


def _urlopen(url: str, timeout: float):
    req = urllib.request.Request(url, headers={"User-Agent": "catchment-mrv"})
    return urllib.request.urlopen(req, timeout=timeout)


def fetch_geoboundaries_adm1(
    iso3: str = "ZAF",
    adm: str = "ADM1",
    cache: str | Path = ZAF_ADM1_CACHE,
) -> gpd.GeoDataFrame:
    """Download + cache the GeoBoundaries gbOpen ADM1 (province) layer (CC-BY 4.0)."""
    cache = Path(cache)
    if cache.exists():
        logger.info("using cached boundaries {}", cache)
        return gpd.read_file(cache)

    try:
        meta_url = GEOBOUNDARIES_API.format(iso3=iso3, adm=adm)
        logger.info("fetching GeoBoundaries metadata {}", meta_url)
        with _urlopen(meta_url, timeout=60) as r:
            gj_url = json.load(r)["gjDownloadURL"]
    except Exception as e:  # API hiccup → fall back to the canonical raw URL
        logger.warning("GeoBoundaries API failed ({}); using direct raw URL", e)
        gj_url = GEOBOUNDARIES_RAW.format(iso3=iso3, adm=adm)

    logger.info("downloading {}", gj_url)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with _urlopen(gj_url, timeout=120) as r:
        cache.write_bytes(r.read())
    return gpd.read_file(cache)


def _n_vertices(geom) -> int:
    return len(get_coordinates(geom))


def fetch_provinces(
    provinces: list[str] | None = None,
    source: str | Path | None = None,
    buffer_m: float = WC_BUFFER_M,
    simplify_m: float = 100.0,
    out_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Dissolve one or more SA provinces (GeoBoundaries gbOpen ADM1) into one AOI polygon.

    ``provinces=None`` → **all** provinces (national South Africa). Cleans the raw
    boundary: make-valid, drop the sub-Antarctic Prince Edward Islands (~2000 km
    offshore), simplify vertices (``simplify_m`` metres, in UTM 34S), dissolve, then a
    +``buffer_m`` buffer. Returns one feature in ``out_crs``.
    """
    gdf = gpd.read_file(source) if source else fetch_geoboundaries_adm1()
    name_col = "shapeName" if "shapeName" in gdf.columns else "PROVINCE"
    norm = gdf[name_col].astype(str).str.strip().str.lower()
    if provinces:
        wanted = {p.strip().lower() for p in provinces}
        sel = gdf[norm.isin(wanted)]
        missing = wanted - set(norm)
        if missing:
            raise ValueError(
                f"provinces {sorted(missing)} not in column {name_col!r}; "
                f"saw {sorted(gdf[name_col].astype(str).unique())}"
            )
        label = provinces[0] if len(provinces) == 1 else f"{len(provinces)} provinces"
    else:
        sel = gdf
        label = "South Africa"
    sel = sel.set_crs("EPSG:4326") if sel.crs is None else sel.to_crs("EPSG:4326")

    # --- clean vertices ---
    parts = sel.explode(index_parts=False, ignore_index=True)
    parts["geometry"] = parts.geometry.apply(make_valid)
    n_parts = len(parts)
    parts = parts[parts.geometry.representative_point().y > MAINLAND_MIN_LAT]
    if len(parts) < n_parts:
        logger.info("dropped {} far-offshore part(s) (Prince Edward Is.)", n_parts - len(parts))

    sel_m = parts.to_crs(SA_ALBERS)  # equal-area: correct area + simplify anywhere in SA
    dissolved = make_valid(unary_union(sel_m.geometry))
    nv_before = _n_vertices(dissolved)
    if simplify_m > 0:
        dissolved = make_valid(dissolved.simplify(simplify_m, preserve_topology=True))
    buffered = dissolved.buffer(buffer_m) if buffer_m > 0 else dissolved

    area_km2 = gpd.GeoSeries([buffered], crs=SA_ALBERS).area.iloc[0] / 1e6
    logger.info(
        "{}: vertices {} → {} (simplify {} m), area {:.0f} km^2, buffer {:.0f} m",
        label,
        nv_before,
        _n_vertices(buffered),
        simplify_m,
        area_km2,
        buffer_m,
    )
    return gpd.GeoDataFrame({"name": [label]}, geometry=[buffered], crs=SA_ALBERS).to_crs(out_crs)


def fetch_western_cape(
    source: str | Path | None = None,
    buffer_m: float = WC_BUFFER_M,
    simplify_m: float = 100.0,
    out_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Western Cape province polygon — thin wrapper over :func:`fetch_provinces`."""
    return fetch_provinces(
        [WC_PROVINCE_NAME], source=source, buffer_m=buffer_m, simplify_m=simplify_m, out_crs=out_crs
    )


def build_tile_grid(
    aoi: gpd.GeoDataFrame,
    tile_km: float,
    crs: str = SA_ALBERS,
    min_overlap_frac: float = 0.01,
) -> gpd.GeoDataFrame:
    """Build a square tile grid covering the AOI. Grid is computed in the given metric CRS.

    Defaults to :data:`SA_ALBERS` (national equal-area) so tiles are true-square across all
    provinces; UTM 34S would skew tiles far from zone 34 (KZN/EC/Limpopo/Mpumalanga).

    `min_overlap_frac` is the minimum fraction of a tile's area that must fall inside the AOI
    for the tile to be kept. The default (1%) filters sliver tiles introduced by reprojection
    drift and edge tiles with too little AOI coverage to be worth processing.
    """
    aoi_m = aoi.to_crs(crs)
    minx, miny, maxx, maxy = aoi_m.total_bounds
    step = tile_km * 1000.0
    xs = np.arange(minx, maxx, step)
    ys = np.arange(miny, maxy, step)
    tiles = [box(x, y, x + step, y + step) for x in xs for y in ys]
    grid = gpd.GeoDataFrame({"tile_id": range(len(tiles))}, geometry=tiles, crs=crs)
    aoi_union = unary_union(aoi_m.geometry)
    tile_area = step * step
    overlap = grid.intersection(aoi_union).area
    keep = grid[overlap > min_overlap_frac * tile_area].reset_index(drop=True)
    keep["tile_id"] = range(len(keep))
    return keep
