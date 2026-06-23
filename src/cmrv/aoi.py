"""AOI utilities: Western Cape province boundary + tile grid (scales to SA later)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from loguru import logger
from shapely.geometry import box
from shapely.ops import unary_union

SA_PROVINCIAL_SHP = Path("data/aoi/SA_Provincial_bnd_dd.shp")

WC_PROVINCE_NAME = "Western Cape"
WC_BUFFER_M = 1_000.0


def fetch_western_cape(
    source: str | Path = SA_PROVINCIAL_SHP,
    buffer_m: float = WC_BUFFER_M,
    out_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Extract the Western Cape province polygon from the SA provincial boundary shapefile.

    Filters ``PROVINCE == "Western Cape"``, dissolves multi-part geometry, and
    applies a +1 km buffer in UTM 34S (metric) before returning in ``out_crs``.
    """
    src = Path(source)
    if not src.exists():
        raise FileNotFoundError(f"Provincial boundary source not found: {src}")
    logger.info("loading provincial boundaries from {}", src)
    gdf = gpd.read_file(src)
    wc = gdf[gdf["PROVINCE"] == WC_PROVINCE_NAME]
    if wc.empty:
        raise ValueError(
            f"no features with PROVINCE == {WC_PROVINCE_NAME!r} in {src}; "
            f"saw {sorted(gdf['PROVINCE'].unique())}"
        )
    if wc.crs is None:
        wc = wc.set_crs("EPSG:4148")
    wc_m = wc.to_crs("EPSG:32734")
    dissolved = unary_union(wc_m.geometry)
    buffered = dissolved.buffer(buffer_m) if buffer_m > 0 else dissolved
    out = gpd.GeoDataFrame(
        {"name": [WC_PROVINCE_NAME], "admin1_iso": ["ZA-WC"]},
        geometry=[buffered],
        crs="EPSG:32734",
    ).to_crs(out_crs)
    area_km2 = gpd.GeoSeries([buffered], crs="EPSG:32734").area.iloc[0] / 1e6
    logger.info("Western Cape polygon: area = {:.0f} km^2 (buffer = {:.0f} m)", area_km2, buffer_m)
    return out


def build_tile_grid(
    aoi: gpd.GeoDataFrame,
    tile_km: float,
    crs: str = "EPSG:32734",
    min_overlap_frac: float = 0.01,
) -> gpd.GeoDataFrame:
    """Build a square tile grid covering the AOI. Grid is computed in the given metric CRS.

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
