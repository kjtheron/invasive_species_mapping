"""Monthly median compositing and COG write for Stage 2.

Pipeline per tile × month:
    MPC STAC query → stackstac lazy stack → SCL cloud mask
    → time-median composite → Cloud-Optimized GeoTIFF on GCS.

Output path convention: ``<raw_prefix>/tile_id=<N>/<month_label>.tif``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
import yaml
from loguru import logger
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from shapely.geometry.base import BaseGeometry

from cmrv.ingest.cloud_mask import apply_scl_mask
from cmrv.ingest.stac import query_items, stack_items
from cmrv.io import write_cog

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_pipeline_config(path: str | Path = "configs/pipeline.yaml") -> dict:
    """Load configs/pipeline.yaml and return as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Compositing
# ---------------------------------------------------------------------------


def monthly_median(da: xr.DataArray) -> xr.DataArray:
    """Compute the pixel-wise median over the time dimension, skipping NaN.

    Args:
        da: (time, band, y, x) masked DataArray (SCL already removed).

    Returns:
        (band, y, x) DataArray.
    """
    return da.median(dim="time", skipna=True)


# ---------------------------------------------------------------------------
# COG writer
# ---------------------------------------------------------------------------


def _transform_from_da(da: xr.DataArray) -> rasterio.transform.Affine:
    """Derive an Affine transform from evenly-spaced x/y pixel-center coords."""
    x = da.x.values
    y = da.y.values
    dx = abs(float(x[1] - x[0]))
    dy = abs(float(y[1] - y[0]))
    west = float(x.min()) - dx / 2
    east = float(x.max()) + dx / 2
    south = float(y.min()) - dy / 2
    north = float(y.max()) + dy / 2
    return from_bounds(west, south, east, north, len(x), len(y))


def write_composite_cog(
    composite: xr.DataArray,
    out_uri: str,
    epsg: int = 32734,
    nodata: float = 0.0,
) -> str:
    """Write a (band, y, x) composite DataArray as a Cloud-Optimized GeoTIFF.

    Delegates to ``cmrv.io.write_cog`` for the temp-file + translate + upload
    pipeline.  See ``write_cog`` for GCS streaming details.
    """
    arr = np.asarray(composite.values, dtype="float32")
    transform = _transform_from_da(composite)
    return write_cog(arr, transform, CRS.from_epsg(epsg), out_uri, nodata=nodata)


# ---------------------------------------------------------------------------
# Per-tile runner
# ---------------------------------------------------------------------------


def ingest_tile_month(
    tile_id: int,
    tile_geom_wgs84: BaseGeometry,
    date_start: str,
    date_end: str,
    month_label: str,
    bands: list[str],
    out_prefix: str,
    *,
    resolution_m: int = 10,
    cloud_cover_max: int = 40,
    epsg: int = 32734,
) -> str | None:
    """Full Stage 2 pipeline for one tile × one month.

    Queries MPC STAC → cloud-masks → median composites → writes COG.

    Returns:
        Output URI string, or ``None`` if no valid scenes were found.
    """
    items = query_items(
        tile_geom_wgs84,
        date_start,
        date_end,
        cloud_cover_max=cloud_cover_max,
    )
    if len(items) == 0:
        logger.warning(
            "tile {}: no S2 items for {}/{} — skipping",
            tile_id,
            date_start,
            date_end,
        )
        return None

    da = stack_items(
        items,
        bands,
        tile_geom_wgs84=tile_geom_wgs84,
        resolution_m=resolution_m,
        epsg=epsg,
    )
    da = apply_scl_mask(da)
    composite = monthly_median(da)

    out_uri = f"{out_prefix}/tile_id={tile_id}/{month_label}.tif"
    write_composite_cog(composite, out_uri, epsg=epsg)
    logger.success("tile {}: wrote {}", tile_id, out_uri)
    return out_uri


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------


def run_ingest(
    pipeline_cfg: dict,
    *,
    month_label: str | None = None,
    tile_id: int | None = None,
) -> list[str]:
    """Run Stage 2 for all tiles × all (or one) month from pipeline config.

    Args:
        pipeline_cfg: parsed ``configs/pipeline.yaml`` dict.
        month_label: restrict to a single month label (e.g. ``"oct"``).
            ``None`` runs all active months in the config.
        tile_id: restrict to a single tile by ID. ``None`` runs all tiles.

    Returns:
        List of output URIs written.
    """
    import geopandas as gpd

    from cmrv.io import read_gdf

    bands: list[str] = pipeline_cfg["s2_bands"]
    cloud_max: int = pipeline_cfg.get("cloud_cover_max", 40)
    raw_prefix: str = pipeline_cfg["paths"]["raw_prefix"]
    tiles_uri: str = pipeline_cfg["paths"]["tiles"]

    months = [m for m in pipeline_cfg["months"] if month_label is None or m["label"] == month_label]
    if not months:
        raise ValueError(f"No months matched label={month_label!r} in pipeline config.")

    tiles_gdf: gpd.GeoDataFrame = read_gdf(tiles_uri)
    if "tile_id" not in tiles_gdf.columns:
        tiles_gdf["tile_id"] = range(len(tiles_gdf))

    if tile_id is not None:
        tiles_gdf = tiles_gdf[tiles_gdf["tile_id"] == tile_id]
        if tiles_gdf.empty:
            raise ValueError(f"tile_id={tile_id} not found in {tiles_uri}")

    # Reproject to WGS84 for STAC bounds query
    tiles_wgs84 = tiles_gdf.to_crs("EPSG:4326")

    written: list[str] = []
    for row, row_wgs84 in zip(tiles_gdf.itertuples(), tiles_wgs84.itertuples(), strict=True):
        tid = int(row.tile_id)
        geom_wgs84 = row_wgs84.geometry
        for month in months:
            uri = ingest_tile_month(
                tile_id=tid,
                tile_geom_wgs84=geom_wgs84,
                date_start=month["start"],
                date_end=month["end"],
                month_label=month["label"],
                bands=bands,
                out_prefix=raw_prefix,
                cloud_cover_max=cloud_max,
            )
            if uri:
                written.append(uri)

    return written
