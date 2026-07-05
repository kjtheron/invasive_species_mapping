"""Shared IO helpers for local paths.

All streaming file access across the pipeline is consolidated here.
Phase 0 is local-first — artifacts live under ``data/`` (see CLAUDE.md).
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.crs import CRS
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from shapely import from_wkb

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"


def ensure_parent(uri: str) -> None:
    """Create parent directories for a local path."""
    Path(uri).parent.mkdir(parents=True, exist_ok=True)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def list_parquet_files(root: str, *, recursive: bool = False) -> list[str]:
    """List ``*.parquet`` files under *root*, excluding ``_tmp_`` dirs."""
    base = Path(root)
    if not base.exists():
        return []
    pattern = "**/*.parquet" if recursive else "*.parquet"
    return [str(p) for p in sorted(base.glob(pattern)) if "/_tmp_" not in str(p)]


# Raster helpers


def write_cog(
    arr: np.ndarray,
    transform: rasterio.transform.Affine,
    crs: str | CRS,
    out_uri: str,
    *,
    dtype: str = "float32",
    nodata: float | int = 0.0,
    profile_name: str = "deflate",
    blockxsize: int = 256,
    blockysize: int = 256,
) -> str:
    """Write a (bands, H, W) or (H, W) array as a COG to a local path. Returns *out_uri*."""
    if arr.ndim == 2:
        arr = arr[np.newaxis, :, :]
    arr = np.asarray(arr, dtype=dtype)

    if isinstance(crs, str):
        crs = CRS.from_string(crs)

    with tempfile.TemporaryDirectory() as tmp:
        raw_tif = str(Path(tmp) / "raw.tif")
        cog_tif = str(Path(tmp) / "cog.tif")

        with rasterio.open(
            raw_tif,
            "w",
            driver="GTiff",
            height=arr.shape[1],
            width=arr.shape[2],
            count=arr.shape[0],
            dtype=dtype,
            crs=crs,
            transform=transform,
            nodata=nodata,
        ) as dst:
            dst.write(arr)

        profile = cog_profiles.get(profile_name)
        profile.update(dtype=dtype, nodata=nodata, blockxsize=blockxsize, blockysize=blockysize)
        cog_translate(
            raw_tif,
            cog_tif,
            profile,
            in_memory=True,
            quiet=True,
            allow_intermediate_compression=True,
        )

        Path(out_uri).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cog_tif, out_uri)

    return out_uri


# Parquet helpers


def read_parquet_df(uri: str) -> pd.DataFrame:
    """Read a (non-geo) Parquet file into a pandas DataFrame."""
    return pd.read_parquet(uri)


def write_parquet_df(df: pd.DataFrame, uri: str) -> None:
    """Write a pandas DataFrame to Parquet (ZSTD)."""
    ensure_parent(uri)
    df.to_parquet(uri, index=False, compression="zstd")


# GeoParquet helpers
#
# ponytail: geometry is stored as WKB bytes + a ``__crs__`` column rather than
# native GeoParquet metadata, for back-compat with artifacts already written in
# this format. ``read_gdf`` falls back to ``gpd.read_parquet`` for native files.
# Switch writes to ``gdf.to_parquet`` once the old artifacts are regenerated.


def write_gdf_parquet(gdf: gpd.GeoDataFrame, uri: str) -> None:
    """Write a GeoDataFrame to Parquet (geometry as WKB + __crs__ column)."""
    geom_col = gdf.geometry.name
    pdf = gdf.drop(columns=[geom_col]).copy()
    pdf["geometry"] = gdf.geometry.to_wkb()
    pdf["__crs__"] = gdf.crs.to_wkt() if gdf.crs else None
    ensure_parent(uri)
    pdf.to_parquet(uri, index=False, compression="zstd")


def read_gdf(uri: str) -> gpd.GeoDataFrame:
    """Read a geo file into a GeoDataFrame.

    Parquet written by :func:`write_gdf_parquet` (geometry WKB + ``__crs__``)
    is decoded directly; native GeoParquet falls back to ``gpd.read_parquet``.
    Non-parquet (shp, geojson) uses ``gpd.read_file``.
    """
    if uri.endswith(".parquet"):
        pdf = pd.read_parquet(uri)
        if "__crs__" not in pdf.columns:
            return gpd.read_parquet(uri)
        crs_val = pdf["__crs__"].iloc[0] if len(pdf) > 0 else None
        if hasattr(crs_val, "item"):
            crs_val = crs_val.item()
        pdf = pdf.drop(columns=["__crs__"])
        return gpd.GeoDataFrame(pdf, geometry=from_wkb(pdf["geometry"].to_numpy()), crs=crs_val)
    return gpd.read_file(uri)
