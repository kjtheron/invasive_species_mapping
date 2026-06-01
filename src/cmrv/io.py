"""Shared IO helpers for local paths and gs:// URIs.

All streaming file access across the pipeline is consolidated here.
See CLAUDE.md § "Consolidated IO helpers" for the full convention list.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import duckdb
import fsspec
import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import rasterio
import yaml
from loguru import logger
from rasterio.crs import CRS
from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from shapely import from_wkb

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
CACHE_DIR: Path = DATA_DIR / "cache"


# Path utilities


def is_gcs(uri: str) -> bool:
    return uri.startswith("gs://")


def to_vsi(uri: str) -> str:
    """Convert a gs:// URI to GDAL's /vsigs/ virtual path for rasterio/GDAL."""
    return "/vsigs/" + uri[len("gs://") :] if is_gcs(uri) else uri


def ensure_parent(uri: str) -> None:
    """Create parent directories for a local path (no-op for gs:// URIs)."""
    if not is_gcs(uri):
        Path(uri).parent.mkdir(parents=True, exist_ok=True)


# Config helpers


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file (local or gs://) and return as a dict."""
    with fsspec.open(str(path), "r") as f:
        return yaml.safe_load(f)


# Partition discovery


def list_parquet_files(root: str, *, recursive: bool = False) -> list[str]:
    """List ``*.parquet`` files under *root* (local or gs://), excluding ``_tmp_`` dirs."""
    fs, fs_path = fsspec.core.url_to_fs(root)
    if not fs.exists(fs_path):
        return []
    protocol = fs.protocol if isinstance(fs.protocol, str) else fs.protocol[0]
    pattern = f"{fs_path}/**/*.parquet" if recursive else f"{fs_path}/*.parquet"
    hits = fs.glob(pattern)
    prefix = f"{protocol}://" if protocol != "file" else ""
    return [f"{prefix}{h}" for h in hits if "/_tmp_" not in h]


# GDAL / GCS authentication
#
# GDAL's /vsigs/ driver does not read Application Default Credentials on its
# own. We extract the OAuth2 triple from the ADC file (or defer to a service
# account when GOOGLE_APPLICATION_CREDENTIALS points to one) and set the
# GS_OAUTH2_* env vars GDAL expects. Runs once per process.

_ADC_PATH = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
_GDAL_CONFIGURED = False


def configure_gdal_gcs() -> None:
    """Configure GDAL so /vsigs/ streaming works with ADC or a service account.

    Idempotent — safe to call repeatedly. Raises nothing; logs a warning if no
    credential source is available (public buckets still work via
    GS_NO_SIGN_REQUEST=YES).
    """
    global _GDAL_CONFIGURED
    if _GDAL_CONFIGURED:
        return

    # A service-account key file takes precedence — GDAL reads it natively.
    sa_key = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if sa_key and Path(sa_key).is_file():
        _GDAL_CONFIGURED = True
        logger.debug("GDAL /vsigs/ auth: service account key ({})", sa_key)
        return

    # Fall back to the gcloud ADC refresh token.
    if _ADC_PATH.is_file():
        try:
            adc = json.loads(_ADC_PATH.read_text())
        except json.JSONDecodeError:
            logger.warning("ADC file at {} is not valid JSON", _ADC_PATH)
            adc = {}
        if adc.get("type") == "authorized_user":
            os.environ.setdefault("GS_OAUTH2_REFRESH_TOKEN", adc["refresh_token"])
            os.environ.setdefault("GS_OAUTH2_CLIENT_ID", adc["client_id"])
            os.environ.setdefault("GS_OAUTH2_CLIENT_SECRET", adc["client_secret"])
            _GDAL_CONFIGURED = True
            logger.debug("GDAL /vsigs/ auth: ADC refresh token ({})", _ADC_PATH)
            return

    logger.warning("GDAL /vsigs/ auth: no credentials found — only public buckets will work")


# Raster helpers


def open_raster(uri: str) -> rasterio.DatasetReader:
    """Open a raster for streaming reads (GCS via /vsigs/, local as-is).

    Transparently configures GDAL GCS auth on first call so no caller needs
    to handle credentials.
    """
    if is_gcs(uri):
        configure_gdal_gcs()
    return rasterio.open(to_vsi(uri))


def upload_file(local_path: str, gcs_uri: str, chunk: int = 8 * 1024 * 1024) -> None:
    """Stream a local file to GCS in chunks (avoids loading the whole file into memory)."""
    with open(local_path, "rb") as src, fsspec.open(gcs_uri, "wb") as dst:
        shutil.copyfileobj(src, dst, length=chunk)


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
    """Write a (bands, H, W) or (H, W) array as a COG (local or GCS). Returns *out_uri*."""
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

        if is_gcs(out_uri):
            upload_file(cog_tif, out_uri)
        else:
            Path(out_uri).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cog_tif, out_uri)

    return out_uri


# DuckDB connection factory


def _gcs_bearer_token() -> str:
    """Fetch a short-lived OAuth2 token from Application Default Credentials."""
    import google.auth
    import google.auth.transport.requests

    creds, _ = google.auth.default()
    creds.refresh(google.auth.transport.requests.Request())
    return creds.token


def _duckdb_con(uri: str = "") -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection; configure GCS credentials when uri is a gs:// path."""
    con = duckdb.connect()
    if is_gcs(uri):
        con.execute("LOAD httpfs;")
        token = _gcs_bearer_token()
        con.execute(
            f"CREATE OR REPLACE SECRET gcs_secret"
            f" (TYPE GCS, PROVIDER CONFIG, bearer_token '{token}');"
        )
    return con


# Plain Parquet helpers (no geometry)


def _duckdb_copy(df: pd.DataFrame, uri: str) -> None:
    """Register a DataFrame and COPY it to Parquet (ZSTD) at *uri*."""
    con = _duckdb_con(uri)
    con.register("_copy_tbl", df)
    ensure_parent(uri)
    con.execute(f"COPY _copy_tbl TO '{uri}' (FORMAT PARQUET, COMPRESSION ZSTD)")


def read_parquet_df(uri: str) -> pd.DataFrame:
    """Read a (non-geo) Parquet file into a pandas DataFrame via DuckDB."""
    return _duckdb_con(uri).execute(f"SELECT * FROM read_parquet('{uri}')").df()


def write_parquet_df(df: pd.DataFrame, uri: str) -> None:
    """Write a pandas DataFrame to Parquet via DuckDB (ZSTD, local or gs://)."""
    _duckdb_copy(df, uri)


# GeoParquet helpers


def write_gdf_parquet(gdf: gpd.GeoDataFrame, uri: str) -> None:
    """Write a GeoDataFrame to Parquet via DuckDB (geometry as WKB + __crs__)."""
    geom_col = gdf.geometry.name
    pdf = gdf.drop(columns=[geom_col]).copy()
    pdf[geom_col] = gdf.geometry.to_wkb()
    pdf["__crs__"] = gdf.crs.to_wkt() if gdf.crs else None
    _duckdb_copy(pdf, uri)


def _read_geoparquet_crs(uri: str) -> tuple[str, str | dict | None]:
    """Extract (primary_column, crs) from a GeoParquet ``geo`` metadata block.

    Used as a backward-compat fallback for files written by geopandas.
    """
    with fsspec.open(uri, "rb") as f:
        meta = pq.ParquetFile(f).schema_arrow.metadata or {}
    geo = json.loads(meta.get(b"geo", b"{}"))
    primary = geo.get("primary_column", "geometry")
    crs = geo.get("columns", {}).get(primary, {}).get("crs")
    return primary, crs


def read_gdf(uri: str) -> gpd.GeoDataFrame:
    """Read a geo file into a GeoDataFrame.

    Parquet: DuckDB streams directly from GCS or local disk. CRS is recovered from
    the ``__crs__`` column (files written by this module) or from the ``geo`` Parquet
    metadata block (legacy geopandas-written files).
    Non-parquet (shp, geojson): geopandas.read_file via fsspec file-object.
    Both local paths and ``gs://`` URIs are supported.
    """
    if uri.endswith(".parquet"):
        con = _duckdb_con(uri)
        pdf = con.execute(f"SELECT * FROM read_parquet('{uri}')").df()

        if "__crs__" in pdf.columns:
            crs_val: str | dict | None = pdf["__crs__"].iloc[0] if len(pdf) > 0 else None
            if hasattr(crs_val, "item"):
                crs_val = crs_val.item()
            pdf = pdf.drop(columns=["__crs__"])
            geom_col = "geometry"
        else:
            geom_col, crs_val = _read_geoparquet_crs(uri)

        pdf[geom_col] = from_wkb(pdf[geom_col].apply(bytes))
        return gpd.GeoDataFrame(pdf, geometry=geom_col, crs=crs_val)

    if is_gcs(uri):
        with fsspec.open(uri, "rb") as f:
            return gpd.read_file(f)
    return gpd.read_file(uri)
