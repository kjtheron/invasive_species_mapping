"""Shared monthly-median compositing helpers.

``monthly_median`` and ``_transform_from_da`` are the compositing primitives used by
both ``cmrv.ingest.chips`` (training chips) and ``cmrv.infer`` (wall-to-wall
inference); ``load_pipeline_config`` reads ``configs/pipeline.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import rasterio
import xarray as xr
import yaml
from rasterio.transform import from_bounds


def load_pipeline_config(path: str | Path = "configs/pipeline.yaml") -> dict:
    """Load configs/pipeline.yaml and return as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def monthly_median(da: xr.DataArray) -> xr.DataArray:
    """Pixel-wise median over time, skipping NaN. ``(time,band,y,x)`` → ``(band,y,x)``."""
    return da.median(dim="time", skipna=True)


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
