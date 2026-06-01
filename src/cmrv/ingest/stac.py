"""STAC query + stackstac lazy loader for Sentinel-2 L2A (MPC).

No subscription key is required — MPC S2 L2A is publicly accessible.
`planetary_computer.sign_inplace` adds time-limited SAS tokens to asset URLs.
"""

from __future__ import annotations

from typing import Any

import planetary_computer as pc
import pystac_client
import stackstac
import xarray as xr
from loguru import logger
from shapely.geometry.base import BaseGeometry

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"


def query_items(
    tile_geom_wgs84: BaseGeometry,
    date_start: str,
    date_end: str,
    *,
    cloud_cover_max: int = 40,
) -> Any:
    """Query MPC STAC for S2 L2A items intersecting a WGS84 tile geometry.

    Returns a signed pystac ItemCollection (may be empty).
    """
    cat = pystac_client.Client.open(STAC_URL, modifier=pc.sign_inplace)
    items = cat.search(
        collections=[COLLECTION],
        intersects=tile_geom_wgs84.__geo_interface__,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": cloud_cover_max}},
    ).item_collection()
    logger.debug(
        "STAC {}/{}: {} items (cloud_cover < {})",
        date_start,
        date_end,
        len(items),
        cloud_cover_max,
    )
    return items


def stack_items(
    items: Any,
    bands: list[str],
    *,
    tile_geom_wgs84: BaseGeometry,
    resolution_m: int = 10,
    epsg: int = 32734,
) -> xr.DataArray:
    """Stack pystac items into a lazy (time, band, y, x) DataArray.

    Includes the SCL band for downstream cloud masking. Spatial extent is
    clipped to ``tile_geom_wgs84`` bounds and reprojected to EPSG:``epsg``
    (UTM 34S for the Berg River catchment).
    """
    return stackstac.stack(
        items,
        assets=bands + ["SCL"],
        resolution=resolution_m,
        epsg=epsg,
        bounds_latlon=tile_geom_wgs84.bounds,
        dtype="float64",
        rescale=False,
        fill_value=float("nan"),
        chunksize=2048,
    )
