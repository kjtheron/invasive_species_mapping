"""Filter helper for the observation store.

``load_training_labels``
    Spatial + species filter over the merged store — the single entry-point
    for training configs. Returns a geopandas GeoDataFrame in EPSG:4326 (native
    geometry), clipped to ``aoi_uri`` and filtered to ``species_subset``
    (scientific-name fragments). The per-source summary is ``observations.write_summary``.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from loguru import logger

from cmrv.io import read_gdf
from cmrv.labels.observations import PROCESSED_ROOT, read_all


def _dedup_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per obs_id — the latest by ingested_at."""
    return df.sort_values("ingested_at").drop_duplicates("obs_id", keep="last")


def load_training_labels(
    aoi_uri: str,
    species_subset: list[str] | None = None,
    sources: list[str] | None = None,
    min_coord_uncertainty_m: float | None = None,
    max_coord_uncertainty_m: float | None = 500.0,
    date_min: str | None = "2018-01-01",
    date_max: str | None = None,
    geom_types: list[str] | None = None,
    min_weight: float | None = None,
    min_cover_pct: float | None = None,
    bbox: tuple[float, float, float, float] | None = None,
    root: str = PROCESSED_ROOT,
) -> gpd.GeoDataFrame:
    """Load training labels filtered to an AOI and optional species subset.

    Filters: source, geom_type, weight, event_date range, coord_uncertainty
    range (nulls kept), cover, and species (scientific-name substrings).
    Deduplicates on obs_id (latest ingested_at), then clips to the AOI polygon.

    ``min_cover_pct`` is the **pure-pixel gate** — keep only rows whose measured
    ``cover_pct`` ≥ threshold (drops null-cover rows). Off by default; enable
    (~60) once cover-bearing data is in the store.

    Returns a GeoDataFrame with native geometry, EPSG:4326.
    """
    df = _dedup_latest(read_all(root, sources=sources, bbox=bbox))  # sources/bbox pushed down
    mask = pd.Series(True, index=df.index)

    if geom_types:
        mask &= df["geom_type"].isin(geom_types)
    if min_weight is not None:
        mask &= df["weight"] >= min_weight
    if min_cover_pct is not None:
        mask &= df["cover_pct"].notna() & (df["cover_pct"] >= min_cover_pct)

    ed = pd.to_datetime(df["event_date"], errors="coerce")
    if date_min:
        mask &= ed.isna() | (ed >= pd.Timestamp(date_min))
    if date_max:
        mask &= ed <= pd.Timestamp(date_max)

    coord = df["coord_uncertainty_m"]
    if max_coord_uncertainty_m is not None:
        mask &= coord.isna() | (coord <= max_coord_uncertainty_m)
    if min_coord_uncertainty_m is not None:
        mask &= coord.isna() | (coord >= min_coord_uncertainty_m)

    if species_subset:
        sp = pd.Series(False, index=df.index)
        for frag in species_subset:
            sp |= df["species_normalized"].str.contains(frag, regex=False, na=False)
        mask &= sp

    df = df[mask]
    logger.info("load_training_labels: {} rows after filters", len(df))

    aoi = read_gdf(aoi_uri).to_crs("EPSG:4326")
    aoi_union = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union
    gdf = df[df.geometry.intersects(aoi_union)].copy()
    logger.info("after AOI clip: {} rows", len(gdf))

    return gdf
