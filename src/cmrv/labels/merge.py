"""Merge + filter helpers for the WC observation store.

``merge_partitions``
    Union all source partitions, deduplicate on ``obs_id`` (keeping max
    ``ingested_at``), write ``summary.parquet``.

``load_training_labels``
    Spatial + species filter over the merged store — the single entry-point
    for training configs. Returns a geopandas GeoDataFrame in EPSG:4326 with
    WKB geometry deserialized to shapely geometries, clipped to ``aoi_uri``
    and filtered to ``species_subset`` (by GBIF usage_key or scientific name).
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely import from_wkb

from cmrv.io import read_gdf
from cmrv.labels.observations import WC_LABELS_ROOT, read_all, write_summary


def _dedup_latest(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per obs_id — the latest by ingested_at."""
    return df.sort_values("ingested_at").drop_duplicates("obs_id", keep="last")


def merge_partitions(
    root: str = WC_LABELS_ROOT,
    summary_uri: str = "data/labels/wc/summary.parquet",
) -> pd.DataFrame:
    """Union all source partitions, deduplicate on obs_id, write summary.

    Returns the summary DataFrame (per-source × per-category counts +
    fraction with non-null coord_uncertainty_m).
    """
    table = read_all(root)
    n_deduped = len(_dedup_latest(table))
    logger.info(
        "merge_partitions: {} raw rows → {} after dedup (dropped {})",
        len(table),
        n_deduped,
        len(table) - n_deduped,
    )
    return write_summary(root=root, out_uri=summary_uri)


def load_training_labels(
    aoi_uri: str,
    species_subset: list[str] | list[int] | None = None,
    sources: list[str] | None = None,
    min_coord_uncertainty_m: float | None = None,
    max_coord_uncertainty_m: float | None = 500.0,
    date_min: str | None = "2018-01-01",
    date_max: str | None = None,
    geom_types: list[str] | None = None,
    min_weight: float | None = None,
    min_cover_pct: float | None = None,
    root: str = WC_LABELS_ROOT,
) -> gpd.GeoDataFrame:
    """Load training labels filtered to an AOI and optional species subset.

    Filters: source, geom_type, weight, event_date range, coord_uncertainty
    range (nulls kept), cover, and species (GBIF usage_key ints or
    scientific-name substrings). Deduplicates on obs_id (latest ingested_at),
    then clips to the AOI polygon.

    ``min_cover_pct`` is the **pure-pixel gate** — keep only rows whose measured
    ``cover_pct`` ≥ threshold (drops null-cover rows). Off by default; enable
    (~60) once cover-bearing data is in the store.

    Returns a GeoDataFrame with WKB geometry deserialized to shapely, EPSG:4326.
    """
    df = _dedup_latest(read_all(root))
    mask = pd.Series(True, index=df.index)

    if sources:
        mask &= df["source"].isin(sources)
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
        int_keys = [s for s in species_subset if isinstance(s, int)]
        str_frags = [s for s in species_subset if isinstance(s, str)]
        sp = pd.Series(False, index=df.index)
        if int_keys:
            sp |= df["gbif_usage_key"].isin(int_keys)
        for frag in str_frags:
            sp |= df["species_normalized"].str.contains(frag, regex=False, na=False)
        mask &= sp

    df = df[mask]
    logger.info("load_training_labels: {} rows after filters", len(df))

    geometries = [from_wkb(bytes(b)) if b is not None else None for b in df["geometry"].tolist()]
    gdf = gpd.GeoDataFrame(df.drop(columns=["geometry"]), geometry=geometries, crs="EPSG:4326")

    aoi = read_gdf(aoi_uri).to_crs("EPSG:4326")
    aoi_union = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union
    gdf = gdf[gdf.geometry.intersects(aoi_union)].copy()
    logger.info("after AOI clip: {} rows", len(gdf))

    return gdf
