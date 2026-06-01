"""Merge + filter helpers for the WC NEMBA observation store.

``merge_partitions``
    Union all source partitions in the store, deduplicate on ``obs_id``
    (keeping max ``ingested_at``), write ``summary.parquet``.

``load_training_labels``
    Spatial + species filter over the merged store — the single entry-point
    for training configs. Returns a geopandas GeoDataFrame in EPSG:4326 with
    WKB geometry deserialized to shapely geometries, clipped to ``aoi_uri``
    and filtered to ``species_subset`` (by GBIF usage_key or scientific name).
"""

from __future__ import annotations

import duckdb
import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely import from_wkb

from cmrv.io import read_gdf
from cmrv.labels.observations import WC_LABELS_ROOT, read_all, write_summary


def merge_partitions(
    root: str = WC_LABELS_ROOT,
    summary_uri: str = "gs://ism-data/labels/wc/summary.parquet",
) -> pd.DataFrame:
    """Union all source partitions, deduplicate on obs_id, write summary.

    Returns the summary DataFrame (per-source × per-category counts +
    fraction with non-null coord_uncertainty_m).
    """
    table = read_all(root)
    con = duckdb.connect()
    con.register("_obs", table)
    n_raw = con.execute("SELECT COUNT(*) FROM _obs").fetchone()[0]
    n_deduped = con.execute("""
        SELECT COUNT(*) FROM (
            SELECT * FROM _obs
            QUALIFY ROW_NUMBER() OVER (PARTITION BY obs_id ORDER BY ingested_at DESC) = 1
        )
    """).fetchone()[0]
    logger.info(
        "merge_partitions: {} raw rows → {} after dedup (dropped {})",
        n_raw,
        n_deduped,
        n_raw - n_deduped,
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
    nemba_categories: list[str] | None = None,
    geom_types: list[str] | None = None,
    min_weight: float | None = None,
    root: str = WC_LABELS_ROOT,
) -> gpd.GeoDataFrame:
    """Load training labels filtered to an AOI and optional species subset.

    Parameters
    ----------
    aoi_uri:
        Path to an AOI GeoParquet (local or gs://). All returned observations
        lie within this polygon.
    species_subset:
        List of GBIF usage_keys (int) or scientific name fragments (str).
        If None, return all species (including nemba_category=None rows).
    sources:
        Restrict to these source values (e.g. ["gbif", "bioscape_line"]).
    min/max_coord_uncertainty_m:
        Filter on coord_uncertainty_m. Null values are kept unless
        ``min_coord_uncertainty_m`` is set.
    date_min:
        ISO date string; keep only rows where event_date ≥ date_min.
    nemba_categories:
        Filter on nemba_category (e.g. ["1a", "1b"]).
    geom_types:
        Filter on geom_type (e.g. ["point"]).
    min_weight:
        Drop rows with weight < min_weight.

    Returns
    -------
    geopandas.GeoDataFrame with WKB geometry deserialized to shapely, CRS=EPSG:4326.
    """
    table = read_all(root)
    con = duckdb.connect()
    con.register("_obs", table)

    conditions: list[str] = []

    if sources:
        quoted = ", ".join(f"'{s}'" for s in sources)
        conditions.append(f"source IN ({quoted})")

    if nemba_categories:
        quoted = ", ".join(f"'{c}'" for c in nemba_categories)
        conditions.append(f"nemba_category IN ({quoted})")

    if geom_types:
        quoted = ", ".join(f"'{g}'" for g in geom_types)
        conditions.append(f"geom_type IN ({quoted})")

    if min_weight is not None:
        conditions.append(f"weight >= {min_weight}")

    if date_min:
        conditions.append(f"(event_date IS NULL OR event_date >= DATE '{date_min}')")

    if date_max:
        conditions.append(f"event_date <= DATE '{date_max}'")

    if max_coord_uncertainty_m is not None:
        conditions.append(
            f"(coord_uncertainty_m IS NULL OR coord_uncertainty_m <= {max_coord_uncertainty_m})"
        )

    if min_coord_uncertainty_m is not None:
        conditions.append(
            f"(coord_uncertainty_m IS NULL OR coord_uncertainty_m >= {min_coord_uncertainty_m})"
        )

    if species_subset:
        int_keys = [s for s in species_subset if isinstance(s, int)]
        str_frags = [s for s in species_subset if isinstance(s, str)]
        sp_parts: list[str] = []
        if int_keys:
            sp_parts.append(f"gbif_usage_key IN ({', '.join(str(k) for k in int_keys)})")
        for frag in str_frags:
            escaped = frag.replace("'", "''")
            sp_parts.append(f"species_normalized LIKE '%{escaped}%'")
        if sp_parts:
            conditions.append(f"({' OR '.join(sp_parts)})")

    where = ("WHERE " + " AND ".join(conditions)) if conditions else ""

    df: pd.DataFrame = con.execute(f"""
        WITH deduped AS (
            SELECT * FROM _obs
            QUALIFY ROW_NUMBER() OVER (PARTITION BY obs_id ORDER BY ingested_at DESC) = 1
        )
        SELECT * FROM deduped {where}
    """).df()

    logger.info("load_training_labels: {} rows after filters", len(df))

    geometries = [from_wkb(bytes(b)) if b is not None else None for b in df["geometry"].tolist()]
    gdf = gpd.GeoDataFrame(df.drop(columns=["geometry"]), geometry=geometries, crs="EPSG:4326")

    aoi = read_gdf(aoi_uri).to_crs("EPSG:4326")
    aoi_union = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union
    gdf = gdf[gdf.geometry.intersects(aoi_union)].copy()
    logger.info("after AOI clip: {} rows", len(gdf))

    return gdf
