"""Unified observation schema for the WC occurrence store.

Canonical store: ``data/labels/processed/<dataset>/`` — one partition directory
per source dataset (mirrors ``data/labels/raw/<dataset>/``), Parquet, deduped on
``obs_id``. Every row carries ``obs_id``, ``source``, ``coord_uncertainty_m``,
and ``ingested_at`` for traceability. Multiple ``source`` values may live in one
dataset folder (e.g. BioSCape line + plot) — they are distinguished by the
``source`` column, not the path.

``class_id`` is deliberately **not** in this schema — training configs crosswalk
``species_normalized → class_id`` via
``configs/labels_schema.yaml → class_maps.<name>`` at ``make-split`` time.
"""

from __future__ import annotations

import datetime as dt
import os
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from cmrv.io import ensure_parent, list_parquet_files, write_parquet_df

PROCESSED_ROOT = "data/labels/processed"
# One partition directory per source dataset (mirrors data/labels/raw/<dataset>/).
KNOWN_DATASETS = frozenset(
    {"BioSCape_VegPlots_Berg_Eerste_2425", "mapwaps_olifants_doring", "sanlc_accuracy_points"}
)

# Canonical column order for all partitions
COLUMNS: tuple[str, ...] = (
    "obs_id",
    "source",
    "source_record_id",
    "source_url",
    "source_doi",
    "license",
    "species",
    "species_normalized",
    "taxon_rank",
    "gbif_usage_key",
    "geom_type",
    "coord_uncertainty_m",
    "event_date",
    "basis_of_record",
    "cover_pct",
    "weight",
    "ingested_at",
    "ingest_run_id",
    "aoi_admin1",
    "geometry",
)

# Subset that must be non-null in every row
REQUIRED_COLUMNS: tuple[str, ...] = (
    "obs_id",
    "source",
    "source_record_id",
    "geom_type",
    "weight",
    "ingested_at",
    "ingest_run_id",
    "aoi_admin1",
)


def make_run_id(source: str, when: dt.datetime | None = None) -> str:
    """Stable ingest run id for traceability: ``<source>_<UTC-compact>``."""
    when = when or dt.datetime.now(tz=dt.UTC)
    return f"{source}_{when.strftime('%Y%m%dT%H%M%SZ')}"


def partition_uri(dataset: str, root: str = PROCESSED_ROOT) -> str:
    """URI of the dataset-specific partition directory."""
    if dataset not in KNOWN_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {sorted(KNOWN_DATASETS)}")
    return f"{root}/{dataset}"


def to_obs_gdf(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Validate + canonicalise a GeoDataFrame for the observation store.

    Reprojects to EPSG:4326, fills missing nullable columns with None, orders columns,
    and keeps geometry **native** (no WKB) so the store is native GeoParquet — queryable
    by ``bbox`` at read time. Missing required columns raise ValueError.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must declare a CRS (expect EPSG:4326)")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    missing = [c for c in REQUIRED_COLUMNS if c not in gdf.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    gdf = gdf.copy()
    if gdf.geometry.name != "geometry":
        gdf = gdf.rename_geometry("geometry")
    for col in COLUMNS:
        if col != "geometry" and col not in gdf.columns:
            gdf[col] = None
    return gdf[[c for c in COLUMNS if c != "geometry"] + ["geometry"]]


def write_partition(
    gdf: gpd.GeoDataFrame,
    dataset: str,
    root: str = PROCESSED_ROOT,
    run_id: str | None = None,
) -> str:
    """Atomic partition overwrite with upsert semantics, keyed by source dataset.

    Concatenates new rows with any existing rows in the dataset partition,
    deduplicates on ``obs_id`` (keeping max ``ingested_at``), and writes the
    result back. Re-running the same ingest is idempotent. Multiple sources
    (e.g. BioSCape line + plot) can write to the same dataset folder; the
    ``source`` column keeps them distinct.

    Writes to a ``_tmp_<run_id>/`` file first, then ``os.replace`` into place
    so readers never see a torn partition.
    """
    if dataset not in KNOWN_DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}")
    partition_dir = partition_uri(dataset, root=root)
    run_id = run_id or make_run_id(dataset)

    new = to_obs_gdf(gdf)
    existing_files = list_parquet_files(partition_dir)
    frames = [gpd.read_parquet(f) for f in existing_files]
    n_existing = sum(len(f) for f in frames)

    combined = pd.concat([*frames, new], ignore_index=True)
    merged = gpd.GeoDataFrame(
        combined.sort_values("ingested_at").drop_duplicates("obs_id", keep="last"),
        geometry="geometry",
        crs="EPSG:4326",
    )

    tmp_file = f"{partition_dir}/_tmp_{run_id}/part-{run_id}.parquet"
    final_file = f"{partition_dir}/part-{run_id}.parquet"
    ensure_parent(tmp_file)
    merged.to_parquet(tmp_file, write_covering_bbox=True)  # native GeoParquet + bbox pushdown

    logger.info(
        "dataset={} new={} existing={} merged={} (dedupe kept {})",
        dataset,
        len(new),
        n_existing,
        n_existing + len(new),
        len(merged),
    )

    # Atomic promote: remove prior partition files then move the new one in.
    for old_file in existing_files:
        os.remove(old_file)
    os.replace(tmp_file, final_file)
    os.rmdir(f"{partition_dir}/_tmp_{run_id}")

    logger.success("wrote dataset={} → {} ({} rows after upsert)", dataset, final_file, len(merged))
    return final_file


def read_all(
    root: str = PROCESSED_ROOT,
    sources: list[str] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> gpd.GeoDataFrame:
    """Read the store as a GeoDataFrame (native geometry, EPSG:4326).

    ``sources`` filters the ``source`` column via pyarrow row-group pushdown; ``bbox``
    (minx, miny, maxx, maxy in lon/lat) prunes to intersecting geometries using the
    GeoParquet covering-bbox. Root-level files (e.g. summary.parquet) are excluded.
    """
    base = Path(root)
    files = [f for f in list_parquet_files(root, recursive=True) if Path(f).parent != base]
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}")
    kw: dict = {}
    if sources:  # pyarrow row-group pushdown on the source column
        kw["filters"] = [("source", "in", list(sources))]
    if bbox:  # GeoParquet covering-bbox pushdown (geopandas crashes on filters=None + bbox)
        kw["bbox"] = bbox
    frames = [gpd.read_parquet(f, **kw) for f in files]
    return gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    )


def summary(root: str = PROCESSED_ROOT) -> pd.DataFrame:
    """Per-source counts + fraction with non-null coord_uncertainty_m / cover_pct."""
    tbl = read_all(root)
    tbl["_has_unc"] = tbl["coord_uncertainty_m"].notna()
    tbl["_has_cover"] = tbl["cover_pct"].notna()
    out = (
        tbl.groupby("source", dropna=False)
        .agg(
            n=("obs_id", "size"),
            n_with_coord_unc=("_has_unc", "sum"),
            n_with_cover=("_has_cover", "sum"),
        )
        .reset_index()
    )
    out["pct_with_coord_unc"] = (100 * out["n_with_coord_unc"] / out["n"]).round(2)
    out["pct_with_cover"] = (100 * out["n_with_cover"] / out["n"]).round(2)
    return out.sort_values("source").reset_index(drop=True)


def write_summary(
    root: str = PROCESSED_ROOT,
    out_uri: str = "data/labels/processed/summary.parquet",
) -> pd.DataFrame:
    """Compute ``summary()`` and persist as a flat Parquet for auditability."""
    df = summary(root)
    write_parquet_df(df, out_uri)
    logger.success("wrote summary → {} ({} rows)", out_uri, len(df))
    return df


__all__ = [
    "COLUMNS",
    "KNOWN_DATASETS",
    "PROCESSED_ROOT",
    "REQUIRED_COLUMNS",
    "make_run_id",
    "partition_uri",
    "read_all",
    "summary",
    "to_obs_gdf",
    "write_partition",
    "write_summary",
]
