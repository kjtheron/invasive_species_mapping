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

import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely import to_wkb

from cmrv.io import list_parquet_files, read_parquet_df, write_parquet_df

PROCESSED_ROOT = "data/labels/processed"
COORD_UNCERTAINTY_DROP_M = 500.0
# One partition directory per source dataset (mirrors data/labels/raw/<dataset>/).
KNOWN_DATASETS = frozenset({"BioSCape_VegPlots_Berg_Eerste_2425", "mapwaps_olifants_doring"})

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


def gdf_to_obs_df(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    """Serialize a GeoDataFrame to a flat pandas DataFrame for the observation store.

    Geometry is serialised to WKB bytes (EPSG:4326). Missing nullable columns are
    filled with None; missing required columns raise ValueError.
    """
    if gdf.crs is None:
        raise ValueError("GeoDataFrame must declare a CRS (expect EPSG:4326)")
    if gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs("EPSG:4326")

    missing = [c for c in REQUIRED_COLUMNS if c not in gdf.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    df = gdf.drop(columns=[gdf.geometry.name]).copy()
    df["geometry"] = gdf.geometry.apply(lambda g: bytes(to_wkb(g, hex=False)))

    for col in COLUMNS:
        if col not in df.columns:
            df[col] = None

    return df[list(COLUMNS)]


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

    df = gdf_to_obs_df(gdf)
    existing_files = list_parquet_files(partition_dir)
    frames = [read_parquet_df(f) for f in existing_files]
    n_existing = sum(len(f) for f in frames)

    combined = pd.concat([*frames, df], ignore_index=True)
    merged = combined.sort_values("ingested_at").drop_duplicates("obs_id", keep="last")

    tmp_file = f"{partition_dir}/_tmp_{run_id}/part-{run_id}.parquet"
    final_file = f"{partition_dir}/part-{run_id}.parquet"
    write_parquet_df(merged, tmp_file)

    logger.info(
        "dataset={} new={} existing={} merged={} (dedupe kept {})",
        dataset,
        len(df),
        n_existing,
        n_existing + len(df),
        len(merged),
    )

    # Atomic promote: remove prior partition files then move the new one in.
    for old_file in existing_files:
        os.remove(old_file)
    os.replace(tmp_file, final_file)
    os.rmdir(f"{partition_dir}/_tmp_{run_id}")

    logger.success("wrote dataset={} → {} ({} rows after upsert)", dataset, final_file, len(merged))
    return final_file


def read_all(root: str = PROCESSED_ROOT) -> pd.DataFrame:
    """Read the full partitioned dataset as a pandas DataFrame.

    The ``geometry`` column contains raw WKB bytes; callers that need shapely
    geometries should apply ``shapely.from_wkb`` (as ``load_training_labels`` does).
    """
    files = list_parquet_files(root, recursive=True)
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}")
    return pd.concat([read_parquet_df(f) for f in files], ignore_index=True)


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
    "COORD_UNCERTAINTY_DROP_M",
    "KNOWN_DATASETS",
    "PROCESSED_ROOT",
    "REQUIRED_COLUMNS",
    "gdf_to_obs_df",
    "make_run_id",
    "partition_uri",
    "read_all",
    "summary",
    "write_partition",
    "write_summary",
]
