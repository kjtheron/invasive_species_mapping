"""Unified observation schema for the WC NEMBA occurrence store.

Canonical store: ``gs://ism-data/labels/wc/obs/source=<source>/`` — partitioned
Parquet, deduped on ``obs_id``. See roadmap §1.5 for why every row carries
``obs_id``, ``source``, ``coord_uncertainty_m``, and ``ingested_at``.

``class_id`` is deliberately **not** in this schema — training configs crosswalk
``(gbif_usage_key, nemba_category) → class_id`` via
``configs/labels_schema.yaml → class_maps.<name>``. Same parquet feeds
upper-Berg-12 and WC-NEMBA-full runs.
"""

from __future__ import annotations

import datetime as dt

import duckdb
import fsspec
import geopandas as gpd
import pandas as pd
from loguru import logger
from shapely import to_wkb

from cmrv.io import _duckdb_con, ensure_parent, list_parquet_files, write_parquet_df

WC_LABELS_ROOT = "gs://ism-data/labels/wc/obs"
COORD_UNCERTAINTY_DROP_M = 500.0
KNOWN_SOURCES = frozenset(
    {
        "gbif",
        "inat_via_gbif",
        "bioscape_line",
        "bioscape_plot",
        "vegmap",
        "nlc_sample",
    }
)

# Canonical column order for all partitions
COLUMNS: tuple[str, ...] = (
    "obs_id",
    "source",
    "source_record_id",
    "source_url",
    "species",
    "species_normalized",
    "gbif_usage_key",
    "nemba_category",
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


def partition_uri(source: str, root: str = WC_LABELS_ROOT) -> str:
    """URI of the source-specific hive partition directory."""
    if source not in KNOWN_SOURCES:
        raise ValueError(f"unknown source {source!r}; expected one of {sorted(KNOWN_SOURCES)}")
    return f"{root}/source={source}"


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


def write_source_partition(
    gdf: gpd.GeoDataFrame,
    source: str,
    root: str = WC_LABELS_ROOT,
    run_id: str | None = None,
) -> str:
    """Atomic partition overwrite with upsert semantics.

    Reads any existing partition via DuckDB, concatenates the new rows,
    deduplicates on ``obs_id`` (keeping max ``ingested_at``), and writes
    the result back. Re-running the same ingest is idempotent.

    Writes to ``_tmp_<run_id>/`` first, then ``fs.mv`` into place so
    readers never see a torn partition.
    """
    if source not in KNOWN_SOURCES:
        raise ValueError(f"unknown source {source!r}")
    partition_dir = partition_uri(source, root=root)
    run_id = run_id or make_run_id(source)

    df = gdf_to_obs_df(gdf)
    existing_files = list_parquet_files(partition_dir)

    con = _duckdb_con(partition_dir)
    con.register("_new", df)

    if existing_files:
        files_sql = "[" + ", ".join(f"'{f}'" for f in existing_files) + "]"
        n_existing = con.execute(f"SELECT COUNT(*) FROM read_parquet({files_sql})").fetchone()[0]
        logger.info(
            "reading {} existing row(s) from {} file(s) for source={}",
            n_existing,
            len(existing_files),
            source,
        )
        combined_sql = f"SELECT * FROM read_parquet({files_sql}) UNION ALL SELECT * FROM _new"
    else:
        n_existing = 0
        combined_sql = "SELECT * FROM _new"

    tmp_dir = f"{partition_dir}/_tmp_{run_id}"
    final_file = f"{partition_dir}/part-{run_id}.parquet"
    tmp_file = f"{tmp_dir}/part-{run_id}.parquet"

    fs, _ = fsspec.core.url_to_fs(partition_dir)
    _, tmp_path = fsspec.core.url_to_fs(tmp_dir)
    if fs.exists(tmp_path):
        fs.rm(tmp_path, recursive=True)

    ensure_parent(tmp_file)  # creates tmp_dir on local filesystem (no-op for gs://)

    con.execute(f"""
        COPY (
            WITH combined AS ({combined_sql})
            SELECT * FROM combined
            QUALIFY ROW_NUMBER() OVER (PARTITION BY obs_id ORDER BY ingested_at DESC) = 1
        ) TO '{tmp_file}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)

    n_out = con.execute(f"SELECT COUNT(*) FROM read_parquet('{tmp_file}')").fetchone()[0]
    logger.info(
        "source={} new={} existing={} merged={} (dedupe kept {})",
        source,
        len(df),
        n_existing,
        n_existing + len(df),
        n_out,
    )

    # Atomic promote: remove prior partition files then move the new one in.
    for old_file in existing_files:
        fs.rm(old_file)
    _, final_path = fsspec.core.url_to_fs(final_file)
    _, tmp_file_path = fsspec.core.url_to_fs(tmp_file)
    fs.mv(tmp_file_path, final_path)
    fs.rm(tmp_path, recursive=True)

    logger.success(
        "wrote source={} → {} ({} rows after upsert)",
        source,
        final_file,
        n_out,
    )
    return final_file


def read_all(root: str = WC_LABELS_ROOT) -> pd.DataFrame:
    """Read the full partitioned dataset as a pandas DataFrame.

    The ``geometry`` column contains raw WKB bytes; callers that need shapely
    geometries should apply ``shapely.from_wkb`` (as ``load_training_labels`` does).
    """
    files = list_parquet_files(root, recursive=True)
    if not files:
        raise FileNotFoundError(f"no parquet files under {root}")
    con = _duckdb_con(root)
    files_sql = "[" + ", ".join(f"'{f}'" for f in files) + "]"
    return con.execute(f"SELECT * FROM read_parquet({files_sql})").df()


def summary(root: str = WC_LABELS_ROOT) -> pd.DataFrame:
    """Per-source × per-category counts + fraction with non-null coord_uncertainty_m."""
    tbl = read_all(root)
    con = duckdb.connect()
    con.register("_tbl", tbl)
    return con.execute("""
        SELECT
            source,
            nemba_category,
            COUNT(*) AS n,
            SUM(CASE WHEN coord_uncertainty_m IS NOT NULL THEN 1 ELSE 0 END)
                AS n_with_coord_unc,
            ROUND(
                AVG(CASE WHEN coord_uncertainty_m IS NOT NULL THEN 1.0 ELSE 0.0 END) * 100,
                2
            ) AS pct_with_coord_unc
        FROM _tbl
        GROUP BY source, nemba_category
        ORDER BY source, nemba_category
    """).df()


def write_summary(
    root: str = WC_LABELS_ROOT,
    out_uri: str = "gs://ism-data/labels/wc/summary.parquet",
) -> pd.DataFrame:
    """Compute ``summary()`` and persist as a flat Parquet for auditability."""
    df = summary(root)
    write_parquet_df(df, out_uri)
    logger.success("wrote summary → {} ({} rows)", out_uri, len(df))
    return df


__all__ = [
    "COLUMNS",
    "COORD_UNCERTAINTY_DROP_M",
    "KNOWN_SOURCES",
    "REQUIRED_COLUMNS",
    "WC_LABELS_ROOT",
    "gdf_to_obs_df",
    "make_run_id",
    "partition_uri",
    "read_all",
    "summary",
    "write_source_partition",
    "write_summary",
]
