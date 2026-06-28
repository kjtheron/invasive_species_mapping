"""Tests for the unified observation schema + upsert semantics."""

from __future__ import annotations

import datetime as dt
import tempfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pytest
from shapely.geometry import Point

from cmrv.labels.observations import (
    COLUMNS,
    gdf_to_obs_df,
    write_partition,
)


def _make_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    """Minimal GeoDataFrame with all required schema columns."""
    base = {
        "obs_id": None,
        "source": "bioscape_plot",
        "source_record_id": "0",
        "source_url": None,
        "species": None,
        "species_normalized": None,
        "gbif_usage_key": None,
        "geom_type": "point",
        "coord_uncertainty_m": None,
        "event_date": None,
        "basis_of_record": None,
        "cover_pct": None,
        "weight": 0.5,
        "ingested_at": dt.datetime(2026, 4, 17, 12, 0, 0, tzinfo=dt.UTC),
        "ingest_run_id": "test_run",
        "aoi_admin1": "western_cape",
    }
    records = [{**base, **r} for r in rows]
    pdf = pd.DataFrame(records)
    geometries = [Point(18.5, -33.9)] * len(pdf)
    return gpd.GeoDataFrame(pdf, geometry=geometries, crs="EPSG:4326")


def test_gdf_to_obs_df_columns() -> None:
    """gdf_to_obs_df produces a DataFrame with the canonical column set."""
    gdf = _make_gdf([{"obs_id": "gbif:1", "source_record_id": "1", "weight": 0.5}])
    df = gdf_to_obs_df(gdf)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == list(COLUMNS)
    assert len(df) == 1
    assert isinstance(df["geometry"].iloc[0], bytes)


def test_upsert_same_obs_id_keeps_latest() -> None:
    """Re-ingesting the same obs_id keeps only the row with max(ingested_at)."""
    t_old = dt.datetime(2026, 4, 1, tzinfo=dt.UTC)
    t_new = dt.datetime(2026, 4, 17, tzinfo=dt.UTC)

    gdf_first = _make_gdf([{"obs_id": "gbif:42", "source_record_id": "42", "ingested_at": t_old}])
    gdf_second = _make_gdf([{"obs_id": "gbif:42", "source_record_id": "42", "ingested_at": t_new}])

    with tempfile.TemporaryDirectory() as tmp:
        root = f"{tmp}/obs"

        write_partition(gdf_first, "BioSCape_VegPlots_Berg_Eerste_2425", root=root, run_id="run1")
        write_partition(gdf_second, "BioSCape_VegPlots_Berg_Eerste_2425", root=root, run_id="run2")

        files = list(Path(f"{root}/BioSCape_VegPlots_Berg_Eerste_2425").glob("*.parquet"))
        assert len(files) == 1, "should be exactly one partition file after upsert"

        df = pd.read_parquet(str(files[0]))
        assert len(df) == 1, "same obs_id should deduplicate to 1 row"
        assert pd.Timestamp(df["ingested_at"].iloc[0]) == pd.Timestamp(t_new), (
            "should keep the row with the later ingested_at"
        )


def test_upsert_distinct_obs_ids_both_retained() -> None:
    """Two distinct obs_ids both survive the upsert."""
    gdf = _make_gdf(
        [
            {"obs_id": "gbif:1", "source_record_id": "1"},
            {"obs_id": "gbif:2", "source_record_id": "2"},
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = f"{tmp}/obs"
        write_partition(gdf, "BioSCape_VegPlots_Berg_Eerste_2425", root=root, run_id="run1")

        files = list(Path(f"{root}/BioSCape_VegPlots_Berg_Eerste_2425").glob("*.parquet"))
        df = pd.read_parquet(str(files[0]))
        assert len(df) == 2


def test_read_all_ignores_root_level_parquet() -> None:
    """A summary.parquet sitting in the root must not pollute the observation rows."""
    from cmrv.labels.observations import read_all

    gdf = _make_gdf([{"obs_id": "x:1", "source_record_id": "1"}])
    with tempfile.TemporaryDirectory() as tmp:
        root = f"{tmp}/obs"
        write_partition(gdf, "BioSCape_VegPlots_Berg_Eerste_2425", root=root, run_id="run1")
        # drop a stray non-partition parquet directly under root (like summary.parquet)
        pd.DataFrame({"source": ["bioscape_plot"], "n": [1]}).to_parquet(f"{root}/summary.parquet")

        df = read_all(root)
        assert len(df) == 1 and set(df["obs_id"]) == {"x:1"}


def test_upsert_idempotent() -> None:
    """Running the same ingest twice produces the same row count."""
    gdf = _make_gdf(
        [
            {"obs_id": "gbif:10", "source_record_id": "10"},
            {"obs_id": "gbif:11", "source_record_id": "11"},
        ]
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = f"{tmp}/obs"
        write_partition(gdf, "BioSCape_VegPlots_Berg_Eerste_2425", root=root, run_id="run1")
        write_partition(gdf, "BioSCape_VegPlots_Berg_Eerste_2425", root=root, run_id="run2")

        files = list(Path(f"{root}/BioSCape_VegPlots_Berg_Eerste_2425").glob("*.parquet"))
        df = pd.read_parquet(str(files[0]))
        assert len(df) == 2, "re-ingest of same rows should remain at 2"


def test_missing_required_column_raises() -> None:
    """gdf_to_obs_df raises ValueError when a non-nullable column is missing."""
    gdf = gpd.GeoDataFrame(
        pd.DataFrame([{"source": "gbif", "weight": 0.5}]),
        geometry=[Point(18.5, -33.9)],
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="missing required"):
        gdf_to_obs_df(gdf)
