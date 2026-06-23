"""Check load_training_labels filtering + AOI clip after the pandas rewrite."""

from __future__ import annotations

import datetime as dt

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point, box

from cmrv.io import write_gdf_parquet
from cmrv.labels.merge import load_training_labels
from cmrv.labels.observations import write_source_partition


def _obs_gdf(rows: list[dict]) -> gpd.GeoDataFrame:
    base = {
        "source": "bioscape_plot",
        "source_record_id": "0",
        "geom_type": "point",
        "coord_uncertainty_m": None,
        "event_date": None,
        "weight": 0.5,
        "ingested_at": dt.datetime(2026, 4, 17, tzinfo=dt.UTC),
        "ingest_run_id": "t",
        "aoi_admin1": "western_cape",
    }
    recs = [{**base, **r} for r in rows]
    return gpd.GeoDataFrame(
        pd.DataFrame(recs), geometry=[Point(18.5, -33.9)] * len(recs), crs="EPSG:4326"
    )


def test_load_training_labels_species_and_aoi(tmp_path):
    root = f"{tmp_path}/obs"
    gdf = _obs_gdf(
        [
            {"obs_id": "gbif:1", "species_normalized": "acacia mearnsii"},
            {"obs_id": "gbif:2", "species_normalized": "pinus radiata"},
        ]
    )
    write_source_partition(gdf, "bioscape_plot", root=root, run_id="r1")

    aoi = f"{tmp_path}/aoi.parquet"
    write_gdf_parquet(
        gpd.GeoDataFrame({"id": [0]}, geometry=[box(18, -34, 19, -33)], crs="EPSG:4326"), aoi
    )

    out = load_training_labels(aoi, species_subset=["mearnsii"], root=root)
    assert list(out["obs_id"]) == ["gbif:1"]  # species filter + AOI clip both applied
    assert out.crs.to_epsg() == 4326
