"""Tests for the MapWAPS adapter's pure transforms."""

from __future__ import annotations

import datetime as dt

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from cmrv.labels.mapwaps import _build_rows, _clean_date, _density_to_cover, _lulc_to_taxon


def test_lulc_alien_classes_map_to_genus() -> None:
    assert _lulc_to_taxon("Alien_Pine") == ("Pinus", "genus")
    assert _lulc_to_taxon("Alien_Gum") == ("Eucalyptus", "genus")
    assert _lulc_to_taxon("Alien_Wattle") == ("Acacia", "genus")
    assert _lulc_to_taxon("Alien_Prosopis") == ("Prosopis", "genus")


def test_lulc_native_kept_verbatim_as_functional() -> None:
    assert _lulc_to_taxon("Fynbos-High density") == ("Fynbos-High density", "functional")
    assert _lulc_to_taxon("Alien_Other") == ("Alien_Other", "functional")


def test_density_zero_is_null_positive_is_cover() -> None:
    assert _density_to_cover(0) is None  # 0 ambiguous → NULL
    assert _density_to_cover(100) == 100.0
    assert _density_to_cover(float("nan")) is None


def test_clean_date_drops_excel_sentinel() -> None:
    assert _clean_date("2025-05-19") == "2025-05-19"
    assert _clean_date(pd.Timestamp("1899-12-30")) is None
    assert _clean_date(None) is None


def test_build_rows_resolves_under_genus_map() -> None:
    """A genus-labelled MapWAPS row resolves to an IAP class under the genus map."""
    from cmrv.labels.classmap import build_lookup

    gdf = gpd.GeoDataFrame(
        pd.DataFrame(
            [
                {
                    "LULC_Class": "Alien_Pine",
                    "X": 230000.0,
                    "Y": 6240000.0,
                    "Density___": 100,
                    "DateTime": pd.Timestamp("2025-05-19"),
                    "Id": 1,
                },
                {
                    "LULC_Class": "Water",
                    "X": 231000.0,
                    "Y": 6241000.0,
                    "Density___": 0,
                    "DateTime": pd.Timestamp("1899-12-30"),
                    "Id": 2,
                },
            ]
        ),
        geometry=[Point(19.0, -32.0), Point(19.1, -32.1)],
        crs="EPSG:4326",
    )
    rows = _build_rows(gdf, "run1", dt.datetime(2026, 6, 24, tzinfo=dt.UTC))
    cm = build_lookup("configs/labels_schema.yaml", "western_cape_iap_genus")

    pine, water = rows
    assert pine["species_normalized"] == "Pinus"
    assert cm.resolve(pine["species_normalized"]) is not None  # IAP → kept
    assert pine["cover_pct"] == 100.0
    assert pine["event_date"] == "2025-05-19"
    assert water["taxon_rank"] == "functional"
    assert water["cover_pct"] is None  # density 0 → NULL
    assert water["event_date"] is None  # sentinel dropped
    assert cm.resolve(water["species_normalized"]) is None  # native → dropped at split


def test_obs_id_from_geometry_not_parent_xy() -> None:
    """Child points share the parent's X/Y but have distinct geometry → distinct obs_id."""
    gdf = gpd.GeoDataFrame(
        pd.DataFrame(
            [
                {"LULC_Class": "Alien_Pine", "X": 230000.0, "Y": 6240000.0, "Density___": 100},
                {"LULC_Class": "Alien_Pine", "X": 230000.0, "Y": 6240000.0, "Density___": 90},
            ]
        ),
        geometry=[Point(19.0, -32.0), Point(19.001, -32.001)],
        crs="EPSG:4326",
    )
    rows = _build_rows(gdf, "run1", dt.datetime(2026, 6, 24, tzinfo=dt.UTC))
    assert rows[0]["obs_id"] != rows[1]["obs_id"]


def test_undated_point_uses_fallback_date() -> None:
    """Undated points get the campaign fallback date (so event_date.year is set)."""
    gdf = gpd.GeoDataFrame(
        pd.DataFrame(
            [
                {
                    "LULC_Class": "Alien_Pine",
                    "X": 1.0,
                    "Y": 2.0,
                    "Density___": 100,
                    "DateTime": pd.Timestamp("1899-12-30"),
                }
            ]
        ),
        geometry=[Point(19.0, -32.0)],
        crs="EPSG:4326",
    )
    rows = _build_rows(
        gdf, "r", dt.datetime(2026, 6, 24, tzinfo=dt.UTC), fallback_date="2025-05-19"
    )
    assert rows[0]["event_date"] == "2025-05-19"
