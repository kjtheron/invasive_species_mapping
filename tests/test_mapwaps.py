"""Tests for the MapWAPS adapter's pure transforms + crosswalk correctness."""

from __future__ import annotations

import datetime as dt

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point

from cmrv.labels.classmap import build_lookup
from cmrv.labels.mapwaps import (
    _LULC_TO_CLASS,
    CATCHMENTS,
    _build_rows,
    _clean_date,
    _density_to_cover,
    _lulc_to_taxon,
)

OLI = CATCHMENTS["mapwaps_olifants_doring"]  # cols: LULC_Class / Density___ / DateTime


def test_lulc_alien_classes_map_to_genus() -> None:
    assert _lulc_to_taxon("Alien_Pine") == ("Pinus", "genus")
    assert _lulc_to_taxon("Alien_Gum") == ("Eucalyptus", "genus")
    assert _lulc_to_taxon("Alien_Wattle") == ("Acacia", "genus")
    assert _lulc_to_taxon("Alien_Black Wattle") == ("Acacia", "genus")  # uMzimvubu split
    assert _lulc_to_taxon("Alien_Silver Wattle") == ("Acacia", "genus")
    assert _lulc_to_taxon("Alien_Prosopis") == ("Prosopis", "genus")
    assert _lulc_to_taxon("Alien_Poplar") == ("Populus", "genus")


def test_lulc_native_and_transformed_map_to_members() -> None:
    assert _lulc_to_taxon("Fynbos-High density") == ("fynbos", "biome")
    assert _lulc_to_taxon("Renosterveld") == ("renosterveld", "biome")  # own class, not fynbos
    assert _lulc_to_taxon("Bushmanland Shrubland") == ("nama_karoo", "biome")
    assert _lulc_to_taxon("Grassland") == ("grassland", "biome")
    assert _lulc_to_taxon("Indigenous Forest") == ("forest", "biome")
    assert _lulc_to_taxon("Indigenous Bush_Vachellia") == ("savanna", "biome")
    assert _lulc_to_taxon("Indigenous Bush_Leucosidea") == ("savanna", "biome")
    assert _lulc_to_taxon("Irrigated Agriculture") == ("cultivated", "landcover")
    assert _lulc_to_taxon("Maize") == ("cultivated", "landcover")
    assert _lulc_to_taxon("Rock") == ("bare", "landcover")


def test_lulc_unmapped_classes_drop() -> None:
    # shadow / transient / indigenous-fern / unspecific → no member, dropped at ingest
    for cls in ("Shade", "Burnt", "Bracken", "Alien_Other"):
        assert _lulc_to_taxon(cls) == (None, "functional")


def test_every_mapped_class_resolves_under_landcover() -> None:
    """The requirement: every _LULC_TO_CLASS target is a real western_cape_landcover member."""
    cm = build_lookup("configs/labels_schema.yaml", "western_cape_landcover")
    for lulc, (member, _rank) in _LULC_TO_CLASS.items():
        assert cm.resolve(member) is not None, f"{lulc} → {member!r} resolves to no class"


def test_poplar_resolves_to_populus_spp() -> None:
    """Alien_Poplar → 'Populus' → genus fallback → its own class (not iap_riparian)."""
    cm = build_lookup("configs/labels_schema.yaml", "western_cape_landcover")
    member, _ = _lulc_to_taxon("Alien_Poplar")
    assert cm.resolve(member) is not None


def test_density_zero_is_null_positive_is_cover() -> None:
    assert _density_to_cover(0) is None  # 0 ambiguous → NULL
    assert _density_to_cover(100) == 100.0
    assert _density_to_cover(float("nan")) is None


def test_clean_date_drops_excel_sentinel() -> None:
    assert _clean_date("2025-05-19") == "2025-05-19"
    assert _clean_date(pd.Timestamp("1899-12-30")) is None
    assert _clean_date(None) is None


def _oli_gdf(records: list[dict], geoms: list[Point]) -> gpd.GeoDataFrame:
    return gpd.GeoDataFrame(pd.DataFrame(records), geometry=geoms, crs="EPSG:4326")


def test_build_rows_resolves_under_landcover() -> None:
    """An alien-genus row and a native row both resolve under western_cape_landcover."""
    gdf = _oli_gdf(
        [
            {"LULC_Class": "Alien_Pine", "Density___": 100, "DateTime": pd.Timestamp("2025-05-19")},
            {"LULC_Class": "Water", "Density___": 0, "DateTime": pd.Timestamp("1899-12-30")},
        ],
        [Point(19.0, -32.0), Point(19.1, -32.1)],
    )
    rows = _build_rows(
        gdf, OLI, "run1", dt.datetime(2026, 6, 24, tzinfo=dt.UTC), fallback_date=None
    )
    cm = build_lookup("configs/labels_schema.yaml", "western_cape_landcover")

    pine, water = rows
    assert pine["species_normalized"] == "Pinus"
    assert cm.resolve(pine["species_normalized"]) is not None  # IAP genus → kept
    assert pine["cover_pct"] == 100.0
    assert pine["event_date"] == "2025-05-19"
    assert water["species_normalized"] == "water"
    assert water["taxon_rank"] == "landcover"
    assert cm.resolve(water["species_normalized"]) is not None  # water → class 17
    assert water["cover_pct"] is None  # density 0 → NULL
    assert water["event_date"] is None  # sentinel dropped, no fallback


def test_obs_id_from_geometry_not_parent_xy() -> None:
    """Child points share parent X/Y but have distinct geometry → distinct obs_id."""
    gdf = _oli_gdf(
        [
            {"LULC_Class": "Alien_Pine", "Density___": 100},
            {"LULC_Class": "Alien_Pine", "Density___": 90},
        ],
        [Point(19.0, -32.0), Point(19.001, -32.001)],
    )
    rows = _build_rows(
        gdf, OLI, "run1", dt.datetime(2026, 6, 24, tzinfo=dt.UTC), fallback_date=None
    )
    assert rows[0]["obs_id"] != rows[1]["obs_id"]


def test_undated_point_uses_fallback_date() -> None:
    """Undated points get the campaign fallback date (so event_date.year is set)."""
    gdf = _oli_gdf(
        [{"LULC_Class": "Alien_Pine", "Density___": 100, "DateTime": pd.Timestamp("1899-12-30")}],
        [Point(19.0, -32.0)],
    )
    rows = _build_rows(
        gdf, OLI, "r", dt.datetime(2026, 6, 24, tzinfo=dt.UTC), fallback_date="2025-05-19"
    )
    assert rows[0]["event_date"] == "2025-05-19"
