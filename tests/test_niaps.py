"""NIAPS distillation: growth-form exclusions and cross-taxon overlap removal."""

from __future__ import annotations

import geopandas as gpd
from shapely.geometry import box

from cmrv.aoi import SA_ALBERS
from cmrv.labels.classmap import build_lookup
from cmrv.labels.niaps import DROPPED_GROWTH_FORM, LAYER_TO_TAXON, _drop_cross_taxon_overlaps


def test_non_woody_taxa_are_not_ingestable() -> None:
    """Arundo (tall grass) and Chromolaena (perennial herb) must have no route in."""
    assert set(DROPPED_GROWTH_FORM) == {"Arundo donax", "Chromolaena odorata"}
    assert not set(DROPPED_GROWTH_FORM) & set(LAYER_TO_TAXON)


def test_every_taxon_resolves_under_sa_landcover() -> None:
    """A layer that ingests but has no class chips at full imagery cost, then vanishes."""
    cm = build_lookup("configs/labels_schema.yaml", "sa_landcover")
    for layer, (sp, _rank) in LAYER_TO_TAXON.items():
        assert cm.resolve(sp) is not None, f"{layer} → {sp!r} resolves to no class"


def test_overlapping_taxa_both_dropped() -> None:
    """A mixed centre has no single correct label, so neither side survives."""
    g = gpd.GeoDataFrame(
        {"layer": ["Pinus_spp", "Wattle_spp", "Pinus_spp"]},
        geometry=[box(0, 0, 10, 10), box(5, 5, 15, 15), box(100, 100, 110, 110)],
        crs=SA_ALBERS,
    )
    out = _drop_cross_taxon_overlaps(g)
    assert len(out) == 1  # only the isolated Pinus core
    assert out.iloc[0].geometry.bounds == (100.0, 100.0, 110.0, 110.0)


def test_same_taxon_overlap_is_kept() -> None:
    """Self-overlap within one taxon is not ambiguity — don't throw it away."""
    g = gpd.GeoDataFrame(
        {"layer": ["Pinus_spp", "Pinus_spp"]},
        geometry=[box(0, 0, 10, 10), box(5, 5, 15, 15)],
        crs=SA_ALBERS,
    )
    assert len(_drop_cross_taxon_overlaps(g)) == 2
