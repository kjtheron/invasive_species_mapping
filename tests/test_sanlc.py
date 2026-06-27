"""Tests for the SANLC/VegMap sampler crosswalks (no raster I/O needed)."""

from __future__ import annotations

from cmrv.labels.classmap import build_lookup
from cmrv.labels.sanlc import BIOME_TO_CLASS, SALCC2_TO_CLASS


def test_emitted_classes_resolve_in_landcover_map() -> None:
    """Every class the sampler emits must exist in western_cape_landcover."""
    cm = build_lookup("configs/labels_schema.yaml", "western_cape_landcover")
    emitted = {c for c in SALCC2_TO_CLASS.values() if c != "NATURAL"} | set(BIOME_TO_CLASS.values())
    unresolved = {c for c in emitted if cm.resolve(c) is None}
    assert not unresolved, f"sampler emits classes missing from the class map: {unresolved}"


def test_natural_salcc2_groups_defer_to_vegmap() -> None:
    """The natural SANLC groups must defer to VegMap (sentinel 'NATURAL')."""
    assert SALCC2_TO_CLASS["Karoo & Fynbos Shubland"] == "NATURAL"
    assert SALCC2_TO_CLASS["Planted Forest"] == "planted_forest"  # alien plantation kept separate
