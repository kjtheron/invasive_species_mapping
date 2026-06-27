"""Tests for the SANLC/VegMap sampler crosswalks (no raster I/O needed)."""

from __future__ import annotations

from cmrv.labels.classmap import build_lookup
from cmrv.labels.sanlc import ACC_CLASS_TO_CLASS, BIOME_TO_CLASS


def test_emitted_classes_resolve_in_landcover_map() -> None:
    """Every class the adapter emits must exist in western_cape_landcover."""
    cm = build_lookup("configs/labels_schema.yaml", "western_cape_landcover")
    emitted = {c for c in ACC_CLASS_TO_CLASS.values() if c != "NATURAL"} | set(
        BIOME_TO_CLASS.values()
    )
    unresolved = {c for c in emitted if cm.resolve(c) is None}
    assert not unresolved, f"adapter emits classes missing from the class map: {unresolved}"


def test_natural_defers_to_vegmap_and_plantation_separate() -> None:
    """Natural shrubland → VegMap (sentinel 'NATURAL'); plantation its own class."""
    assert ACC_CLASS_TO_CLASS["low shrubland (fynbos)"] == "NATURAL"
    assert ACC_CLASS_TO_CLASS["plantation"] == "planted_forest"  # alien plantation ≠ invasive IAP
