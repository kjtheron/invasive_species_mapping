"""Tests for the SANLC/VegMap sampler crosswalks (no raster I/O needed)."""

from __future__ import annotations

from cmrv.labels.classmap import build_lookup
from cmrv.labels.sanlc import ACC_CLASS_TO_CLASS, BIOME_TO_CLASS


def test_emitted_classes_resolve_in_landcover_map() -> None:
    """Every class the adapter emits must exist in sa_landcover."""
    cm = build_lookup("configs/labels_schema.yaml", "sa_landcover")
    emitted = {c for c in ACC_CLASS_TO_CLASS.values() if c != "NATURAL"} | set(
        BIOME_TO_CLASS.values()
    )
    # Emitted but deliberately not trained: the store keeps what SANLC surveys and
    # make-split drops these. planted_forest left 2026-09-13 (gum OR pine, so it
    # overlapped eucalyptus_spp and pinus_spp).
    not_trained = {"planted_forest"}
    unresolved = {c for c in emitted if cm.resolve(c) is None}
    assert unresolved == not_trained, (
        f"adapter emits classes missing from the class map: {unresolved - not_trained}; "
        f"listed as not trained but resolves: {not_trained - unresolved}"
    )


def test_natural_defers_to_vegmap_and_plantation_separate() -> None:
    """Natural shrubland → VegMap (sentinel 'NATURAL'); plantation its own class."""
    assert ACC_CLASS_TO_CLASS["low shrubland (fynbos)"] == "NATURAL"
    assert ACC_CLASS_TO_CLASS["plantation"] == "planted_forest"  # alien plantation ≠ invasive IAP
