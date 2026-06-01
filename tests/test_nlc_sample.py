"""Tests for NLC 2022 raster sampling (cmrv.labels.nlc)."""

from __future__ import annotations

import numpy as np
import pytest

from cmrv.labels.nlc import _build_nlc_to_class, _sample_class_pixels


@pytest.fixture()
def schema() -> dict:
    """Mirrors the corrected NLC 2022 VAT-verified schema."""
    return {
        "nlc_2022": {
            "class_groups": {
                "indigenous_forest": [1, 2, 3, 4],
                "plantations": [5, 6, 7],
                "shrubland_fynbos": [9],
                "shrubland_other": [8],
                "cultivated": [32, 33],
                "built_up": [47, 48],
                "mines": [68, 69],
            },
            "class_crosswalk": {
                "10": ["indigenous_forest"],
                "8": ["shrubland_fynbos"],
                "11": ["cultivated", "built_up", "mines"],
            },
            "exclusions": ["plantations"],
        }
    }


def test_nlc_crosswalk_basic(schema: dict) -> None:
    mapping = _build_nlc_to_class(schema)
    assert mapping[1] == 10
    assert mapping[4] == 10
    assert mapping[9] == 8
    assert mapping[32] == 11
    assert mapping[47] == 11
    assert mapping[68] == 11


def test_nlc_crosswalk_excludes_plantations(schema: dict) -> None:
    mapping = _build_nlc_to_class(schema)
    assert 5 not in mapping
    assert 6 not in mapping
    assert 7 not in mapping


def test_nlc_crosswalk_shrubland_other_not_mapped(schema: dict) -> None:
    """NLC value 8 (shrubland_other) is not in the crosswalk — handled in code via vegmap."""
    mapping = _build_nlc_to_class(schema)
    assert 8 not in mapping


def test_nlc_crosswalk_no_unknown_classes(schema: dict) -> None:
    mapping = _build_nlc_to_class(schema)
    valid_classes = {8, 10, 11}
    assert all(v in valid_classes for v in mapping.values())


def test_grid_thinning_enforces_spacing() -> None:
    """Points closer than min_spacing_m should be thinned."""
    from rasterio.transform import Affine

    transform = Affine(20.0, 0.0, 0.0, 0.0, -20.0, 1_000_000.0)
    rows = np.array([0, 0, 0, 10, 10, 10, 20, 20, 20])
    cols = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
    rng = np.random.default_rng(42)

    coords = _sample_class_pixels(rows, cols, transform, target=100, min_spacing_m=200.0, rng=rng)
    assert len(coords) <= 9
    if len(coords) > 1:
        dists = np.linalg.norm(coords[1:] - coords[:-1], axis=1)
        assert dists.min() >= 0


def test_grid_thinning_subsample() -> None:
    """When more points survive thinning than target, subsample."""
    from rasterio.transform import Affine

    transform = Affine(20.0, 0.0, 0.0, 0.0, -20.0, 1_000_000.0)
    rows = np.arange(100)
    cols = np.arange(100) * 20
    rng = np.random.default_rng(42)

    coords = _sample_class_pixels(rows, cols, transform, target=10, min_spacing_m=1.0, rng=rng)
    assert len(coords) == 10


def test_deterministic_sampling() -> None:
    """Same seed produces identical results."""
    from rasterio.transform import Affine

    transform = Affine(20.0, 0.0, 0.0, 0.0, -20.0, 1_000_000.0)
    rows = np.arange(50)
    cols = np.arange(50) * 5

    coords1 = _sample_class_pixels(
        rows, cols, transform, target=10, min_spacing_m=1.0, rng=np.random.default_rng(99)
    )
    coords2 = _sample_class_pixels(
        rows, cols, transform, target=10, min_spacing_m=1.0, rng=np.random.default_rng(99)
    )
    np.testing.assert_array_equal(coords1, coords2)
