"""Tests for cmrv.labels.classmap — single-source-of-truth crosswalk."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cmrv.labels.classmap import build_lookup


def _write_schema(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "schema.yaml"
    p.write_text(yaml.safe_dump(body, sort_keys=False))
    return p


def test_members_exact_match(tmp_path):
    schema = _write_schema(
        tmp_path,
        {
            "class_maps": {
                "test12": {
                    0: {"name": "mearnsii", "members": ["Acacia mearnsii"]},
                    1: {"name": "saligna", "members": ["Acacia saligna", "Acacia cyclops"]},
                }
            }
        },
    )
    cm = build_lookup(schema, "test12")
    assert cm.resolve("Acacia mearnsii") == 0
    assert cm.resolve("acacia cyclops") == 1
    assert cm.resolve("Acacia saligna") == 1


def test_genus_fallback_only_when_enabled(tmp_path):
    schema = _write_schema(
        tmp_path,
        {
            "class_maps": {
                "test12": {
                    4: {
                        "name": "pinus_spp",
                        "members": ["Pinus pinaster", "Pinus radiata"],
                        "genus_fallback": True,
                        "genus": "pinus",
                    },
                    6: {
                        "name": "hakea_sericea",
                        "members": ["Hakea sericea"],
                    },
                }
            }
        },
    )
    cm = build_lookup(schema, "test12")
    assert cm.resolve("Pinus halepensis") == 4  # genus fallback
    assert cm.resolve("Hakea drupacea") is None  # hakea has no genus fallback
    assert cm.resolve("Pinus pinaster") == 4  # exact


def test_unmapped_returns_none(tmp_path):
    schema = _write_schema(
        tmp_path,
        {"class_maps": {"test12": {0: {"name": "x", "members": ["Acacia mearnsii"]}}}},
    )
    cm = build_lookup(schema, "test12")
    assert cm.resolve("Banksia integrifolia") is None
    assert cm.resolve("") is None
    assert cm.resolve(None) is None


def test_validation_does_not_raise_on_duplicate_binomial(tmp_path):
    schema = _write_schema(
        tmp_path,
        {
            "class_maps": {
                "test12": {
                    0: {"name": "a", "members": ["Acacia saligna"]},
                    1: {"name": "b", "members": ["Acacia saligna"]},
                }
            }
        },
    )
    # Duplicate binomial across classes warns (loguru→stderr) but never raises.
    build_lookup(schema, "test12")


def test_genus_fallback_without_genus_infers_from_member(tmp_path):
    schema = _write_schema(
        tmp_path,
        {
            "class_maps": {
                "test12": {
                    4: {
                        "name": "pinus_spp",
                        "members": ["Pinus pinaster"],
                        "genus_fallback": True,
                        # missing 'genus:' field — should infer from first member
                    }
                }
            }
        },
    )
    cm = build_lookup(schema, "test12")
    assert cm.resolve("Pinus halepensis") == 4


def test_no_members_raises(tmp_path):
    schema = _write_schema(tmp_path, {"class_maps": {"test12": {0: {"name": "a"}}}})
    with pytest.raises(ValueError, match="no members"):
        build_lookup(schema, "test12")


def test_unknown_class_map_raises(tmp_path):
    schema = _write_schema(tmp_path, {"class_maps": {"foo": {0: {"members": ["X"]}}}})
    with pytest.raises(KeyError):
        build_lookup(schema, "bar")


def test_real_schema_round_trip():
    """Smoke test against the real configs/labels_schema.yaml."""
    repo_root = Path(__file__).resolve().parents[1]
    schema = repo_root / "configs" / "labels_schema.yaml"
    if not schema.exists():
        pytest.skip("real schema not present")
    cm = build_lookup(schema, "upper_berg_12")
    assert cm.resolve("Acacia mearnsii") == 0
    assert cm.resolve("Acacia cyclops") == 1
    assert cm.resolve("Pinus halepensis") == 4  # member of pinus_spp
    assert cm.resolve("Eucalyptus diversicolor") == 5  # genus fallback
    assert cm.resolve("Hakea drupacea") is None  # hakea has no genus_fallback
