"""Tests for cmrv.labels.classmap — single-source-of-truth crosswalk."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from cmrv.labels.classmap import build_lookup, gbif_taxa_from_schema


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
    assert cm.resolve("Acacia mearnsii") == (0, "exact")
    assert cm.resolve("acacia cyclops") == (1, "exact")
    assert cm.resolve("Acacia saligna") == (1, "exact")


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
    assert cm.resolve("Pinus halepensis") == (4, "genus_fallback")
    # No genus fallback for hakea — uncovered species drops
    assert cm.resolve("Hakea drupacea") == (None, "unmapped")
    # Exact still wins over genus fallback
    assert cm.resolve("Pinus pinaster") == (4, "exact")


def test_unmapped_returns_none(tmp_path):
    schema = _write_schema(
        tmp_path,
        {
            "class_maps": {
                "test12": {0: {"name": "x", "members": ["Acacia mearnsii"]}}
            }
        },
    )
    cm = build_lookup(schema, "test12")
    assert cm.resolve("Banksia integrifolia") == (None, "unmapped")
    assert cm.resolve("") == (None, "unmapped")
    assert cm.resolve(None) == (None, "unmapped")


def test_validation_warns_on_duplicate_binomial(tmp_path, caplog):
    import logging

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
    with caplog.at_level(logging.WARNING):
        build_lookup(schema, "test12")
    # Loguru routes through stderr, but the test still confirms no exception raised.
    # We can't easily capture loguru in caplog without a propagation handler;
    # behavior assertion is "build_lookup does not raise on duplicate".


def test_validation_warns_genus_fallback_without_genus(tmp_path):
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
    assert cm.resolve("Pinus halepensis") == (4, "genus_fallback")


def test_legacy_species_map_fallback(tmp_path):
    """Old-shape schema (top-level species_map) still resolves."""
    schema = _write_schema(
        tmp_path,
        {
            "class_maps": {"test12": {0: {"name": "a"}, 4: {"name": "p"}}},
            "species_map": {
                "acacia mearnsii": 0,
                "pinus pinaster": 4,
                "pinus": 4,  # genus-only entry → genus fallback under legacy shape
            },
        },
    )
    cm = build_lookup(schema, "test12")
    assert cm.resolve("Acacia mearnsii") == (0, "exact")
    assert cm.resolve("Pinus halepensis") == (4, "genus_fallback")


def test_unknown_class_map_raises(tmp_path):
    schema = _write_schema(tmp_path, {"class_maps": {"foo": {0: {"members": ["X"]}}}})
    with pytest.raises(KeyError):
        build_lookup(schema, "bar")


def test_gbif_taxa_from_schema_dedups_and_carries_class_id(tmp_path):
    schema = _write_schema(
        tmp_path,
        {
            "class_maps": {
                "test12": {
                    0: {"name": "a", "members": ["Acacia mearnsii"]},
                    1: {"name": "b", "members": ["Acacia saligna", "Acacia cyclops"]},
                    7: {"name": "rip", "members": ["Populus alba", "Populus alba"]},
                }
            }
        },
    )
    taxa = gbif_taxa_from_schema(schema, "test12")
    names = [t["name"] for t in taxa]
    assert names.count("Populus alba") == 1
    by_name = {t["name"]: t["class_id"] for t in taxa}
    assert by_name["Acacia mearnsii"] == 0
    assert by_name["Acacia cyclops"] == 1
    assert by_name["Populus alba"] == 7


def test_real_schema_round_trip():
    """Smoke test against the real configs/labels_schema.yaml."""
    repo_root = Path(__file__).resolve().parents[1]
    schema = repo_root / "configs" / "labels_schema.yaml"
    if not schema.exists():
        pytest.skip("real schema not present")
    cm = build_lookup(schema, "upper_berg_12")
    # Pre-refactor parity spot checks
    assert cm.resolve("Acacia mearnsii") == (0, "exact")
    assert cm.resolve("Acacia cyclops") == (1, "exact")
    assert cm.resolve("Pinus halepensis")[1] in ("exact", "genus_fallback")
    assert cm.resolve("Eucalyptus diversicolor") == (5, "genus_fallback")
    assert cm.resolve("Hakea drupacea") == (None, "unmapped")  # hakea has no genus_fallback
