"""Region-aware months: pipeline.yaml zones must stay consistent with the embedder.

These are the failure modes that only bite at chip/embed time (expensive), so pin
them here: a zone whose month label has no day-of-year, a province mapped to a
non-existent zone, or zones with differing month counts (breaks the uniform-T batch).
"""

from __future__ import annotations

from cmrv.embeddings.base import MONTH_DOY
from cmrv.io import load_config

CFG = load_config("configs/pipeline.yaml")


def test_every_province_zone_has_a_month_set() -> None:
    zones = CFG["months_by_zone"]
    for admin1, zone in CFG["admin1_zone"].items():
        assert zone in zones, f"{admin1} → {zone!r} has no months_by_zone entry"


def test_every_configured_month_has_a_doy() -> None:
    # embed.py looks up MONTH_DOY[label] per chip — a missing label is a KeyError at embed.
    for zone, months in CFG["months_by_zone"].items():
        for m in months:
            assert m["label"] in MONTH_DOY, f"{zone} month {m['label']!r} missing from MONTH_DOY"


def test_all_zones_same_month_count() -> None:
    # Uniform T across zones → obs from different zones batch into one tensor at embed.
    counts = {len(m) for m in CFG["months_by_zone"].values()}
    assert len(counts) == 1, f"zones have differing month counts {counts} — breaks uniform T"


def test_default_months_match_winter_zone() -> None:
    # `months` (used by infer / ingest-month) is the winter-rainfall set via a YAML anchor.
    assert CFG["months"] == CFG["months_by_zone"]["winter_rainfall"]
