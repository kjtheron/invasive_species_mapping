"""Region-aware months: pipeline.yaml zones must stay consistent with the embedder.

These are the failure modes that only bite at chip/embed time (expensive), so pin
them here: a zone whose month label has no day-of-year, a province mapped to a
non-existent zone, or zones with differing month counts (breaks the uniform-T batch).
"""

from __future__ import annotations

from cmrv.embeddings.constants import MONTH_DOY
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


def test_no_default_month_set() -> None:
    # There is no top-level `months`: every consumer resolves a zone first, so a
    # default calendar could only ever be the wrong one for half the country.
    assert "months" not in CFG


def test_month_year_is_a_placeholder() -> None:
    # Both consumers slice the year off and prepend their own (label year / --year).
    for months in CFG["months_by_zone"].values():
        for m in months:
            assert m["start"].startswith("0000-"), m
            assert m["end"].startswith("0000-"), m


def test_months_for_geom_picks_the_province_calendar() -> None:
    """Inference must composite on the same calendar its training chips used.

    A KZN chip was embedded from jul/sep/dec (day-of-year 196/258/349). Compositing
    that ground on the winter feb/may/sep set feeds the head a temporal encoding it
    never saw for that region — skew, not just different imagery.
    """
    from shapely.geometry import box

    from cmrv.aoi import months_for_geom

    assert months_for_geom(box(18.4, -34.0, 18.6, -33.8), CFG)[0] == "winter_rainfall"  # Cape Town
    assert months_for_geom(box(30.9, -29.9, 31.1, -29.7), CFG)[0] == "summer_rainfall"  # Durban
    assert [m["label"] for m in months_for_geom(box(30.9, -29.9, 31.1, -29.7), CFG)[1]] == [
        "jul",
        "sep",
        "dec",
    ]


def test_months_for_geom_raises_outside_sa() -> None:
    """Refuse to guess, exactly as ingest-chips does for an unmapped province."""
    import pytest
    from shapely.geometry import box

    from cmrv.aoi import months_for_geom

    with pytest.raises(ValueError, match="outside every SA province"):
        months_for_geom(box(0.0, 0.0, 0.1, 0.1), CFG)
