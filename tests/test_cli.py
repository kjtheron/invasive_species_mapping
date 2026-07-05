"""CLI helper tests — bbox parsing (negatives must survive)."""

from __future__ import annotations

import pytest

from cmrv.cli import parse_bbox


def test_parse_bbox_ok_with_negative_lat() -> None:
    assert parse_bbox("19.21,-33.20,19.25,-33.16") == (19.21, -33.20, 19.25, -33.16)


@pytest.mark.parametrize(
    "s",
    [
        "19.21,-33.20,19.25",  # too few
        "19.21,-33.20,19.25,-33.16,0",  # too many
        "19.25,-33.20,19.21,-33.16",  # min_lon >= max_lon
        "19.21,-33.16,19.25,-33.20",  # min_lat >= max_lat
        "a,b,c,d",  # non-numeric
    ],
)
def test_parse_bbox_rejects_bad(s: str) -> None:
    with pytest.raises(ValueError):
        parse_bbox(s)
