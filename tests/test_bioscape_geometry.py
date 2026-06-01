"""Tests for BioSCape geodesic geometry helpers."""

from __future__ import annotations

import math

import pytest

from cmrv.labels.bioscape import (
    _GEOD,
    LINE_COORD_UNCERTAINTY_M,
    PLOT_COORD_UNCERTAINTY_M,
    _mark_to_offset_m,
    _offset_point,
)

# Cape Town CBD as a known reference point (WGS84)
_LON0, _LAT0 = 18.4241, -33.9249


class TestCoordUncertainty:
    def test_line_uncertainty(self) -> None:
        """Line coord uncertainty = sqrt(5^2 + 1^2) ≈ 5.099 m."""
        expected = math.sqrt(5.0**2 + 1.0**2)
        assert abs(LINE_COORD_UNCERTAINTY_M - expected) < 1e-6

    def test_plot_uncertainty(self) -> None:
        """Plot coord uncertainty = sqrt(5^2 + 2.5^2) ≈ 5.590 m."""
        expected = math.sqrt(5.0**2 + 2.5**2)
        assert abs(PLOT_COORD_UNCERTAINTY_M - expected) < 1e-6


class TestMarkToOffset:
    def test_center_mark(self) -> None:
        """Mark 'C' → 0 m offset."""
        assert _mark_to_offset_m("C") == 0.0
        assert _mark_to_offset_m("c") == 0.0

    def test_start_mark(self) -> None:
        """Mark '1' on a 0–10 m transect → -4 m (1 - 5)."""
        assert _mark_to_offset_m("1") == pytest.approx(-4.0)

    def test_end_mark(self) -> None:
        """Mark '10' → +5 m (10 - 5)."""
        assert _mark_to_offset_m("10") == pytest.approx(5.0)

    def test_middle_mark(self) -> None:
        """Mark '5' → 0 m (5 - 5)."""
        assert _mark_to_offset_m("5") == pytest.approx(0.0)

    def test_mark_with_side_suffix(self) -> None:
        """Mark '3L' → strip 'L', result = 3 - 5 = -2 m."""
        assert _mark_to_offset_m("3L") == pytest.approx(-2.0)
        assert _mark_to_offset_m("7R") == pytest.approx(2.0)


class TestOffsetPoint:
    def test_north_offset(self) -> None:
        """Moving due north by 10 m increases latitude by ≈0.00009°."""
        lon2, lat2 = _offset_point(_LON0, _LAT0, az_deg=0.0, dist_m=10.0)
        assert lon2 == pytest.approx(_LON0, abs=1e-6), "longitude unchanged for due-north move"
        assert lat2 > _LAT0, "latitude increases when moving north"
        delta_lat = lat2 - _LAT0
        # 10 m / 111_320 m/° ≈ 0.0000898°
        assert 8e-5 < delta_lat < 1e-4

    def test_east_offset(self) -> None:
        """Moving due east by 10 m increases longitude."""
        lon2, lat2 = _offset_point(_LON0, _LAT0, az_deg=90.0, dist_m=10.0)
        assert lat2 == pytest.approx(_LAT0, abs=1e-6), "latitude unchanged for due-east move"
        assert lon2 > _LON0, "longitude increases when moving east"

    def test_negative_dist_reverses_azimuth(self) -> None:
        """Negative dist_m should move in the opposite direction."""
        lon_pos, lat_pos = _offset_point(_LON0, _LAT0, az_deg=0.0, dist_m=100.0)
        lon_neg, lat_neg = _offset_point(_LON0, _LAT0, az_deg=0.0, dist_m=-100.0)
        assert lat_pos > _LAT0
        assert lat_neg < _LAT0
        assert lat_pos == pytest.approx(-lat_neg + 2 * _LAT0, abs=1e-6)

    def test_roundtrip_precision(self) -> None:
        """Offset 10 m north then 10 m south returns to within 0.001 m of origin."""
        lon1, lat1 = _offset_point(_LON0, _LAT0, az_deg=0.0, dist_m=10.0)
        lon2, lat2 = _offset_point(lon1, lat1, az_deg=180.0, dist_m=10.0)
        # Convert degree difference to metres (rough)
        d_m = math.sqrt(
            ((lon2 - _LON0) * 111320 * math.cos(math.radians(_LAT0))) ** 2
            + ((lat2 - _LAT0) * 111320) ** 2
        )
        assert d_m < 0.001, f"round-trip error {d_m:.6f} m exceeds 0.001 m"

    def test_known_offset_within_tolerance(self) -> None:
        """Geodesic offset of 5.1 m at 45° (NE quadrant centre) stays within 0.1 m."""
        lon2, lat2 = _offset_point(_LON0, _LAT0, az_deg=45.0, dist_m=LINE_COORD_UNCERTAINTY_M)
        # Verify both coords moved in the right direction
        assert lon2 > _LON0
        assert lat2 > _LAT0
        # Back-compute distance using geodesic inverse
        _, _, dist_back = _GEOD.inv(lon2, lat2, _LON0, _LAT0)
        assert abs(abs(dist_back) - LINE_COORD_UNCERTAINTY_M) < 0.1
