"""Tests for Stage 2 STAC ingest + compositing (cloud_mask + composite)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import xarray as xr

from cmrv.ingest.chips import _reconcile_manifest, _window_medians
from cmrv.ingest.cloud_mask import BAD_SCL, apply_scl_mask
from cmrv.ingest.composite import monthly_median, write_composite_cog

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_da_with_scl(scl_values: np.ndarray) -> xr.DataArray:
    """Build (time=1, band=[B02, SCL], y=H, x=W) DataArray from a 2-D SCL array."""
    H, W = scl_values.shape
    b02 = np.ones((1, 1, H, W), dtype="float32")
    scl = scl_values[np.newaxis, np.newaxis, :, :].astype("float32")
    data = np.concatenate([b02, scl], axis=1)
    return xr.DataArray(
        data,
        dims=["time", "band", "y", "x"],
        coords={"band": ["B02", "SCL"]},
    )


def _make_composite(bands: int = 3, H: int = 16, W: int = 16) -> xr.DataArray:
    """Synthetic (band, y, x) composite with UTM-like x/y coordinates."""
    arr = np.random.default_rng(42).random((bands, H, W)).astype("float32")
    y_coords = np.linspace(6_240_500.0, 6_240_500.0 - (H - 1) * 10.0, H)
    x_coords = np.linspace(230_000.0, 230_000.0 + (W - 1) * 10.0, W)
    return xr.DataArray(
        arr,
        dims=["band", "y", "x"],
        coords={
            "band": [f"B0{i}" for i in range(bands)],
            "y": y_coords,
            "x": x_coords,
        },
    )


# ---------------------------------------------------------------------------
# apply_scl_mask
# ---------------------------------------------------------------------------


class TestApplySclMask:
    def test_bad_scl_pixels_become_nan(self) -> None:
        scl = np.array([[3, 4], [8, 5]], dtype="float32")  # 3,8 bad; 4,5 clear
        da = _make_da_with_scl(scl)
        result = apply_scl_mask(da)
        assert np.isnan(result.values[0, 0, 0, 0]), "SCL=3 (cloud shadow) not masked"
        assert np.isnan(result.values[0, 0, 1, 0]), "SCL=8 (cloud med) not masked"
        assert not np.isnan(result.values[0, 0, 0, 1]), "SCL=4 (vegetation) masked by mistake"
        assert not np.isnan(result.values[0, 0, 1, 1]), "SCL=5 (bare) masked by mistake"

    def test_scl_band_excluded_from_output(self) -> None:
        scl = np.array([[4, 5]], dtype="float32")
        da = _make_da_with_scl(scl)
        result = apply_scl_mask(da)
        assert "SCL" not in result.band.values.tolist()

    def test_all_bad_scl_returns_all_nan(self) -> None:
        scl = np.array([[3, 8], [9, 10]], dtype="float32")
        da = _make_da_with_scl(scl)
        result = apply_scl_mask(da)
        assert np.isnan(result.values).all()

    def test_all_clear_scl_values_unchanged(self) -> None:
        scl = np.array([[4, 5], [6, 7]], dtype="float32")
        da = _make_da_with_scl(scl)
        result = apply_scl_mask(da)
        np.testing.assert_array_equal(result.values, np.ones((1, 1, 2, 2), dtype="float32"))

    def test_all_bad_scl_codes_are_masked(self) -> None:
        """Every value in BAD_SCL should produce NaN in the output."""
        for bad_val in BAD_SCL:
            scl = np.array([[bad_val]], dtype="float32")
            da = _make_da_with_scl(scl)
            result = apply_scl_mask(da)
            assert np.isnan(result.values).all(), f"SCL={bad_val} not masked"

    def test_multiple_sr_bands_all_masked(self) -> None:
        """When SCL is bad, all SR bands at that pixel become NaN."""
        H, W = 2, 2
        scl_vals = np.array([[3, 4], [4, 4]], dtype="float32")
        b02 = np.ones((1, 1, H, W), dtype="float32") * 0.1
        b03 = np.ones((1, 1, H, W), dtype="float32") * 0.2
        scl = scl_vals[np.newaxis, np.newaxis]
        data = np.concatenate([b02, b03, scl], axis=1)
        da = xr.DataArray(
            data,
            dims=["time", "band", "y", "x"],
            coords={"band": ["B02", "B03", "SCL"]},
        )
        result = apply_scl_mask(da)
        # pixel (0,0) has SCL=3 → both bands NaN
        assert np.isnan(result.values[0, 0, 0, 0])
        assert np.isnan(result.values[0, 1, 0, 0])
        # pixel (0,1) has SCL=4 → both bands valid
        assert not np.isnan(result.values[0, 0, 0, 1])
        assert not np.isnan(result.values[0, 1, 0, 1])


# ---------------------------------------------------------------------------
# monthly_median
# ---------------------------------------------------------------------------


class TestMonthlyMedian:
    def test_reduces_time_dim(self) -> None:
        arr = np.ones((3, 2, 4, 4), dtype="float32")  # (T=3, B=2, H=4, W=4)
        da = xr.DataArray(arr, dims=["time", "band", "y", "x"])
        result = monthly_median(da)
        assert result.dims == ("band", "y", "x")
        assert result.shape == (2, 4, 4)

    def test_correct_median_value(self) -> None:
        # (T=3, B=1, H=1, W=1) with time values [1, 3, 5] → median = 3
        data = np.array([1.0, 3.0, 5.0], dtype="float32").reshape(3, 1, 1, 1)
        da = xr.DataArray(data, dims=["time", "band", "y", "x"])
        result = monthly_median(da)
        assert float(result.values[0, 0, 0]) == pytest.approx(3.0)

    def test_skipna_uses_remaining_valid_pixels(self) -> None:
        """NaN at one time step → median over remaining valid observations."""
        # (T=2, B=1, H=1, W=2): time 0 has NaN at x=1; time 1 is fully valid
        data = np.array([[[[1.0, np.nan]]], [[[3.0, 5.0]]]], dtype="float32")
        da = xr.DataArray(data, dims=["time", "band", "y", "x"])
        result = monthly_median(da)
        # x=0: median of [1.0, 3.0] = 2.0
        assert float(result.values[0, 0, 0]) == pytest.approx(2.0)
        # x=1: median of [5.0] (NaN skipped) = 5.0
        assert float(result.values[0, 0, 1]) == pytest.approx(5.0)

    def test_all_nan_slice_stays_nan(self) -> None:
        arr = np.full((2, 1, 1, 1), np.nan, dtype="float32")
        da = xr.DataArray(arr, dims=["time", "band", "y", "x"])
        result = monthly_median(da)
        assert np.isnan(result.values).all()


# ---------------------------------------------------------------------------
# write_composite_cog
# ---------------------------------------------------------------------------


class TestWriteCompositeCog:
    def test_writes_valid_geotiff(self) -> None:
        composite = _make_composite(bands=10, H=32, W=32)
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "composite.tif")
            write_composite_cog(composite, out)
            assert Path(out).exists()
            with rasterio.open(out) as src:
                assert src.count == 10
                assert src.dtypes[0] == "float32"
                assert src.crs.to_epsg() == 32734
                data = src.read()
                assert data.shape == (10, 32, 32)

    def test_cog_passes_rio_cogeo_validate(self) -> None:
        from rio_cogeo.cogeo import cog_validate

        composite = _make_composite(bands=3, H=64, W=64)
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "composite.tif")
            write_composite_cog(composite, out)
            is_valid, _, _ = cog_validate(out)
            assert is_valid, "COG failed rio-cogeo validation"

    def test_nodata_preserved(self) -> None:
        composite = _make_composite(bands=1, H=16, W=16)
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "nodata.tif")
            write_composite_cog(composite, out, nodata=-9999.0)
            with rasterio.open(out) as src:
                assert src.nodata == pytest.approx(-9999.0)

    def test_2d_array_written_as_single_band(self) -> None:
        """A (y, x) DataArray (squeezed single band) is written as 1-band GeoTIFF."""
        composite = _make_composite(bands=1, H=16, W=16).squeeze("band", drop=True)
        assert composite.ndim == 2
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "single.tif")
            write_composite_cog(composite, out)
            with rasterio.open(out) as src:
                assert src.count == 1


# ---------------------------------------------------------------------------
# _window_medians  (per-label window compute — only kept pixels materialised)
# ---------------------------------------------------------------------------


def _make_stack(T: int = 2, B: int = 2, H: int = 8, W: int = 8, fill: float = 1.0) -> xr.DataArray:
    """Synthetic (time, band, y, x) stack with UTM-like x/y coords."""
    arr = np.full((T, B, H, W), fill, dtype="float32")
    y = np.linspace(6_240_000.0, 6_240_000.0 - (H - 1) * 10.0, H)
    x = np.linspace(230_000.0, 230_000.0 + (W - 1) * 10.0, W)
    return xr.DataArray(arr, dims=["time", "band", "y", "x"], coords={"y": y, "x": x})


class TestWindowMedians:
    def test_in_bounds_window_median(self) -> None:
        stack = _make_stack(fill=2.0)
        cx, cy = float(stack.x.values[4]), float(stack.y.values[4])
        (result,) = _window_medians(stack, [(cx, cy)], chip_px=4)
        arr, _tf, valid_frac = result
        assert arr.shape == (2, 4, 4)
        assert valid_frac == pytest.approx(1.0)
        assert np.allclose(arr, 2.0)

    def test_out_of_bounds_rejected(self) -> None:
        stack = _make_stack()
        cx, cy = float(stack.x.values[0]), float(stack.y.values[0])  # corner → window off-edge
        assert _window_medians(stack, [(cx, cy)], chip_px=4)[0] == "oob"

    def test_low_valid_frac_rejected(self) -> None:
        stack = _make_stack(fill=np.nan)  # all-NaN window → valid_frac 0
        cx, cy = float(stack.x.values[4]), float(stack.y.values[4])
        assert _window_medians(stack, [(cx, cy)], chip_px=4)[0].startswith("low_valid_frac")


def test_thin_labels_order_independent_and_stable() -> None:
    """Thinning survivor is the smallest obs_id per cell — independent of input order."""
    import geopandas as gpd
    from shapely.geometry import Point

    from cmrv.ingest.chips import thin_labels

    pts = gpd.GeoDataFrame(
        {"obs_id": [f"obs{i}" for i in range(8)], "species_normalized": ["Pinus"] * 8},
        geometry=[Point(19.0 + 1e-5 * i, -32.0) for i in range(8)],
        crs="EPSG:4326",
    )
    a = set(thin_labels(pts, thin_m=20.0)["obs_id"])
    b = set(thin_labels(pts.iloc[::-1].reset_index(drop=True), thin_m=20.0)["obs_id"])
    assert a == b  # independent of input row order
    assert "obs0" in a  # survivor = smallest obs_id in the cell
    assert len(a) < 8  # near-duplicates collapsed


def test_reconcile_manifest_prunes_thinned_out_obs(tmp_path):
    """Stale obs (not in the thinned set) lose their chips + manifest rows."""
    (tmp_path / "keep").mkdir()
    (tmp_path / "drop").mkdir()
    ck = tmp_path / "keep" / "feb.tif"
    cd = tmp_path / "drop" / "feb.tif"
    ck.write_bytes(b"x")
    cd.write_bytes(b"x")
    man = pd.DataFrame(
        {"obs_id": ["keep", "drop"], "chip_uri": [str(ck), str(cd)], "month_label": ["feb", "feb"]}
    )
    muri = str(tmp_path / "manifest.parquet")

    kept = _reconcile_manifest(man, {"keep"}, muri)

    assert set(kept["obs_id"]) == {"keep"}
    assert ck.exists() and not cd.exists()  # stale chip file deleted
    assert not (tmp_path / "drop").exists()  # emptied obs dir removed
    assert set(pd.read_parquet(muri)["obs_id"]) == {"keep"}  # manifest rewritten
