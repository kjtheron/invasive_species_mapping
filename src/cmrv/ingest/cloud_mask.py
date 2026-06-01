"""SCL-based cloud and shadow masking for Sentinel-2 stackstac DataArrays."""

from __future__ import annotations

import xarray as xr

# SCL values that indicate unusable pixels.
# 3 = cloud shadow, 8 = cloud medium probability,
# 9 = cloud high probability, 10 = thin cirrus.
BAD_SCL: frozenset[int] = frozenset({3, 8, 9, 10})


def apply_scl_mask(
    da: xr.DataArray,
    bad_values: frozenset[int] = BAD_SCL,
) -> xr.DataArray:
    """Mask cloud/shadow pixels using the SCL band.

    Args:
        da: (time, band, y, x) DataArray that includes a ``'SCL'`` band.
        bad_values: SCL integer codes to treat as invalid.

    Returns:
        DataArray with the SCL band removed; pixels whose SCL value is in
        ``bad_values`` are set to NaN across all remaining bands.
    """
    scl = da.sel(band="SCL")  # (time, y, x)
    valid = ~scl.isin(list(bad_values))  # True where pixel is usable
    sr_bands = [str(b) for b in da.band.values if b != "SCL"]
    return da.sel(band=sr_bands).where(valid)
