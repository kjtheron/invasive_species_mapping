"""Plot random IAP chip samples with label point overlay → PNGs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer

from cmrv.io import open_raster
from cmrv.labels.merge import load_training_labels


def percentile_stretch(rgb: np.ndarray, p_lo: float = 2, p_hi: float = 98) -> np.ndarray:
    """Per-band percentile stretch for display."""
    out = np.empty_like(rgb, dtype=np.float32)
    for b in range(rgb.shape[-1]):
        lo, hi = np.nanpercentile(rgb[..., b], [p_lo, p_hi])
        out[..., b] = np.clip((rgb[..., b] - lo) / max(hi - lo, 1e-6), 0, 1)
    return out


def plot_chip(
    chip_uri: str,
    obs_id: str,
    species: str,
    month: str,
    year: int,
    lon: float,
    lat: float,
    out_path: Path,
) -> None:
    with open_raster(chip_uri) as src:
        # Sentinel-2: assume B02,B03,B04,B08 ordering. RGB = B04,B03,B02.
        band_count = src.count
        arr = src.read().astype(np.float32)  # (bands, H, W)
        transform = src.transform
        crs = src.crs

        rgb = np.dstack([arr[2], arr[1], arr[0]]) if band_count >= 4 else np.dstack([arr[0]] * 3)

    rgb = percentile_stretch(rgb)

    # Convert label lon/lat to the chip's CRS, then to pixel coords
    t = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x_crs, y_crs = t.transform(lon, lat)
    col = (x_crs - transform.c) / transform.a
    row = (y_crs - transform.f) / transform.e

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb)
    ax.scatter([col], [row], s=120, facecolors="none", edgecolors="red", linewidths=2)
    ax.scatter([col], [row], s=15, c="red", marker="+")
    ax.set_title(f"{species}\n{obs_id} | {month} {year}", fontsize=10)
    ax.set_xlabel(f"lon {lon:.5f}, lat {lat:.5f}", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main(n: int = 10, seed: int = 42, out_dir: str = ".tmp/chip_previews") -> None:
    manifest = pd.read_parquet("gs://ism-data/chips/train/manifest.parquet")
    iap_pattern = "acacia|pinus|eucalyptus|hakea|populus|sesbania|leptospermum"
    iap = manifest[manifest["species"].str.lower().str.contains(iap_pattern, na=False)]

    labels = load_training_labels(aoi_uri="gs://ism-data/aoi/western_cape.parquet")
    labels_wgs = labels.to_crs("EPSG:4326")
    labels_wgs["lon"] = labels_wgs.geometry.x
    labels_wgs["lat"] = labels_wgs.geometry.y
    obs_xy = labels_wgs[["obs_id", "lon", "lat"]].drop_duplicates("obs_id")

    rng = np.random.default_rng(seed)
    sample = iap.sample(n=n, random_state=rng.integers(2**31 - 1))
    sample = sample.merge(obs_xy, on="obs_id", how="left").dropna(subset=["lon", "lat"])

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for i, row in enumerate(sample.itertuples(), start=1):
        fname = out / f"{i:02d}_{row.species.replace(' ', '_')}_{row.month_label}{row.year}.png"
        plot_chip(
            chip_uri=row.chip_uri,
            obs_id=row.obs_id,
            species=row.species,
            month=row.month_label,
            year=row.year,
            lon=row.lon,
            lat=row.lat,
            out_path=fname,
        )
        print(f"wrote {fname}")


if __name__ == "__main__":
    main()
