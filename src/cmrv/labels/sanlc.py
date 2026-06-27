"""SANLC 2022 + VegMap 2024 native-vegetation / land-cover label sampler.

Samples pixel-**interior** points within the WC AOI and labels them for the
unified land-cover map:

- **transformed / other** classes from SANLC 2022 (actual cover), grouped via its
  `SALCC_2` scheme and collapsed (built_up, cultivated, planted_forest, bare,
  water, wetland);
- **natural vegetation** pixels by VegMap 2024 biome (`T_BIOME`) — SANLC merges
  Fynbos + Karoo into one class, so VegMap supplies the floristic split.

Known-IAP areas are excluded (those labels come from the field adapters). Emits
`source=sanlc` rows to the obs store; the `western_cape_landcover` class map
crosswalks them alongside the IAP genera into one every-pixel map.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
import rasterio
from loguru import logger
from shapely import from_wkb

from cmrv.io import read_gdf
from cmrv.labels.observations import PROCESSED_ROOT, make_run_id, read_all, write_partition

DATASET = "sanlc_2022"
SOURCE = "sanlc"
SANLC_TIF = Path("data/labels/raw/sanlc_2022/SA_NLC_2022_ALBERS.tif")
VEGMAP_SHP = Path("data/labels/raw/vegmap_2024/Shapefile/NVM2024Final_IEM5_12_07012025.shp")
AOI = "data/aoi/processed/western_cape.parquet"

SANLC_URL = "https://www.dffe.gov.za/egis"
VEGMAP_URL = "https://bgis.sanbi.org/Projects/Detail/2258"
LICENSE = "SANLC 2022 (DFFE) + VegMap 2024 (SANBI) — free, cite sources"
COORD_UNCERTAINTY_M = 20.0  # one SANLC pixel
UTM34S = "EPSG:32734"

# SANLC SALCC_2 group → our class. "NATURAL" defers the label to the VegMap biome.
SALCC2_TO_CLASS: dict[str, str] = {
    "Karoo & Fynbos Shubland": "NATURAL",
    "Natural Wooded Land": "NATURAL",
    "Natural Grassland": "NATURAL",
    "Shrubs": "NATURAL",
    "Natural Waterbodies": "water",
    "Artificial Waterbodies": "water",
    "Herbaceous Wetlands": "wetland",
    "Woody Wetlands": "wetland",
    "Temporary Crops": "cultivated",
    "Permanent Crops": "cultivated",
    "Fallow Lands & Old Fields": "cultivated",
    "Planted Forest": "planted_forest",
    "Residential": "built_up",
    "Commercial": "built_up",
    "Industrial": "built_up",
    "Village": "built_up",
    "Smallholdings": "built_up",
    "Transport": "built_up",
    "Surface Infrastructure": "built_up",
    "Urban Vegetation": "built_up",
    "Extraction Sites": "bare",
    "Waste & Resource Dumps": "bare",
    "Consolidated": "bare",
    "Unconsolidated": "bare",
}
# VegMap T_BIOME → biome class.
BIOME_TO_CLASS: dict[str, str] = {
    "Fynbos": "fynbos",
    "Succulent Karoo": "succulent_karoo",
    "Nama-Karoo": "nama_karoo",
    "Albany Thicket": "albany_thicket",
    "Forests": "forest",
    "Azonal Vegetation": "azonal",
    "Grassland": "grassland",
}
NATURAL_CLASSES = frozenset(BIOME_TO_CLASS.values())


def _sanlc_class(cand: gpd.GeoDataFrame, src, vat: pd.Series, homogeneity_m: float) -> pd.Series:
    """Collapsed SANLC class per candidate, NaN unless the ``homogeneity_m`` window
    around it is a single class (pixel interior — avoids mixed/edge pixels)."""
    pa = cand.to_crs(src.crs)
    xy = np.column_stack([pa.geometry.x.to_numpy(), pa.geometry.y.to_numpy()])

    def vals(arr):
        return np.array([v[0] for v in src.sample(arr)])

    center = vals(xy)
    homog = np.ones(len(xy), dtype=bool)
    for off in [(homogeneity_m, 0), (-homogeneity_m, 0), (0, homogeneity_m), (0, -homogeneity_m)]:
        homog &= vals(xy + np.array(off)) == center

    cls = pd.Series(center, index=cand.index).map(vat).map(SALCC2_TO_CLASS)
    return cls.where(homog)


def _vegmap_biome(points: gpd.GeoDataFrame) -> pd.Series:
    """Natural points → VegMap biome class via point-in-polygon (NaN if unmatched)."""
    vcrs = pyogrio.read_info(str(VEGMAP_SHP))["crs"]
    bbox = tuple(points.to_crs(vcrs).total_bounds)
    veg = gpd.read_file(str(VEGMAP_SHP), bbox=bbox, columns=["T_BIOME"])
    joined = gpd.sjoin(points.to_crs(veg.crs), veg, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]  # edge: a point in >1 polygon
    return joined["T_BIOME"].map(BIOME_TO_CLASS).reindex(points.index)


def ingest_sanlc(
    n_per_class: int = 500,
    pool: int = 40000,
    homogeneity_m: float = 40.0,
    iap_buffer_m: float = 320.0,
    root: str = PROCESSED_ROOT,
    run_id: str | None = None,
    seed: int = 42,
) -> str:
    """Sample SANLC/VegMap land-cover labels in the WC AOI → ``source=sanlc`` store."""
    run_id = run_id or make_run_id(SOURCE)
    ingested_at = dt.datetime.now(tz=dt.UTC)

    aoi = read_gdf(AOI)
    pts = aoi.geometry.sample_points(pool, rng=seed).explode(index_parts=False, ignore_index=True)
    cand = gpd.GeoDataFrame(geometry=list(pts), crs=aoi.crs)
    logger.info("SANLC sampler: {} candidate points in AOI", len(cand))

    # SANLC class + pixel-interior gate
    with rasterio.open(SANLC_TIF) as src:
        vat = gpd.read_file(str(SANLC_TIF) + ".vat.dbf", ignore_geometry=True).set_index("Value")[
            "SALCC_2"
        ]
        cand["cls"] = _sanlc_class(cand, src, vat, homogeneity_m)
    cand = cand[cand["cls"].notna()].reset_index(drop=True)
    logger.info("after interior + known-class gate: {}", len(cand))

    # natural pixels → VegMap biome
    nat = cand["cls"] == "NATURAL"
    if nat.any():
        cand.loc[nat, "cls"] = _vegmap_biome(cand[nat]).to_numpy()
    cand = cand[cand["cls"].notna() & (cand["cls"] != "NATURAL")].reset_index(drop=True)
    logger.info("after VegMap biome join: {}", len(cand))

    # drop natives whose 640 m chip would contain a known IAP point
    # (buffer ≈ chip half-width); only knows field IAP, not unmapped invasion
    store = read_all(root)
    store = store[store["source"] != SOURCE]
    iap = gpd.GeoSeries(
        [from_wkb(bytes(b)) for b in store["geometry"] if b is not None], crs="EPSG:4326"
    ).to_crs(UTM34S)
    iap_buf = iap.buffer(iap_buffer_m).union_all()
    inside = cand.to_crs(UTM34S).geometry.within(iap_buf).to_numpy()
    cand = cand[~inside].reset_index(drop=True)
    logger.info("after IAP exclusion ({} m): {}", iap_buffer_m, len(cand))

    # stratified cap per class
    parts = [g.sample(min(len(g), n_per_class), random_state=seed) for _, g in cand.groupby("cls")]
    cand = pd.concat(parts).reset_index(drop=True)
    logger.info("per-class counts:\n{}", cand["cls"].value_counts().to_string())

    cand = cand.to_crs("EPSG:4326")
    rows = []
    for i, geom, cls in zip(range(len(cand)), cand.geometry, cand["cls"], strict=True):
        natural = cls in NATURAL_CLASSES
        rows.append(
            {
                "obs_id": f"{SOURCE}:{geom.x:.6f}:{geom.y:.6f}",
                "source": SOURCE,
                "source_record_id": str(i),
                "source_url": VEGMAP_URL if natural else SANLC_URL,
                "source_doi": None,
                "license": LICENSE,
                "species": cls,
                "species_normalized": cls,
                "taxon_rank": "biome" if natural else "landcover",
                "gbif_usage_key": None,
                "geom_type": "point",
                "coord_uncertainty_m": COORD_UNCERTAINTY_M,
                "event_date": "2022-01-01",  # SANLC 2022 epoch; native cover is stable
                "basis_of_record": "SANLC_VEGMAP_SAMPLE",
                "cover_pct": 100.0,  # homogeneous-interior sample
                "weight": 1.0,
                "ingested_at": ingested_at,
                "ingest_run_id": run_id,
                "aoi_admin1": "western_cape",
            }
        )

    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=list(cand.geometry), crs="EPSG:4326")
    path = write_partition(out, DATASET, root=root, run_id=run_id)
    logger.success("sanlc: {} rows → {}", len(rows), path)
    return path
