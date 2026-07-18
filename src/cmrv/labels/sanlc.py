"""SANLC accuracy-assessment points + VegMap 2024 → land-cover training labels.

The SANLC 2018/2020/2022 **accuracy-assessment points** are field/reference-verified
land-cover reference data — the truth used to *validate* the SANLC maps. They are
far better training labels than sampling the SANLC raster (which is the model's own
prediction). Each point carries a year, slotting into per-label imagery-year
alignment (2018→2018 S2, …).

Pipeline: load all years → map each point's land-cover class to our scheme via
``ACC_CLASS_TO_CLASS`` → de-duplicate points identical across years (same location
+ class, keep latest) → clip to WC → name natural-vegetation points by VegMap 2024
biome (``T_BIOME``; SANLC's natural classes don't resolve the Cape biomes) →
exclude known-IAP areas → emit ``source=sanlc``. Feeds the unified
``western_cape_landcover`` class map alongside the IAP genera.
"""

from __future__ import annotations

import datetime as dt
import glob
from pathlib import Path

import geopandas as gpd
import pandas as pd
import pyogrio
from loguru import logger

from cmrv.io import read_gdf
from cmrv.labels.observations import PROCESSED_ROOT, make_run_id, read_all, write_partition

DATASET = "sanlc_accuracy_points"
SOURCE = "sanlc"
POINTS_DIR = Path("data/labels/raw/sanlc_accuracy_points")
VEGMAP_SHP = Path("data/labels/raw/vegmap_2024/Shapefile/NVM2024Final_IEM5_12_07012025.shp")
AOI = "data/aoi/processed/western_cape.parquet"
YEARS = (2018, 2020, 2022)

SANLC_URL = "https://www.dffe.gov.za/egis"
VEGMAP_URL = "https://bgis.sanbi.org/Projects/Detail/2258"
LICENSE = "SANLC accuracy points (DFFE) + VegMap 2024 (SANBI) — free, cite sources"
COORD_UNCERTAINTY_M = 20.0
UTM34S = "EPSG:32734"

# SANLC accuracy-point class name → our class. "NATURAL" → named by VegMap biome.
ACC_CLASS_TO_CLASS: dict[str, str] = {
    # natural vegetation → VegMap biome
    "low shrubland (fynbos)": "NATURAL",
    "low shrubland (succulent karoo)": "NATURAL",
    "low shrubland (nama karoo)": "NATURAL",
    "low shrubland (other)": "NATURAL",
    "dense forest & woodland": "NATURAL",
    "contiguous (indigenous) forest": "NATURAL",
    "contiguous low forest & thicket": "NATURAL",
    "open woodland": "NATURAL",
    "sparsely wooded grassland": "NATURAL",
    "natural grassland": "NATURAL",
    "grassland": "NATURAL",
    # water
    "natural ocean & coastal": "water",
    "natural estuaries & lagoons": "water",
    "natural rivers": "water",
    "natural lakes": "water",
    "natural pans (flooded @ observation times)": "water",
    "artificial dams (including canals)": "water",
    "artificial sewage ponds": "water",
    "artificial flooded mine pits": "water",
    # wetland
    "wetlands": "wetland",
    "mangrove wetlands": "wetland",
    # bare / exposed surfaces
    "other bare": "bare",
    "natural rock surfaces": "bare",
    "coastal sand & dunes": "bare",
    "sand dunes (terrestrial)": "bare",
    "bare riverbed material": "bare",
    "dry pans": "bare",
    "eroded lands": "bare",
    "land-fills": "bare",
    "mines": "bare",
    # cultivated
    "commercial annual crops non-pivot irrigated": "cultivated",
    "commercial annual crops pivot irrigated": "cultivated",
    "commercial annual crops rain-fed / dryland": "cultivated",
    "cultivated commercial permanent orchards": "cultivated",
    "cultivated commercial permanent vines": "cultivated",
    "cultivated commercial permanent pineapples": "cultivated",
    "cultivated commercial sugarcane non-pivot": "cultivated",
    "cultivated commercial sugarcane pivot irrigated": "cultivated",
    "cultivated emerging farmer sugarcane non-pivot": "cultivated",
    "subsistence / small-scale annual crops": "cultivated",
    # built-up
    "commercial": "built_up",
    "industrial": "built_up",
    "residential formal": "built_up",
    "residential informal": "built_up",
    "urban recreational": "built_up",
    "village": "built_up",
    "smallholdings": "built_up",
    # planted forest — commercial alien plantation, distinct from invasive IAP
    "plantation": "planted_forest",
}
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


def _load_points() -> gpd.GeoDataFrame:
    """Load every year's accuracy points → ``(cname, year, geometry)`` in EPSG:4326."""
    frames = []
    for yr in YEARS:
        shp = next(s for s in glob.glob(f"{POINTS_DIR}/{yr}/*.shp") if "integrity" not in s.lower())
        g = gpd.read_file(shp)
        name_col = "Acc_Cls_Na" if "Acc_Cls_Na" in g.columns else "Class_name"
        g = g.assign(cname=g[name_col].str.lower().str.strip(), year=yr)
        frames.append(g[["cname", "year", "geometry"]].to_crs("EPSG:4326"))
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs="EPSG:4326")


def _vegmap_biome(points: gpd.GeoDataFrame) -> pd.Series:
    """Natural points → VegMap biome class via point-in-polygon (NaN if unmatched)."""
    vcrs = pyogrio.read_info(str(VEGMAP_SHP))["crs"]
    bbox = tuple(points.to_crs(vcrs).total_bounds)
    veg = gpd.read_file(str(VEGMAP_SHP), bbox=bbox, columns=["T_BIOME"])
    joined = gpd.sjoin(points.to_crs(veg.crs), veg, how="left", predicate="within")
    joined = joined[~joined.index.duplicated(keep="first")]  # edge: a point in >1 polygon
    return joined["T_BIOME"].map(BIOME_TO_CLASS).reindex(points.index)


def ingest_sanlc(
    iap_buffer_m: float = 320.0,
    root: str = PROCESSED_ROOT,
    run_id: str | None = None,
) -> str:
    """Ingest SANLC accuracy points + VegMap biome → ``source=sanlc`` store."""
    run_id = run_id or make_run_id(SOURCE)
    ingested_at = dt.datetime.now(tz=dt.UTC)

    pts = _load_points()
    logger.info("accuracy points (all years): {}", len(pts))
    pts["cls"] = pts["cname"].map(ACC_CLASS_TO_CLASS)
    unmapped = sorted(pts.loc[pts["cls"].isna(), "cname"].unique())
    if unmapped:
        logger.warning("unmapped class names dropped: {}", unmapped)
    pts = pts[pts["cls"].notna()].reset_index(drop=True)

    # de-dup points identical across years (same location + class) — keep latest year.
    key = pts.geometry.x.round(5).astype(str) + "_" + pts.geometry.y.round(5).astype(str)
    pts["_key"] = key + "_" + pts["cname"]
    n0 = len(pts)
    pts = pts.sort_values("year").drop_duplicates("_key", keep="last").reset_index(drop=True)
    logger.info(
        "after de-dup identical (loc+class) across years: {} (-{})", len(pts), n0 - len(pts)
    )

    aoi = read_gdf(AOI)
    pts = pts[pts.geometry.within(aoi.union_all())].reset_index(drop=True)
    logger.info("in WC AOI: {}", len(pts))

    nat = pts["cls"] == "NATURAL"
    if nat.any():
        pts.loc[nat, "cls"] = _vegmap_biome(pts[nat]).to_numpy()
    pts = pts[pts["cls"].notna() & (pts["cls"] != "NATURAL")].reset_index(drop=True)
    logger.info("after VegMap biome join: {}", len(pts))

    # exclude points whose 640 m chip would contain a known IAP field point.
    # Only species/genus rows are IAP observations — MapWAPS also contributes native
    # (biome) + transformed (landcover) points, which must NOT trigger exclusion.
    store = read_all(root)  # native GeoParquet → geometry already shapely
    iap = store.loc[store["taxon_rank"].isin(("species", "genus")), "geometry"].to_crs(UTM34S)
    iap_buf = iap.buffer(iap_buffer_m).union_all()
    pts = pts[~pts.to_crs(UTM34S).geometry.within(iap_buf).to_numpy()].reset_index(drop=True)
    logger.info("after IAP exclusion ({} m): {}", iap_buffer_m, len(pts))
    logger.info("per-class:\n{}", pts["cls"].value_counts().to_string())

    rows = []
    for i, (geom, cls, yr) in enumerate(zip(pts.geometry, pts["cls"], pts["year"], strict=True)):
        natural = cls in NATURAL_CLASSES
        rows.append(
            {
                "obs_id": f"{SOURCE}:{geom.x:.6f}:{geom.y:.6f}:{yr}",
                "source": SOURCE,
                "source_record_id": str(i),
                "source_url": VEGMAP_URL if natural else SANLC_URL,
                "source_doi": None,
                "license": LICENSE,
                "species": cls,
                "species_normalized": cls,
                "taxon_rank": "biome" if natural else "landcover",
                "geom_type": "point",
                "coord_uncertainty_m": COORD_UNCERTAINTY_M,
                "event_date": f"{yr}-01-01",
                "basis_of_record": "SANLC_ACCURACY_POINT",
                "cover_pct": 100.0,
                "weight": 1.0,
                "ingested_at": ingested_at,
                "ingest_run_id": run_id,
                "aoi_admin1": "western_cape",
            }
        )

    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=list(pts.geometry), crs="EPSG:4326")
    path = write_partition(out, DATASET, root=root, run_id=run_id)
    logger.success("sanlc: {} rows → {}", len(rows), path)
    return path
