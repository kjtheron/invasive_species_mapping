"""SANLC accuracy-assessment points + VegMap 2024 → land-cover training labels.

The SANLC 2018/2020/2022 **accuracy-assessment points** are field/reference-verified
land-cover reference data — the truth used to *validate* the SANLC maps.
Each point carries a year, slotting into per-label imagery-year
alignment (2018→2018 S2, …).

Pipeline: load all years → map each point's land-cover class to our scheme via
``ACC_CLASS_TO_CLASS`` → de-duplicate points identical across years (same location
+ class, keep latest) → clip to the national AOI → name natural-vegetation points by
VegMap 2024 biome (``T_BIOME``; SANLC's natural classes don't resolve the Cape biomes)
→ stamp each point's province by ADM1 join → exclude known-IAP areas → emit
``source=sanlc``. Feeds the unified ``sa_landcover`` class map alongside the IAP genera.

The points are national, so ``aoi_admin1`` is derived **per point** — it picks the
rainfall-zone month calendar at chip time, and one hardcoded province would silently
chip half the country on the wrong calendar.
"""

from __future__ import annotations

import datetime as dt
import glob
from pathlib import Path

import geopandas as gpd  # type: ignore
import pandas as pd  # type: ignore
import pyogrio  # type: ignore
from loguru import logger  # type: ignore

from cmrv.aoi import SA_ALBERS, province_of
from cmrv.io import load_config, read_gdf
from cmrv.labels.classmap import warn_unmapped
from cmrv.labels.observations import PROCESSED_ROOT, make_run_id, read_all, write_partition

DATASET = "sanlc_accuracy_points"
SOURCE = "sanlc"
POINTS_DIR = Path("data/labels/raw/sanlc_accuracy_points")
VEGMAP_SHP = Path("data/labels/raw/vegmap_2024/Shapefile/NVM2024Final_IEM5_12_07012025.shp")
PIPELINE = "configs/pipeline.yaml"
YEARS = (2018, 2020, 2022)

SANLC_URL = "https://www.dffe.gov.za/egis"
VEGMAP_URL = "https://bgis.sanbi.org/Projects/Detail/2258"
LICENSE = "SANLC accuracy points (DFFE) + VegMap 2024 (SANBI) — free, cite sources"
COORD_UNCERTAINTY_M = 20.0

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
    joined = gpd.sjoin(points.to_crs(veg.crs), veg, how="left", predicate="within") # type: ignore
    joined = joined[~joined.index.duplicated(keep="first")]  # edge: a point in >1 polygon
    return joined["T_BIOME"].map(BIOME_TO_CLASS).reindex(points.index)


def ingest_sanlc(
    iap_buffer_m: float = 320.0,
    root: str = PROCESSED_ROOT,
    run_id: str | None = None,
    pipeline: str = PIPELINE,
    replace: bool = True,
) -> str:
    """Ingest SANLC accuracy points + VegMap biome → ``source=sanlc`` store.

    ``replace=True`` by default, unlike every other adapter. This one's output is a
    function of the *whole store* — the IAP-exclusion buffer grows whenever another
    source adds species/genus rows — so an upsert would keep points a stricter run has
    since excluded, and the exclusion would silently do nothing.
    """
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

    aoi = read_gdf(load_config(pipeline)["aoi"]["train_path"])
    pts = pts[pts.geometry.within(aoi.union_all())].reset_index(drop=True)
    logger.info("in AOI: {}", len(pts))

    nat = pts["cls"] == "NATURAL"
    if nat.any():
        pts.loc[nat, "cls"] = _vegmap_biome(pts[nat]).to_numpy()
    pts = pts[pts["cls"].notna() & (pts["cls"] != "NATURAL")].reset_index(drop=True)
    logger.info("after VegMap biome join: {}", len(pts))

    # exclude points whose 640 m chip would contain a known IAP field point.
    # Only species/genus rows are IAP observations — MapWAPS also contributes native
    # (biome) + transformed (landcover) points, which must NOT trigger exclusion.
    store = read_all(root)  # native GeoParquet → geometry already shapely
    iap = store.loc[store["taxon_rank"].isin(("species", "genus")), "geometry"].to_crs(SA_ALBERS)
    iap_buf = iap.buffer(iap_buffer_m).union_all()
    pts = pts[~pts.to_crs(SA_ALBERS).geometry.within(iap_buf).to_numpy()].reset_index(drop=True)
    logger.info("after IAP exclusion ({} m): {}", iap_buffer_m, len(pts))
    logger.info("per-class:\n{}", pts["cls"].value_counts().to_string())

    # Province per point — it selects the rainfall-zone month calendar at chip time.
    # A point outside every ADM1 polygon (coastal/estuary jitter) has no calendar, so
    # drop it here rather than let ingest-chips raise on the whole run.
    pts["admin1"] = province_of(pts)
    n_nop = int(pts["admin1"].isna().sum())
    if n_nop:
        logger.warning("dropping {} point(s) outside every ADM1 province polygon", n_nop)
        pts = pts[pts["admin1"].notna()].reset_index(drop=True)
    logger.info("per-province:\n{}", pts["admin1"].value_counts().to_string())

    rows = []
    for i, (geom, cls, yr, prov) in enumerate(
        zip(pts.geometry, pts["cls"], pts["year"], pts["admin1"], strict=True)
    ):
        natural = cls in NATURAL_CLASSES
        rows.append(
            {
                "obs_id": f"{SOURCE}:{geom.x:.6f}:{geom.y:.6f}:{yr}", # type: ignore
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
                "aoi_admin1": prov,
            }
        )

    warn_unmapped({r["species_normalized"] for r in rows}, source=DATASET)
    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=list(pts.geometry), crs="EPSG:4326")
    path = write_partition(out, DATASET, root=root, run_id=run_id, replace=replace)
    logger.success("sanlc: {} rows → {}", len(rows), path)
    return path
