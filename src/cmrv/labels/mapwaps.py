"""MapWAPS ingest — field IAP + land-cover training points → unified store.

Multiple MapWAPS catchments (Rebelo, Cogill, Skosana et al., Stellenbosch Univ. /
figshare, CC-BY 4.0), each a shapefile of field-surveyed points. Every catchment
uses a slightly different column layout / CRS / class vocabulary, so a
per-catchment :class:`Catchment` config normalises them and a single master
``_LULC_TO_CLASS`` crosswalk maps every class string to a ``sa_landcover``
member (the same strings SANLC emits, so ``make-split`` resolves them).

Decisions baked in (see each catchment's metadata PDF):

- **Geometry used as-is.** Points are already distance/direction corrected; we only
  reproject to EPSG:4326 (setting a source CRS first where the shapefile lacks one).
- **cover_pct ← the density column** (estimated %); ``0 → None`` (ambiguous: truly
  sparse vs not recorded). Native / transformed classes carry density 0 → cover None.
- **taxon_rank = genus** for the Alien_* IAP classes — the survey is genus-level.
- **Only mappable classes ingested.** Shade / Burnt / Bracken / Alien_Other resolve
  to no class and are dropped at ingest.

Catchments processed: Olifants-Doring (WC), Tugela (KZN), uMzimvubu (EC). The
Luvuvhu and Sabie-Crocodile figshare articles ship broken TrainingData (a duplicate
of Tugela / an empty folder respectively) and are omitted — see download/README.md.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from cmrv.labels.observations import PROCESSED_ROOT, make_run_id, write_partition

SOURCE = "mapwaps"
RAW_ROOT = Path("data/labels/raw")

# Residual positional uncertainty (m): field-GPS + ~10 m S2-pixel residual after the
# published distance/direction correction. ponytail: constant, refine if a figure surfaces.
COORD_UNCERTAINTY_M = 15.0

# MapWAPS class string → (species_normalized, taxon_rank) for the
# ``sa_landcover`` map. Union across all catchments. Alien_* → IAP genus;
# native veg → VegMap-biome member; transformed → land-cover member. Classes NOT here
# are dropped at ingest: "Shade" (shadow), "Burnt" (transient scar), "Bracken"
# (indigenous fern, no land-cover class), "Alien_Other" (unspecific alien).
_LULC_TO_CLASS: dict[str, tuple[str, str]] = {
    # --- alien invasive trees → IAP genus (survey didn't resolve to species) ---
    "Alien_Pine": ("Pinus", "genus"),
    "Alien_Gum": ("Eucalyptus", "genus"),
    "Alien_Wattle": ("Acacia", "genus"),
    "Alien_Black Wattle": ("Acacia", "genus"),  # A. mearnsii (uMzimvubu)
    "Alien_Silver Wattle": ("Acacia", "genus"),  # A. dealbata (uMzimvubu)
    "Alien_Prosopis": ("Prosopis", "genus"),
    "Alien_Poplar": ("Populus", "genus"),  # Populus × canescens
    # --- native biomes / vegetation → VegMap-biome member ---
    "Fynbos-High density": ("fynbos", "biome"),
    "Fynbos - Low density": ("fynbos", "biome"),
    "Renosterveld": ("renosterveld", "biome"),  # own class, kept separate from fynbos
    "Succulent Karoo": ("succulent_karoo", "biome"),
    "Bushmanland Shrubland": ("nama_karoo", "biome"),  # Bushmanland bioregion ⊂ Nama-Karoo
    "Grassland": ("grassland", "biome"),
    "Indigenous Forest": ("forest", "biome"),
    "Indigenous Bush_Other": ("savanna", "biome"),  # native woody non-forest
    "Indigenous Bush_Vachellia": ("savanna", "biome"),  # thornveld (Vachellia)
    "Indigenous Bush_Leucosidea": ("savanna", "biome"),  # montane scrub (Leucosidea)
    "Riparian Bush": ("azonal", "biome"),  # riparian = azonal (intrazonal) vegetation
    "Riparian Trees": ("azonal", "biome"),
    # --- transformed / land cover ---
    "Irrigated Agriculture": ("cultivated", "landcover"),
    "Dryland Agriculture": ("cultivated", "landcover"),
    "Maize": ("cultivated", "landcover"),
    "Urban": ("built_up", "landcover"),
    "Bare Ground": ("bare", "landcover"),
    "Rock": ("bare", "landcover"),
    "Water": ("water", "landcover"),
    "Wetland": ("wetland", "landcover"),
    "Wetland - Reed": ("wetland", "landcover"),
    "Wetland_Other": ("wetland", "landcover"),
    "Wetland - Palmiet": ("wetland", "landcover"),
}


@dataclass(frozen=True)
class Catchment:
    """One MapWAPS shapefile + how to read it (columns / CRS / provenance differ)."""

    dataset: str  # partition dir under data/labels/{raw,processed}/ (∈ KNOWN_DATASETS)
    shp: str  # path to the .shp, relative to data/labels/raw/<dataset>/
    class_col: str  # the LULC class column
    density_col: str | None  # IAP density % column (→ cover_pct); None if absent
    date_col: str | None  # survey-date column; None if the shapefile carries no date
    src_crs: str | None  # source CRS to assume when the shapefile declares none
    aoi_admin1: str  # province (provenance)
    campaign_date: str  # ISO fallback date for undated points (single field campaign)
    doi: str
    url: str
    license: str


_CCBY = "CC-BY-4.0"
CATCHMENTS: dict[str, Catchment] = {
    "mapwaps_olifants_doring": Catchment(
        dataset="mapwaps_olifants_doring",
        shp="OlifantsDoring_TrainingData_23Classes/OlifantsDoring_trainingdata.shp",
        class_col="LULC_Class",
        density_col="Density___",
        date_col="DateTime",
        src_crs=None,
        aoi_admin1="western_cape",
        campaign_date="2023-01-01",
        doi="10.25413/sun.29958053",
        url="https://doi.org/10.25413/sun.29958053",
        license="CC-BY-4.0 / CC-BY-SA (ambiguous); co-authorship offer expected for academic use",
    ),
    "mapwaps_tugela": Catchment(
        dataset="mapwaps_tugela",
        shp="MapWAPS_Tugela_TrainingData/Trainingdata_Tugela_18Classes/Trainingdata_Tugela_18Classes.shp",
        class_col="LULC_Class",
        density_col="IAP_Densit",
        date_col=None,  # shapefile carries no survey date
        src_crs=None,  # already EPSG:4326
        aoi_admin1="kwazulu_natal",
        campaign_date="2023-01-01",  # ponytail: MapWAPS Tugela campaign ~2023; refine from metadata
        doi="10.25413/sun.25066151",
        url="https://doi.org/10.25413/sun.25066151",
        license=_CCBY,
    ),
    "mapwaps_umzimvubu": Catchment(
        dataset="mapwaps_umzimvubu",
        shp="MapWAPS_uMzimvubu_TrainingData/TrainingData_uMzim_18Classes/TrainingData_uMzim_18Classes.shp",
        class_col="LULC",
        density_col="Density",
        date_col="DateTime",
        src_crs="EPSG:32735",  # UTM 35S; shapefile declares no CRS
        aoi_admin1="eastern_cape",
        campaign_date="2023-05-19",  # observed DateTime range 2023-05-19..06-06
        doi="10.25413/sun.25050401",
        url="https://doi.org/10.25413/sun.25050401",
        license=_CCBY,
    ),
}


def _lulc_to_taxon(lulc: str) -> tuple[str | None, str]:
    """LULC class → (species_normalized, taxon_rank); ``(None, "functional")`` if unmapped."""
    return _LULC_TO_CLASS.get(lulc, (None, "functional"))


def _density_to_cover(d: object) -> float | None:
    """Density (%) → cover_pct. ``0 → None`` (ambiguous: low vs not recorded)."""
    if pd.isna(d):
        return None
    val = float(d)  # type: ignore[arg-type]
    return val if val > 0 else None


def _clean_date(v: object) -> str | None:
    """Survey date → ISO date string; drop the 1899-12-30 Excel-null sentinel."""
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts) or ts.year < 1990:
        return None
    return ts.date().isoformat()


def _build_rows(
    gdf: gpd.GeoDataFrame,
    cat: Catchment,
    run_id: str,
    ingested_at: dt.datetime,
    fallback_date: str | None,
) -> list[dict]:
    """Map a (reprojected, EPSG:4326) MapWAPS GeoDataFrame to observation rows."""
    rows: list[dict] = []
    for i, rec in enumerate(gdf.to_dict("records")):
        lulc = str(rec.get(cat.class_col) or "").strip()
        sp_norm, rank = _lulc_to_taxon(lulc)
        # obs_id from the (unique) geometry — NOT the X/Y columns, which are the
        # parent-point coords shared across GIS-harvested child points.
        geom = rec["geometry"]
        date_raw = rec.get(cat.date_col) if cat.date_col else None
        density = rec.get(cat.density_col) if cat.density_col else None
        rows.append(
            {
                "obs_id": f"{SOURCE}:{geom.x:.6f}:{geom.y:.6f}",
                "source": SOURCE,
                "source_record_id": str(i),
                "source_url": cat.url,
                "source_doi": cat.doi,
                "license": cat.license,
                "species": lulc,
                "species_normalized": sp_norm,
                "taxon_rank": rank,
                "geom_type": "point",
                "coord_uncertainty_m": COORD_UNCERTAINTY_M,
                "event_date": _clean_date(date_raw) or fallback_date,
                "basis_of_record": "MAPWAPS_FIELD",
                "cover_pct": _density_to_cover(density),
                "weight": 1.0,
                "ingested_at": ingested_at,
                "ingest_run_id": run_id,
                "aoi_admin1": cat.aoi_admin1,
            }
        )
    return rows


def ingest_mapwaps(
    catchment: str,
    root: str = PROCESSED_ROOT,
    run_id: str | None = None,
) -> str:
    """Ingest one MapWAPS catchment's training points → unified store (``source=mapwaps``).

    Every class in ``_LULC_TO_CLASS`` is crosswalked to a ``sa_landcover``
    member (IAP genus / native biome / transformed land cover); unmapped classes
    (Shade / Burnt / Bracken / Alien_Other) are dropped. Geometry used as-is,
    reprojected to 4326 (assuming ``cat.src_crs`` when the shapefile declares none).
    """
    cat = CATCHMENTS[catchment]
    run_id = run_id or make_run_id(cat.dataset)
    ingested_at = dt.datetime.now(tz=dt.UTC)

    gdf = gpd.read_file(RAW_ROOT / cat.dataset / cat.shp)
    logger.info("{}: {} rows (CRS {})", cat.dataset, len(gdf), gdf.crs)
    if gdf.crs is None:
        if not cat.src_crs:
            raise ValueError(f"{cat.dataset}: shapefile has no CRS and no src_crs configured")
        gdf = gdf.set_crs(cat.src_crs)
    gdf = gdf.to_crs("EPSG:4326")

    n0 = len(gdf)
    keep = gdf[cat.class_col].astype(str).str.strip().isin(_LULC_TO_CLASS)
    gdf = gdf[keep].reset_index(drop=True)
    logger.info("{}: kept {} of {} rows (dropped unmapped)", cat.dataset, len(gdf), n0)

    # Single field campaign → fill undated points with the modal date (or the
    # configured campaign date), so per-label year alignment (event_date.year) holds.
    fallback_date = cat.campaign_date
    if cat.date_col and cat.date_col in gdf.columns:
        valid = pd.to_datetime(gdf[cat.date_col], errors="coerce")
        valid = valid[valid.dt.year > 1990]
        if len(valid):
            fallback_date = valid.mode().iloc[0].date().isoformat()

    rows = _build_rows(gdf, cat, run_id, ingested_at, fallback_date=fallback_date)
    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=list(gdf.geometry), crs="EPSG:4326")

    path = write_partition(out, cat.dataset, root=root, run_id=run_id)
    logger.success("{}: {} rows → {}", cat.dataset, len(rows), path)
    return path
