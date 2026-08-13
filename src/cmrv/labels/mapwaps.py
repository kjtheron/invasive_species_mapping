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
- **Only mappable classes ingested.** Shade (shadow artefact) and Alien_Other
  (unspecified alien, no genus to assign) resolve to no class and are dropped.

Catchments are hydrological, so they cross provincial borders: ``aoi_admin1`` is
derived **per point** (:func:`cmrv.aoi.province_of`), not from the catchment. The
declared province is only a fallback for points outside every ADM1 polygon.

Catchments processed: Olifants-Doring (WC + NC), Tugela (KZN), uMzimvubu (EC + KZN). The
Luvuvhu and Sabie-Crocodile figshare articles ship broken TrainingData (a duplicate
of Tugela / an empty folder respectively) and are omitted — see download/README.md.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd  # type: ignore
import pandas as pd  # type: ignore
from loguru import logger  # type: ignore

from cmrv.aoi import province_of
from cmrv.labels.classmap import warn_unmapped
from cmrv.labels.observations import PROCESSED_ROOT, make_run_id, write_partition

SOURCE = "mapwaps"
RAW_ROOT = Path("data/labels/raw")

# Residual positional uncertainty (m): field-GPS + ~10 m S2-pixel residual after the
# published distance/direction correction. ponytail: constant, refine if a figure surfaces.
COORD_UNCERTAINTY_M = 15.0

# MapWAPS class string → (species_normalized, taxon_rank) for the
# ``sa_landcover`` map. Union across all catchments. Alien_* → IAP genus;
# native veg → VegMap-biome member; transformed → land-cover member. Classes NOT here
# are dropped at ingest: "Shade" (shadow artefact), plus "Alien_Other" and "Other
# Invasive Alien Plants" (unspecific aliens — no genus to assign, so no class).
_LULC_TO_CLASS: dict[str, tuple[str, str]] = {
    # --- alien invasive trees → IAP genus (survey didn't resolve to species) ---
    "Alien_Pine": ("Pinus", "genus"),
    "Alien_Gum": ("Eucalyptus", "genus"),
    "Alien_Wattle": ("Acacia", "genus"),
    "Alien_Black Wattle": ("Acacia", "genus"),  # A. mearnsii (uMzimvubu)
    "Alien_Silver Wattle": ("Acacia", "genus"),  # A. dealbata (uMzimvubu)
    "Alien_Prosopis": ("Prosopis", "genus"),
    "Alien_Poplar": ("Populus", "genus"),  # Populus × canescens
    # The 2026-08-13 re-release is inconsistent across catchments: Tugela and Luvuvhu
    # use bare names where uMzimvubu and Sabie-Croc keep the Alien_ prefix. Same taxa.
    "Wattle": ("Acacia", "genus"),
    "Gum": ("Eucalyptus", "genus"),
    "Pine": ("Pinus", "genus"),
    "Poplar": ("Populus", "genus"),
    "Lantana": ("Lantana camara", "species"),
    "Alien_Lantana": ("Lantana camara", "species"),
    # "Bugweed" is the South African common name for Solanum mauritianum.
    "Bugweed": ("Solanum mauritianum", "species"),
    "Alien_Bugweed": ("Solanum mauritianum", "species"),
    # Tecoma stans — yellow bells, a NEMBA 1b woody shrub / small tree. New taxon.
    "Alien_Yellow Bells": ("Tecoma stans", "species"),
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
    "Indigenous Bush": ("savanna", "biome"),  # unqualified — same class as _Other
    "Indigenous Bush_Mopane": ("savanna", "biome"),  # mopane veld ⊂ savanna
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
    # Permanent crops from the subtropical Luvuvhu / Sabie-Crocodile catchments. All
    # are cultivated land; the map has no orchard class and does not need one.
    "Orchards": ("cultivated", "landcover"),
    "Orchards_Banana": ("cultivated", "landcover"),
    "Orchards_Nuts": ("cultivated", "landcover"),
    "Orchards_Other": ("cultivated", "landcover"),
    "Bananas": ("cultivated", "landcover"),
    "Macadamias": ("cultivated", "landcover"),
    "Tea": ("cultivated", "landcover"),
    # --- other cover states ---
    # Bracken is Pteridium aquilinum — a taxon, but deliberately NOT rank species/genus:
    # that rank marks IAP observations, and `sanlc.py` buffers around them to exclude
    # land-cover points. Bracken is indigenous, so it must not trigger that exclusion.
    "Bracken": ("bracken", "landcover"),
    "Burnt": ("burnt", "landcover"),
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
    aoi_admin1: str  # declared province — FALLBACK only; province_of decides per point
    campaign_date: str  # ISO fallback date for undated points (single field campaign)
    doi: str
    url: str
    license: str


_LICENSE = "CC-BY-4.0 (SUNScholar/figshare); cite the DOI"


def _cat(dataset, shp, class_col, date_col, admin1, campaign, doi_id):
    """One 2026-re-release catchment — they share every field except these six.

    All four declare their own UTM CRS, name the density column ``Density``, and are
    CC-BY-4.0, so only the varying parts are spelled out at the call site.
    """
    return Catchment(
        dataset=dataset,
        shp=shp,
        class_col=class_col,
        density_col="Density",
        date_col=date_col,
        src_crs=None,  # every 2026 shapefile declares a CRS
        aoi_admin1=admin1,
        campaign_date=campaign,
        doi=f"10.25413/sun.{doi_id}",
        url=f"https://doi.org/10.25413/sun.{doi_id}",
        license=_LICENSE,
    )


# The 2026-08-13 re-release re-cut all four non-Olifants catchments: new filenames,
# renamed columns, declared CRSs, and Luvuvhu / Sabie-Crocodile finally carrying their
# own data. ``aoi_admin1`` here is only the fallback — ingest derives it per point.
CATCHMENTS: dict[str, Catchment] = {
    "mapwaps_olifants_doring": Catchment(
        dataset="mapwaps_olifants_doring",
        shp="OlifantsDoring_TrainingData_23Classes/OlifantsDoring_trainingdata.shp",
        class_col="LULC_Class",
        density_col="Density___",
        date_col="DateTime",
        src_crs=None,
        aoi_admin1="western_cape",  # ~17 % of it is actually Northern Cape
        campaign_date="2025-05-19",  # DateTime 2025-05-17..22 — a sync batch, not 6 field days
        doi="10.25413/sun.29958053",
        url="https://doi.org/10.25413/sun.29958053",
        license=_LICENSE,
    ),
    "mapwaps_tugela": _cat(
        "mapwaps_tugela",
        "Tugela_TrainingData/Tugela_TrainingData.shp",
        "LULC_Class",
        "Date_Capt",
        "kwazulu_natal",
        "2023-10-26",  # observed 2023-10-12..26
        "25066151",
    ),
    "mapwaps_umzimvubu": _cat(
        "mapwaps_umzimvubu",
        "uMzim_Train.shp",
        "LULC",
        "DateTime",
        "eastern_cape",  # ~17 % is KwaZulu-Natal
        "2023-06-06",  # observed 2023-05-19..06-06
        "25050401",
    ),
    "mapwaps_luvuvhu": _cat(
        "mapwaps_luvuvhu",
        "Luvuvhu_TrainingData/Luvuvhu_TrainingData.shp",
        "LULC",
        "DateTime",
        "limpopo",
        "2023-07-27",  # observed 2023-07-18..30
        "25050314",
    ),
    "mapwaps_sabie_crocodile": _cat(
        "mapwaps_sabie_crocodile",
        "SabieCroc_Train.shp",
        "LULC_Class",
        None,  # shapefile carries no survey date
        "mpumalanga",
        "2023-07-01",  # ponytail: undated; mid-2023 to match the neighbouring catchments
        "25050368",
    ),
}


def _lulc_to_taxon(lulc: str) -> tuple[str | None, str]:
    """LULC class → (species_normalized, taxon_rank); ``(None, "functional")`` if unmapped."""
    return _LULC_TO_CLASS.get(lulc, (None, "functional"))


def _density_to_cover(d: object) -> float | None:
    """Density (%) → cover_pct. ``0 → None`` (ambiguous: low vs not recorded)."""
    if pd.isna(d): # type: ignore
        return None
    val = float(d)  # type: ignore[arg-type]
    return val if val > 0 else None


def _clean_date(v: object) -> str | None:
    """Survey date → ISO date string; drop the 1899-12-30 Excel-null sentinel."""
    ts = pd.to_datetime(v, errors="coerce") # type: ignore
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
        # obs_id from the geometry — never the X/Y columns. Geometry is the
        # distance/direction-corrected target position and is unique per row. X/Y is
        # the raw observer reading, and it is inconsistent across catchments: WGS84
        # degrees in Olifants-Doring, UTM 35S metres in uMzimvubu, where 670 of 5479
        # rows also share a value. Neither CRS-portable nor unique, so it can't key an id.
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
                "aoi_admin1": rec.get("_prov") or cat.aoi_admin1,
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
    (Shade / Alien_Other) are dropped. Geometry used as-is,
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
    classes = gdf[cat.class_col].astype(str).str.strip()
    keep = classes.isin(_LULC_TO_CLASS)
    if not keep.all():
        # Name the classes, don't just count them: a new catchment's unknown
        # Alien_* string is dropped here, and a bare count can't tell you whether
        # you lost a survey artefact or a whole genus. Add real ones to
        # _LULC_TO_CLASS (and the genus to labels_schema members[]) before ingest.
        dropped = classes[~keep].value_counts()
        logger.warning(
            "{}: dropping {} rows in {} unmapped class(es): {}",
            cat.dataset,
            int((~keep).sum()),
            len(dropped),
            dropped.to_dict(),
        )
    gdf = gdf[keep].reset_index(drop=True)
    logger.info("{}: kept {} of {} rows", cat.dataset, len(gdf), n0)

    # Single field campaign → fill undated points with the modal date (or the
    # configured campaign date), so per-label year alignment (event_date.year) holds.
    fallback_date = cat.campaign_date
    if cat.date_col and cat.date_col in gdf.columns:
        valid = pd.to_datetime(gdf[cat.date_col], errors="coerce")
        valid = valid[valid.dt.year > 1990]
        if len(valid):
            fallback_date = valid.mode().iloc[0].date().isoformat()

    # Province per point, not per catchment. Catchments are hydrological and cross
    # provincial borders: Olifants-Doring is 17% Northern Cape (the metadata PDF says
    # so outright), uMzimvubu is 16% KwaZulu-Natal. The declared `aoi_admin1` is only
    # the fallback for points outside every ADM1 polygon (border jitter).
    gdf["_prov"] = province_of(gdf).fillna(cat.aoi_admin1)
    other = gdf.loc[gdf["_prov"] != cat.aoi_admin1, "_prov"].value_counts()
    if not other.empty:
        logger.info("{}: declared {} but {}", cat.dataset, cat.aoi_admin1, other.to_dict())

    rows = _build_rows(gdf, cat, run_id, ingested_at, fallback_date=fallback_date)
    warn_unmapped({r["species_normalized"] for r in rows}, source=cat.dataset)
    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=list(gdf.geometry), crs="EPSG:4326")

    path = write_partition(out, cat.dataset, root=root, run_id=run_id)
    logger.success("{}: {} rows → {}", cat.dataset, len(rows), path)
    return path
