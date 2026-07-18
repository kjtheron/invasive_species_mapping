"""MapWAPS Olifants-Doring ingest — field training points → unified store.

~28k IAP + land-cover field points (WGS84 / UTM 34S) from the MapWAPS project
(Rebelo et al. 2025, DOI 10.25413/sun.29958053).

Decisions baked in (see the Training-dataset metadata PDF):

- **Geometry used as-is.** The published points are *already* distance/direction
  corrected ("The points were then corrected for distance and direction using
  the metadata"); many are GIS-harvested child points that inherit the parent's
  ``Distance``/``Direction``. Re-applying an offset would double-correct, so we
  only reproject UTM 34S → EPSG:4326.
- **cover_pct ← ``Density___``** (estimated density %); ``0 → None`` (ambiguous:
  truly sparse vs not recorded).
- **taxon_rank = genus** for the Alien_* IAP classes (Pine/Gum/Wattle/Prosopis) —
  the survey did not resolve them to species. Species-level resolution is left to
  a hierarchical-loss head later; for now ``make-split`` uses a genus class map.
- **All mappable classes ingested.** ``_LULC_TO_CLASS`` crosswalks each of the 23
  MapWAPS classes to a ``western_cape_landcover`` member (IAP genus, VegMap biome,
  or transformed land cover) — the same strings SANLC emits, so make-split resolves
  them. Only Shade / Burnt / Alien_Other are dropped (shadow / transient / unspecific).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger

from cmrv.labels.observations import PROCESSED_ROOT, make_run_id, write_partition

DATASET = "mapwaps_olifants_doring"
SOURCE = "mapwaps"
SHP_PATH = Path(
    "data/labels/raw/mapwaps_olifants_doring/"
    "OlifantsDoring_TrainingData_23Classes/OlifantsDoring_trainingdata.shp"
)
DOI = "10.25413/sun.29958053"
SOURCE_URL = "https://doi.org/10.25413/sun.29958053"
LICENSE = "CC-BY-4.0 / CC-BY-SA (ambiguous); co-authorship offer expected for academic use"

# Residual positional uncertainty (m). Points are already distance/direction
# corrected and mixed-pixel points removed; this is the leftover field-GPS +
# ~10 m S2-pixel residual. ponytail: constant, refine if a per-point figure surfaces.
COORD_UNCERTAINTY_M = 15.0

# MapWAPS LULC_Class → (species_normalized, taxon_rank) for the
# ``western_cape_landcover`` map. Alien_* → IAP genus; native veg → VegMap-biome
# member; transformed → land-cover member — the SAME strings SANLC emits, so
# make-split's class_map resolves them (genus fallback for Alien_*, binomial match
# for the rest). Classes NOT listed are dropped at ingest: "Shade" (terrain/cloud
# shadow), "Burnt" (transient post-fire scar), "Alien_Other" (unspecific alien).
_LULC_TO_CLASS: dict[str, tuple[str, str]] = {
    # IAP genera (survey didn't resolve to species)
    "Alien_Pine": ("Pinus", "genus"),
    "Alien_Gum": ("Eucalyptus", "genus"),
    "Alien_Wattle": ("Acacia", "genus"),
    "Alien_Prosopis": ("Prosopis", "genus"),
    # native biomes (→ VegMap T_BIOME member strings)
    "Fynbos-High density": ("fynbos", "biome"),
    "Fynbos - Low density": ("fynbos", "biome"),
    "Renosterveld": ("fynbos", "biome"),  # Renosterveld ⊂ Fynbos biome
    "Succulent Karoo": ("succulent_karoo", "biome"),
    "Bushmanland Shrubland": ("nama_karoo", "biome"),  # Bushmanland bioregion ⊂ Nama-Karoo
    "Riparian Bush": ("azonal", "biome"),  # riparian = azonal (intrazonal) vegetation
    "Riparian Trees": ("azonal", "biome"),
    # transformed / land cover
    "Irrigated Agriculture": ("cultivated", "landcover"),
    "Dryland Agriculture": ("cultivated", "landcover"),
    "Urban": ("built_up", "landcover"),
    "Bare Ground": ("bare", "landcover"),
    "Rock": ("bare", "landcover"),
    "Water": ("water", "landcover"),
    "Wetland - Reed": ("wetland", "landcover"),
    "Wetland_Other": ("wetland", "landcover"),
    "Wetland - Palmiet": ("wetland", "landcover"),
}


def _lulc_to_taxon(lulc: str) -> tuple[str | None, str]:
    """LULC_Class → (species_normalized, taxon_rank); ``(None, "functional")`` if unmapped."""
    return _LULC_TO_CLASS.get(lulc, (None, "functional"))


def _density_to_cover(d: object) -> float | None:
    """``Density___`` (%) → cover_pct. ``0 → None`` (ambiguous: low vs not recorded)."""
    if pd.isna(d):
        return None
    val = float(d)  # type: ignore[arg-type]
    return val if val > 0 else None


def _clean_date(v: object) -> str | None:
    """``DateTime`` → ISO date string; drop the 1899-12-30 Excel-null sentinel."""
    ts = pd.to_datetime(v, errors="coerce")
    if pd.isna(ts) or ts.year < 1990:
        return None
    return ts.date().isoformat()


def _build_rows(
    gdf: gpd.GeoDataFrame,
    run_id: str,
    ingested_at: dt.datetime,
    fallback_date: str | None = None,
) -> list[dict]:
    """Map a (reprojected, EPSG:4326) MapWAPS GeoDataFrame to observation rows.

    ``fallback_date`` fills undated points — the survey is a single campaign, so
    the year is well-defined, and chip extraction keys imagery on ``event_date.year``.
    """
    rows: list[dict] = []
    for i, rec in enumerate(gdf.to_dict("records")):
        lulc = str(rec.get("LULC_Class") or "").strip()
        sp_norm, rank = _lulc_to_taxon(lulc)
        # obs_id from the (unique) geometry — NOT the X/Y columns, which are the
        # parent-point coords shared across GIS-harvested child points.
        geom = rec["geometry"]
        obs_id = f"{SOURCE}:{geom.x:.6f}:{geom.y:.6f}"
        rows.append(
            {
                "obs_id": obs_id,
                "source": SOURCE,
                "source_record_id": str(rec.get("Id", i)),
                "source_url": SOURCE_URL,
                "source_doi": DOI,
                "license": LICENSE,
                "species": lulc,
                "species_normalized": sp_norm,
                "taxon_rank": rank,
                "geom_type": "point",
                "coord_uncertainty_m": COORD_UNCERTAINTY_M,
                "event_date": _clean_date(rec.get("DateTime")) or fallback_date,
                "basis_of_record": "MAPWAPS_FIELD",
                "cover_pct": _density_to_cover(rec.get("Density___")),
                "weight": 1.0,
                "ingested_at": ingested_at,
                "ingest_run_id": run_id,
                "aoi_admin1": "western_cape",
            }
        )
    return rows


def ingest_mapwaps(
    shp_path: str | Path = SHP_PATH,
    root: str = PROCESSED_ROOT,
    run_id: str | None = None,
) -> str:
    """Ingest MapWAPS Olifants-Doring training points → unified store (``source=mapwaps``).

    Every class in ``_LULC_TO_CLASS`` is kept and crosswalked to a
    ``western_cape_landcover`` member — IAP genera (Alien_*), native biomes, and
    transformed land cover. Rows outside it (Shade / Burnt / Alien_Other) are dropped
    at ingest. Geometry used as-is (already corrected), reprojected to 4326.
    """
    run_id = run_id or make_run_id(SOURCE)
    ingested_at = dt.datetime.now(tz=dt.UTC)

    gdf = gpd.read_file(shp_path)
    logger.info("MapWAPS rows: {} (CRS {})", len(gdf), gdf.crs)
    gdf = gdf.to_crs("EPSG:4326")

    n0 = len(gdf)
    gdf = gdf[gdf["LULC_Class"].astype(str).isin(_LULC_TO_CLASS)].reset_index(drop=True)
    logger.info("kept {} of {} rows (dropped Shade/Burnt/Alien_Other)", len(gdf), n0)

    # Single field campaign → fill undated points with the modal observation date
    # so per-label year alignment (event_date.year) holds for all of them.
    valid = pd.to_datetime(gdf.get("DateTime"), errors="coerce")
    valid = valid[valid.dt.year > 1990]
    fallback_date = valid.mode().iloc[0].date().isoformat() if len(valid) else None

    rows = _build_rows(gdf, run_id, ingested_at, fallback_date=fallback_date)
    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=list(gdf.geometry), crs="EPSG:4326")

    path = write_partition(out, DATASET, root=root, run_id=run_id)
    logger.success("mapwaps: {} rows → {}", len(rows), path)
    return path
