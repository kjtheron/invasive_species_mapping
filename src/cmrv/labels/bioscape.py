"""BioSCape VegPlots ingest — Berg+Eerste 2022-2023 field plots.

Two emitters (distinct ``source`` values in the unified store):

``ingest_lineintercept``
    One row per (SiteCode_Plot, LineTransect, MetersAlongLine, species_normalized).
    Geometry: plot-center GPS + transect offset via ``pyproj.Geod.fwd``.
    coord_uncertainty_m ≈ 5.1 m (plot-center QA 5 m ⊕ 1 m transect position).

``ingest_plotcoverage``
    One row per (SiteCode_Plot_Quadrant, species_normalized).
    Geometry: quadrant centroid = plot center + 2.5 m along quadrant bisector.
    coord_uncertainty_m ≈ 5.6 m (plot-center ⊕ quadrant half-radius 2.5 m).
    weight = 0.95 when IAP cover ≥ 40 %; 0.5 when present but minor.

Both loaders:
- Emit the unified ``observations.SCHEMA`` via ``write_partition``.
- Do NOT assign class_id (training config does that via labels_schema.yaml).
- Decide IAP membership from the active class-map ``members[]`` (single source
  of truth in ``configs/labels_schema.yaml``) — no separate NEMBA taxa file.
- Filter to WC (both berg and eerste SiteCodes are within Western Cape —
  the old ``SiteCode == 'berg'`` filter is dropped).

Coord uncertainty: ``sqrt(5² + 1²) ≈ 5.1 m`` for line intercept,
``sqrt(5² + 2.5²) ≈ 5.6 m`` for plot coverage (quadrant centroid offset).
"""

from __future__ import annotations

import datetime as dt
import math
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd
from loguru import logger
from pyproj import Geod
from shapely.geometry import Point

from cmrv.labels.classmap import ClassMap, build_lookup
from cmrv.labels.observations import PROCESSED_ROOT, make_run_id, write_partition

DEFAULT_SCHEMA_PATH = "configs/labels_schema.yaml"
DEFAULT_CLASS_MAP = "western_cape_iap"
DATASET = "BioSCape_VegPlots_Berg_Eerste_2425"
DOI = "10.3334/ORNLDAAC/2425"
SOURCE_URL = "https://doi.org/10.3334/ORNLDAAC/2425"
LICENSE = "NASA ORNL DAAC — free (Earthdata login); cite per DOI"

# ---------------------------------------------------------------------------
# Data file paths (ORNL DAAC archive layout)
# ---------------------------------------------------------------------------

BIOSCAPE_DATA_DIR = Path("data/labels/raw/BioSCape_VegPlots_Berg_Eerste_2425/data")
SITE_CSV = BIOSCAPE_DATA_DIR / "Berg_Eerste_Veg_SiteData.csv"
LINE_CSV = BIOSCAPE_DATA_DIR / "Berg_Eerste_Veg_LineIntercept.csv"
PLOT_CSV = BIOSCAPE_DATA_DIR / "Berg_Eerste_Veg_PlotCoverage.csv"

# Coord uncertainty constants (metres)
LINE_COORD_UNCERTAINTY_M = math.sqrt(5.0**2 + 1.0**2)  # ≈ 5.10 m
PLOT_COORD_UNCERTAINTY_M = math.sqrt(5.0**2 + 2.5**2)  # ≈ 5.59 m

# IAP cover threshold for weight assignment in PlotCoverage
DOMINANT_COVER_PCT = 40.0

_GEOD = Geod(ellps="WGS84")

# Quadrant bisector azimuths (degrees from north, clockwise)
_QUADRANT_AZ = {"NE": 45.0, "NW": 315.0, "SE": 135.0, "SW": 225.0}

# Transect azimuths (north = 0°, east = 90°)
_TRANSECT_AZ = {"NS": 0.0, "WE": 90.0}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_name(name: str | None) -> str | None:
    """Strip hybrid markers, trailing spaces; return None for null/unident entries."""
    if not name or not isinstance(name, str):
        return None
    s = re.sub(r"\s*[×x]\s*", " ", name).strip()
    if s in ("NA", "") or s.startswith("(") or s.startswith("[") or " " not in s:
        return None
    return s


def _mark_to_offset_m(meters_along: str) -> float:
    """Convert MetersAlongLine label to signed offset from plot center (metres).

    The rope runs from –5 m to +5 m relative to centre.
    '0' = one end (–5 m offset), '10' = other end (+5 m offset).
    'C' = centre (0 m offset).
    """
    tag = meters_along.rstrip("LR").strip()
    if tag.upper() == "C":
        return 0.0
    return float(tag) - 5.0


def _offset_point(lon: float, lat: float, az_deg: float, dist_m: float) -> tuple[float, float]:
    """Return (lon2, lat2) after moving dist_m along az_deg from (lon, lat)."""
    if dist_m < 0:
        az_deg = (az_deg + 180.0) % 360.0
        dist_m = -dist_m
    if dist_m == 0.0:
        return lon, lat
    lon2, lat2, _ = _GEOD.fwd(lon, lat, az_deg, dist_m)
    return float(lon2), float(lat2)


def _is_iap(species: str | None, classmap: ClassMap) -> bool:
    """True if *species* resolves to an IAP class in the active class-map.

    Membership comes from ``class_maps.<name>.members[]`` (single source of
    truth) — exact binomial first, then per-class genus fallback.
    """
    normalized = _normalize_name(species)
    if not normalized:
        return False
    return classmap.resolve(normalized) is not None


def _load_site_data(site_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(site_csv)
    df["GPS_PlotCenter_Latitude"] = pd.to_numeric(df["GPS_PlotCenter_Latitude"], errors="coerce")
    df["GPS_PlotCenter_Longitude"] = pd.to_numeric(df["GPS_PlotCenter_Longitude"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# LineIntercept ingest
# ---------------------------------------------------------------------------


def ingest_lineintercept(
    line_csv: str | Path = LINE_CSV,
    site_csv: str | Path = SITE_CSV,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    class_map_name: str = DEFAULT_CLASS_MAP,
    root: str = PROCESSED_ROOT,
    run_id: str | None = None,
    iap_only: bool = True,
) -> str:
    """Ingest BioSCape LineIntercept → unified observation store.

    Parameters
    ----------
    schema_path / class_map_name:
        IAP membership is decided from ``class_maps.<class_map_name>.members[]``.
    iap_only:
        If True (default) emit only IAP species rows. Pass False to also
        include non-IAP rows (e.g., for negative-sample construction).
    """
    source = "bioscape_line"
    run_id = run_id or make_run_id(source)
    ingested_at = dt.datetime.now(tz=dt.UTC)

    classmap = build_lookup(schema_path, class_map_name)

    sd = _load_site_data(Path(site_csv))
    li = pd.read_csv(Path(line_csv))
    logger.info("LineIntercept rows: {}", len(li))

    # Join GPS from SiteData via SiteCode_Plot
    site_gps = sd[
        ["SiteCode_Plot", "GPS_PlotCenter_Latitude", "GPS_PlotCenter_Longitude", "Date"]
    ].drop_duplicates(subset=["SiteCode_Plot"])
    li = li.merge(site_gps, on="SiteCode_Plot", how="inner")

    rows: list[dict] = []
    for rec in li.to_dict("records"):
        sp_raw = rec.get("AcceptedSpecies") or ""
        sp_norm = _normalize_name(sp_raw)
        is_iap = _is_iap(sp_raw, classmap)

        if iap_only and not is_iap:
            continue

        transect = rec["LineTransect"]
        meters_along = str(rec["MetersAlongLine"])
        az = _TRANSECT_AZ.get(transect, 0.0)
        offset = _mark_to_offset_m(meters_along)
        lon, lat = _offset_point(
            rec["GPS_PlotCenter_Longitude"],
            rec["GPS_PlotCenter_Latitude"],
            az,
            offset,
        )

        obs_id = (
            f"bioscape_line:{rec['SiteCode_Plot']}:{transect}:{meters_along}"
            f":{sp_norm or 'GROUND_COVER'}"
        )
        rows.append(
            {
                "obs_id": obs_id,
                "source": source,
                "source_record_id": obs_id,
                "source_url": SOURCE_URL,
                "source_doi": DOI,
                "license": LICENSE,
                "species": sp_raw,
                "species_normalized": sp_norm,
                "taxon_rank": "species",
                "geom_type": "point",
                "coord_uncertainty_m": LINE_COORD_UNCERTAINTY_M,
                "event_date": rec.get("Date"),
                "basis_of_record": "BIOSCAPE_LINE" if sp_norm else "GROUND_COVER",
                "cover_pct": None,
                "weight": 0.95 if is_iap else 0.0,
                "ingested_at": ingested_at,
                "ingest_run_id": run_id,
                "aoi_admin1": "western_cape",
                "lon": lon,
                "lat": lat,
            }
        )

    if not rows:
        logger.warning("no rows to ingest for source={}", source)
        return ""

    df_rows = pd.DataFrame(rows)
    geom_col = [Point(lo, la) for lo, la in zip(df_rows["lon"], df_rows["lat"], strict=True)]
    gdf = gpd.GeoDataFrame(df_rows.drop(columns=["lon", "lat"]), geometry=geom_col, crs="EPSG:4326")

    out_path = write_partition(gdf, DATASET, root=root, run_id=run_id)
    logger.success("bioscape_line: {} rows → {}", len(rows), out_path)
    return out_path


# ---------------------------------------------------------------------------
# PlotCoverage ingest
# ---------------------------------------------------------------------------


def ingest_plotcoverage(
    plot_csv: str | Path = PLOT_CSV,
    site_csv: str | Path = SITE_CSV,
    schema_path: str | Path = DEFAULT_SCHEMA_PATH,
    class_map_name: str = DEFAULT_CLASS_MAP,
    root: str = PROCESSED_ROOT,
    run_id: str | None = None,
    iap_only: bool = True,
) -> str:
    """Ingest BioSCape PlotCoverage → unified observation store.

    Geometry is the quadrant centroid: plot center + 2.5 m along the quadrant
    bisector azimuth (NE→45°, NW→315°, SE→135°, SW→225°).
    Weight = 0.95 when sum(IAP PercentCoverAlive) ≥ 40; 0.5 otherwise.
    IAP membership comes from ``class_maps.<class_map_name>.members[]``.
    """
    source = "bioscape_plot"
    run_id = run_id or make_run_id(source)
    ingested_at = dt.datetime.now(tz=dt.UTC)

    classmap = build_lookup(schema_path, class_map_name)

    sd = _load_site_data(Path(site_csv))
    pc = pd.read_csv(Path(plot_csv))
    pc["PercentCoverAlive"] = pd.to_numeric(pc["PercentCoverAlive"], errors="coerce")
    pc["PercentCoverDead"] = pd.to_numeric(pc.get("PercentCoverDead"), errors="coerce")
    logger.info("PlotCoverage rows: {}", len(pc))

    # Join GPS + quadrant info via SiteCode_Plot_Quadrant
    site_cols = sd[
        [
            "SiteCode_Plot_Quadrant",
            "GPS_PlotCenter_Latitude",
            "GPS_PlotCenter_Longitude",
            "Quadrant",
            "Date",
        ]
    ].drop_duplicates(subset=["SiteCode_Plot_Quadrant"])
    pc = pc.merge(site_cols, on="SiteCode_Plot_Quadrant", how="inner")

    # Compute per-quadrant IAP cover total for weight assignment
    pc["is_iap"] = pc["AcceptedSpecies"].apply(lambda sp: _is_iap(sp, classmap))
    quad_iap_cover = (
        pc[pc["is_iap"]]
        .groupby("SiteCode_Plot_Quadrant")["PercentCoverAlive"]
        .sum()
        .reset_index()
        .rename(columns={"PercentCoverAlive": "total_iap_cover"})
    )
    pc = pc.merge(quad_iap_cover, on="SiteCode_Plot_Quadrant", how="left")

    rows: list[dict] = []
    for rec in pc.to_dict("records"):
        sp_raw = rec.get("AcceptedSpecies") or ""
        sp_norm = _normalize_name(sp_raw)

        if iap_only and not _is_iap(sp_raw, classmap):
            continue

        quadrant = rec.get("Quadrant") or "NE"
        az = _QUADRANT_AZ.get(quadrant, 45.0)
        lon, lat = _offset_point(
            rec["GPS_PlotCenter_Longitude"],
            rec["GPS_PlotCenter_Latitude"],
            az,
            2.5,  # quadrant half-radius
        )

        # Pandas left-join non-matches return NaN (not None); use pd.isna() to check
        raw_iap = rec.get("total_iap_cover")
        total_iap = 0.0 if pd.isna(raw_iap) else float(raw_iap)
        weight = 0.95 if total_iap >= DOMINANT_COVER_PCT else 0.5

        obs_id = f"bioscape_plot:{rec['SiteCode_Plot_Quadrant']}:{sp_norm or 'GROUND_COVER'}"
        rows.append(
            {
                "obs_id": obs_id,
                "source": source,
                "source_record_id": obs_id,
                "source_url": SOURCE_URL,
                "source_doi": DOI,
                "license": LICENSE,
                "species": sp_raw,
                "species_normalized": sp_norm,
                "taxon_rank": "species",
                "geom_type": "point",
                "coord_uncertainty_m": PLOT_COORD_UNCERTAINTY_M,
                "event_date": rec.get("Date"),
                "basis_of_record": "BIOSCAPE_PLOT",
                "cover_pct": float(rec.get("PercentCoverAlive") or 0.0),
                "weight": float(weight),
                "ingested_at": ingested_at,
                "ingest_run_id": run_id,
                "aoi_admin1": "western_cape",
                "lon": lon,
                "lat": lat,
            }
        )

    if not rows:
        logger.warning("no rows to ingest for source={}", source)
        return ""

    df_rows = pd.DataFrame(rows)
    geom_col = [Point(lo, la) for lo, la in zip(df_rows["lon"], df_rows["lat"], strict=True)]
    gdf = gpd.GeoDataFrame(df_rows.drop(columns=["lon", "lat"]), geometry=geom_col, crs="EPSG:4326")

    out_path = write_partition(gdf, DATASET, root=root, run_id=run_id)
    logger.success("bioscape_plot: {} rows → {}", len(rows), out_path)
    return out_path
