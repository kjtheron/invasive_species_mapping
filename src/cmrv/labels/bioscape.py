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
- Emit the unified ``observations.SCHEMA`` via ``write_source_partition``.
- Do NOT assign class_id (training config does that via labels_schema.yaml).
- Cross-reference species against ``nemba_taxa_resolved.parquet`` for
  nemba_category and gbif_usage_key.
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

from cmrv.io import read_parquet_df
from cmrv.labels.observations import WC_LABELS_ROOT, make_run_id, write_source_partition

# ---------------------------------------------------------------------------
# Data file paths (ORNL DAAC archive layout)
# ---------------------------------------------------------------------------

BIOSCAPE_DATA_DIR = Path("data/labels/BioSCape_VegPlots_Berg_Eerste_2425/data")
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


def _load_nemba_lookup(
    nemba_resolved: str | Path,
) -> dict[str, tuple[int | None, str | None]]:
    """Return {binomial_lower: (gbif_usage_key, nemba_category)} from resolved parquet."""
    df = read_parquet_df(str(nemba_resolved))
    out: dict[str, tuple[int | None, str | None]] = {}
    for rec in df.to_dict("records"):
        cat = rec.get("nemba_category")
        key = rec.get("gbif_usage_key")
        qn = str(rec.get("query_name") or "").lower().strip()
        if qn:
            out[qn] = (key, cat)
        canon = str(rec.get("canonical_name") or "")
        parts = canon.split()
        if len(parts) >= 2:
            binomial = f"{parts[0]} {parts[1]}".lower()
            out.setdefault(binomial, (key, cat))
    return out


def _lookup_species(
    species: str | None,
    nemba_map: dict[str, tuple[int | None, str | None]],
) -> tuple[int | None, str | None]:
    """Resolve AcceptedSpecies → (gbif_usage_key, nemba_category) or (None, None)."""
    if not species:
        return None, None
    normalized = _normalize_name(species)
    if not normalized:
        return None, None
    tokens = normalized.lower().split()
    for n in (2, 1):
        key = " ".join(tokens[:n])
        if key in nemba_map:
            return nemba_map[key]
    return None, None


def _load_site_data(site_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(site_csv)
    df["GPS_PlotCenter_Latitude"] = pd.to_numeric(df["GPS_PlotCenter_Latitude"], errors="coerce")
    df["GPS_PlotCenter_Longitude"] = pd.to_numeric(df["GPS_PlotCenter_Longitude"], errors="coerce")
    return df


# ---------------------------------------------------------------------------
# LineIntercept ingest
# ---------------------------------------------------------------------------


def ingest_lineintercept(
    nemba_resolved: str | Path,
    line_csv: str | Path = LINE_CSV,
    site_csv: str | Path = SITE_CSV,
    root: str = WC_LABELS_ROOT,
    run_id: str | None = None,
    iap_only: bool = True,
) -> str:
    """Ingest BioSCape LineIntercept → unified observation store.

    Parameters
    ----------
    nemba_resolved:
        Path to ``nemba_taxa_resolved.parquet`` (output of labels-nemba-resolve).
    iap_only:
        If True (default) emit only NEMBA-listed species rows. Pass False to
        also include non-IAP rows (e.g., for negative-sample construction).
    """
    source = "bioscape_line"
    run_id = run_id or make_run_id(source)
    ingested_at = dt.datetime.now(tz=dt.UTC)

    nemba_map = _load_nemba_lookup(nemba_resolved)
    logger.info("loaded {} NEMBA entries for BioSCape line lookup", len(nemba_map))

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
        gbif_key, nemba_cat = _lookup_species(sp_raw, nemba_map)

        if iap_only and nemba_cat is None:
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
                "source_url": None,
                "species": sp_raw,
                "species_normalized": sp_norm,
                "gbif_usage_key": gbif_key,
                "nemba_category": nemba_cat,
                "geom_type": "point",
                "coord_uncertainty_m": LINE_COORD_UNCERTAINTY_M,
                "event_date": rec.get("Date"),
                "basis_of_record": "BIOSCAPE_LINE" if sp_norm else "GROUND_COVER",
                "cover_pct": None,
                "weight": 0.95 if nemba_cat else 0.0,
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

    out_path = write_source_partition(gdf, source, root=root, run_id=run_id)
    logger.success("bioscape_line: {} rows → {}", len(rows), out_path)
    return out_path


# ---------------------------------------------------------------------------
# PlotCoverage ingest
# ---------------------------------------------------------------------------


def ingest_plotcoverage(
    nemba_resolved: str | Path,
    plot_csv: str | Path = PLOT_CSV,
    site_csv: str | Path = SITE_CSV,
    root: str = WC_LABELS_ROOT,
    run_id: str | None = None,
    iap_only: bool = True,
) -> str:
    """Ingest BioSCape PlotCoverage → unified observation store.

    Geometry is the quadrant centroid: plot center + 2.5 m along the quadrant
    bisector azimuth (NE→45°, NW→315°, SE→135°, SW→225°).
    Weight = 0.95 when sum(NEMBA PercentCoverAlive) ≥ 40; 0.5 otherwise.
    """
    source = "bioscape_plot"
    run_id = run_id or make_run_id(source)
    ingested_at = dt.datetime.now(tz=dt.UTC)

    nemba_map = _load_nemba_lookup(nemba_resolved)
    logger.info("loaded {} NEMBA entries for BioSCape plot lookup", len(nemba_map))

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
    def is_nemba(sp: str | None) -> bool:
        norm = _normalize_name(sp)
        if not norm:
            return False
        tokens = norm.lower().split()
        return (" ".join(tokens[:2]) in nemba_map) or (" ".join(tokens[:1]) in nemba_map)

    pc["is_nemba"] = pc["AcceptedSpecies"].apply(is_nemba)
    quad_iap_cover = (
        pc[pc["is_nemba"]]
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
        gbif_key, nemba_cat = _lookup_species(sp_raw, nemba_map)

        if iap_only and nemba_cat is None:
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
                "source_url": None,
                "species": sp_raw,
                "species_normalized": sp_norm,
                "gbif_usage_key": gbif_key,
                "nemba_category": nemba_cat,
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

    out_path = write_source_partition(gdf, source, root=root, run_id=run_id)
    logger.success("bioscape_plot: {} rows → {}", len(rows), out_path)
    return out_path
