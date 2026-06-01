"""GBIF Download API client + Darwin Core parser.

Two primary entry points:

``ingest_gbif``
    Full WC-wide NEMBA occurrence ingest.  Reads resolved taxa from a NEMBA
    parquet (+ optionally the 12-class schema taxa list), submits GBIF Download
    API jobs (chunked to stay under the 3 M record cap), parses DwC archives,
    and writes two source partitions to the unified observation store:

        gs://ism-data/labels/wc/obs/source=gbif/
        gs://ism-data/labels/wc/obs/source=inat_via_gbif/

    iNat records are identified by ``datasetKey == INAT_DATASET_KEY``.
    ``class_id`` is **not** assigned here — training configs crosswalk
    ``(gbif_usage_key, nemba_category) → class_id`` via YAML.

``resolve_taxa``
    vernacular_map → GBIF /species/match → pytaxize → /species/suggest.
    Writes a parquet of {name, usage_key, canonical_name, …}.

GBIF Download API reference:
https://techdocs.gbif.org/en/data-use/api-downloads
"""

from __future__ import annotations

import csv
import datetime as dt
import io
import os
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from loguru import logger
from shapely.geometry import Point

from cmrv.io import DATA_DIR, load_config, read_gdf, read_parquet_df
from cmrv.labels.observations import (
    COORD_UNCERTAINTY_DROP_M,
    WC_LABELS_ROOT,
    make_run_id,
    write_source_partition,
)

GBIF_API = "https://api.gbif.org/v1"
DOWNLOAD_ENDPOINT = f"{GBIF_API}/occurrence/download/request"
STATUS_ENDPOINT = f"{GBIF_API}/occurrence/download"
MATCH_ENDPOINT = f"{GBIF_API}/species/match"
SUGGEST_ENDPOINT = f"{GBIF_API}/species/suggest"

# iNat occurrences are redistributed through this GBIF dataset
INAT_DATASET_KEY = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"

# Max taxon keys per download job — keeps result sets under the 3 M record cap
TAXON_CHUNK_SIZE = 50

# Source trust weights (matches labels_schema.yaml)
GBIF_WEIGHT = 0.5
INAT_WEIGHT = 0.6  # research-grade iNat has tighter positional accuracy


# -- schema helpers ----------------------------------------------------------

load_schema = load_config  # re-export for existing callers


# -- name resolution ---------------------------------------------------------


def _match_direct(name: str) -> dict[str, Any] | None:
    """Primary resolver — direct HTTP hit against GBIF /species/match."""
    try:
        r = requests.get(
            MATCH_ENDPOINT,
            params={"name": name, "kingdom": "Plantae", "strict": "false"},
            timeout=30,
        )
        r.raise_for_status()
        j = r.json()
        if j.get("usageKey") and j.get("matchType") != "NONE":
            return j
    except Exception as e:
        logger.debug("direct match miss for {!r}: {}", name, e)
    return None


def _match_pytaxize(name: str) -> dict[str, Any] | None:
    """Secondary resolver — pytaxize GBIF backbone."""
    try:
        from pytaxize import gbif as px_gbif  # type: ignore[import-untyped]

        r = px_gbif.name_backbone(name=name, kingdom="Plantae", strict=False)
        if r and r.get("usageKey") and r.get("matchType") != "NONE":
            return r
    except Exception as e:
        logger.debug("pytaxize backbone miss for {!r}: {}", name, e)
    return None


def _suggest_direct(name: str) -> dict[str, Any] | None:
    """Last-resort fuzzy suggest via /species/suggest."""
    try:
        r = requests.get(
            SUGGEST_ENDPOINT,
            params={"q": name, "rank": "SPECIES", "kingdom": "Plantae", "limit": 5},
            timeout=30,
        )
        r.raise_for_status()
        cands = r.json() or []
        if cands:
            c = cands[0]
            return {
                "usageKey": c.get("key"),
                "scientificName": c.get("scientificName"),
                "matchType": "SUGGEST",
                "confidence": 50,
            }
    except Exception as e:
        logger.debug("direct suggest miss for {!r}: {}", name, e)
    return None


def resolve_taxa(
    taxa: list[dict[str, Any]], vernacular_map: dict[str, str]
) -> list[dict[str, Any]]:
    """Resolve each taxon → GBIF usageKey via 4-layer cascade.

    Input rows: ``{"name": str, "class_id": int}``.
    Output rows: input fields + ``{usage_key, canonical_name, match_type, confidence}``.
    """
    out: list[dict[str, Any]] = []
    for t in taxa:
        raw = str(t["name"])
        scientific = vernacular_map.get(raw.lower(), raw)
        r = _match_direct(scientific) or _match_pytaxize(scientific) or _suggest_direct(scientific)
        if not r or not r.get("usageKey"):
            logger.warning("could not resolve taxon {!r} — skipping", raw)
            continue
        out.append(
            {
                **t,
                "resolved_from": scientific,
                "usage_key": int(r["usageKey"]),
                "canonical_name": r.get("scientificName") or r.get("canonicalName"),
                "match_type": r.get("matchType", "UNKNOWN"),
                "confidence": int(r.get("confidence") or 0),
            }
        )
        logger.info(
            "resolved {!r:<30} → key={} match={} conf={}",
            raw,
            r["usageKey"],
            r.get("matchType"),
            r.get("confidence"),
        )
    return out


# -- download predicate ------------------------------------------------------


@dataclass(frozen=True)
class GBIFAuth:
    user: str
    password: str
    email: str

    @classmethod
    def from_env(cls) -> GBIFAuth:
        u, p, e = (
            os.environ.get("GBIF_USER"),
            os.environ.get("GBIF_PASS"),
            os.environ.get("GBIF_EMAIL"),
        )
        if not (u and p and e):
            raise RuntimeError("GBIF_USER / GBIF_PASS / GBIF_EMAIL must be set (see .env.example).")
        return cls(u, p, e)


def build_predicate(
    usage_keys: list[int],
    bbox: tuple[float, float, float, float],
    year_min: int,
    coord_uncertainty_max_m: int,
    basis_of_record: list[str],
    email: str,
) -> dict[str, Any]:
    """Build a GBIF Download predicate JSON.

    ``bbox`` is (min_lon, min_lat, max_lon, max_lat) in WGS84. We use coordinate
    predicates rather than a WKT polygon to sidestep GBIF's "polygon too complex"
    rejections — exact AOI polygon clip happens at parse time.
    """
    minx, miny, maxx, maxy = bbox
    return {
        "creator": email.split("@")[0],
        "notificationAddresses": [email],
        "sendNotification": True,
        "format": "DWCA",
        "predicate": {
            "type": "and",
            "predicates": [
                {"type": "in", "key": "TAXON_KEY", "values": [str(k) for k in usage_keys]},
                {"type": "greaterThanOrEquals", "key": "DECIMAL_LATITUDE", "value": str(miny)},
                {"type": "lessThanOrEquals", "key": "DECIMAL_LATITUDE", "value": str(maxy)},
                {"type": "greaterThanOrEquals", "key": "DECIMAL_LONGITUDE", "value": str(minx)},
                {"type": "lessThanOrEquals", "key": "DECIMAL_LONGITUDE", "value": str(maxx)},
                {"type": "greaterThanOrEquals", "key": "YEAR", "value": str(year_min)},
                {
                    "type": "lessThanOrEquals",
                    "key": "COORDINATE_UNCERTAINTY_IN_METERS",
                    "value": str(coord_uncertainty_max_m),
                },
                {"type": "in", "key": "BASIS_OF_RECORD", "values": basis_of_record},
                {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
                {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
            ],
        },
    }


def submit_download(predicate: dict[str, Any], auth: GBIFAuth) -> str:
    """POST predicate → returns a download_key (string)."""
    r = requests.post(
        DOWNLOAD_ENDPOINT,
        json=predicate,
        auth=(auth.user, auth.password),
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    if r.status_code not in (200, 201):
        raise RuntimeError(f"GBIF download request failed: {r.status_code} {r.text}")
    key = r.text.strip().strip('"')
    logger.info("GBIF download submitted — key={}", key)
    return key


def wait_for_download(
    download_key: str, auth: GBIFAuth, poll_seconds: int = 15, timeout_seconds: int = 3600
) -> dict[str, Any]:
    """Poll status until SUCCEEDED or FAILED."""
    start = time.time()
    while True:
        r = requests.get(
            f"{STATUS_ENDPOINT}/{download_key}", auth=(auth.user, auth.password), timeout=30
        )
        r.raise_for_status()
        meta = r.json()
        status = meta.get("status")
        logger.info(
            "GBIF download {} — status={} records={}",
            download_key,
            status,
            meta.get("totalRecords"),
        )
        if status == "SUCCEEDED":
            return meta
        if status in {"KILLED", "FAILED", "CANCELLED"}:
            raise RuntimeError(f"GBIF download ended with status={status}: {meta}")
        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"GBIF download {download_key} did not complete in {timeout_seconds}s"
            )
        time.sleep(poll_seconds)


def fetch_download_zip(download_key: str, out_path: Path) -> Path:
    """Fetch the DwC zip. No auth needed for the archive URL itself."""
    url = f"{STATUS_ENDPOINT}/request/{download_key}.zip"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(1 << 16):
                f.write(chunk)
    logger.success("wrote DwC zip → {} ({:.1f} MB)", out_path, out_path.stat().st_size / 1e6)
    return out_path


# -- parse + unified schema --------------------------------------------------

_DWC_COLUMNS = [
    "gbifID",
    "occurrenceID",
    "catalogNumber",  # iNat observation ID (present when datasetKey == INAT_DATASET_KEY)
    "species",
    "scientificName",
    "taxonKey",
    "acceptedTaxonKey",
    "decimalLatitude",
    "decimalLongitude",
    "coordinateUncertaintyInMeters",
    "eventDate",
    "year",
    "basisOfRecord",
    "datasetKey",
    "institutionCode",
    "countryCode",
]


def _parse_dwca_to_obs_gdf(
    zip_path: Path,
    aoi_gdf: gpd.GeoDataFrame,
    key_to_info: dict[int, dict[str, Any]],
    rejection_names: list[str],
    ingested_at: dt.datetime,
    run_id: str,
) -> gpd.GeoDataFrame:
    """Parse ``occurrence.txt`` from a DwC archive → GeoDataFrame with unified schema.

    Parameters
    ----------
    key_to_info:
        Maps GBIF usage_key → dict with keys ``usage_key``, ``nemba_category``
        (str or None), ``weight`` (float).  Built from either the NEMBA resolved
        parquet or the legacy schema taxa list.
    """
    with zipfile.ZipFile(zip_path) as zf, zf.open("occurrence.txt") as f:
        raw = f.read()

    df = pd.read_csv(
        io.BytesIO(raw),
        sep="\t",
        dtype=str,
        on_bad_lines="skip",
        quoting=csv.QUOTE_NONE,
    )
    have = [c for c in _DWC_COLUMNS if c in df.columns]
    df = df[have].copy()

    # Cast numeric columns
    for col in ("decimalLatitude", "decimalLongitude", "coordinateUncertaintyInMeters"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("taxonKey", "acceptedTaxonKey"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Drop rows with no coordinates
    df = df.dropna(subset=["decimalLongitude", "decimalLatitude"])

    # Reject native look-alikes
    if rejection_names:
        pattern = "|".join(re.escape(n) for n in rejection_names)
        mask = df["scientificName"].str.contains(pattern, na=False, regex=True)
        n_before = len(df)
        df = df[~mask].copy()
        logger.info("rejected {} record(s) by taxa_rejection list", n_before - len(df))

    # Keep null uncertainty; drop > threshold (defensive — API predicate already filters)
    unc = df["coordinateUncertaintyInMeters"]
    df = df[unc.isna() | (unc <= COORD_UNCERTAINTY_DROP_M)].copy()

    if df.empty:
        logger.warning("no records remain after initial filters in {}", zip_path.name)
        return gpd.GeoDataFrame()

    # Build join table from key_to_info
    info_rows = [
        {
            "_key": int(k),
            "gbif_usage_key": int(v["usage_key"]),
            "nemba_category": v.get("nemba_category"),
            "_base_weight": float(v.get("weight", GBIF_WEIGHT)),
        }
        for k, v in key_to_info.items()
    ]
    key_df = pd.DataFrame(info_rows)
    key_df["_key"] = key_df["_key"].astype("Int64")

    # Join on taxonKey; fall back to acceptedTaxonKey for unresolved rows
    df_joined = df.merge(key_df.rename(columns={"_key": "taxonKey"}), on="taxonKey", how="left")
    unresolved = df_joined[df_joined["gbif_usage_key"].isna()].drop(
        columns=["gbif_usage_key", "nemba_category", "_base_weight"]
    )
    df = df_joined[df_joined["gbif_usage_key"].notna()].copy()

    if len(unresolved) > 0 and "acceptedTaxonKey" in unresolved.columns:
        resolved2 = unresolved.merge(
            key_df.rename(columns={"_key": "acceptedTaxonKey"}),
            on="acceptedTaxonKey",
            how="inner",
        )
        df = pd.concat([df, resolved2], ignore_index=True)

    n_matched = len(df)
    logger.info("{} records matched key_to_info (of {} after filters)", n_matched, n_matched)

    if df.empty:
        return gpd.GeoDataFrame()

    # Build obs_id: iNat records get "inat:<catalogNumber>", others get "gbif:<gbifID>"
    has_catalog = "catalogNumber" in df.columns
    if has_catalog:
        is_inat = df["datasetKey"] == INAT_DATASET_KEY
        catalog_ok = df["catalogNumber"].notna() & (df["catalogNumber"].str.len() > 0)
        obs_id = np.where(
            is_inat & catalog_ok,
            "inat:" + df["catalogNumber"].fillna(""),
            "gbif:" + df["gbifID"].fillna(""),
        )
    else:
        obs_id = "gbif:" + df["gbifID"].fillna("")

    is_inat = df["datasetKey"] == INAT_DATASET_KEY
    source = np.where(is_inat, "inat_via_gbif", "gbif")
    weight = np.where(is_inat, INAT_WEIGHT, df["_base_weight"]).astype("float32")

    # species_normalized: prefer DwC `species` (binomial); fall back to
    # `scientificName` for genus-only records.  Extract up to two words
    # (genus + epithet); single-word genus names are kept as-is.
    raw_name = df["species"].fillna(df.get("scientificName", pd.Series(dtype=str)))
    cleaned = raw_name.str.replace("×", "", regex=False).str.strip().str.lower()
    binomial = cleaned.str.extract(r"^(\w+\s+\w+)")[0]
    genus_only = cleaned.str.extract(r"^(\w+)")[0]
    species_normalized = binomial.fillna(genus_only)

    n_genus_only = species_normalized.notna().sum() - binomial.notna().sum()
    if n_genus_only > 0:
        logger.info("{} records resolved to genus-only (no species epithet)", n_genus_only)

    event_date = pd.to_datetime(
        df["eventDate"].str[:10] if "eventDate" in df.columns else None,
        format="%Y-%m-%d",
        errors="coerce",
    ).dt.date

    df = df.assign(
        obs_id=obs_id,
        source=source,
        source_record_id=df["gbifID"].astype(str),
        source_url="https://www.gbif.org/occurrence/" + df["gbifID"].fillna(""),
        species=raw_name,
        species_normalized=species_normalized,
        event_date=event_date,
        geom_type="point",
        cover_pct=None,
        weight=weight,
        ingested_at=ingested_at,
        ingest_run_id=run_id,
        aoi_admin1="western_cape",
        coord_uncertainty_m=df["coordinateUncertaintyInMeters"],
        basis_of_record=df.get("basisOfRecord"),
    )

    geometries = [
        Point(lon, lat)
        for lon, lat in zip(df["decimalLongitude"], df["decimalLatitude"], strict=True)
    ]
    gdf = gpd.GeoDataFrame(df, geometry=geometries, crs="EPSG:4326")

    # Exact AOI polygon clip (bbox was the download bbox; AOI polygon is the precise shape)
    aoi_wgs = aoi_gdf.to_crs("EPSG:4326")
    aoi_union = aoi_wgs.union_all() if hasattr(aoi_wgs, "union_all") else aoi_wgs.unary_union
    gdf = gdf[gdf.geometry.within(aoi_union)].copy()
    logger.info("retained {} records within AOI polygon (of {})", len(gdf), n_matched)

    return gdf


# -- top-level ingest --------------------------------------------------------


def ingest_gbif(
    aoi_uri: str = "gs://ism-data/aoi/western_cape.parquet",
    nemba_resolved_uri: str | None = None,
    schema_path: str | Path = "configs/labels_schema.yaml",
    class_map_name: str = "upper_berg_12",
    root: str = WC_LABELS_ROOT,
    cache_dir: Path | None = None,
    taxon_chunk_size: int = TAXON_CHUNK_SIZE,
) -> gpd.GeoDataFrame:
    """Full WC-wide NEMBA occurrence ingest → partitioned observation store.

    Taxa sourced from ``nemba_resolved_uri`` parquet (preferred; columns:
    ``usage_key``, ``nemba_category``) or from the ``gbif.taxa`` block in
    ``schema_path`` (legacy; no ``nemba_category``).

    Usage keys are chunked into groups of ``taxon_chunk_size`` to stay under
    the GBIF 3 M records-per-job cap.  Each chunk triggers one download job.

    Writes two hive partitions to ``root``:
        ``source=gbif/``         — all non-iNat records
        ``source=inat_via_gbif/`` — iNat research-grade records

    Returns the combined GeoDataFrame (both sources merged).
    """
    from cmrv.labels.classmap import gbif_taxa_from_schema

    schema = load_schema(schema_path)
    gblk = schema.get("gbif", {})
    rejection = schema.get("taxa_rejection", [])
    schema_taxa = gbif_taxa_from_schema(schema_path, class_map_name)

    aoi_gdf = read_gdf(aoi_uri)
    bbox_raw = tuple(aoi_gdf.to_crs("EPSG:4326").total_bounds.tolist())
    buf_deg = float(gblk.get("aoi_buffer_km", 5.0)) / 111.0
    bbox = (
        bbox_raw[0] - buf_deg,
        bbox_raw[1] - buf_deg,
        bbox_raw[2] + buf_deg,
        bbox_raw[3] + buf_deg,
    )

    # Build key_to_info dict
    key_to_info: dict[int, dict[str, Any]] = {}
    if nemba_resolved_uri:
        nemba_df = read_parquet_df(nemba_resolved_uri)
        for rec in nemba_df.to_dict("records"):
            uk = int(rec.get("gbif_usage_key") or rec.get("usage_key"))
            key_to_info[uk] = {
                "usage_key": uk,
                "nemba_category": rec.get("nemba_category"),
                "weight": GBIF_WEIGHT,
            }
        logger.info("loaded {} NEMBA taxa from {}", len(key_to_info), nemba_resolved_uri)
    else:
        vernacular_map = schema.get("vernacular_map", {})
        resolved = resolve_taxa(schema_taxa, vernacular_map)
        for t in resolved:
            uk = int(t["usage_key"])
            key_to_info[uk] = {
                "usage_key": uk,
                "nemba_category": None,
                "weight": GBIF_WEIGHT,
            }
        logger.info("resolved {} taxa from schema (no NEMBA parquet provided)", len(key_to_info))

    # Always merge schema taxa (species-level keys) so the parser can match
    # records that the NEMBA file resolved to genus-level keys only.
    vernacular_map = schema.get("vernacular_map", {})
    if schema_taxa:
        resolved = resolve_taxa(schema_taxa, vernacular_map)
        n_added = 0
        for t in resolved:
            uk = int(t["usage_key"])
            if uk not in key_to_info:
                key_to_info[uk] = {
                    "usage_key": uk,
                    "nemba_category": None,
                    "weight": GBIF_WEIGHT,
                }
                n_added += 1
        if n_added:
            logger.info("merged {} additional species-level keys from schema gbif.taxa", n_added)

    if not key_to_info:
        raise RuntimeError("No taxa resolved — check nemba_resolved_uri or schema gbif.taxa")

    usage_keys = sorted(key_to_info.keys())
    chunks = [
        usage_keys[i : i + taxon_chunk_size] for i in range(0, len(usage_keys), taxon_chunk_size)
    ]
    logger.info(
        "total {} usage keys — {} chunk(s) of ≤{}", len(usage_keys), len(chunks), taxon_chunk_size
    )

    auth = GBIFAuth.from_env()
    cache = cache_dir or (DATA_DIR / "cache" / "gbif")
    run_id = make_run_id("gbif")
    ingested_at = dt.datetime.now(tz=dt.UTC)

    year_min = int(str(gblk.get("date_range", ["2018-01-01"])[0])[:4])
    coord_unc_max = int(gblk.get("coordinate_uncertainty_max_m", 500))
    basis = list(gblk.get("basis_of_record", ["HUMAN_OBSERVATION"]))

    all_gdfs: list[gpd.GeoDataFrame] = []
    for i, chunk in enumerate(chunks):
        logger.info("chunk {}/{}: {} keys", i + 1, len(chunks), len(chunk))
        predicate = build_predicate(
            usage_keys=chunk,
            bbox=bbox,
            year_min=year_min,
            coord_uncertainty_max_m=coord_unc_max,
            basis_of_record=basis,
            email=auth.email,
        )
        dk = submit_download(predicate, auth)
        meta = wait_for_download(dk, auth)
        logger.info("chunk {} total records: {}", i + 1, meta.get("totalRecords"))

        zip_path = Path(cache) / f"{dk}.zip"
        fetch_download_zip(dk, zip_path)

        chunk_gdf = _parse_dwca_to_obs_gdf(
            zip_path=zip_path,
            aoi_gdf=aoi_gdf,
            key_to_info=key_to_info,
            rejection_names=rejection,
            ingested_at=ingested_at,
            run_id=run_id,
        )
        if not chunk_gdf.empty:
            all_gdfs.append(chunk_gdf)

    if not all_gdfs:
        logger.warning("ingest_gbif: no records returned across all chunks")
        return gpd.GeoDataFrame()

    combined = gpd.GeoDataFrame(
        pd.concat(all_gdfs, ignore_index=True), geometry="geometry", crs="EPSG:4326"
    )
    # Deduplicate across chunks (same obs_id from overlapping accepted keys)
    combined = combined.drop_duplicates(subset=["obs_id"], keep="first")
    logger.info("combined: {} records after cross-chunk dedup", len(combined))

    # Write source=gbif partition
    gbif_gdf = combined[combined["source"] == "gbif"].copy()
    if not gbif_gdf.empty:
        write_source_partition(gbif_gdf, "gbif", root=root, run_id=run_id)
        logger.success("source=gbif: {} records", len(gbif_gdf))

    # Write source=inat_via_gbif partition
    inat_gdf = combined[combined["source"] == "inat_via_gbif"].copy()
    if not inat_gdf.empty:
        write_source_partition(inat_gdf, "inat_via_gbif", root=root, run_id=run_id)
        logger.success("source=inat_via_gbif: {} records", len(inat_gdf))

    return combined
