"""Vegmap 2024 loader — SANBI NVM2024 (IEM5) native vegetation polygons.

``ingest_vegmap``
    Clips to Western Cape AOI, emits unified observation schema with
    ``geom_type=polygon``, ``coord_uncertainty_m=None``, ``nemba_category=None``.
    ``species`` stores the vegtype name; ``species_normalized`` encodes
    ``"biome|bioregion"`` so that ``fuse.py`` can apply the biome/bioregion →
    class_id crosswalk at fuse time without re-reading the shapefile.

Field names verified against NVM2024Final_IEM5_12_07012025.shp (2026-04-17):
  T_BIOME, T_BIOREGIO, T_Name.
"""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import geopandas as gpd
from loguru import logger
from shapely import to_wkb

from cmrv.io import read_gdf
from cmrv.labels.observations import WC_LABELS_ROOT, make_run_id, write_source_partition

_NS_VEGMAP = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")  # UUID namespace


def _vegmap_obs_id(objectid: int | None, biome: str, bioregion: str, wkb: bytes) -> str:
    if objectid is not None:
        return f"vegmap_iem5:{objectid}"
    seed = f"{biome}|{bioregion}|{wkb.hex()[:32]}"
    return f"vegmap_iem5:{uuid.uuid5(_NS_VEGMAP, seed)}"


def ingest_vegmap(
    shp_path: str | Path,
    aoi_uri: str = "gs://ism-data/aoi/western_cape.parquet",
    root: str = WC_LABELS_ROOT,
    run_id: str | None = None,
    field_biome: str = "T_BIOME",
    field_bioregion: str = "T_BIOREGIO",
    field_vegtype: str = "T_Name",
    weight: float = 0.8,
) -> str:
    """Clip NVM2024 IEM5 to WC AOI and write to unified observation store.

    Emits rows with ``source=vegmap``, ``geom_type=polygon``,
    ``coord_uncertainty_m=None``, ``nemba_category=None`` (native vegetation).
    These serve as negative-sample anchors for training.
    """
    source = "vegmap"
    run_id = run_id or make_run_id(source)
    ingested_at = dt.datetime.now(tz=dt.UTC)

    aoi = read_gdf(aoi_uri)
    vm_sample = gpd.read_file(shp_path, rows=1)
    aoi_vm = aoi.to_crs(vm_sample.crs)
    bbox = tuple(aoi_vm.total_bounds)
    vm = gpd.read_file(shp_path, bbox=bbox)
    logger.info("loaded {} vegmap polygons within WC bbox", len(vm))

    clipped = gpd.overlay(vm, aoi_vm[["geometry"]], how="intersection")
    clipped = clipped.to_crs("EPSG:4326")
    logger.info("after exact WC clip: {} polygons", len(clipped))

    rows: list[dict] = []
    for _, row in clipped.iterrows():
        biome = str(row.get(field_biome) or "")
        bioregion = str(row.get(field_bioregion) or "")
        vegtype = str(row.get(field_vegtype) or "")
        objectid = row.get("OBJECTID") or row.get("objectid")

        try:
            wkb_bytes = bytes(to_wkb(row.geometry, hex=False))
        except Exception:
            continue

        obs_id = _vegmap_obs_id(objectid, biome, bioregion, wkb_bytes)
        rows.append(
            {
                "obs_id": obs_id,
                "source": source,
                "source_record_id": str(objectid or obs_id),
                "source_url": None,
                "species": vegtype,
                "species_normalized": f"{biome}|{bioregion}",
                "gbif_usage_key": None,
                "nemba_category": None,
                "geom_type": "polygon",
                "coord_uncertainty_m": None,
                "event_date": None,
                "basis_of_record": "VEG_POLYGON",
                "cover_pct": None,
                "weight": float(weight),
                "ingested_at": ingested_at,
                "ingest_run_id": run_id,
                "aoi_admin1": "western_cape",
            }
        )

    if not rows:
        logger.warning("no rows after WC clip for vegmap")
        return ""

    import pandas as pd

    pdf = pd.DataFrame(rows)
    gdf = gpd.GeoDataFrame(pdf, geometry=clipped.geometry.values, crs="EPSG:4326")
    out_path = write_source_partition(gdf, source, root=root, run_id=run_id)
    logger.success("vegmap: {} polygons → {}", len(rows), out_path)
    return out_path
