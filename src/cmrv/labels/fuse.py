"""Label fusion + rasterization — VIZ-ONLY sanity output, NOT training input.

Phase 0 trains on the chip / point regime: ``ingest-chips`` extracts a 64×64
chip per (obs_id, month) and the model consumes embedding-patch vectors at
each obs coordinate across T months.  Class assignment is per-obs_id via
``cmrv.labels.classmap``, not per-pixel via raster lookup.  The training
pipeline never reads ``label.tif``.

This module is retained as a visualization helper:
  * Run ``cmrv labels-fuse`` to produce a per-tile sparse label COG that you
    can drop into QGIS / a Streamlit map to spot-check label coverage,
    weight-resolution conflicts, and AOI gaps.
  * The post-fuse stage of ``cmrv labels-audit-classmap`` ingests these COGs
    if they exist; if you don't fuse, run the audit with ``--stage pre``.

Earlier roadmap drafts (Phase_0_Build_Roadmap.md §3 Stage 4, §7) assumed
dense per-pixel segmentation training, which would have made this raster a
required training input.  That regime was dropped in favour of the
chip-centred design — see ``tasks/lessons.md`` for context.

Sources loaded via ``load_training_labels()``:

1. **IAP sources** (classes 0–7): class assignment via
   ``cmrv.labels.classmap.build_lookup`` (members[] in
   ``class_maps.<name>``).

2. **Native-vegetation sources** (classes 8–10): loaded from ``source=vegmap``
   in the unified store.  Class assignment uses the ``biome|bioregion``
   encoding in ``species_normalized`` (set by ``ingest_vegmap()``) together
   with the ``biome_to_class`` / ``bioregion_to_class`` crosswalk in the
   schema YAML.

Fusion rule — **highest-weight-wins** within each pixel:
   Observations are sorted by ``weight`` ascending then rasterized in order
   so the highest-weight observation overwrites lower ones.
   Points are buffered by ``POINT_BUFFER_M`` (= 10 m = 4 px at 2.5 m) before
   rasterization.  Polygons (Vegmap) are rasterized directly.
   Pixels with no observation → ``nodata=NODATA`` (255).

Output per tile: ``<out_prefix>/tile_id={tile_id}/label.tif``
   uint8, EPSG:32734, nodata=255, Cloud-Optimized GeoTIFF validated by
   rio-cogeo.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
from loguru import logger
from rasterio.transform import from_bounds
from shapely.geometry import mapping

from cmrv.io import load_config, open_raster, read_gdf, write_cog, write_gdf_parquet
from cmrv.labels.classmap import ClassMap, build_lookup
from cmrv.labels.merge import load_training_labels
from cmrv.labels.observations import WC_LABELS_ROOT

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESOLUTION_M: float = 2.5  # SEN2SR output GSD
NODATA: int = 255  # uint8 fill for unlabelled pixels
POINT_BUFFER_M: float = 10.0  # 4 px at 2.5 m, per roadmap §3 Stage 4
RASTER_CRS: str = "EPSG:32734"  # UTM zone 34S — matches tile grid
DEFAULT_OUT_PREFIX: str = "gs://ism-data/labels"
DEFAULT_TILES_URI: str = "gs://ism-data/aoi/tiles.parquet"
DEFAULT_CLASS_MAP: str = "upper_berg_12"


# ---------------------------------------------------------------------------
# Class crosswalk helpers
# ---------------------------------------------------------------------------


def assign_class_id(
    gdf: gpd.GeoDataFrame,
    classmap: ClassMap,
) -> gpd.GeoDataFrame:
    """Attach ``class_id`` (int or NaN) to each observation row.

    Resolution delegated to :meth:`ClassMap.resolve`: exact binomial first,
    then per-class genus fallback.  Rows with no match get ``NaN`` and are
    dropped by the caller.
    """
    result = gdf.copy()
    class_ids: list[int | float] = []
    for _, row in result.iterrows():
        cid, _ = classmap.resolve(row.get("species_normalized"))
        class_ids.append(float("nan") if cid is None else cid)
    result["class_id"] = class_ids
    return result


# ---------------------------------------------------------------------------
# Rasterization helpers
# ---------------------------------------------------------------------------


def _tile_transform_shape(
    tile_geom_utm: Any,
    resolution_m: float,
) -> tuple[rasterio.transform.Affine, int, int]:
    """Compute rasterio Affine transform + (height, width) for a UTM tile."""
    minx, miny, maxx, maxy = tile_geom_utm.bounds
    width = int(np.ceil((maxx - minx) / resolution_m))
    height = int(np.ceil((maxy - miny) / resolution_m))
    transform = from_bounds(minx, miny, maxx, maxy, width, height)
    return transform, height, width


def rasterize_tile(
    gdf: gpd.GeoDataFrame,
    tile_geom_utm: Any,
    resolution_m: float = RESOLUTION_M,
    nodata: int = NODATA,
    raster_crs: str = RASTER_CRS,
    point_buffer_m: float = POINT_BUFFER_M,
) -> tuple[np.ndarray, Any, str]:
    """Rasterize labeled observations for a single tile.

    Observations in ``gdf`` must have ``class_id`` (int) and ``weight``
    (float) columns.  Points are buffered by ``point_buffer_m`` before
    rasterization; polygon geometries are used as-is.

    Returns ``(label_array, affine_transform, crs_string)``.
    """
    transform, height, width = _tile_transform_shape(tile_geom_utm, resolution_m)
    label = np.full((height, width), nodata, dtype=np.uint8)

    if gdf.empty:
        return label, transform, raster_crs

    # Reproject to raster CRS
    gdf_utm = gdf.to_crs(raster_crs)

    # Buffer points; keep polygons as-is
    point_mask = gdf_utm.geom_type == "Point"
    geoms = gdf_utm.geometry.copy()
    geoms[point_mask] = gdf_utm.geometry[point_mask].buffer(point_buffer_m)

    # Sort ascending by weight so highest-weight overwrites
    order = gdf_utm["weight"].argsort().values
    shapes = [
        (mapping(geoms.iloc[i]), int(gdf_utm["class_id"].iloc[i]))
        for i in order
        if int(gdf_utm["class_id"].iloc[i]) != nodata
    ]

    if shapes:
        burned = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(height, width),
            transform=transform,
            fill=nodata,
            dtype=np.uint8,
            all_touched=False,
        )
        # Only overwrite background pixels if observation lands there
        label = burned

    return label, transform, raster_crs


def write_label_cog(
    arr: np.ndarray,
    transform: Any,
    crs: str,
    out_uri: str,
    nodata: int = NODATA,
) -> str:
    """Write a uint8 label array as a Cloud-Optimized GeoTIFF.

    Delegates to ``cmrv.io.write_cog`` for the temp-file + translate + upload
    pipeline.  Returns ``out_uri``.
    """
    uri = write_cog(arr, transform, crs, out_uri, dtype="uint8", nodata=nodata)
    logger.success("wrote label COG → {} ({}×{})", out_uri, arr.shape[1], arr.shape[0])
    return uri


# ---------------------------------------------------------------------------
# Per-tile orchestrator
# ---------------------------------------------------------------------------


def _assign_vegmap_class_id(
    gdf: gpd.GeoDataFrame,
    biome_to_class: dict[str, int],
    bioregion_to_class: dict[str, int],
) -> gpd.GeoDataFrame:
    """Assign class_id to vegmap observations using the biome|bioregion encoding."""
    class_ids: list[int | float] = []
    for _, row in gdf.iterrows():
        parts = (row.get("species_normalized") or "").split("|")
        biome = parts[0].strip() if len(parts) > 0 else ""
        bioregion = parts[1].strip() if len(parts) > 1 else ""
        cid = bioregion_to_class.get(bioregion) or biome_to_class.get(biome)
        class_ids.append(cid if cid is not None else float("nan"))
    gdf = gdf.copy()
    gdf["class_id"] = class_ids
    return gdf


def fuse_tile(
    tile_id: int,
    tile_row: gpd.GeoSeries,
    schema_path: str | Path = "configs/labels_schema.yaml",
    class_map_name: str = DEFAULT_CLASS_MAP,
    labels_root: str = WC_LABELS_ROOT,
    out_prefix: str = DEFAULT_OUT_PREFIX,
    resolution_m: float = RESOLUTION_M,
    max_coord_uncertainty_m: float = 500.0,
    date_min: str = "2018-01-01",
) -> str:
    """Fuse observation store labels for one tile → label COG.

    ``tile_row`` is a row from the tiles GeoDataFrame (EPSG:32734).
    The tile polygon is converted to EPSG:4326 for ``load_training_labels``
    spatial clip, then back to EPSG:32734 for rasterization.

    Both IAP sources (classes 0–7) and native-vegetation sources (classes 8–10)
    are loaded from the unified observation store. Native class assignment uses
    the ``biome|bioregion`` encoding in ``species_normalized`` set by
    ``ingest_vegmap()``.

    Returns the output URI of the written COG.
    """
    schema = load_config(schema_path)
    classmap = build_lookup(schema_path, class_map_name)

    tile_geom_utm = tile_row.geometry
    tile_gdf_utm = gpd.GeoDataFrame(
        {"tile_id": [tile_id]}, geometry=[tile_geom_utm], crs=RASTER_CRS
    )
    tile_gdf_4326 = tile_gdf_utm.to_crs("EPSG:4326")

    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        tmp_aoi = f.name
    try:
        write_gdf_parquet(tile_gdf_4326, tmp_aoi)

        # --- IAP observations (classes 0–7) from unified store ---
        iap_sources = ["gbif", "inat_via_gbif", "bioscape_line", "bioscape_plot"]
        try:
            iap_gdf = load_training_labels(
                aoi_uri=tmp_aoi,
                sources=iap_sources,
                root=labels_root,
                max_coord_uncertainty_m=max_coord_uncertainty_m,
                date_min=date_min,
            )
        except FileNotFoundError:
            logger.warning(
                "tile {}: no unified label store at {} — skipping IAP sources",
                tile_id,
                labels_root,
            )
            iap_gdf = gpd.GeoDataFrame()

        if not iap_gdf.empty:
            iap_gdf = assign_class_id(iap_gdf, classmap)
            n_before = len(iap_gdf)
            iap_gdf = iap_gdf.dropna(subset=["class_id"]).copy()
            iap_gdf["class_id"] = iap_gdf["class_id"].astype(int)
            logger.info(
                "tile {}: {} IAP obs, {} after class crosswalk", tile_id, n_before, len(iap_gdf)
            )

        # --- Native-vegetation observations (classes 8–10) from unified store ---
        native_gdf = gpd.GeoDataFrame()
        try:
            native_raw = load_training_labels(
                aoi_uri=tmp_aoi,
                sources=["vegmap"],
                root=labels_root,
            )
        except FileNotFoundError:
            native_raw = gpd.GeoDataFrame()

        if not native_raw.empty:
            vm_cfg = schema.get("vegmap_2024", {})
            biome_to_class: dict[str, int] = vm_cfg.get("biome_to_class", {})
            bioregion_to_class: dict[str, int] = vm_cfg.get("bioregion_to_class", {})
            native_weight = float(vm_cfg.get("weight", 0.8))

            native_gdf = _assign_vegmap_class_id(native_raw, biome_to_class, bioregion_to_class)
            native_gdf = native_gdf.dropna(subset=["class_id"]).copy()
            native_gdf["class_id"] = native_gdf["class_id"].astype(int)
            if "weight" not in native_gdf.columns or native_gdf["weight"].isna().all():
                native_gdf["weight"] = native_weight
            logger.info("tile {}: {} native vegmap polygons", tile_id, len(native_gdf))
    finally:
        Path(tmp_aoi).unlink(missing_ok=True)

    # --- Merge IAP + native ---
    parts = [df for df in [iap_gdf, native_gdf] if not df.empty]
    if not parts:
        logger.warning("tile {}: no labeled observations — writing all-nodata COG", tile_id)
        transform, height, width = _tile_transform_shape(tile_geom_utm, resolution_m)
        arr = np.full((height, width), NODATA, dtype=np.uint8)
    else:
        combined = gpd.GeoDataFrame(
            pd.concat(parts, ignore_index=True),
            geometry="geometry",
            crs="EPSG:4326",
        )
        required_cols = {"class_id", "weight", "geometry"}
        missing = required_cols - set(combined.columns)
        if missing:
            raise ValueError(f"fuse_tile: combined GDF missing columns {missing}")

        arr, transform, _ = rasterize_tile(
            combined,
            tile_geom_utm=tile_geom_utm,
            resolution_m=resolution_m,
        )

    labeled_px = int((arr != NODATA).sum())
    total_px = arr.size
    logger.info(
        "tile {}: {}/{} labeled pixels ({:.1f}%)",
        tile_id,
        labeled_px,
        total_px,
        100.0 * labeled_px / total_px if total_px else 0,
    )

    out_uri = f"{out_prefix}/tile_id={tile_id}/label.tif"
    return write_label_cog(arr, transform, RASTER_CRS, out_uri, nodata=NODATA)


# ---------------------------------------------------------------------------
# Batch orchestrator
# ---------------------------------------------------------------------------


def fuse_all(
    tiles_uri: str = DEFAULT_TILES_URI,
    schema_path: str | Path = "configs/labels_schema.yaml",
    class_map_name: str = DEFAULT_CLASS_MAP,
    labels_root: str = WC_LABELS_ROOT,
    out_prefix: str = DEFAULT_OUT_PREFIX,
    resolution_m: float = RESOLUTION_M,
) -> list[str]:
    """Fuse labels for all tiles in the grid.  Returns list of written URIs."""
    tiles = read_gdf(tiles_uri)
    if "tile_id" not in tiles.columns:
        tiles["tile_id"] = range(len(tiles))

    results: list[str] = []
    for _, row in tiles.iterrows():
        tid = int(row["tile_id"])
        try:
            uri = fuse_tile(
                tile_id=tid,
                tile_row=row,
                schema_path=schema_path,
                class_map_name=class_map_name,
                labels_root=labels_root,
                out_prefix=out_prefix,
                resolution_m=resolution_m,
            )
            results.append(uri)
        except Exception as e:
            logger.error("tile {}: fuse failed — {}", tid, e)
    return results


# ---------------------------------------------------------------------------
# Per-class pixel counts (DoD check helper)
# ---------------------------------------------------------------------------


def label_class_counts(uri: str, nodata: int = NODATA) -> dict[int, int]:
    """Read a label COG and return per-class pixel counts (excludes nodata).

    GCS URIs are streamed via /vsigs/ — no local download.
    """
    with open_raster(uri) as src:
        arr = src.read(1)
    unique, counts = np.unique(arr[arr != nodata], return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist(), strict=True))
