"""NLC 2022 raster sampling — balanced non-IAP training labels.

Reads the SA National Land Cover 2022 raster (20 m, uint8) windowed to the
WC AOI, crosswalks NLC values to training classes via
``configs/labels_schema.yaml → nlc_2022``, and samples spatially-uniform
points per class.  Vegmap 2024 polygons provide a stratification layer to
disambiguate NLC class 8 ("low shrubland other") into renosterveld (class 9)
where the vegmap bioregion confirms it.

Output is written to the unified observation store as ``source="nlc_sample"``
with ``geom_type="point"``, compatible with the chip extraction pipeline.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
import rasterio.features
from loguru import logger
from pyproj import Transformer
from rasterio.windows import from_bounds
from shapely.geometry import Point, mapping

from cmrv.io import load_config, read_gdf
from cmrv.labels.observations import WC_LABELS_ROOT, make_run_id, write_source_partition

_SOURCE = "nlc_sample"

_TRAINING_CLASS_NAMES: dict[int, str] = {
    8: "fynbos",
    9: "renosterveld",
    10: "indigenous_forest",
    11: "other_landcover",
}


def _build_nlc_to_class(schema: dict) -> dict[int, int]:
    """Build NLC pixel value → training class_id from the schema crosswalk."""
    nlc_cfg = schema["nlc_2022"]
    groups: dict[str, list[int]] = nlc_cfg["class_groups"]
    crosswalk: dict[str, list[str]] = nlc_cfg["class_crosswalk"]
    exclusions: list[str] = nlc_cfg.get("exclusions", [])

    excluded_vals: set[int] = set()
    for grp_name in exclusions:
        excluded_vals.update(groups.get(grp_name, []))

    mapping_out: dict[int, int] = {}
    for class_id_str, group_names in crosswalk.items():
        class_id = int(class_id_str)
        for grp_name in group_names:
            for nlc_val in groups.get(grp_name, []):
                if nlc_val not in excluded_vals:
                    mapping_out[nlc_val] = class_id
    return mapping_out


def _read_nlc_windowed(
    nlc_path: str | Path, aoi: gpd.GeoDataFrame
) -> tuple[np.ndarray, rasterio.transform.Affine, rasterio.crs.CRS]:
    """Read the NLC raster windowed to the AOI bounding box."""
    with rasterio.open(nlc_path) as src:
        nlc_crs = src.crs
        t = Transformer.from_crs("EPSG:4326", nlc_crs, always_xy=True)
        aoi_wgs = aoi.to_crs("EPSG:4326")
        minx, miny, maxx, maxy = aoi_wgs.total_bounds
        x1, y1 = t.transform(minx, miny)
        x2, y2 = t.transform(maxx, maxy)
        win = from_bounds(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2), src.transform)
        arr = src.read(1, window=win)
        win_transform = src.window_transform(win)
        logger.info(
            "NLC window: {}×{} pixels ({:.0f} km²)",
            arr.shape[1],
            arr.shape[0],
            arr.shape[0] * arr.shape[1] * 20 * 20 / 1e6,
        )
    return arr, win_transform, nlc_crs


def _build_renosterveld_mask(
    vegmap_path: str | Path,
    aoi: gpd.GeoDataFrame,
    schema: dict,
    nlc_shape: tuple[int, int],
    nlc_transform: rasterio.transform.Affine,
    nlc_crs: rasterio.crs.CRS,
) -> np.ndarray:
    """Rasterize renosterveld vegmap bioregion polygons onto the NLC grid."""
    vm_cfg = schema.get("vegmap_2024", {})
    bioregion_to_class: dict[str, int] = vm_cfg.get("bioregion_to_class", {})
    renosterveld_bioregions = {k for k, v in bioregion_to_class.items() if v == 9}

    if not renosterveld_bioregions:
        logger.warning("no renosterveld bioregions in schema — mask will be empty")
        return np.zeros(nlc_shape, dtype=bool)

    vm_sample = gpd.read_file(vegmap_path, rows=1)
    aoi_vm = aoi.to_crs(vm_sample.crs)
    bbox = tuple(aoi_vm.total_bounds)
    vm = gpd.read_file(vegmap_path, bbox=bbox)
    vm = gpd.overlay(vm, aoi_vm[["geometry"]], how="intersection")

    field_bioregion = vm_cfg.get("field_bioregion", "T_BIOREGIO")
    field_vegtype = vm_cfg.get("field_vegtype", "T_Name")
    vegtype_pattern = vm_cfg.get("vegtype_renosterveld_pattern", "")

    in_bioregion = vm[field_bioregion].isin(renosterveld_bioregions)
    if vegtype_pattern and field_vegtype in vm.columns:
        has_vegtype = vm[field_vegtype].str.contains(vegtype_pattern, case=False, na=False)
        renoster = vm[in_bioregion & has_vegtype]
    else:
        renoster = vm[in_bioregion]

    logger.info(
        "vegmap renosterveld: {} polygons from {} bioregions (vegtype filter: {!r})",
        len(renoster),
        len(renosterveld_bioregions),
        vegtype_pattern or "none",
    )

    if renoster.empty:
        return np.zeros(nlc_shape, dtype=bool)

    renoster_nlc = renoster.to_crs(nlc_crs)
    shapes = [(mapping(g), 1) for g in renoster_nlc.geometry if g is not None]
    mask = rasterio.features.rasterize(
        shapes,
        out_shape=nlc_shape,
        transform=nlc_transform,
        fill=0,
        dtype=np.uint8,
    )
    n_px = int(mask.sum())
    logger.info(
        "renosterveld mask: {} pixels ({:.0f} km²)",
        n_px,
        n_px * 20 * 20 / 1e6,
    )
    return mask.astype(bool)


def _sample_class_pixels(
    rows: np.ndarray,
    cols: np.ndarray,
    transform: rasterio.transform.Affine,
    target: int,
    min_spacing_m: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Grid-thin and subsample pixel coordinates for one class.

    Returns an (N, 2) array of (x, y) in the NLC CRS.
    """
    pixel_size = abs(transform.a)
    cell_px = max(1, int(min_spacing_m / pixel_size))

    cell_r = (rows // cell_px).astype(np.int32)
    cell_c = (cols // cell_px).astype(np.int32)
    stride = int(cell_c.max() - cell_c.min()) + 1 if len(cell_c) else 1
    keys = (cell_r - cell_r.min()).astype(np.int64) * stride + (cell_c - cell_c.min())

    _, idx = np.unique(keys, return_index=True)
    rows, cols = rows[idx], cols[idx]
    logger.info(
        "  after {}m grid thin: {} points",
        int(min_spacing_m),
        len(rows),
    )

    if len(rows) > target:
        pick = rng.choice(len(rows), size=target, replace=False)
        rows, cols = rows[pick], cols[pick]
        logger.info("  subsampled to {} points", target)

    xs = transform.c + (cols + 0.5) * transform.a
    ys = transform.f + (rows + 0.5) * transform.e
    return np.column_stack([xs, ys])


def sample_nlc_points(
    nlc_path: str | Path,
    schema_path: str | Path = "configs/labels_schema.yaml",
    aoi_uri: str = "gs://ism-data/aoi/western_cape.parquet",
    vegmap_path: str | Path | None = None,
    root: str = WC_LABELS_ROOT,
    target_per_class: int = 2_500,
    min_spacing_m: float = 200.0,
    seed: int = 42,
    weight: float = 0.8,
) -> str:
    """Sample balanced, spatially-uniform points from the NLC 2022 raster.

    Uses vegmap renosterveld polygons to disambiguate NLC class 8 ("low
    shrubland other") from fynbos.  Writes to the unified observation store
    as ``source="nlc_sample"``.

    Parameters
    ----------
    nlc_path : path to SA_NLC_2022_ALBERS.tif
    schema_path : labels_schema.yaml with nlc_2022 crosswalk
    aoi_uri : WC AOI GeoParquet
    vegmap_path : NVM2024 shapefile (needed for renosterveld stratification)
    target_per_class : max samples per training class
    min_spacing_m : spatial thinning grid cell size
    seed : random seed for reproducible subsampling
    weight : observation weight in the store
    """
    schema = load_config(schema_path)
    aoi = read_gdf(aoi_uri)
    rng = np.random.default_rng(seed)
    run_id = make_run_id(_SOURCE)
    ingested_at = dt.datetime.now(tz=dt.UTC)

    nlc_to_class = _build_nlc_to_class(schema)
    arr, win_tf, nlc_crs = _read_nlc_windowed(nlc_path, aoi)

    # NLC class 8 needs vegmap stratification for renosterveld
    if vegmap_path is not None:
        renoster_mask = _build_renosterveld_mask(
            vegmap_path, aoi, schema, arr.shape, win_tf, nlc_crs
        )
    else:
        renoster_mask = np.zeros(arr.shape, dtype=bool)
        logger.warning("no vegmap path — skipping renosterveld stratification")

    # Build per-training-class pixel masks
    class_masks: dict[int, np.ndarray] = {}

    for nlc_val, class_id in nlc_to_class.items():
        if nlc_val == 8:
            continue  # handled separately via renosterveld mask
        mask = arr == nlc_val
        if class_id in class_masks:
            class_masks[class_id] = class_masks[class_id] | mask
        else:
            class_masks[class_id] = mask

    # Class 9 (renosterveld): NLC=8 AND vegmap confirms renosterveld
    renoster_pixels = (arr == 8) & renoster_mask
    if 9 in class_masks:
        class_masks[9] = class_masks[9] | renoster_pixels
    else:
        class_masks[9] = renoster_pixels

    to_wgs = Transformer.from_crs(nlc_crs, "EPSG:4326", always_xy=True)

    all_rows: list[dict] = []
    all_geoms: list[Point] = []

    for class_id in sorted(class_masks):
        mask = class_masks[class_id]
        n_px = int(mask.sum())
        class_name = _TRAINING_CLASS_NAMES.get(class_id, f"class_{class_id}")
        logger.info(
            "class {} ({}): {} candidate pixels ({:.0f} km²)",
            class_id,
            class_name,
            n_px,
            n_px * 20 * 20 / 1e6,
        )
        if n_px == 0:
            continue

        rows_px, cols_px = np.where(mask)
        coords = _sample_class_pixels(
            rows_px, cols_px, win_tf, target_per_class, min_spacing_m, rng
        )
        lons, lats = to_wgs.transform(coords[:, 0], coords[:, 1])

        for i in range(len(coords)):
            x_nlc, y_nlc = coords[i]
            r_orig = int(round((y_nlc - win_tf.f) / win_tf.e))
            c_orig = int(round((x_nlc - win_tf.c) / win_tf.a))
            r_orig = max(0, min(r_orig, arr.shape[0] - 1))
            c_orig = max(0, min(c_orig, arr.shape[1] - 1))
            nlc_val = int(arr[r_orig, c_orig])

            all_rows.append(
                {
                    "obs_id": f"nlc2022:{nlc_val}:{r_orig}:{c_orig}",
                    "source": _SOURCE,
                    "source_record_id": f"{nlc_val}:{r_orig}:{c_orig}",
                    "source_url": None,
                    "species": class_name,
                    "species_normalized": class_name,
                    "gbif_usage_key": None,
                    "nemba_category": None,
                    "geom_type": "point",
                    "coord_uncertainty_m": 10.0,
                    "event_date": dt.date(2022, 1, 1),
                    "basis_of_record": "NLC_RASTER_SAMPLE",
                    "cover_pct": None,
                    "weight": float(weight),
                    "ingested_at": ingested_at,
                    "ingest_run_id": run_id,
                    "aoi_admin1": "western_cape",
                }
            )
            all_geoms.append(Point(float(lons[i]), float(lats[i])))

    if not all_rows:
        logger.warning("no NLC points sampled — check schema crosswalk")
        return ""

    gdf = gpd.GeoDataFrame(all_rows, geometry=all_geoms, crs="EPSG:4326")
    logger.info("assembled {} NLC sample points across {} classes", len(gdf), len(class_masks))

    out_path = write_source_partition(gdf, _SOURCE, root=root, run_id=run_id)
    return out_path
