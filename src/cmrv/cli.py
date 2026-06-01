"""cmrv root CLI — stage subcommands dispatched via tyro."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
import tyro
from loguru import logger

from cmrv.aoi import (
    BERG_UPPER_CODES,
    DWS_QUATERNARY_SHP,
    SA_PROVINCIAL_SHP,
    build_tile_grid,
    fetch_western_cape,
    select_quaternaries_from_file,
)
from cmrv.ingest.chips import (
    build_spatial_blocks,
    extract_training_chips,
    flag_burned_labels,
    make_split,
)
from cmrv.ingest.composite import load_pipeline_config, run_ingest
from cmrv.io import ensure_parent, load_config, read_gdf, write_gdf_parquet
from cmrv.labels.fuse import (
    DEFAULT_OUT_PREFIX,
    DEFAULT_TILES_URI,
    RESOLUTION_M,
    fuse_all,
    fuse_tile,
    label_class_counts,
)
from cmrv.labels.gbif import ingest_gbif, resolve_taxa
from cmrv.labels.merge import load_training_labels, merge_partitions
from cmrv.labels.nemba import (
    NEMBA_OUT_LOCAL,
    NEMBA_PRIMARY_PDF,
    NEMBA_RESOLVED_GCS,
    resolve_nemba_taxa,
    write_nemba_plants,
)
from cmrv.labels.nlc import sample_nlc_points
from cmrv.labels.observations import WC_LABELS_ROOT
from cmrv.labels.vegmap import ingest_vegmap


def aoi_fetch(
    out: str = "gs://ism-data/aoi/berg_upper.parquet",
    source: str = str(DWS_QUATERNARY_SHP),
    codes: list[str] | None = None,
    field: str = "QUATERNARY",
    target_crs: str = "EPSG:4326",
) -> None:
    """Select quaternary catchments from the DWS shapefile and write as GeoParquet.

    Source shapefile from waterresourceswr2012.co.za (free registration required).
    """
    codes = codes or list(BERG_UPPER_CODES)
    gdf = select_quaternaries_from_file(source, codes=codes, field=field, out_crs=target_crs)
    area_km2 = gdf.to_crs("EPSG:32734").area.sum() / 1e6
    logger.info("Fetched {} feature(s), total area = {:.1f} km^2", len(gdf), area_km2)
    write_gdf_parquet(gdf, out)
    logger.success("wrote {}", out)


def aoi_wc(
    out: str = "gs://ism-data/aoi/western_cape.parquet",
    source: str = str(SA_PROVINCIAL_SHP),
    buffer_m: float = 1000.0,
    target_crs: str = "EPSG:4326",
) -> None:
    """Extract the Western Cape province polygon, buffer, and write as GeoParquet.

    Source shapefile from waterresourceswr2012.co.za (free registration required).
    """
    gdf = fetch_western_cape(source=source, buffer_m=buffer_m, out_crs=target_crs)
    area_km2 = gdf.to_crs("EPSG:32734").area.sum() / 1e6
    logger.info("Western Cape AOI: {} feature, area = {:.0f} km^2", len(gdf), area_km2)
    write_gdf_parquet(gdf, out)
    logger.success("wrote {}", out)


def aoi_tiles(
    aoi: str = "gs://ism-data/aoi/berg_upper.geojson",
    km: float = 10.0,
    out: str = "gs://ism-data/aoi/tiles.parquet",
    crs: str = "EPSG:32734",
) -> None:
    """Build a square tile grid over the AOI and write as GeoParquet."""
    gdf = read_gdf(aoi)
    tiles = build_tile_grid(gdf, tile_km=km, crs=crs)
    logger.info("built {} tiles of {} km in {}", len(tiles), km, crs)
    write_gdf_parquet(tiles, out)
    logger.success("wrote {}", out)


def labels_nemba_extract(
    pdf: str = str(NEMBA_PRIMARY_PDF),
    out: str = str(NEMBA_OUT_LOCAL),
    lists: list[int] | None = None,
) -> None:
    """Parse NEMBA AIS plant list from gazette PDF → local parquet.

    Lists 1 (terrestrial+freshwater plants) and 2 (marine plants) are
    extracted by default; pass --lists 1 to restrict to terrestrial only.
    """
    include = tuple(lists) if lists else (1, 2)
    write_nemba_plants(pdf_path=pdf, out_path=out, include_lists=include)


def labels_nemba_resolve(
    plants: str = str(NEMBA_OUT_LOCAL),
    out: str = NEMBA_RESOLVED_GCS,
    pause_s: float = 0.3,
) -> None:
    """Resolve NEMBA plant names → GBIF backbone usage keys and write to GCS.

    Requires network access. Reads the local parquet written by
    labels-nemba-extract; writes resolved rows to ``out`` (GCS or local).
    Target: ≥95% of taxa matched.
    """
    resolve_nemba_taxa(plants_parquet=plants, out_uri=out, pause_s=pause_s)


def labels_gbif_resolve(
    schema: str = "configs/labels_schema.yaml",
    class_map: str = "upper_berg_12",
    out: str = "data/labels/gbif_taxa_resolved.parquet",
) -> None:
    """Resolve GBIF taxa (vernacular → scientific → usage_key) and cache to parquet.

    Taxa list is derived from ``class_maps.<class_map>.members[]``; falls back
    to the legacy ``gbif.taxa`` block if no members[] are present.
    Run this first to inspect the resolution before kicking off the full download.
    """
    from cmrv.io import write_parquet_df
    from cmrv.labels.classmap import gbif_taxa_from_schema

    cfg = load_config(schema)
    taxa = gbif_taxa_from_schema(schema, class_map)
    resolved = resolve_taxa(taxa, cfg.get("vernacular_map", {}))
    logger.info("resolved {} of {} taxa", len(resolved), len(taxa))
    ensure_parent(out)
    write_parquet_df(pd.DataFrame(resolved), out)
    logger.success("wrote {}", out)


def labels_vegmap_ingest(
    shp: str = "data/vegmap/NVM2024final_Shapefile/NVM2024Final_IEM5_12_07012025.shp",
    aoi: str = "gs://ism-data/aoi/western_cape.parquet",
) -> None:
    """Clip SANBI NVM2024 IEM5 to WC AOI → unified observation store.

    Emits ``source=vegmap`` rows with polygon geometry and no class_id.
    Run ``cmrv aoi-wc`` first to ensure the WC AOI parquet exists.
    """
    ingest_vegmap(shp_path=shp, aoi_uri=aoi)


def labels_ingest(
    source: str = "all",
    aoi: str = "gs://ism-data/aoi/western_cape.parquet",
    nemba_resolved: str = "gs://ism-data/labels/nemba_taxa_resolved.parquet",
    schema: str = "configs/labels_schema.yaml",
    class_map: str = "upper_berg_12",
    shp: str = "data/vegmap/NVM2024final_Shapefile/NVM2024Final_IEM5_12_07012025.shp",
    root: str = WC_LABELS_ROOT,
) -> None:
    """Ingest one or all label sources into the unified WC observation store.

    ``--source`` accepts: ``gbif``, ``vegmap``, ``all``.
    (bioscape requires its own CSV paths; use ``labels-bioscape-ingest`` for now.)

    Requires GBIF_USER / GBIF_PASS / GBIF_EMAIL for the ``gbif`` source.
    Run ``cmrv labels-nemba-resolve`` first to populate ``nemba_resolved``.
    """
    sources = {"gbif", "vegmap"} if source == "all" else {source}

    if "gbif" in sources:
        gdf = ingest_gbif(
            aoi_uri=aoi,
            nemba_resolved_uri=nemba_resolved,
            schema_path=schema,
            class_map_name=class_map,
            root=root,
        )
        logger.success(
            "gbif ingest complete — {} total records ({} gbif / {} inat)",
            len(gdf),
            (gdf["source"] == "gbif").sum() if not gdf.empty else 0,
            (gdf["source"] == "inat_via_gbif").sum() if not gdf.empty else 0,
        )

    if "vegmap" in sources:
        ingest_vegmap(shp_path=shp, aoi_uri=aoi, root=root)


def labels_merge(
    root: str = WC_LABELS_ROOT,
    summary_out: str = "gs://ism-data/labels/wc/summary.parquet",
) -> None:
    """Union all source partitions, dedup on obs_id, write summary.parquet.

    Prints per-source × per-category row counts + coord_uncertainty coverage.
    """
    df = merge_partitions(root=root, summary_uri=summary_out)
    logger.success("summary written → {}", summary_out)
    print(df)


def labels_sample(
    aoi: str = "gs://ism-data/aoi/berg_upper.parquet",
    root: str = WC_LABELS_ROOT,
    out: str | None = None,
    max_coord_uncertainty_m: float = 500.0,
    date_min: str = "2018-01-01",
) -> None:
    """Load filtered training labels for an AOI and print a summary.

    Thin wrapper on ``load_training_labels`` for CLI-driven exploration.
    Pass ``--out <path>`` to write the resulting GeoParquet to disk.
    """
    gdf = load_training_labels(
        aoi_uri=aoi,
        root=root,
        max_coord_uncertainty_m=max_coord_uncertainty_m,
        date_min=date_min,
    )
    logger.success(
        "labels_sample: {} rows, {} sources, {} species",
        len(gdf),
        gdf["source"].nunique() if not gdf.empty else 0,
        gdf["species_normalized"].nunique() if not gdf.empty else 0,
    )
    if out and not gdf.empty:
        write_gdf_parquet(gdf, out)
        logger.success("wrote → {}", out)


def labels_fuse(
    tiles: str = DEFAULT_TILES_URI,
    schema: str = "configs/labels_schema.yaml",
    class_map: str = "upper_berg_12",
    root: str = WC_LABELS_ROOT,
    out_prefix: str = DEFAULT_OUT_PREFIX,
    resolution_m: float = RESOLUTION_M,
    tile_id: int | None = None,
    class_floor: int = 200,
) -> None:
    """Rasterize label observations → per-tile sparse label COGs (VIZ ONLY).

    NOT a training input.  Training (chip / point regime) consumes the
    observation store directly via ``ingest-chips`` + ``make-split``;
    ``label.tif`` rasters are only emitted as a QGIS / Streamlit overlay
    so you can spot-check coverage and conflict resolution.

    If ``--tile-id`` is given, fuse only that tile.  Otherwise fuse every
    tile in the grid (``--tiles``).

    After writing, reports per-class pixel counts for each COG.  Warns if
    any class falls below ``--class-floor`` pixels (default 200).
    """
    if tile_id is not None:
        tiles_gdf = read_gdf(tiles)
        if "tile_id" not in tiles_gdf.columns:
            tiles_gdf["tile_id"] = range(len(tiles_gdf))
        row = tiles_gdf[tiles_gdf["tile_id"] == tile_id]
        if row.empty:
            raise ValueError(f"tile_id={tile_id} not found in {tiles}")
        uris = [
            fuse_tile(
                tile_id=tile_id,
                tile_row=row.iloc[0],
                schema_path=schema,
                class_map_name=class_map,
                labels_root=root,
                out_prefix=out_prefix,
                resolution_m=resolution_m,
            )
        ]
    else:
        uris = fuse_all(
            tiles_uri=tiles,
            schema_path=schema,
            class_map_name=class_map,
            labels_root=root,
            out_prefix=out_prefix,
            resolution_m=resolution_m,
        )

    for uri in uris:
        try:
            counts = label_class_counts(uri)
            for cid, n in sorted(counts.items()):
                flag = " ⚠ BELOW FLOOR" if n < class_floor else ""
                logger.info("  class {:>2d}: {:>8,} px{}", cid, n, flag)
        except Exception as e:
            logger.warning("could not read counts from {}: {}", uri, e)

    logger.success("labels-fuse complete — {} tile(s) written", len(uris))


def labels_nlc_sample(
    nlc: str = "data/landuse/SA_NLC_2022_ALBERS/SA_NLC_2022_ALBERS.tif",
    schema: str = "configs/labels_schema.yaml",
    aoi: str = "gs://ism-data/aoi/western_cape.parquet",
    vegmap: str = "data/vegmap/NVM2024final_Shapefile/NVM2024Final_IEM5_12_07012025.shp",
    target_per_class: int = 2_500,
    min_spacing_m: float = 200.0,
    seed: int = 42,
    root: str = WC_LABELS_ROOT,
) -> None:
    """Sample balanced non-IAP training labels from SA NLC 2022 raster.

    Reads the NLC raster windowed to the WC AOI, crosswalks NLC pixel values
    to training classes (fynbos, renosterveld, indigenous forest, other
    landcover) via labels_schema.yaml, and spatially-thins to balanced
    per-class samples. Vegmap 2024 polygons disambiguate renosterveld from
    other low shrubland.

    --nlc: path to SA_NLC_2022_ALBERS.tif (20m, uint8, Albers SA CRS).
    --vegmap: NVM2024 shapefile for renosterveld stratification.
    --target-per-class: max samples per training class (default 8000).
    --min-spacing-m: spatial thinning grid cell size (default 200m).
    """
    out = sample_nlc_points(
        nlc_path=nlc,
        schema_path=schema,
        aoi_uri=aoi,
        vegmap_path=vegmap,
        root=root,
        target_per_class=target_per_class,
        min_spacing_m=min_spacing_m,
        seed=seed,
    )
    if out:
        logger.success("labels-nlc-sample complete → {}", out)
    else:
        logger.warning("labels-nlc-sample produced no output")


def ingest_month(
    month: str | None = None,
    tile_id: int | None = None,
    pipeline: str = "configs/pipeline.yaml",
) -> None:
    """Download S2 L2A composites → 10 m COGs on GCS (Stage 2).

    Queries Microsoft Planetary Computer (no subscription key required),
    applies SCL cloud masking, computes monthly pixel-wise median, and writes
    a Cloud-Optimized GeoTIFF to ``<raw_prefix>/tile_id=<N>/<month_label>.tif``.

    --month: label from pipeline.yaml (e.g. ``oct``). Omit to run all active months.
    --tile-id: single tile to process. Omit to run all tiles.
    --pipeline: path to pipeline.yaml (default: configs/pipeline.yaml).
    """
    cfg = load_pipeline_config(pipeline)
    uris = run_ingest(cfg, month_label=month, tile_id=tile_id)
    logger.success("ingest-month complete — {} COG(s) written", len(uris))
    for u in uris:
        logger.info("  {}", u)


def ingest_chips(
    aoi: str = "gs://ism-data/aoi/western_cape.parquet",
    pipeline: str = "configs/pipeline.yaml",
    out_prefix: str = "gs://ism-data/chips/train",
    root: str = WC_LABELS_ROOT,
    block_km: float = 20.0,
    max_coord_uncertainty_m: float = 40.0,
    date_min: str = "2018-01-01",
    date_max: str = "2025-12-31",
    default_year: int = 2023,
    species: list[str] | None = None,
    nemba_categories: list[str] | None = None,
    max_workers: int = 6,
) -> None:
    """Extract temporally-aligned training chips for label points (Stage 2b).

    Builds spatial blocks over the AOI, then extracts 64x64 px (10 m) chips
    per label per month.  No fold assignment — splitting is done at training
    time via ``cmrv make-split``.

    Uses manifest-based incremental extraction — existing chips are skipped
    automatically. Safe to re-run after adding new label sources.

    Fire filter is source-aware: IAP labels (gbif, inat_via_gbif, bioscape)
    are always fire-filtered; non-IAP labels (nlc_sample, vegmap_native)
    always skip it — fynbos and other native cover types burn naturally.

    --aoi: AOI for label filtering and block grid.
    --block-km: spatial block size in km (default 20).
    --species: restrict to these species (by name fragment). Omit for all.
    --nemba-categories: restrict to these NEMBA categories (e.g. 1a 1b).
    """
    cfg = load_pipeline_config(pipeline)

    labels = load_training_labels(
        aoi_uri=aoi,
        root=root,
        max_coord_uncertainty_m=max_coord_uncertainty_m,
        date_min=date_min,
        date_max=date_max,
        species_subset=species,
        nemba_categories=nemba_categories,
        geom_types=["point"],
    )
    if labels.empty:
        logger.warning("no labels found — nothing to extract")
        return

    non_iap_sources = {"nlc_sample", "vegmap_native"}
    is_iap = ~labels["source"].isin(non_iap_sources)
    iap_labels = labels[is_iap]
    non_iap_labels = labels[~is_iap]

    iap_n_before = len(iap_labels)
    iap_labels = flag_burned_labels(iap_labels)
    iap_labels = iap_labels[~iap_labels["burned"]].drop(columns=["burned"])
    logger.info(
        "fire filter (IAP, always on): removed {} of {} labels",
        iap_n_before - len(iap_labels),
        iap_n_before,
    )

    if len(non_iap_labels):
        logger.info(
            "fire filter (non-IAP): skipped for {} labels — native cover types burn naturally",
            len(non_iap_labels),
        )

    labels = gpd.GeoDataFrame(
        pd.concat([iap_labels, non_iap_labels], ignore_index=True),
        crs=labels.crs,
    )

    aoi_gdf = read_gdf(aoi)
    blocks = build_spatial_blocks(aoi_gdf, block_km=block_km)

    blocks_wgs = blocks[["block_id", "geometry"]].to_crs("EPSG:4326")
    labels = gpd.sjoin(labels, blocks_wgs, how="inner", predicate="within")
    if "index_right" in labels.columns:
        labels = labels.drop(columns=["index_right"])
    logger.info("{} labels assigned to {} blocks", len(labels), labels["block_id"].nunique())

    manifest = extract_training_chips(
        labels=labels,
        blocks=blocks,
        months_cfg=cfg["months"],
        bands=cfg["s2_bands"],
        out_prefix=out_prefix,
        cloud_cover_max=cfg.get("cloud_cover_max", 40),
        default_year=default_year,
        max_workers=max_workers,
    )
    logger.success(
        "ingest-chips complete — {} chips for {} labels",
        len(manifest),
        manifest["obs_id"].nunique() if not manifest.empty else 0,
    )


def chips_make_split(
    aoi: str = "gs://ism-data/aoi/western_cape.parquet",
    manifest: str = "gs://ism-data/chips/train/manifest.parquet",
    out_prefix: str = "gs://ism-data/chips/train",
    species: list[str] | None = None,
    class_map_name: str | None = None,
    schema_path: str = "configs/labels_schema.yaml",
    seed: int = 42,
    block_km: float = 20.0,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    min_months: int = 4,
    thin_m: float = 20.0,
    lock_folds: bool = True,
) -> None:
    """Generate a reproducible spatial split from the chip manifest.

    Reads the manifest, optionally filters to a species subset, drops
    labels with fewer than ``min-months`` temporal chips, spatially thins
    near-duplicates, assigns spatial blocks to train/val/test folds,
    and writes split files.
    Run this before training — decoupled from chip extraction so the same
    chips support different experiments.

    --species: species names (exact match) to include. Omit for all.
    --class-map-name: name of a class_maps entry in the schema YAML (e.g.
                      "upper_berg_12"). Adds a class_id column by collapsing
                      multiple species to a shared training class (e.g. all
                      Eucalyptus spp → class 5, Pinus spp → class 4).
                      Rows with no class_id mapping are dropped.
    --seed: random seed for reproducible splits.
    --min-months: minimum month-chips per label (default 4 = all months).
    --thin-m: thinning grid cell size in metres (0 to disable). Default 20m
              matches Clay v1.5 patch footprint at 2.5m SEN2SR resolution.
    --lock-folds: if True, re-use existing block_folds.parquet assignments.
    """
    result = make_split(
        manifest_uri=manifest,
        aoi_uri=aoi,
        species=species,
        class_map_name=class_map_name,
        schema_path=schema_path,
        seed=seed,
        block_km=block_km,
        train_frac=train_frac,
        val_frac=val_frac,
        min_months=min_months,
        thin_m=thin_m,
        out_prefix=out_prefix,
        lock_folds=lock_folds,
    )
    logger.success(
        "make-split complete — {} obs_ids across {} species, {} classes",
        result["obs_id"].nunique(),
        result["species"].nunique(),
        result["class_id"].nunique() if "class_id" in result.columns else "n/a",
    )

    # --- Balance summary ---
    obs_only = result.drop_duplicates(subset=["obs_id"])
    fold_order = ["train", "val", "test"]

    print()
    print("=== Fold × species (obs_id counts) ===")
    sp_table = (
        obs_only.groupby(["fold", "species"]).size().unstack(fill_value=0)
    )
    sp_table = sp_table.reindex([f for f in fold_order if f in sp_table.index])
    sp_table["TOTAL"] = sp_table.sum(axis=1)
    print(sp_table.T.to_string())

    if "class_id" in obs_only.columns and obs_only["class_id"].notna().any():
        print()
        print("=== Fold × class_id (obs_id counts) ===")
        cls_table = (
            obs_only.dropna(subset=["class_id"])
            .assign(class_id=lambda d: d["class_id"].astype(int))
            .groupby(["fold", "class_id"])
            .size()
            .unstack(fill_value=0)
        )
        cls_table = cls_table.reindex([f for f in fold_order if f in cls_table.index])
        cls_table["TOTAL"] = cls_table.sum(axis=1)
        print(cls_table.T.to_string())

        print()
        print("=== Class composition (species → class_id) ===")
        comp = (
            obs_only.dropna(subset=["class_id"])
            .assign(class_id=lambda d: d["class_id"].astype(int))
            .groupby(["class_id", "species"])
            .size()
            .reset_index(name="n_obs")
            .sort_values(["class_id", "n_obs"], ascending=[True, False])
        )
        for cid, grp in comp.groupby("class_id"):
            members = ", ".join(f"{r.species}={r.n_obs}" for r in grp.itertuples())
            print(f"  class {cid}: {members}")

        unmapped_obs = obs_only[obs_only["class_id"].isna()]
        if len(unmapped_obs):
            print()
            print(
                f"=== Unmapped (kept, no class_id): {len(unmapped_obs)} obs_ids "
                "— class_map labelling skipped ==="
            )
            unmapped_sp = unmapped_obs.groupby("species").size().sort_values(ascending=False)
            print(unmapped_sp.head(15).to_string())


def chips_stats(
    manifest: str = "gs://ism-data/chips/train/manifest.parquet",
    top_species: int = 30,
    top_blocks: int = 10,
) -> None:
    """Print species × spatial × temporal stats for a chip manifest.

    Reads ``manifest.parquet`` from a chip-extraction run and reports:

    * total chips / unique obs_ids / unique species / spatial extent
    * top-N species by obs_id count, with chips-per-obs and block coverage
    * cumulative coverage (top-10 / top-30) and long-tail count
    * month-completeness histogram per obs_id
    * densest spatial blocks
    * spatially-dominated species (>50% of obs_ids in a single block)
    * fold × species table (only if ``make-split`` has populated ``fold``)
    * obs_ids per chip year

    No schema or class_map needed — the manifest is the source of truth for
    what got chipped.  Class assignment happens later in ``make-split`` for
    the species you actually want to train on.
    """
    from cmrv.chips.stats import chip_stats

    chip_stats(manifest_uri=manifest, top_species=top_species, top_blocks=top_blocks)


def main() -> None:
    import sys
    from datetime import datetime
    from pathlib import Path

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    # One file per invocation: logs/cmrv_<subcommand>_<timestamp>.log
    subcommand = next((a for a in sys.argv[1:] if not a.startswith("-")), "unknown")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"cmrv_{subcommand}_{timestamp}.log"
    logger.add(
        str(log_path),
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
        enqueue=True,   # thread-safe writes from ThreadPoolExecutor workers
    )
    logger.info("log file: {}", log_path)

    tyro.extras.subcommand_cli_from_dict(
        {
            "aoi-fetch": aoi_fetch,
            "aoi-wc": aoi_wc,
            "aoi-tiles": aoi_tiles,
            "labels-nemba-extract": labels_nemba_extract,
            "labels-nemba-resolve": labels_nemba_resolve,
            "labels-gbif-resolve": labels_gbif_resolve,
            "labels-vegmap-ingest": labels_vegmap_ingest,
            "labels-ingest": labels_ingest,
            "labels-merge": labels_merge,
            "labels-sample": labels_sample,
            "labels-fuse": labels_fuse,
            "chips-stats": chips_stats,
            "labels-nlc-sample": labels_nlc_sample,
            "ingest-month": ingest_month,
            "ingest-chips": ingest_chips,
            "make-split": chips_make_split,
        }
    )


if __name__ == "__main__":
    main()
