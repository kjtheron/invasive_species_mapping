"""cmrv root CLI — stage subcommands dispatched via tyro.

Phase 0 is local-first: all artifacts live under ``data/`` (see CLAUDE.md).
"""

from __future__ import annotations

import geopandas as gpd
import tyro
from loguru import logger

from cmrv.aoi import SA_PROVINCIAL_SHP, build_tile_grid, fetch_western_cape
from cmrv.ingest.chips import (
    build_spatial_blocks,
    extract_training_chips,
    make_split,
    thin_labels,
)
from cmrv.ingest.composite import load_pipeline_config, run_ingest
from cmrv.io import read_gdf, write_gdf_parquet
from cmrv.labels.bioscape import ingest_lineintercept, ingest_plotcoverage
from cmrv.labels.mapwaps import ingest_mapwaps
from cmrv.labels.merge import load_training_labels, merge_partitions
from cmrv.labels.observations import PROCESSED_ROOT


def aoi_wc(
    out: str = "data/aoi/western_cape.parquet",
    source: str = str(SA_PROVINCIAL_SHP),
    buffer_m: float = 1000.0,
    target_crs: str = "EPSG:4326",
) -> None:
    """Extract the Western Cape province polygon, buffer, and write as GeoParquet.

    Source shapefile from waterresourceswr2012.co.za (free registration required).
    Scaling to SA later = dissolve all provinces (same machinery, bigger polygon).
    """
    gdf = fetch_western_cape(source=source, buffer_m=buffer_m, out_crs=target_crs)
    area_km2 = gdf.to_crs("EPSG:32734").area.sum() / 1e6
    logger.info("Western Cape AOI: {} feature, area = {:.0f} km^2", len(gdf), area_km2)
    write_gdf_parquet(gdf, out)
    logger.success("wrote {}", out)


def aoi_tiles(
    aoi: str = "data/aoi/western_cape.parquet",
    km: float = 10.0,
    out: str = "data/aoi/tiles.parquet",
    crs: str = "EPSG:32734",
) -> None:
    """Build a square tile grid over the AOI and write as GeoParquet (inference unit)."""
    gdf = read_gdf(aoi)
    tiles = build_tile_grid(gdf, tile_km=km, crs=crs)
    logger.info("built {} tiles of {} km in {}", len(tiles), km, crs)
    write_gdf_parquet(tiles, out)
    logger.success("wrote {}", out)


def labels_bioscape_ingest(
    schema: str = "configs/labels_schema.yaml",
    class_map: str = "western_cape_iap",
    root: str = PROCESSED_ROOT,
    iap_only: bool = True,
) -> None:
    """Ingest BioSCape VegPlots (Berg+Eerste) → unified observation store.

    Writes ``source=bioscape_line`` + ``source=bioscape_plot`` partitions.
    IAP membership decided from the class-map ``members[]``. CSV paths default
    to the ORNL DAAC archive layout under
    ``data/labels/raw/BioSCape_VegPlots_Berg_Eerste_2425/``.

    One adapter per scientific dataset — add a sibling ``labels-<dataset>-ingest``
    verb for each new source, all emitting the same observation schema.
    """
    line_path = ingest_lineintercept(
        schema_path=schema, class_map_name=class_map, root=root, iap_only=iap_only
    )
    plot_path = ingest_plotcoverage(
        schema_path=schema, class_map_name=class_map, root=root, iap_only=iap_only
    )
    logger.success("bioscape ingest complete — line={} plot={}", line_path, plot_path)


def labels_mapwaps_ingest(
    root: str = PROCESSED_ROOT,
) -> None:
    """Ingest MapWAPS Olifants-Doring training points → store (source=mapwaps).

    Keeps all 23 LULC classes; IAP labels are genus-level (Alien_Pine/Gum/Wattle/
    Prosopis). Crosswalk at make-split via ``--class-map-name western_cape_iap_genus``.
    Geometry used as-is (already distance/direction corrected upstream).
    """
    path = ingest_mapwaps(root=root)
    logger.success("mapwaps ingest complete — {}", path)


def labels_inspect(
    aoi: str | None = None,
    species: list[str] | None = None,
    out: str | None = None,
    root: str = PROCESSED_ROOT,
    summary_out: str = "data/labels/processed/summary.parquet",
    max_coord_uncertainty_m: float = 500.0,
    date_min: str = "2018-01-01",
) -> None:
    """Inspect the observation store.

    Always prints per-source counts + coord-uncertainty/cover coverage and
    writes ``summary.parquet``. With ``--aoi`` (and optional ``--species``) it
    also prints a filtered training-label preview; ``--out`` writes that
    filtered GeoParquet.
    """
    print(merge_partitions(root=root, summary_uri=summary_out))
    logger.success("summary → {}", summary_out)

    if not aoi:
        return
    gdf = load_training_labels(
        aoi_uri=aoi,
        root=root,
        species_subset=species,
        max_coord_uncertainty_m=max_coord_uncertainty_m,
        date_min=date_min,
    )
    logger.success(
        "filtered preview: {} rows, {} sources, {} species",
        len(gdf),
        gdf["source"].nunique() if not gdf.empty else 0,
        gdf["species_normalized"].nunique() if not gdf.empty else 0,
    )
    if out and not gdf.empty:
        write_gdf_parquet(gdf, out)
        logger.success("wrote → {}", out)


def ingest_month(
    month: str | None = None,
    tile_id: int | None = None,
    pipeline: str = "configs/pipeline.yaml",
) -> None:
    """Download S2 L2A composites → 10 m COGs (Stage 2 — inference imagery).

    Queries Microsoft Planetary Computer (no subscription key required),
    applies SCL cloud masking, computes monthly pixel-wise median, and writes
    a Cloud-Optimized GeoTIFF to ``<raw_prefix>/tile_id=<N>/<month_label>.tif``.

    --month: label from pipeline.yaml (e.g. ``feb``). Omit to run all months.
    --tile-id: single tile to process. Omit to run all tiles.
    """
    cfg = load_pipeline_config(pipeline)
    uris = run_ingest(cfg, month_label=month, tile_id=tile_id)
    logger.success("ingest-month complete — {} COG(s) written", len(uris))
    for u in uris:
        logger.info("  {}", u)


def ingest_chips(
    aoi: str = "data/aoi/western_cape.parquet",
    pipeline: str = "configs/pipeline.yaml",
    out_prefix: str = "data/chips/train",
    root: str = PROCESSED_ROOT,
    block_km: float = 10.0,
    thin_m: float = 20.0,
    max_coord_uncertainty_m: float = 40.0,
    date_min: str = "2018-01-01",
    date_max: str = "2025-12-31",
    default_year: int = 2023,
    species: list[str] | None = None,
    max_workers: int = 6,
) -> None:
    """Extract temporally-aligned training chips for label points (Stage 2b).

    Pipeline: load labels → **spatial-thin (before any imagery)** → group into
    spatial blocks → extract a 64×64 px (10 m) chip per (label, month). No fold
    assignment — that's done at training time via ``cmrv make-split``.

    Manifest-based incremental extraction — existing chips are skipped, so it's
    safe to re-run after adding a label source.

    --block-km: spatial-block size in km (default 10; STAC-query batching + CV unit).
    --thin-m: keep one label per species per thin-m cell, before download (default 20).
    --species: restrict to these species (by name fragment). Omit for all.
    """
    cfg = load_pipeline_config(pipeline)

    labels = load_training_labels(
        aoi_uri=aoi,
        root=root,
        max_coord_uncertainty_m=max_coord_uncertainty_m,
        date_min=date_min,
        date_max=date_max,
        species_subset=species,
        geom_types=["point"],
    )
    if labels.empty:
        logger.warning("no labels found — nothing to extract")
        return

    # Thin BEFORE fetching imagery so we never download chips we'd discard.
    labels = thin_labels(labels, thin_m=thin_m)

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
    aoi: str = "data/aoi/western_cape.parquet",
    manifest: str = "data/chips/train/manifest.parquet",
    out_prefix: str = "data/chips/train",
    species: list[str] | None = None,
    class_map_name: str | None = None,
    schema_path: str = "configs/labels_schema.yaml",
    seed: int = 42,
    block_km: float = 10.0,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    lock_folds: bool = True,
) -> None:
    """Generate a reproducible spatial split from the chip manifest.

    Reads the manifest, optionally filters to a species subset, assigns spatial
    blocks to train/val/test folds, and writes split files. Obs with 1–3 of the
    configured months are all kept (the temporal head masks missing timesteps);
    thinning already happened at ``ingest-chips`` time.

    --species: species names (exact match) to include. Omit for all.
    --class-map-name: a class_maps entry in the schema YAML (e.g. "western_cape_iap").
                      Adds a class_id column collapsing species to a shared class
                      (e.g. all Eucalyptus spp → class 5). Unmapped rows dropped
                      unless --species is given.
    --lock-folds: re-use existing block_folds.parquet assignments.
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
    sp_table = obs_only.groupby(["fold", "species"]).size().unstack(fill_value=0)
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
    manifest: str = "data/chips/train/manifest.parquet",
    top_species: int = 30,
    top_blocks: int = 10,
) -> None:
    """Print species × spatial × temporal stats for a chip manifest.

    Reads ``manifest.parquet`` and reports total chips / obs_ids / species /
    extent, top-N species, month-completeness, densest blocks,
    spatially-dominated species, fold × species (if ``make-split`` has run),
    and obs_ids per chip year. No schema or class_map needed.
    """
    from cmrv.chips.stats import chip_stats

    chip_stats(manifest_uri=manifest, top_species=top_species, top_blocks=top_blocks)


def main() -> None:
    tyro.extras.subcommand_cli_from_dict(
        {
            "aoi-wc": aoi_wc,
            "aoi-tiles": aoi_tiles,
            "labels-bioscape-ingest": labels_bioscape_ingest,
            "labels-mapwaps-ingest": labels_mapwaps_ingest,
            "labels": labels_inspect,
            "chips-stats": chips_stats,
            "ingest-month": ingest_month,
            "ingest-chips": ingest_chips,
            "make-split": chips_make_split,
        }
    )


if __name__ == "__main__":
    main()
