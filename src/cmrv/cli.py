"""cmrv root CLI — stage subcommands dispatched via tyro.

Phase 0 is local-first: all artifacts live under ``data/`` (see CLAUDE.md).
"""

from __future__ import annotations

import geopandas as gpd  # type: ignore
import tyro  # type: ignore
from loguru import logger  # type: ignore

from cmrv.aoi import SA_ALBERS, build_tile_grid, fetch_provinces
from cmrv.ingest.chips import (
    build_spatial_blocks,
    extract_training_chips,
    make_split,
    thin_labels,
)
from cmrv.io import load_config, read_gdf, write_gdf_parquet
from cmrv.labels.mapwaps import CATCHMENTS, ingest_mapwaps
from cmrv.labels.merge import load_training_labels
from cmrv.labels.niaps import ingest_niaps
from cmrv.labels.observations import PROCESSED_ROOT, write_summary
from cmrv.labels.sanlc import ingest_sanlc


def parse_bbox(s: str) -> tuple[float, float, float, float]:
    """Parse ``"min_lon,min_lat,max_lon,max_lat"`` → 4-float tuple, validated min<max."""
    try:
        vals = tuple(float(x) for x in s.split(","))
    except ValueError as e:
        raise ValueError(f"bbox must be comma-separated numbers, got {s!r}") from e
    if len(vals) != 4:
        raise ValueError(f"bbox needs 4 values (min_lon,min_lat,max_lon,max_lat), got {len(vals)}")
    min_lon, min_lat, max_lon, max_lat = vals
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError(f"bbox needs min_lon<max_lon and min_lat<max_lat, got {vals}")
    return vals


def aoi_sa(
    out: str = "data/aoi/processed/south_africa.parquet",
    source: str | None = None,
    buffer_m: float = 1000.0,
    simplify_m: float = 200.0,
    target_crs: str = "EPSG:4326",
) -> None:
    """Build a national South Africa AOI (all 9 provinces dissolved) → GeoParquet.

    The only AOI: training chips and the delivered map both use it. Boundary from
    GeoBoundaries gbOpen ADM1; drops the offshore Prince Edward Islands. Pass
    ``--source`` for a local file.
    """
    gdf = fetch_provinces(
        None, source=source, buffer_m=buffer_m, simplify_m=simplify_m, out_crs=target_crs
    )
    area_km2 = gdf.to_crs(SA_ALBERS).area.sum() / 1e6
    logger.info("South Africa AOI: {} feature, area = {:.0f} km^2", len(gdf), area_km2)
    write_gdf_parquet(gdf, out)
    logger.success("wrote {}", out)


def aoi_tiles(
    aoi: str | None = None,
    km: float = 10.0,
    out: str = "data/aoi/processed/tiles.parquet",
    crs: str = SA_ALBERS,
    pipeline: str = "configs/pipeline.yaml",
) -> None:
    """Build a square tile grid over the AOI and write as GeoParquet (inference unit).

    Grid CRS defaults to national equal-area SA Albers (true-square tiles country-wide).
    --aoi defaults to ``aoi.infer_path`` (the delivered map's extent, not the training one).
    """
    aoi = aoi or load_config(pipeline)["aoi"]["infer_path"]
    gdf = read_gdf(aoi)  # type: ignore
    tiles = build_tile_grid(gdf, tile_km=km, crs=crs)
    logger.info("built {} tiles of {} km", len(tiles), km)
    write_gdf_parquet(tiles, out)
    logger.success("wrote {}", out)


def labels_mapwaps_ingest(
    root: str = PROCESSED_ROOT,
    catchment: str | None = None,
) -> None:
    """Ingest MapWAPS field points → store (source=mapwaps), all catchments by default.

    Registered catchments: Olifants-Doring (WC), Tugela (KZN), uMzimvubu (EC).
    All mappable classes ingested — IAP genera (Alien_*), native biomes, transformed
    land cover — each crosswalked to a ``sa_landcover`` member (only Shade and
    Alien_Other are dropped). Pass ``--catchment <name>`` for one.
    Assign class_id at make-split via ``--class-map-name sa_landcover``.
    """
    keys = [catchment] if catchment else list(CATCHMENTS)
    for key in keys:
        ingest_mapwaps(key, root=root)
    logger.success("mapwaps ingest complete — {} catchment(s)", len(keys))


def labels_niaps_ingest(
    min_density: int = 50,
    min_area_ha: float = 1.0,
    erode_m: float = 20.0,
    root: str = PROCESSED_ROOT,
) -> None:
    """Distil NIAPS 2023 national IAP polygons → points in the store (source=niaps).

    **These are not field observations.** Kotze et al. (2025) extrapolated 47,830 field
    plots to all pixels by Sentinel-2 spectral matching, so training on them distils
    ARC's classifier. Rows carry weight=0.5 and basis_of_record=NIAPS_S2_EXTRAPOLATED;
    always report test metrics per source (`train-head` does).

    Filters, in cost order: density → area → make-valid/explode → negative buffer
    (also removes slivers) → drop cross-taxon overlaps → point_on_surface.
    --min-density: gridcode (percent cover) floor. --erode-m: purity margin.
    """
    path = ingest_niaps(
        min_density=min_density, min_area_ha=min_area_ha, erode_m=erode_m, root=root
    )
    logger.success("niaps ingest complete — {}", path)


def labels_sanlc_ingest(
    root: str = PROCESSED_ROOT,
    replace: bool = True,
) -> None:
    """Ingest SANLC 2018/2020/2022 accuracy-assessment points + VegMap 2024 → store.

    Field-verified land-cover reference points → our classes (natural points named
    by VegMap biome); identical points across years de-duplicated; known-IAP areas
    excluded. Feeds the unified ``sa_landcover`` class map at make-split.

    **Run this LAST.** The 320 m IAP-exclusion buffer is built from every species/genus
    row in the store, so it grows each time another source lands. ``--replace`` (on by
    default) rewrites the partition rather than upserting, so a re-run can actually
    *drop* points that are now inside the buffer. Pass ``--no-replace`` to merge instead.
    """
    path = ingest_sanlc(root=root, replace=replace)
    logger.success("sanlc ingest complete — {}", path)


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
    print(write_summary(root=root, out_uri=summary_out))
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


def ingest_chips(
    aoi: str | None = None,
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
    max_workers: int = 8,
    read_pool: int | None = None,
    max_dates: int | None = None,
) -> None:
    """Extract temporally-aligned training chips for label points (Stage 2b).

    Pipeline: load labels → **spatial-thin (before any imagery)** → group into
    spatial blocks → extract a 64x64 px (10 m) chip per (label, month). No fold
    assignment — that's done at training time via ``cmrv make-split``.

    Manifest-based incremental extraction — existing chips are skipped, so it's
    safe to re-run after adding a label source.

    One work unit is one observation. Per month window it searches the label's
    coarse cell (memoised), drops the dates whose footprints miss the chip, reads
    SCL alone to find the date clearest over *this* chip, then reads the bands for
    that date only. One uint16 GeoTIFF per (obs, year) holds every month as bands.

    --aoi: defaults to ``aoi.train_path`` (national SA — so KZN/EC labels aren't clipped).
    --block-km: spatial-block size in km (default 10; the CV unit, not a query batch).
    --thin-m: keep one label per species per thin-m cell, before download (default 20).
    --species: restrict to these species (by name fragment). Omit for all.
    --max-workers: chips in flight. Each also opens ``--read-pool`` asset reads,
                   so total in-flight sockets is the product. On a slow link keep
                   that product near a few dozen — past the point that saturates
                   the link, extra streams only stall each other.
    --read-pool: concurrent asset reads inside one chip (default from config).
    --max-dates: override ``max_dates_per_month`` — raise it if too many chips
                 drop on cloud, then re-chip everything rather than mixing.
    """
    cfg = load_config(pipeline)
    aoi = aoi or cfg["aoi"]["train_path"]

    labels = load_training_labels(
        aoi_uri=aoi,  # type: ignore
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

    # Region-aware months: each label's rainfall zone (from its province) picks its
    # month set at extraction time (winter- vs summer-rainfall phenology).
    # Refuse to guess a zone — silently defaulting an unlisted province to
    # winter_rainfall chips a summer-rainfall label on the wrong calendar, and
    # nothing downstream can tell. Cheapest possible moment to catch it.
    admin1_zone = cfg.get("admin1_zone", {})
    labels["_zone"] = labels["aoi_admin1"].map(admin1_zone)
    unzoned = labels[labels["_zone"].isna()]
    if not unzoned.empty:
        missing = sorted(unzoned["aoi_admin1"].fillna("<missing>").unique())
        raise ValueError(
            f"no rainfall zone for province(s) {missing} "
            f"({len(unzoned)} labels). Add them to `admin1_zone` in {pipeline} "
            f"(known: {sorted(admin1_zone)}) before chipping."
        )
    logger.info("label zones: {}", dict(labels["_zone"].value_counts()))

    # Thin BEFORE fetching imagery so we never download chips we'd discard.
    labels = thin_labels(labels, thin_m=thin_m)

    aoi_gdf = read_gdf(aoi)  # type: ignore
    blocks = build_spatial_blocks(aoi_gdf, block_km=block_km)

    blocks_wgs = blocks[["block_id", "geometry"]].to_crs("EPSG:4326")
    labels = gpd.sjoin(labels, blocks_wgs, how="inner", predicate="within")
    if "index_right" in labels.columns:
        labels = labels.drop(columns=["index_right"])
    logger.info("{} labels assigned to {} blocks", len(labels), labels["block_id"].nunique())

    manifest = extract_training_chips(
        labels=labels,
        months_by_zone=cfg["months_by_zone"],
        bands=cfg["s2_bands"],
        out_prefix=out_prefix,
        chip_px=cfg.get("chip_px", 128),
        max_dates=max_dates if max_dates is not None else cfg.get("max_dates_per_month", 1),
        min_coverage=cfg.get("min_coverage", 0.98),
        min_valid_frac=cfg.get("min_valid_frac", 0.85),
        screen_max_dates=cfg.get("screen_max_dates", 24),
        read_pool=read_pool if read_pool is not None else cfg.get("read_pool", 4),
        date_cell_m=cfg.get("date_cell_m", 2000.0),
        # A --species run only knows about its own subset, so it must not prune
        # everything else's chips as "no longer in the thinned set".
        reconcile=species is None,
        default_year=default_year,
        max_workers=max_workers,
    )
    logger.success(
        "ingest-chips complete — {} chips for {} labels",
        len(manifest),
        manifest["obs_id"].nunique() if not manifest.empty else 0,
    )


def chips_make_split(
    aoi: str | None = None,
    manifest: str = "data/chips/train/manifest.parquet",
    out_prefix: str = "data/chips/train",
    species: list[str] | None = None,
    class_map_name: str | None = None,
    schema_path: str = "configs/labels_schema.yaml",
    pipeline: str = "configs/pipeline.yaml",
    seed: int = 42,
    block_km: float = 10.0,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    min_class_obs: int = 0,
    lock_folds: bool = True,
) -> None:
    """Generate a reproducible spatial split from the chip manifest.

    Reads the manifest, optionally filters to a species subset, assigns spatial
    blocks to train/val/test folds via iterative stratification (whole blocks, no
    leakage; each class spread across folds), and writes split files. Obs with 1-3
    of the configured months are all kept (the temporal head masks missing months);
    thinning already happened at ``ingest-chips`` time.

    --species: species names (exact match) to include. Omit for all.
    --class-map-name: a class_maps entry in the schema YAML (e.g. "sa_landcover").
                      Adds a class_id column collapsing species to a shared class
                      (e.g. all Eucalyptus spp → class 5); the split is stratified on
                      class_id. Unmapped rows dropped unless --species is given.
    --min-class-obs: drop classes with fewer than N obs before splitting (0 = keep
                     all). Use for classes too rare to appear in every fold.
    --lock-folds: re-use existing block_folds.parquet assignments.
    --aoi: defaults to ``aoi.train_path`` — must match what ``ingest-chips`` used,
           or nationally-chipped labels get clipped out of the split.
    """
    aoi = aoi or load_config(pipeline)["aoi"]["train_path"]
    result = make_split(
        manifest_uri=manifest,
        aoi_uri=aoi,  # type: ignore
        species=species,
        class_map_name=class_map_name,
        schema_path=schema_path,
        seed=seed,
        block_km=block_km,
        train_frac=train_frac,
        val_frac=val_frac,
        min_class_obs=min_class_obs,
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
    print("=== Fold x species (obs_id counts) ===")
    sp_table = obs_only.groupby(["fold", "species"]).size().unstack(fill_value=0)
    sp_table = sp_table.reindex([f for f in fold_order if f in sp_table.index])
    sp_table["TOTAL"] = sp_table.sum(axis=1)
    print(sp_table.T.to_string())

    if "class_id" in obs_only.columns and obs_only["class_id"].notna().any():
        print()
        print("=== Fold x class_id (obs_id counts) ===")
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
    """Print species x spatial x temporal stats for a chip manifest.

    Reads ``manifest.parquet`` and reports total chips / obs_ids / species /
    extent, top-N species, month-completeness, densest blocks,
    spatially-dominated species, fold x species (if ``make-split`` has run),
    and obs_ids per chip year. No schema or class_map needed.
    """
    from cmrv.ingest.stats import chip_stats

    chip_stats(manifest_uri=manifest, top_species=top_species, top_blocks=top_blocks)


def embed(
    manifest: str = "data/chips/train/manifest.parquet",
    out: str = "data/embeddings/universat_center.zarr",
    output_grid: int | None = None,
    pipeline: str = "configs/pipeline.yaml",
    device: str = "cpu",
    batch: int = 2,
    num_workers: int = 4,
    amp: bool = False,
) -> None:
    """Embed training chips → UniverSat center-token vectors (single Zarr).

    One 768-d vector per obs. --output-grid defaults to ``chip_px`` from
    pipeline.yaml, which is the only value that keeps one token per 10 m pixel and
    so matches wall-to-wall inference. Do not set it lower to save time: it halves
    the map resolution instead. Center pooling = the per-location representation
    the frozen head replicates densely at inference. CRS-less + tiny, so it is a
    single Zarr regardless of source UTM zone. Needs the ``embed`` group.

    Resumable: vectors checkpoint to ``<out>.parts/`` every 500 obs, so after any stop
    (crash, reboot, Ctrl+C, pkill) re-run the same command and it continues.

    --device: ``cpu`` or ``cuda`` (cloud). --num-workers: chip-prefetch workers that
    overlap disk reads with the forward (raise on GPU to keep it fed). --amp: fp16/bf16
    autocast: a big GPU win, and 2.6x on an AMX CPU (Xeon 4th gen+, measured 4.11 ->
    1.56 s/chip at cosine 0.99998 to fp32). Leave it off on older CPUs, where it can
    be slower. --batch default 2: attention runs over the 32x32 latent grid and one
    upsampling pass makes the 128x128 output, so on CPU a bigger batch adds only RAM
    (5.2 GB at 2, 8.7 GB at 4) and neither batch nor thread count changes s/chip.
    Raise it on a GPU.
    """
    from cmrv.embeddings.embed import embed_chips
    from cmrv.embeddings.universat import UniverSatEmbedder

    if output_grid is None:
        output_grid = load_config(pipeline).get("chip_px", 128)
    enc = UniverSatEmbedder(
        pool="center", output_grid=output_grid, device=device, batch=batch, amp=amp
    )
    embed_chips(manifest, out, enc, batch=batch, num_workers=num_workers)


def train_head(
    emb: str = "data/embeddings/universat_center.zarr",
    split: str = "data/chips/train/split.parquet",
    arch: str = "linear",
    weight: str = "balanced",
    save: str | None = None,
) -> None:
    """Train a light head on frozen embeddings + report per-class test metrics.

    --arch: ``linear`` (the adopted head — beat MLP on the current set) or ``mlp``.
    --weight: ``balanced`` (N/(K·n_c)), ``sqrt`` (gentler), or ``none``. Computed
              live from the train fold, so it tracks label updates automatically.
    --save: checkpoint path (weights + mu/sd + class ids) for `cmrv infer`.
    """
    from cmrv.embeddings.head import train_head as _train

    per, macro = _train(emb, split, arch=arch, weight=weight, save=save)
    print(per.to_string(index=False))
    logger.success("{} head ({} CE): test macro-F1 = {:.3f}", arch, weight, macro)


def infer(
    bbox: str,
    ckpt: str = "data/runs/head_linear.pt",
    out: str = "data/outputs/infer_class.tif",
    year: int = 2023,
    device: str = "cpu",
    tta_views: int = 1,
) -> None:
    """Wall-to-wall per-pixel class map over a lon/lat box → COG.

    bbox = "min_lon,min_lat,max_lon,max_lat" — a single comma-separated string so
    negative latitudes aren't mistaken for CLI flags (e.g.
    ``--bbox=19.21,-33.20,19.25,-33.16``). 3-month composite → UniverSat dense tokens
    → frozen head per token (the center-token rep, applied to every token),
    overlap-blended into a seamless class/confidence/OOD COG. Needs the ``embed`` group
    + a saved head (``train-head --save``). --tta-views soft-averages augmented views
    (1 = off, 4 = rotations, 8 = full D4 flips+rotations); ~Nx slower.
    """
    from cmrv.infer import infer_box

    try:
        box = parse_bbox(bbox)
    except ValueError as e:
        raise SystemExit(f"infer: {e}") from None  # clean one-line error, no traceback
    infer_box(box, ckpt, out, year=year, device=device, tta_views=tta_views)


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
        enqueue=True,  # thread-safe writes from ThreadPoolExecutor workers
    )
    logger.info("log file: {}", log_path)

    # A stop is a normal way to end a long run, not a crash. Without this the
    # KeyboardInterrupt (raised by Ctrl+C, or by SIGTERM via _term_saves_work)
    # printed a full traceback over the log, which reads like a failure.
    try:
        _run_cli()
    except KeyboardInterrupt:
        logger.warning("stopped by request — finished work was saved")
        sys.exit(130)


def _run_cli() -> None:
    tyro.extras.subcommand_cli_from_dict(
        {
            "aoi-sa": aoi_sa,
            "aoi-tiles": aoi_tiles,
            "labels-mapwaps-ingest": labels_mapwaps_ingest,
            "labels-niaps-ingest": labels_niaps_ingest,
            "labels-sanlc-ingest": labels_sanlc_ingest,
            "labels": labels_inspect,
            "chips-stats": chips_stats,
            "ingest-chips": ingest_chips,
            "make-split": chips_make_split,
            "embed": embed,
            "train-head": train_head,
            "infer": infer,
        }
    )


if __name__ == "__main__":
    main()
