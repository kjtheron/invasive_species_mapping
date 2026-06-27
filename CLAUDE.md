# Project conventions — catchment-mrv

## What this is

One platform sold as **verification + intelligence** through four "lenses", all
sharing a single spine:

- **Lens A — Invasive Alien Plant Intelligence** ← *Phase 0 builds this, and only this.*
- Lens B — Rehabilitation & Closure Assurance (mining)
- Lens C — Deforestation-Free Verification (EUDR)
- Lens D — Biodiversity & Nature-Risk Intelligence (bioacoustics + camera traps)

The spine (build once, reused by every lens): **AOI/tiling → sensor ingest &
harmonization → foundation-model embeddings (UniverSat, native 10 m) → light
per-lens heads → per-pixel uncertainty + OOD verification → delivery (COGs,
report, viewer).** Each lens is "spine + a head + a metric + a report".

**Do not scope-creep into Lens B/C/D.** They are deferred. When in doubt, cut.

## Phase 0 goal

A 10 m, 3-timestep IAP species map for the Western Cape, with per-pixel
uncertainty and out-of-distribution novelty flags. Phase 0 is
**IAP-species-only** — native/background classes are deferred; "not-IAP" is an
OOD/threshold decision, not a trained class.

Week-by-week plan: [tasks/todo.md](tasks/todo.md). Accumulated lessons:
[tasks/lessons.md](tasks/lessons.md).

## Local-first storage layout

Phase 0 runs **locally** — no cloud bucket. All artifacts live under `data/`
(gitignored), not `~/.cache/`:

- `data/aoi/` — AOI polygons, tile grid
- `data/raw/` — S2 L2A 10 m COGs
- `data/labels/raw/<dataset>/` — raw downloads (untouched, gitignored)
- `data/labels/processed/<dataset>/` — unified observation store (partitioned Parquet, mirrors raw)
- `data/chips/train/` — 64×64 training chips + `manifest.parquet`
- `data/embeddings/` — UniverSat embedding Zarr cubes
- `data/runs/` — checkpoints, MLflow db
- `data/outputs/` — triplet COGs, reports

## Python & packaging

- **Always `uv`** — never `pip install`, never `conda`, never `poetry`. `uv add <pkg>` to add a dep; `uv sync` to install; `uv run <cmd>` to execute.
- The virtualenv lives at `.venv` (uv default). **It is not committed and is currently absent — recreate it with `uv sync`.**
- Python 3.12 pinned. `uv.lock` is committed; regenerate only via `uv lock --upgrade` or `uv add`.
- Run the CLI via `uv run cmrv <verb> ...`.

## Storage format rules

- **Vector / tabular → Parquet (GeoParquet for geometry).** No GeoJSON, no shapefile, no CSV in the pipeline. Read/write via `cmrv.io` (`read_gdf`, `write_gdf_parquet`).
- **Raster → Cloud-Optimized GeoTIFF (COG)**, validated with `rio-cogeo validate`. Write via `cmrv.io.write_cog`.
- **N-d arrays (embeddings) → Zarr v3** with chunks matching the DataLoader (`(1, 256, 256, D)`).

## Consolidated IO helpers (`cmrv.io`)

All shared IO lives in `src/cmrv/io.py` — no module duplicates these:
`load_config`, `list_parquet_files`, `open_raster`, `write_cog`,
`read_parquet_df`/`write_parquet_df`, `read_gdf`/`write_gdf_parquet`. pandas +
geopandas/pyogrio do all Parquet I/O (no DuckDB). `read_gdf` decodes the WKB +
`__crs__` GeoParquet this module writes, falling back to native GeoParquet.

## Label sources

Training signal comes from **field datasets that measure cover/density**
(occurrence points without abundance — GBIF/iNat — are dropped; a lone-tree GPS
point is a mixed 10 m pixel). Raw downloads live in `data/labels/raw/<dataset>/`;
one **adapter per scientific dataset** writes the unified schema to
`data/labels/processed/<dataset>/` (mirrors raw):

1. **BioSCape VegPlots** (`cmrv labels-bioscape-ingest`) — Berg+Eerste field
   plots, per-species cover %. **Species-level** labels (`taxon_rank=species`).
2. **MapWAPS Olifants-Doring** (`cmrv labels-mapwaps-ingest`) — ~28k field points
   (NW WC); **genus/functional** labels (Alien_Pine/Gum/Wattle/Prosopis),
   `Density___` → `cover_pct`. Geometry is already distance/direction corrected.
3. Future datasets — add a sibling `labels-<dataset>-ingest` verb (a small loader
   mapping the source's columns → the schema).

Sources differ in label granularity, so each row carries `taxon_rank`. Phase 0
trains the **genus** class map `western_cape_iap_genus` (Acacia→`acacia_spp`);
the species map + a hierarchical-loss head is the later upgrade.

Every observation carries provenance + quality fields: `source`, `source_url`,
`source_doi`, `license`, `coord_uncertainty_m`, `cover_pct`. Training gates on
`coord_uncertainty_m` (≤40 m at chip time) and, once enabled, on `cover_pct`
(the pure-pixel gate — `load_training_labels(min_cover_pct=…)`).

**The species list is a single source of truth: `class_maps.<name>.members[]` in
[configs/labels_schema.yaml](configs/labels_schema.yaml).** Append a binomial to
a class's `members[]` and both the runtime species→class lookup and each
adapter's IAP membership derive from it. `class_id` is never assigned at ingest —
it's applied at `make-split` time via `--class-map`.

## AOI boundary data

The AOI is the **Western Cape province** (scales to SA later by dissolving more
provinces). Boundary auto-downloaded from **GeoBoundaries gbOpen ADM1** (CC-BY 4.0)
by `cmrv aoi-wc` — no manual download. Cleaned on ingest: make-valid, drop the
offshore Prince Edward Islands, simplify vertices (`--simplify-m`), +1 km buffer;
cached at `data/aoi/geoBoundaries-ZAF-ADM1.geojson`. Pass `--source <file>` for a
local boundary instead.

## Pipeline verbs

`uv run cmrv <verb>`. Label flow:

```
labels-bioscape-ingest → BioSCape field plots into the obs store (one adapter per dataset)
labels-mapwaps-ingest  → MapWAPS Olifants-Doring field points into the obs store
labels                 → inspect the store (per-source counts + coverage); optional AOI/species preview
ingest-chips           → thin → 64×64 chip per (obs_id, month); writes manifest.parquet
chips-stats            → species × spatial × temporal stats from the manifest
make-split             → spatial-block train/val/test split + --class-map → class_id
```

Per-tile imagery + downstream: `aoi-wc` / `aoi-tiles` (once), `ingest-month`
(S2 composites for inference), then SR → embed → train → infer (later stages).
Months for the chip stack are **Feb/May/Sep** (WC phenology — see `pipeline.yaml`).

## Common tripwires

- **CRS drift.** Vector labels in WGS84, rasters in UTM 34S (EPSG:32734), embeddings in pixel coords. Always declare CRS; convert via `rioxarray.reproject_match`. Tile/block grids are built in UTM, never WGS84.
- **Output grid.** UniverSat ingests native 10 m chips; `output_grid` sets the dense token resolution, decoupled from input. Center-token embedding for point labels; full dense grid for wall-to-wall inference.
- **Month availability.** Some tiles have 0 valid scenes in a month — pass a `month_mask` so the temporal head ignores missing months; don't drop the tile.
- **Label leakage.** Build spatial-block splits *before* training; never train on a block overlapping a held-out one.
- **Silent fp16 NaNs.** The embedding model + AMP can emit NaN on bad input — assert `torch.isfinite(...)` after each forward.
- **No distributed dask.** `stackstac`'s internal dask is enough; one box.

## Working style

- Plan-first: non-trivial tasks get a `tasks/todo.md` update before code.
- One corrected mistake → one new rule in `tasks/lessons.md`.
- Don't mark a task done until verified (tests green, artifact exists, behaviour demonstrated).
- Don't scope-creep — defer Lens B/C/D, multi-catchment, production API, native classes.
