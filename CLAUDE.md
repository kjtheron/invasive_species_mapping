# Project conventions — catchment-mrv Phase 0

Primary source-of-truth: [Phase_0_Build_Roadmap.md](Phase_0_Build_Roadmap.md). Week-by-week plan: [tasks/todo.md](tasks/todo.md). Accumulated lessons: [tasks/lessons.md](tasks/lessons.md) (create on first correction).

## Python & packaging

- **Always `uv`** — never `pip install` directly, never `conda`, never `poetry`. `uv add <pkg>` to add a dep; `uv sync` to install; `uv run <cmd>` to execute inside the venv.
- Virtualenv lives at `.venv` (uv default) — do not commit it, do not rename it.
- Python 3.12 pinned in `pyproject.toml`.
- `uv.lock` is committed. Regenerate only via `uv lock --upgrade` or `uv add`.
- Run the CLI via `uv run cmrv <stage> ...` or the `just` recipe that wraps it.

## Task runner

- `just` recipes in `justfile` are the canonical entry points. Prefer `just <recipe>` over ad-hoc commands so everything is reproducible.
- `just default` = `just lint test`. Run before declaring any stage done.

## Google Cloud

- **Project**: `focus-vim-493513-r1`
- **Bucket**: `gs://ism-data` (region: `northamerica-northeast1`, Montréal)
- **Vertex AI region**: `northamerica-northeast1` — must match the bucket to avoid egress.
- **Demo host**: Cloud Run (scales to zero).

All pipeline artifacts live in GCS, not local disk. Path convention:
- `gs://ism-data/aoi/` — AOI polygons, tile grids
- `gs://ism-data/raw/` — S2 L2A COGs
- `gs://ism-data/sr/` — SEN2SR 2.5 m COGs
- `gs://ism-data/labels/` — fused label rasters + unified observation store (`wc/obs/source=<source>/`)
- `gs://ism-data/chips/train/` — 64×64 training chips + manifest.parquet
- `gs://ism-data/embeddings/` — Clay Zarr cubes
- `gs://ism-data/pca/` — IncrementalPCA artifacts
- `gs://ism-data/runs/` — Lightning checkpoints, MLflow db
- `gs://ism-data/outputs/` — triplet COGs, reports

## Authentication

- **Local dev**: Application Default Credentials via `gcloud auth application-default login`. `gcsfs` and `google-cloud-storage` read ADC automatically. GDAL `/vsigs/` does **not** — `cmrv.io.configure_gdal_gcs()` extracts the OAuth2 triple from ADC and sets `GS_OAUTH2_REFRESH_TOKEN`/`GS_OAUTH2_CLIENT_ID`/`GS_OAUTH2_CLIENT_SECRET` so streaming works. It is invoked automatically by `cmrv.io.open_raster()` — always use that helper, never `rasterio.open("/vsigs/...")` directly. **Never** commit service-account JSON keys.
- **Vertex AI / Cloud Run**: attach a least-privilege service account with `storage.objectAdmin` on `ism-data`. Workload Identity preferred over JSON keys.
- **Secrets** (PC subscription key, HF token, etc.): local dev uses `.env` (gitignored); prod uses Google Secret Manager. Reference `.env.example` for required vars.

## Data sourcing rule

Always prefer South African sources first: SAEON, SANBI (BGIS, Vegmap, SAPIA, POSA), SANSA, DWS, DFFE, NGI, SAWS, ARC, CapeNature, Elsenburg, City of Cape Town open data. Only fall back to global (MPC Sentinel-2, Copernicus DEM) when no SA equivalent exists or coverage is insufficient.

## AOI boundary data

Quaternary catchment boundaries and SA provincial boundaries are from **waterresourceswr2012.co.za** (free registration required). Files live in `data/aoi/`:
- `SA_Catchm_Quaternary.shp` — DWS quaternary catchments (field: `QUATERNARY`, CRS: EPSG:4148)
- `SA_Provincial_bnd_dd.shp` — SA provincial boundaries (field: `PROVINCE`, CRS: EPSG:4148)

HydroBASINS and geoBoundaries are no longer used — all AOI work uses the SA-specific layers above.

## Storage format rules

- **Vector / tabular → Parquet (GeoParquet for geometry)**. No GeoJSON, no shapefile, no CSV in the pipeline. Single file by default — **partition only when size justifies it** (rough trigger: >100 MB or >1M rows). Read/write via `cmrv.io` (`read_gdf`, `write_gdf_parquet`) which handles both local and `gs://`.
- **Raster → Cloud-Optimized GeoTIFF (COG)**. Validated with `rio-cogeo validate`.
- **N-d arrays (embeddings) → Zarr v3** with chunks matching the DataLoader (`(1, 256, 256, 768)` per roadmap §5).
- Caches, downloads, and scratch artifacts live in `<project>/data/` (gitignored), not `~/.cache/`.

## Consolidated IO helpers (`cmrv.io`)

All shared IO patterns live in `src/cmrv/io.py` — no module should duplicate these:

- **`load_config`** — single YAML loader (local or `gs://` via fsspec).
- **`_duckdb_con`** — single DuckDB connection factory with GCS credential chain. All modules use this; no duplicate factories.
- **`list_parquet_files`** — partition discovery (`*.parquet`, excludes `_tmp_` dirs). Replaces ad-hoc walk/listing in individual modules.
- **`write_cog`** — single COG writer (temp file + `cog_translate` + upload). Both `composite.py` and `fuse.py` delegate here.
- **`read_gdf` / `write_gdf_parquet`** — DuckDB-based GeoParquet streaming (geometry as WKB + `__crs__`).
- **`open_raster`** — streaming raster reads via GDAL `/vsigs/`.

## Chip extraction & splitting

Extraction and splitting are **decoupled** — different concerns, different commands:

- **`cmrv ingest-chips`** — extracts 64×64 px chips (10m) per label × 4 months from MPC STAC. Writes chips to `gs://ism-data/chips/train/{obs_id}/{year}/{month}.tif` and a fold-free `manifest.parquet`. Manifest-based incremental resume: re-running skips already-chipped obs_ids. Labels must have `block_id` (assigned via spatial join to 20km blocks in CLI).
- **`cmrv make-split`** — runs at training time. Reads manifest, filters by species + min_months, spatially thins (default 20m = Clay patch footprint at 2.5m GSD), assigns blocks to train/val/test folds via stratified greedy algorithm with seed-controlled tie-breaking, tags boundary blocks for leakage diagnostics. Different seeds produce different splits for ensemble uncertainty estimation.

## One-time setup commands

These CLI subcommands produce reference data and only need to run once (unless the source is updated):

- `cmrv labels-nemba-extract` — parses NEMBA Gazette PDF → `data/labels/nemba_plants.parquet`
- `cmrv labels-nemba-resolve` — resolves NEMBA taxa via GBIF API → `gs://ism-data/labels/nemba_taxa_resolved.parquet`

## Common tripwires

- Use DuckDB for streaming I/O (via `cmrv.io` helpers); pandas is fine for in-memory label processing.
- Raster/vector CRS: always declare, always convert via `rioxarray.reproject_match`. Tile grid work happens in UTM 34S (EPSG:32734), not WGS84.
- Zarr chunks match DataLoader sampling: `(1, 256, 256, 768)` — see roadmap §5.
- Every COG must pass `rio-cogeo validate`.
- No dask cluster. `stackstac`'s internal dask is enough.

## Working style (also in ~/.claude/CLAUDE.md, reinforced here)

- Plan-first: non-trivial tasks get a `tasks/todo.md` update before code.
- One corrected mistake → one new rule in `tasks/lessons.md`.
- Don't mark tasks done until verified (tests green, artifact exists, behaviour demonstrated).
- Don't scope-creep the roadmap — defer Phase 1 ambitions (U-Net student, multi-catchment, prod API, SAM).
