# Geospatial pipeline to classify alien invasive tree species within South Africa

## catchment-mrv

Phase 0 prototype: a 2.5 m, 4-timestep invasive alien plant (IAP) + native-class map for a slice of the Western Cape, with per-pixel uncertainty and out-of-distribution novelty flags.

- Build plan: [Phase_0_Build_Roadmap.md](Phase_0_Build_Roadmap.md)
- Tooling conventions: [CLAUDE.md](CLAUDE.md)
- Current week's plan: [tasks/todo.md](tasks/todo.md)
- Lessons from prior corrections: [tasks/lessons.md](tasks/lessons.md)

## Pipeline architecture (read this first)

The pipeline is **four verbs**: `ingest → chip → stats → split`.

```
labels-ingest         Pull a label source (GBIF, BioSCape, NLC, vegmap, …)
                      into the unified observation store
                          gs://ism-data/labels/wc/obs/source=<src>/
                      Names are standardized via GBIF backbone at this point.
        ↓
ingest-chips          Extract a 64×64 px (10 m) chip per (obs_id, month) for
                      every observation that survives basic filters (AOI clip,
                      coord_uncertainty ≤ 40 m, date range, geom_type=point).
                      Writes manifest.parquet — the empirical record of what
                      got chipped.  No class assignment here.
        ↓
chips-stats           Print species × spatial × temporal stats from the
                      manifest.  Use this for exploration: "what's in my
                      chips, is it balanced, is anything spatially dominated?"
        ↓
make-split            Filter the manifest by --species (optional), assign
                      train/val/test folds via spatial blocks, optionally
                      assign class_id via --class-map.  This is the only
                      stage that touches the class crosswalk.
```

Two side-helpers, neither on the critical path:

| Verb | Purpose |
|---|---|
| `labels-fuse` | **Viz only.** Per-tile sparse label COG for QGIS / Streamlit overlay. Training never reads this. |
| `labels-merge` | Print per-source × per-category row counts + coord-uncertainty coverage on the obs store. |

## Quickstart

```bash
# 1. Toolchain (one-off)
curl -LsSf https://astral.sh/uv/install.sh | sh
curl -sSL https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-linux-x86_64.tar.gz \
  | tar -xzC ~ && ~/google-cloud-sdk/install.sh --quiet --path-update true

# 2. GCP auth (one-off, opens browser)
gcloud auth login
gcloud auth application-default login
gcloud auth application-default set-quota-project focus-vim-493513-r1
gcloud config set project focus-vim-493513-r1

# 3. Sync Python environment
uv sync

# 4. Run Stage 1 end-to-end
just aoi-fetch    # WC quaternary catchments → gs://ism-data/aoi/
just tiles        # 10 km tile grid       → gs://ism-data/aoi/tiles.parquet
```

All pipeline artifacts live in `gs://ism-data/` — see [CLAUDE.md](CLAUDE.md) for the path convention.

## CLI reference

`uv run cmrv <verb>` (or `just <recipe>` for the common ones).

### AOI / tile setup — once per AOI

| Verb | When to run | Why |
|---|---|---|
| `aoi-fetch` | Once. | Pull SA quaternary catchments + provincial boundaries. |
| `aoi-wc` | Once. | Build the Western Cape province polygon. |
| `aoi-tiles` | Once after `aoi-wc`. | Build the 10 km UTM-34S tile grid that drives all per-tile stages. |

### Labels — per source

| Verb | When to run | Why |
|---|---|---|
| `labels-nemba-extract` | Once (NEMBA Gazette PDF parse). | Get the canonical NEMBA AIS plant list. |
| `labels-nemba-resolve` | Once after extract. | Resolve NEMBA taxa → GBIF backbone usage keys (≥95% match). |
| `labels-vegmap-ingest` | Once per Vegmap release. | Clip SANBI NVM2024 polygons to WC AOI → `source=vegmap` partition. |
| `labels-gbif-resolve` | Inspect-before-download (optional). | Resolve `class_maps.<name>.members[]` taxa → GBIF usage keys; cache as parquet so you can verify resolutions before the full ingest. |
| `labels-ingest --source gbif` | Whenever you want fresh GBIF / iNat occurrences. | Triggers GBIF Download API jobs and writes `source=gbif/` + `source=inat_via_gbif/` partitions. |
| `labels-ingest --source vegmap` | Same as `labels-vegmap-ingest`. | Convenience wrapper. |
| `labels-ingest --source all` | Both of the above in one go. | |
| `labels-nlc-sample` | Once (or after NLC release). | Sample balanced points from SA NLC 2022 raster → `source=nlc_sample`. |
| `labels-bioscape-ingest` | Once (BioSCape VegPlots, held-out eval). | High-quality field plots; flagged as held-out — never trained on. |
| `labels-merge` | Anytime, after ingestion changes. | Union all source partitions, dedup on `obs_id`, print per-source counts. Sanity check that ingestion produced what you expected. |

### Chips — extract, explore, split

| Verb | When to run | Why |
|---|---|---|
| `ingest-chips` | After labels are in the obs store; re-run any time a new source lands. | Manifest-based incremental — already-chipped (obs_id, month) pairs are skipped. Chips *every* survivor of basic filters; no schema or class concerns at this stage. |
| `chips-stats` | After `ingest-chips`, and any time you want to understand the training data. | Source of truth for "what's in my chips". Prints species counts, top-species cumulative coverage, month-completeness, densest blocks, spatially-dominated species (split-risk), per-year obs counts. **No schema, no class_map** — purely empirical. |
| `make-split` | Once per training experiment. | Filter manifest → spatial-block stratified split → optionally assign `class_id` from `class_maps.<name>` for labelling. With `--species [list]` you keep unmapped species (warn-only); without `--species` and with `--class-map`, unmapped are dropped. |

### Per-tile pipeline (Stage 5+)

| Verb | When to run | Why |
|---|---|---|
| `ingest-month` | Stage 2 of the per-tile pipeline. | MPC STAC → SCL mask → monthly median → per-tile COG. |
| `labels-fuse` | **Optional, viz only.** | Per-tile sparse label COG (`label.tif`) for QGIS / Streamlit overlay. Training never reads it; the chip / point regime consumes manifest + obs store directly. |

### Useful `just` recipes

```bash
just default          # = lint + test
just labels-ingest    # all sources
just chips-stats      # explore chip manifest
just labels-merge     # per-source obs-store summary
just labels-fuse      # viz raster (only if you want overlays)
```

## Common workflows

### "I just added a new label source"

```bash
# 1. Ingest it
uv run cmrv labels-ingest --source <new_src>

# 2. Chip everything new (incremental — only new obs get chipped)
uv run cmrv ingest-chips

# 3. See what landed
uv run cmrv chips-stats
```

### "I want to train on a specific species set"

```bash
# Filter manifest to your species, assign class_id via class_map, write splits
uv run cmrv make-split \
    --species "Acacia mearnsii" "Acacia saligna" "Pinus pinaster" \
    --class-map upper_berg_12 \
    --out-prefix gs://ism-data/chips/train
```

### "I want to know what's spatially imbalanced"

```bash
uv run cmrv chips-stats
# Look for:
#   - "Spatially-dominated species" section (>50% of obs in one block → split risk)
#   - "Top N densest blocks" (concentration → over-fit risk)
#   - Long-tail count (species with <50 obs → data-poor for training)
```

## Conventions

- **Always `uv`** — never `pip install`, never `conda`, never `poetry`.
- **Always `just`** for pipeline entry points so commands stay reproducible.
- All artifacts in `gs://ism-data/`; see [CLAUDE.md](CLAUDE.md) for the path schema.
- Vector / tabular → Parquet (GeoParquet for geometry); raster → COG; embeddings → Zarr v3.
- Class crosswalk lives only in `configs/labels_schema.yaml → class_maps.<name>.<id>.members[]` — single source of truth.
