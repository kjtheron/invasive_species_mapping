# catchment-mrv

Phase 0 prototype for **Lens A — Invasive Alien Plant Intelligence**: a 2.5 m,
multi-season IAP species map for the Western Cape, with per-pixel uncertainty
and out-of-distribution novelty flags.

It is the beachhead of a larger platform (one shared spine, four verification
"lenses"). Phase 0 builds Lens A only — see [CLAUDE.md](CLAUDE.md).

- Conventions & architecture: [CLAUDE.md](CLAUDE.md)
- Current plan: [tasks/todo.md](tasks/todo.md)
- Lessons from prior corrections: [tasks/lessons.md](tasks/lessons.md)

Phase 0 is **local-first** — all artifacts live under `data/` (gitignored), no
cloud bucket. See [CLAUDE.md](CLAUDE.md) for the path layout.

## Quickstart

```bash
# 1. Toolchain (one-off)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Recreate the Python environment (.venv is not committed)
uv sync

# 3. AOI + tile grid (once)
uv run cmrv aoi-wc      # Western Cape province   → data/aoi/western_cape.parquet
uv run cmrv aoi-tiles   # 10 km UTM-34S tile grid → data/aoi/tiles.parquet
```

No secrets/credentials are needed — label data comes from local scientific
datasets (see below).

## Label pipeline

```
labels-bioscape-ingest  Load BioSCape field plots into the unified observation
                        store (data/labels/wc/obs/source=<src>/). One adapter per
                        scientific dataset — add labels-<dataset>-ingest as more land.
      ↓
labels                  Inspect the store: per-source counts + coord-uncertainty /
                        cover coverage. With --aoi/--species, preview the filtered
                        training labels.
      ↓
ingest-chips            Spatial-thin (before any download), then extract a 64×64 px
                        (10 m) chip per (obs_id, month) for survivors of the basic
                        filters (AOI clip, coord_uncertainty ≤ 40 m, date, point).
                        Writes manifest.parquet — the record of what got chipped.
      ↓
chips-stats             Species × spatial × temporal stats from the manifest.
                        Empirical — no schema, no class_map.
      ↓
make-split              Filter by --species (optional), assign train/val/test folds
                        via spatial blocks, assign class_id via --class-map.
```

## CLI reference

`uv run cmrv <verb>`.

| Verb | Why |
|---|---|
| `aoi-wc` | Build the Western Cape province polygon (the AOI). |
| `aoi-tiles` | Build the 10 km UTM-34S tile grid (inference unit). |
| `labels-bioscape-ingest` | BioSCape VegPlots (line + plot) → `source=bioscape_line/` + `source=bioscape_plot/`. |
| `labels` | Inspect the obs store (counts + coverage); `--aoi`/`--species` previews filtered labels, `--out` writes them. |
| `ingest-month` | MPC STAC → SCL mask → monthly median → per-tile 10 m COG (inference imagery). |
| `ingest-chips` | Thin, then 64×64 chips per (obs_id, month); manifest-based incremental resume. |
| `chips-stats` | Explore the chip manifest. |
| `make-split` | Spatial-block train/val/test split; `--class-map-name` assigns `class_id`. |

## Common workflows

```bash
# I just added a label source
uv run cmrv labels-bioscape-ingest
uv run cmrv ingest-chips       # incremental — only new obs get chipped
uv run cmrv chips-stats        # see what landed

# Train on a specific species set
uv run cmrv make-split \
    --species "Acacia mearnsii" "Acacia saligna" "Pinus pinaster" \
    --class-map-name western_cape_iap
```
