# Mapping alien invasive trees across South Africa

Wattle, pine, gum and prosopis have spread across South African catchments to the point where they measurably reduce river flow — invasive alien plants cost the country an estimated billions of rand a year in lost water, and clearing programmes need to know *where the trees actually are* before they can spend a cent well.

This project builds that map: a **10 m resolution, multi-season map of invasive alien tree genera**, derived from Sentinel-2 satellite imagery and trained on field survey data. Every pixel carries not just a prediction but a **confidence score** and a **novelty flag** — so a user can tell the difference between "this is wattle", "this might be wattle", and "this doesn't look like anything the model was trained on".

That last part matters more than it sounds. A map that is confidently wrong is worse than no map, because someone budgets against it.

## What it produces

For any area of interest, a three-band Cloud-Optimized GeoTIFF:

| Band | Contents |
|---|---|
| `class_id` | Predicted genus — acacia, pinus, eucalyptus, hakea, prosopis, populus — or native / transformed land cover |
| `confidence` | Per-pixel model confidence |
| `ood` | Out-of-distribution score: how unlike the training data this pixel is |

Resolution is 10 m — roughly a tennis court — fine enough to pick out riparian invasion corridors along river lines, where the water cost is highest.

## How it works

```
area of interest
      ↓
Sentinel-2 imagery, three phenology-tuned seasons per year
      ↓                   (timed so evergreen aliens stand out against
      ↓                    the native vegetation's seasonal cycle)
foundation-model embeddings  (UniverSat, native 10 m)
      ↓
lightweight classification head trained on field survey labels
      ↓
per-pixel class + uncertainty + novelty  →  COG, report, viewer
```

The seasonal timing does real work. In the winter-rainfall Western Cape the model looks at February (peak dry summer, when evergreen pine and gum stand out against senesced fynbos), May, and September (spring green-up, when wattle flowers bright yellow). In the summer-rainfall provinces the calendar flips to July / September / December. Same machinery, different phenology.

## Training data

The model learns from **field surveys that measure cover or density**, not from opportunistic occurrence records. A GPS point marking a single tree tells you almost nothing about a 10 m pixel that is mostly something else, so presence-only records are deliberately excluded.

| Source | What it contributes |
|---|---|
| **MapWAPS** | ~36k field points across Olifants-Doring (WC), Tugela (KZN) and uMzimvubu (EC) — alien genera plus native and transformed land cover |
| **SANLC + VegMap** | National land-cover accuracy-assessment points and biome boundaries — the native and transformed classes |

Sources name things at different levels of precision — MapWAPS records "Alien Wattle", not *Acacia mearnsii* — so every observation carries its taxonomic rank and the model trains at genus level, where the sources agree. BioSCape VegPlots (Berg + Eerste, the only species-level source) was removed on 2026-08-12: the pre-embargo release carried 83 plots and no cover data. It returns when the full release lands.

## Repository layout

Data artifacts live under `data/`:

| Path | Contents |
|---|---|
| `data/aoi/` | Area-of-interest boundaries and tile grids |
| `data/labels/raw/` · `processed/` | Source downloads, and the unified observation store |
| `data/chips/train/` | 64×64 training chips + manifest |
| `data/embeddings/` | Embedding cubes (Zarr) |
| `data/outputs/` | Output maps and reports |

Vector data is GeoParquet, rasters are COGs, arrays are Zarr.

## Getting started

```bash
# Toolchain (one-off)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Python environment (.venv is not committed)
uv sync

# Area of interest + tile grid
uv run cmrv aoi-sa       # national boundary — the training extent
uv run cmrv aoi-tiles    # 10 km tile grid (the inference unit)
```

No API keys or credentials needed. Boundaries download automatically from GeoBoundaries (CC-BY 4.0); imagery comes from the Microsoft Planetary Computer's open Sentinel-2 archive.

## Pipeline

```
labels-*-ingest   Field datasets → unified observation store.
                  One adapter per source, all emitting the same schema.
      ↓
labels            Inspect the store: per-source counts, coordinate
                  uncertainty and cover coverage.
      ↓
ingest-chips      Spatially thin, then extract a 64×64 px (10 m) chip per
                  (observation, month). Incremental and resumable.
      ↓
chips-stats       Species × spatial × temporal breakdown of what was chipped.
      ↓
make-split        Spatial-block train/val/test split (whole blocks, so no
                  leakage between folds) + class assignment.
      ↓
embed → train-head → infer
```

### Command reference

`uv run cmrv <verb>`

| Verb | Purpose |
|---|---|
| `aoi-sa` | Build the national South Africa boundary. |
| `aoi-tiles` | Build the tile grid used as the inference unit. |
| `labels-mapwaps-ingest` | MapWAPS field points across three catchments. |
| `labels-sanlc-ingest` | SANLC accuracy points + VegMap biomes. |
| `labels` | Inspect the observation store; preview filtered labels. |
| `ingest-chips` | Extract training chips; resumable, self-reconciling. |
| `chips-stats` | Explore the chip manifest. |
| `make-split` | Spatial-block split with class assignment. |
| `embed` | Foundation-model embeddings → Zarr cube. |
| `train-head` | Train the classification head; report per-class metrics. |
| `infer` | Wall-to-wall map → class / confidence / novelty COG. |

### Common workflows

```bash
# A new label source just landed
uv run cmrv labels-mapwaps-ingest
uv run cmrv ingest-chips        # incremental — only new observations get chipped
uv run cmrv chips-stats

# Build the split (alien genera + native biomes + transformed cover)
uv run cmrv make-split --class-map-name sa_landcover
```

Class definitions — which genera and cover types roll up into which class — live in [configs/labels_schema.yaml](configs/labels_schema.yaml). Each adapter decides what it *ingests* from its own vocabulary dict, and ingest warns by name about anything the class map cannot place.

## Licence and attribution

Boundary data from [GeoBoundaries](https://www.geoboundaries.org) (CC-BY 4.0).
Sentinel-2 imagery courtesy of ESA / Copernicus via the Microsoft Planetary Computer. 
Field datasets retain their original licences, recorded per observation in the store.
