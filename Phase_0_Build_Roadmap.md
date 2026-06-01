# Phase 0 — Catchment MRV Build Roadmap

**Audience:** A second coding agent (or a collaborator) who will implement the Phase 0 prototype end-to-end.
**Goal:** Ship a working 2.5 m, 4-timestep IAP + native-class map of a ~500 km² slice of the Berg River catchment, with per-pixel uncertainty + OOD novelty flags, in 12 calendar weeks.
**Philosophy:** Minimal code. Best-in-class packages. Lazy I/O everywhere. No premature abstractions. Prefer deleting code to writing it.

This roadmap is the *build plan*. Scientific rationale and DoD live in `Phase_0_Wedge_Prototype.md`; read that first.

---

## 0. Repo Strategy — One or Two?

**Use one monorepo for Phase 0.** Name it `catchment-mrv`.

Reasons:
- Data engineering (STAC → SR → embeddings → Zarr) and inference (embeddings → temporal head → triplet COG) share the same Zarr schema, the same Clay weights, and the same coordinate reference frame. A split duplicates this.
- You are one person. A second repo doubles CI, release, and environment surface area.
- The embedding cube is the *boundary* between the two halves. That boundary is a Zarr path, not a package boundary.

**Split in Phase 1**, not Phase 0, when:
- Inference moves behind an API with different SLAs and dependencies (FastAPI + ONNX Runtime).
- A second biome / catchment pipeline spawns and data-engineering needs its own release cadence.
- An ML student needs the training half without the ingest half.

Phase 0 `catchment-mrv/` is a single Python package with clear stage boundaries inside `src/cmrv/`, each stage runnable as a Typer CLI subcommand. This is lighter than two repos but lets you carve off `src/cmrv/ingest` and `src/cmrv/inference` into separate packages later without a rewrite.

---

## 1. Tech Stack — Chosen for Speed and Simplicity

| Concern | Pick | Why |
|---|---|---|
| Package manager | **`uv`** (Astral) | 10–100× faster than pip; lockfile; venv built in. No poetry, no conda. |
| Python version | **3.12** | Fast. Wide wheel coverage for geo stack. |
| STAC query | **`pystac-client`** | Only way to talk to Microsoft Planetary Computer and Element84 Earth Search. |
| Lazy raster stacks | **`stackstac`** | xarray-native, dask-lazy, handles reprojection. Preferred over `odc-stac` for Phase 0 because of the simpler API. |
| Raster I/O | **`rasterio`** + **`rioxarray`** | Standard. GDAL under the hood. |
| Vector | **`geopandas` ≥1.0** + **`pyogrio`** | Arrow-backed reads are 10× faster than fiona. |
| Array store | **`zarr` v3** + **`numcodecs`** with Blosc-Zstd | Chunked, cloud-ready, fast random reads for tile sampling. |
| DataFrames | **`DuckDB`** + **`pandas`** | DuckDB for streaming I/O; pandas for in-memory processing. |
| Config | **`pydantic-settings`** + **`tyro`** | Typed CLI configs without Hydra ceremony. |
| Training | **`torch` 2.5+** + **`lightning` 2.4** | Lightning handles AMP, checkpointing, logging. No manual loops. |
| Foundation model | **`terratorch`** (IBM) + Clay v1.5 (`made-with-clay/Clay`, `v1.5/clay-v1.5.ckpt`) | Primary backbone, frozen. TerraTorch provides model factory + fine-tuning hooks. Patch embedding D=1024. |
| Backbone swap | **`terratorch`** factory | Already installed for Clay v1.5. Week 7 backbone comparison (TerraMind / Prithvi) uses the same factory — no extra install. |
| LoRA | **`peft`** | Standard. Needed only for Week 10b. |
| Dim reduction | ~~`scikit-learn` IncrementalPCA~~ **dropped** | Replaced by `nn.Linear(1024, 128)` as the entry layer of the temporal head (Stage 7). Jointly optimised; no offline artefact. |
| OOD | Hand-rolled Mahalanobis (≈30 LoC) | No external dep worth adding. |
| Logging | **`loguru`** | One-line setup. |
| Metrics/tracking | **`mlflow`** (file backend, no server) | Local tracking, SQLite. Free. |
| COG writer | **`rio-cogeo`** | Validated COGs in one call. |
| Tile server for demo | **`titiler`** + **`leafmap`** or **Streamlit** | Pick one at Week 11. |
| Tests | **`pytest`** + **`hypothesis`** for a few property tests | |
| Lint / format | **`ruff`** (lint + format) + **`pyright`** (types) | Ruff replaces black, isort, flake8. |
| Task runner | **`just`** | Simpler than make. Recipes for common ops. |

**Hard rules for the implementing agent:**
1. Use DuckDB for streaming I/O; pandas is fine for in-memory label processing.
2. No nested for-loops over pixels at inference — rely on batched tensor ops + stride tricks.
3. Every stage writes to a cacheable artifact (Zarr, Parquet, or COG). Stages must be re-runnable in isolation.
4. `torch.compile(model, mode="reduce-overhead")` on the temporal head once it's stable.
5. `xarray.open_zarr` everywhere instead of re-reading COGs.
6. Use `dask` only when `stackstac` or `xarray` forces it. Don't reach for distributed dask on one GPU box.
7. GPU box is a single RunPod A100/L40S. No multi-node.

---

## 1.5 Strategic Context & Auditability — Why The Schema Looks The Way It Does

Phase 0 is not a research demo. It is positioned against a specific market shift: the **GCTWF Cape Water Performance Bond** closed April 2026 (≈USD $8.8M from FirstRand/RMB via TNC-SA), and clearing spend is now **contractually conditional on independent MRV**. Conservation Alpha is the incumbent verification agent and uses random-forest classifiers on BioSCape imagery. Our wedge — frozen foundation-model embeddings + calibrated per-pixel uncertainty + OOD novelty flags — is a *complement* (higher-frequency change layer) first, competitor second.

**Anchor-customer tiers** (drives prioritisation of label coverage + source traceability):
1. TNC-SA / GCTWF — the anchor; land it and the ecosystem follows.
2. City of Cape Town Invasive Species Unit (Biodiversity Management Branch).
3. DFFE Natural Resource Management / Working for Water — chronic "ghost hectares" problem; exactly our pain point.
4. CapeNature (provincial reserves + stewardship estate).
5. SANParks (TMNP, Agulhas, Garden Route).
Second tier: FirstRand/RMB directly, Santam (WUI fire-risk), local FPAs (Overberg / Cape Winelands / Garden Route), Berg-Olifants + Breede-Gouritz CMAs, WWF-SA Water Source Areas, corporate water stewards (SAB, Distell/Heineken, Woolworths, CCBSA), LandCare WC.

**Design implication — labels are MRV-grade from day one.** Every observation in `gs://ism-data/labels/wc/obs/` carries:

- `obs_id` — stable `"<source>:<source_record_id>"`, idempotent across re-ingests.
- `source` + `source_record_id` + `source_url` — a partner can re-audit any row to its origin.
- `coord_uncertainty_m` — indicated when attainable (GBIF DwC field, BioSCape derived from 5 m plot-center QA), null when not (Vegmap polygons).
- `ingested_at` + `ingest_run_id` — dedupe tiebreaker + traceability.

`class_id` is explicitly **not** assigned at ingest. The canonical store is species-keyed (`gbif_usage_key`, `nemba_category`); training configs crosswalk to class IDs via YAML. One parquet feeds the upper-Berg 12-class run and a future WC-wide NEMBA-full run without re-ingestion.

This framing is what justifies §3 Stage 4's broadened label scope (WC-wide NEMBA) against the tight inference scope (upper-Berg only) of §9.

---

## 2. Repo Structure

```
catchment-mrv/
├── pyproject.toml               # uv + tyro + lightning + ...
├── uv.lock
├── justfile                     # task recipes (see §3)
├── README.md
├── .env.example                 # PC_SDK_SUBSCRIPTION_KEY, HF_TOKEN, MLFLOW_URI
├── configs/
│   ├── aoi.geojson              # Berg River catchment polygon
│   ├── pipeline.yaml            # months, bands, chip size, strides
│   ├── model.yaml               # temporal head hyperparams, ensemble size
│   └── labels_schema.yaml       # 14-class + 10-class fallback mapping
├── data/                        # gitignored; external storage in practice
│   ├── raw/                     # S2 L2A scenes, COGs from MPC
│   ├── sr/                      # SEN2SR 2.5 m outputs
│   ├── labels/                  # parquet of point labels, fused raster
│   ├── embeddings/              # Zarr cube (T, H, W, D)
│   ├── pca/                     # fitted IncrementalPCA + compressed cube
│   ├── runs/                    # lightning checkpoints, MLflow db
│   └── outputs/                 # triplet COGs, reports
├── notebooks/                   # EDA only; no production logic here
│   ├── 01_aoi_inspection.ipynb
│   ├── 02_label_audit.ipynb
│   └── 03_embedding_sanity.ipynb
├── src/cmrv/
│   ├── __init__.py
│   ├── cli.py                   # tyro root — subcommands dispatch to stages
│   ├── config.py                # pydantic-settings models
│   ├── io.py                    # zarr / cog / parquet helpers
│   ├── aoi.py                   # bbox, tile grid, chip iterator
│   ├── ingest/
│   │   ├── stac.py              # pystac-client + stackstac
│   │   ├── cloud_mask.py        # s2cloudless or SCL-based
│   │   └── composite.py         # monthly median compositing
│   ├── sr/
│   │   ├── sen2sr.py            # SEN2SR wrapper, tiling, overlap-blend
│   │   └── checks.py            # PSNR/SSIM vs bicubic control
│   ├── labels/
│   │   ├── sources.py           # SANBI Vegmap, Forestry DB, Elsenburg, OSM, POSA, MODIS loaders
│   │   ├── fuse.py              # 7-tier fusion rules
│   │   └── schema.py            # 14-class, 10-class fallback
│   ├── embed/
│   │   ├── clay.py              # frozen Clay v1.5 extractor
│   │   ├── cube.py              # Zarr writer, (T,H,W,D) schema
│   │   └── pca.py               # IncrementalPCA fit + apply
│   ├── models/
│   │   ├── temporal_head.py     # 2-layer Transformer, CLS token, month-pos
│   │   ├── ensemble.py          # 5 seeds, saved as a bundle
│   │   └── lora.py              # Week 10b only
│   ├── train/
│   │   ├── datamodule.py        # lightning DataModule — samples (T,D) vectors + label
│   │   ├── lit_module.py        # CrossEntropyLoss(ignore_index=255)
│   │   └── callbacks.py         # early stop, ckpt, metrics
│   ├── ood/
│   │   ├── mahalanobis.py       # fit μ_c, shared Σ, τ_OOD
│   │   └── calibration.py       # ECE, reliability
│   ├── infer/
│   │   ├── stride8.py           # overlapping-patch inference over the cube
│   │   ├── triplet.py           # class / uncertainty / OOD COG writer
│   │   └── report.py            # hectare triplet, PDF via reportlab
│   ├── eval/
│   │   ├── heldout.py           # partner polygons + BioSCape holdout
│   │   └── metrics.py           # macro-F1, per-class, ECE
│   └── demo/
│       └── app.py               # Streamlit or leafmap viewer
└── tests/
    ├── conftest.py
    ├── fixtures/                # tiny 256×256 synthetic rasters + points
    ├── test_ingest.py
    ├── test_sr.py
    ├── test_labels.py
    ├── test_embed.py
    ├── test_models.py
    ├── test_ood.py
    ├── test_infer.py
    └── test_e2e_smoke.py        # 3-tile end-to-end on fixtures
```

**Conventions:**
- Every stage has one CLI entrypoint: `cmrv <stage> ...` (implemented with tyro subcommands).
- Every stage reads from and writes to fixed paths under `data/` controlled by `configs/pipeline.yaml` — no ad-hoc paths in code.
- All xarray DataArrays carry CRS, transform, and `nodata=255` for label rasters.
- Zarr chunk shape: `(T, 256, 256, D)` — matched to the training chip size so DataLoader reads one chunk per sample.

---

## 3. Pipeline Stages — Build and Test

Each stage below has: **what it does**, **key code sketch**, **DoD**, **tests**, **pitfalls**. The implementing agent must hit DoD before moving on.

### Stage 1 — AOI & Tile Grid (Week 1, day 1)
**Does:** Load `configs/aoi.geojson`, produce a tile grid of N × N km cells covering the AOI, persist to `data/aoi/tiles.parquet` (geopandas parquet).

**Sketch:**
```python
# src/cmrv/aoi.py
import geopandas as gpd
from shapely.geometry import box

def build_tile_grid(aoi_path: str, tile_km: float, crs: str = "EPSG:32734") -> gpd.GeoDataFrame:
    aoi = gpd.read_file(aoi_path).to_crs(crs)
    minx, miny, maxx, maxy = aoi.total_bounds
    step = tile_km * 1000
    tiles = [box(x, y, x+step, y+step)
             for x in np.arange(minx, maxx, step)
             for y in np.arange(miny, maxy, step)]
    gdf = gpd.GeoDataFrame({"tile_id": range(len(tiles))}, geometry=tiles, crs=crs)
    return gdf[gdf.intersects(aoi.unary_union)].reset_index(drop=True)
```
**DoD:** `just tiles` produces a parquet with ~50–80 tiles for a 500 km² AOI. Map plots cleanly in `notebooks/01_aoi_inspection.ipynb`.
**Tests:** golden-file check on tile count for a fixture AOI.
**Pitfalls:** Do the work in UTM (EPSG:32734 for Berg River), not WGS84, so tiles are true-area squares.

---

### Stage 2 — STAC Ingest & Monthly Median Compositing (Week 2)
**Does:** Query Microsoft Planetary Computer for S2 L2A scenes over the AOI for 4 months (Aug, Oct, Jan, Apr). Cloud-mask via the SCL band. Median-composite per month. Write 10 m COGs per tile per month to `data/raw/<tile>/<month>.tif`.

**Sketch:**
```python
# src/cmrv/ingest/stac.py
import planetary_computer as pc, pystac_client, stackstac, xarray as xr

# No subscription key required — MPC public S2 L2A is anonymously accessible.
# pc.sign_inplace adds time-limited SAS tokens to asset URLs.
def load_month(tile_geom, month_start, month_end, bands):
    cat = pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=pc.sign_inplace,
    )
    items = cat.search(
        collections=["sentinel-2-l2a"],
        intersects=tile_geom.__geo_interface__,
        datetime=f"{month_start}/{month_end}",
        query={"eo:cloud_cover": {"lt": 40}},
    ).item_collection()
    da = stackstac.stack(items, assets=bands + ["SCL"], resolution=10,
                         epsg=32734, bounds_latlon=tile_geom.bounds,
                         dtype="float32", chunksize=2048)
    # cloud mask
    scl = da.sel(band="SCL")
    mask = ~scl.isin([3, 8, 9, 10])  # shadow + clouds + cirrus
    da = da.sel(band=bands).where(mask)
    return da.median(dim="time", skipna=True)
```
**DoD:** One COG per (tile, month) on disk. 4-month × 50–80-tile matrix is ≥95% complete (cells with ≥1 valid scene per month).
**Tests:** unit test on SCL masking with a synthetic scene; integration test that runs one tile one month and checks COG validity with `rio-cogeo validate`.
**Pitfalls:**
- Planetary Computer URLs expire — sign right before reading, do not cache signed URLs.
- `stackstac.stack` with `bounds_latlon` is lazy; `.compute()` only inside the monthly median step.
- Set `chunksize=2048` to avoid tiny-chunk overhead.

---

### Stage 3 — SEN2SR Super-Resolution to 2.5 m (Week 3)
**Does:** Upsample each 10 m composite to 2.5 m using SEN2SR. Write `data/sr/<tile>/<month>.tif` as 2.5 m COGs. Hold out one tile × two months for QC against bicubic.

**Sketch:**
```python
# src/cmrv/sr/sen2sr.py
import torch, rasterio as rio
from tacofoundation.sen2sr import load_model  # hypothetical; adjust to actual HF path

MODEL = load_model("tacofoundation/SEN2SR").eval().cuda()

@torch.inference_mode()
def super_resolve_tile(in_cog: str, out_cog: str, chip: int = 256, overlap: int = 32):
    with rio.open(in_cog) as src:
        arr = src.read().astype("float32") / 10000.0  # S2 reflectance scaling
        out = torch.zeros((arr.shape[0], arr.shape[1]*4, arr.shape[2]*4), dtype=torch.float32)
        # chipwise inference with feather blending on overlap
        ...
    # write COG at 2.5 m via rio.open with transform / 4
```
**DoD:**
- All tile × month 2.5 m COGs on disk.
- Held-out comparison: SEN2SR PSNR ≥ bicubic PSNR + 2 dB on a BioSCape overlap scene, SSIM improvement logged in MLflow.
- Visual check of a fynbos vs plantation boundary in `notebooks/02_sr_sanity.ipynb`.

**Tests:** contract test on tile dimensions (4× upsample), AMP-safe check that output is in [0,1], reflected-padding on overlap is not introducing edge artifacts.
**Pitfalls:**
- Memory: a full tile at 2.5 m is 4× larger. Chip with 32 px feathered overlap.
- SEN2SR expects specific band order and scale — wrap in a single `prepare_input()` function, test it.
- Write with the correct `transform` (new pixel size is old / 4), not just by changing `width`/`height`.

---

### Stage 4 — Labels: Canonical WC Occurrence Store + Chip Everything (Weeks 2–4, parallel to 2 & 3)

**AS BUILT — chip / point regime, not pixel-segmentation regime.**  Earlier drafts of this stage assumed labels would be rasterised into a sparse 2.5 m label COG that the DataLoader sampled per-pixel for a U-Net-style head.  The actual implementation pivoted to a chip / point regime: 64×64 chips per (`obs_id`, month) extracted at the obs coordinate, four months per obs, model consumes embedding-patch vectors at each obs across T months → temporal head.  Rasterised labels (`labels-fuse`) survive only as a viz-side QGIS overlay; training never reads them.

**Scope shift from earlier drafts**: the label AOI broadens to **Western Cape province, full NEMBA AIS plant list** (~380 taxa × categories 1a/1b/2/3). The inference AOI stays upper-Berg (§9). The canonical store is reusable, species-keyed, and idempotent across re-ingests; class assignment is purely a downstream training-subset concern, not a chip-time concern.

**Does:**
1. Ingest every label source into the unified observation schema (see below) — names standardised via GBIF backbone at this point.
2. Persist partitioned GeoParquet at `gs://ism-data/labels/wc/obs/source=<source>/` (one partition per source — re-ingesting GBIF never touches BioSCape rows).
3. Union + dedupe by `obs_id` into `gs://ism-data/labels/wc/summary.parquet` (auditability metrics: per-source × per-NEMBA-category counts + fraction with non-null `coord_uncertainty_m`).
4. **Chip everything** that survives basic filters via `ingest-chips` — AOI clip, `coord_uncertainty_m ≤ 40`, date range, `geom_type='point'`.  No schema or class_map at this stage.
5. **Explore via `chips-stats`** — manifest is the source of truth for "what's in my chips, is it balanced, what's spatially dominated?"  Reads `manifest.parquet` directly.
6. **Subset for training via `make-split`** — filter manifest by `--species`, optionally assign `class_id` via `--class-map`.  The class crosswalk is a labelling shim applied here, not a chip-time gate.

**Optional viz path (not on the critical path):** `cmrv labels-fuse` produces a per-tile sparse label COG via `load_training_labels` + highest-weight-wins fusion for QGIS / Streamlit overlays only.

Schema + class-map crosswalks live in [configs/labels_schema.yaml](configs/labels_schema.yaml).

---

**Unified observation schema** (`src/cmrv/labels/observations.py`)

| column | type | notes |
|---|---|---|
| `obs_id` | str (PK) | `"<source>:<source_record_id>"`, stable across re-ingests |
| `source` | cat | `gbif` \| `inat_via_gbif` \| `bioscape_line` \| `bioscape_plot` \| `vegmap` \| `sapia` |
| `source_record_id` | str | native ID |
| `source_url` | str? | back-link |
| `species` / `species_normalized` | str / str | raw + hybrid-stripped binomial |
| `gbif_usage_key` | i64? | resolved via backbone |
| `nemba_category` | cat? | `"1a"`\|`"1b"`\|`"2"`\|`"3"`\|null |
| `geometry` | wkb (EPSG:4326) | Point or Polygon |
| `geom_type` | cat | `"point"`\|`"polygon"` |
| `coord_uncertainty_m` | f64? | **indicated when attainable; null when not** |
| `event_date`, `basis_of_record`, `cover_pct`, `weight` | — | provenance + training signal |
| `ingested_at`, `ingest_run_id` | ts, str | upsert tiebreaker + traceability |
| `aoi_admin1` | str | `"western_cape"` |

Explicitly **not** in schema: `class_id`. Assigned by training config via `configs/labels_schema.yaml → class_maps.<name>`.

---

**Class maps** (`configs/labels_schema.yaml → class_maps:`)

- `class_maps.upper_berg_12` — the original 12-class biogeographically-tuned map (Berg inference): IAP upland (7), IAP riparian (1), native (3), other landcover (1). Unchanged from earlier drafts.
- `class_maps.wc_nemba_full` — one class per NEMBA-listed taxon group (~50 after merges); stub for a future WC-wide run.

`vernacular_map` + `taxa_rejection` remain top-level, shared across maps.

---

**Active label sources for Phase 0** (WC-scope; others deferred, see `data/labels/sources.md`)

| # | Source | Type | Role | `coord_unc` attainable? | Weight |
|---|---|---|---|---|---|
| 1 | **BioSCape VegPlots (Berg+Eerste, 2022–23)** — 36 sites, LineIntercept (42 IDs/plot) + PlotCoverage (per-quadrant cover) | Field points | Held-out evaluation + IAP-dominant training signal | yes (≈5 m derived) | 0.95 |
| 2 | **GBIF Download API** — WC bbox × NEMBA plants × 2018+ × ≤500 m DwC uncertainty | Points | Broad presence signal across WC | yes (DwC `coordinateUncertaintyInMeters`) | 0.5 |
| 3 | **iNaturalist-research-grade via GBIF** (`datasetKey=50c9509d-22c7-4a22-a47d-8c48425ef4a7`) — split out so weight + trust profile is distinct | Points | Citizen-science presence | yes (= iNat `public_positional_accuracy`) | 0.4 |
| 4 | **SANBI Vegmap 2024 (NVM2024 IEM5)** — biome/bioregion polygons, clipped to WC | Polygon | Native anchors (fynbos, renosterveld, indigenous forest) | **no** (polygon — `geom_type=polygon`) | 0.8 |
| 5 | **SAPIA (SANBI)** — if endpoint reachable (time-boxed) | Points/plots | SA-first IAP density records | variable (pass through) | 0.7 |

**Deferred** (per `data/labels/sources.md`): SA NLC 2022, NFEPA rivers+wetlands, WRC Report 3193 (FynBase), BioSCape IAP map (Sun ScholarData), NASA CFR acoustic, POSA native points, CapeNature partner polygons, WfW managed-clearance, Commercial Forestry DB. Each carries the reason (license / no API / awaiting outreach).

---

**BioSCape "majority IAP" — how the plot signal becomes a label**

Per the ORNL DAAC methods PDF (plot center ≤5 m GPS QA):
- Circular 10-m-diameter plot with two 10-m rope transects (N–S and W–E crossing at center).
- At every 1-m mark, observer records the plant on left and right of the rope (canopy view). 42 IDs per plot (`LineIntercept.csv`). If no plant is at the mark → ground cover recorded.
- 2022 plots: fractional cover per species recorded for the **NE quadrant only**. 2023 plots: all four quadrants + plot-level fractional vegetation dominance (`PlotCoverage.csv`).

Two rules are computed and cross-checked:

- **Transect rule (sanity check)**: fraction of the 42 LineIntercept IDs matching a NEMBA-listed species. >50% ⇒ transect-majority IAP.
- **Areal rule (the pixel-mappable training signal)**: per quadrant, `sum(PercentCoverAlive WHERE species ∈ NEMBA)`; plot-mean across available quadrants (1 in 2022, 4 in 2023). Assigned to the plot polygon geometry at rasterisation.

Agreement ⇒ `weight=0.95`. Disagreement (e.g., one tall wattle on the line but minor quadrant cover) flagged, not dropped; `weight=0.5`.

LineIntercept rope-positions with **no plant** at the mark are kept as `species_normalized=null, basis_of_record="GROUND_COVER"` — useful for negative sampling.

---

**Idempotent re-ingest**

Each source loader writes to its own partition (`source=<source>/`). Upsert semantics per partition: read existing → concat new → `group_by(obs_id).agg(*, max(ingested_at))` → atomic overwrite via temp-prefix + `gcsfs.mv`. Re-running any single source leaves the merged summary row count unchanged; that's an explicit handoff check (§10).

**Species name alignment:** 4-layer resolver in `src/cmrv/labels/gbif.py` cascades vernacular_map → GBIF `/species/match` → pytaxize → GBIF `/species/suggest` fuzzy. Catches SAPIA/WRC synonyms; resolved usage-keys cached to `gs://ism-data/labels/nemba_taxa_resolved.parquet`.

**Fusion rule (at training-config time, not ingest time):** `load_training_labels` returns deduped rows; rasterisation applies highest-weight-wins with conflicts below margin → `nodata=255`. Riparian mask (NFEPA ±30 m, once NFEPA is promoted from deferred) upweights class 7 + *A. melanoxylon*.

**Image-year alignment:** BioSCape 2022–2023, GBIF rich from 2018+, NLC 2022. → **Target imagery year = 2023** to maximise label–image temporal overlap. See `configs/pipeline.yaml → months`.

---

**DoD:**
- `gs://ism-data/labels/wc/obs/source=*/` populated by GBIF + iNat-via-GBIF + BioSCape line + BioSCape plot + Vegmap + NLC sample.
- `summary.parquet` reports per-source × per-NEMBA-category counts and **fraction with non-null `coord_uncertainty_m`** per source.
- `load_training_labels(aoi=berg_upper, species=upper_berg_12_taxa)` returns a GeoDataFrame for the optional viz-side `labels-fuse`.
- `gs://ism-data/chips/train/manifest.parquet` exists and `cmrv chips-stats` reports >100 obs_ids per IAP class in `class_maps.upper_berg_12` after a `make-split --class-map upper_berg_12 --species [...]`.
- BioSCape held-out set excluded from training splits via `eval_bioscape.parquet` manifest.
- Re-running any single-source ingest is a no-op on the summary row count (idempotency).

**Tests:**
- Synthetic upsert fixture: same `obs_id` twice → one row at max `ingested_at`; distinct `obs_id` → both retained.
- BioSCape geodesic geometry fixture: known plot center + meter + side, `pyproj.Geod.fwd` returns expected lon/lat within 0.1 m.
- Species-name alignment fixture: GBIF sample with known synonyms → all map to correct `(gbif_usage_key, nemba_category)`.
- `rio-cogeo validate` passes on every rasterised tile label COG.

**Pitfalls:**
- Reproject vector to raster CRS *before* rasterising — `rasterio.features.rasterize` with matching transform.
- GBIF/iNat point buffer: 10 m = 4 px at 2.5 m — verify in pixel units, not metres.
- *Acacia karroo* (native), *Sesbania sesban* (native), genus-level `Acacia sp.` / `Pinus sp.` / `Hakea sp.` / `Eucalyptus sp.` — all in `taxa_rejection`; filtered at parse time.
- BioSCape ground-cover-only rows (rope-position without plant) — do **not** drop; keep with `species_normalized=null` for later negative-sample use.
- BioSCape transect geometry: use `pyproj.Geod.fwd` (geodesic forward), not the naive `111320/cos(lat)` degree approximation — the latter drifts at WC latitudes.
- Coord uncertainty: GBIF DwC field is variable, often null for museum records. Drop at ingest when `coord_uncertainty_m > 500`; training configs can tighten.

---

### Stage 5 — Clay v1.5 Frozen Embedding Extraction (Week 5)
**Does:** For each (tile, month), chip the 2.5 m composite into Clay's expected input, run Clay frozen, write patch embeddings to a Zarr cube `data/embeddings/cube.zarr` with layout `(T=4, H, W, D=768)` per tile. H, W are in *patch* coordinates at Clay's 8×8 patch size at native GSD — at 2.5 m input, each patch is 20 m on the ground.

**Sketch:**
```python
# src/cmrv/embed/clay.py
import torch
from terratorch.models.backbones.clay_v1 import ClayMAEModule

# HF repo: made-with-clay/Clay  |  file: v1.5/clay-v1.5.ckpt (5.16 GB)
CLAY = ClayMAEModule.load_from_checkpoint(
    "path/to/clay-v1.5.ckpt", map_location="cuda"
).eval()

@torch.inference_mode()
def embed_chip(
    chip: torch.Tensor,        # (B, C, 256, 256) float32 at 2.5 m
    wavelengths: torch.Tensor, # (B, C) wavelengths in nm
    timestamps: torch.Tensor,  # (B, 4) [week, hour, lat, lon]
) -> torch.Tensor:
    out = CLAY.encoder(chip, wavelengths=wavelengths, timestamps=timestamps)
    # out: (B, N_patches, 1024); N_patches = 32×32 = 1024 for a 256-px chip
    h = w = 256 // 8  # 32 patch tokens per side
    return out.reshape(chip.shape[0], h, w, 1024)
```
```python
# src/cmrv/embed/cube.py
import zarr, numcodecs as nc
def open_cube(path, shape, dtype="float16"):
    compressor = nc.Blosc(cname="zstd", clevel=3, shuffle=nc.Blosc.BITSHUFFLE)
    # D=1024: Clay v1.5 patch token width. Stage 7 head projects to 128 internally.
    return zarr.open(path, mode="a", shape=shape, dtype=dtype,
                     chunks=(1, 256, 256, 1024), compressor=compressor)
```

**DoD:**
- Cube exists, opens with `xr.open_zarr`, chunk reads return in <100 ms on the GPU box.
- A sanity decode: pick 5 random patches, visualize the corresponding 20 m ground footprint, confirm embeddings cluster by broad land cover in a UMAP in `notebooks/03_embedding_sanity.ipynb`.

**Tests:** golden-file test on a fixed chip — Clay output bitwise-equal across runs (set `torch.use_deterministic_algorithms(True)`).
**Pitfalls:**
- Clay expects specific band order and wavelength metadata — pass both explicitly.
- Store in fp16 to halve disk. Decompress to fp32 just before feeding the temporal head.
- Don't try to cube all months together in memory — stream tile-by-tile into Zarr.

---

### ~~Stage 6 — IncrementalPCA~~ (dropped)

**Dropped.** Offline PCA replaced by `nn.Linear(1024, 128, bias=False)` as the first layer of the temporal head (Stage 7). This projection is jointly optimised with the classification task, removes the `cube_pca128.zarr` artefact, and simplifies the data path. The `cube.zarr` stays at `D=1024` (float16); the linear layer decompresses on the fly during training and inference.

*No code to ship for this stage.*

---

### Stage 7 — Temporal Head + 5-Seed Ensemble Training (Week 6)
**Does:** Train the 2-layer Transformer temporal head (architecture already specified in `Phase_0_Wedge_Prototype.md`). 5 seeds → ensemble. Lightning `Trainer` with AMP.

**Sketch:**
```python
# src/cmrv/train/lit_module.py
import torch, lightning as L
from cmrv.models.temporal_head import TemporalHead

class LitHead(L.LightningModule):
    def __init__(self, n_classes=14, d_in=1024, d=128, lr=3e-4):
        super().__init__()
        self.save_hyperparameters()
        # d_in=1024 (Clay v1.5 patch tokens) → d=128 (temporal attention width)
        self.model = TemporalHead(d_in=d_in, d=d, num_classes=n_classes, T=4)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    def training_step(self, batch, _):
        x, y, mask = batch
        logits = self.model(x, month_mask=mask)
        return self.loss(logits, y)
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=0.01)
```
DataModule emits `(T=4, D=128)` vectors **per labeled obs_id** (point-centred) + `y` + missing-month `mask`.  At each obs coordinate the DataModule reads the embedding patch (Clay 8×8 at 2.5 m = 20 m footprint) for each of the four months, stacks them as the `(T=4, D=128)` input, and assigns the obs's `class_id` from the `make-split` output as `y`.

**DoD:**
- 5 checkpoints in `data/runs/ens_{0..4}/`.
- Macro-F1 on held-out blocks ≥ 0.65 on the 12-class `upper_berg_12` schema.
- ECE ≤ 0.10 post-temperature-scaling.

**Tests:** overfit one batch to F1=1.0 to prove gradient flow; test `ignore_index` truly removes those obs from loss.
**Pitfalls:**
- Class imbalance: use inverse-frequency-weighted sampler in the DataLoader, not class weights in loss (simpler and as effective).
- Per-obs sampling already avoids the polygon-boundary problem — no erosion needed because there are no rasterised polygon labels in this regime.

---

### Stage 8 — Mahalanobis OOD Fit (Week 6, end)
**Does:** From training-set penultimate-layer features, compute per-class mean μ_c and shared covariance Σ (Ledoit-Wolf shrinkage). Save `data/ood/maha.joblib`. Set τ_OOD at the 95th percentile of in-distribution Mahalanobis distances.

**Sketch:**
```python
# src/cmrv/ood/mahalanobis.py
from sklearn.covariance import LedoitWolf
import numpy as np, joblib

def fit(feats: np.ndarray, y: np.ndarray) -> dict:
    classes = np.unique(y)
    mus = {c: feats[y == c].mean(0) for c in classes}
    sigma = LedoitWolf().fit(feats - np.stack([mus[c] for c in y])).covariance_
    inv = np.linalg.pinv(sigma)
    return {"mus": mus, "inv": inv}

def distance(x, mus, inv):
    return np.min([((x - m) @ inv * (x - m)).sum(-1) for m in mus.values()], axis=0)
```
**DoD:** 95% of held-in validation pixels below τ; a planted OOD control (e.g., urban pixels held out entirely from training) flagged ≥85%.
**Pitfalls:** Use the temporal head's CLS token as the feature, not logits.

---

### Stage 9 — Stride-8 Overlapping Inference → Triplet COG (Weeks 7–8)
**Does:** For each tile, slide a window with stride=8 patches over the Clay PCA cube, run the ensemble, average softmax probs, write three co-registered COGs at 10 m effective resolution:
1. `class.tif` — argmax class ID
2. `uncertainty.tif` — ensemble entropy or variance of softmax
3. `ood.tif` — Mahalanobis distance (or boolean at τ)

**Sketch:**
```python
# src/cmrv/infer/stride8.py
@torch.inference_mode()
def infer_tile(cube_path, ensemble, out_dir, stride=8):
    cube = xr.open_zarr(cube_path)  # (T, H, W, D=128)
    H, W = cube.sizes["y"], cube.sizes["x"]
    logits_sum = torch.zeros((H*4, W*4, 14))  # stride=8 → 4× oversample
    counts = torch.zeros_like(logits_sum[..., 0])
    for yy in range(0, H, 8):
        for xx in range(0, W, 8):
            patch = cube.isel(y=slice(yy, yy+64), x=slice(xx, xx+64))
            x = torch.as_tensor(patch.values, device="cuda").permute(1,2,0,3)  # (H,W,T,D)
            probs = torch.stack([m(x.reshape(-1, 4, 128)) for m in ensemble]).softmax(-1).mean(0)
            ...  # accumulate with spatial offset
    # write three COGs via rio-cogeo
```
**DoD:**
- Three valid COGs per tile.
- `rio-cogeo validate` passes.
- Hectare triplet produced: `confirmed IAP ha` (class + ood=False + uncertainty<τ_ent), `uncertain ha`, `novel ha`.

**Pitfalls:**
- Don't soft-blend by naive averaging — weight by a Hann window over the patch to suppress seam artifacts.
- Keep all three COGs on the same grid, CRS, and nodata convention.

---

### Stage 10 — Held-Out Evaluation (Week 9)
**Does:** Score against (a) partner polygons withheld from training, (b) BioSCape reference, (c) field points from the CIB MSc student in Week 10. Metrics: macro-F1, per-class F1, ECE, precision@confidence-threshold curve. Report triplet recall: what fraction of confirmed IAP area matches partner polygon ground truth?

**DoD:** Metrics dumped to `data/outputs/eval.parquet` and rendered in a 1-page evaluation sheet via `src/cmrv/infer/report.py`.

---

### Stage 11 — Demo UI (Week 11)
**Does:** Streamlit app with three toggleable raster layers (class, uncertainty, OOD) over an ESRI basemap, a pixel-click inspector that pulls the full temporal embedding vector and per-class probability + Mahalanobis distance, and a hectare-triplet sidebar summary.

**DoD:** `just demo` spins up the app on port 8501, a partner can click around a held-out tile, see the triplet, and download a 1-page PDF.

---

### Stage 12 — LoRA-Plus Fine-Tune Comparison (Week 10b, for preprint) *[optional if behind schedule]*
**Does:** Unfreeze Clay patch embed + first transformer block + inject LoRA adapters (`peft` `LoraConfig(r=8, lora_alpha=16, target_modules=["qkv","proj"])`). Retrain with differential LRs: 1e-5 (patch embed), 1e-4 (block-1 + LoRA), 3e-4 (temporal head). Re-run evaluation. Ablation row in preprint.

**DoD:** macro-F1 delta vs frozen baseline logged in MLflow, training curves screenshot for preprint.

---

## 4. Testing Strategy

| Layer | Tooling | Target |
|---|---|---|
| Unit | pytest + hypothesis | ≥80% line coverage on `src/cmrv/` excluding `demo/` and `notebooks/` |
| Fixtures | 256×256 synthetic TIFFs + tiny geojson | Run in <10 s |
| Integration | pytest-marked `@slow` | One tile × one month, full pipeline, on CPU, in <3 min |
| Golden-file | `syrupy` snapshots | Clay output, PCA output, Mahalanobis distances on fixed input |
| Smoke e2e | `just smoke` | 3 tiles × 4 months, runs on the GPU box in <15 min, asserts COG validity + hectare triplet sanity |
| Validation | `rio-cogeo validate` | Every written COG |
| Type | `pyright --strict src/cmrv/` | Zero errors before each stage completes |
| Lint | `ruff check && ruff format --check` | Pre-commit hook |

**Rule:** every new stage PR must add tests for that stage. Don't ship a stage with a green CI if only the e2e smoke covers it.

---

## 5. Performance Optimization Notes

In priority order, apply these as you hit bottlenecks — don't pre-optimize.

1. **Zarr chunking matches DataLoader sampling.** If the DataLoader samples 256×256 chips, Zarr chunks are (1, 256, 256, 768). Otherwise you decompress 10× more than you need.
2. **`fp16` for cached embeddings, `bf16` for training compute.** Clay can output fp16 directly. PCA operates on fp32 upcasted just for the fit.
3. **`torch.compile(..., mode="reduce-overhead")` on the temporal head** only after architecture is frozen. Recompilation on every batch is worse than eager mode.
4. **`num_workers` = vCPU − 2, `persistent_workers=True`, `prefetch_factor=4`.** Measure the data-loading vs compute ratio with Lightning's `profiler="simple"` and fix whichever dominates.
5. **Batch the Clay forward pass.** Never call Clay on one chip — batch 32+ chips per forward.
6. **Inference: half-precision everywhere.** Ensemble in fp16, accumulate softmax in fp32.
7. **Skip dask unless forced.** stackstac uses dask under the hood; let it, but don't spin up a distributed cluster. `ThreadPoolExecutor(max_workers=8)` is usually enough for the download stage.
8. **Cache STAC item collections to disk** as GeoJSON so re-runs don't re-hit the STAC API. These are cheap to serialize.
9. **COG overviews.** Always write with `OVERVIEW_RESAMPLING=nearest` for class, `average` for uncertainty/OOD. Titiler needs overviews to render fast.

Phase 0 target budget on a single A100:
- Full pipeline (50 tiles × 4 months): SR ~6 h, embeddings ~2 h, training ~30 min per seed, ensemble inference ~1.5 h.
- Total cold-run wallclock: <12 h.

---

## 6. Common Pitfalls (the ones that will cost days)

1. **CRS drift.** Vector labels in WGS84, rasters in UTM, embedding cube in pixel coords. Always declare CRS, always convert via `rioxarray.reproject_match`.
2. **Off-by-one patch math.** Clay patch=8, SEN2SR upsample=4×. At 2.5 m input, one patch = 20 m. A 256-pixel chip gives 32×32 patch tokens. Write these numbers on a sticky note.
3. **Month availability asymmetry.** Some tiles have 0 valid Apr scenes. Don't drop the tile — pass a `month_mask` into the temporal head so the attention ignores missing months. This is already baked into the head design; honor it in the DataModule.
4. **Label leakage.** Never train on a tile that overlaps a held-out partner polygon. Build the spatial splits *before* training and check intersection.
5. **Silent fp16 NaNs.** Clay + AMP can emit NaN on bad input. Add a `torch.isfinite(embed).all()` assert after every Clay forward during the first day of Stage 5.
6. **MLflow overflow.** Log per-tile metrics as artifacts, not as individual `log_metric` calls — thousands of metric points break the UI.
7. **Streamlit reloading on every file change during demo build.** Use `@st.cache_resource` for the raster loader; disable the file watcher on the output dir.
8. **SEN2SR tiling seams.** Without feathered overlap, you'll see grid artifacts at chip boundaries. Catch this in `sr/checks.py` with an edge-gradient test.
9. **Ensemble divergence when reloading checkpoints.** Lightning saves optimizer state; on restart use `Trainer.fit(ckpt_path=...)` not manual `load_state_dict` — otherwise scheduler drifts.

---

## 7. justfile (task runner)

```make
set shell := ["bash", "-cu"]

default: lint test

install:
    uv sync

lint:
    uv run ruff check .
    uv run ruff format --check .
    uv run pyright src/cmrv

test:
    uv run pytest -x

smoke:
    uv run pytest -x -m smoke

tiles:
    uv run cmrv aoi tiles --aoi configs/aoi.geojson --km 10 --out data/aoi/tiles.parquet

ingest MONTH:
    uv run cmrv ingest month --month {{MONTH}} --tiles data/aoi/tiles.parquet

sr TILE MONTH:
    uv run cmrv sr run --tile {{TILE}} --month {{MONTH}}

labels-ingest:
    uv run cmrv labels-ingest

ingest-chips:
    uv run cmrv ingest-chips

chips-stats:
    uv run cmrv chips-stats

# Optional: per-tile sparse label COG for QGIS / Streamlit overlays only.
# Training never reads this — see Stage 4 "AS BUILT" note.
labels-fuse:
    uv run cmrv labels-fuse

embed:
    uv run cmrv embed clay --cube data/embeddings/cube.zarr

pca:
    uv run cmrv embed pca --in data/embeddings/cube.zarr --out data/embeddings/cube_pca128.zarr

train SEED:
    uv run cmrv train head --seed {{SEED}} --out data/runs/ens_{{SEED}}

ood:
    uv run cmrv ood fit --runs data/runs

infer TILE:
    uv run cmrv infer tile --tile {{TILE}} --out data/outputs

eval:
    uv run cmrv eval run --out data/outputs/eval.parquet

demo:
    uv run streamlit run src/cmrv/demo/app.py
```

---

## 8. 12-Week Execution Timeline (matches `Phase_0_Wedge_Prototype.md`)

| Wk | Repo milestone | DoD artifact |
|----|---|---|
| 1 | Repo scaffolded + Stage 1 + `just lint test` green | `tiles.parquet`, CI green |
| 2 | Stage 2 (STAC ingest) + Stage 4 kickoff (label sources 1–3) | 4-month COGs for 5 tiles, label summary parquet |
| 3 | Stage 3 (SEN2SR) + Stage 4 complete (fusion) | SR COGs, sparse label raster, per-class floors met |
| 4 | Stage 5 (Clay extraction) | Zarr embedding cube populated |
| 5 | Stage 7 scaffold (temporal head with Linear(1024→128) projection) | 1-seed checkpoint |
| 6 | Stage 7 full (5-seed ensemble) + Stage 8 (OOD) | 5 checkpoints, maha.joblib, ECE ≤ 0.10 |
| 7 | Stage 9 inference pipeline + backbone swap (TerraTorch) | Triplet COG for one tile, TerraMind comparison row |
| 8 | Stage 9 at scale (all tiles) + active-learning loop | Full AOI triplet COGs, hectare-triplet report |
| 9 | Stage 10 evaluation | `eval.parquet`, preliminary F1/ECE |
| 10a | Field validation with CIB student | Field-verified points, updated metrics |
| 10b | Stage 12 LoRA-Plus (optional) | Ablation row in MLflow |
| 11 | Stage 11 demo UI | Streamlit app live, PDF export works |
| 12 | Preprint draft, ablation tables, artifact archive | arXiv-ready PDF, repo tagged `v0.1.0` |

---

## 9. What *not* to build in Phase 0

- ❌ U-Net student (that is Phase 1 probe-as-labeler distillation).
- ❌ Multi-catchment generalization (stays in Berg River).
- ❌ Production API / containerization / k8s.
- ❌ SAM or any segmentation-from-scratch architecture.
- ❌ PlanetScope ingest.
- ❌ Manual polygon digitization tools.
- ❌ Any deep-dask multi-node setup.
- ❌ Custom UI frameworks — Streamlit or leafmap only.

Everything above has a tempting reason to be in scope and none of them is. Defer aggressively.

---

## 10. Handoff Checklist for the Implementing Agent

Before declaring Phase 0 done, verify every item:

- [ ] All 12 stages have passing unit + integration tests.
- [ ] `just smoke` runs end-to-end on CI hardware in <15 min.
- [ ] AOI triplet COGs exist, validated by `rio-cogeo validate`.
- [ ] Held-out macro-F1 ≥ 0.65 on 10-class fallback.
- [ ] ECE ≤ 0.10 after temperature scaling.
- [ ] Mahalanobis OOD flags ≥85% of planted OOD control.
- [ ] Hectare-triplet (confirmed / uncertain / novel) reported, denominators disclosed honestly.
- [ ] Field validation (Week 10a) points incorporated into final metrics.
- [ ] Demo UI runs locally, PDF report downloads.
- [ ] Preprint draft submitted to arXiv, repo tagged `v0.1.0`, README has install + quickstart.
- [ ] `Phase_0_Wedge_Prototype.md` success criteria (all 7) reviewed and signed off.

**Auditability (§1.5 — MRV-grade labels from day one):**
- [ ] Every row in `gs://ism-data/labels/wc/obs/` has non-null `obs_id`, `source`, `source_record_id`, `ingested_at`.
- [ ] Re-running any single source loader (`cmrv labels-ingest --source <name>`) leaves `summary.parquet` row count unchanged (idempotency).
- [ ] `summary.parquet` reports per-source fraction-with-`coord_uncertainty_m` (auditability metric; Vegmap polygons exempt).
- [ ] `load_training_labels(aoi=berg_upper, species=upper_berg_12_taxa)` reproduces the Stage 4 training input used by current Stage 5+ code (backward compatibility check).

When every box is checked: ship, write the Phase 1 plan, and book the partner demo.
