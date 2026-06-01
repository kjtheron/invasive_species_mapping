# Phase 0 Prototype — Task Tracker

Source of truth: [Phase_0_Build_Roadmap.md](../Phase_0_Build_Roadmap.md).

---

## Completed Stages

### Stage 0 — Repo scaffold ✅
- [x] Project skeleton, uv env, cmrv.io, .env.example, ruff + pytest

### Stage 1 — AOI & Tile Grid ✅
- [x] `cmrv aoi-fetch`, `cmrv aoi-wc`, `cmrv aoi-tiles`
- [x] `gs://ism-data/aoi/{berg_upper,western_cape,tiles}.parquet`

### Stage 2 — STAC Ingest & Compositing ✅
- [x] `cmrv ingest-month` — MPC STAC → SCL mask → monthly median → COGs

### Stage 2b — IAP + iNat Chip Extraction ✅
- [x] `cmrv ingest-chips` — 23,435 GBIF/iNat obs_ids chipped, ~90k manifest rows
- [x] MODIS MCD64A1 fire flag, spatial blocks (20km)

### Stage 3 — Spatial Splitting ✅
- [x] `cmrv make-split` — species filter, spatial thinning, stratified block assignment, --lock-folds

### Stage 4 — Label Ingestion & Fusion ✅
- [x] Unified obs store, GBIF + iNat ingest, BioSCape VegPlots, Vegmap NVM2024
- [x] NEMBA Gazette PDF extraction + taxa resolution
- [x] `cmrv labels-fuse`

---

## Current: Label Verification (BLOCKING — must complete before SR)

### 1. Fix NEMBA resolved parquet ✅
- [x] mearnsii patched: gbif_usage_key 2978223 → 2979775 (Acacia mearnsii, EXACT)
- [x] melanoxylon patched: gbif_usage_key 2978223 → 2979000 (Acacia melanoxylon, EXACT)
- Note: eucalyptus spp covered via schema `gbif.taxa` fallback (4 species listed)

### 2. Genus collapsing at split time ✅
- [x] `make_split` now accepts `--class-map-name` param
- [x] Applies `species_map` from schema → adds `class_id` column to manifest
- [x] Collapses e.g. E. globulus + E. camaldulensis → class 5 (eucalyptus_spp)
- [x] Collapses A. cyclops → class 1 (same as saligna), all Pinus spp → class 4
- Usage: `uv run cmrv make-split --class-map-name upper_berg_12`

### 3. Re-run GBIF ingest (YOU MUST RUN)
Schema `gbif.taxa` has all 16 species incl. mearnsii, melanoxylon, eucalyptus spp, pinus spp.
NEMBA parquet now has correct species keys. Fix is in code — just run:
- [ ] `uv run cmrv labels-ingest --source gbif`
  - Expected: ~2,600 mearnsii + ~1,650 melanoxylon + eucalyptus + pinus records added
  - Overwrites `source=gbif/` and `source=inat_via_gbif/` partitions
- [ ] `uv run cmrv ingest-chips` — chip new GBIF obs_ids (manifest incremental resume handles this)

### 4. Complete NLC chip extraction (YOU MUST RUN)
NLC obs store: **7,877 obs_ids**. Only 122 chipped (3 blocks). ~7,755 remaining.
- [ ] `uv run cmrv ingest-chips --skip-fire-filter` — chips all remaining NLC blocks
  - Target classes: fynbos (2500 obs), indigenous_forest (2500), other_landcover (2500)
  - renosterveld (377 obs) — will be excluded at split time (not a tree, too few obs)

### 5. Post-ingest distribution check (YOU MUST RUN after 3 + 4)
- [ ] Re-run species obs count — confirm mearnsii + melanoxylon + eucalyptus now present
- [ ] Identify top 7 IAP trees by obs count, cross-check against NEMBA 1a/1b regulation
- [ ] `uv run cmrv make-split --class-map-name upper_berg_12` — verify class_id counts per fold
- [ ] Drop renosterveld: exclude from split via `--species` filter or by removing from class_map

### Bucket cleanup (non-blocking)
- [ ] Delete legacy: `gs://ism-data/labels/{gbif_iap_berg,bioscape_obs,vegmap_native}.parquet`

---

## Next: Stage 5 — SEN2SR Super-Resolution (after label verification done)

Package: `sen2sr` (pip), model: `tacofoundation/RS-SR-LTDF` via `mlstac.download()`.
Example in `super_res.md`. Input: `B x 10 x 128x128` (10m S2 bands), output: 4× upsampled.
For tiles >128px use `sen2sr.predict_large(model, X, overlap=16)`.

- [ ] `uv add sen2sr mlstac` — verify install + GPU available
- [ ] `src/cmrv/sr/sen2sr.py`:
  - `download_model(out_dir)` — `mlstac.download(HF_URL, output_dir=out_dir)`
  - `load_model(model_dir, device)` — `mlstac.load(...).compiled_model(device)`
  - `prepare_input(arr_float32)` — scale to [0,1], nan→0, validate band order
  - `super_resolve_tile(in_cog, out_cog, overlap=16)` — reads COG, calls `predict_large`, writes 2.5m COG with transform/4
- [ ] `cmrv super-resolve` CLI — tile_id + month, reads `gs://ism-data/raw/`, writes `gs://ism-data/sr/`
- [ ] Download model checkpoint once to `data/models/LDSRS2-SEN2SR/`
- [ ] QC: PSNR ≥ bicubic + 2 dB; visual check `notebooks/02_sr_sanity.ipynb`

---

## Next: Stage 6 — Clay Embedding Extraction (after SR done)

- [ ] Verify terratorch Clay v1.5 checkpoint load
- [ ] `src/cmrv/embeddings/clay.py` — frozen encoder, batch ≥32, band order + wavelength metadata
- [ ] `cmrv extract-embeddings` CLI — SR chips → Zarr v3 `(1, 256, 256, 1024)` at `gs://ism-data/embeddings/`
- [ ] fp16 cached embeddings; AMP NaN guard
- [ ] Patch math: 64px@10m → 256px@2.5m → 32×32 patches (Clay patch=8)

---

## Deferred: Stage 7+ (not started)

- [ ] Temporal head training (linear probe, 4-month stack, calibrated uncertainty)
- [ ] Wall-to-wall inference, uncertainty maps, Cloud Run demo
