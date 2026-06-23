# Phase 0 Prototype — Task Tracker

Conventions & architecture: [CLAUDE.md](../CLAUDE.md). Phase 0 builds **Lens A
(Invasive Alien Plants)** only, **locally** (no cloud bucket), from
**cover-bearing scientific datasets** (BioSCape; GBIF/iNat dropped),
**IAP-species-only** (native classes deferred), tuned to **WC phenology**.

---

## Completed

- **AOI & tile grid** — `cmrv aoi-wc`, `aoi-tiles` → `data/aoi/*.parquet` (WC; tiles are the inference unit)
- **S2 ingest** — `cmrv ingest-month` (MPC STAC → SCL mask → monthly median → COG)
- **Label adapter** — `cmrv labels-bioscape-ingest` (VegPlots line + plot; IAP membership from class-map `members[]`)
- **Store inspector** — `cmrv labels` (per-source counts + coord-uncertainty/cover coverage; optional AOI/species preview)
- **Chip extraction** — `cmrv ingest-chips` (thin → 64×64 per obs×month, 10 km blocks, per-label window compute, incremental resume)
- **Split** — `cmrv chips-stats`, `cmrv make-split` (stratified block folds, `--class-map`, `--lock-folds`)

---

## Current — needs data

No label data is in the store yet. When a cover-bearing dataset arrives:
1. Write/confirm its adapter (one loader → the observation schema; fill its `source_doi` + `license`).
2. `uv run cmrv labels-bioscape-ingest` (or the new dataset's verb).
3. `uv run cmrv labels` — sanity-check per-source counts + cover coverage.
4. `uv run cmrv ingest-chips` → `chips-stats` → `make-split --class-map-name upper_berg_12`.

---

## Next — Stage 5: SEN2SR Super-Resolution (10 m → 2.5 m)

Package `sen2sr` (`uv add sen2sr mlstac`), model `tacofoundation/RS-SR-LTDF` via `mlstac.download()`.
Input `B × 10 × 128×128`, output 4× upsampled; `sen2sr.predict_large(model, X, overlap=16)` for larger.

- [ ] `src/cmrv/sr/sen2sr.py`: `download_model`, `load_model`, `prepare_input`, `super_resolve_tile`
- [ ] `cmrv super-resolve` CLI — `data/raw/` → `data/sr/`
- [ ] QC: PSNR ≥ bicubic + 2 dB

## Next — Stage 6: Clay Embedding Extraction

- [ ] `src/cmrv/embeddings/clay.py` — frozen Clay v1.5, batch ≥32, band order + wavelength metadata
- [ ] Chain composite → SR → Clay; persist **embeddings** (Zarr cube) as the durable artifact; chips/SR are transient cache
- [ ] Embedding manifest (sample → shard/offset, label, cover, fold). Patch math: 64px@10m → 256px@2.5m → 32×32 patches

## Deferred (designed, not built)

- [ ] **Region-aware months** — rainfall-seasonality zone layer → per-zone month set (WC = Feb/May/Sep now; add summer-rainfall calendars when other-province datasets land)
- [ ] **Cover gate** — flip `load_training_labels(min_cover_pct≈60)` on once cover-bearing data exists
- [ ] **Spatial-CV upgrades** — buffered/dead-zone folds, variogram-informed block size, leave-one-eco-region-out (before quoting accuracy)
- [ ] **Embedding store at scale** — Zarr → WebDataset shards; GEE→bucket compositing option for Vertex
- [ ] Temporal head training (mask missing months) + Mahalanobis OOD + wall-to-wall inference + demo viewer
- [ ] Re-add native/background label sources + native classes 8–11
- [ ] Lenses B (mine rehab), C (EUDR), D (biodiversity/bioacoustics)
