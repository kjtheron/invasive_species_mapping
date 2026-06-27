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
4. `uv run cmrv ingest-chips` → `chips-stats` → `make-split --class-map-name western_cape_iap`.

---

## Stage 5–6: Embedding — UniverSat (adopted)

**Bakeoff done → adopted UniverSat, dropped Clay + SEN2SR.** On 240 labelled chips
(4 genus classes, spatial-block CV macro-F1): UniverSat (center-token) + linear probe
≈ **0.55 ±0.17**, beating the conventional Sentinel-2 **Random Forest** baseline
(0.38) and the raw-spectral linear floor (0.45). RF underperformed the linear probe,
so the head stays **linear / MLP**. UniverSat ingests native 10 m chips
`(B, T=3, 10, 64, 64)` directly — **no super-resolution, no separate temporal head**.

Built (`src/cmrv/embeddings/`, heavy deps in the `embed` dependency group):
`Embedder` interface · `UniverSatEmbedder` (`g-astruc/UniverSat`, MIT, ~201 M, 768-d;
center-token for point labels, dense grid for inference) · `RawStatsEmbedder` baseline ·
`bakeoff` linear probe + `load_bakeoff_arrays`.

- [ ] Embedding extraction over all chips → persist embeddings (Zarr) as the durable artifact
- [ ] Light **linear / MLP** head on the frozen encoder → per-class logits
- [ ] Per-pixel uncertainty + Mahalanobis OOD on the embeddings

**VHR / resolution decision:** spine stays **S2-only (train + inference)** — temporal
consistency, no SR, UniverSat already wins on S2 alone. NGI 0.25 m ortho is free but
flown every 3–5 yr (static snapshot) → temporally misaligned with S2, unusable as an
inference input. SPOT 6/7 (1.5 m, **annual** SANSA mosaic, free for research) is the
better VHR *if* ever pursued — training-only enrichment via UniverSat's missing-modality
inference, deferred + validated separately.

## Embedding store (after backend chosen)
- [ ] Persist **embeddings** (Zarr cube) as the durable artifact; chips are transient cache
- [ ] Embedding manifest (sample → shard/offset, label, cover, fold)

## Deferred (designed, not built)

- [ ] **Region-aware months** — rainfall-seasonality zone layer → per-zone month set (WC = Feb/May/Sep now; add summer-rainfall calendars when other-province datasets land)
- [ ] **Cover gate** — flip `load_training_labels(min_cover_pct≈60)` on once cover-bearing data exists
- [ ] **Spatial-CV upgrades** — buffered/dead-zone folds, variogram-informed block size, leave-one-eco-region-out (before quoting accuracy)
- [ ] **Embedding store at scale** — Zarr → WebDataset shards; GEE→bucket compositing option for Vertex
- [ ] Temporal head training (mask missing months) + Mahalanobis OOD + wall-to-wall inference + demo viewer
- [ ] Re-add native/background label sources + native classes 8–11
- [ ] Lenses B (mine rehab), C (EUDR), D (biodiversity/bioacoustics)
