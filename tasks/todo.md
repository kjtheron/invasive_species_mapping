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

## Active — Stage 5–6: Embedding bakeoff (Clay+SR vs UniverSat)

Pick the embedding backend **before** building the spine's embed stage. Keep it
**encoder-agnostic** so the loser is a clean delete.

**Interface (so Clay+SR is removable):** an `Embedder` protocol —
`embed(stack[T,C,H,W], dates) -> np.ndarray[N, D]`. All SEN2SR code lives *inside*
the Clay backend; adopting UniverSat = delete `sr/` + the Clay backend, flip the
default. Chips / manifest / make-split stay untouched (already encoder-agnostic).

Backends:
- **ClaySR** — SEN2SR (`tacofoundation/RS-SR-LTDF`) 10 m→2.5 m → frozen Clay v1.5
  (patch 8 → 32×32 tokens). The SR stage exists only to feed Clay fine pixels.
- **UniverSat** (`g-astruc/UniverSat`, MIT, ~201 M, 768-d) — ingests native 10 m S2
  time-series + dates directly (no SR), `output_grid` set at inference; subsumes the
  temporal head. Our 64×64@10 m chips feed it as `(B, T=3, 10, 64, 64)` as-is.

Protocol (does **not** need the province AOI — uses the labels' own extent):
- [ ] Pull a small real-S2 chip set for the stored labels (label-bbox AOI, Planetary Computer)
- [ ] `Embedder` interface + both backends (`uv add` torch + clay + sen2sr + universat)
- [ ] Embed all chips with both; linear probe on `western_cape_iap_genus`; compare macro-F1 / separability
- [ ] Decide → if UniverSat wins: delete `sr/` + Clay backend, set UniverSat default, update roadmap + docs

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
