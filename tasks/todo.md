# Phase 0 Prototype — Task Tracker

Conventions & architecture: [CLAUDE.md](../CLAUDE.md). Phase 0 builds **Lens A
(Invasive Alien Plants)**, **locally** (no cloud bucket), from **cover/abundance-bearing
field datasets**. Training extent is national SA (`aoi.train_path`) so KZN/EC labels
aren't clipped; the delivered map stays WC (`aoi.infer_path`). The unified
`sa_landcover` map classifies every pixel (IAP genera + native biomes + transformed);
under the IAP-only maps, "not-IAP" is an OOD/threshold call instead.

---

## Completed

- **AOI & tile grid** — `aoi-wc` / `aoi-sa` / `aoi-tiles`; grids in SA Albers equal-area.
- **S2 compositing** — inline in `ingest-chips` + `infer`: MPC STAC → SCL mask → monthly
  median. No separate composite-to-disk step.
- **Label adapters (3)** — `labels-bioscape-ingest` (VegPlots), `labels-mapwaps-ingest`
  (3 catchments), `labels-sanlc-ingest` (SANLC accuracy points + VegMap biome).
- **Store** — **37,516 obs** across WC/KZN/EC, all resolving under `sa_landcover`.
  Inspect with `cmrv labels`.
- **Region-aware months** — `months_by_zone` + `admin1_zone`; winter feb/may/sep,
  summer jul/sep/dec (Masemola et al. 2020, IJAEOG 93). Chips in per-image native S2
  UTM; inference warps to SA Albers so tiles mosaic.
- **Chip extraction** — `ingest-chips`: thin → 64×64 per obs×month, 10 km blocks,
  incremental + self-reconciling to the thinned set.
- **Memory-bounded chipping** — one `dask.compute` per 4 km sub-cell (`SUBCELL_M`) +
  `max_scenes_per_composite: 20`, so peak RSS no longer scales with labels-per-block.
  See [lessons.md](lessons.md) — this OOM-killed two multi-hour runs.
- **Split** — `make-split`: iterative-stratification block folds on `class_id`.
- **Embedding** — `embed`: UniverSat center-token → lon/lat-indexed Zarr cube.
- **Head** — `train-head`: linear / MLP, balanced CE, per-class test metrics.
  **Linear macro-F1 0.60** (≈0.74 over ≥10-support classes); pinus 0.94, prosopis 0.77,
  built_up/water 0.92. Linear > MLP.

---

## Next

1. **Re-chip the full store** — chips wiped 2026-07-20 (class scope changed to
   IAP + non-IAP nationally). ~19.3k thinned labels × 3 months; expect a long run.
   Then re-split on `sa_landcover` and retrain.
2. **⚠️ RE-RUN `ingest-chips` ONCE THE 2026-07-20 RUN FINISHES — to backfill
   skipped months.** That run was started *before* the SAS-refresh fix, so any
   month whose retries were exhausted mid-run was logged
   `all attempts failed — skipping month` and left out of the manifest. The
   re-run is cheap and safe: it's incremental, so it fetches **only** the missing
   months, and the fix means they should now succeed. Count what was lost with:
   ```bash
   grep -c "all attempts failed — skipping month" data/chips_run.log
   ```
   Then just `uv run cmrv ingest-chips --max-workers 15` again (no flags, no wipe).
3. **Wall-to-wall inference** (Stage 7, issue #3): tile → 3-month composite → dense
   UniverSat token grid → frozen head → class + uncertainty + Mahalanobis OOD →
   triplet COGs in `data/outputs/` → viewer.
4. **More training data — the biggest accuracy lever.** The set is small + imbalanced
   (rare classes have 1–4 test samples). BioSCape cover-bearing data is embargoed until
   **~Oct 2026**; source interim cover-bearing datasets meanwhile. Iterating
   loss/architecture won't move the needle — data will.

---

## Decisions on record

- **Bakeoff → UniverSat** (dropped Clay + SEN2SR): on 240 chips (spatial-block CV),
  UniverSat center-token + linear ≈ 0.55, beating a Sentinel-2 Random Forest (0.38) and
  the raw-spectral floor (0.45). Head stays linear/MLP; native 10 m, no SR.
- **VHR:** spine stays S2-only — temporal consistency, and UniverSat wins on S2 alone.
  SPOT 6/7 (1.5 m annual) the better VHR *if* ever pursued; deferred.
- **Training extent ≠ inference extent** — split into `aoi.train_path` (national) and
  `aoi.infer_path` (WC) in `pipeline.yaml`; both are read by the CLI, not decorative.
- **Chip everything, subset at training time** — `ingest-chips` consults no class map;
  `make-split --class-map-name` decides scope. Keeps the manifest the empirical record.

## Deferred (designed, not built)

- [ ] **Caching (perf)** — as the store + AOI grow:
  - *Observation store:* `read_all` rebuilds the whole store on every call — cache by
    (root, partition mtimes, sources, bbox). `summary` / `sanlc` ingest ride the same cache.
  - *Ingest:* skip a source partition whose raw inputs are unchanged (hash raw / figshare md5).
  - *STAC:* **uncacheable** — `_query_items` signs hrefs inline and MPC SAS tokens are
    ~1 h TTL. Splitting signing out of search isn't worth it. Leave uncached.
  - *Encoder forward:* training embeddings already persist to Zarr; an inference-time
    embedding cache is **not wanted** (decided).
- [ ] **Spatial-CV upgrades** — buffered/dead-zone folds, variogram-informed block size,
  leave-one-eco-region-out (before quoting accuracy publicly).
- [ ] **True rainfall-seasonality zones** (Schulze) where a province is mixed (EC south coast).
- [ ] **Embedding store at scale** — Zarr→WebDataset shards only when embeddings outgrow
  memory / for cloud-scale training (issue #8).
- [ ] Lenses B (mine rehab), C (EUDR), D (biodiversity/bioacoustics).
