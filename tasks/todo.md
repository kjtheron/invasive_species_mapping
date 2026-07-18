# Phase 0 Prototype — Task Tracker

Conventions & architecture: [CLAUDE.md](../CLAUDE.md). Phase 0 builds **Lens A
(Invasive Alien Plants)** for the **Western Cape**, **locally** (no cloud bucket),
from **cover/abundance-bearing field datasets**, tuned to **WC phenology**. The
unified `western_cape_landcover` map classifies every pixel (IAP genera + native
biomes + transformed); "not-IAP" is ultimately an OOD/threshold call.

---

## Completed

- **AOI & tile grid** — `cmrv aoi-wc`, `aoi-tiles` → `data/aoi/*.parquet` (WC; tiles = inference unit)
- **S2 compositing** — inline in `ingest-chips` (training windows) + `infer` (inference boxes): MPC STAC → SCL mask → monthly median (no separate composite-to-disk step)
- **Label adapters (3)** — `labels-bioscape-ingest` (VegPlots), `labels-mapwaps-ingest`
  (Olifants-Doring IAP genera), `labels-sanlc-ingest` (SANLC 2018/20/22 accuracy points
  + VegMap biome). IAP membership from class-map `members[]`.
- **Store inspector** — `cmrv labels` (per-source counts + coord-uncertainty/cover coverage)
- **MapWAPS native/land-cover classes** — `_LULC_TO_CLASS` crosswalks all 23 MapWAPS
  classes to `western_cape_landcover` members (native→VegMap biome, transformed→land
  cover); `iap_only` dropped. Store: mapwaps 4818→27009. SANLC IAP-exclusion now
  buffers only species/genus rows (not the new native points).
- **Chip extraction** — `cmrv ingest-chips` (thin → 64×64 per obs×month, 10 km blocks,
  per-label window compute, incremental + self-reconcile to the thinned set)
- **Split** — `cmrv make-split` (iterative-stratification block folds on `class_id`,
  `--min-class-obs`, writes `split.parquet`). `western_cape_landcover`: 14 classes,
  2,448 obs, 12/13 classes present in all three folds.
- **Embedding** — `cmrv embed` (UniverSat center-token, 10 m per-pixel → single
  lon/lat-indexed Zarr cube, 2,454×768; DataLoader prefetch, `--device {cpu,cuda}`/`--amp`).
- **Head** — `cmrv train-head` (linear / MLP, on-the-fly balanced CE, per-class test
  metrics). **Linear macro-F1 0.60** on the spatial-block test fold (≈0.74 over the
  ≥10-support classes); pinus 0.94, prosopis 0.77, built_up/water 0.92. Linear > MLP.

---

## Next

1. **Wall-to-wall inference** (Stage 7, issue #3): tile → 3-month composite → UniverSat
   **dense** token grid → frozen head per token → per-pixel class + uncertainty +
   Mahalanobis OOD → triplet COGs in `data/outputs/` → viewer. Port chips.py's STAC
   retry/re-sign/download-fallback robustness into `composite.py` first.
2. **More training data — the biggest accuracy lever.** The set is small + imbalanced
   (rare classes have 1–4 test samples). BioSCape cover-bearing data is under embargo
   until **~Oct 2026**; source interim cover-bearing datasets meanwhile. Iterating
   loss/architecture won't move the needle — data will.

---

## Decisions on record

- **Bakeoff → UniverSat** (dropped Clay + SEN2SR): on 240 chips (spatial-block CV),
  UniverSat center-token + linear ≈ 0.55, beating a Sentinel-2 Random Forest (0.38)
  and the raw-spectral floor (0.45). Head stays linear/MLP; native 10 m, no SR.
- **VHR:** spine stays S2-only (train + inference) — temporal consistency, UniverSat
  wins on S2 alone. SPOT 6/7 (1.5 m annual) the better VHR *if* ever pursued
  (training-only via missing-modality inference), deferred.

## Deferred (designed, not built)

- [ ] **Region-aware months** — per-zone month set (WC = Feb/May/Sep; add summer-rainfall calendars when other-province data lands)
- [ ] **Cover gate** — flip `load_training_labels(min_cover_pct≈60)` on once enough cover-bearing data exists
- [ ] **Spatial-CV upgrades** — buffered/dead-zone folds, variogram-informed block size, leave-one-eco-region-out (before quoting accuracy)
- [ ] **Embedding store at scale** — single Zarr cube built; Zarr→WebDataset shards + GEE→bucket only when embeddings outgrow memory / for cloud-scale training (issue #8)
- [ ] Lenses B (mine rehab), C (EUDR), D (biodiversity/bioacoustics)
