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
  cover); `iap_only` dropped. SANLC IAP-exclusion now buffers only species/genus rows.
- **MapWAPS multi-catchment (SA-wide)** — generic adapter + `CATCHMENTS` registry
  (per-catchment cols/CRS/date) over Olifants-Doring (WC) + Tugela (KZN) + uMzimvubu
  (EC); `download/mapwaps.py` fetches TrainingData from figshare. New classes:
  `renosterveld` (split from fynbos), `savanna` (Indigenous Bush_*), `populus_spp`
  (Alien_Poplar). Store **37,516 obs** across 3 provinces, all resolve under
  `western_cape_landcover`. Luvuvhu (dup of Tugela) + Sabie-Croc (empty) broken upstream.
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

- [x] **Region-aware months per province** — DONE. `pipeline.yaml` `months_by_zone`
  (winter_rainfall feb/may/sep; summer_rainfall **jul/sep/dec**) + `admin1_zone`; `ingest-chips`
  tags each label's zone from its province and composites per-zone; `embed` reads each obs's
  own months from the manifest (per-obs day-of-year). Summer months from Masemola et al. 2020
  (IJAEOG 93): dry-winter senescence / Acacia flowering / peak summer growth, >80% each. AOI
  expanded to national SA (`aoi-sa`, `fetch_provinces`) so KZN/EC labels aren't clipped; tile+
  block grids in **SA Albers equal-area** (`cmrv.aoi.SA_ALBERS`), not UTM 34S. Chip pixel grid
  is **per-image native S2 UTM zone** (`utm_epsg` per group; manifest stores lon/lat, so embed
  is CRS-agnostic); inference composites+embeds in the native zone then warps the output map to
  SA Albers (`infer_box(out_crs=SA_ALBERS)`) so tiles mosaic. *Deferred:* a true rainfall-
  seasonality-zone layer (Schulze) where a province is mixed (EC south coast).
- [ ] **Caching (perf)** — memoise repeated / expensive work as the store + AOI grow:
  - *Observation store:* `observations.read_all` rebuilds the whole store (`gpd.read_parquet` per
    partition + `pd.concat`) on every call — cache by (root, partition mtimes, sources, bbox);
    invalidate when a partition file changes. `summary`/`write_summary` and `sanlc` ingest each
    re-read via `read_all` — they'd ride the same cache.
  - *Ingest:* skip re-ingesting a source partition whose raw inputs are unchanged (hash the raw
    file(s) / figshare md5 → skip `write_partition` if the partition already reflects them).
  - *STAC:* cache `_query_items` results (STAC item search is the slow, rate-limited step) keyed by
    (bbox, date-window, cloud_cover_max); optionally cache signed asset hrefs within their TTL.
  - *Encoder forward:* the UniverSat forward is the dominant inference cost and is deterministic for
    a frozen encoder — cache dense-token embeddings per (tile, month-set, year) so re-runs
    (new head, threshold sweep, TTA compare) skip re-embedding; keys off the composite, not the head.
- [ ] **Spatial-CV upgrades** — buffered/dead-zone folds, variogram-informed block size, leave-one-eco-region-out (before quoting accuracy)
- [ ] **Embedding store at scale** — single Zarr cube built; Zarr→WebDataset shards + GEE→bucket only when embeddings outgrow memory / for cloud-scale training (issue #8)
- [ ] Lenses B (mine rehab), C (EUDR), D (biodiversity/bioacoustics)
