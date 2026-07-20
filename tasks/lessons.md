# Lessons Learned

Running log of mistakes made and the rules that prevent recurrence. Reviewed at the start of every session.

---

## 2026-04-16 — Reprojection round-trip inflates `total_bounds`

**Mistake:** `build_tile_grid` used `gdf.intersects(aoi_union)` to filter candidate tiles. When the AOI came in already reprojected (e.g. a user-supplied WGS84 GeoJSON that then gets reprojected to UTM), the reprojected polygon's bbox drifted by ~1 m past the original extent. `intersects` returned True for sliver tiles that only touched the AOI by a fraction of a meter, producing ghost tiles in the grid.

**Rule:** When filtering a grid against a reprojected polygon, always use an area-fraction threshold, never a boolean predicate like `intersects` or `touches`. The default in this repo is `min_overlap_frac=0.01` (1% of tile area). This also correctly drops genuine tiny-edge slivers that would waste compute downstream.

**Why this matters:** Any `to_crs()` → `to_crs()` round-trip introduces sub-meter drift because rectangles become curvilinear quads. Two pipelines hit this silently: user-supplied AOIs in WGS84, and DWS REST responses that arrive in WGS84 and get converted to UTM 34S for tiling.

---

## 2026-04-25 — PC SDK `sign_url` short-circuits on already-signed hrefs

**Mistake:** Long-running `cmrv ingest-chips` (>25h) failed with 403 `AuthenticationFailed` on MPC asset downloads. Retry paths called `pc.sign_inplace(item)` to refresh tokens, but the failing URLs kept the same expired `st`/`se` params — re-signing was a no-op.

**Rule:** Before re-signing an item with `pc.sign_inplace`, strip the SAS query from each asset href. `planetary-computer` 1.0.0 `sas.py:142-145` — `sign_url` returns the URL unchanged if any of `st`/`se`/`sp` is present in the query string, so the cached SAS token never gets a chance to refresh.

```python
for asset in item.assets.values():
    p = urlparse(asset.href)
    if p.netloc.endswith(".blob.core.windows.net"):
        asset.href = urlunparse(p._replace(query=""))
pc.sign_inplace(item)
```

**Why this matters:** PC SAS tokens have a finite TTL (~25h observed for sentinel-2-l2a). Items signed at month-query time are downloaded much later in long jobs. Without the strip-then-sign pattern, every retry/fallback path appears to refresh but actually keeps reusing the dead token. Applies to any long-running pipeline that signs items once upstream and consumes them downstream — sign just-in-time at the consumer (e.g. inside `_download_item_assets`) for an extra safety layer.

---

## 2026-04-26 — Class crosswalk: single source of truth in `class_maps.<name>.members[]`

**Mistake:** Three places encoded parts of the same crosswalk: `species_map` (string→class), `gbif.taxa[]` (download list with `class_id` per taxon), and `class_maps.<name>` (id → semantics). Adding a new species meant editing 2-3 blocks. Genus fallback was implicit (any `species_map` key without a space acted as a genus key) and silently swept all species in the genus into one class.

**Rule:** Class crosswalk lives only in `class_maps.<name>.<id>.members[]`. Each class declares its scientific binomials. Genus fallback is opt-in per class via `genus_fallback: true` + an explicit `genus:` field — default off. The GBIF download taxa list and the runtime species→class lookup both derive from `members[]` via `cmrv.labels.classmap.build_lookup` / `gbif_taxa_from_schema`. `species_map` and `gbif.taxa[]` blocks are removed.

**How to apply:** When adding a new species to training, append the canonical binomial to the relevant class's `members[]`. Run `just audit-labels` after to confirm no observations dropped silently — pre-chip stage lists every unmapped (source, species) pair sorted by count; post-fuse stage flags raster anomalies. Validation in `build_lookup` is warn-only, so duplicates and conflicts surface in logs without breaking the pipeline.

**Why this matters:** Adding a new ingest source previously meant: chip the data, watch chips.py warn `N rows dropped`, grep the warning for species names, edit `species_map`, edit `gbif.taxa[]`, re-run. The audit subcommand makes the unmapped set explicit and the single-block edit makes it one change instead of three.

---

## 2026-04-26 — `fuse.py` is viz-only, not a training input

**Mistake:** Earlier roadmap drafts (since removed) assumed dense per-pixel segmentation training that consumed `label.tif` rasters via a DataLoader.  The implementation pivoted to a chip / point regime (`ingest-chips` → 64×64 chips per obs_id × 4 months → `make-split` → embedding-patch lookup at obs coordinate) but the docstrings, justfile recipe, and audit code didn't say so.  Running `labels-fuse` looked like a required step it isn't.

**Rule:** `cmrv labels-fuse` produces sparse label COGs for **visualization only** (QGIS / Streamlit overlays).  Training consumes the unified observation store directly via `cmrv.labels.classmap.build_lookup`, not via raster pixel values.  When auditing label coverage for an actual training run, use `cmrv labels-audit-classmap --stage pre` (no fuse needed).  The `--stage post` and `--stage both` modes only matter if you want to sanity-check the viz raster.

**How to apply:** Don't add `labels-fuse` to required setup steps.  Don't gate training on `label.tif` existing.  If a future model genuinely needs dense pixel labels (U-Net student, semantic segmentation experiment), reinstate fuse as a training input — until then, point regime + classmap is the path.

**Why this matters:** Phase 0 IAP labels are sparse points to begin with.  Rasterizing them with a 10 m point buffer fabricates ~16 "labeled" pixels per real observation, which is fine for *visualization* but biases pixel-uniform sampling for training.  The chip-centred design avoids that fabrication entirely.

---

## 2026-04-26 — Chip everything; subset at training time, not at chip time

**Mistake:** Built up a class-map-centric audit (`labels-audit-classmap`) that re-derived "what *would* be chipped under filters" from the obs store.  That duplicated work the manifest already records, gated chipping decisions on a schema the chip extractor doesn't actually consult, and forced two different "what observations exist" answers (audit vs manifest) that drifted apart.

**Rule:** The pipeline is **ingest → chip everything → manifest → subset for training**.  Names are already standardized in the obs store (GBIF backbone), so the chip extractor doesn't need to know about classes — it chips every obs that survives basic AOI / coord / date filters.  Class assignment happens *only* when forming a training subset via `cmrv make-split`.  Exploration ("what's in my chips, and is it balanced?") reads `manifest.parquet` directly via `cmrv chips-stats` — no schema, no class_map, no obs-store re-derivation.

**How to apply:** New label source?  Ingest it, run `ingest-chips`, run `chips-stats`.  Want to train on a specific species set?  `make-split --species [...]` filters the manifest; pass `--class-map` only if you want `class_id` columns assigned.  When `--species` is explicit, unmapped species are *kept* (warning only) — class_map is a labelling shim, not a gatekeeper.

**Why this matters:** The manifest is the empirical record of what got chipped.  Treating it as the source of truth for exploration eliminates an entire category of "audit says X, manifest says Y" bugs (the NLC `class_crosswalk` mismatch and the AOI confusion both came from re-deriving instead of reading).  Four CLI verbs cover the path: `labels-ingest → ingest-chips → chips-stats → make-split`.

---

## `ingest-chips` is additive — reconcile the manifest against the current thinned set

**Rule:** Chipping only ever *adds*. When the label store changes (a source re-ingested, a `iap_only` filter, or deterministic thinning picking a different one-per-cell representative), a prior run's chips for now-dropped obs stay on disk + in the manifest. The manifest silently becomes a **superset** of the canonical one-rep-per-(species,20 m-cell) set — near-duplicate training samples concentrated in whatever source churned (here: +741 obs, 66% Pinus).

**How to apply:** `extract_training_chips` captures the passed-in thinned set as `canonical_obs` *before* incremental filtering, and `_reconcile_manifest` prunes any manifest obs (and its chip files + emptied dirs) outside it at the end of a top-level run — gated off the year-fallback recursion. Re-running `ingest-chips` is now self-healing and reproducible; it also runs on the "all already chipped — nothing to do" path. Disk-only, no re-download.

**Why this matters:** Spatial-block CV stops near-dups leaking across folds, so this never inflates held-out accuracy — but it skews class balance + per-class metrics toward the churned source and makes the manifest a function of run *history*, not current inputs. "manifest == chips on disk" was always true; the real invariant is "manifest == current thinned set".

---

## The chip→encoder contract: scale ÷10000 and fill NaN before UniverSat

**Rule:** Chips are stored as raw S2 DN with cloud-masked pixels left as NaN (so
`valid_frac` is honest). UniverSat (and any ViT) NaN-poisons — one NaN token makes a
mean-pooled vector NaN — and expects reflectance, not DN. So the loader, not the
chipper, owns the encoder contract: `load_chip_arrays` defaults `scale=1/10000` and
`np.nan_to_num(..., nan=0.0)` after scaling, and `UniverSatEmbedder.embed` asserts
`torch.isfinite` after the forward. `RawStatsEmbedder` hid this (it uses `nanmean`),
so the bakeoff looked fine while the production path would have trained on NaN.

**How to apply:** Never feed stored chips straight to an encoder — scale + fill first.
SCL mask now also drops 0 (no-data) and 1 (saturated/defective); `MIN_VALID_FRAC` is
0.5 and `load_chip_arrays(min_valid_frac=…, split=…)` filters poor chips + honors the
train/val/test fold files (the split is only useful if the loader reads it).

**Why this matters:** A silent NaN/scale bug degrades the *adopted* model while the
baseline masks it — the worst kind of regression. Also: `read_all` now reads only
dataset partition files (`root/<dataset>/*.parquet`), never root-level `summary.parquet`,
which was inflating per-source counts by 1 and injecting a geometry-less phantom row.

---

## 2026-07-20 — Size the work unit by cost, not by geography

**Mistake:** `ingest-chips` OOM-killed twice (7.0 GB, then 9.2 GB RSS on a 15 GB box),
hours into runs, on *some* blocks but not others. The work unit is a
`(block_id, year, zone)` group — a 10 km square — and every label in it went into one
`dask.compute`. Dask holds each root chunk resident until its last dependent window
finishes, so a block whose labels are scattered across its full extent materialises the
**dense** cube, not just the 64×64 windows. Peak was therefore
`bbox_area × n_bands × n_scenes × 4` where the first two terms are chosen by the *data*.
Under BioSCape (83 plots province-wide) groups held ~1–3 points and the bbox collapsed
to a few hundred metres. Under MapWAPS (36k survey points; one block held 558) the bbox
is the whole block — ~2.6 GB per worker, × 6 workers. Nothing in the code changed; the
input distribution did. The earlier `chunksize 1024→256` fix reduced per-window read
amplification but not the aggregate, because the ceiling is the whole cube either way.

**Rule:** Any batched compute over spatially-distributed points must bound its own
extent. Batch by a fixed spatial cell (`SUBCELL_M`, 4 km), one `dask.compute` per cell,
so bbox — and therefore peak RSS — is a constant the code picks. 5 points or 500 in a
cell now cost the same. Bound the time axis too: `max_scenes_per_composite` keeps the N
least-cloudy scenes, since a permissive `cloud_cover_max: 95` plus ±15 d padding pulls
60+ scenes into one median.

**How to apply:** Query STAC once per group-month over the group bbox (items are just
metadata), then restack per cell with narrower `bounds_latlon` — same network cost,
bounded memory. `_check_compute_size` raises above `MAX_COMPUTE_BYTES` (2 GiB) so a
mis-sized stack fails in seconds instead of OOM-ing at hour four. When a cost is
data-dependent, the guard belongs at the point of allocation, not in a tuned constant.

**Why this matters:** Density-dependent memory is invisible until a denser dataset
lands, then it costs multi-hour runs and looks like a flaky bug — some chips work, some
don't. The next survey will be denser still. Also: label counts per block were skewed
186× (median 3, max 558) and nothing surfaced that, so a distribution check on a new
source is worth more than it looks.

---

## Never `to_zarr()` to a path you opened lazily — it corrupts the store

**Rule:** `xr.open_zarr(p)` is lazy. Doing `ds = open_zarr(p); ds2 = ds.assign_coords(...); ds2.to_zarr(p, mode="w")` overwrites chunks while they're still being read back to write — it silently shreds the data (here: `emb` filled with NaN, `obs_id` emptied). zarr-v3 strings are fine; the in-place rewrite was the bug.

**How to apply:** Write the cube **complete in one pass** (emb + obs_id + coords together, as `embed_chips` now does) so no post-hoc patch is needed. If you must add to an existing store, write to a *new* path and replace, or `.load()` into memory first to break the lazy link. Re-deriving row→obs_id order to "recover" a coord-less cube is unsafe (silent label misalignment) — re-embed instead.

**Why this matters:** A corrupted training cube trains on garbage with no error. The empty-fold guard in `train_head` now turns the downstream symptom (obs_id join → empty fold) into a loud failure instead of a cryptic `argmax` error.
