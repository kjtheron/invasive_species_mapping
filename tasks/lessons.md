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

**Mistake:** Earlier roadmap drafts (`Phase_0_Build_Roadmap.md` §3 Stage 4, §7, line 472) assumed dense per-pixel segmentation training that consumed `label.tif` rasters via a DataLoader.  The implementation pivoted to a chip / point regime (`ingest-chips` → 64×64 chips per obs_id × 4 months → `make-split` → embedding-patch lookup at obs coordinate) but the docstrings, justfile recipe, and audit code didn't say so.  Running `labels-fuse` looked like a required step it isn't.

**Rule:** `cmrv labels-fuse` produces sparse label COGs for **visualization only** (QGIS / Streamlit overlays).  Training consumes the unified observation store directly via `cmrv.labels.classmap.build_lookup`, not via raster pixel values.  When auditing label coverage for an actual training run, use `cmrv labels-audit-classmap --stage pre` (no fuse needed).  The `--stage post` and `--stage both` modes only matter if you want to sanity-check the viz raster.

**How to apply:** Don't add `labels-fuse` to required setup steps.  Don't gate training on `label.tif` existing.  If a future model genuinely needs dense pixel labels (U-Net student, semantic segmentation experiment), reinstate fuse as a training input — until then, point regime + classmap is the path.

**Why this matters:** Phase 0 IAP labels are sparse points to begin with.  Rasterizing them with a 10 m point buffer fabricates ~16 "labeled" pixels per real observation, which is fine for *visualization* but biases pixel-uniform sampling for training.  The chip-centred design avoids that fabrication entirely.

---

## 2026-04-26 — Chip everything; subset at training time, not at chip time

**Mistake:** Built up a class-map-centric audit (`labels-audit-classmap`) that re-derived "what *would* be chipped under filters" from the obs store.  That duplicated work the manifest already records, gated chipping decisions on a schema the chip extractor doesn't actually consult, and forced two different "what observations exist" answers (audit vs manifest) that drifted apart.

**Rule:** The pipeline is **ingest → chip everything → manifest → subset for training**.  Names are already standardized in the obs store (GBIF backbone), so the chip extractor doesn't need to know about classes — it chips every obs that survives basic AOI / coord / date filters.  Class assignment happens *only* when forming a training subset via `cmrv make-split`.  Exploration ("what's in my chips, and is it balanced?") reads `manifest.parquet` directly via `cmrv chips-stats` — no schema, no class_map, no obs-store re-derivation.

**How to apply:** New label source?  Ingest it, run `ingest-chips`, run `chips-stats`.  Want to train on a specific species set?  `make-split --species [...]` filters the manifest; pass `--class-map` only if you want `class_id` columns assigned.  When `--species` is explicit, unmapped species are *kept* (warning only) — class_map is a labelling shim, not a gatekeeper.

**Why this matters:** The manifest is the empirical record of what got chipped.  Treating it as the source of truth for exploration eliminates an entire category of "audit says X, manifest says Y" bugs (the NLC `class_crosswalk` mismatch and the AOI confusion both came from re-deriving instead of reading).  Four CLI verbs cover the path: `labels-ingest → ingest-chips → chips-stats → make-split`.
