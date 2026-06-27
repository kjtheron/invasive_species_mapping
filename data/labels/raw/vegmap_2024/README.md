# VegMap 2024 — SANBI National Vegetation Map (NVM 2024)

*Potential* natural vegetation polygons for South Africa. In the native-vegetation
label sampler it supplies the **biome name** for pixels that SANLC says are
natural (SANLC = actual-cover mask, VegMap = biome name). Not yet ingested —
needs the `labels-sanlc-ingest` sampler.

## Provenance

- **Title:** National Vegetation Map 2024 (NVM2024 Final, IEM 5_12)
- **Publisher:** SANBI — South African National Biodiversity Institute
- **Source:** https://bgis.sanbi.org/Projects/Detail/2258
- **Version:** `NVM2024Final_IEM5_12_07012025` (released 2025-01)
- **License:** SANBI BGIS terms — free for research, cite the National Vegetation Map
- **Date accessed:** 2026-06-27

## Files

- `Shapefile/NVM2024Final_IEM5_12_07012025.shp` (+ sidecars) — **47,438 polygons**
- `Symbology/` — ArcGIS/QGIS style files (biome + veg-type symbology)

## Schema

- **CRS:** Albers Equal Area (`AEA_RSA_WGS84`)
- **`T_BIOME`** — biome name, **10 biomes** (the label granularity we use):
  Fynbos · Succulent Karoo · Nama-Karoo · Albany Thicket · Forests · Grassland ·
  Savanna · Azonal Vegetation · Indian Ocean Coastal Belt · Desert
- `T_BIOMEID` — biome id · `T_BIOREGIO` — bioregion (~finer than biome) ·
  `T_Name` — veg type (~450, too fine for 10 m S2) · `T_MAPCODE` · `T_SUBTYPNM`

## Sampler notes

- Collapse to **`T_BIOME`**. In the WC that's mostly Fynbos, Succulent Karoo,
  Nama-Karoo, Albany Thicket, Forests, Azonal. **Renosterveld folds into Fynbos**
  at biome level — use `T_BIOREGIO` if it should be separate.
- VegMap is *potential* vegetation → only label pixels SANLC marks **natural**;
  sample polygon **interiors** (erode) to avoid 1:250k-vs-20 m boundary noise.
