# SANLC 2022 — South African National Land Cover (2022)

20 m national land-cover raster — the **actual-cover** truth for the native-veg
sampler. It decides natural vs transformed (cultivated / plantation / built /
bare / water) and supplies the transformed classes directly. Not yet ingested —
needs the `labels-sanlc-ingest` sampler.

## Provenance

- **Title:** South African National Land Cover 2022 (Albers)
- **Publisher:** DFFE — Dept. of Forestry, Fisheries & the Environment
- **Source:** https://www.dffe.gov.za/egis
- **License:** DFFE EGIS terms — free, cite DFFE
- **Date accessed:** 2026-06-27

## File

- `SA_NLC_2022_ALBERS.tif` — uint8, **20 m**, Albers (WGS84), 86309 × 72429
- `SA_NLC_2022_ALBERS.tif.vat.dbf` — value attribute table: `Value`, `Class_Name`,
  `SALCC_1` / `SALCC_2` (grouped class schemes), `Count`

## Classes

~73 detailed classes via **`Class_Name`**, e.g.:
- **natural** — `low shrubland (fynbos)`, `low shrubland (succulent karoo)`,
  `low shrubland (nama karoo)`, `natural grassland`, `contiguous (indigenous)
  forest`, `open woodland`, `natural rivers/lakes/estuaries`
- **transformed** — plantation forest, cultivated, built-up/urban, mines, bare,
  artificial dams

## Sampler notes

- Use `Class_Name` (or grouped `SALCC_*`) to split **natural vs transformed**.
- Natural pixels → VegMap `T_BIOME`; transformed pixels → the SANLC class group;
  IAP pixels → the IAP model ⇒ one unified land-cover map.
- SANLC and VegMap are both Albers → reproject to UTM 34S / EPSG:4326 in the sampler.
