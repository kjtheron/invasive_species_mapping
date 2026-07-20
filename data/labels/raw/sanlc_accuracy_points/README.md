# SANLC accuracy-assessment points (2018 / 2020 / 2022)

Field/reference-verified land-cover reference points — the independent truth used to
**validate** the SANLC national land-cover maps. We train on these (not on the SANLC
raster, which is the model's own prediction): each point is a checked land-cover
observation, and each carries a year that aligns with its imagery
(2018→2018 S2, 2020→2020, 2022→2022).

## Source

South African National Land-Cover (SANLC) accuracy-assessment reports, DFFE EGIS
(<https://www.dffe.gov.za/egis>). Downloaded zips:

| Year | Zip | Points (national) |
|------|-----|-------------------|
| 2018 | `SA_NLC_2018_Accuracy_Assessment_Report.zip`    | 6,570 |
| 2020 | `SA_NLC_2020_Accuracy_Assessment_Report.zip`    | 6,836 |
| 2022 | `SA_NLC_2022_11_Acc_Assessment_Report.zip`      | 7,500 |

License: free to use, cite DFFE. The points re-use many locations across years, so
~54% are exact duplicates (same location + class) once pooled.

## Layout

```
sanlc_accuracy_points/
  2018/  …accuracy_assessment_points*.shp  + report PDF + confusion_matrix.xlsx
  2020/  …accuracy_assessment_points*.shp  + report PDF + confusion_matrix.xlsx
  2022/  …Finalized_2022_SANLC_Accuracy_Points*.shp + report PDF + confusion_matrix.xlsx
```

Each `*.shp` is **EPSG:4326**. The class field name differs by year — `Acc_Cls_Na`
(2018) vs `Class_name` (2020/2022) — both hold the SANLC land-cover class name from a
shared 48-class vocabulary; the `*_integrity_index.shp` and PDFs are QA material, not
used. (The separate SANLC raster is intentionally **not** kept — these points supersede
it as the label source.)

## Ingest

`uv run cmrv labels-sanlc-ingest` → [`cmrv.labels.sanlc`](../../../../src/cmrv/labels/sanlc.py)
writes `source=sanlc` rows to `data/labels/processed/sanlc_accuracy_points/`. The adapter:

1. pools all three years, maps each point's class to our scheme via `ACC_CLASS_TO_CLASS`
   (collapsed transformed classes; natural vegetation → sentinel `NATURAL`);
2. de-duplicates points identical across years (same location + class, keep latest);
3. clips to the Western Cape AOI;
4. names natural points by **VegMap 2024** biome (`T_BIOME`) — SANLC's natural classes
   don't resolve the Cape biomes;
5. excludes points within 320 m of a known IAP field point (chip-leakage guard).

Feeds the unified `sa_landcover` class map alongside the IAP genera.
