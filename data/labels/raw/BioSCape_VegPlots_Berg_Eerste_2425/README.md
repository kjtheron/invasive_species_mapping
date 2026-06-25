# BioSCape Vegetation Surveys — Berg & Eerste River Catchments, 2022–2023

Field vegetation plots (Berg + Eerste, Western Cape) — high-quality IAP ground
truth: GPS + date + per-species cover %. Consumed by the
`labels-bioscape-ingest` adapter (`src/cmrv/labels/bioscape.py`) →
`source=bioscape_line` + `source=bioscape_plot` in the obs store.

## Provenance

- **Title:** BioSCape Vegetation Surveys Berg and Eerste River Catchments, South Africa, 2022-2023 (Version 1)
- **Authors:** Slimp, M., Malindi, J., Meyer, R., Cunningham, A., Rourke, J., Hayden, M., Hargey, A., Nesslage, J., Johnson, S., Rossi, M., & Stavros, N. (2025)
- **Publisher:** ORNL Distributed Active Archive Center (ORNL DAAC)
- **DOI:** https://doi.org/10.3334/ORNLDAAC/2425
- **Landing page:** https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-bioscape-vegplots-berg-eerste-2425-1
- **Access:** free NASA Earthdata Login (manual download)
- **Date accessed:** 2026-06-24

### Citation

> Slimp, M., Malindi, J., Meyer, R., Cunningham, A., Rourke, J., Hayden, M.,
> Hargey, A., Nesslage, J., Johnson, S., Rossi, M., & Stavros, N. (2025).
> *BioSCape Vegetation Surveys Berg and Eerste River Catchments, South Africa,
> 2022-2023* (Version 1). ORNL Distributed Active Archive Center.
> https://doi.org/10.3334/ORNLDAAC/2425

## Files

| Path | Used by adapter | Notes |
|------|-----------------|-------|
| `data/Berg_Eerste_Veg_SiteData.csv` | ✅ | plot-center GPS + date (joined into both emitters) |
| `data/Berg_Eerste_Veg_PlotCoverage.csv` | ✅ | per-quadrant species cover % → `bioscape_plot` |
| `data/Berg_Eerste_Veg_LineIntercept.csv` | ✅ | point-intercept transects → `bioscape_line` |
| `data/Berg_Eerste_Veg_Rarefaction2023.csv` | — | species-accumulation, not ingested |
| `data/Berg_Eerste_Veg_vegplotboundaries_{2022,2023}.kml` | — | plot polygons, not ingested |
| `comp/`, `guide/` | — | ORNL DAAC companion docs + user guide |
