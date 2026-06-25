# Label data downloads

Reproducible fetch of the IAP training datasets. Scripts here only
**download raw files** into `data/labels/raw/<dataset>/`. Conversion to the
unified GeoParquet observation store (`data/labels/processed/<dataset>/`) is done
later by per-dataset **adapters** in `src/cmrv/labels/`.

| # | Dataset | DOI / source | License | How | Raw lands in |
|---|---------|--------------|---------|-----|--------------|
| 1 | BioSCape VegPlots (Berg+Eerste) | `10.3334/ORNLDAAC/2425` | NASA ORNL DAAC (free Earthdata login) | **manual** | `data/labels/raw/BioSCape_VegPlots_Berg_Eerste_2425/data/` |
| 2 | MapWAPS Olifants-Doring | `10.25413/sun.29958053` | CC-BY 4.0 / CC-BY-SA (ambiguous) | `mapwaps_olifants_doring.py` | `data/labels/raw/mapwaps_olifants_doring/` |

> iNaturalist project points were evaluated and **dropped**: the public API
> returns geoprivacy-obscured coordinates (~29 km), unusable for pixel-level
> training. Re-add only via authenticated curator export.

## Run

```bash
python3 download/mapwaps_olifants_doring.py   # figshare public API, md5-verified
```

Stdlib only (no project venv, no API key).

## 1. BioSCape VegPlots — MANUAL (Earthdata login)

ORNL DAAC files are behind a **free NASA Earthdata Login** (not an API key).
Download these three CSVs into
`data/labels/raw/BioSCape_VegPlots_Berg_Eerste_2425/data/` (the
`labels-bioscape-ingest` adapter expects this path + filenames):

- `Berg_Eerste_Veg_SiteData.csv`
- `Berg_Eerste_Veg_PlotCoverage.csv`
- `Berg_Eerste_Veg_LineIntercept.csv`

Landing page: https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-bioscape-vegplots-berg-eerste-2425-1
(DOI https://doi.org/10.3334/ORNLDAAC/2425). Then `uv run cmrv labels-bioscape-ingest`.

## Pipeline after raw download

`raw file → adapter (cols → observation schema; IAP membership from
class_maps.<name>.members[]; coord_uncertainty_m) → write_source_partition() →
data/labels/processed/<dataset>/ → cmrv labels → ingest-chips → make-split`.

New sources must be added to `KNOWN_SOURCES` in
`src/cmrv/labels/observations.py` when their adapters are written.
