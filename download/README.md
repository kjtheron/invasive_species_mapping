# Label data downloads

Reproducible fetch of the IAP occurrence/training datasets. Scripts here only
**download raw files** into `data/labels/<dataset>/`. Conversion to the unified
GeoParquet observation store (`data/labels/wc/obs/source=<src>/`) is done later
by per-dataset **adapters** in `src/cmrv/labels/` — written once the real files
are present and their columns are known.

| # | Dataset | DOI / source | License | How | Raw lands in |
|---|---------|--------------|---------|-----|--------------|
| 1 | BioSCape VegPlots (Berg+Eerste) | `10.3334/ORNLDAAC/2425` | NASA ORNL DAAC (free Earthdata login) | **manual** | `data/labels/BioSCape_VegPlots_Berg_Eerste_2425/data/` |
| 2 | MapWAPS Olifants-Doring | `10.25413/sun.29958053` | CC-BY 4.0 | `mapwaps_olifants_doring.py` | `data/labels/mapwaps_olifants_doring/` |
| 3 | BioSCape iNaturalist project | iNat `bioscape-invasive-alien-tree-mapping-project` | per-obs (CC-BY-NC / CC0) | `bioscape_inat.py` | `data/labels/bioscape_inat/` |

## Run

```bash
python3 download/mapwaps_olifants_doring.py   # figshare public API, md5-verified
python3 download/bioscape_inat.py             # iNaturalist API v1, NDJSON
```

Both use only the Python stdlib (no project venv, no API key).

## 1. BioSCape VegPlots — MANUAL (Earthdata login)

ORNL DAAC files are behind a **free NASA Earthdata Login** (not an API key).
Download these three CSVs and drop them in
`data/labels/BioSCape_VegPlots_Berg_Eerste_2425/data/` (the
`labels-bioscape-ingest` adapter already expects this path + filenames):

- `Berg_Eerste_Veg_SiteData.csv`
- `Berg_Eerste_Veg_PlotCoverage.csv`
- `Berg_Eerste_Veg_LineIntercept.csv`

Landing page: https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-bioscape-vegplots-berg-eerste-2425-1
(or DOI https://doi.org/10.3334/ORNLDAAC/2425). Adapter already exists →
`uv run cmrv labels-bioscape-ingest`.

## Pipeline after raw download

`raw file → adapter (cols → observations.SCHEMA; IAP membership from
class_maps.<name>.members[]; coord_uncertainty_m) → write_source_partition() →
data/labels/wc/obs/source=<src>/ → cmrv labels → ingest-chips → make-split`.

New sources (MapWAPS, iNat) must be added to `KNOWN_SOURCES` in
`src/cmrv/labels/observations.py` when their adapters are written.
