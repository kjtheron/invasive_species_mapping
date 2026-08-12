# Label data downloads

Reproducible fetch of the IAP training datasets. Scripts here only
**download raw files** into `data/labels/raw/<dataset>/`. Conversion to the
unified GeoParquet observation store (`data/labels/processed/<dataset>/`) is done
later by per-dataset **adapters** in `src/cmrv/labels/`.

| # | Dataset | DOI / source | Province | How | Raw lands in |
|---|---------|--------------|----------|-----|--------------|
| 1 | MapWAPS Olifants-Doring | `10.25413/sun.29958053` | WC | `mapwaps.py` | `data/labels/raw/mapwaps_olifants_doring/` |
| 2 | MapWAPS Tugela | `10.25413/sun.25066151` | KZN | `mapwaps.py` | `data/labels/raw/mapwaps_tugela/` |
| 3 | MapWAPS uMzimvubu | `10.25413/sun.25050401` | EC | `mapwaps.py` | `data/labels/raw/mapwaps_umzimvubu/` |

All MapWAPS catchments CC-BY 4.0 (Olifants-Doring also flags CC-BY-SA ambiguity).

> **MapWAPS datasets omitted (broken upstream, verified 2026-07):**
> **Luvuvhu** (`10.25413/sun.25050314`) — its `TrainingData.zip` ships the *Tugela*
> shapefile by mistake (identical 5267 rows, folder named `Trainingdata_Tugela`),
> so it carries no Luvuvhu data. **Sabie-Crocodile** (`10.25413/sun.25050368`) —
> `TrainingData.zip` contains only a metadata PDF and an empty folder (no shapefile).
> Both are wired in `download/mapwaps.py`'s registry; re-enable if the authors re-upload.

## Run

```bash
python3 download/mapwaps.py                     # all catchments, figshare API, md5-verified
python3 download/mapwaps.py mapwaps_tugela      # one catchment
```

Only the field **TrainingData** + metadata are fetched; the large `AlienMap_*`
rasters (the RF prediction map — not a training label) are skipped. 
Then `uv run cmrv labels-mapwaps-ingest` (all catchments).

## Pipeline after raw download

`raw file → adapter (cols → observation schema; IAP membership from
class_maps.<name>.members[]; coord_uncertainty_m) → write_source_partition() →
data/labels/processed/<dataset>/ → cmrv labels → ingest-chips → make-split`.

New datasets must be added to `KNOWN_DATASETS` in
`src/cmrv/labels/observations.py` (and, for MapWAPS catchments, to the
`CATCHMENTS` registry in `src/cmrv/labels/mapwaps.py`).
