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
| 4 | NIAPS 2023 | DFFE / Working for Water | **all 9** | **manual** (`niaps.py` verifies) | `data/labels/raw/niaps_2023/` |

All MapWAPS catchments CC-BY 4.0 (Olifants-Doring also flags CC-BY-SA ambiguity).

## 4. NIAPS 2023 — MANUAL (SharePoint browser login)

**N**ational **I**nvasive **A**lien **P**lant **S**urvey, DFFE / Working for Water.
Aerial-survey polygons for **14 alien taxa across all of South Africa**, each carrying
`gridcode` = **percent density (1-100)** — confirmed against the "Density" legend on
the A0 map sheet. Underlies Kotze et al. (2025). ~1.27 M polygons in one GeoPackage.

The file is a **personal OneDrive share** and answers `403 Access denied` to any
unauthenticated client, so nothing can script it. Download in a browser:

- index page: https://sites.google.com/site/wfwplanning/assessment (under "Abundance") —
  this is a Google **Sites** index page; the file itself is on DFFE's SharePoint.
- direct: `.../2023 NIAPS GeoPackage.gpkg` on `environmentza-my.sharepoint.com`

Drop the files in `data/labels/raw/niaps_2023/`, then verify — a truncated 606 MB
GeoPackage opens fine and silently returns fewer polygons:

```bash
python3 download/niaps.py --sha    # size + sha256 of the .gpkg and the A0 PDF
```

**Licence: permission, not a licence.** Andrew Wannenburgh (Ecologist, Working for
Water) confirmed by email on 2026-08-11 that there are **"No restrictions"** on use.
He also stated DFFE "is not set up to provide data use agreements" and declined to
issue CC-BY or CC0. So there is written permission from the data custodian but **no
formal licence grant**, which is not the same thing as an open licence and does not
automatically flow to downstream users of a model trained on it. The email chain is
kept verbatim at `data/labels/raw/niaps_2023/NIAPS_2023.txt`. Cite DFFE / Working for
Water and Kotze et al. (2025); get a written licence before redistributing the data
itself or claiming an open licence on derived weights.

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
