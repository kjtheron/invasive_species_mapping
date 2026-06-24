# MapWAPS — Invasive Alien Plant map for the Olifants-Doring Catchments

Random-Forest classification of invasive alien plants in the Olifants-Doring
catchments (NW Western Cape) from 10 m Sentinel-2, plus the **field training
points** behind it. Downloaded via `download/mapwaps_olifants_doring.py`
(figshare public API, md5-verified; zips auto-extracted + deleted). Not yet
ingested — needs a `labels-mapwaps-ingest` adapter → `source=mapwaps` in the obs
store. The training-point `.shp` is the part useful for our training signal.

## Provenance

- **Title:** MapWAPS Invasive Alien Plant map for the Olifants-Doring Catchments
- **Authors:** Rebelo, A.J., Skosana, T.E., Cogill, L.S. (2025)
- **Publisher:** SUNScholar Data Repository (Stellenbosch University, figshare)
- **DOI:** https://doi.org/10.25413/sun.29958053 (v1, published 2025-08-21)
- **Landing page:** https://wrcwro01.arc.agric.za/eu/dataset/mapwaps-invasive-alien-plant-map-for-the-olifants-doring-catchments
- **Funder:** Water Research Commission (MapWAPS project)
- **Date accessed:** 2026-06-24
- Full raw figshare metadata (md5s, etc.): `_figshare_metadata.json`

### Citation

> Rebelo, A.J., Skosana, T.E., & Cogill, L.S. (2025). *MapWAPS Invasive Alien
> Plant map for the Olifants-Doring Catchments*. Stellenbosch University.
> https://doi.org/10.25413/sun.29958053

## ⚠️ License / conditions of use

The figshare metadata field says **CC BY 4.0**, but the dataset description text
says **CC-BY-SA** *and* adds a condition of use:

> "All academics using this dataset in any scientific publication shall offer
> first right of refusal for co-authorship on any manuscripts. Practitioners are
> free to use the dataset."

Treat as **share-alike + co-authorship offer expected for academic publication**.
Provided "as is", no warranty. Resolve the CC-BY vs CC-BY-SA ambiguity with the
authors before any publication.

## Files

```
OlifantsDoring_classification_1.tif / _2.tif   10 m RF IAP map (split, large)
OlifantsDoring_classification.lyrx             ArcGIS style for the map
Metadata_OlifantsDoring.pdf                    map metadata
OlifantsDoring_TrainingData_23Classes/         <-- field training points (.shp + sidecars)
  OlifantsDoring_trainingdata.shp              CRS: WGS84 / UTM 34S (EPSG:32734)
  MAPWAPS_Training Dataset_Metadata.pdf
OliDor_IAT_Map_Certainty/                       per-pixel map certainty polygons (.shp, WGS84)
```

## Training-point schema (`OlifantsDoring_trainingdata.dbf`)

CRS **EPSG:32734** (UTM 34S — matches our raster CRS).

| field | type | use |
|-------|------|-----|
| `LULC_Class` | C254 | **the class label** — 23 classes incl. pines, wattles, gums, poplars, *Solanum mauritianum*, plus non-IAP/land-cover. Crosswalk → 8-class `western_cape_iap`. |
| `DateTime` | D | observation date |
| `X`, `Y` | F | coordinates |
| `Age__m_` | F | stand age (m), where recorded |
| `Density___` | N | density class |
| `Id`, `Direction`, `Notes`, `Name`, `Distance__` | | survey metadata |

### Adapter notes (for later)

- 23 LULC classes → map only the IAP ones onto our `members[]`; the rest are
  native/land-cover → drop (Phase 0 is IAP-only) or hold as OOD negatives.
- Reproject UTM 34S → EPSG:4326 for the obs store (`gdf_to_obs_df` expects 4326).
- No per-point coord-uncertainty field; set a constant from the survey method
  (field-walked points — likely a few m; confirm in the training metadata PDF).
- Carry the license/co-authorship condition into `license` on each row.
