# BioSCape Invasive Alien Tree Mapping Project — iNaturalist observations

Community-science occurrence points collected as training data for the
spring-2023 Cape Floristic Region invasive-alien-tree classification. Downloaded
via `download/bioscape_inat.py` (iNaturalist API v1). Not yet ingested — needs a
`labels-inat-ingest` adapter → `source=inat` in the obs store.

## Provenance

- **Project:** BioSCape Invasive Alien Tree Mapping Project
- **Slug / URL:** `bioscape-invasive-alien-tree-mapping-project` — https://www.inaturalist.org/projects/bioscape-invasive-alien-tree-mapping-project
- **Source API:** iNaturalist API v1, `GET /v1/observations?project_id=<slug>`
- **Supported by:** NASA BioSCape ("Impacts of Invasive Alien Species on Biodiversity and Ecosystem Functioning")
- **Date accessed:** 2026-06-24
- **License:** per-observation `license_code` (see caveat below — mostly null)

## File

- `observations.ndjson` — one full raw iNaturalist observation per line (1,373 records).

## Snapshot (at download)

- **1,373 observations**, all geolocated (`geojson` Point), `positional_accuracy` present, **median ~3 m**.
- **Date range:** 2023-10-19 → 2025-07-11.
- **Quality grade:** 358 research · 828 needs_id · 187 casual.
- **Top taxa:** Plantae 329, *Eucalyptus* 195, *Pinus* 155, *Acacia mearnsii* 120, *Hakea sericea* 66, *Pinus pinaster* 44, *Acacia saligna* 36, *Acacia cyclops* 30, *Pinus radiata* 26, …

## Caveats for the adapter

- **License:** 1,371 of 1,373 have `license_code: null` (all-rights-reserved); only 2 are `cc-by-nc`. Treat as not openly licensed — gate/decide use accordingly; carry `license_code` through to the obs store.
- **Coarse IDs:** many records resolve only to genus or higher (*Eucalyptus*, *Pinus*, `Plantae`). Genus-level → `pinus_spp` / `eucalyptus_spp` via `genus_fallback`; anything above genus (Plantae, Tracheophyta, Angiospermae) and the **7 no-taxon** records are unusable — drop.
- **Quality grade:** consider filtering to `research` (358) or gating non-research-grade.
- **Schema:** taxon at `obs.taxon.name`, coords at `obs.geojson.coordinates` ([lon, lat], WGS84), accuracy at `obs.positional_accuracy`, date at `obs.observed_on`.
