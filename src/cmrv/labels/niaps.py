"""NIAPS 2023 ingest — national IAP polygons distilled to pure-core points.

**These polygons are not observations.** Kotze et al. (2025) built them by taking
47,830 field-surveyed 1 ha plots and extrapolating: "All pixels in the HEU were
examined and those pixels that **matched** the sample point were **assumed** to be
invaded by the same taxa at the same cover." The match was made against Sentinel-2.

So training a Sentinel-2 model on them is **distillation of ARC's classifier**, not
learning from ground truth, and the ceiling is that classifier's accuracy. Two
consequences are wired in here and must not be undone:

- ``weight = 0.5`` — these inform, they do not outvote observed field data.
- ``basis_of_record = "NIAPS_S2_EXTRAPOLATED"`` — so every downstream consumer can
  separate them. **Report test metrics per source**; a model evaluated on NIAPS is
  only being measured against the rule that generated NIAPS.

The plot data behind it would be far better (1 ha, observer-estimated cover, two
epochs) but is "in the possession of Johann Kotze and currently not publically
available".

Distillation, in the order that keeps it cheap — attribute filters cut 1.27 M
polygons to ~61 k before any spatial work runs:

1. ``gridcode >= min_density`` — gridcode is percent density (1-100), per the A0
   map legend. Dense stands are also where the extrapolation is most confident.
2. ``area >= min_area_ha`` in equal-area SA Albers.
3. make-valid, explode, keep polygonal parts only.
4. **Negative buffer** — the load-bearing step. It enforces a purity margin around
   the centre *and* removes linear/sliver geometry for free: a 1 ha riparian reed
   bed 300 x 30 m erodes to nothing, a 1 ha compact stand survives.
5. Drop any polygon overlapping a different taxon — no mixed-taxon centres.
6. ``point_on_surface`` (never centroid: that can fall outside a concave polygon).

The chip is 640 x 640 m = 40.96 ha and the median polygon is 0.25 ha, so a pure
*chip* is unreachable and is not the goal. The label describes the **centre**; the
surrounding context is what the encoder sees but nothing claims it is pure. That is
exactly how the MapWAPS point labels already behave.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import geopandas as gpd  # type: ignore
import pandas as pd  # type: ignore
import pyogrio  # type: ignore
from loguru import logger  # type: ignore

from cmrv.aoi import SA_ALBERS, province_of
from cmrv.labels.classmap import warn_unmapped
from cmrv.labels.observations import PROCESSED_ROOT, make_run_id, write_partition

DATASET = "niaps_2023"
SOURCE = "niaps"
GPKG = Path("data/labels/raw/niaps_2023/2023 NIAPS GeoPackage.gpkg")

SOURCE_URL = "https://sites.google.com/site/wfwplanning/assessment"
PAPER_DOI = "10.1007/s10530-025-03558-9"
LICENSE = (
    "DFFE / Working for Water — written permission to use (email 2026-08-11, "
    "'No restrictions'); NO formal licence issued"
)

# The survey ran 2016-2023 but the delivered product is the 2023 layer and the
# extrapolation matched against recent Sentinel-2. Per-polygon survey dates are not
# in the GeoPackage, so every row takes the product year and `ingest-chips` fetches
# 2023 imagery. ponytail: refine if a dated plot table ever becomes available.
EVENT_DATE = "2023-01-01"

# Eroded >=20 m inside a boundary that is itself a model output. Passes the <=40 m
# chip gate; deliberately worse than MapWAPS (15 m) and SANLC (20 m).
COORD_UNCERTAINTY_M = 25.0

# Distilled, not observed — halve their say in the loss. See module docstring.
WEIGHT = 0.5

# GeoPackage layer -> (species_normalized, taxon_rank). Component species per taxon
# are Table 1 of Kotze et al. (2025); "Wattle_spp" is A. dealbata/decurrens/mearnsii
# and "Opuntia spp" is the paper's Cactaceae taxon (Cylindropuntia, Opuntia,
# Trichocereus). Species rank where the layer resolves to one binomial.
LAYER_TO_TAXON: dict[str, tuple[str, str]] = {
    "Wattle_spp": ("Acacia", "genus"),
    "Acacia cyclops": ("Acacia cyclops", "species"),
    "Acacia saligna": ("Acacia saligna", "species"),
    "Eucalyptus_spp": ("Eucalyptus", "genus"),
    "Pinus_spp": ("Pinus", "genus"),
    "Populus spp": ("Populus", "genus"),
    "Prosopis spp": ("Prosopis", "genus"),
    "Hakea spp": ("Hakea", "genus"),
    "Lantana camara": ("Lantana camara", "species"),
    "Melia azedarach": ("Melia azedarach", "species"),
    "Solanum mauritianum": ("Solanum mauritianum", "species"),
    "Opuntia spp": ("Opuntia", "genus"),
}

# Dropped on growth form (Table 1) — neither tree nor shrub, so their invaded form is
# not a woody canopy at 10 m. Both are also mostly linear riparian/disturbance strips
# that the negative buffer would erode away regardless.
DROPPED_GROWTH_FORM: dict[str, str] = {
    "Arundo donax": "tall grass",
    "Chromolaena odorata": "perennial herb",
}


def _distil_layer(
    layer: str, min_density: int, min_area_m2: float, erode_m: float
) -> gpd.GeoDataFrame:
    """One layer -> eroded pure-core polygons in SA Albers (may be empty)."""
    # Push the density filter into the read: it drops ~78% of rows before any
    # geometry is parsed, which is what makes a 1.27 M-polygon source tractable.
    gdf = pyogrio.read_dataframe(
        GPKG, layer=layer, columns=["gridcode"], where=f"gridcode >= {min_density}"
    )
    if gdf.empty:
        return gdf
    gdf = gdf.to_crs(SA_ALBERS)
    gdf["geometry"] = gdf.geometry.make_valid()
    gdf = gdf.explode(index_parts=False, ignore_index=True)
    gdf = gdf[gdf.geom_type.isin(("Polygon", "MultiPolygon"))]  # make_valid can emit lines
    gdf = gdf[gdf.geometry.area >= min_area_m2]
    if gdf.empty:
        return gdf
    gdf["geometry"] = gdf.geometry.buffer(-erode_m)
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.is_valid].reset_index(drop=True)
    gdf["layer"] = layer
    return gdf


def _drop_cross_taxon_overlaps(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop every polygon that touches a polygon of a *different* taxon.

    The 14 layers are independent, so one place can be both Wattle and Eucalyptus.
    A mixed centre has no single correct label, so both sides go — cheaper and more
    honest than trying to pick a winner. Runs after the size/density filters, on
    ~5% of the original polygons.
    """
    if gdf.empty:
        return gdf
    j = gpd.sjoin(
        gdf[["layer", "geometry"]], gdf[["layer", "geometry"]], how="inner", predicate="intersects"
    )
    conflicted = j.index[j["layer_left"].to_numpy() != j["layer_right"].to_numpy()].unique()
    if len(conflicted):
        logger.info("cross-taxon overlap: dropping {} polygons", len(conflicted))
    return gdf.drop(index=conflicted).reset_index(drop=True)


def ingest_niaps(
    min_density: int = 50,
    min_area_ha: float = 1.0,
    erode_m: float = 20.0,
    root: str = PROCESSED_ROOT,
    run_id: str | None = None,
) -> str:
    """Distil the NIAPS polygons to one point per pure core → ``source=niaps`` store."""
    run_id = run_id or make_run_id(SOURCE)
    ingested_at = dt.datetime.now(tz=dt.UTC)
    logger.info(
        "niaps: density>={} area>={}ha erode={}m — dropping {} on growth form",
        min_density,
        min_area_ha,
        erode_m,
        sorted(DROPPED_GROWTH_FORM),
    )

    parts = []
    for layer in LAYER_TO_TAXON:
        g = _distil_layer(layer, min_density, min_area_ha * 10_000, erode_m)
        logger.info("  {:22s} {:6d} cores", layer, len(g))
        if not g.empty:
            parts.append(g)
    if not parts:
        raise ValueError("no polygons survived the distillation filters")

    gdf = _drop_cross_taxon_overlaps(
        gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=SA_ALBERS)
    )

    # point_on_surface, never centroid — a centroid can land outside a concave polygon.
    pts = gdf.geometry.representative_point()
    gdf = gdf.set_geometry(pts).to_crs("EPSG:4326")
    gdf["_prov"] = province_of(gdf)
    outside = gdf["_prov"].isna()
    if outside.any():  # coastal/border slivers — no province means no month calendar
        logger.warning("dropping {} points outside every SA province", int(outside.sum()))
        gdf = gdf[~outside].reset_index(drop=True)

    rows = []
    for i, rec in enumerate(gdf.to_dict("records")):
        sp, rank = LAYER_TO_TAXON[rec["layer"]]
        geom = rec["geometry"]
        rows.append(
            {
                "obs_id": f"{SOURCE}:{geom.x:.6f}:{geom.y:.6f}",
                "source": SOURCE,
                "source_record_id": str(i),
                "source_url": SOURCE_URL,
                "source_doi": PAPER_DOI,
                "license": LICENSE,
                "species": rec["layer"],
                "species_normalized": sp,
                "taxon_rank": rank,
                "geom_type": "point",
                "coord_uncertainty_m": COORD_UNCERTAINTY_M,
                "event_date": EVENT_DATE,
                "basis_of_record": "NIAPS_S2_EXTRAPOLATED",
                "cover_pct": float(rec["gridcode"]),
                "weight": WEIGHT,
                "ingested_at": ingested_at,
                "ingest_run_id": run_id,
                "aoi_admin1": rec["_prov"],
            }
        )

    warn_unmapped({r["species_normalized"] for r in rows}, source=DATASET)
    out = gpd.GeoDataFrame(pd.DataFrame(rows), geometry=list(gdf.geometry), crs="EPSG:4326")
    logger.info("per-taxon:\n{}", out["species_normalized"].value_counts().to_string())
    path = write_partition(out, DATASET, root=root, run_id=run_id)
    logger.success("niaps: {} rows → {}", len(rows), path)
    return path
