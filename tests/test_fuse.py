"""Tests for label fusion + rasterization (fuse.py)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from shapely.geometry import Point, box

from cmrv.labels.classmap import build_lookup
from cmrv.labels.fuse import (
    NODATA,
    RASTER_CRS,
    RESOLUTION_M,
    assign_class_id,
    label_class_counts,
    rasterize_tile,
    write_label_cog,
)

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "labels_schema.yaml"

# A 500 m × 500 m UTM 34S tile centred in the Berg catchment area
_TILE_UTM = box(230_000.0, 6_240_000.0, 230_500.0, 6_240_500.0)


# ---------------------------------------------------------------------------
# Crosswalk tests
# ---------------------------------------------------------------------------


class TestSpeciestoClass:
    def test_loads_from_schema(self) -> None:
        cm = build_lookup(SCHEMA_PATH, "upper_berg_12")
        assert len(cm.binomial_to_class) > 0

    def test_known_species_resolved(self) -> None:
        cm = build_lookup(SCHEMA_PATH, "upper_berg_12")
        assert cm.resolve("Acacia mearnsii")[0] == 0
        assert cm.resolve("Acacia saligna")[0] == 1
        assert cm.resolve("Pinus pinaster")[0] == 4
        assert cm.resolve("Eucalyptus globulus")[0] == 5
        assert cm.resolve("Hakea sericea")[0] == 6
        assert cm.resolve("Sesbania punicea")[0] == 7

    def test_synonym_resolved(self) -> None:
        cm = build_lookup(SCHEMA_PATH, "upper_berg_12")
        assert cm.resolve("Acacia cyclops")[0] == 1  # merged into saligna class
        # 'blackwood' / 'rooikrans' were vernacular keys in the legacy
        # species_map; they live in vernacular_map (pre-GBIF resolution)
        # rather than in class_maps.members[] now.


class TestAssignClassId:
    def _make_gdf(self, rows: list[dict]) -> gpd.GeoDataFrame:
        import pandas as pd

        pdf = pd.DataFrame(rows)
        geoms = [Point(18.5, -33.9)] * len(pdf)
        return gpd.GeoDataFrame(pdf, geometry=geoms, crs="EPSG:4326")

    def test_species_normalized_lookup(self) -> None:
        cm = build_lookup(SCHEMA_PATH, "upper_berg_12")
        gdf = self._make_gdf(
            [
                {"species_normalized": "Acacia mearnsii", "weight": 0.5},
                {"species_normalized": "Pinus radiata", "weight": 0.5},
                {"species_normalized": "totally unknown species", "weight": 0.5},
            ]
        )
        out = assign_class_id(gdf, cm)
        assert int(out.loc[0, "class_id"]) == 0
        assert int(out.loc[1, "class_id"]) == 4
        assert np.isnan(out.loc[2, "class_id"])

    def test_genus_fallback(self) -> None:
        """Pinus halepensis falls back to pinus_spp via genus_fallback flag."""
        cm = build_lookup(SCHEMA_PATH, "upper_berg_12")
        gdf = self._make_gdf([{"species_normalized": "Pinus halepensis", "weight": 0.5}])
        out = assign_class_id(gdf, cm)
        assert int(out.loc[0, "class_id"]) == 4

    def test_unknown_species_is_nan(self) -> None:
        cm = build_lookup(SCHEMA_PATH, "upper_berg_12")
        gdf = self._make_gdf([{"species_normalized": "Quercus robur", "weight": 0.5}])
        out = assign_class_id(gdf, cm)
        assert np.isnan(out.loc[0, "class_id"])


# ---------------------------------------------------------------------------
# Rasterization tests
# ---------------------------------------------------------------------------


def _obs_gdf(class_id: int, weight: float, geom: object) -> gpd.GeoDataFrame:
    import pandas as pd

    return gpd.GeoDataFrame(
        pd.DataFrame(
            [
                {
                    "class_id": class_id,
                    "weight": weight,
                    "species_normalized": "test",
                    "geom_type": "point",
                }
            ]
        ),
        geometry=[geom],
        crs="EPSG:4326",
    )


class TestRasterizeTile:
    def test_empty_gdf_returns_all_nodata(self) -> None:
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        arr, transform, crs = rasterize_tile(gdf, _TILE_UTM)
        assert arr.dtype == np.uint8
        assert (arr == NODATA).all()
        assert crs == RASTER_CRS

    def test_point_burns_class_in_tile(self) -> None:
        # Place a point at the centre of the tile in WGS84
        tile_utm = _TILE_UTM
        cx = (tile_utm.bounds[0] + tile_utm.bounds[2]) / 2
        cy = (tile_utm.bounds[1] + tile_utm.bounds[3]) / 2
        from pyproj import Transformer

        t = Transformer.from_crs(RASTER_CRS, "EPSG:4326", always_xy=True)
        lon, lat = t.transform(cx, cy)
        gdf = _obs_gdf(class_id=0, weight=0.5, geom=Point(lon, lat))
        arr, _, _ = rasterize_tile(gdf, tile_utm)
        assert (arr != NODATA).any(), "at least one pixel should be labeled"
        labeled = arr[arr != NODATA]
        assert set(labeled.tolist()) == {0}

    def test_higher_weight_wins(self) -> None:
        """Two observations at the same location — higher weight class wins."""
        tile_utm = _TILE_UTM
        cx = (tile_utm.bounds[0] + tile_utm.bounds[2]) / 2
        cy = (tile_utm.bounds[1] + tile_utm.bounds[3]) / 2
        from pyproj import Transformer

        t = Transformer.from_crs(RASTER_CRS, "EPSG:4326", always_xy=True)
        lon, lat = t.transform(cx, cy)

        import pandas as pd

        gdf = gpd.GeoDataFrame(
            pd.DataFrame(
                [
                    {"class_id": 0, "weight": 0.3, "species_normalized": "a"},
                    {"class_id": 5, "weight": 0.9, "species_normalized": "b"},
                ]
            ),
            geometry=[Point(lon, lat), Point(lon, lat)],
            crs="EPSG:4326",
        )
        arr, _, _ = rasterize_tile(gdf, tile_utm)
        labeled = arr[arr != NODATA]
        # The weight-0.9 class (5) should dominate
        assert set(labeled.tolist()) == {5}

    def test_array_shape_matches_tile_size(self) -> None:
        gdf = gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        arr, _, _ = rasterize_tile(gdf, _TILE_UTM, resolution_m=RESOLUTION_M)
        expected_px = int(np.ceil(500.0 / RESOLUTION_M))
        assert arr.shape == (expected_px, expected_px)

    def test_polygon_burns_correctly(self) -> None:
        """A polygon covering half the tile should label ~half the pixels."""
        tile_utm = _TILE_UTM
        half_poly_utm = box(
            tile_utm.bounds[0],
            tile_utm.bounds[1],
            (tile_utm.bounds[0] + tile_utm.bounds[2]) / 2,
            tile_utm.bounds[3],
        )
        half_poly_4326 = gpd.GeoSeries([half_poly_utm], crs=RASTER_CRS).to_crs("EPSG:4326").iloc[0]

        import pandas as pd

        gdf = gpd.GeoDataFrame(
            pd.DataFrame([{"class_id": 8, "weight": 0.8, "species_normalized": "fynbos"}]),
            geometry=[half_poly_4326],
            crs="EPSG:4326",
        )
        arr, _, _ = rasterize_tile(gdf, tile_utm)
        labeled_frac = (arr == 8).sum() / arr.size
        assert 0.35 < labeled_frac < 0.65, f"half-tile polygon labeled {labeled_frac:.2%}"


# ---------------------------------------------------------------------------
# COG write + validation
# ---------------------------------------------------------------------------


class TestWriteLabelCog:
    def test_writes_valid_cog(self) -> None:
        """write_label_cog produces a valid GeoTIFF readable by rasterio."""
        arr = np.full((200, 200), NODATA, dtype=np.uint8)
        arr[50:100, 50:100] = 4  # burn class 4 in a 50×50 patch

        from rasterio.transform import from_bounds

        transform = from_bounds(230_000.0, 6_240_000.0, 230_500.0, 6_240_500.0, 200, 200)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "label.tif")
            write_label_cog(arr, transform, RASTER_CRS, out_path)

            assert Path(out_path).exists()
            with rasterio.open(out_path) as src:
                assert src.count == 1
                assert src.dtypes[0] == "uint8"
                assert src.nodata == NODATA
                data = src.read(1)
                assert data.shape == (200, 200)
                assert set(data[data != NODATA].tolist()) == {4}

    def test_cog_passes_rio_cogeo_validate(self) -> None:
        """Written COG passes rio-cogeo structural validation."""
        from rio_cogeo.cogeo import cog_validate

        arr = np.full((256, 256), NODATA, dtype=np.uint8)
        arr[64:192, 64:192] = 1

        from rasterio.transform import from_bounds

        transform = from_bounds(230_000.0, 6_240_000.0, 230_640.0, 6_240_640.0, 256, 256)

        with tempfile.TemporaryDirectory() as tmp:
            out_path = str(Path(tmp) / "label.tif")
            write_label_cog(arr, transform, RASTER_CRS, out_path)
            is_valid, _, _ = cog_validate(out_path)
            assert is_valid, "COG failed rio-cogeo validation"


# ---------------------------------------------------------------------------
# label_class_counts
# ---------------------------------------------------------------------------


def test_label_class_counts_correct() -> None:
    arr = np.full((100, 100), NODATA, dtype=np.uint8)
    arr[0:50, :] = 0  # 5000 px class 0
    arr[50:75, :] = 4  # 2500 px class 4

    from rasterio.transform import from_bounds

    transform = from_bounds(230_000.0, 6_240_000.0, 230_250.0, 6_240_250.0, 100, 100)

    with tempfile.TemporaryDirectory() as tmp:
        out_path = str(Path(tmp) / "label.tif")
        write_label_cog(arr, transform, RASTER_CRS, out_path)
        counts = label_class_counts(out_path)

    assert counts[0] == 5000
    assert counts[4] == 2500
    assert NODATA not in counts
