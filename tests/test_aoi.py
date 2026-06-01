import geopandas as gpd
from shapely.geometry import Polygon

from cmrv.aoi import build_tile_grid


def _square_utm34s(minx: float, miny: float, size_m: float) -> gpd.GeoDataFrame:
    poly = Polygon(
        [
            (minx, miny),
            (minx + size_m, miny),
            (minx + size_m, miny + size_m),
            (minx, miny + size_m),
        ]
    )
    return gpd.GeoDataFrame(geometry=[poly], crs="EPSG:32734")


def test_build_tile_grid_counts_20km_square_with_10km_tiles():
    aoi = _square_utm34s(260_000, 6_240_000, 20_000)
    tiles = build_tile_grid(aoi, tile_km=10.0, crs="EPSG:32734")
    assert len(tiles) == 4
    assert tiles.crs is not None and tiles.crs.to_epsg() == 32734
    assert list(tiles.columns) == ["tile_id", "geometry"]
    assert tiles["tile_id"].tolist() == [0, 1, 2, 3]


def test_build_tile_grid_accepts_wgs84_aoi_and_reprojects():
    aoi_wgs = _square_utm34s(260_000, 6_240_000, 20_000).to_crs("EPSG:4326")
    tiles = build_tile_grid(aoi_wgs, tile_km=10.0, crs="EPSG:32734")
    assert len(tiles) == 4
    assert tiles.crs is not None and tiles.crs.to_epsg() == 32734
