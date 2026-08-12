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


def test_province_of_slugs_match_admin1_zone_keys():
    """Every ADM1 slug must be a key in pipeline.yaml's admin1_zone — else chips raise."""
    import geopandas as gpd
    import yaml

    from cmrv.aoi import fetch_geoboundaries_adm1, province_of

    adm1 = fetch_geoboundaries_adm1()
    pts = gpd.GeoDataFrame(
        geometry=[g.representative_point() for g in adm1.geometry], crs="EPSG:4326"
    )
    slugs = set(province_of(pts).dropna())
    with open("configs/pipeline.yaml") as fh:
        zones = yaml.safe_load(fh)["admin1_zone"]
    assert len(slugs) == 9, slugs
    assert slugs <= set(zones), f"unmapped provinces: {sorted(slugs - set(zones))}"
