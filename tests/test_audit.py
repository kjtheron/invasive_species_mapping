"""Tests for cmrv.labels.audit — viz-side post-fuse raster sanity check."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml
from rasterio.transform import from_origin

from cmrv.labels.audit import audit_post_fuse


@pytest.fixture
def schema_file(tmp_path: Path) -> Path:
    body = {
        "class_maps": {
            "test12": {
                0: {"name": "mearnsii", "members": ["Acacia mearnsii"]},
                4: {
                    "name": "pinus_spp",
                    "members": ["Pinus pinaster", "Pinus radiata"],
                    "genus_fallback": True,
                    "genus": "pinus",
                },
                6: {"name": "hakea_sericea", "members": ["Hakea sericea"]},
            }
        }
    }
    p = tmp_path / "schema.yaml"
    p.write_text(yaml.safe_dump(body, sort_keys=False))
    return p


def _write_label_tif(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    transform = from_origin(0, arr.shape[0], 1, 1)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=1,
        dtype="uint8",
        nodata=255,
        transform=transform,
        crs="EPSG:32734",
    ) as dst:
        dst.write(arr, 1)


def test_post_fuse_flags_empty_and_unknown(tmp_path: Path, schema_file: Path):
    raster_root = tmp_path / "labels"
    nodata = 255

    arr1 = np.full((4, 4), nodata, dtype="uint8")
    arr1[0, 0] = 0
    arr1[0, 1] = 4  # known classes, hakea (6) absent
    _write_label_tif(raster_root / "tile_id=0" / "label.tif", arr1)

    arr2 = np.full((4, 4), nodata, dtype="uint8")
    arr2[0, 0] = 0
    arr2[0, 1] = 99  # unknown class_id — corruption
    _write_label_tif(raster_root / "tile_id=1" / "label.tif", arr2)

    out = audit_post_fuse(
        raster_prefix=str(raster_root),
        schema_path=schema_file,
        class_map_name="test12",
        out_dir=str(tmp_path / "audit"),
    )

    anomalies = out["anomalies"]
    unknown = anomalies[anomalies["kind"] == "unknown_class_id"]
    empty = anomalies[anomalies["kind"] == "empty_class"]

    assert len(unknown) == 1
    assert int(unknown.iloc[0]["class_id"]) == 99
    assert int(unknown.iloc[0]["tile_id"]) == 1

    empty_pairs = {(int(r["tile_id"]), int(r["class_id"])) for _, r in empty.iterrows()}
    assert (0, 6) in empty_pairs
    assert (1, 4) in empty_pairs
    assert (1, 6) in empty_pairs

    hist = out["hist"].set_index(["tile_id", "class_id"])
    assert hist.loc[(0, 0), "n_pixels"] == 1
    assert hist.loc[(0, 4), "n_pixels"] == 1
    assert hist.loc[(1, 99), "n_pixels"] == 1
