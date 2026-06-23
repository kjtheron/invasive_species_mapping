"""Tests for cmrv.io shared helpers."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rio_cogeo.cogeo import cog_validate

from cmrv.io import list_parquet_files, load_config, write_cog

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "configs" / "labels_schema.yaml"


def test_load_config_reads_yaml() -> None:
    cfg = load_config(str(SCHEMA_PATH))
    assert isinstance(cfg, dict)
    assert "class_maps" in cfg


def test_list_parquet_files_finds_files(tmp_path: Path) -> None:
    (tmp_path / "a.parquet").write_bytes(b"x")
    (tmp_path / "b.parquet").write_bytes(b"x")
    (tmp_path / "c.txt").write_bytes(b"x")
    result = list_parquet_files(str(tmp_path))
    assert len(result) == 2
    assert all(r.endswith(".parquet") for r in result)


def test_list_parquet_files_excludes_tmp(tmp_path: Path) -> None:
    tmp_dir = tmp_path / "_tmp_run"
    tmp_dir.mkdir()
    (tmp_dir / "part.parquet").write_bytes(b"x")
    (tmp_path / "good.parquet").write_bytes(b"x")
    result = list_parquet_files(str(tmp_path), recursive=True)
    assert len(result) == 1
    assert result[0].endswith("good.parquet")


def test_list_parquet_files_recursive(tmp_path: Path) -> None:
    sub = tmp_path / "source=gbif"
    sub.mkdir()
    (sub / "part0.parquet").write_bytes(b"x")
    (tmp_path / "root.parquet").write_bytes(b"x")
    result = list_parquet_files(str(tmp_path), recursive=True)
    assert len(result) == 2


def test_list_parquet_files_nonexistent(tmp_path: Path) -> None:
    result = list_parquet_files(str(tmp_path / "nope"))
    assert result == []


def test_write_cog_roundtrip() -> None:
    arr = np.random.randint(0, 10, (1, 64, 64), dtype=np.uint8)
    transform = from_bounds(0, 0, 640, 640, 64, 64)
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "test.tif")
        result = write_cog(arr, transform, "EPSG:32734", out, dtype="uint8", nodata=255)
        assert result == out
        with rasterio.open(out) as src:
            assert src.width == 64
            assert src.height == 64
            assert src.dtypes[0] == "uint8"
            assert src.nodata == 255
            data = src.read(1)
            np.testing.assert_array_equal(data, arr[0])


def test_write_cog_validates() -> None:
    arr = np.zeros((32, 32), dtype=np.float32)
    transform = from_bounds(0, 0, 320, 320, 32, 32)
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "cog.tif")
        write_cog(arr, transform, "EPSG:4326", out)
        is_valid, _, _ = cog_validate(out)
        assert is_valid


def test_write_cog_2d_input() -> None:
    arr = np.ones((16, 16), dtype=np.uint8)
    transform = from_bounds(0, 0, 160, 160, 16, 16)
    with tempfile.TemporaryDirectory() as tmp:
        out = str(Path(tmp) / "2d.tif")
        write_cog(arr, transform, "EPSG:32734", out, dtype="uint8", nodata=0)
        with rasterio.open(out) as src:
            assert src.count == 1
            assert src.read(1).shape == (16, 16)
