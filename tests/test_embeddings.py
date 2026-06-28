"""Tests for the embedding probe harness (encoder-agnostic, no torch needed)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # probe is torch; needs the `embed` dependency group

from cmrv.embeddings import RawStatsEmbedder, evaluate_embedders, linear_probe_scores


def _synthetic(n_per_class: int = 10, n_groups: int = 4):
    """Two classes separable by band mean, spread across spatial-block groups."""
    rng = np.random.default_rng(0)
    stacks, y, groups = [], [], []
    for g in range(n_groups):
        for cls in (0, 1):
            for _ in range(n_per_class // 2):
                base = 0.0 if cls == 0 else 5.0
                stacks.append(rng.normal(base, 0.5, (3, 10, 8, 8)).astype("float32"))
                y.append(cls)
                groups.append(g)
    dates = np.tile([46, 135, 258], (len(stacks), 1))
    return np.stack(stacks), dates, np.array(y), np.array(groups)


def test_rawstats_embed_shape() -> None:
    stacks, dates, _y, _g = _synthetic()
    X = RawStatsEmbedder().embed(stacks, dates)
    assert X.shape == (len(stacks), 2 * 3 * 10)  # mean+std over (T=3, C=10)
    assert np.isfinite(X).all()


def test_probe_separates_synthetic() -> None:
    stacks, dates, y, groups = _synthetic()
    X = RawStatsEmbedder().embed(stacks, dates)
    mean_f1, _ = linear_probe_scores(X, y, groups, n_splits=2)
    assert mean_f1 > 0.8  # band-mean signal is trivially separable


def test_evaluate_embedders_table() -> None:
    stacks, dates, y, groups = _synthetic()
    df = evaluate_embedders(stacks, dates, y, groups, [RawStatsEmbedder()])
    assert list(df.columns) == ["embedder", "dim", "macro_f1", "f1_std"]
    assert df.iloc[0]["embedder"] == "rawstats"


def test_embed_chips_writes_keyed_zarr(tmp_path):
    """embed_chips streams the manifest → a Zarr keyed by obs_id + block_id."""
    import pandas as pd
    import rasterio
    import xarray as xr

    from cmrv.embeddings.base import Embedder
    from cmrv.embeddings.embed import embed_chips

    months = ("feb", "may", "sep")
    rows = []
    for oid in ("a", "b"):
        for mo in months:
            p = tmp_path / oid / f"{mo}.tif"
            p.parent.mkdir(exist_ok=True)
            with rasterio.open(
                p, "w", driver="GTiff", height=4, width=4, count=10, dtype="float32"
            ) as d:
                d.write(np.full((10, 4, 4), 5000.0, dtype="float32"))
            rows.append(
                {
                    "obs_id": oid,
                    "month_label": mo,
                    "chip_uri": str(p),
                    "block_id": 7,
                    "x_utm": 300000.0,
                    "y_utm": 6200000.0,
                    "valid_frac": 1.0,
                }
            )
    pd.DataFrame(rows).to_parquet(tmp_path / "manifest.parquet")

    class _Stub(Embedder):
        name = "stub"

        def embed(self, stacks, dates):
            return np.zeros((len(stacks), 768), dtype="float32")

    out = embed_chips(
        str(tmp_path / "manifest.parquet"), str(tmp_path / "emb.zarr"), _Stub(), batch=1
    )
    ds = xr.open_zarr(out)
    assert ds["emb"].shape == (2, 768)
    assert set(ds["obs_id"].values) == {"a", "b"}
    assert set(ds["block_id"].values) == {7}
    assert ds.attrs["crs"] == "EPSG:4326"  # zone-agnostic point index
    assert {"lon", "lat"} <= set(ds.coords) and bool(np.isfinite(ds["lon"].values).all())
