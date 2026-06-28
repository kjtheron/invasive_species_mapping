"""Tests for the embedding stage: RawStats baseline, embed_chips, train_head."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # train_head is torch; needs the `embed` dependency group

from cmrv.embeddings import RawStatsEmbedder


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
        str(tmp_path / "manifest.parquet"),
        str(tmp_path / "emb.zarr"),
        _Stub(),
        batch=1,
        num_workers=0,
    )
    ds = xr.open_zarr(out)
    assert ds["emb"].shape == (2, 768)
    assert set(ds["obs_id"].values) == {"a", "b"}
    assert set(ds["block_id"].values) == {7}
    assert ds.attrs["crs"] == "EPSG:4326"  # zone-agnostic point index
    assert {"lon", "lat"} <= set(ds.coords) and bool(np.isfinite(ds["lon"].values).all())


def test_train_head_separates_synthetic(tmp_path):
    """A linear head must separate two linearly-separable classes across folds."""
    import pandas as pd
    import xarray as xr

    from cmrv.embeddings.head import train_head

    rng = np.random.default_rng(0)
    n = 80
    y = np.tile([0, 1], n // 2)
    X = rng.normal(0, 0.1, (n, 8)).astype("float32")
    X[y == 0, 0] += 5.0
    X[y == 1, 1] += 5.0
    obs = [f"o{i}" for i in range(n)]
    xr.Dataset({"emb": (("obs", "feat"), X)}, coords={"obs_id": ("obs", np.array(obs))}).to_zarr(
        tmp_path / "e.zarr"
    )
    folds = np.array((["train", "train", "val", "test"] * n)[:n])
    pd.DataFrame({"obs_id": obs, "fold": folds, "class_id": y}).to_parquet(tmp_path / "s.parquet")

    ckpt = str(tmp_path / "head.pt")
    per, macro = train_head(
        str(tmp_path / "e.zarr"), str(tmp_path / "s.parquet"), arch="linear", epochs=200, save=ckpt
    )
    assert macro > 0.8
    assert set(per["class_id"]) == {0, 1}

    # save → load → predict round-trip (the inference path) recovers the labels
    from cmrv.embeddings.head import load_head, predict_dense

    model, mu, sd, classes, ood = load_head(ckpt)
    cls, conf, ood_score = predict_dense(model, mu, sd, classes, ood, X)
    assert (cls == y).mean() > 0.8
    assert conf.shape == y.shape and ood_score.shape == y.shape
    assert ood_score.min() >= 0 and ood_score.max() <= 1  # normalized OOD score
    assert (ood_score < 0.5).mean() > 0.8  # in-distribution points mostly not flagged


def test_d4_ops_roundtrip():
    """Each dihedral (fwd, inv) pair must invert: inv∘fwd = identity on the spatial grid."""
    from cmrv.infer import _d4_ops

    a = np.random.default_rng(0).random((6, 6)).astype("float32")
    for fwd, inv in _d4_ops(tta=True):
        aug = fwd(a[None, None, None])[0, 0, 0]  # transform as input-spatial (last-2 axes)
        back = inv(aug[..., None])[..., 0]  # inverse as probs-spatial (first-2 axes)
        assert np.allclose(back, a)
    assert len(_d4_ops(tta=True)) == 8 and len(_d4_ops(tta=False)) == 1
