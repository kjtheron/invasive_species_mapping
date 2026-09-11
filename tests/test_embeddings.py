"""Tests for the embedding stage: embed_chips, train_head."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # train_head is torch; needs the `embed` dependency group


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


def test_embed_chips_writes_keyed_zarr(tmp_path):
    """embed_chips streams the manifest → a Zarr keyed by obs_id + block_id."""
    import pandas as pd
    import rasterio
    import xarray as xr

    from cmrv.embeddings.embed import embed_chips

    # One uint16 file per obs, months as bands, month-major: feb_B02..feb_B12,
    # may_B02.., sep_B12. The loader reshapes (T*C, H, W) -> (T, C, H, W) using
    # the manifest's `months`, so band ORDER in the file is load-bearing.
    months = ("feb", "may", "sep")
    rows = []
    for oid in ("a", "b"):
        p = tmp_path / oid / "2023.tif"
        p.parent.mkdir(exist_ok=True)
        with rasterio.open(
            p, "w", driver="GTiff", height=4, width=4, count=30, dtype="uint16", nodata=0
        ) as d:
            d.write(np.full((30, 4, 4), 5000, dtype="uint16"))
        rows.append(
            {
                "obs_id": oid,
                "species": "pinus",
                "year": 2023,
                "chip_uri": str(p),
                "months": ",".join(months),
                "n_months": 3,
                "block_id": 7,
                "lon": 25.0,
                "lat": -30.0,
                "valid_frac": 1.0,
            }
        )
    pd.DataFrame(rows).to_parquet(tmp_path / "manifest.parquet")

    class _Stub:
        """Duck-typed stand-in for UniverSatEmbedder — embed_chips only calls .embed()."""

        def embed(self, stacks, dates):
            # The stack must arrive as (B, T, C, H, W) with T from `months`, not
            # as the raw (T*C, H, W) the file holds.
            assert stacks.shape[1:] == (3, 10, 4, 4), stacks.shape
            assert dates.shape[1] == 3, dates.shape
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
    for fwd, inv in _d4_ops(8):
        aug = fwd(a[None, None, None])[0, 0, 0]  # transform as input-spatial (last-2 axes)
        back = inv(aug[..., None])[..., 0]  # inverse as probs-spatial (first-2 axes)
        assert np.allclose(back, a)
    assert [len(_d4_ops(n)) for n in (1, 4, 8)] == [1, 4, 8]  # 1=identity, 4=rotations, 8=full D4


def _traceable_manifest(tmp_path, n: int) -> str:
    """n one-obs chips whose DN encodes the obs index, so a vector names its own chip."""
    import pandas as pd
    import rasterio

    rows = []
    for k in range(n):
        p = tmp_path / f"o{k}" / "2023.tif"
        p.parent.mkdir()
        with rasterio.open(
            p, "w", driver="GTiff", height=4, width=4, count=30, dtype="uint16", nodata=0
        ) as d:
            d.write(np.full((30, 4, 4), 1000 * (k + 1), dtype="uint16"))
        rows.append(
            {
                "obs_id": f"o{k}",
                "chip_uri": str(p),
                "months": "feb,may,sep",
                "block_id": k,
                "lon": 25.0,
                "lat": -30.0,
                "valid_frac": 1.0,
            }
        )
    pd.DataFrame(rows).to_parquet(tmp_path / "manifest.parquet")
    return str(tmp_path / "manifest.parquet")


class _Tracer:
    """Stub encoder: the vector is the chip's mean reflectance; can die on a chosen call."""

    output_grid = 128

    def __init__(self, die_on_call: int | None = None):
        self.calls, self.die_on_call = 0, die_on_call

    def embed(self, stacks, dates):
        if self.calls == self.die_on_call:
            raise KeyboardInterrupt  # what Ctrl+C, or pkill via _term_saves_work, raises
        self.calls += 1
        return np.repeat(stacks.mean(axis=(1, 2, 3, 4))[:, None], 768, axis=1).astype("float32")


def test_embed_chips_resumes_after_a_kill(tmp_path):
    """A killed run keeps every finished vector; the re-run embeds only the rest."""
    import xarray as xr

    from cmrv.embeddings.embed import embed_chips

    man, out = _traceable_manifest(tmp_path, 5), str(tmp_path / "emb.zarr")
    # parts_every=2 with a kill on the 4th call exercises both save paths: a full
    # part (o0, o1) and the finally-flush of a partial one (o2).
    with pytest.raises(KeyboardInterrupt):
        embed_chips(man, out, _Tracer(die_on_call=3), batch=1, num_workers=0, parts_every=2)

    rerun = _Tracer()
    embed_chips(man, out, rerun, batch=1, num_workers=0, parts_every=2)
    assert rerun.calls == 2  # only o3 and o4 — nothing finished before the kill is redone

    ds = xr.open_zarr(out)
    ids = [str(o) for o in ds["obs_id"].values]
    assert sorted(ids) == [f"o{k}" for k in range(5)]
    for oid, v in zip(
        ids, ds["emb"].values[:, 0], strict=True
    ):  # each vector still belongs to its chip
        assert np.isclose(v, 1000 * (int(oid[1:]) + 1) * 1e-4)


def test_embed_chips_refuses_parts_from_another_model(tmp_path):
    """Vectors from a different grid must never be merged into the same store."""
    from cmrv.embeddings.embed import embed_chips

    man, out = _traceable_manifest(tmp_path, 2), str(tmp_path / "emb.zarr")
    embed_chips(man, out, _Tracer(), batch=1, num_workers=0)
    other = _Tracer()
    other.output_grid = 64
    with pytest.raises(ValueError, match="Delete"):
        embed_chips(man, out, other, batch=1, num_workers=0)
