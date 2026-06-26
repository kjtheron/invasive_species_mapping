"""Tests for the embedding bakeoff harness (encoder-agnostic, no torch needed)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")  # probe is torch; needs the `embed` dependency group

from cmrv.embeddings import RawStatsEmbedder, linear_probe_scores, run_bakeoff


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


def test_run_bakeoff_table() -> None:
    stacks, dates, y, groups = _synthetic()
    df = run_bakeoff(stacks, dates, y, groups, [RawStatsEmbedder()])
    assert list(df.columns) == ["embedder", "dim", "macro_f1", "f1_std"]
    assert df.iloc[0]["embedder"] == "rawstats"
