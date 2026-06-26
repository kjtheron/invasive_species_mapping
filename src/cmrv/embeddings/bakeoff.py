"""Embedding bakeoff — compare embedders by a spatial-block-CV linear probe.

Each embedder turns the same labelled chips into features; a 1-layer torch linear
probe is scored with **group (spatial-block) cross-validation** (whole blocks go
to train or test, never split — the leakage guard), and macro-F1s are tabled.
Cheap heads on frozen embeddings — the standard way to rank frozen geospatial
encoders.

Torch-only (no sklearn): the probe, standardization, grouped folds and macro-F1
are hand-rolled here. Chip→array loading from the manifest is written at
bakeoff-time; this module is array-in so it's testable now and encoder-agnostic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from cmrv.embeddings.base import Embedder


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro-F1 over the classes present in ``y_true``."""
    f1s = []
    for c in np.unique(y_true):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def linear_probe_scores(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    seed: int = 42,
    n_splits: int = 4,
    epochs: int = 300,
    lr: float = 0.05,
) -> tuple[float, float]:
    """Spatial-block (group) CV macro-F1 of a 1-layer torch probe (class-balanced CE).

    Returns ``(mean_macro_f1, std)``. ``groups`` is the spatial-block id per chip —
    whole groups go to train or test so no block is split across the fold.
    """
    import torch

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)
    groups = np.asarray(groups)
    classes = np.unique(y)
    y_idx = np.searchsorted(classes, y)  # → 0..K-1
    n_classes = len(classes)

    uniq = np.random.default_rng(seed).permutation(np.unique(groups))
    n_splits = min(n_splits, len(uniq))
    torch.manual_seed(seed)

    f1s = []
    for fold in np.array_split(uniq, n_splits):
        te = np.isin(groups, fold)
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue

        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6  # standardize on train
        xtr = torch.tensor((X[tr] - mu) / sd)
        xte = torch.tensor((X[te] - mu) / sd)
        ytr = torch.tensor(y_idx[tr], dtype=torch.long)

        counts = np.bincount(y_idx[tr], minlength=n_classes).astype(np.float32)
        weight = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1.0)))  # balanced

        clf = torch.nn.Linear(X.shape[1], n_classes)
        opt = torch.optim.Adam(clf.parameters(), lr=lr, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        clf.train()
        for _ in range(epochs):
            opt.zero_grad()
            loss_fn(clf(xtr), ytr).backward()
            opt.step()

        clf.eval()
        with torch.no_grad():
            pred = clf(xte).argmax(1).numpy()
        f1s.append(_macro_f1(y_idx[te], pred))

    return float(np.mean(f1s)), float(np.std(f1s))


def run_bakeoff(
    stacks: np.ndarray,
    dates: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    embedders: list[Embedder],
    seed: int = 42,
) -> pd.DataFrame:
    """Embed with each embedder, probe, return a comparison table (best first)."""
    rows = []
    for emb in embedders:
        X = emb.embed(stacks, dates)
        mean_f1, std_f1 = linear_probe_scores(X, y, groups, seed=seed)
        rows.append(
            {
                "embedder": emb.name,
                "dim": X.shape[1],
                "macro_f1": round(mean_f1, 4),
                "f1_std": round(std_f1, 4),
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
