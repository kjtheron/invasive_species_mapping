"""Linear-probe evaluation for the embedding stage.

Turns labelled chips into features (via any :class:`Embedder`) and scores a
1-layer torch linear probe with **group (spatial-block) cross-validation** — whole
blocks go to train or test, never split, so the score can't leak across nearby
chips. This is how the frozen encoder + a light head are evaluated, and the same
probe is the basis for the production linear/MLP head.

``load_chip_arrays`` reads real chips from the chip manifest; everything else is
array-in, so it's testable without torch or model weights.
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


def load_chip_arrays(
    manifest_uri: str,
    class_map_name: str,
    schema_path: str = "configs/labels_schema.yaml",
    months: tuple[str, ...] = ("feb", "may", "sep"),
    min_class_count: int = 10,
    max_per_class: int | None = None,
    scale: float = 1.0 / 10000,
    min_valid_frac: float = 0.5,
    split: str | None = None,
    seed: int = 42,
):
    """Load chips from the manifest → ``(stacks, dates, y, groups, obs_ids)``.

    One ``(T, C, H, W)`` stack per obs_id with all ``months`` present; ``y`` is the
    class_id under ``class_map_name`` (unmapped/native dropped); ``groups`` is the
    spatial block. Rare classes (< ``min_class_count``) are dropped so grouped CV
    is well-posed; ``max_per_class`` optionally subsamples for speed.

    ``scale`` multiplies reflectance — **default 1/10000** for raw S2 DN → reflectance
    (UniverSat expects this range). Cloud-masked NaN pixels are filled with 0 after
    scaling so the encoder never sees NaN. ``min_valid_frac`` drops obs whose any
    month's chip is below that finite-pixel fraction. ``split`` (``"train"``/``"val"``
    /``"test"``) restricts to that fold's obs from ``<dir>/<split>.txt``.
    """
    from collections import Counter
    from pathlib import Path

    import rasterio

    from cmrv.embeddings.base import MONTH_DOY
    from cmrv.labels.classmap import build_lookup

    cm = build_lookup(schema_path, class_map_name)
    man = pd.read_parquet(manifest_uri)
    rng = np.random.default_rng(seed)

    if split:  # restrict to one fold from make-split's split files
        ids = set(Path(manifest_uri).parent.joinpath(f"{split}.txt").read_text().split())
        man = man[man["obs_id"].isin(ids)]
    if min_valid_frac > 0 and "valid_frac" in man.columns:
        keep = man.groupby("obs_id")["valid_frac"].transform("min") >= min_valid_frac
        man = man[keep]

    rows = []
    for obs_id, g in man.groupby("obs_id"):
        cls = cm.resolve(g["species"].iloc[0])
        by_month = dict(zip(g["month_label"], g["chip_uri"], strict=False))
        if cls is None or not all(mo in by_month for mo in months):
            continue
        rows.append((obs_id, int(cls), int(g["block_id"].iloc[0]), [by_month[mo] for mo in months]))

    counts = Counter(r[1] for r in rows)
    rows = [r for r in rows if counts[r[1]] >= min_class_count]
    if max_per_class:
        by_cls: dict[int, list] = {}
        for r in rows:
            by_cls.setdefault(r[1], []).append(r)
        rows = [
            r
            for cls_rows in by_cls.values()
            for r in (
                [cls_rows[i] for i in rng.permutation(len(cls_rows))[:max_per_class]]
                if len(cls_rows) > max_per_class
                else cls_rows
            )
        ]

    dvec = [MONTH_DOY[mo] for mo in months]
    stacks, dates, y, groups, obs_ids = [], [], [], [], []
    for obs_id, cls, blk, uris in rows:
        frames = []
        for uri in uris:
            with rasterio.open(uri) as src:
                # scale DN→reflectance, then fill cloud-NaN with 0 (UniverSat NaN-poisons)
                frames.append(np.nan_to_num(src.read().astype("float32") * scale, nan=0.0))
        stacks.append(np.stack(frames))
        dates.append(dvec)
        y.append(cls)
        groups.append(blk)
        obs_ids.append(obs_id)
    return np.stack(stacks), np.array(dates), np.array(y), np.array(groups), obs_ids


def evaluate_embedders(
    stacks: np.ndarray,
    dates: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    embedders: list[Embedder],
    seed: int = 42,
) -> pd.DataFrame:
    """Embed with each embedder, linear-probe, return a comparison table (best first).

    Handy sanity check (e.g. UniverSat vs the rawstats floor) on a fixed chip set.
    """
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
