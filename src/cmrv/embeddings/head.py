"""Train a light head on frozen embeddings — the per-location classifier.

Reads the embedding cube + the split artifact, trains a **linear** or 1-hidden-layer
**MLP** head with on-the-fly class-balanced cross-entropy (weights from the TRAIN
fold only — recomputed every run, so updating labels just re-derives them),
standardizes on train, early-stops on val macro-F1, and reports per-class
precision/recall/F1 on the held-out test fold. Everything is in memory.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


def _per_class_prf(y_true: np.ndarray, y_pred: np.ndarray, class_ids: np.ndarray):
    """Per-class precision/recall/f1 (index space) + macro-F1 over present classes."""
    rows = []
    for i, cid in enumerate(class_ids):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        rows.append(
            {
                "class_id": int(cid),
                "support": int((y_true == i).sum()),
                "precision": round(prec, 3),
                "recall": round(rec, 3),
                "f1": round(f1, 3),
            }
        )
    df = pd.DataFrame(rows)
    macro = float(df.loc[df["support"] > 0, "f1"].mean())
    return df, macro


def _class_weights(counts: np.ndarray, scheme: str) -> np.ndarray:
    """Loss weights from TRAIN-fold counts. balanced = N/(K·n_c); sqrt = gentler."""
    k = len(counts)
    safe = np.maximum(counts, 1.0).astype("float32")
    if scheme == "balanced":
        return counts.sum() / (k * safe)
    if scheme == "sqrt":
        w = 1.0 / np.sqrt(safe)
        return w * k / w.sum()
    return np.ones(k, dtype="float32")


def _maha(z: np.ndarray, means: np.ndarray, prec: np.ndarray) -> np.ndarray:
    """Min Mahalanobis² distance of each row of ``z`` (N,D) to the class means (K,D)."""
    zpz = (z @ prec * z).sum(1)  # (N,)
    pmu = means @ prec  # (K, D)
    cross = z @ pmu.T  # (N, K)
    const = (pmu * means).sum(1)  # (K,)
    return (zpz[:, None] - 2 * cross + const[None, :]).min(1)


def _ood_stats(
    xfit: np.ndarray, yfit: np.ndarray, xthr: np.ndarray, k: int, eps: float = 1e-3
) -> dict:
    """Per-class means + shared precision + distance threshold (Mahalanobis OOD).

    Lee et al. 2018: fit a Gaussian per class with a pooled within-class covariance;
    a pixel far from every class mean (distance > threshold) is novel / not-IAP. The
    threshold is the 97.5-pct of distances on the **held-out** ``xthr`` (out-of-sample),
    not the in-sample fit set — in-sample distances are far too tight and over-flag.
    """
    means = np.stack([xfit[yfit == c].mean(0) for c in range(k)]).astype("float32")
    cov = np.cov(xfit - means[yfit], rowvar=False) + eps * np.eye(xfit.shape[1])
    prec = np.linalg.inv(cov).astype("float32")
    thr = float(np.quantile(_maha(xthr.astype("float32"), means, prec), 0.975))
    return {"means": means, "prec": prec, "threshold": thr}


def _build_model(arch: str, in_dim: int, k: int, hidden: int):
    """Linear or 1-hidden-layer MLP — shared by training and the inference reload."""
    import torch

    if arch == "linear":
        return torch.nn.Linear(in_dim, k)
    if arch == "mlp":
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, k),
        )
    raise ValueError(f"arch must be 'linear' or 'mlp', got {arch!r}")


def train_head(
    emb_uri: str,
    split_uri: str,
    arch: str = "linear",
    *,
    weight: str = "balanced",
    hidden: int = 256,
    epochs: int = 500,
    lr: float = 0.05,
    patience: int = 60,
    seed: int = 42,
    save: str | None = None,
):
    """Train a frozen-embedding head → ``(per_class_df, test_macro_f1)``.

    ``save`` writes a checkpoint (weights + standardization mu/sd + class ids) for
    wall-to-wall inference — reload with ``load_head``.
    """
    import torch

    ds = xr.open_zarr(emb_uri)
    emb = ds["emb"].values.astype("float32")
    idx = pd.DataFrame({"obs_id": ds["obs_id"].values.astype(str), "row": range(emb.shape[0])})
    split = pd.read_parquet(split_uri)
    split["obs_id"] = split["obs_id"].astype(str)
    df = split.merge(idx, on="obs_id", how="inner").dropna(subset=["class_id"])
    df["class_id"] = df["class_id"].astype(int)

    classes = np.sort(df["class_id"].unique())
    to_idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)

    def fold(name: str):
        d = df[df["fold"] == name]
        return emb[d["row"].to_numpy()], d["class_id"].map(to_idx).to_numpy().astype(np.int64)

    xtr, ytr = fold("train")
    xva, yva = fold("val")
    xte, yte = fold("test")
    if min(len(xtr), len(xva), len(xte)) == 0:
        raise ValueError(
            f"empty fold (train={len(xtr)} val={len(xva)} test={len(xte)}) — "
            "obs_id mismatch between the embedding cube and split.parquet?"
        )

    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    torch.manual_seed(seed)
    t = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)  # noqa: E731 (standardize)
    xtr_t, xva_t, xte_t = t(xtr), t(xva), t(xte)
    ytr_t = torch.tensor(ytr, dtype=torch.long)

    model = _build_model(arch, int(emb.shape[1]), k, hidden)

    w = _class_weights(np.bincount(ytr, minlength=k).astype("float32"), weight)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_f1, best_state, bad = -1.0, None, 0
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss_fn(model(xtr_t), ytr_t).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            vf1 = _per_class_prf(yva, model(xva_t).argmax(1).numpy(), classes)[1]
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {kk: v.clone() for kk, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        per, macro = _per_class_prf(yte, model(xte_t).argmax(1).numpy(), classes)
    logger.success(
        "{} head ({} CE): val macro-F1 {:.3f} | test macro-F1 {:.3f}", arch, weight, best_f1, macro
    )
    if save:
        from cmrv.io import ensure_parent

        ood = _ood_stats(
            ((xtr - mu) / sd).astype("float32"), ytr, ((xva - mu) / sd).astype("float32"), k
        )
        ensure_parent(save)
        torch.save(
            {
                "state_dict": best_state,
                "mu": mu,
                "sd": sd,
                "classes": classes,
                "arch": arch,
                "in_dim": int(emb.shape[1]),
                "hidden": hidden,
                "ood": ood,
            },
            save,
        )
        logger.success("saved head (+ OOD stats) → {}", save)
    return per, macro


def load_head(ckpt_path: str):
    """Load a saved head → ``(model.eval(), mu, sd, class_ids, ood)`` for inference."""
    import torch

    ck = torch.load(ckpt_path, weights_only=False)
    model = _build_model(ck["arch"], ck["in_dim"], len(ck["classes"]), ck["hidden"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck["mu"], ck["sd"], np.asarray(ck["classes"]), ck["ood"]


def predict_probs(model, mu, sd, ood: dict, x: np.ndarray):
    """Features ``(N, D)`` → ``(softmax probs (N, K), ood_score (N,) 0-1)``.

    ood_score = Mahalanobis distance / threshold, scaled so 0.5 = the train 97.5-pct
    cutoff (``>0.5`` ⇒ novel / not-IAP). Returns full probs so overlapping inference
    windows can be blended before the argmax.
    """
    import torch

    xstd = ((x - mu) / sd).astype("float32")
    with torch.no_grad():
        probs = torch.softmax(model(torch.tensor(xstd)), dim=1).numpy()
    d = _maha(xstd, ood["means"], ood["prec"])
    return probs, np.clip(d / ood["threshold"], 0, 2) / 2


def predict_dense(model, mu, sd, classes: np.ndarray, ood: dict, x: np.ndarray):
    """Features ``(N, D)`` → ``(class_ids, confidence 0-1, ood_score 0-1)`` (argmax of probs)."""
    probs, ood_score = predict_probs(model, mu, sd, ood, x)
    return classes[probs.argmax(1)], probs.max(1), ood_score
