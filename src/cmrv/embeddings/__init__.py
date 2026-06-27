"""Embedding stage — encoder-agnostic backend + the linear-probe evaluation.

Light exports only (numpy/pandas). The UniverSat backend pulls torch and is
imported explicitly:

    from cmrv.embeddings.universat import UniverSatEmbedder   # needs the `embed` group
"""

from cmrv.embeddings.base import MONTH_DOY, Embedder, RawStatsEmbedder
from cmrv.embeddings.probe import evaluate_embedders, linear_probe_scores, load_chip_arrays

__all__ = [
    "MONTH_DOY",
    "Embedder",
    "RawStatsEmbedder",
    "evaluate_embedders",
    "linear_probe_scores",
    "load_chip_arrays",
]
