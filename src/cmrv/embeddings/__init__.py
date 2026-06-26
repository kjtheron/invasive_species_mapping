"""Embedding stage — encoder-agnostic backends + the bakeoff probe.

Light exports only (numpy/pandas). The heavy backends pull their own deps and
are imported explicitly:

    from cmrv.embeddings.universat import UniverSatEmbedder   # needs torch
    from cmrv.embeddings.claysr import ClaySREmbedder         # needs sen2sr + clay
"""

from cmrv.embeddings.bakeoff import linear_probe_scores, run_bakeoff
from cmrv.embeddings.base import MONTH_DOY, Embedder, RawStatsEmbedder, mean_pool

__all__ = [
    "MONTH_DOY",
    "Embedder",
    "RawStatsEmbedder",
    "linear_probe_scores",
    "mean_pool",
    "run_bakeoff",
]
