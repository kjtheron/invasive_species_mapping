"""Embedding stage — encoder-agnostic backend.

Light exports only (numpy). The UniverSat backend pulls torch and is imported
explicitly:

    from cmrv.embeddings.universat import UniverSatEmbedder   # needs the `embed` group

``embed_chips`` (embed.py) writes the cube; ``train_head`` (head.py) trains on it.
"""

from cmrv.embeddings.base import MONTH_DOY, Embedder, RawStatsEmbedder

__all__ = [
    "MONTH_DOY",
    "Embedder",
    "RawStatsEmbedder",
]
