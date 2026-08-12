"""Embedding stage — UniverSat encoder → feature vector per chip.

Light exports only (no torch). The encoder pulls torch and is imported explicitly:

    from cmrv.embeddings.universat import UniverSatEmbedder   # needs the `embed` group

``embed_chips`` (embed.py) writes the cube; ``train_head`` (head.py) trains on it.
"""

from cmrv.embeddings.constants import MONTH_DOY

__all__ = ["MONTH_DOY"]
