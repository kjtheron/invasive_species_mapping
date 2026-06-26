"""ClaySR embedding backend — SEN2SR (10 m→2.5 m) then frozen Clay v1.5.

**Throwaway baseline.** ALL super-resolution lives here; if UniverSat wins the
bakeoff, delete this file + drop the `sen2sr`/`clay` deps and nothing else
references them.

Wiring is deliberately deferred to **bakeoff-time** — the SEN2SR + Clay APIs are
far easier to get right with real chips on hand than blind. Install when ready:

    uv add --group embed sen2sr mlstac
    # + Clay v1.5 (clay-foundation/model checkpoint via Hugging Face)

Per chip-stack ``(N, T, C=10, 64, 64)``:
    1. SEN2SR each of the T frames: 10 m → 2.5 m  ⇒ ``(N, T, C, 256, 256)``
    2. frozen Clay v1.5 per frame (patch 8 → 32×32 tokens), patch-mean ⇒ ``(N, T, D)``
    3. mean over T ⇒ ``(N, D)``
"""

from __future__ import annotations

import numpy as np

from cmrv.embeddings.base import Embedder


class ClaySREmbedder(Embedder):
    name = "clay_sr"

    def __init__(self, device: str = "cpu", batch: int = 8) -> None:
        self.device = device
        self.batch = batch

    def embed(self, stacks: np.ndarray, dates: np.ndarray) -> np.ndarray:
        raise NotImplementedError(
            "ClaySR is wired at bakeoff-time (needs real chips to validate the "
            "SEN2SR + Clay APIs). See this module's docstring + tasks/todo.md."
        )
