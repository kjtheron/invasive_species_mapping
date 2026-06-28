"""UniverSat embedding backend — native 10 m S2, no super-resolution.

Wraps ``g-astruc/UniverSat`` (torch.hub, MIT, ~201 M, 768-d). Smoke-validated:
our ``(B, T=3, 10, 64, 64)`` chips feed in directly and return a dense token
grid, which we mean-pool to one vector per chip.

Requires the ``embed`` dependency group (``torch`` + torch.hub deps).
"""

from __future__ import annotations

import numpy as np
import torch

from cmrv.embeddings.base import Embedder


class UniverSatEmbedder(Embedder):
    name = "universat"

    def __init__(
        self,
        device: str = "cpu",
        patch_size: int = 40,
        output_grid: int = 9,
        pool: str = "mean",
        repo: str = "gastruc/UniverSat",
        batch: int = 8,
        amp: bool = False,
    ) -> None:
        self.device = device
        self.patch_size = patch_size
        self.output_grid = output_grid
        self.pool = pool  # "mean" over all tokens, or "center" token (point labels)
        self.name = f"universat_{pool}"
        self.batch = batch
        self.amp = amp  # autocast (fp16/bf16) — big GPU win, usually no-op/slower on CPU
        self.model = torch.hub.load(repo, "from_pretrained", trust_repo=True).eval().to(device)

    @torch.no_grad()
    def embed(self, stacks: np.ndarray, dates: np.ndarray) -> np.ndarray:
        out = []
        for i in range(0, len(stacks), self.batch):
            s = torch.as_tensor(stacks[i : i + self.batch], dtype=torch.float32, device=self.device)
            d = torch.as_tensor(dates[i : i + self.batch], dtype=torch.long, device=self.device)
            dev_type = "cuda" if self.device.startswith("cuda") else "cpu"
            with torch.autocast(device_type=dev_type, enabled=self.amp):
                feats = self.model.encode(
                    {"s2": s, "s2_dates": d},
                    patch_size=self.patch_size,
                    output_grid=self.output_grid,
                )
            feats = feats[0] if isinstance(feats, (tuple, list)) else feats  # (b, L, D)
            if self.pool == "center":
                g = int(feats.shape[1] ** 0.5)
                vec = feats.reshape(feats.shape[0], g, g, feats.shape[2])[:, g // 2, g // 2, :]
            else:
                vec = feats.mean(dim=1)
            # fp16/NaN tripwire — NaN in → NaN out; catch unfilled cloud pixels or
            # an unscaled DN range rather than silently training on NaN features.
            if not torch.isfinite(vec).all():
                raise ValueError("non-finite UniverSat embedding — check chip NaN fill + scale")
            out.append(vec.cpu().numpy())  # (b, D)
        return np.concatenate(out, axis=0).astype("float32")
