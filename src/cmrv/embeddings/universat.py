"""UniverSat embedding backend — native 10 m S2, no super-resolution.

Wraps ``g-astruc/UniverSat`` (torch.hub, MIT, ~201 M, 768-d). Validated on real
``(B, T=3, 10, 128, 128)`` chips: they feed in directly and return a dense token
grid, which we mean-pool to one vector per chip.

Requires the ``embed`` dependency group (``torch`` + torch.hub deps).
"""

from __future__ import annotations

import numpy as np
import torch  # type: ignore

# Pinned. The head is only valid on vectors from the exact model it was trained on,
# and torch.hub would otherwise fetch whatever ``main`` is on the day a new machine
# first loads it: embed on one VM, infer on another, and they silently differ.
# Code: github gastruc/UniverSat @ d4e8712 (2026-07-31, the throughput rewrite).
# Weights: HF g-astruc/UniverSat @ b6456ee (model.safetensors unchanged since 2026-06-19).
# Move both together, then delete the embedding parts and re-embed.
UNIVERSAT_REPO = "gastruc/UniverSat:d4e8712d29651c34c4fdb9f9d2c0b168dd5c678b"
UNIVERSAT_REVISION = "b6456ee87a162128ea5600cf45d2767a6e48f2f9"


class UniverSatEmbedder:
    def __init__(
        self,
        device: str = "cpu",
        patch_size: int = 40,
        output_grid: int = 9,
        pool: str = "mean",
        repo: str = UNIVERSAT_REPO,
        revision: str = UNIVERSAT_REVISION,
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
        self.repo, self.revision = repo, revision  # part of the embedding store's signature
        # Freezing folds the (never-trained) weights into UniverSat's compiled graphs as
        # constants: 1.60 -> 1.31 s/chip on the Azure CPU, measured. Inference only,
        # which is all this wrapper does. Must be set before the first forward compiles.
        import torch._inductor.config as inductor_config  # type: ignore

        inductor_config.freezing = True
        self.model = (
            torch.hub.load(repo, "from_pretrained", trust_repo=True, revision=revision)
            .eval()
            .to(device)
        )

    @torch.no_grad()
    def _grids(self, stacks: np.ndarray, dates: np.ndarray) -> np.ndarray:
        """Forward → dense token grid per sample: ``(N, G, G, D)``."""
        out = []
        dev_type = "cuda" if self.device.startswith("cuda") else "cpu"
        for i in range(0, len(stacks), self.batch):
            s = torch.as_tensor(stacks[i : i + self.batch], dtype=torch.float32, device=self.device)
            d = torch.as_tensor(dates[i : i + self.batch], dtype=torch.long, device=self.device)
            with torch.autocast(device_type=dev_type, enabled=self.amp):
                feats = self.model.encode(
                    {"s2": s, "s2_dates": d},
                    patch_size=self.patch_size,
                    output_grid=self.output_grid,
                )
            feats = feats[0] if isinstance(feats, (tuple, list)) else feats  # (b, L, D)
            g = int(feats.shape[1] ** 0.5)
            out.append(feats.reshape(feats.shape[0], g, g, feats.shape[2]).cpu().numpy())
        return np.concatenate(out, axis=0)

    def embed(self, stacks: np.ndarray, dates: np.ndarray) -> np.ndarray:
        """Pooled vector per sample ``(N, D)`` — center token (point labels) or mean."""
        grids = self._grids(stacks, dates)  # (N, G, G, D)
        g = grids.shape[1]
        vec = grids[:, g // 2, g // 2, :] if self.pool == "center" else grids.mean(axis=(1, 2))
        # fp16/NaN tripwire — NaN in → NaN out; catch unfilled cloud pixels or unscaled DN.
        if not np.isfinite(vec).all():
            raise ValueError("non-finite UniverSat embedding — check chip NaN fill + scale")
        return vec.astype("float32")

    def embed_dense(self, stacks: np.ndarray, dates: np.ndarray) -> np.ndarray:
        """Full token grid per sample ``(N, G, G, D)`` — the frozen head runs on every token."""
        return self._grids(stacks, dates).astype("float32")
