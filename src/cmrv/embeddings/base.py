"""Encoder-agnostic embedding interface for the Clay-vs-UniverSat bakeoff.

An ``Embedder`` maps a stack of monthly S2 composites for one chip to a single
feature vector. The bakeoff swaps implementations; chips / manifest / make-split
/ probe never know which encoder ran. **All super-resolution lives inside
``ClaySREmbedder``** — adopting UniverSat = delete ``claysr.py`` + flip the
default, nothing else changes.

Chip-stack convention: float32 ``(N, T, C, H, W)`` — N chips, T months, C bands
(10), H×W pixels at native 10 m. ``dates`` is ``(N, T)`` day-of-year ints.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

# Mid-month day-of-year for the configured WC months (pipeline.yaml feb/may/sep).
MONTH_DOY: dict[str, int] = {"feb": 46, "may": 135, "sep": 258}


class Embedder(ABC):
    """One feature vector per chip. Implementations: universat, clay_sr, rawstats."""

    name: str

    @abstractmethod
    def embed(self, stacks: np.ndarray, dates: np.ndarray) -> np.ndarray:
        """``(N, T, C, H, W)`` float32 + ``(N, T)`` day-of-year → ``(N, D)`` float32."""


def mean_pool(tokens: np.ndarray) -> np.ndarray:
    """``(N, L, D)`` token grid → ``(N, D)`` mean over tokens."""
    return tokens.mean(axis=1)


class RawStatsEmbedder(Embedder):
    """Dependency-free baseline: per-(month, band) spatial mean + std.

    The bakeoff's sanity floor — a foundation model must beat plain spectral
    statistics to justify itself. Also the test fixture (no torch/weights).
    """

    name = "rawstats"

    def embed(self, stacks: np.ndarray, dates: np.ndarray) -> np.ndarray:
        mean = np.nanmean(stacks, axis=(3, 4))  # (N, T, C)
        std = np.nanstd(stacks, axis=(3, 4))  # (N, T, C)
        x = np.concatenate([mean, std], axis=2).reshape(len(stacks), -1)  # (N, 2*T*C)
        return np.nan_to_num(x).astype("float32")
