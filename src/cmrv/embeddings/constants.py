"""Shared constants for the embedding stage.

Chip-stack convention: float32 ``(N, T, C, H, W)`` — N chips, T months, C bands
(10), HxW pixels at native 10 m. ``dates`` is ``(N, T)`` day-of-year ints.

Kept torch-free so ``infer`` and the config tests can import it without the
``embed`` dependency group.
"""

from __future__ import annotations

# Mid-month day-of-year for every configured month across zones (pipeline.yaml
# months_by_zone): winter-rainfall feb/may/sep, summer-rainfall jul/sep/dec.
MONTH_DOY: dict[str, int] = {"feb": 46, "may": 135, "jul": 196, "sep": 258, "dec": 349}
