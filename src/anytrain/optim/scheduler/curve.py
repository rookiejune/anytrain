"""Learning-rate curve definitions for scheduler phases."""

from __future__ import annotations

from math import cos, pi

from .types import CurveShape


def lr_ratio(
    shape: CurveShape,
    progress: float,
    *,
    start_lr_ratio: float,
    end_lr_ratio: float,
) -> float:
    clamped_progress = min(max(progress, 0.0), 1.0)
    if shape is CurveShape.CONSTANT:
        return end_lr_ratio
    if shape is CurveShape.LINEAR:
        return start_lr_ratio + (end_lr_ratio - start_lr_ratio) * clamped_progress
    if shape is CurveShape.COSINE:
        cosine_progress = 0.5 * (1.0 - cos(pi * clamped_progress))
        return start_lr_ratio + (end_lr_ratio - start_lr_ratio) * cosine_progress
    raise ValueError("curve shape must be constant, linear, or cosine.")


__all__ = ["lr_ratio"]
