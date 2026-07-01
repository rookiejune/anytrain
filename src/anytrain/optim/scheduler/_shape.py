"""Shared scheduler curve-shape helpers."""

from __future__ import annotations

from enum import StrEnum, auto


class CurveShape(StrEnum):
    CONSTANT = auto()
    LINEAR = auto()
    COSINE = auto()


def normalize_curve_shape(shape: CurveShape | str) -> CurveShape:
    if isinstance(shape, CurveShape):
        return shape
    if not isinstance(shape, str):
        raise TypeError("curve shape must be a string or CurveShape.")
    try:
        return CurveShape(shape)
    except ValueError as error:
        raise ValueError("curve shape must be constant, linear, or cosine.") from error


__all__ = ["CurveShape", "normalize_curve_shape"]
