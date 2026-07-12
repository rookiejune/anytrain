"""Scheduler phase and schedule types."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import auto
from typing import Union

from anytrain._compat import StrEnum

from ._shape import CurveShape, curve_shape


class SchedulerOption(StrEnum):
    CONSTANT = auto()
    WARMUP_COSINE = auto()
    WSD = auto()


@dataclass(frozen=True)
class Phase:
    shape: CurveShape | str
    duration_steps: int = -1
    start_lr_ratio: float | None = None
    end_lr_ratio: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", curve_shape(self.shape))
        if not isinstance(self.duration_steps, int) or isinstance(self.duration_steps, bool):
            raise TypeError("duration_steps must be an integer.")
        if self.duration_steps == 0 or self.duration_steps < -1:
            raise ValueError("duration_steps must be positive, or -1 for an infinite phase.")
        if self.start_lr_ratio is not None:
            _validate_ratio(self.start_lr_ratio, name="start_lr_ratio")
        _validate_ratio(self.end_lr_ratio, name="end_lr_ratio")


PhaseLike = Union[Phase, tuple[Union[CurveShape, str], int]]


@dataclass(frozen=True)
class Schedule:
    phases: tuple[Phase, ...] = field(
        default_factory=lambda: (
            Phase(shape=CurveShape.CONSTANT, duration_steps=-1),
        )
    )

    def __post_init__(self) -> None:
        if not isinstance(self.phases, tuple):
            raise TypeError("phases must be a tuple of Phase.")
        if not self.phases:
            raise ValueError("phases must contain at least one scheduler phase.")
        for index, phase in enumerate(self.phases):
            if not isinstance(phase, Phase):
                raise TypeError(f"phases[{index}] must be a Phase.")
            if phase.duration_steps == -1 and index != len(self.phases) - 1:
                raise ValueError("duration_steps=-1 makes later phases unreachable.")
            if phase.duration_steps == -1 and phase.shape is not CurveShape.CONSTANT:
                raise ValueError("duration_steps=-1 is only supported for constant phases.")


def _validate_ratio(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float.")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


__all__ = [
    "CurveShape",
    "Phase",
    "PhaseLike",
    "Schedule",
]
