from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from math import cos, pi

import torch


class CurveShape(StrEnum):
    CONSTANT = auto()
    LINEAR = auto()
    COSINE = auto()


class SchedulerOption(StrEnum):
    CONSTANT = auto()
    WARMUP_COSINE = auto()
    WSD = auto()


def _validate_ratio(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a float.")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


def _normalize_curve_shape(shape: CurveShape | str) -> CurveShape:
    if isinstance(shape, CurveShape):
        return shape
    if not isinstance(shape, str):
        raise TypeError("curve shape must be a string or CurveShape.")
    try:
        return CurveShape(shape)
    except ValueError as error:
        raise ValueError("curve shape must be constant, linear, or cosine.") from error


@dataclass(frozen=True)
class SchedulerPhaseConfig:
    shape: CurveShape | str
    duration_steps: int = -1
    start_lr_ratio: float | None = None
    end_lr_ratio: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", _normalize_curve_shape(self.shape))
        if not isinstance(self.duration_steps, int) or isinstance(self.duration_steps, bool):
            raise TypeError("duration_steps must be an integer.")
        if self.duration_steps == 0 or self.duration_steps < -1:
            raise ValueError("duration_steps must be positive, or -1 for an infinite phase.")
        if self.start_lr_ratio is not None:
            _validate_ratio(self.start_lr_ratio, name="start_lr_ratio")
        _validate_ratio(self.end_lr_ratio, name="end_lr_ratio")


SchedulerPhaseLike = SchedulerPhaseConfig | tuple[CurveShape | str, int]


@dataclass(frozen=True)
class SchedulerConfig:
    phases: tuple[SchedulerPhaseConfig, ...] = field(
        default_factory=lambda: (
            SchedulerPhaseConfig(shape=CurveShape.CONSTANT, duration_steps=-1),
        )
    )

    def __post_init__(self) -> None:
        if not isinstance(self.phases, tuple):
            raise TypeError("phases must be a tuple of SchedulerPhaseConfig.")
        if not self.phases:
            raise ValueError("phases must contain at least one scheduler phase.")
        for index, phase in enumerate(self.phases):
            if not isinstance(phase, SchedulerPhaseConfig):
                raise TypeError(f"phases[{index}] must be a SchedulerPhaseConfig.")
            if phase.duration_steps == -1 and index != len(self.phases) - 1:
                raise ValueError("duration_steps=-1 makes later phases unreachable.")
            if phase.duration_steps == -1 and phase.shape is not CurveShape.CONSTANT:
                raise ValueError("duration_steps=-1 is only supported for constant phases.")


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    schedule: str = "constant",
    warmup_steps: int = 0,
    total_steps: int | None = None,
    stable_steps: int | None = None,
    decay_steps: int | None = None,
    min_lr_ratio: float = 0.1,
) -> torch.optim.lr_scheduler.LambdaLR:
    return create_scheduler_from_config(
        optimizer,
        make_named_scheduler_config(
            schedule=schedule,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
            min_lr_ratio=min_lr_ratio,
        ),
    )


def create_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    phases = _resolve_phases(config)

    def lr_lambda(step: int) -> float:
        clamped_step = max(step, 0)
        for phase in phases:
            if phase.end_step is None or clamped_step <= phase.end_step:
                return _lr_ratio_for_phase(phase, clamped_step)
        return phases[-1].end_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def make_scheduler_config(
    *phases: SchedulerPhaseLike,
) -> SchedulerConfig:
    return SchedulerConfig(phases=tuple(_coerce_scheduler_phase(phase) for phase in phases))


def make_named_scheduler_config(
    *,
    schedule: str = "constant",
    warmup_steps: int = 0,
    total_steps: int | None = None,
    stable_steps: int | None = None,
    decay_steps: int | None = None,
    min_lr_ratio: float = 0.1,
) -> SchedulerConfig:
    _validate_non_negative_int(warmup_steps, name="warmup_steps")
    _validate_ratio(min_lr_ratio, name="min_lr_ratio")

    option = _normalize_scheduler_option(schedule)
    if option is SchedulerOption.CONSTANT:
        _reject_constant_scheduler_steps(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
        )
        return SchedulerConfig()

    if option is SchedulerOption.WARMUP_COSINE:
        resolved_total_steps = _require_positive_int(total_steps, name="total_steps")
        if resolved_total_steps <= warmup_steps:
            raise ValueError("total_steps must be greater than warmup_steps.")
        return _make_warmup_cosine_config(
            warmup_steps=warmup_steps,
            decay_steps=resolved_total_steps - warmup_steps,
            min_lr_ratio=min_lr_ratio,
        )

    if option is SchedulerOption.WSD:
        resolved_stable_steps = _require_non_negative_int(stable_steps, name="stable_steps")
        resolved_decay_steps = _require_positive_int(decay_steps, name="decay_steps")
        if total_steps is not None:
            expected_total_steps = warmup_steps + resolved_stable_steps + resolved_decay_steps
            if total_steps != expected_total_steps:
                raise ValueError(
                    "total_steps must equal warmup_steps + stable_steps + decay_steps for wsd."
                )
        return _make_wsd_config(
            warmup_steps=warmup_steps,
            stable_steps=resolved_stable_steps,
            decay_steps=resolved_decay_steps,
            min_lr_ratio=min_lr_ratio,
        )

    raise ValueError("schedule must be constant, warmup_cosine, or wsd.")


@dataclass(frozen=True)
class _ResolvedSchedulerPhase:
    shape: CurveShape
    start_step: int
    end_step: int | None
    start_lr_ratio: float
    end_lr_ratio: float


def _resolve_phases(
    config: SchedulerConfig,
) -> tuple[_ResolvedSchedulerPhase, ...]:
    resolved_phases: list[_ResolvedSchedulerPhase] = []
    start_step = 0
    previous_lr_ratio = 1.0
    for phase in config.phases:
        end_step = None if phase.duration_steps == -1 else start_step + phase.duration_steps
        start_lr_ratio = previous_lr_ratio if phase.start_lr_ratio is None else phase.start_lr_ratio
        resolved_phases.append(
            _ResolvedSchedulerPhase(
                shape=_normalize_curve_shape(phase.shape),
                start_step=start_step,
                end_step=end_step,
                start_lr_ratio=start_lr_ratio,
                end_lr_ratio=phase.end_lr_ratio,
            )
        )
        if end_step is None:
            break
        start_step = end_step
        previous_lr_ratio = phase.end_lr_ratio

    return tuple(resolved_phases)


def _lr_ratio_for_phase(phase: _ResolvedSchedulerPhase, step: int) -> float:
    if phase.shape is CurveShape.CONSTANT:
        return phase.end_lr_ratio

    if phase.end_step is None:
        raise ValueError("duration_steps=-1 is only supported for constant phases.")
    duration_steps = max(phase.end_step - phase.start_step, 1)
    progress = min(max((step - phase.start_step) / duration_steps, 0.0), 1.0)
    if phase.shape is CurveShape.LINEAR:
        return phase.start_lr_ratio + (phase.end_lr_ratio - phase.start_lr_ratio) * progress
    if phase.shape is CurveShape.COSINE:
        cosine_progress = 0.5 * (1.0 - cos(pi * progress))
        return phase.start_lr_ratio + (phase.end_lr_ratio - phase.start_lr_ratio) * cosine_progress
    raise ValueError("curve shape must be constant, linear, or cosine.")


def _coerce_scheduler_phase(phase: SchedulerPhaseLike) -> SchedulerPhaseConfig:
    if isinstance(phase, SchedulerPhaseConfig):
        return phase
    if not isinstance(phase, tuple) or len(phase) != 2:
        raise TypeError(
            "scheduler phase must be a SchedulerPhaseConfig or a (shape, duration_steps) tuple."
        )
    shape, duration_steps = phase
    curve_shape = _normalize_curve_shape(shape)
    if curve_shape is CurveShape.LINEAR:
        return SchedulerPhaseConfig(
            shape=curve_shape,
            duration_steps=duration_steps,
            start_lr_ratio=0.0,
            end_lr_ratio=1.0,
        )
    if curve_shape is CurveShape.COSINE:
        return SchedulerPhaseConfig(
            shape=curve_shape,
            duration_steps=duration_steps,
            end_lr_ratio=0.1,
        )
    return SchedulerPhaseConfig(
        shape=curve_shape,
        duration_steps=duration_steps,
        end_lr_ratio=1.0,
    )


def _make_warmup_cosine_config(
    *,
    warmup_steps: int,
    decay_steps: int,
    min_lr_ratio: float,
) -> SchedulerConfig:
    phases: list[SchedulerPhaseConfig] = []
    if warmup_steps > 0:
        phases.append(
            SchedulerPhaseConfig(
                shape=CurveShape.LINEAR,
                duration_steps=warmup_steps,
                start_lr_ratio=0.0,
                end_lr_ratio=1.0,
            )
        )
    phases.append(
        SchedulerPhaseConfig(
            shape=CurveShape.COSINE,
            duration_steps=decay_steps,
            end_lr_ratio=min_lr_ratio,
        )
    )
    return SchedulerConfig(phases=tuple(phases))


def _make_wsd_config(
    *,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float,
) -> SchedulerConfig:
    phases: list[SchedulerPhaseConfig] = []
    if warmup_steps > 0:
        phases.append(
            SchedulerPhaseConfig(
                shape=CurveShape.LINEAR,
                duration_steps=warmup_steps,
                start_lr_ratio=0.0,
                end_lr_ratio=1.0,
            )
        )
    if stable_steps > 0:
        phases.append(
            SchedulerPhaseConfig(
                shape=CurveShape.CONSTANT,
                duration_steps=stable_steps,
                end_lr_ratio=1.0,
            )
        )
    phases.append(
        SchedulerPhaseConfig(
            shape=CurveShape.COSINE,
            duration_steps=decay_steps,
            end_lr_ratio=min_lr_ratio,
        )
    )
    return SchedulerConfig(phases=tuple(phases))


def _reject_constant_scheduler_steps(
    *,
    warmup_steps: int,
    total_steps: int | None,
    stable_steps: int | None,
    decay_steps: int | None,
) -> None:
    if warmup_steps != 0:
        raise ValueError("warmup_steps is only valid for warmup_cosine or wsd schedules.")
    if total_steps is not None:
        raise ValueError("total_steps is only valid for warmup_cosine or wsd schedules.")
    if stable_steps is not None:
        raise ValueError("stable_steps is only valid for wsd schedules.")
    if decay_steps is not None:
        raise ValueError("decay_steps is only valid for wsd schedules.")


def _normalize_scheduler_option(schedule: str) -> SchedulerOption:
    if not isinstance(schedule, str):
        raise TypeError("schedule must be a string.")
    try:
        return SchedulerOption(schedule)
    except ValueError as error:
        raise ValueError("schedule must be constant, warmup_cosine, or wsd.") from error


def _validate_non_negative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _require_non_negative_int(value: int | None, *, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required.")
    _validate_non_negative_int(value, name=name)
    return value


def _require_positive_int(value: int | None, *, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required.")
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value


__all__ = [
    "CurveShape",
    "SchedulerOption",
    "SchedulerConfig",
    "SchedulerPhaseLike",
    "SchedulerPhaseConfig",
    "create_scheduler",
    "create_scheduler_from_config",
    "make_named_scheduler_config",
    "make_scheduler_config",
]
