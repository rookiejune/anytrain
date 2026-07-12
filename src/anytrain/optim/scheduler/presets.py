"""Named scheduler presets and phase-list construction helpers."""

from __future__ import annotations

import torch

from ._shape import curve_shape
from .compose import create_scheduler_from_config
from .types import (
    CurveShape,
    Phase,
    PhaseLike,
    Schedule,
    SchedulerOption,
)


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


def make_scheduler_config(*phases: PhaseLike) -> Schedule:
    return Schedule(phases=tuple(_coerce_phase(phase) for phase in phases))


def make_named_scheduler_config(
    *,
    schedule: str = "constant",
    warmup_steps: int = 0,
    total_steps: int | None = None,
    stable_steps: int | None = None,
    decay_steps: int | None = None,
    min_lr_ratio: float = 0.1,
) -> Schedule:
    _validate_non_negative_int(warmup_steps, name="warmup_steps")
    _validate_ratio(min_lr_ratio, name="min_lr_ratio")

    option = _scheduler_option(schedule)
    if option is SchedulerOption.CONSTANT:
        _reject_constant_scheduler_steps(
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            stable_steps=stable_steps,
            decay_steps=decay_steps,
        )
        return Schedule()

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


def _coerce_phase(phase: PhaseLike) -> Phase:
    if isinstance(phase, Phase):
        return phase
    if not isinstance(phase, tuple) or len(phase) != 2:
        raise TypeError(
            "scheduler phase must be a Phase or a (shape, duration_steps) tuple."
    )
    shape, duration_steps = phase
    shape = curve_shape(shape)
    if shape is CurveShape.LINEAR:
        return Phase(
            shape=shape,
            duration_steps=duration_steps,
            start_lr_ratio=0.0,
            end_lr_ratio=1.0,
        )
    if shape is CurveShape.COSINE:
        return Phase(
            shape=shape,
            duration_steps=duration_steps,
            end_lr_ratio=0.1,
        )
    return Phase(
        shape=shape,
        duration_steps=duration_steps,
        end_lr_ratio=1.0,
    )


def _make_warmup_cosine_config(
    *,
    warmup_steps: int,
    decay_steps: int,
    min_lr_ratio: float,
) -> Schedule:
    phases: list[Phase] = []
    if warmup_steps > 0:
        phases.append(
            Phase(
                shape=CurveShape.LINEAR,
                duration_steps=warmup_steps,
                start_lr_ratio=0.0,
                end_lr_ratio=1.0,
            )
        )
    phases.append(
        Phase(
            shape=CurveShape.COSINE,
            duration_steps=decay_steps,
            end_lr_ratio=min_lr_ratio,
        )
    )
    return Schedule(phases=tuple(phases))


def _make_wsd_config(
    *,
    warmup_steps: int,
    stable_steps: int,
    decay_steps: int,
    min_lr_ratio: float,
) -> Schedule:
    phases: list[Phase] = []
    if warmup_steps > 0:
        phases.append(
            Phase(
                shape=CurveShape.LINEAR,
                duration_steps=warmup_steps,
                start_lr_ratio=0.0,
                end_lr_ratio=1.0,
            )
        )
    if stable_steps > 0:
        phases.append(
            Phase(
                shape=CurveShape.CONSTANT,
                duration_steps=stable_steps,
                end_lr_ratio=1.0,
            )
        )
    phases.append(
        Phase(
            shape=CurveShape.COSINE,
            duration_steps=decay_steps,
            end_lr_ratio=min_lr_ratio,
        )
    )
    return Schedule(phases=tuple(phases))


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


def _scheduler_option(schedule: str) -> SchedulerOption:
    if not isinstance(schedule, str):
        raise TypeError("schedule must be a string.")
    try:
        return SchedulerOption(schedule)
    except ValueError as error:
        raise ValueError("schedule must be constant, warmup_cosine, or wsd.") from error


def _validate_ratio(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a float.")
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1.")


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
    "create_scheduler",
    "make_named_scheduler_config",
    "make_scheduler_config",
]
