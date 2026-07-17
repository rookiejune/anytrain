"""Compose scheduler phases into PyTorch LambdaLR schedulers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .curve import lr_ratio
from .types import CurveShape, Schedule


@dataclass(frozen=True)
class _ResolvedPhase:
    shape: CurveShape
    start_step: int
    end_step: int | None
    start_lr_ratio: float
    end_lr_ratio: float


def create_scheduler_from_config(
    optimizer: torch.optim.Optimizer,
    config: Schedule,
) -> torch.optim.lr_scheduler.LambdaLR:
    phases = _phases(config)

    def lr_lambda(step: int) -> float:
        clamped_step = max(step, 0)
        for phase in phases:
            if phase.end_step is None or clamped_step <= phase.end_step:
                return _lr_ratio_for_phase(phase, clamped_step)
        return phases[-1].end_lr_ratio

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _phases(config: Schedule) -> tuple[_ResolvedPhase, ...]:
    resolved_phases: list[_ResolvedPhase] = []
    start_step = 0
    previous_lr_ratio = 1.0
    for phase in config.phases:
        end_step = None if phase.duration_steps == -1 else start_step + phase.duration_steps
        start_lr_ratio = previous_lr_ratio if phase.start_lr_ratio is None else phase.start_lr_ratio
        resolved_phases.append(
            _ResolvedPhase(
                shape=phase.shape,
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


def _lr_ratio_for_phase(phase: _ResolvedPhase, step: int) -> float:
    if phase.shape is CurveShape.CONSTANT:
        return phase.end_lr_ratio

    if phase.end_step is None:
        raise ValueError("duration_steps=-1 is only supported for constant phases.")
    duration_steps = max(phase.end_step - phase.start_step, 1)
    progress = (step - phase.start_step) / duration_steps
    return lr_ratio(
        phase.shape,
        progress,
        start_lr_ratio=phase.start_lr_ratio,
        end_lr_ratio=phase.end_lr_ratio,
    )


__all__ = ["create_scheduler_from_config"]
