"""LLM optimizer preset and options assembly helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum, auto
from typing import TypeGuard

from .options import (
    DEFAULT_MUON_ADJUST_LR_FN,
    AdamWOptions,
    MuonAdamWOptions,
    MuonOptions,
    OptimizerOptions,
)
from .scheduler import PhaseLike, Schedule
from .scheduler import make_scheduler_config as make_phase_scheduler_config

type SchedulerInput = Sequence[PhaseLike]
DEFAULT_ADAMW_WEIGHT_DECAY = 0.01
DEFAULT_MUON_WEIGHT_DECAY = 0.0


class OptimizationPreset(StrEnum):
    PRETRAIN = auto()
    CPT = auto()
    SFT = auto()


class OptimizerOption(StrEnum):
    ADAMW = auto()
    MUON = auto()


def make_optimizer_options(
    preset: str,
    *,
    optimizer: str,
    lr: float | None,
    weight_decay: float | None,
    betas: tuple[float, float] | None,
    eps: float | None,
    fused: bool | None,
) -> OptimizerOptions:
    if not isinstance(optimizer, str):
        raise TypeError("optimizer must be a string.")
    if not isinstance(preset, str):
        raise TypeError("preset must be a string.")
    try:
        optimizer_option = OptimizerOption(optimizer)
    except ValueError as error:
        raise ValueError("optimizer must be adamw or muon.") from error
    try:
        preset_option = OptimizationPreset(preset)
    except ValueError as error:
        raise ValueError("preset must be pretrain, cpt, or sft.") from error

    adamw = _preset_adamw_options(
        preset_option,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        fused=fused,
    )
    if optimizer_option is OptimizerOption.ADAMW:
        return adamw

    muon: MuonOptions = {
        "lr": adamw["lr"],
        "weight_decay": DEFAULT_MUON_WEIGHT_DECAY,
        "adjust_lr_fn": DEFAULT_MUON_ADJUST_LR_FN,
    }
    return {"muon": muon, "adamw": adamw}


def make_scheduler_config_from_input(scheduler: SchedulerInput | None) -> Schedule:
    if scheduler is None:
        return Schedule()
    if isinstance(scheduler, (str, bytes)) or not isinstance(scheduler, Sequence):
        raise TypeError("scheduler must be a sequence of (shape, duration_steps) tuples.")
    return make_phase_scheduler_config(*scheduler)


def is_muon_adamw_options(options: object) -> TypeGuard[MuonAdamWOptions]:
    return isinstance(options, Mapping) and "muon" in options and "adamw" in options


def _preset_adamw_options(
    preset: OptimizationPreset,
    *,
    lr: float | None,
    weight_decay: float | None,
    betas: tuple[float, float] | None,
    eps: float | None,
    fused: bool | None,
) -> AdamWOptions:
    if preset is OptimizationPreset.PRETRAIN:
        default_lr = 3e-4
        default_betas = (0.9, 0.95)
    elif preset is OptimizationPreset.CPT:
        default_lr = 5e-5
        default_betas = (0.9, 0.95)
    elif preset is OptimizationPreset.SFT:
        default_lr = 2e-5
        default_betas = (0.9, 0.999)
    else:
        raise ValueError("preset must be pretrain, cpt, or sft.")
    return {
        "lr": default_lr if lr is None else lr,
        "weight_decay": DEFAULT_ADAMW_WEIGHT_DECAY if weight_decay is None else weight_decay,
        "betas": default_betas if betas is None else betas,
        "eps": 1e-8 if eps is None else eps,
        "fused": fused,
    }
