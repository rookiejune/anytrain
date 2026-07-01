"""LLM optimizer preset and config assembly helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum, auto

from .config import AdamWConfig, MuonAdamWConfig, MuonConfig
from .scheduler import SchedulerConfig, SchedulerPhaseLike
from .scheduler import make_scheduler_config as make_phase_scheduler_config

type OptimizerConfig = AdamWConfig | MuonAdamWConfig
type SchedulerInput = Sequence[SchedulerPhaseLike]


class OptimizationPreset(StrEnum):
    PRETRAIN = auto()
    CPT = auto()
    SFT = auto()

    @classmethod
    def parse(cls, preset: str) -> OptimizationPreset:
        if not isinstance(preset, str):
            raise TypeError("preset must be a string.")
        try:
            return cls(preset)
        except ValueError as error:
            raise ValueError("preset must be pretrain, cpt, or sft.") from error


class OptimizerOption(StrEnum):
    ADAMW = auto()
    MUON = auto()

    @classmethod
    def parse(cls, optimizer: str) -> OptimizerOption:
        if not isinstance(optimizer, str):
            raise TypeError("optimizer must be a string.")
        try:
            return cls(optimizer)
        except ValueError as error:
            raise ValueError("optimizer must be adamw or muon.") from error


@dataclass(frozen=True)
class PresetDefaults:
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float = 1e-8


def make_optimizer_config(
    preset: str,
    *,
    optimizer: str,
    lr: float | None,
    weight_decay: float | None,
    betas: tuple[float, float] | None,
    eps: float | None,
    fused: bool | None,
) -> OptimizerConfig:
    optimizer_option = OptimizerOption.parse(optimizer)
    defaults = preset_defaults(OptimizationPreset.parse(preset))
    adamw = AdamWConfig(
        lr=defaults.lr if lr is None else lr,
        weight_decay=defaults.weight_decay if weight_decay is None else weight_decay,
        betas=defaults.betas if betas is None else betas,
        eps=defaults.eps if eps is None else eps,
        fused=fused,
    )
    if optimizer_option is OptimizerOption.ADAMW:
        return adamw

    return MuonAdamWConfig(
        muon=MuonConfig(
            lr=adamw.lr,
            weight_decay=adamw.weight_decay,
        ),
        adamw=adamw,
    )


def make_scheduler_config_from_input(scheduler: SchedulerInput | None) -> SchedulerConfig:
    if scheduler is None:
        return SchedulerConfig()
    if isinstance(scheduler, (str, bytes)) or not isinstance(scheduler, Sequence):
        raise TypeError("scheduler must be a sequence of (shape, duration_steps) tuples.")
    return make_phase_scheduler_config(*scheduler)


def as_adamw_config(optimizer_config: OptimizerConfig) -> AdamWConfig:
    if isinstance(optimizer_config, AdamWConfig):
        return optimizer_config
    if isinstance(optimizer_config, MuonAdamWConfig):
        return optimizer_config.adamw
    raise TypeError("optimizer_config must be an AdamWConfig or MuonAdamWConfig.")


def as_muon_config(optimizer_config: OptimizerConfig) -> MuonConfig:
    if isinstance(optimizer_config, MuonAdamWConfig):
        return optimizer_config.muon
    raise TypeError("optimizer_config must be a MuonAdamWConfig.")


def preset_defaults(preset: OptimizationPreset) -> PresetDefaults:
    if preset is OptimizationPreset.PRETRAIN:
        return PresetDefaults(lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    if preset is OptimizationPreset.CPT:
        return PresetDefaults(lr=5e-5, weight_decay=0.1, betas=(0.9, 0.95))
    if preset is OptimizationPreset.SFT:
        return PresetDefaults(lr=2e-5, weight_decay=0.01, betas=(0.9, 0.999))
    raise ValueError("preset must be pretrain, cpt, or sft.")
