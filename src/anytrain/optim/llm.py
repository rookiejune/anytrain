from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import TypedDict

import torch
from torch import nn

from .adamw import AdamWDecayPolicy, create_adamw_optimizer
from .config import AdamWConfig, MuonAdamWConfig, MuonConfig
from .muon import (
    ExcludedModules,
    ExcludedModuleTypes,
    create_muon_adamw_optimizer,
)
from .scheduler import SchedulerConfig, create_scheduler, make_scheduler_config

type _OptimizerConfig = AdamWConfig | MuonAdamWConfig
type _SchedulerInput = list[tuple[str, int]] | tuple[tuple[str, int], ...]


class LLMLRSchedulerConfig(TypedDict):
    scheduler: torch.optim.lr_scheduler.LambdaLR
    interval: str


class LLMLightningOptimizerConfig(TypedDict):
    optimizer: torch.optim.Optimizer
    lr_scheduler: LLMLRSchedulerConfig


class _OptimizationPreset(StrEnum):
    PRETRAIN = auto()
    CPT = auto()
    SFT = auto()


class _OptimizerOption(StrEnum):
    ADAMW = auto()
    MUON = auto()


@dataclass(frozen=True)
class LLMOptimizationConfig:
    optimizer_config: _OptimizerConfig
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    excluded_modules: ExcludedModules = ()
    excluded_module_types: ExcludedModuleTypes = ()

    def __post_init__(self) -> None:
        _validate_optimizer_config(self.optimizer_config)
        _validate_scheduler_config(self.scheduler)
        _validate_excluded_modules(self.excluded_modules)
        _validate_excluded_module_types(self.excluded_module_types)

    @classmethod
    def from_preset(
        cls,
        preset: str = "pretrain",
        *,
        optimizer: str = "adamw",
        scheduler: _SchedulerInput | None = None,
        excluded_modules: ExcludedModules = (),
        excluded_module_types: ExcludedModuleTypes = (),
    ) -> LLMOptimizationConfig:
        return cls(
            optimizer_config=_make_optimizer_config(preset, optimizer=optimizer),
            scheduler=_make_scheduler_config(scheduler),
            excluded_modules=excluded_modules,
            excluded_module_types=excluded_module_types,
        )


def create_llm_optimizer(
    module: nn.Module,
    config: LLMOptimizationConfig,
) -> torch.optim.Optimizer:
    optimizer_config = config.optimizer_config
    if isinstance(optimizer_config, AdamWConfig):
        return create_adamw_optimizer(
            module,
            optimizer_config,
            excluded_modules=config.excluded_modules,
            excluded_module_types=config.excluded_module_types,
            decay_policy=AdamWDecayPolicy.MUON_ELIGIBLE,
        )

    return create_muon_adamw_optimizer(
        module,
        muon=_resolve_muon_config(optimizer_config),
        adamw=_resolve_adamw_config(optimizer_config),
        excluded_modules=config.excluded_modules,
        excluded_module_types=config.excluded_module_types,
    )


def create_llm_lightning_optimizers(
    module: nn.Module,
    config: LLMOptimizationConfig,
) -> LLMLightningOptimizerConfig:
    optimizer = create_llm_optimizer(module, config)
    scheduler = create_scheduler(optimizer, config.scheduler)
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }


def _make_optimizer_config(
    preset: str,
    *,
    optimizer: str,
) -> _OptimizerConfig:
    optimizer_option = _normalize_optimizer_option(optimizer)
    defaults = _preset_defaults(_normalize_optimization_preset(preset))
    adamw_config = AdamWConfig(
        lr=defaults.lr,
        weight_decay=defaults.weight_decay,
        betas=defaults.betas,
        eps=defaults.eps,
    )
    if optimizer_option is _OptimizerOption.ADAMW:
        return adamw_config
    if optimizer_option is _OptimizerOption.MUON:
        return MuonAdamWConfig(
            muon=MuonConfig(
                lr=adamw_config.lr,
                weight_decay=adamw_config.weight_decay,
            ),
            adamw=adamw_config,
    )
    raise ValueError("optimizer must be adamw or muon.")


def _make_scheduler_config(scheduler: _SchedulerInput | None) -> SchedulerConfig:
    if scheduler is None:
        return SchedulerConfig()
    if not isinstance(scheduler, (list, tuple)):
        raise TypeError("scheduler must be a sequence of (shape, duration_steps) tuples.")
    return make_scheduler_config(*scheduler)


def _resolve_adamw_config(optimizer_config: _OptimizerConfig) -> AdamWConfig:
    if isinstance(optimizer_config, AdamWConfig):
        return optimizer_config
    if isinstance(optimizer_config, MuonAdamWConfig):
        return optimizer_config.adamw
    raise TypeError("optimizer_config must be an AdamWConfig or MuonAdamWConfig.")


def _resolve_muon_config(optimizer_config: _OptimizerConfig) -> MuonConfig:
    if isinstance(optimizer_config, MuonAdamWConfig):
        return optimizer_config.muon
    raise TypeError("optimizer_config must be a MuonAdamWConfig.")


@dataclass(frozen=True)
class _PresetDefaults:
    lr: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float = 1e-8


def _preset_defaults(preset: _OptimizationPreset) -> _PresetDefaults:
    if preset is _OptimizationPreset.PRETRAIN:
        return _PresetDefaults(lr=3e-4, weight_decay=0.1, betas=(0.9, 0.95))
    if preset is _OptimizationPreset.CPT:
        return _PresetDefaults(lr=5e-5, weight_decay=0.1, betas=(0.9, 0.95))
    if preset is _OptimizationPreset.SFT:
        return _PresetDefaults(lr=2e-5, weight_decay=0.01, betas=(0.9, 0.999))
    raise ValueError("preset must be pretrain, cpt, or sft.")


def _normalize_optimization_preset(preset: str) -> _OptimizationPreset:
    if not isinstance(preset, str):
        raise TypeError("preset must be a string.")
    try:
        return _OptimizationPreset(preset)
    except ValueError as error:
        raise ValueError("preset must be pretrain, cpt, or sft.") from error


def _normalize_optimizer_option(optimizer: str) -> _OptimizerOption:
    if not isinstance(optimizer, str):
        raise TypeError("optimizer must be a string.")
    try:
        return _OptimizerOption(optimizer)
    except ValueError as error:
        raise ValueError("optimizer must be adamw or muon.") from error


def _validate_optimizer_config(
    optimizer_config: _OptimizerConfig,
) -> None:
    if isinstance(optimizer_config, (AdamWConfig, MuonAdamWConfig)):
        return
    raise TypeError("optimizer_config must be an AdamWConfig or MuonAdamWConfig.")


def _validate_scheduler_config(config: SchedulerConfig) -> None:
    if isinstance(config, SchedulerConfig):
        return
    raise TypeError("scheduler must be a SchedulerConfig.")


def _validate_excluded_modules(excluded_modules: ExcludedModules) -> None:
    if not isinstance(excluded_modules, tuple):
        raise TypeError("excluded_modules must be a tuple of nn.Module instances.")
    for index, excluded_module in enumerate(excluded_modules):
        if not isinstance(excluded_module, nn.Module):
            raise TypeError(f"excluded_modules[{index}] must be an nn.Module instance.")


def _validate_excluded_module_types(
    excluded_module_types: ExcludedModuleTypes,
) -> None:
    if not isinstance(excluded_module_types, tuple):
        raise TypeError("excluded_module_types must be a tuple of nn.Module types.")
    for index, module_type in enumerate(excluded_module_types):
        if not isinstance(module_type, type) or not issubclass(module_type, nn.Module):
            raise TypeError(f"excluded_module_types[{index}] must be a nn.Module subclass.")


__all__ = [
    "LLMLightningOptimizerConfig",
    "LLMLRSchedulerConfig",
    "LLMOptimizationConfig",
    "create_llm_lightning_optimizers",
    "create_llm_optimizer",
]
