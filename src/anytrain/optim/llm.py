from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import TypedDict

import torch
from torch import nn

from .adamw import AdamWDecayPolicy, create_adamw_optimizer_from_config
from .config import AdamWConfig, MuonAdamWConfig, MuonAdjustLRFn, MuonConfig
from .muon import (
    ExcludedModules,
    ExcludedModuleTypes,
    create_muon_adamw_optimizer_from_config,
)
from .scheduler import (
    SchedulerConfig,
    create_scheduler,
    create_scheduler_from_config,
    make_scheduler_config,
)

type _OptimizerConfig = AdamWConfig | MuonAdamWConfig
type _SchedulerInput = list[tuple[str, int]] | tuple[tuple[str, int], ...]


class LRSchedulerConfig(TypedDict):
    scheduler: torch.optim.lr_scheduler.LambdaLR
    interval: str


class LightningOptimizerConfig(TypedDict):
    optimizer: torch.optim.Optimizer
    lr_scheduler: LRSchedulerConfig


class _OptimizationPreset(StrEnum):
    PRETRAIN = auto()
    CPT = auto()
    SFT = auto()


class _OptimizerOption(StrEnum):
    ADAMW = auto()
    MUON = auto()


_DEFAULT_MUON_MOMENTUM = 0.95
_DEFAULT_MUON_NESTEROV = True
_DEFAULT_MUON_NS_COEFFICIENTS = (3.4445, -4.775, 2.0315)
_DEFAULT_MUON_EPS = 1e-7
_DEFAULT_MUON_NS_STEPS = 5
_DEFAULT_MUON_ADJUST_LR_FN = MuonAdjustLRFn.MATCH_RMS_ADAMW


@dataclass(frozen=True)
class OptimizationConfig:
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
        lr: float | None = None,
        weight_decay: float | None = None,
        betas: tuple[float, float] | None = None,
        eps: float | None = None,
        fused: bool | None = None,
        muon_lr: float | None = None,
        muon_weight_decay: float | None = None,
        muon_momentum: float = _DEFAULT_MUON_MOMENTUM,
        muon_nesterov: bool = _DEFAULT_MUON_NESTEROV,
        muon_ns_coefficients: tuple[float, float, float] = _DEFAULT_MUON_NS_COEFFICIENTS,
        muon_eps: float = _DEFAULT_MUON_EPS,
        muon_ns_steps: int = _DEFAULT_MUON_NS_STEPS,
        muon_adjust_lr_fn: MuonAdjustLRFn | str = _DEFAULT_MUON_ADJUST_LR_FN,
        scheduler: _SchedulerInput | None = None,
        excluded_modules: ExcludedModules = (),
        excluded_module_types: ExcludedModuleTypes = (),
    ) -> OptimizationConfig:
        return cls(
            optimizer_config=_make_optimizer_config(
                preset,
                optimizer=optimizer,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                fused=fused,
                muon_lr=muon_lr,
                muon_weight_decay=muon_weight_decay,
                muon_momentum=muon_momentum,
                muon_nesterov=muon_nesterov,
                muon_ns_coefficients=muon_ns_coefficients,
                muon_eps=muon_eps,
                muon_ns_steps=muon_ns_steps,
                muon_adjust_lr_fn=muon_adjust_lr_fn,
            ),
            scheduler=_make_scheduler_config(scheduler),
            excluded_modules=excluded_modules,
            excluded_module_types=excluded_module_types,
        )


def create_optimizer(
    module: nn.Module,
    *,
    preset: str = "pretrain",
    optimizer: str = "adamw",
    lr: float | None = None,
    weight_decay: float | None = None,
    betas: tuple[float, float] | None = None,
    eps: float | None = None,
    fused: bool | None = None,
    muon_lr: float | None = None,
    muon_weight_decay: float | None = None,
    muon_momentum: float = _DEFAULT_MUON_MOMENTUM,
    muon_nesterov: bool = _DEFAULT_MUON_NESTEROV,
    muon_ns_coefficients: tuple[float, float, float] = _DEFAULT_MUON_NS_COEFFICIENTS,
    muon_eps: float = _DEFAULT_MUON_EPS,
    muon_ns_steps: int = _DEFAULT_MUON_NS_STEPS,
    muon_adjust_lr_fn: MuonAdjustLRFn | str = _DEFAULT_MUON_ADJUST_LR_FN,
    excluded_modules: ExcludedModules = (),
    excluded_module_types: ExcludedModuleTypes = (),
) -> torch.optim.Optimizer:
    return create_optimizer_from_config(
        module,
        OptimizationConfig.from_preset(
            preset,
            optimizer=optimizer,
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            fused=fused,
            muon_lr=muon_lr,
            muon_weight_decay=muon_weight_decay,
            muon_momentum=muon_momentum,
            muon_nesterov=muon_nesterov,
            muon_ns_coefficients=muon_ns_coefficients,
            muon_eps=muon_eps,
            muon_ns_steps=muon_ns_steps,
            muon_adjust_lr_fn=muon_adjust_lr_fn,
            excluded_modules=excluded_modules,
            excluded_module_types=excluded_module_types,
        ),
    )


def create_optimizer_from_config(
    module: nn.Module,
    config: OptimizationConfig,
) -> torch.optim.Optimizer:
    optimizer_config = config.optimizer_config
    if isinstance(optimizer_config, AdamWConfig):
        return create_adamw_optimizer_from_config(
            module,
            optimizer_config,
            excluded_modules=config.excluded_modules,
            excluded_module_types=config.excluded_module_types,
            decay_policy=AdamWDecayPolicy.MUON_ELIGIBLE,
        )

    return create_muon_adamw_optimizer_from_config(
        module,
        muon=_resolve_muon_config(optimizer_config),
        adamw=_resolve_adamw_config(optimizer_config),
        excluded_modules=config.excluded_modules,
        excluded_module_types=config.excluded_module_types,
    )


def create_lightning_optimizers(
    module: nn.Module,
    *,
    preset: str = "pretrain",
    optimizer: str = "adamw",
    lr: float | None = None,
    weight_decay: float | None = None,
    betas: tuple[float, float] | None = None,
    eps: float | None = None,
    fused: bool | None = None,
    muon_lr: float | None = None,
    muon_weight_decay: float | None = None,
    muon_momentum: float = _DEFAULT_MUON_MOMENTUM,
    muon_nesterov: bool = _DEFAULT_MUON_NESTEROV,
    muon_ns_coefficients: tuple[float, float, float] = _DEFAULT_MUON_NS_COEFFICIENTS,
    muon_eps: float = _DEFAULT_MUON_EPS,
    muon_ns_steps: int = _DEFAULT_MUON_NS_STEPS,
    muon_adjust_lr_fn: MuonAdjustLRFn | str = _DEFAULT_MUON_ADJUST_LR_FN,
    schedule: str = "constant",
    warmup_steps: int = 0,
    total_steps: int | None = None,
    stable_steps: int | None = None,
    decay_steps: int | None = None,
    min_lr_ratio: float = 0.1,
    excluded_modules: ExcludedModules = (),
    excluded_module_types: ExcludedModuleTypes = (),
) -> LightningOptimizerConfig:
    optimizer_instance = create_optimizer(
        module,
        preset=preset,
        optimizer=optimizer,
        lr=lr,
        weight_decay=weight_decay,
        betas=betas,
        eps=eps,
        fused=fused,
        muon_lr=muon_lr,
        muon_weight_decay=muon_weight_decay,
        muon_momentum=muon_momentum,
        muon_nesterov=muon_nesterov,
        muon_ns_coefficients=muon_ns_coefficients,
        muon_eps=muon_eps,
        muon_ns_steps=muon_ns_steps,
        muon_adjust_lr_fn=muon_adjust_lr_fn,
        excluded_modules=excluded_modules,
        excluded_module_types=excluded_module_types,
    )
    scheduler = create_scheduler(
        optimizer_instance,
        schedule=schedule,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        stable_steps=stable_steps,
        decay_steps=decay_steps,
        min_lr_ratio=min_lr_ratio,
    )
    return {
        "optimizer": optimizer_instance,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
        },
    }


def create_lightning_optimizers_from_config(
    module: nn.Module,
    config: OptimizationConfig,
) -> LightningOptimizerConfig:
    optimizer = create_optimizer_from_config(module, config)
    scheduler = create_scheduler_from_config(optimizer, config.scheduler)
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
    lr: float | None,
    weight_decay: float | None,
    betas: tuple[float, float] | None,
    eps: float | None,
    fused: bool | None,
    muon_lr: float | None,
    muon_weight_decay: float | None,
    muon_momentum: float,
    muon_nesterov: bool,
    muon_ns_coefficients: tuple[float, float, float],
    muon_eps: float,
    muon_ns_steps: int,
    muon_adjust_lr_fn: MuonAdjustLRFn | str,
) -> _OptimizerConfig:
    optimizer_option = _normalize_optimizer_option(optimizer)
    defaults = _preset_defaults(_normalize_optimization_preset(preset))
    adamw_config = AdamWConfig(
        lr=defaults.lr if lr is None else lr,
        weight_decay=defaults.weight_decay if weight_decay is None else weight_decay,
        betas=defaults.betas if betas is None else betas,
        eps=defaults.eps if eps is None else eps,
        fused=fused,
    )
    if optimizer_option is _OptimizerOption.ADAMW:
        _reject_muon_options_for_adamw(
            muon_lr=muon_lr,
            muon_weight_decay=muon_weight_decay,
            muon_momentum=muon_momentum,
            muon_nesterov=muon_nesterov,
            muon_ns_coefficients=muon_ns_coefficients,
            muon_eps=muon_eps,
            muon_ns_steps=muon_ns_steps,
            muon_adjust_lr_fn=muon_adjust_lr_fn,
        )
        return adamw_config
    if optimizer_option is _OptimizerOption.MUON:
        return MuonAdamWConfig(
            muon=MuonConfig(
                lr=adamw_config.lr if muon_lr is None else muon_lr,
                weight_decay=adamw_config.weight_decay
                if muon_weight_decay is None
                else muon_weight_decay,
                momentum=muon_momentum,
                nesterov=muon_nesterov,
                ns_coefficients=muon_ns_coefficients,
                eps=muon_eps,
                ns_steps=muon_ns_steps,
                adjust_lr_fn=muon_adjust_lr_fn,
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


def _reject_muon_options_for_adamw(
    *,
    muon_lr: float | None,
    muon_weight_decay: float | None,
    muon_momentum: float,
    muon_nesterov: bool,
    muon_ns_coefficients: tuple[float, float, float],
    muon_eps: float,
    muon_ns_steps: int,
    muon_adjust_lr_fn: MuonAdjustLRFn | str,
) -> None:
    default_adjust_lr_values = {
        _DEFAULT_MUON_ADJUST_LR_FN,
        _DEFAULT_MUON_ADJUST_LR_FN.value,
    }
    if (
        muon_lr is not None
        or muon_weight_decay is not None
        or muon_momentum != _DEFAULT_MUON_MOMENTUM
        or muon_nesterov != _DEFAULT_MUON_NESTEROV
        or muon_ns_coefficients != _DEFAULT_MUON_NS_COEFFICIENTS
        or muon_eps != _DEFAULT_MUON_EPS
        or muon_ns_steps != _DEFAULT_MUON_NS_STEPS
        or muon_adjust_lr_fn not in default_adjust_lr_values
    ):
        raise ValueError("muon_* options require optimizer='muon'.")


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
    "LightningOptimizerConfig",
    "LRSchedulerConfig",
    "OptimizationConfig",
    "create_lightning_optimizers",
    "create_lightning_optimizers_from_config",
    "create_optimizer",
    "create_optimizer_from_config",
]
