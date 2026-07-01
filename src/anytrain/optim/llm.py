from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypedDict, cast

import torch
from torch import nn

from ._llm_config import (
    SchedulerInput,
    is_muon_adamw_options,
    make_optimizer_options,
    make_scheduler_config_from_input,
)
from .adamw import create_adamw_optimizer
from .muon import create_muon_adamw_optimizer
from .options import AdamWOptions, OptimizerOptions
from .rules import (
    ExcludedModules,
    LRScaleRules,
    validate_excluded_modules,
)
from .scheduler import (
    Schedule,
    create_scheduler,
    create_scheduler_from_config,
)


class LRSchedulerConfig(TypedDict):
    scheduler: torch.optim.lr_scheduler.LambdaLR
    interval: str


class LightningOptimizerConfig(TypedDict):
    optimizer: torch.optim.Optimizer
    lr_scheduler: LRSchedulerConfig


@dataclass(frozen=True)
class OptimizationConfig:
    optimizer_options: OptimizerOptions
    scheduler: Schedule = field(default_factory=Schedule)
    excluded_modules: ExcludedModules = ()
    lr_scale_rules: LRScaleRules = ()

    def __post_init__(self) -> None:
        _validate_scheduler_config(self.scheduler)
        validate_excluded_modules(self.excluded_modules)

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
        scheduler: SchedulerInput | None = None,
        excluded_modules: ExcludedModules = (),
        lr_scale_rules: LRScaleRules = (),
    ) -> OptimizationConfig:
        return cls(
            optimizer_options=make_optimizer_options(
                preset,
                optimizer=optimizer,
                lr=lr,
                weight_decay=weight_decay,
                betas=betas,
                eps=eps,
                fused=fused,
            ),
            scheduler=make_scheduler_config_from_input(scheduler),
            excluded_modules=excluded_modules,
            lr_scale_rules=lr_scale_rules,
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
    excluded_modules: ExcludedModules = (),
    lr_scale_rules: LRScaleRules = (),
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
            excluded_modules=excluded_modules,
            lr_scale_rules=lr_scale_rules,
        ),
    )


def create_optimizer_from_config(
    module: nn.Module,
    config: OptimizationConfig,
) -> torch.optim.Optimizer:
    optimizer_options = config.optimizer_options
    if is_muon_adamw_options(optimizer_options):
        return create_muon_adamw_optimizer(
            module,
            muon=optimizer_options["muon"],
            adamw=optimizer_options["adamw"],
            excluded_modules=config.excluded_modules,
            lr_scale_rules=config.lr_scale_rules,
        )

    adamw_options = cast(AdamWOptions, optimizer_options)
    return create_adamw_optimizer(
        module,
        adamw_options,
        excluded_modules=config.excluded_modules,
        lr_scale_rules=config.lr_scale_rules,
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
    schedule: str = "constant",
    warmup_steps: int = 0,
    total_steps: int | None = None,
    stable_steps: int | None = None,
    decay_steps: int | None = None,
    min_lr_ratio: float = 0.1,
    excluded_modules: ExcludedModules = (),
    lr_scale_rules: LRScaleRules = (),
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
        excluded_modules=excluded_modules,
        lr_scale_rules=lr_scale_rules,
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


def _validate_scheduler_config(config: Schedule) -> None:
    if isinstance(config, Schedule):
        return
    raise TypeError("scheduler must be a Schedule.")


__all__ = [
    "LightningOptimizerConfig",
    "LRSchedulerConfig",
    "OptimizationConfig",
    "create_lightning_optimizers",
    "create_lightning_optimizers_from_config",
    "create_optimizer",
    "create_optimizer_from_config",
]
