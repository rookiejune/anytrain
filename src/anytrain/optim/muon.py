from __future__ import annotations

import torch
from torch import nn

from ._params import make_scaled_param_groups, split_parameters_by_predicate, validate_module
from .adamw import AdamWDecayPolicy, create_adamw_optimizer_from_config
from .compose import CompositeOptimizer
from .config import AdamWConfig, MuonAdjustLRFn, MuonConfig
from .rules import (
    ExcludedModules,
    ExcludedModuleTypes,
    LRScaleRules,
    is_muon_parameter_for_module,
    resolve_excluded_module_ids,
)


def create_muon_adamw_optimizer(
    module: nn.Module,
    *,
    muon_lr: float,
    adamw_lr: float,
    muon_weight_decay: float = 0.1,
    adamw_weight_decay: float = 0.1,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
    adamw_fused: bool | None = None,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
    muon_eps: float = 1e-7,
    muon_ns_steps: int = 5,
    muon_adjust_lr_fn: MuonAdjustLRFn | str = MuonAdjustLRFn.MATCH_RMS_ADAMW,
    requires_grad_only: bool = True,
    excluded_modules: ExcludedModules = (),
    excluded_module_types: ExcludedModuleTypes = (),
    lr_scale_rules: LRScaleRules = (),
) -> CompositeOptimizer:
    return create_muon_adamw_optimizer_from_config(
        module,
        muon=MuonConfig(
            lr=muon_lr,
            weight_decay=muon_weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_coefficients=muon_ns_coefficients,
            eps=muon_eps,
            ns_steps=muon_ns_steps,
            adjust_lr_fn=muon_adjust_lr_fn,
        ),
        adamw=AdamWConfig(
            lr=adamw_lr,
            weight_decay=adamw_weight_decay,
            betas=adamw_betas,
            eps=adamw_eps,
            fused=adamw_fused,
        ),
        requires_grad_only=requires_grad_only,
        excluded_modules=excluded_modules,
        excluded_module_types=excluded_module_types,
        lr_scale_rules=lr_scale_rules,
    )


def create_muon_adamw_optimizer_from_config(
    module: nn.Module,
    *,
    muon: MuonConfig,
    adamw: AdamWConfig,
    requires_grad_only: bool = True,
    excluded_modules: ExcludedModules = (),
    excluded_module_types: ExcludedModuleTypes = (),
    lr_scale_rules: LRScaleRules = (),
) -> CompositeOptimizer:
    muon_parameters, adamw_parameters = split_muon_params(
        module,
        requires_grad_only=requires_grad_only,
        excluded_modules=excluded_modules,
        excluded_module_types=excluded_module_types,
    )
    if not muon_parameters:
        raise ValueError("No parameters are eligible for Muon.")

    optimizers: dict[str, torch.optim.Optimizer] = {
        "muon": _create_muon_optimizer(
            module,
            muon_parameters,
            muon,
            lr_scale_rules=lr_scale_rules,
        ),
    }
    if adamw_parameters:
        optimizers["adamw"] = create_adamw_optimizer_from_config(
            module,
            adamw,
            requires_grad_only=requires_grad_only,
            selected_params=adamw_parameters,
            excluded_modules=excluded_modules,
            excluded_module_types=excluded_module_types,
            decay_policy=AdamWDecayPolicy.STANDARD,
            lr_scale_rules=lr_scale_rules,
        )

    return CompositeOptimizer(optimizers)


def split_muon_params(
    module: nn.Module,
    *,
    requires_grad_only: bool = True,
    excluded_modules: ExcludedModules = (),
    excluded_module_types: ExcludedModuleTypes = (),
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split module parameters into Muon and non-Muon groups.

    By default, Muon receives trainable 2D weight matrices, except parameters
    owned by embedding or normalization modules. Modules passed in
    ``excluded_modules`` and modules matching ``excluded_module_types`` are also
    excluded. Shared parameters are only assigned to Muon when every owner is
    Muon-eligible.
    """

    validate_module(module)
    excluded_module_ids = resolve_excluded_module_ids(module, excluded_modules)

    def is_muon(
        child_module: nn.Module,
        parameter_name: str,
        parameter: nn.Parameter,
    ) -> bool:
        return is_muon_parameter_for_module(
            child_module,
            parameter_name,
            parameter,
            excluded_module_ids=excluded_module_ids,
            excluded_module_types=excluded_module_types,
        )

    return split_parameters_by_predicate(
        module,
        is_muon,
        requires_grad_only=requires_grad_only,
    )


def _create_muon_optimizer(
    module: nn.Module,
    parameters: list[nn.Parameter],
    config: MuonConfig,
    *,
    lr_scale_rules: LRScaleRules = (),
) -> torch.optim.Optimizer:
    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError("torch.optim.Muon is not available in this PyTorch version.")
    param_groups = make_scaled_param_groups(
        module,
        ((parameters, {}),),
        base_lr=config.lr,
        lr_scale_rules=lr_scale_rules,
    )
    return muon_cls(
        param_groups,
        lr=config.lr,
        weight_decay=config.weight_decay,
        momentum=config.momentum,
        nesterov=config.nesterov,
        ns_coefficients=config.ns_coefficients,
        eps=config.eps,
        ns_steps=config.ns_steps,
        adjust_lr_fn=config.adjust_lr_fn,
    )


__all__ = [
    "ExcludedModules",
    "ExcludedModuleTypes",
    "create_muon_adamw_optimizer",
    "create_muon_adamw_optimizer_from_config",
    "split_muon_params",
]
