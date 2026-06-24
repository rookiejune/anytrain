from __future__ import annotations

import torch
from torch import nn

from .adamw import AdamWDecayPolicy, create_adamw_optimizer_from_config
from .compose import CompositeOptimizer
from .config import AdamWConfig, MuonAdjustLRFn, MuonConfig
from .rules import (
    ExcludedModules,
    ExcludedModuleTypes,
    is_muon_parameter_for_module,
    resolve_excluded_module_ids,
)


def create_muon_adamw_optimizer(
    module: nn.Module,
    *,
    lr: float,
    weight_decay: float = 0.1,
    adamw_lr: float | None = None,
    adamw_weight_decay: float | None = None,
    adamw_betas: tuple[float, float] = (0.9, 0.95),
    adamw_eps: float = 1e-8,
    adamw_fused: bool | None = None,
    muon_lr: float | None = None,
    muon_weight_decay: float | None = None,
    muon_momentum: float = 0.95,
    muon_nesterov: bool = True,
    muon_ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315),
    muon_eps: float = 1e-7,
    muon_ns_steps: int = 5,
    muon_adjust_lr_fn: MuonAdjustLRFn | str = MuonAdjustLRFn.MATCH_RMS_ADAMW,
    requires_grad_only: bool = True,
    excluded_modules: ExcludedModules = (),
    excluded_module_types: ExcludedModuleTypes = (),
) -> CompositeOptimizer:
    return create_muon_adamw_optimizer_from_config(
        module,
        muon=MuonConfig(
            lr=lr if muon_lr is None else muon_lr,
            weight_decay=weight_decay if muon_weight_decay is None else muon_weight_decay,
            momentum=muon_momentum,
            nesterov=muon_nesterov,
            ns_coefficients=muon_ns_coefficients,
            eps=muon_eps,
            ns_steps=muon_ns_steps,
            adjust_lr_fn=muon_adjust_lr_fn,
        ),
        adamw=AdamWConfig(
            lr=lr if adamw_lr is None else adamw_lr,
            weight_decay=weight_decay if adamw_weight_decay is None else adamw_weight_decay,
            betas=adamw_betas,
            eps=adamw_eps,
            fused=adamw_fused,
        ),
        requires_grad_only=requires_grad_only,
        excluded_modules=excluded_modules,
        excluded_module_types=excluded_module_types,
    )


def create_muon_adamw_optimizer_from_config(
    module: nn.Module,
    *,
    muon: MuonConfig,
    adamw: AdamWConfig,
    requires_grad_only: bool = True,
    excluded_modules: ExcludedModules = (),
    excluded_module_types: ExcludedModuleTypes = (),
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
        "muon": _create_muon_optimizer(muon_parameters, muon),
    }
    if adamw_parameters:
        optimizers["adamw"] = create_adamw_optimizer_from_config(
            module,
            adamw,
            requires_grad_only=requires_grad_only,
            selected_params=adamw_parameters,
            excluded_modules=excluded_modules,
            excluded_module_types=excluded_module_types,
            decay_selected_params=False,
            decay_policy=AdamWDecayPolicy.MUON_ELIGIBLE,
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

    if not isinstance(module, nn.Module):
        raise TypeError("`module` must be an instance of torch.nn.Module.")

    parameter_entries_by_id: dict[int, tuple[nn.Parameter, bool]] = {}
    excluded_module_ids = resolve_excluded_module_ids(module, excluded_modules)

    for _, child_module in module.named_modules():
        for parameter_name, parameter in child_module.named_parameters(recurse=False):
            if requires_grad_only and not parameter.requires_grad:
                continue

            is_muon = is_muon_parameter_for_module(
                child_module,
                parameter_name,
                parameter,
                excluded_module_ids=excluded_module_ids,
                excluded_module_types=excluded_module_types,
            )

            parameter_id = id(parameter)
            previous_entry = parameter_entries_by_id.get(parameter_id)
            if previous_entry is None:
                parameter_entries_by_id[parameter_id] = (parameter, is_muon)
            else:
                parameter_entries_by_id[parameter_id] = (parameter, previous_entry[1] and is_muon)

    muon_parameters: list[nn.Parameter] = []
    non_muon_parameters: list[nn.Parameter] = []
    for parameter, is_muon in parameter_entries_by_id.values():
        if is_muon:
            muon_parameters.append(parameter)
        else:
            non_muon_parameters.append(parameter)

    return muon_parameters, non_muon_parameters


def _create_muon_optimizer(
    parameters: list[nn.Parameter],
    config: MuonConfig,
) -> torch.optim.Optimizer:
    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError("torch.optim.Muon is not available in this PyTorch version.")
    return muon_cls(
        parameters,
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
