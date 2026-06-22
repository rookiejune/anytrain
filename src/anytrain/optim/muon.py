from __future__ import annotations

import torch
from torch import nn

from .adamw import AdamWDecayPolicy, create_adamw_optimizer
from .compose import CompositeOptimizer
from .config import AdamWConfig, MuonConfig
from .rules import (
    ExcludedModules,
    ExcludedModuleTypes,
    is_muon_parameter_for_module,
    resolve_excluded_module_ids,
)


def create_muon_adamw_optimizer(
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
        optimizers["adamw"] = create_adamw_optimizer(
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
    "split_muon_params",
]
