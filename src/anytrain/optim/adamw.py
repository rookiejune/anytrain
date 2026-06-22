from __future__ import annotations

from collections.abc import Collection
from enum import StrEnum, auto

import torch
from torch import nn

from .config import AdamWConfig
from .rules import (
    ExcludedModules,
    ExcludedModuleTypes,
    is_embedding_module,
    is_muon_parameter_for_module,
    is_normalization_module,
    resolve_excluded_module_ids,
)


class AdamWDecayPolicy(StrEnum):
    STANDARD = auto()
    MUON_ELIGIBLE = auto()


def create_adamw_optimizer(
    module: nn.Module,
    config: AdamWConfig,
    *,
    requires_grad_only: bool = True,
    selected_params: Collection[nn.Parameter] | None = None,
    excluded_modules: ExcludedModules = (),
    excluded_module_types: ExcludedModuleTypes = (),
    decay_selected_params: bool = True,
    decay_policy: AdamWDecayPolicy | str = AdamWDecayPolicy.STANDARD,
) -> torch.optim.AdamW:
    decay_params, no_decay_params = split_adamw_decay_params(
        module,
        requires_grad_only=requires_grad_only,
        selected_params=selected_params,
        excluded_modules=excluded_modules,
        excluded_module_types=excluded_module_types,
        decay_selected_params=decay_selected_params,
        decay_policy=decay_policy,
    )
    param_groups: list[dict[str, object]] = []
    if decay_params:
        param_groups.append({"params": decay_params, "weight_decay": config.weight_decay})
    if no_decay_params:
        param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
    if not param_groups:
        raise ValueError("No parameters are available for AdamW.")

    if config.fused is None:
        return torch.optim.AdamW(
            param_groups,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
        )
    return torch.optim.AdamW(
        param_groups,
        lr=config.lr,
        betas=config.betas,
        eps=config.eps,
        fused=config.fused,
    )


def split_adamw_decay_params(
    module: nn.Module,
    *,
    requires_grad_only: bool = True,
    selected_params: Collection[nn.Parameter] | None = None,
    excluded_modules: ExcludedModules = (),
    excluded_module_types: ExcludedModuleTypes = (),
    decay_selected_params: bool = True,
    decay_policy: AdamWDecayPolicy | str = AdamWDecayPolicy.STANDARD,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    if not isinstance(module, nn.Module):
        raise TypeError("`module` must be an instance of torch.nn.Module.")
    if not isinstance(decay_selected_params, bool):
        raise TypeError("decay_selected_params must be a bool.")

    resolved_decay_policy = _normalize_decay_policy(decay_policy)
    selected_param_ids = _resolve_selected_param_ids(module, selected_params)
    excluded_module_ids = resolve_excluded_module_ids(module, excluded_modules)
    parameter_entries_by_id: dict[int, tuple[nn.Parameter, bool]] = {}

    for _, child_module in module.named_modules():
        for parameter_name, parameter in child_module.named_parameters(recurse=False):
            if requires_grad_only and not parameter.requires_grad:
                continue
            if selected_param_ids is not None and id(parameter) not in selected_param_ids:
                continue

            should_decay = False
            if decay_selected_params:
                should_decay = _should_decay_parameter(
                    child_module,
                    parameter_name,
                    parameter,
                    excluded_module_ids=excluded_module_ids,
                    excluded_module_types=excluded_module_types,
                    decay_policy=resolved_decay_policy,
                )
            parameter_id = id(parameter)
            previous_entry = parameter_entries_by_id.get(parameter_id)
            if previous_entry is None:
                parameter_entries_by_id[parameter_id] = (parameter, should_decay)
            else:
                parameter_entries_by_id[parameter_id] = (
                    parameter,
                    previous_entry[1] and should_decay,
                )

    decay_params: list[nn.Parameter] = []
    no_decay_params: list[nn.Parameter] = []
    for parameter, should_decay in parameter_entries_by_id.values():
        if should_decay:
            decay_params.append(parameter)
        else:
            no_decay_params.append(parameter)
    return decay_params, no_decay_params


def _should_decay_parameter(
    module: nn.Module,
    parameter_name: str,
    parameter: nn.Parameter,
    *,
    excluded_module_ids: frozenset[int],
    excluded_module_types: ExcludedModuleTypes,
    decay_policy: AdamWDecayPolicy,
) -> bool:
    if decay_policy is AdamWDecayPolicy.MUON_ELIGIBLE:
        return is_muon_parameter_for_module(
            module,
            parameter_name,
            parameter,
            excluded_module_ids=excluded_module_ids,
            excluded_module_types=excluded_module_types,
        )
    return _is_standard_decay_parameter(
        module,
        parameter_name,
        parameter,
        excluded_module_ids=excluded_module_ids,
        excluded_module_types=excluded_module_types,
    )


def _is_standard_decay_parameter(
    module: nn.Module,
    parameter_name: str,
    parameter: nn.Parameter,
    *,
    excluded_module_ids: frozenset[int],
    excluded_module_types: ExcludedModuleTypes,
) -> bool:
    if is_muon_parameter_for_module(
        module,
        parameter_name,
        parameter,
        excluded_module_ids=excluded_module_ids,
        excluded_module_types=excluded_module_types,
    ):
        return True
    if id(module) in excluded_module_ids or isinstance(module, excluded_module_types):
        return False
    if is_embedding_module(module):
        return False
    if is_normalization_module(module):
        return False
    return parameter_name == "weight" and parameter.ndim >= 2


def _normalize_decay_policy(decay_policy: AdamWDecayPolicy | str) -> AdamWDecayPolicy:
    if isinstance(decay_policy, AdamWDecayPolicy):
        return decay_policy
    if not isinstance(decay_policy, str):
        raise TypeError("decay_policy must be a string or AdamWDecayPolicy.")
    try:
        return AdamWDecayPolicy(decay_policy)
    except ValueError as error:
        raise ValueError("decay_policy must be standard or muon_eligible.") from error


def _resolve_selected_param_ids(
    module: nn.Module,
    selected_params: Collection[nn.Parameter] | None,
) -> set[int] | None:
    if selected_params is None:
        return None
    if not isinstance(selected_params, Collection):
        raise TypeError("selected_params must be a collection of nn.Parameter.")

    module_param_ids = {id(parameter) for parameter in module.parameters()}
    selected_param_ids: set[int] = set()
    for index, parameter in enumerate(selected_params):
        if not isinstance(parameter, nn.Parameter):
            raise TypeError(f"selected_params[{index}] must be an nn.Parameter.")
        if id(parameter) not in module_param_ids:
            raise ValueError(f"selected_params[{index}] must belong to `module`.")
        selected_param_ids.add(id(parameter))
    return selected_param_ids


__all__ = [
    "AdamWDecayPolicy",
    "create_adamw_optimizer",
    "split_adamw_decay_params",
]
