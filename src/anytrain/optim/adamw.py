from __future__ import annotations

from collections.abc import Collection

import torch
from torch import nn

from ._params import make_scaled_param_groups, split_parameters_by_predicate, validate_module
from .options import AdamWOptions
from .rules import (
    ExcludedModules,
    LRScaleRules,
    is_embedding_module,
    is_normalization_module,
    resolve_excluded_module_ids,
)


def create_adamw_optimizer(
    module: nn.Module,
    options: AdamWOptions,
    *,
    requires_grad_only: bool = True,
    selected_params: Collection[nn.Parameter] | None = None,
    excluded_modules: ExcludedModules = (),
    decay_selected_params: bool = True,
    lr_scale_rules: LRScaleRules = (),
) -> torch.optim.AdamW:
    optimizer_options = dict(options)
    lr = optimizer_options.pop("lr")
    weight_decay = optimizer_options.pop("weight_decay", 0.1)

    decay_params, no_decay_params = split_adamw_decay_params(
        module,
        requires_grad_only=requires_grad_only,
        selected_params=selected_params,
        excluded_modules=excluded_modules,
        decay_selected_params=decay_selected_params,
    )
    raw_param_groups: list[tuple[list[nn.Parameter], dict[str, object]]] = []
    if decay_params:
        raw_param_groups.append((decay_params, {"weight_decay": weight_decay}))
    if no_decay_params:
        raw_param_groups.append((no_decay_params, {"weight_decay": 0.0}))
    if not raw_param_groups:
        raise ValueError("No parameters are available for AdamW.")

    param_groups = make_scaled_param_groups(
        module,
        raw_param_groups,
        base_lr=lr,
        lr_scale_rules=lr_scale_rules,
    )

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        **optimizer_options,
    )


def split_adamw_decay_params(
    module: nn.Module,
    *,
    requires_grad_only: bool = True,
    selected_params: Collection[nn.Parameter] | None = None,
    excluded_modules: ExcludedModules = (),
    decay_selected_params: bool = True,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    validate_module(module)
    if not isinstance(decay_selected_params, bool):
        raise TypeError("decay_selected_params must be a bool.")

    excluded_module_ids = resolve_excluded_module_ids(module, excluded_modules)

    def should_decay(
        child_module: nn.Module,
        parameter_name: str,
        parameter: nn.Parameter,
    ) -> bool:
        if not decay_selected_params:
            return False
        return _is_standard_decay_parameter(
            child_module,
            parameter_name,
            parameter,
            excluded_module_ids=excluded_module_ids,
        )

    return split_parameters_by_predicate(
        module,
        should_decay,
        requires_grad_only=requires_grad_only,
        selected_params=selected_params,
    )


def _is_standard_decay_parameter(
    module: nn.Module,
    parameter_name: str,
    parameter: nn.Parameter,
    *,
    excluded_module_ids: frozenset[int],
) -> bool:
    if id(module) in excluded_module_ids:
        return False
    if is_embedding_module(module):
        return False
    if is_normalization_module(module):
        return False
    return parameter_name == "weight" and parameter.ndim >= 2


__all__ = [
    "create_adamw_optimizer",
    "split_adamw_decay_params",
]
