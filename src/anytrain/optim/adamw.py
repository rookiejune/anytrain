from __future__ import annotations

from collections.abc import Collection
from enum import StrEnum, auto

import torch
from torch import nn

from ._params import make_scaled_param_groups, split_parameters_by_predicate, validate_module
from .config import AdamWConfig
from .rules import (
    ExcludedModules,
    ExcludedModuleTypes,
    LRScaleRules,
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
    *,
    lr: float,
    weight_decay: float = 0.1,
    betas: tuple[float, float] = (0.9, 0.95),
    eps: float = 1e-8,
    fused: bool | None = None,
    requires_grad_only: bool = True,
    selected_params: Collection[nn.Parameter] | None = None,
    excluded_modules: ExcludedModules = (),
    excluded_module_types: ExcludedModuleTypes = (),
    decay_selected_params: bool = True,
    decay_policy: AdamWDecayPolicy | str = AdamWDecayPolicy.STANDARD,
    lr_scale_rules: LRScaleRules = (),
) -> torch.optim.AdamW:
    return create_adamw_optimizer_from_config(
        module,
        AdamWConfig(
            lr=lr,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            fused=fused,
        ),
        requires_grad_only=requires_grad_only,
        selected_params=selected_params,
        excluded_modules=excluded_modules,
        excluded_module_types=excluded_module_types,
        decay_selected_params=decay_selected_params,
        decay_policy=decay_policy,
        lr_scale_rules=lr_scale_rules,
    )


def create_adamw_optimizer_from_config(
    module: nn.Module,
    config: AdamWConfig,
    *,
    requires_grad_only: bool = True,
    selected_params: Collection[nn.Parameter] | None = None,
    excluded_modules: ExcludedModules = (),
    excluded_module_types: ExcludedModuleTypes = (),
    decay_selected_params: bool = True,
    decay_policy: AdamWDecayPolicy | str = AdamWDecayPolicy.STANDARD,
    lr_scale_rules: LRScaleRules = (),
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
    raw_param_groups: list[tuple[list[nn.Parameter], dict[str, object]]] = []
    if decay_params:
        raw_param_groups.append((decay_params, {"weight_decay": config.weight_decay}))
    if no_decay_params:
        raw_param_groups.append((no_decay_params, {"weight_decay": 0.0}))
    if not raw_param_groups:
        raise ValueError("No parameters are available for AdamW.")

    param_groups = make_scaled_param_groups(
        module,
        raw_param_groups,
        base_lr=config.lr,
        lr_scale_rules=lr_scale_rules,
    )

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
    validate_module(module)
    if not isinstance(decay_selected_params, bool):
        raise TypeError("decay_selected_params must be a bool.")

    resolved_decay_policy = _normalize_decay_policy(decay_policy)
    excluded_module_ids = resolve_excluded_module_ids(module, excluded_modules)

    def should_decay(
        child_module: nn.Module,
        parameter_name: str,
        parameter: nn.Parameter,
    ) -> bool:
        if not decay_selected_params:
            return False
        return _should_decay_parameter(
            child_module,
            parameter_name,
            parameter,
            excluded_module_ids=excluded_module_ids,
            excluded_module_types=excluded_module_types,
            decay_policy=resolved_decay_policy,
        )

    return split_parameters_by_predicate(
        module,
        should_decay,
        requires_grad_only=requires_grad_only,
        selected_params=selected_params,
    )


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


__all__ = [
    "AdamWDecayPolicy",
    "create_adamw_optimizer",
    "create_adamw_optimizer_from_config",
    "split_adamw_decay_params",
]
