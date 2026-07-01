from __future__ import annotations

import torch
from torch import nn

from ._params import make_scaled_param_groups, split_parameters_by_predicate, validate_module
from .adamw import create_adamw_optimizer
from .compose import CompositeOptimizer
from .options import (
    DEFAULT_MUON_ADJUST_LR_FN,
    AdamWOptions,
    MuonOptions,
)
from .rules import (
    ExcludedModules,
    LRScaleRules,
    is_muon_parameter_for_module,
    resolve_excluded_module_ids,
)


def create_muon_adamw_optimizer(
    module: nn.Module,
    *,
    muon: MuonOptions,
    adamw: AdamWOptions,
    requires_grad_only: bool = True,
    excluded_modules: ExcludedModules = (),
    lr_scale_rules: LRScaleRules = (),
) -> CompositeOptimizer:
    muon_parameters, adamw_parameters = split_muon_params(
        module,
        requires_grad_only=requires_grad_only,
        excluded_modules=excluded_modules,
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
        optimizers["adamw"] = create_adamw_optimizer(
            module,
            adamw,
            requires_grad_only=requires_grad_only,
            selected_params=adamw_parameters,
            excluded_modules=excluded_modules,
            lr_scale_rules=lr_scale_rules,
        )

    return CompositeOptimizer(optimizers)


def split_muon_params(
    module: nn.Module,
    *,
    requires_grad_only: bool = True,
    excluded_modules: ExcludedModules = (),
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split module parameters into Muon and non-Muon groups.

    By default, Muon receives trainable 2D weight matrices, except parameters
    owned by embedding or normalization modules. Modules passed in
    ``excluded_modules`` are also excluded. Shared parameters are only assigned
    to Muon when every owner is Muon-eligible.
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
        )

    return split_parameters_by_predicate(
        module,
        is_muon,
        requires_grad_only=requires_grad_only,
    )


def _create_muon_optimizer(
    module: nn.Module,
    parameters: list[nn.Parameter],
    options: MuonOptions,
    *,
    lr_scale_rules: LRScaleRules = (),
) -> torch.optim.Optimizer:
    muon_cls = getattr(torch.optim, "Muon", None)
    if muon_cls is None:
        raise RuntimeError("torch.optim.Muon is not available in this PyTorch version.")

    optimizer_options = dict(options)
    lr = optimizer_options.pop("lr")
    weight_decay = optimizer_options.pop("weight_decay", 0.1)
    optimizer_options.setdefault("adjust_lr_fn", DEFAULT_MUON_ADJUST_LR_FN)

    param_groups = make_scaled_param_groups(
        module,
        ((parameters, {}),),
        base_lr=lr,
        lr_scale_rules=lr_scale_rules,
    )
    return muon_cls(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
        **optimizer_options,
    )


__all__ = [
    "ExcludedModules",
    "create_muon_adamw_optimizer",
    "split_muon_params",
]
