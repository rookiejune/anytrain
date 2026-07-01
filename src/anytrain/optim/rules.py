from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypedDict

from torch import nn

ExcludedModules = tuple[nn.Module, ...]
ExcludedModuleTypes = tuple[type[nn.Module], ...]


class LRScaleRule(TypedDict):
    name: str
    lr_scale: float


type LRScaleRules = Sequence[LRScaleRule]

_EMBEDDING_MODULE_TYPES: Final[tuple[type[nn.Module], ...]] = (nn.Embedding, nn.EmbeddingBag)
_NORMALIZATION_MODULE_TYPES: Final[tuple[type[nn.Module], ...]] = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.InstanceNorm1d,
    nn.InstanceNorm2d,
    nn.InstanceNorm3d,
    nn.LayerNorm,
    nn.LocalResponseNorm,
    nn.SyncBatchNorm,
)


def resolve_excluded_module_ids(
    module: nn.Module,
    excluded_modules: ExcludedModules,
) -> frozenset[int]:
    validate_excluded_modules(excluded_modules)

    module_ids = {id(child_module) for child_module in module.modules()}
    excluded_module_ids: set[int] = set()
    for index, excluded_module in enumerate(excluded_modules):
        if id(excluded_module) not in module_ids:
            raise ValueError(f"excluded_modules[{index}] must belong to `module`.")
        excluded_module_ids.update(id(child_module) for child_module in excluded_module.modules())
    return frozenset(excluded_module_ids)


def validate_excluded_modules(excluded_modules: ExcludedModules) -> None:
    if not isinstance(excluded_modules, tuple):
        raise TypeError("excluded_modules must be a tuple of nn.Module instances.")
    for index, excluded_module in enumerate(excluded_modules):
        if not isinstance(excluded_module, nn.Module):
            raise TypeError(f"excluded_modules[{index}] must be an nn.Module instance.")


def validate_excluded_module_types(excluded_module_types: ExcludedModuleTypes) -> None:
    if not isinstance(excluded_module_types, tuple):
        raise TypeError("excluded_module_types must be a tuple of nn.Module types.")
    for index, module_type in enumerate(excluded_module_types):
        if not isinstance(module_type, type) or not issubclass(module_type, nn.Module):
            raise TypeError(f"excluded_module_types[{index}] must be a nn.Module subclass.")


def is_muon_parameter_for_module(
    module: nn.Module,
    parameter_name: str,
    parameter: nn.Parameter,
    *,
    excluded_module_ids: frozenset[int] | None = None,
    excluded_module_types: ExcludedModuleTypes = (),
) -> bool:
    excluded_ids = frozenset() if excluded_module_ids is None else excluded_module_ids
    if (
        id(module) in excluded_ids
        or isinstance(module, excluded_module_types)
        or is_embedding_module(module)
        or is_normalization_module(module)
    ):
        return False
    return parameter_name == "weight" and parameter.ndim == 2


def is_embedding_module(module: nn.Module) -> bool:
    return isinstance(module, _EMBEDDING_MODULE_TYPES)


def is_normalization_module(module: nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    return isinstance(module, _NORMALIZATION_MODULE_TYPES) or class_name.endswith("norm")


__all__ = [
    "ExcludedModules",
    "ExcludedModuleTypes",
    "LRScaleRule",
    "LRScaleRules",
    "is_embedding_module",
    "is_muon_parameter_for_module",
    "is_normalization_module",
    "resolve_excluded_module_ids",
    "validate_excluded_module_types",
    "validate_excluded_modules",
]
