from __future__ import annotations

import re
from typing import Final

from torch import nn

OutputHeadNamePattern = str | re.Pattern[str]
ExcludedModuleTypes = tuple[type[nn.Module], ...]
DEFAULT_OUTPUT_HEAD_NAME_PATTERN: Final[re.Pattern[str]] = re.compile("head")

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


def split_muon_params(
    module: nn.Module,
    *,
    requires_grad_only: bool = True,
    output_head_name_pattern: OutputHeadNamePattern | None = DEFAULT_OUTPUT_HEAD_NAME_PATTERN,
    excluded_module_types: ExcludedModuleTypes = (),
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split module parameters into Muon and non-Muon groups.

    By default, Muon receives trainable 2D weight matrices, except parameters
    owned by embedding or normalization modules. Output heads are excluded when
    ``output_head_name_pattern`` matches the owning module name or full
    parameter name; the default pattern excludes names containing ``"head"``.
    Modules matching ``excluded_module_types`` are also excluded. Pass ``None``
    to disable output-head exclusion. All other selected parameters are
    returned in the second list.
    """

    if not isinstance(module, nn.Module):
        raise TypeError("`module` must be an instance of torch.nn.Module.")

    parameter_entries_by_id: dict[int, tuple[nn.Parameter, bool]] = {}
    resolved_output_head_name_pattern = _compile_output_head_name_pattern(
        output_head_name_pattern
    )

    for module_name, child_module in module.named_modules():
        for parameter_name, parameter in child_module.named_parameters(recurse=False):
            if requires_grad_only and not parameter.requires_grad:
                continue

            is_muon = _is_muon_parameter_for_module(
                child_module,
                parameter_name,
                parameter,
                excluded_module_types=excluded_module_types,
            )
            if is_muon:
                is_muon = not _matches_output_head_name(
                    resolved_output_head_name_pattern,
                    parameter_name=_join_parameter_name(module_name, parameter_name),
                    module_name=module_name,
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


def _is_muon_parameter_for_module(
    module: nn.Module,
    parameter_name: str,
    parameter: nn.Parameter,
    *,
    excluded_module_types: ExcludedModuleTypes = (),
) -> bool:
    if (
        isinstance(module, excluded_module_types)
        or _is_embedding_module(module)
        or _is_normalization_module(module)
    ):
        return False
    return _is_muon_parameter_name(parameter_name, parameter)


def _is_muon_parameter_name(parameter_name: str, parameter: nn.Parameter) -> bool:
    return parameter_name == "weight" and parameter.ndim == 2


def _is_embedding_module(module: nn.Module) -> bool:
    return isinstance(module, _EMBEDDING_MODULE_TYPES)


def _is_normalization_module(module: nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    return isinstance(module, _NORMALIZATION_MODULE_TYPES) or class_name.endswith("norm")


def _matches_output_head_name(
    output_head_name_pattern: re.Pattern[str] | None,
    *,
    parameter_name: str,
    module_name: str,
) -> bool:
    if output_head_name_pattern is None:
        return False
    return _pattern_matches(output_head_name_pattern, module_name) or _pattern_matches(
        output_head_name_pattern,
        parameter_name,
    )


def _compile_output_head_name_pattern(
    pattern: OutputHeadNamePattern | None,
) -> re.Pattern[str] | None:
    if pattern is None or isinstance(pattern, re.Pattern):
        return pattern
    return re.compile(pattern)


def _pattern_matches(pattern: re.Pattern[str], value: str) -> bool:
    return pattern.search(value) is not None


def _join_parameter_name(module_name: str, parameter_name: str) -> str:
    if not module_name:
        return parameter_name
    return f"{module_name}.{parameter_name}"


__all__ = [
    "DEFAULT_OUTPUT_HEAD_NAME_PATTERN",
    "ExcludedModuleTypes",
    "OutputHeadNamePattern",
    "split_muon_params",
]
