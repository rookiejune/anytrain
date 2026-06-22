from __future__ import annotations

from collections.abc import Callable, Collection, Iterator
from dataclasses import dataclass
from typing import Final

from torch import nn

MuonParameterPredicate = Callable[[str, nn.Parameter], bool]

DEFAULT_OUTPUT_HEAD_MODULE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "head",
        "heads",
        "lm_head",
        "output_head",
        "output_projection",
        "classifier",
        "classification_head",
        "regressor",
        "regression_head",
        "score",
        "cls",
        "prediction_head",
        "predictions",
        "to_logits",
        "logits",
    }
)

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
_EMBEDDING_MODULE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "embed",
        "emb",
        "embedding",
        "embeddings",
        "embed_tokens",
        "token_embedding",
        "token_embeddings",
        "position_embedding",
        "position_embeddings",
        "pos_embedding",
        "pos_embeddings",
        "wte",
        "wpe",
    }
)
_NORMALIZATION_MODULE_NAMES: Final[frozenset[str]] = frozenset(
    {
        "norm",
        "ln",
        "ln_f",
        "layer_norm",
        "layernorm",
        "rms_norm",
        "rmsnorm",
        "final_norm",
        "post_norm",
        "pre_norm",
    }
)


@dataclass(frozen=True)
class _ParameterContext:
    name: str
    parameter: nn.Parameter
    module_name: str
    module: nn.Module
    parameter_name: str


def is_default_muon_parameter(
    name: str,
    parameter: nn.Parameter,
    *,
    module: nn.Module | None = None,
    module_name: str | None = None,
    parameter_name: str | None = None,
    output_head_module_names: Collection[str] = DEFAULT_OUTPUT_HEAD_MODULE_NAMES,
) -> bool:
    """Return whether a parameter should use Muon under the default rule."""

    resolved_module_name = _resolve_module_name(name, module_name)
    resolved_parameter_name = parameter_name or _parameter_name_from_name(name)

    if resolved_parameter_name != "weight" or parameter.ndim != 2:
        return False
    if _is_embedding_parameter(resolved_module_name, module):
        return False
    if _is_normalization_parameter(resolved_module_name, module):
        return False
    return not _is_output_head_parameter(resolved_module_name, output_head_module_names)


def split_muon_params(
    module: nn.Module,
    *,
    requires_grad_only: bool = True,
    is_muon_parameter: MuonParameterPredicate | None = None,
    output_head_module_names: Collection[str] = DEFAULT_OUTPUT_HEAD_MODULE_NAMES,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Split module parameters into Muon and non-Muon groups.

    By default, Muon receives trainable 2D weight matrices, except parameters
    owned by embedding modules, normalization modules, or common output-head
    module names. All other selected parameters are returned in the second
    list.
    """

    if not isinstance(module, nn.Module):
        raise TypeError("`module` must be an instance of torch.nn.Module.")

    muon_parameters: list[nn.Parameter] = []
    non_muon_parameters: list[nn.Parameter] = []

    for parameter_contexts in _iter_parameter_context_groups(module):
        parameter = parameter_contexts[0].parameter

        if requires_grad_only and not parameter.requires_grad:
            continue

        if _selects_muon_parameter(
            parameter_contexts,
            is_muon_parameter=is_muon_parameter,
            output_head_module_names=output_head_module_names,
        ):
            muon_parameters.append(parameter)
        else:
            non_muon_parameters.append(parameter)

    return muon_parameters, non_muon_parameters


def _selects_muon_parameter(
    parameter_contexts: list[_ParameterContext],
    *,
    is_muon_parameter: MuonParameterPredicate | None,
    output_head_module_names: Collection[str],
) -> bool:
    if is_muon_parameter is not None:
        return any(_context_matches(context, is_muon_parameter) for context in parameter_contexts)
    return _is_default_muon_parameter_group(
        parameter_contexts,
        output_head_module_names=output_head_module_names,
    )


def _context_matches(
    context: _ParameterContext,
    is_muon_parameter: MuonParameterPredicate,
) -> bool:
    return is_muon_parameter(context.name, context.parameter)


def _is_default_muon_parameter_group(
    parameter_contexts: list[_ParameterContext],
    *,
    output_head_module_names: Collection[str],
) -> bool:
    if any(
        _is_default_muon_excluded_context(
            context,
            output_head_module_names=output_head_module_names,
        )
        for context in parameter_contexts
    ):
        return False
    return any(
        context.parameter_name == "weight" and context.parameter.ndim == 2
        for context in parameter_contexts
    )


def _is_default_muon_excluded_context(
    context: _ParameterContext,
    *,
    output_head_module_names: Collection[str],
) -> bool:
    return (
        _is_embedding_parameter(context.module_name, context.module)
        or _is_normalization_parameter(context.module_name, context.module)
        or _is_output_head_parameter(context.module_name, output_head_module_names)
    )


def _iter_parameter_context_groups(module: nn.Module) -> Iterator[list[_ParameterContext]]:
    parameter_contexts_by_id: dict[int, list[_ParameterContext]] = {}

    for module_name, child_module in module.named_modules():
        for parameter_name, parameter in child_module.named_parameters(recurse=False):
            parameter_id = id(parameter)
            parameter_contexts = parameter_contexts_by_id.setdefault(parameter_id, [])
            parameter_contexts.append(
                _ParameterContext(
                    name=_join_parameter_name(module_name, parameter_name),
                    parameter=parameter,
                    module_name=module_name,
                    module=child_module,
                    parameter_name=parameter_name,
                )
            )

    yield from parameter_contexts_by_id.values()


def _is_embedding_parameter(module_name: str, module: nn.Module | None) -> bool:
    if module is not None and isinstance(module, _EMBEDDING_MODULE_TYPES):
        return True
    return _has_exact_module_name(module_name, _EMBEDDING_MODULE_NAMES)


def _is_normalization_parameter(module_name: str, module: nn.Module | None) -> bool:
    if module is not None and _is_normalization_module(module):
        return True
    return _has_normalization_module_name(module_name)


def _is_normalization_module(module: nn.Module) -> bool:
    class_name = module.__class__.__name__.lower()
    return isinstance(module, _NORMALIZATION_MODULE_TYPES) or class_name.endswith("norm")


def _is_output_head_parameter(
    module_name: str,
    output_head_module_names: Collection[str],
) -> bool:
    return _has_exact_module_name(module_name, output_head_module_names)


def _has_normalization_module_name(module_name: str) -> bool:
    return any(
        part in _NORMALIZATION_MODULE_NAMES or part.endswith("norm")
        for part in _module_name_parts(module_name)
    )


def _has_exact_module_name(module_name: str, expected_names: Collection[str]) -> bool:
    expected = {name.lower() for name in expected_names}
    return any(part in expected for part in _module_name_parts(module_name))


def _module_name_parts(module_name: str) -> tuple[str, ...]:
    if not module_name:
        return ()
    return tuple(part.lower() for part in module_name.split(".") if part)


def _resolve_module_name(name: str, module_name: str | None) -> str:
    if module_name is not None:
        return module_name
    if "." not in name:
        return ""
    return name.rsplit(".", 1)[0]


def _parameter_name_from_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _join_parameter_name(module_name: str, parameter_name: str) -> str:
    if not module_name:
        return parameter_name
    return f"{module_name}.{parameter_name}"


__all__ = [
    "DEFAULT_OUTPUT_HEAD_MODULE_NAMES",
    "MuonParameterPredicate",
    "is_default_muon_parameter",
    "split_muon_params",
]
