from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from typing import cast

import torch
from torch import nn

from .protocol import EmbeddingProtocol
from .space import IdSpace, Modality


def infer_dim(
    dim: int | None,
    special_embeddings: nn.ParameterDict | None,
    modality_embeddings: Mapping[Modality, EmbeddingProtocol] | None,
    adapters: Mapping[Modality, nn.Module] | None = None,
) -> int:
    if dim is not None:
        from ._ids import validate_positive_int

        validate_positive_int(dim, name="dim")
        return dim

    adapted = set(adapters or ())
    if special_embeddings is not None:
        for param in special_embeddings.values():
            if isinstance(param, nn.Parameter):
                return param.size(0)
    if modality_embeddings is not None:
        for modality, embed in modality_embeddings.items():
            if modality in adapted:
                continue
            if isinstance(embed, nn.Module):
                return embed.embedding_dim
    if adapted:
        raise ValueError("dim must be provided when all explicit modality embeddings use adapters.")
    raise ValueError("dim must be provided when no explicit embeddings are given.")


def embedding_weight(embed: EmbeddingProtocol) -> torch.Tensor:
    try:
        weight = embed.weight
    except AttributeError as error:
        raise TypeError("embedding module must expose a tensor weight.") from error
    if not isinstance(weight, torch.Tensor):
        raise TypeError("embedding module must expose a tensor weight.")
    return weight


def init_special_embeddings(
    space: IdSpace,
    dim: int,
    special_embeddings: nn.ParameterDict | None,
) -> nn.ParameterDict:
    if special_embeddings is None:
        special_embeddings = nn.ParameterDict()

    if not isinstance(special_embeddings, nn.ParameterDict):
        raise TypeError("special_embeddings must be an nn.ParameterDict.")
    unknown = set(special_embeddings) - set(space.special_token_ids)
    if unknown:
        raise ValueError("special_embeddings keys must be id space special token names.")
    for name, param in special_embeddings.items():
        if not isinstance(param, nn.Parameter):
            raise TypeError(f"special_embeddings[{name!r}] must be an nn.Parameter.")
        if param.dim() != 1:
            raise ValueError(f"special_embeddings[{name!r}] must be a 1D parameter.")
        if param.size(0) != dim:
            raise ValueError(f"special_embeddings[{name!r}] dimension must match dim.")

    missing_names = [
        name
        for name, token_id in space.special_token_ids.items()
        if name not in special_embeddings and space.block_containing_id(token_id) is None
    ]
    if missing_names:
        warn_default_initialization("special embeddings", missing_names)
    for name in missing_names:
        param = nn.Parameter(torch.empty(dim))
        nn.init.normal_(param)
        special_embeddings[name] = param
    return special_embeddings


def init_modality_embeddings(
    space: IdSpace,
    dim: int,
    modality_embeddings: Mapping[Modality, EmbeddingProtocol] | None,
    *,
    adapted_modalities: set[Modality] | None = None,
) -> nn.ModuleDict:
    if modality_embeddings is None:
        modality_embeddings = {}
    elif not isinstance(modality_embeddings, Mapping):
        raise TypeError("modality_embeddings must be a mapping of Modality to embedding modules.")

    expected = {modality_block.modality for modality_block in space.modality_blocks}
    adapted = adapted_modalities or set()
    for modality in modality_embeddings:
        if not isinstance(modality, Modality):
            raise TypeError("modality_embeddings keys must be Modality values.")
        if modality not in expected:
            raise ValueError("modality_embeddings keys must be id space modalities.")

    modules = nn.ModuleDict()
    for modality_block in space.modality_blocks:
        modality = modality_block.modality
        embed = modality_embeddings.get(modality)
        if embed is None:
            if modality in adapted:
                raise ValueError(
                    f"adapters[{modality!r}] requires an explicit modality embedding."
                )
            warn_default_initialization(
                "modality embeddings",
                [modality.value],
            )
            embed = nn.Embedding(modality_block.vocab_size, dim)

        if not isinstance(embed, nn.Module):
            raise TypeError(f"modality_embeddings[{modality!r}] must be an nn.Module.")
        if embed.num_embeddings != modality_block.vocab_size:
            raise ValueError(
                f"modality_embeddings[{modality!r}].num_embeddings must match id space."
            )
        if modality not in adapted and embed.embedding_dim != dim:
            raise ValueError(
                f"modality_embeddings[{modality!r}].embedding_dim must match dim "
                "when no adapter is provided."
            )
        modules[modality.value] = cast(nn.Module, embed)
    return modules


def init_adapters(
    space: IdSpace,
    dim: int,
    modality_embeddings: nn.ModuleDict,
    adapters: Mapping[Modality, nn.Module] | None,
) -> nn.ModuleDict:
    if adapters is None:
        return nn.ModuleDict()
    if not isinstance(adapters, Mapping):
        raise TypeError("adapters must be a mapping of Modality to modules.")

    expected = {modality_block.modality for modality_block in space.modality_blocks}
    modules = nn.ModuleDict()
    for modality, adapter in adapters.items():
        if not isinstance(modality, Modality):
            raise TypeError("adapters keys must be Modality values.")
        if modality not in expected:
            raise ValueError("adapters keys must be id space modalities.")
        if not isinstance(adapter, nn.Module):
            raise TypeError(f"adapters[{modality!r}] must be an nn.Module.")

        embed = cast(EmbeddingProtocol, modality_embeddings[modality.value])
        weight = embedding_weight(embed)
        probe = torch.zeros(1, embed.embedding_dim, device=weight.device, dtype=weight.dtype)
        try:
            projected = adapter(probe)
        except Exception as error:
            raise ValueError(
                f"adapters[{modality!r}] failed on embedding_dim={embed.embedding_dim}."
            ) from error
        if not isinstance(projected, torch.Tensor):
            raise TypeError(f"adapters[{modality!r}] must return a tensor.")
        if projected.shape[-1] != dim:
            raise ValueError(
                f"adapters[{modality!r}] output dim must match dim={dim}, "
                f"got {projected.shape[-1]}."
            )
        modules[modality.value] = adapter
    return modules


def warn_default_initialization(kind: str, names: Sequence[str]) -> None:
    joined = ", ".join(names)
    warnings.warn(
        f"IdSpaceEmbedding is using PyTorch default initialization for {kind}: {joined}. "
        "This is often a poor production default, especially when adding tokens to "
        "a pretrained model or tying embeddings as an output head; pass explicit "
        "embeddings when initialization scale matters.",
        UserWarning,
        stacklevel=3,
    )
