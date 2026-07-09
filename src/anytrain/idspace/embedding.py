from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import cast

import torch
from torch import nn

from ._embedding_init import (
    embedding_weight,
    infer_dim,
    init_adapters,
    init_modality_embeddings,
    init_special_embeddings,
)
from ._head import (
    HeadBlock,
    HeadSpecial,
    IdSpaceHead,
    ModalityHeadSpecial,
    ParameterHeadSpecial,
)
from ._ids import validate_id_tensor
from .protocol import EmbeddingProtocol
from .space import IdSpace, Modality

EmbeddingWeight = Callable[[EmbeddingProtocol], torch.Tensor]


class IdSpaceEmbedding(nn.Module):
    def __init__(
        self,
        space: IdSpace,
        dim: int | None = None,
        *,
        special_embeddings: nn.ParameterDict | None = None,
        modality_embeddings: Mapping[Modality, EmbeddingProtocol] | None = None,
        adapters: Mapping[Modality, nn.Module] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(space, IdSpace):
            raise TypeError("space must be an IdSpace.")
        if not space.special_token_ids and not space.modality_blocks:
            raise ValueError("space must contain at least one special token or modality.")
        dim = infer_dim(dim, special_embeddings, modality_embeddings, adapters)
        adapted = set(adapters or ())

        self.space = space
        self.special_embeddings = init_special_embeddings(space, dim, special_embeddings)
        self.modality_embeddings = init_modality_embeddings(
            space,
            dim,
            modality_embeddings,
            adapted_modalities=adapted,
        )
        self.adapters = init_adapters(space, dim, self.modality_embeddings, adapters)
        self._dim = dim
        self._weight_cache_depth = 0
        self._weight_cache: dict[int, torch.Tensor] = {}

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def embedding_dim(self) -> int:
        return self._dim

    @property
    def num_embeddings(self) -> int:
        return self.space.vocab_size

    @property
    def vocab_size(self) -> int:
        return self.num_embeddings

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        validate_id_tensor(input_ids, name="input_ids")
        device, dtype = self._weight_device_dtype()
        if input_ids.device != device:
            raise ValueError("input_ids device must match embedding weights.")

        output = torch.empty((*input_ids.shape, self.dim), dtype=dtype, device=device)
        covered = torch.zeros(input_ids.shape, dtype=torch.bool, device=device)

        for name, global_id in self.space.special_token_ids.items():
            if name not in self.special_embeddings:
                continue
            mask = input_ids == global_id
            if not bool(mask.any()):
                continue
            output[mask] = self.special_embeddings[name]
            covered |= mask

        for modality_block in self.space.modality_blocks:
            mask = (input_ids >= modality_block.start) & (input_ids < modality_block.end) & ~covered
            if not bool(mask.any()):
                continue
            ids = input_ids[mask] - modality_block.start
            vectors = self._modality_embedding(modality_block.modality)(ids)
            key = modality_block.modality.value
            if key in self.adapters:
                vectors = self.adapters[key](vectors)
            output[mask] = vectors
            covered |= mask

        if not bool(covered.all()):
            bad_id = int(input_ids[~covered].reshape(-1)[0].detach().cpu())
            raise ValueError(f"input_ids contains id outside space: {bad_id}.")
        return output

    @property
    def weight(self) -> torch.Tensor:
        for modality_block in self.space.modality_blocks:
            embed = self._modality_embedding(modality_block.modality)
            if embed.embedding_dim != self.dim:
                raise ValueError(
                    "weight requires all modality embedding dims to match dim; "
                    "use modality_embeddings[*].weight for native tables when adapters "
                    "or variable-dim embeddings are present."
                )
        device, dtype = self._weight_device_dtype()
        weight = torch.zeros(self.num_embeddings, self.embedding_dim, device=device, dtype=dtype)
        for modality_block in self.space.modality_blocks:
            embed = self._modality_embedding(modality_block.modality)
            weight[modality_block.start : modality_block.end] = self._embedding_weight(embed)
        for name, global_id in self.space.special_token_ids.items():
            if name not in self.special_embeddings:
                continue
            weight[global_id] = self.special_embeddings[name]
        return weight

    @contextmanager
    def cache_weights(self) -> Iterator[None]:
        self._weight_cache_depth += 1
        try:
            yield
        finally:
            self._weight_cache_depth -= 1
            if self._weight_cache_depth == 0:
                self._weight_cache.clear()

    def head_view(
        self,
        *,
        special_tokens: bool | Sequence[str] = True,
        modalities: Sequence[Modality] | None = None,
    ) -> IdSpaceHead:
        special_names = normalize_head_special_tokens(self.space, special_tokens)
        specials: tuple[HeadSpecial, ...] = tuple(
            self._head_special(name) for name in special_names
        )

        blocks: list[HeadBlock] = []
        for modality in normalize_head_modalities(self.space, modalities):
            modality_block = self.space.modality_block(modality)
            embed = self._modality_embedding(modality)
            for global_start, size in self.space.regular_blocks(modality):
                local_start = global_start - modality_block.start
                blocks.append(HeadBlock(global_start, local_start, size, embed))

        return IdSpaceHead(
            resolve_head_dim(specials, blocks, self._embedding_weight),
            specials,
            blocks,
            self._embedding_weight,
        )

    def _head_special(self, name: str) -> HeadSpecial:
        global_id = self.space.special_token_id(name)
        if name in self.special_embeddings:
            return ParameterHeadSpecial(global_id, self.special_embeddings[name])
        modality_block = self.space.block_containing_id(global_id)
        if modality_block is None:
            raise ValueError(f"special token {name!r} has no embedding.")
        local_id = global_id - modality_block.start
        return ModalityHeadSpecial(
            global_id,
            local_id,
            self._modality_embedding(modality_block.modality),
        )

    def _modality_embedding(self, modality: Modality) -> EmbeddingProtocol:
        return cast(EmbeddingProtocol, self.modality_embeddings[modality.value])

    def _weight_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        if self.special_embeddings:
            param = next(iter(self.special_embeddings.values()))
            return param.device, param.dtype
        modality = self.space.modality_blocks[0].modality
        embed = self._modality_embedding(modality)
        weight = self._embedding_weight(embed)
        return weight.device, weight.dtype

    def _embedding_weight(self, embed: EmbeddingProtocol) -> torch.Tensor:
        if self._weight_cache_depth <= 0:
            return embedding_weight(embed)
        key = id(embed)
        weight = self._weight_cache.get(key)
        if weight is None:
            weight = embedding_weight(embed)
            self._weight_cache[key] = weight
        return weight


def resolve_head_dim(
    specials: Sequence[HeadSpecial],
    blocks: Sequence[HeadBlock],
    embedding_weight_fn: EmbeddingWeight,
) -> int:
    dims: list[int] = []
    for special in specials:
        dims.append(int(special.tensor(embedding_weight_fn).shape[-1]))
    for block in blocks:
        dims.append(int(embedding_weight_fn(block.embed).shape[-1]))
    if not dims:
        raise ValueError("head must contain at least one weight.")
    head_dim = dims[0]
    if any(dim != head_dim for dim in dims):
        raise ValueError(
            "head_view requires selected specials and modality blocks to share the same "
            "native embedding dim; split heads by modality or project hidden yourself."
        )
    return head_dim


def normalize_head_special_tokens(
    space: IdSpace,
    special_tokens: bool | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(special_tokens, bool):
        if not special_tokens:
            return ()
        return tuple(name for name, _ in sorted(space.special_token_ids.items(), key=lambda item: item[1]))
    if not isinstance(special_tokens, Sequence) or isinstance(special_tokens, str | bytes):
        raise TypeError("special_tokens must be a bool or a sequence of special token names.")

    normalized: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(special_tokens):
        if not isinstance(name, str):
            raise TypeError(f"special_tokens[{index}] must be a string.")
        if name not in space.special_token_ids:
            raise KeyError(f"unknown special token {name!r}.")
        if name in seen:
            raise ValueError("special token names must be unique.")
        seen.add(name)
        normalized.append(name)
    return tuple(normalized)


def normalize_head_modalities(
    space: IdSpace,
    modalities: Sequence[Modality] | None,
) -> tuple[Modality, ...]:
    if modalities is None:
        return tuple(modality_block.modality for modality_block in space.modality_blocks)
    if not isinstance(modalities, Sequence) or isinstance(modalities, str | bytes):
        raise TypeError("modalities must be a sequence of Modality values.")

    normalized: list[Modality] = []
    seen: set[Modality] = set()
    for index, modality in enumerate(modalities):
        if not isinstance(modality, Modality):
            raise TypeError(f"modalities[{index}] must be a Modality.")
        space.modality_block(modality)
        if modality in seen:
            raise ValueError("modalities must be unique.")
        seen.add(modality)
        normalized.append(modality)
    return tuple(normalized)


__all__ = [
    "IdSpaceEmbedding",
]
