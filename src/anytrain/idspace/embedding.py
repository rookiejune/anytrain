from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager

import torch
import torch.nn.functional as F
from torch import nn

from ._ids import id_sequence, validate_id_tensor, validate_non_negative_int, validate_positive_int
from .protocol import EmbeddingProtocol
from .space import IdSpace, Modality

_HeadSpecial = tuple[int, nn.Parameter]
_HeadBlock = tuple[int, int, int, EmbeddingProtocol]


class IdSpaceEmbedding(nn.Module):
    def __init__(
        self,
        space: IdSpace,
        dim: int | None = None,
        *,
        special_embeddings: nn.ParameterDict | None = None,
        modality_embeddings: Mapping[Modality, EmbeddingProtocol] | None = None,
        init_missing_special_embeddings: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(space, IdSpace):
            raise TypeError("space must be an IdSpace.")
        if not space.special_token_ids and not space.modality_blocks:
            raise ValueError("space must contain at least one special token or modality.")
        dim = _resolve_dim(dim, special_embeddings, modality_embeddings)
        device, dtype = _resolve_device_dtype(
            special_embeddings,
            modality_embeddings,
            device=device,
            dtype=dtype,
        )

        self.space = space
        self.special_embeddings = _init_special_embeddings(
            space,
            dim,
            special_embeddings,
            init_missing=init_missing_special_embeddings,
            device=device,
            dtype=dtype,
        )
        self.modality_embeddings = _init_modality_embeddings(
            space,
            dim,
            modality_embeddings,
            device=device,
            dtype=dtype,
        )
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
            output[mask] = self._modality_embedding(modality_block.modality)(ids)
            covered |= mask

        if not bool(covered.all()):
            bad_id = int(input_ids[~covered].reshape(-1)[0].detach().cpu())
            raise ValueError(f"input_ids contains id outside space: {bad_id}.")
        return output

    @property
    def weight(self) -> torch.Tensor:
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
    ) -> _IdSpaceHead:
        special_names = _normalize_head_special_tokens(self.space, special_tokens)
        missing = [name for name in special_names if name not in self.special_embeddings]
        if missing:
            raise ValueError("head special tokens must have explicit special embeddings.")
        specials = tuple(
            (self.space.special_token_id(name), self.special_embeddings[name])
            for name in special_names
        )

        blocks: list[_HeadBlock] = []
        for modality in _normalize_head_modalities(self.space, modalities):
            modality_block = self.space.modality_block(modality)
            embed = self._modality_embedding(modality)
            for global_start, size in self.space.regular_blocks(modality):
                local_start = global_start - modality_block.start
                blocks.append((global_start, local_start, size, embed))

        return _IdSpaceHead(
            self.dim,
            specials,
            blocks,
            self._embedding_weight,
        )

    def _modality_embedding(self, modality: Modality) -> EmbeddingProtocol:
        embed = self.modality_embeddings[modality.value]
        return embed

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
            return _embedding_weight(embed)
        key = id(embed)
        weight = self._weight_cache.get(key)
        if weight is None:
            weight = _embedding_weight(embed)
            self._weight_cache[key] = weight
        return weight


class _IdSpaceHead(nn.Module):
    def __init__(
        self,
        dim: int,
        specials: Sequence[_HeadSpecial],
        blocks: Sequence[_HeadBlock],
        embedding_weight: Callable[[EmbeddingProtocol], torch.Tensor],
    ) -> None:
        super().__init__()
        validate_positive_int(dim, name="dim")
        if not specials and not blocks:
            raise ValueError("head must contain at least one weight.")
        self._dim: int = dim
        self._specials: tuple[_HeadSpecial, ...] = tuple(specials)
        self._blocks: tuple[_HeadBlock, ...] = tuple(blocks)
        self._embedding_weight = embedding_weight
        self._global_ids = _global_ids(self._specials, self._blocks)
        self._head_id_by_global_id = {
            global_id: head_id for head_id, global_id in enumerate(self._global_ids)
        }

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def vocab_size(self) -> int:
        return len(self._global_ids)

    @property
    def global_ids(self) -> tuple[int, ...]:
        return self._global_ids

    def to_head_ids(self, ids: Sequence[int] | torch.Tensor) -> list[int] | torch.Tensor:
        if isinstance(ids, torch.Tensor):
            return self._to_head_tensor(ids)
        return [self._to_head_id(token_id) for token_id in id_sequence(ids, name="ids")]

    def to_global_ids(self, ids: Sequence[int] | torch.Tensor) -> list[int] | torch.Tensor:
        if isinstance(ids, torch.Tensor):
            return self._to_global_tensor(ids)
        return [self._to_global_id(head_id) for head_id in id_sequence(ids, name="ids")]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 0:
            raise ValueError("x must have at least one dimension.")
        if x.size(-1) != self.dim:
            raise ValueError("x last dimension must match embedding dim.")
        logits = x.new_empty((*x.shape[:-1], self.vocab_size))
        for head_id, (_, weight) in enumerate(self._specials):
            logits[..., head_id] = F.linear(
                x,
                weight.unsqueeze(0),
            ).squeeze(-1)
        head_start = len(self._specials)
        for _, local_start, size, embed in self._blocks:
            head_end = head_start + size
            logits[..., head_start:head_end] = F.linear(
                x,
                self._embedding_weight(embed)[local_start : local_start + size],
            )
            head_start = head_end
        return logits

    def _to_head_id(self, token_id: int) -> int:
        validate_non_negative_int(token_id, name="global_id")
        try:
            return self._head_id_by_global_id[token_id]
        except KeyError as error:
            raise ValueError(f"global_id is outside head: {token_id}.") from error

    def _to_global_id(self, head_id: int) -> int:
        validate_non_negative_int(head_id, name="head_id")
        if head_id >= self.vocab_size:
            raise ValueError(f"head_id is outside head: {head_id}.")
        return self._global_ids[head_id]

    def _to_head_tensor(self, ids: torch.Tensor) -> torch.Tensor:
        validate_id_tensor(ids, name="ids")
        head_ids = torch.empty_like(ids)
        covered = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
        for head_id, (global_id, _) in enumerate(self._specials):
            mask = ids == global_id
            if not bool(mask.any()):
                continue
            head_ids[mask] = head_id
            covered |= mask
        head_start = len(self._specials)
        for global_start, _, size, _ in self._blocks:
            mask = (ids >= global_start) & (ids < global_start + size)
            if not bool(mask.any()):
                head_start += size
                continue
            head_ids[mask] = head_start + ids[mask] - global_start
            covered |= mask
            head_start += size
        if not bool(covered.all()):
            bad_id = int(ids[~covered].reshape(-1)[0].detach().cpu())
            raise ValueError(f"ids contains token outside head: {bad_id}.")
        return head_ids

    def _to_global_tensor(self, ids: torch.Tensor) -> torch.Tensor:
        validate_id_tensor(ids, name="ids")
        global_ids = torch.empty_like(ids)
        covered = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
        for head_id, (global_id, _) in enumerate(self._specials):
            mask = ids == head_id
            if not bool(mask.any()):
                continue
            global_ids[mask] = global_id
            covered |= mask
        head_start = len(self._specials)
        for global_start, _, size, _ in self._blocks:
            head_end = head_start + size
            mask = (ids >= head_start) & (ids < head_end)
            if not bool(mask.any()):
                head_start = head_end
                continue
            global_ids[mask] = global_start + ids[mask] - head_start
            covered |= mask
            head_start = head_end
        if not bool(covered.all()):
            bad_id = int(ids[~covered].reshape(-1)[0].detach().cpu())
            raise ValueError(f"ids contains head id outside head: {bad_id}.")
        return global_ids


def _resolve_device_dtype(
    special_embeddings: nn.ParameterDict | None,
    modality_embeddings: Mapping[Modality, EmbeddingProtocol] | None,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> tuple[torch.device | str | None, torch.dtype | None]:
    if special_embeddings is not None and not isinstance(special_embeddings, nn.ParameterDict):
        raise TypeError("special_embeddings must be an nn.ParameterDict.")
    if modality_embeddings is not None and not isinstance(modality_embeddings, Mapping):
        raise TypeError("modality_embeddings must be a mapping of Modality to embedding modules.")

    if device is not None and dtype is not None:
        return device, dtype

    weight = _first_explicit_weight(special_embeddings, modality_embeddings)
    if weight is not None:
        if device is None:
            device = weight.device
        if dtype is None:
            dtype = weight.dtype
    return device, dtype


def _resolve_dim(
    dim: int | None,
    special_embeddings: nn.ParameterDict | None,
    modality_embeddings: Mapping[Modality, EmbeddingProtocol] | None,
) -> int:
    if dim is not None:
        validate_positive_int(dim, name="dim")
        return dim

    inferred = _first_explicit_dim(special_embeddings, modality_embeddings)
    if inferred is None:
        raise ValueError("dim must be provided when no explicit embeddings are given.")
    return inferred


def _first_explicit_dim(
    special_embeddings: nn.ParameterDict | None,
    modality_embeddings: Mapping[Modality, EmbeddingProtocol] | None,
) -> int | None:
    if special_embeddings is not None:
        for param in special_embeddings.values():
            if isinstance(param, nn.Parameter):
                return param.size(0)
    if modality_embeddings is not None:
        for embed in modality_embeddings.values():
            if isinstance(embed, nn.Module):
                return embed.embedding_dim
    return None


def _first_explicit_weight(
    special_embeddings: nn.ParameterDict | None,
    modality_embeddings: Mapping[Modality, EmbeddingProtocol] | None,
) -> torch.Tensor | None:
    if special_embeddings is not None:
        for param in special_embeddings.values():
            if isinstance(param, nn.Parameter):
                return param
    if modality_embeddings is not None:
        for embed in modality_embeddings.values():
            if isinstance(embed, nn.Module):
                return _embedding_weight(embed)
    return None


def _embedding_weight(embed: EmbeddingProtocol) -> torch.Tensor:
    weight = getattr(embed, "weight", None)
    if not isinstance(weight, torch.Tensor):
        raise TypeError("embedding module must expose a tensor weight.")
    return weight


def _init_special_embeddings(
    space: IdSpace,
    dim: int,
    special_embeddings: nn.ParameterDict | None,
    *,
    init_missing: bool,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
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
    if not init_missing:
        _validate_missing_special_embeddings_are_covered(space, special_embeddings)
        return special_embeddings
    missing_names = [name for name in space.special_token_ids if name not in special_embeddings]
    if missing_names:
        _warn_default_initialization("special embeddings", missing_names)
    for name in space.special_token_ids:
        if name in special_embeddings:
            continue
        param = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        nn.init.normal_(param)
        special_embeddings[name] = param
    return special_embeddings


def _validate_missing_special_embeddings_are_covered(
    space: IdSpace,
    special_embeddings: nn.ParameterDict,
) -> None:
    for name, token_id in space.special_token_ids.items():
        if name in special_embeddings:
            continue
        if not any(block.contains(token_id) for block in space.modality_blocks):
            raise ValueError(
                "special tokens without explicit embeddings must be inside a modality block."
            )


def _init_modality_embeddings(
    space: IdSpace,
    dim: int,
    modality_embeddings: Mapping[Modality, EmbeddingProtocol] | None,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> nn.ModuleDict:
    if modality_embeddings is None:
        modality_embeddings = {}
    elif not isinstance(modality_embeddings, Mapping):
        raise TypeError("modality_embeddings must be a mapping of Modality to embedding modules.")

    expected = {modality_block.modality for modality_block in space.modality_blocks}
    for modality in modality_embeddings:
        if not isinstance(modality, Modality):
            raise TypeError("modality_embeddings keys must be Modality values.")
        if modality not in expected:
            raise ValueError("modality_embeddings keys must be id space modalities.")

    modules = nn.ModuleDict()
    for modality_block in space.modality_blocks:
        embed = modality_embeddings.get(modality_block.modality)
        if embed is None:
            _warn_default_initialization(
                "modality embeddings",
                [modality_block.modality.value],
            )
            embed = nn.Embedding(
                modality_block.vocab_size,
                dim,
                device=device,
                dtype=dtype,
            )

        if embed.num_embeddings != modality_block.vocab_size:
            raise ValueError(
                f"modality_embeddings[{modality_block.modality!r}].num_embeddings must match id space."
            )
        if embed.embedding_dim != dim:
            raise ValueError(
                f"modality_embeddings[{modality_block.modality!r}].embedding_dim must match dim."
            )
        modules[modality_block.modality.value] = embed
    return modules


def _warn_default_initialization(kind: str, names: Sequence[str]) -> None:
    joined = ", ".join(names)
    warnings.warn(
        f"IdSpaceEmbedding is using PyTorch default initialization for {kind}: {joined}. "
        "This is often a poor production default, especially when adding tokens to "
        "a pretrained model or tying embeddings as an output head; pass explicit "
        "embeddings when initialization scale matters.",
        UserWarning,
        stacklevel=3,
    )


def _normalize_head_special_tokens(
    space: IdSpace,
    special_tokens: bool | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(special_tokens, bool):
        if not special_tokens:
            return ()
        return tuple(
            name for name, _ in sorted(space.special_token_ids.items(), key=lambda item: item[1])
        )
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


def _normalize_head_modalities(
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


def _global_ids(
    specials: Sequence[_HeadSpecial],
    blocks: Sequence[_HeadBlock],
) -> tuple[int, ...]:
    global_ids = [global_id for global_id, _ in specials]
    for global_start, _, size, _ in blocks:
        global_ids.extend(range(global_start, global_start + size))
    return tuple(global_ids)


__all__ = [
    "IdSpaceEmbedding",
]
