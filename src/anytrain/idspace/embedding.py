from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .layout import Modality, ModalityRange, TokenLayout

type _HeadSpecial = tuple[int, nn.Parameter]
type _HeadBlock = tuple[int, int, int, nn.Embedding]


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        layout: TokenLayout,
        dim: int | None,
        *,
        special_embeddings: nn.ParameterDict | None = None,
        modality_embeddings: Mapping[Modality, nn.Embedding] | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(layout, TokenLayout):
            raise TypeError("layout must be a TokenLayout.")
        if not layout.special_token_ids and not layout.modality_ranges:
            raise ValueError("layout must contain at least one special token or modality.")
        dim = _resolve_dim(dim, special_embeddings, modality_embeddings)
        device, dtype = _resolve_device_dtype(
            special_embeddings,
            modality_embeddings,
            device=device,
            dtype=dtype,
        )

        self.layout = layout
        self.special_embeddings = _init_special_embeddings(
            layout,
            dim,
            special_embeddings,
            device=device,
            dtype=dtype,
        )
        self.modality_embeddings = _init_modality_embeddings(
            layout,
            dim,
            modality_embeddings,
            device=device,
            dtype=dtype,
        )
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def vocab_size(self) -> int:
        return self.layout.vocab_size

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        _validate_ids(input_ids, name="input_ids")
        device, dtype = self._weight_device_dtype()
        if input_ids.device != device:
            raise ValueError("input_ids device must match embedding weights.")

        output = torch.empty((*input_ids.shape, self.dim), dtype=dtype, device=device)
        covered = torch.zeros(input_ids.shape, dtype=torch.bool, device=device)

        for name, global_id in self.layout.special_token_ids.items():
            mask = input_ids == global_id
            if not bool(mask.any()):
                continue
            output[mask] = self.special_embeddings[name]
            covered |= mask

        for modality_range in self.layout.modality_ranges:
            mask = (input_ids >= modality_range.start) & (input_ids < modality_range.end) & ~covered
            if not bool(mask.any()):
                continue
            ids = input_ids[mask] - modality_range.start
            output[mask] = self._modality_embedding(modality_range.modality)(ids)
            covered |= mask

        if not bool(covered.all()):
            bad_id = int(input_ids[~covered].reshape(-1)[0].detach().cpu())
            raise ValueError(f"input_ids contains id outside layout: {bad_id}.")
        return output

    def dense_weight(self) -> torch.Tensor:
        device, dtype = self._weight_device_dtype()
        weight = torch.zeros(self.vocab_size, self.dim, device=device, dtype=dtype)
        for modality_range in self.layout.modality_ranges:
            embed = self._modality_embedding(modality_range.modality)
            weight[modality_range.start : modality_range.end] = embed.weight
        for name, global_id in self.layout.special_token_ids.items():
            weight[global_id] = self.special_embeddings[name]
        return weight

    def as_head(
        self,
        *,
        special_tokens: bool | Sequence[str] = True,
        modalities: Sequence[Modality] | None = None,
    ) -> _TokenHead:
        specials = tuple(
            (self.layout.special_token_id(name), self.special_embeddings[name])
            for name in _normalize_head_special_tokens(self.layout, special_tokens)
        )

        blocks: list[_HeadBlock] = []
        for modality in _normalize_head_modalities(self.layout, modalities):
            modality_range = self.layout.modality_range(modality)
            embed = self._modality_embedding(modality)
            for global_start, size in _regular_blocks(self.layout, modality_range):
                local_start = global_start - modality_range.start
                blocks.append((global_start, local_start, size, embed))

        return _TokenHead(
            self.dim,
            specials,
            blocks,
        )

    def _modality_embedding(self, modality: Modality) -> nn.Embedding:
        embed = self.modality_embeddings[modality.value]
        if not isinstance(embed, nn.Embedding):
            raise TypeError(f"modality_embeddings[{modality!r}] must be an nn.Embedding.")
        return embed

    def _weight_device_dtype(self) -> tuple[torch.device, torch.dtype]:
        if self.special_embeddings:
            param = next(iter(self.special_embeddings.values()))
            return param.device, param.dtype
        modality = self.layout.modality_ranges[0].modality
        embed = self._modality_embedding(modality)
        return embed.weight.device, embed.weight.dtype


class _TokenHead(nn.Module):
    def __init__(
        self,
        dim: int,
        specials: Sequence[_HeadSpecial],
        blocks: Sequence[_HeadBlock],
    ) -> None:
        super().__init__()
        _validate_positive_int(dim, name="dim")
        if not specials and not blocks:
            raise ValueError("head must contain at least one weight.")
        self._dim: int = dim
        self._specials: tuple[_HeadSpecial, ...] = tuple(specials)
        self._blocks: tuple[_HeadBlock, ...] = tuple(blocks)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def vocab_size(self) -> int:
        return len(self.global_ids)

    @property
    def global_ids(self) -> tuple[int, ...]:
        global_ids = [global_id for global_id, _ in self._specials]
        for global_start, _, size, _ in self._blocks:
            global_ids.extend(range(global_start, global_start + size))
        return tuple(global_ids)

    def to_head_ids(self, ids: Sequence[int] | torch.Tensor) -> list[int] | torch.Tensor:
        if isinstance(ids, torch.Tensor):
            return self._to_head_tensor(ids)
        return [self._to_head_id(token_id) for token_id in _normalize_id_sequence(ids, name="ids")]

    def to_global_ids(self, ids: Sequence[int] | torch.Tensor) -> list[int] | torch.Tensor:
        if isinstance(ids, torch.Tensor):
            return self._to_global_tensor(ids)
        return [self._to_global_id(head_id) for head_id in _normalize_id_sequence(ids, name="ids")]

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
                embed.weight[local_start : local_start + size],
            )
            head_start = head_end
        return logits

    def _to_head_id(self, token_id: int) -> int:
        _validate_non_negative_int(token_id, name="global_id")
        for head_id, (global_id, _) in enumerate(self._specials):
            if global_id == token_id:
                return head_id
        head_start = len(self._specials)
        for global_start, _, size, _ in self._blocks:
            if global_start <= token_id < global_start + size:
                return head_start + token_id - global_start
            head_start += size
        raise ValueError(f"global_id is outside head: {token_id}.")

    def _to_global_id(self, head_id: int) -> int:
        _validate_non_negative_int(head_id, name="head_id")
        if head_id >= self.vocab_size:
            raise ValueError(f"head_id is outside head: {head_id}.")
        return self.global_ids[head_id]

    def _to_head_tensor(self, ids: torch.Tensor) -> torch.Tensor:
        _validate_ids(ids, name="ids")
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
        _validate_ids(ids, name="ids")
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
    modality_embeddings: Mapping[Modality, nn.Embedding] | None,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> tuple[torch.device | str | None, torch.dtype | None]:
    if special_embeddings is not None and not isinstance(special_embeddings, nn.ParameterDict):
        raise TypeError("special_embeddings must be an nn.ParameterDict.")
    if modality_embeddings is not None and not isinstance(modality_embeddings, Mapping):
        raise TypeError("modality_embeddings must be a mapping of Modality to nn.Embedding.")

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
    modality_embeddings: Mapping[Modality, nn.Embedding] | None,
) -> int:
    if dim is not None:
        _validate_positive_int(dim, name="dim")
        return dim

    inferred = _first_explicit_dim(special_embeddings, modality_embeddings)
    if inferred is None:
        raise ValueError("dim must be provided when no explicit embeddings are given.")
    return inferred


def _first_explicit_dim(
    special_embeddings: nn.ParameterDict | None,
    modality_embeddings: Mapping[Modality, nn.Embedding] | None,
) -> int | None:
    if special_embeddings is not None:
        for param in special_embeddings.values():
            if isinstance(param, nn.Parameter):
                return param.size(0)
    if modality_embeddings is not None:
        for embed in modality_embeddings.values():
            if isinstance(embed, nn.Embedding):
                return embed.embedding_dim
    return None


def _first_explicit_weight(
    special_embeddings: nn.ParameterDict | None,
    modality_embeddings: Mapping[Modality, nn.Embedding] | None,
) -> torch.Tensor | None:
    if special_embeddings is not None:
        for param in special_embeddings.values():
            if isinstance(param, nn.Parameter):
                return param
    if modality_embeddings is not None:
        for embed in modality_embeddings.values():
            if isinstance(embed, nn.Embedding):
                return embed.weight
    return None


def _init_special_embeddings(
    layout: TokenLayout,
    dim: int,
    special_embeddings: nn.ParameterDict | None,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> nn.ParameterDict:
    if special_embeddings is None:
        special_embeddings = nn.ParameterDict()

    if not isinstance(special_embeddings, nn.ParameterDict):
        raise TypeError("special_embeddings must be an nn.ParameterDict.")
    unknown = set(special_embeddings) - set(layout.special_token_ids)
    if unknown:
        raise ValueError("special_embeddings keys must be layout special token names.")
    for name, param in special_embeddings.items():
        if not isinstance(param, nn.Parameter):
            raise TypeError(f"special_embeddings[{name!r}] must be an nn.Parameter.")
        if param.dim() != 1:
            raise ValueError(f"special_embeddings[{name!r}] must be a 1D parameter.")
        if param.size(0) != dim:
            raise ValueError(f"special_embeddings[{name!r}] dimension must match dim.")
    for name in layout.special_token_ids:
        if name in special_embeddings:
            continue
        param = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        nn.init.normal_(param)
        special_embeddings[name] = param
    return special_embeddings


def _init_modality_embeddings(
    layout: TokenLayout,
    dim: int,
    modality_embeddings: Mapping[Modality, nn.Embedding] | None,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype | None,
) -> nn.ModuleDict:
    if modality_embeddings is None:
        modality_embeddings = {}
    elif not isinstance(modality_embeddings, Mapping):
        raise TypeError("modality_embeddings must be a mapping of Modality to nn.Embedding.")

    expected = {modality_range.modality for modality_range in layout.modality_ranges}
    for modality in modality_embeddings:
        if not isinstance(modality, Modality):
            raise TypeError("modality_embeddings keys must be Modality values.")
        if modality not in expected:
            raise ValueError("modality_embeddings keys must be layout modalities.")

    modules = nn.ModuleDict()
    for modality_range in layout.modality_ranges:
        embed = modality_embeddings.get(modality_range.modality)
        if embed is None:
            embed = nn.Embedding(
                modality_range.vocab_size,
                dim,
                device=device,
                dtype=dtype,
            )
        if not isinstance(embed, nn.Embedding):
            raise TypeError(
                f"modality_embeddings[{modality_range.modality!r}] must be an nn.Embedding."
            )
        if embed.num_embeddings != modality_range.vocab_size:
            raise ValueError(
                f"modality_embeddings[{modality_range.modality!r}].num_embeddings must match layout."
            )
        if embed.embedding_dim != dim:
            raise ValueError(
                f"modality_embeddings[{modality_range.modality!r}].embedding_dim must match dim."
            )
        modules[modality_range.modality.value] = embed
    return modules


def _normalize_head_special_tokens(
    layout: TokenLayout,
    special_tokens: bool | Sequence[str],
) -> tuple[str, ...]:
    if isinstance(special_tokens, bool):
        if not special_tokens:
            return ()
        return tuple(name for name, _ in sorted(layout.special_token_ids.items(), key=lambda item: item[1]))
    if not isinstance(special_tokens, Sequence) or isinstance(special_tokens, str | bytes):
        raise TypeError("special_tokens must be a bool or a sequence of special token names.")

    normalized: list[str] = []
    seen: set[str] = set()
    for index, name in enumerate(special_tokens):
        if not isinstance(name, str):
            raise TypeError(f"special_tokens[{index}] must be a string.")
        if name not in layout.special_token_ids:
            raise KeyError(f"unknown special token {name!r}.")
        if name in seen:
            raise ValueError("special token names must be unique.")
        seen.add(name)
        normalized.append(name)
    return tuple(normalized)


def _normalize_head_modalities(
    layout: TokenLayout,
    modalities: Sequence[Modality] | None,
) -> tuple[Modality, ...]:
    if modalities is None:
        return tuple(modality_range.modality for modality_range in layout.modality_ranges)
    if not isinstance(modalities, Sequence) or isinstance(modalities, str | bytes):
        raise TypeError("modalities must be a sequence of Modality values.")

    normalized: list[Modality] = []
    seen: set[Modality] = set()
    for index, modality in enumerate(modalities):
        if not isinstance(modality, Modality):
            raise TypeError(f"modalities[{index}] must be a Modality.")
        layout.modality_range(modality)
        if modality in seen:
            raise ValueError("modalities must be unique.")
        seen.add(modality)
        normalized.append(modality)
    return tuple(normalized)


def _regular_blocks(
    layout: TokenLayout,
    modality_range: ModalityRange,
) -> tuple[tuple[int, int], ...]:
    blocks: list[tuple[int, int]] = []
    cursor = modality_range.start
    for special_id in layout.all_special_ids:
        if special_id < cursor:
            continue
        if special_id >= modality_range.end:
            break
        if cursor < special_id:
            blocks.append((cursor, special_id - cursor))
        cursor = special_id + 1
    if cursor < modality_range.end:
        blocks.append((cursor, modality_range.end - cursor))
    return tuple(blocks)


def _normalize_id_sequence(ids: Sequence[int], *, name: str) -> list[int]:
    if not isinstance(ids, Sequence) or isinstance(ids, str | bytes):
        raise TypeError(f"{name} must be a sequence of integer ids.")
    normalized: list[int] = []
    for index, token_id in enumerate(ids):
        _validate_non_negative_int(token_id, name=f"{name}[{index}]")
        normalized.append(token_id)
    return normalized


def _validate_ids(ids: torch.Tensor, *, name: str) -> None:
    if not isinstance(ids, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if ids.dtype == torch.bool or torch.is_floating_point(ids) or torch.is_complex(ids):
        raise TypeError(f"{name} must contain integer ids.")


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_non_negative_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


__all__ = [
    "TokenEmbedding",
]
