from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import NamedTuple, Protocol

import torch
import torch.nn.functional as F
from torch import nn

from ._ids import id_sequence, validate_id_tensor, validate_non_negative_int, validate_positive_int
from .protocol import EmbeddingProtocol

EmbeddingWeight = Callable[[EmbeddingProtocol], torch.Tensor]


class HeadSpecial(Protocol):
    global_id: int

    def tensor(self, embedding_weight: EmbeddingWeight) -> torch.Tensor: ...


class ParameterHeadSpecial(NamedTuple):
    global_id: int
    param: nn.Parameter

    def tensor(self, embedding_weight: EmbeddingWeight) -> torch.Tensor:
        return self.param


class ModalityHeadSpecial(NamedTuple):
    global_id: int
    local_id: int
    embed: EmbeddingProtocol

    def tensor(self, embedding_weight: EmbeddingWeight) -> torch.Tensor:
        return embedding_weight(self.embed)[self.local_id]


class HeadBlock(NamedTuple):
    global_start: int
    local_start: int
    size: int
    embed: EmbeddingProtocol


class IdSpaceHead(nn.Module):
    def __init__(
        self,
        dim: int,
        specials: Sequence[HeadSpecial],
        blocks: Sequence[HeadBlock],
        embedding_weight: EmbeddingWeight,
    ) -> None:
        super().__init__()
        validate_positive_int(dim, name="dim")
        if not specials and not blocks:
            raise ValueError("head must contain at least one weight.")
        self._dim: int = dim
        self._specials: tuple[HeadSpecial, ...] = tuple(specials)
        self._blocks: tuple[HeadBlock, ...] = tuple(blocks)
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
        for head_id, special in enumerate(self._specials):
            weight = special.tensor(self._embedding_weight)
            logits[..., head_id] = F.linear(x, weight.unsqueeze(0)).squeeze(-1)
        head_start = len(self._specials)
        for block in self._blocks:
            head_end = head_start + block.size
            logits[..., head_start:head_end] = F.linear(
                x,
                self._embedding_weight(block.embed)[block.local_start : block.local_start + block.size],
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
        for head_id, special in enumerate(self._specials):
            mask = ids == special.global_id
            if not bool(mask.any()):
                continue
            head_ids[mask] = head_id
            covered |= mask
        head_start = len(self._specials)
        for block in self._blocks:
            mask = (ids >= block.global_start) & (ids < block.global_start + block.size)
            if not bool(mask.any()):
                head_start += block.size
                continue
            head_ids[mask] = head_start + ids[mask] - block.global_start
            covered |= mask
            head_start += block.size
        if not bool(covered.all()):
            bad_id = int(ids[~covered].reshape(-1)[0].detach().cpu())
            raise ValueError(f"ids contains token outside head: {bad_id}.")
        return head_ids

    def _to_global_tensor(self, ids: torch.Tensor) -> torch.Tensor:
        validate_id_tensor(ids, name="ids")
        global_ids = torch.empty_like(ids)
        covered = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
        for head_id, special in enumerate(self._specials):
            mask = ids == head_id
            if not bool(mask.any()):
                continue
            global_ids[mask] = special.global_id
            covered |= mask
        head_start = len(self._specials)
        for block in self._blocks:
            head_end = head_start + block.size
            mask = (ids >= head_start) & (ids < head_end)
            if not bool(mask.any()):
                head_start = head_end
                continue
            global_ids[mask] = block.global_start + ids[mask] - head_start
            covered |= mask
            head_start = head_end
        if not bool(covered.all()):
            bad_id = int(ids[~covered].reshape(-1)[0].detach().cpu())
            raise ValueError(f"ids contains head id outside head: {bad_id}.")
        return global_ids


def _global_ids(
    specials: Sequence[HeadSpecial],
    blocks: Sequence[HeadBlock],
) -> tuple[int, ...]:
    global_ids = [special.global_id for special in specials]
    for block in blocks:
        global_ids.extend(range(block.global_start, block.global_start + block.size))
    return tuple(global_ids)
