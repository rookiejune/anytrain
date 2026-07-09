from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import auto
from types import MappingProxyType

import torch

from anytrain._compat import StrEnum

from ._ids import (
    int_sequence,
    validate_id_tensor,
    validate_non_negative_int,
    validate_positive_int,
)


class Modality(StrEnum):
    TEXT = auto()
    AUDIO = auto()


@dataclass(frozen=True)
class ModalityBlock:
    modality: Modality
    start: int
    vocab_size: int

    def __post_init__(self) -> None:
        _validate_modality(self.modality, name="modality")
        validate_non_negative_int(self.start, name="start")
        validate_positive_int(self.vocab_size, name="vocab_size")

    @property
    def end(self) -> int:
        return self.start + self.vocab_size

    def contains(self, token_id: int) -> bool:
        return self.start <= token_id < self.end


class IdSpace:
    __slots__ = (
        "_all_special_ids",
        "_block_by_modality",
        "_special_token_name_by_id",
        "_vocab_size",
        "modality_blocks",
        "special_token_ids",
    )

    def __init__(
        self,
        special_token_ids: Mapping[str, int],
        modality_blocks: Sequence[ModalityBlock],
    ) -> None:
        self.special_token_ids = MappingProxyType(_normalize_special_token_ids(special_token_ids))
        self._special_token_name_by_id = {
            token_id: name for name, token_id in self.special_token_ids.items()
        }
        self._all_special_ids = tuple(sorted(self._special_token_name_by_id))
        self.modality_blocks = _normalize_modality_blocks(modality_blocks)
        self._block_by_modality = {
            modality_block.modality: modality_block for modality_block in self.modality_blocks
        }
        self._vocab_size = _vocab_size(self._all_special_ids, self.modality_blocks)

    @property
    def all_special_ids(self) -> tuple[int, ...]:
        return self._all_special_ids

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def modality_block(self, modality: Modality) -> ModalityBlock:
        _validate_modality(modality, name="modality")
        try:
            return self._block_by_modality[modality]
        except KeyError as error:
            raise KeyError(f"unknown modality {modality!r}.") from error

    def special_token_id(self, name: str) -> int:
        try:
            return self.special_token_ids[name]
        except KeyError as error:
            raise KeyError(f"unknown special token {name!r}.") from error

    def is_special_token_id(self, token_id: int) -> bool:
        return token_id in self._special_token_name_by_id

    def block_containing_id(self, token_id: int) -> ModalityBlock | None:
        validate_non_negative_int(token_id, name="token_id")
        for modality_block in self.modality_blocks:
            if modality_block.contains(token_id):
                return modality_block
        return None

    def modality_block_for_id(self, token_id: int) -> ModalityBlock:
        if self.is_special_token_id(token_id):
            raise ValueError(f"token_id is a special token id: {token_id}.")
        modality_block = self.block_containing_id(token_id)
        if modality_block is not None:
            return modality_block
        raise ValueError(f"token_id is outside all modality blocks: {token_id}.")

    def regular_blocks(self, modality: Modality) -> tuple[tuple[int, int], ...]:
        modality_block = self.modality_block(modality)
        blocks: list[tuple[int, int]] = []
        cursor = modality_block.start
        for special_id in self.all_special_ids:
            if special_id < cursor:
                continue
            if special_id >= modality_block.end:
                break
            if cursor < special_id:
                blocks.append((cursor, special_id - cursor))
            cursor = special_id + 1
        if cursor < modality_block.end:
            blocks.append((cursor, modality_block.end - cursor))
        return tuple(blocks)

    def to_global(self, modality: Modality, ids: Sequence[int] | torch.Tensor) -> list[int] | torch.Tensor:
        modality_block = self.modality_block(modality)
        if isinstance(ids, torch.Tensor):
            return self._to_global_tensor(modality_block, ids)
        local_ids = int_sequence(ids, name="ids")
        global_ids: list[int] = []
        for local_id in local_ids:
            _validate_local_id(local_id, modality_block.vocab_size, name="ids")
            token_id = modality_block.start + local_id
            if self.is_special_token_id(token_id):
                raise ValueError(f"ids contains special token id: {token_id}.")
            global_ids.append(token_id)
        return global_ids

    def to_local(
        self,
        modality: Modality,
        ids: Sequence[int] | torch.Tensor,
        *,
        skip_special: bool = False,
    ) -> list[int] | torch.Tensor:
        modality_block = self.modality_block(modality)
        if isinstance(ids, torch.Tensor):
            return self._to_local_tensor(modality_block, ids, skip_special=skip_special)
        global_ids = int_sequence(ids, name="ids")
        local_ids: list[int] = []
        for token_id in global_ids:
            if self.is_special_token_id(token_id):
                if skip_special:
                    continue
                raise ValueError(f"ids contains special token id: {token_id}.")
            if not modality_block.contains(token_id):
                raise ValueError(f"ids contains token outside modality {modality!r}: {token_id}.")
            local_ids.append(token_id - modality_block.start)
        return local_ids

    def _to_local_tensor(
        self,
        modality_block: ModalityBlock,
        ids: torch.Tensor,
        *,
        skip_special: bool,
    ) -> torch.Tensor:
        validate_id_tensor(ids, name="ids")
        special = torch.zeros(ids.shape, dtype=torch.bool, device=ids.device)
        for special_id in self.all_special_ids:
            if special_id < modality_block.start:
                continue
            if special_id >= modality_block.end:
                break
            special |= ids == special_id
        if bool(special.any()) and not skip_special:
            bad_id = int(ids[special].reshape(-1)[0].detach().cpu())
            raise ValueError(f"ids contains special token id: {bad_id}.")

        regular = (
            (ids >= modality_block.start)
            & (ids < modality_block.end)
            & ~special
        )
        if not bool((regular | special).all()):
            bad_id = int(ids[~(regular | special)].reshape(-1)[0].detach().cpu())
            raise ValueError(
                f"ids contains token outside modality {modality_block.modality!r}: {bad_id}."
            )
        if skip_special:
            return ids[regular] - modality_block.start
        return ids - modality_block.start

    def _to_global_tensor(self, modality_block: ModalityBlock, ids: torch.Tensor) -> torch.Tensor:
        validate_id_tensor(ids, name="ids")
        out_of_range = (ids < 0) | (ids >= modality_block.vocab_size)
        if bool(out_of_range.any()):
            raise ValueError("ids must be in [0, vocab_size).")
        global_ids = ids + modality_block.start
        for special_id in self.all_special_ids:
            if special_id < modality_block.start:
                continue
            if special_id >= modality_block.end:
                break
            special = global_ids == special_id
            if bool(special.any()):
                bad_id = int(global_ids[special].reshape(-1)[0].detach().cpu())
                raise ValueError(f"ids contains special token id: {bad_id}.")
        return global_ids


def _normalize_special_token_ids(special_token_ids: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(special_token_ids, Mapping):
        raise TypeError("special_token_ids must be a mapping of names to token ids.")

    normalized: dict[str, int] = {}
    ids: set[int] = set()
    for name, token_id in special_token_ids.items():
        _validate_name(name, name="special name")
        validate_non_negative_int(token_id, name=f"special_token_ids[{name!r}]")
        if name in normalized:
            raise ValueError("special names must be unique.")
        if token_id in ids:
            raise ValueError("special token ids must be unique.")
        normalized[name] = token_id
        ids.add(token_id)

    return normalized


def _normalize_modality_blocks(
    modality_blocks: Sequence[ModalityBlock],
) -> tuple[ModalityBlock, ...]:
    if not isinstance(modality_blocks, Sequence) or isinstance(modality_blocks, str | bytes):
        raise TypeError("modality_blocks must be a sequence of ModalityBlock values.")

    blocks: list[ModalityBlock] = []
    for index, modality_block in enumerate(modality_blocks):
        if not isinstance(modality_block, ModalityBlock):
            raise TypeError(f"modality_blocks[{index}] must be a ModalityBlock.")
        if any(_blocks_overlap(modality_block, existing) for existing in blocks):
            raise ValueError("modality blocks must not overlap.")
        blocks.append(modality_block)

    modalities = {modality_block.modality for modality_block in blocks}
    if len(modalities) != len(blocks):
        raise ValueError("modalities must be unique.")
    return tuple(blocks)


def _blocks_overlap(left: ModalityBlock, right: ModalityBlock) -> bool:
    return left.start < right.end and right.start < left.end


def _vocab_size(
    all_special_ids: Sequence[int],
    modality_blocks: Sequence[ModalityBlock],
) -> int:
    end = 0
    if all_special_ids:
        end = max(all_special_ids) + 1
    for modality_block in modality_blocks:
        end = max(end, modality_block.end)
    return end


def _validate_local_id(token_id: int, vocab_size: int, *, name: str) -> None:
    if token_id < 0 or token_id >= vocab_size:
        raise ValueError(f"{name} must be in [0, vocab_size).")


def _validate_name(value: str, *, name: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string.")
    if "." in value:
        raise ValueError(f"{name} must not contain '.'.")


def _validate_modality(value: Modality, *, name: str) -> None:
    if not isinstance(value, Modality):
        raise TypeError(f"{name} must be a Modality.")


__all__ = [
    "IdSpace",
    "Modality",
    "ModalityBlock",
]
