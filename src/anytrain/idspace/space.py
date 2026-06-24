from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto
from types import MappingProxyType


class Modality(StrEnum):
    TEXT = auto()
    AUDIO = auto()


@dataclass(frozen=True, slots=True)
class ModalityBlock:
    modality: Modality
    start: int
    vocab_size: int

    def __post_init__(self) -> None:
        _validate_modality(self.modality, name="modality")
        _validate_non_negative_int(self.start, name="start")
        _validate_positive_int(self.vocab_size, name="vocab_size")

    @property
    def end(self) -> int:
        return self.start + self.vocab_size

    def contains(self, token_id: int) -> bool:
        _validate_int(token_id, name="token_id")
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
        _validate_int(token_id, name="token_id")
        return token_id in self._special_token_name_by_id

    def modality_block_for_id(self, token_id: int) -> ModalityBlock:
        _validate_int(token_id, name="token_id")
        if self.is_special_token_id(token_id):
            raise ValueError(f"token_id is a special token id: {token_id}.")
        for modality_block in self.modality_blocks:
            if modality_block.contains(token_id):
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

    def to_global(self, modality: Modality, ids: Sequence[int]) -> list[int]:
        modality_block = self.modality_block(modality)
        local_ids = _normalize_ids(ids, name="ids")
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
        ids: Sequence[int],
        *,
        skip_special: bool = False,
    ) -> list[int]:
        modality_block = self.modality_block(modality)
        global_ids = _normalize_ids(ids, name="ids")
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


def _normalize_ids(ids: Sequence[int], *, name: str) -> list[int]:
    if not isinstance(ids, Sequence) or isinstance(ids, str | bytes):
        raise TypeError(f"{name} must be a sequence of integer ids.")
    normalized: list[int] = []
    for index, token_id in enumerate(ids):
        _validate_int(token_id, name=f"{name}[{index}]")
        normalized.append(token_id)
    return normalized


def _normalize_special_token_ids(special_token_ids: Mapping[str, int]) -> dict[str, int]:
    if not isinstance(special_token_ids, Mapping):
        raise TypeError("special_token_ids must be a mapping of names to token ids.")

    normalized: dict[str, int] = {}
    ids: set[int] = set()
    for name, token_id in special_token_ids.items():
        _validate_name(name, name="special name")
        _validate_non_negative_int(token_id, name=f"special_token_ids[{name!r}]")
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
    _validate_int(token_id, name=name)
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


def _validate_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")


def _validate_positive_int(value: int, *, name: str) -> None:
    _validate_int(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_non_negative_int(value: int, *, name: str) -> None:
    _validate_int(value, name=name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


__all__ = [
    "IdSpace",
    "Modality",
    "ModalityBlock",
]
