from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum, auto


class Modality(StrEnum):
    TEXT = auto()
    AUDIO = auto()


@dataclass(frozen=True, slots=True)
class ModalityRange:
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


class TokenLayout:
    def __init__(
        self,
        special_token_ids: Mapping[str, int],
        modality_ranges: Sequence[ModalityRange],
    ) -> None:
        self.special_token_ids = _normalize_special_token_ids(special_token_ids)
        self._special_token_name_by_id = {
            token_id: name for name, token_id in self.special_token_ids.items()
        }
        self.modality_ranges = _normalize_modality_ranges(modality_ranges)
        self._range_by_modality = {
            modality_range.modality: modality_range for modality_range in self.modality_ranges
        }

    @property
    def all_special_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._special_token_name_by_id))

    @property
    def vocab_size(self) -> int:
        end = 0
        if self.all_special_ids:
            end = max(self.all_special_ids) + 1
        for modality_range in self.modality_ranges:
            end = max(end, modality_range.end)
        return end

    def modality_range(self, modality: Modality) -> ModalityRange:
        _validate_modality(modality, name="modality")
        try:
            return self._range_by_modality[modality]
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

    def modality_range_for_id(self, token_id: int) -> ModalityRange:
        _validate_int(token_id, name="token_id")
        if self.is_special_token_id(token_id):
            raise ValueError(f"token_id is a special token id: {token_id}.")
        for modality_range in self.modality_ranges:
            if modality_range.contains(token_id):
                return modality_range
        raise ValueError(f"token_id is outside all modality ranges: {token_id}.")

    def to_global(self, modality: Modality, ids: Sequence[int]) -> list[int]:
        modality_range = self.modality_range(modality)
        local_ids = normalize_ids(ids, name="ids")
        global_ids: list[int] = []
        for local_id in local_ids:
            _validate_local_id(local_id, modality_range.vocab_size, name="ids")
            token_id = modality_range.start + local_id
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
        modality_range = self.modality_range(modality)
        global_ids = normalize_ids(ids, name="ids")
        local_ids: list[int] = []
        for token_id in global_ids:
            if self.is_special_token_id(token_id):
                if skip_special:
                    continue
                raise ValueError(f"ids contains special token id: {token_id}.")
            if not modality_range.contains(token_id):
                raise ValueError(f"ids contains token outside modality {modality!r}: {token_id}.")
            local_ids.append(token_id - modality_range.start)
        return local_ids


def normalize_ids(ids: Sequence[int], *, name: str) -> list[int]:
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


def _normalize_modality_ranges(
    modality_ranges: Sequence[ModalityRange],
) -> tuple[ModalityRange, ...]:
    if not isinstance(modality_ranges, Sequence) or isinstance(modality_ranges, str | bytes):
        raise TypeError("modality_ranges must be a sequence of ModalityRange values.")

    ranges: list[ModalityRange] = []
    for index, modality_range in enumerate(modality_ranges):
        if not isinstance(modality_range, ModalityRange):
            raise TypeError(f"modality_ranges[{index}] must be a ModalityRange.")
        if any(_ranges_overlap(modality_range, existing) for existing in ranges):
            raise ValueError("modality ranges must not overlap.")
        ranges.append(modality_range)

    modalities = {modality_range.modality for modality_range in ranges}
    if len(modalities) != len(ranges):
        raise ValueError("modalities must be unique.")
    return tuple(ranges)


def _ranges_overlap(left: ModalityRange, right: ModalityRange) -> bool:
    return left.start < right.end and right.start < left.end


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
    "Modality",
    "ModalityRange",
    "TokenLayout",
    "normalize_ids",
]
