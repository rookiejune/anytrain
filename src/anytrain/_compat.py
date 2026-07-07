from __future__ import annotations

from collections.abc import Iterable, Iterator
from enum import Enum
from itertools import zip_longest
from typing import TypeVar

from typing_extensions import NotRequired, Self

T1 = TypeVar("T1")
T2 = TypeVar("T2")
T3 = TypeVar("T3")

__all__ = ["NotRequired", "Self", "StrEnum", "strict_zip", "strict_zip3"]


class StrEnum(str, Enum):
    @staticmethod
    def _generate_next_value_(
        name: str,
        start: int,
        count: int,
        last_values: list[str],
    ) -> str:
        return name.lower()


_MISSING = object()


def strict_zip(first: Iterable[T1], second: Iterable[T2]) -> Iterator[tuple[T1, T2]]:
    for left, right in zip_longest(first, second, fillvalue=_MISSING):
        if left is _MISSING or right is _MISSING:
            raise ValueError("zip() argument lengths differ")
        yield left, right


def strict_zip3(
    first: Iterable[T1],
    second: Iterable[T2],
    third: Iterable[T3],
) -> Iterator[tuple[T1, T2, T3]]:
    for left, middle, right in zip_longest(first, second, third, fillvalue=_MISSING):
        if left is _MISSING or middle is _MISSING or right is _MISSING:
            raise ValueError("zip() argument lengths differ")
        yield left, middle, right
