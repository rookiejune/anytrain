from __future__ import annotations

from collections.abc import Sequence

from anytrain._compat import strict_zip

from .types import Frame, FrameInput


class _FrameCodec:
    def __init__(self, codebook_sizes: Sequence[int]) -> None:
        self.codebook_sizes = _normalize_codebook_sizes(codebook_sizes)
        self.num_codebooks = len(self.codebook_sizes)

        strides: list[int] = []
        stride = 1
        for size in self.codebook_sizes:
            strides.append(stride)
            stride *= size
        self.strides = tuple(strides)
        self.vocab_size = stride

    def normalize(self, frame: FrameInput) -> Frame:
        return _normalize_frame(frame, self.codebook_sizes)

    def encode(self, frame: FrameInput) -> int:
        values = self.normalize(frame)
        return sum(value * stride for value, stride in strict_zip(values, self.strides))

    def decode(self, base_id: int) -> Frame:
        base_id = _normalize_base_id(base_id)
        if base_id >= self.vocab_size:
            raise ValueError("frame id is outside the configured codebook space")

        values: list[int] = []
        remaining = base_id
        for size in self.codebook_sizes:
            values.append(remaining % size)
            remaining //= size
        return tuple(values)




def _normalize_codebook_sizes(codebook_sizes: Sequence[int]) -> tuple[int, ...]:
    if not codebook_sizes:
        raise ValueError("codebook_sizes must not be empty")

    sizes: list[int] = []
    for size in codebook_sizes:
        if isinstance(size, bool) or not isinstance(size, int):
            raise TypeError("codebook_sizes must contain integer sizes")
        if size <= 0:
            raise ValueError("codebook sizes must be positive")
        sizes.append(size)
    return tuple(sizes)


def _normalize_frame(frame: FrameInput, codebook_sizes: Sequence[int]) -> Frame:
    if isinstance(frame, int | str):
        raise TypeError("frames must be integer sequences; use [id] for single-codebook frames")

    values = tuple(_normalize_code_id(value) for value in frame)
    if len(values) != len(codebook_sizes):
        raise ValueError("frames must match the configured number of codebooks")

    for index, (value, size) in enumerate(strict_zip(values, codebook_sizes)):
        if value >= size:
            raise ValueError(f"frame code id at codebook {index} must be in [0, {size})")
    return values


def _normalize_code_id(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("frame code ids must be integers")
    if value < 0:
        raise ValueError("frame code ids must be non-negative")
    return value


def _normalize_base_id(base_id: int) -> int:
    if isinstance(base_id, bool) or not isinstance(base_id, int):
        raise TypeError("frame ids must be integers")
    if base_id < 0:
        raise ValueError("frame ids must be non-negative")
    return base_id


