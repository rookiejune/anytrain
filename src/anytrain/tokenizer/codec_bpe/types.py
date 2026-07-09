from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

Frame = tuple[int, ...]
FrameInput = Sequence[int]


class CodecBPEState(TypedDict):
    codebook_sizes: list[int]
    tokens: dict[str, list[list[int]]]
    merges: list[dict[str, int]]
