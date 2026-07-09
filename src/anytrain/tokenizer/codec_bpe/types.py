from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TypedDict, TypeVar

import torch

from anytrain._compat import NotRequired

Frame = tuple[int, ...]
FrameInput = Sequence[int]
FrameState = list[int]
BaseCorpus = Iterable[Sequence[int]]
BaseCorpusFactory = Callable[[], BaseCorpus]
FrameCorpus = Iterable[Sequence[FrameInput]]
FrameCorpusFactory = Callable[[], FrameCorpus]
CorpusT = TypeVar("CorpusT", bound=Iterable[object])

RepeatInterleaveOutput = (
    tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
)


class _TokenizersBPEKwargs(TypedDict):
    vocab: dict[str, int]
    merges: list[tuple[str, str]]
    cache_capacity: NotRequired[int]
    dropout: NotRequired[float]
    unk_token: NotRequired[str]
    continuing_subword_prefix: NotRequired[str]
    end_of_word_suffix: NotRequired[str]
    fuse_unk: NotRequired[bool]
    byte_fallback: bool
    ignore_merges: bool


class CodecBPEState(TypedDict):
    codebook_sizes: list[int]
    tokens: dict[str, list[FrameState]]
    merges: list[dict[str, int]]
