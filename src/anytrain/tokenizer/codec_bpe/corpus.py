from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import TypeVar

from .frame import _FrameCodec, _normalize_base_id
from .types import FrameInput

PRIVATE_USE_RANGES = (
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)

_CorpusT = TypeVar("_CorpusT", bound=Iterable[object])
_ProgressT = TypeVar("_ProgressT")


def _private_use_capacity() -> int:
    return sum(end - start + 1 for start, end in PRIVATE_USE_RANGES)


def _private_use_char(index: int) -> str:
    for start, end in PRIVATE_USE_RANGES:
        size = end - start + 1
        if index < size:
            return chr(start + index)
        index -= size
    raise ValueError("private-use character index out of range")


def _progress(
    iterable: Iterable[_ProgressT],
    *,
    enabled: bool,
    desc: str,
) -> Iterable[_ProgressT]:
    if not enabled:
        return iterable
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise ImportError("CodecBPE progress requires the `tqdm` package") from error
    return tqdm(iterable, desc=desc)


def _corpus_factory(
    corpus: _CorpusT | Callable[[], _CorpusT],
) -> Callable[[], _CorpusT]:
    if callable(corpus):
        return corpus
    if isinstance(corpus, Iterator):
        raise TypeError("corpus must be re-iterable or a callable returning a fresh iterator")
    return lambda: corpus


def _scan_base_corpus(
    corpus_factory: Callable[[], Iterable[Sequence[int]]],
    *,
    show_progress: bool,
) -> tuple[set[int], int]:
    base: set[int] = set()
    num_sequences = 0
    for seq in _progress(corpus_factory(), enabled=show_progress, desc="CodecBPE alphabet"):
        ids = tuple(_normalize_base_id(base_id) for base_id in seq)
        num_sequences += 1
        if not ids:
            raise ValueError("corpus must not contain empty sequences")
        base.update(ids)

    if num_sequences == 0:
        raise ValueError("corpus must not be empty")
    if not base:
        raise ValueError("corpus must contain at least one frame")
    return base, num_sequences


def _text_corpus(
    corpus_factory: Callable[[], Iterable[Sequence[int]]],
    base_tokens: Mapping[int, str],
) -> Iterable[str]:
    for seq in corpus_factory():
        ids = tuple(_normalize_base_id(base_id) for base_id in seq)
        if not ids:
            raise ValueError("corpus must not contain empty sequences")
        yield "".join(base_tokens[base_id] for base_id in ids)


def _encoded_corpus(
    corpus: Iterable[Sequence[FrameInput]],
    codec: _FrameCodec,
) -> Iterable[list[int]]:
    for frames in corpus:
        yield [codec.encode(frame) for frame in frames]
