from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence

from .corpus import _progress
from .frame import _FrameCodec
from .stats import (
    CodecBPEEvalStats,
    CompressionStats,
    TokenCount,
    TokenFrequencyStats,
    TokenLengthStats,
)
from .types import FrameInput

LENGTH_QUANTILES = (0.5, 0.9, 0.95, 0.99)


def _eval(
    corpus: Iterable[Sequence[FrameInput]],
    codec: _FrameCodec,
    encode: Callable[[Sequence[FrameInput]], Sequence[int]],
    *,
    token_lengths: Mapping[int, int],
    vocab_size: int,
    show_progress: bool,
    top_k: int,
) -> CodecBPEEvalStats:
    num_sequences = 0
    original_frames = 0
    encoded_tokens = 0
    token_counts: Counter[int] = Counter()
    for seq in _progress(corpus, enabled=show_progress, desc="CodecBPE eval"):
        frames = tuple(codec.normalize(frame) for frame in seq)
        encoded = tuple(encode(frames))
        num_sequences += 1
        original_frames += len(frames)
        encoded_tokens += len(encoded)
        token_counts.update(encoded)

    return _eval_stats(
        num_sequences=num_sequences,
        original_frames=original_frames,
        encoded_tokens=encoded_tokens,
        token_counts=token_counts,
        token_lengths=token_lengths,
        vocab_size=vocab_size,
        top_k=top_k,
    )


def _eval_stats(
    *,
    num_sequences: int,
    original_frames: int,
    encoded_tokens: int,
    token_counts: Counter[int],
    token_lengths: Mapping[int, int],
    vocab_size: int,
    top_k: int,
) -> CodecBPEEvalStats:
    if num_sequences == 0:
        raise ValueError("corpus must not be empty")
    if original_frames == 0:
        raise ValueError("corpus must contain at least one frame")

    compression_ratio = encoded_tokens / original_frames
    compression_factor = original_frames / encoded_tokens if encoded_tokens else float("inf")
    compression = CompressionStats(
        num_sequences=num_sequences,
        original_frames=original_frames,
        encoded_tokens=encoded_tokens,
        mean_original_length=original_frames / num_sequences,
        mean_encoded_length=encoded_tokens / num_sequences,
        compression_ratio=compression_ratio,
        compression_factor=compression_factor,
        compression_gain=1.0 - compression_ratio,
    )
    return CodecBPEEvalStats(
        compression=compression,
        token_frequency=_token_frequency_stats(
            token_counts,
            token_lengths,
            encoded_tokens,
            vocab_size,
            top_k=top_k,
        ),
        token_length=_token_length_stats(token_counts, token_lengths),
    )


def _token_frequency_stats(
    token_counts: Counter[int],
    token_lengths: Mapping[int, int],
    total_tokens: int,
    vocab_size: int,
    *,
    top_k: int,
) -> TokenFrequencyStats:
    count_histogram = Counter(token_counts.values())
    entropy = -sum(
        (count / total_tokens) * math.log(count / total_tokens) for count in token_counts.values()
    )
    top_counts = tuple(
        TokenCount(
            token_id=token_id,
            count=count,
            frequency=count / total_tokens,
            length=token_lengths[token_id],
        )
        for token_id, count in sorted(
            token_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
    )
    return TokenFrequencyStats(
        total_tokens=total_tokens,
        token_count_histogram=dict(sorted(count_histogram.items())),
        top_token_counts=top_counts,
        num_used_tokens=len(token_counts),
        vocab_coverage=len(token_counts) / vocab_size,
        entropy=entropy,
    )


def _token_length_stats(
    token_counts: Counter[int],
    token_lengths: Mapping[int, int],
) -> TokenLengthStats:
    vocab_counts = Counter(token_lengths.values())
    used_counts: Counter[int] = Counter()
    used_length_total = 0
    total_used_tokens = sum(token_counts.values())
    for token_id, count in token_counts.items():
        if token_id not in token_lengths:
            raise KeyError(f"encoded unknown token_id: {token_id}")
        length = token_lengths[token_id]
        used_counts[length] += count
        used_length_total += length * count

    used_length_counts = _dense_length_counts(used_counts)
    vocab_length_counts = _dense_length_counts(vocab_counts)
    return TokenLengthStats(
        used_token_length_counts=used_length_counts,
        used_token_length_frequencies=_length_frequencies(used_length_counts),
        vocab_token_length_counts=vocab_length_counts,
        vocab_token_length_frequencies=_length_frequencies(vocab_length_counts),
        mean_used_token_length=used_length_total / total_used_tokens,
        mean_vocab_token_length=sum(length * count for length, count in vocab_counts.items())
        / sum(vocab_counts.values()),
        max_used_token_length=max(used_counts),
        max_vocab_token_length=max(vocab_counts),
        used_token_length_quantiles=_length_quantiles(used_counts),
        vocab_token_length_quantiles=_length_quantiles(vocab_counts),
    )


def _dense_length_counts(length_counts: Mapping[int, int]) -> tuple[int, ...]:
    max_length = max(length_counts)
    return tuple(length_counts.get(length, 0) for length in range(max_length + 1))


def _length_frequencies(length_counts: Sequence[int]) -> tuple[float, ...]:
    total = sum(length_counts)
    return tuple(count / total for count in length_counts)


def _length_quantiles(length_counts: Mapping[int, int]) -> dict[str, float]:
    total = sum(length_counts.values())
    quantiles: dict[str, float] = {}
    for quantile in LENGTH_QUANTILES:
        rank = max(1, math.ceil(quantile * total))
        cumulative = 0
        for length, count in sorted(length_counts.items()):
            cumulative += count
            if cumulative >= rank:
                quantiles[f"p{round(quantile * 100):02d}"] = float(length)
                break
    return quantiles


def _normalize_top_k(top_k: int) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k < 0:
        raise ValueError("top_k must be non-negative")
    return top_k
