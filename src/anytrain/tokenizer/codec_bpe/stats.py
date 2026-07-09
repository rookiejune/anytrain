from __future__ import annotations

from dataclasses import dataclass


# `tokenizers.models.BPE` is a Rust extension type and cannot be subclassed in Python.
@dataclass(frozen=True)
class Merge:
    left: int
    right: int
    token_id: int


@dataclass(frozen=True)
class CodecBPEEvalStats:
    num_sequences: int
    original_frames: int
    encoded_tokens: int
    mean_original_length: float
    mean_encoded_length: float
    compression_ratio: float
    compression_factor: float
    compression_gain: float
    token_count_histogram: dict[int, int]
    top_token_counts: tuple[tuple[int, int, float, int], ...]
    num_used_tokens: int
    vocab_coverage: float
    entropy: float
    used_token_length_counts: tuple[int, ...]
    used_token_length_frequencies: tuple[float, ...]
    vocab_token_length_counts: tuple[int, ...]
    vocab_token_length_frequencies: tuple[float, ...]
    mean_used_token_length: float
    mean_vocab_token_length: float
    max_used_token_length: int
    max_vocab_token_length: int
    used_token_length_quantiles: dict[str, float]
    vocab_token_length_quantiles: dict[str, float]
