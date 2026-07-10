from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, TypeVar

import torch

from anytrain._compat import Self

if TYPE_CHECKING:
    from tokenizers import Tokenizer

PRIVATE_USE_RANGES = (
    (0xE000, 0xF8FF),
    (0xF0000, 0xFFFFD),
    (0x100000, 0x10FFFD),
)

Merge = tuple[int, int, int]
_ProgressT = TypeVar("_ProgressT")


class CoreBPE:
    def __init__(
        self,
        tokens: Mapping[int, Sequence[int]],
        merges: Sequence[Merge | tuple[int, int, int]] = (),
    ) -> None:
        if not tokens:
            raise ValueError("tokens must not be empty")

        self.tokens = {
            int(token_id): tuple(base_ids)
            for token_id, base_ids in tokens.items()
        }
        for token_id, base_ids in self.tokens.items():
            if not base_ids:
                raise ValueError(f"token {token_id} must contain at least one frame")

        self.merges = tuple(merges)
        self.base_to_id = self.build_base_to_id(self.tokens)

    @property
    def vocab_size(self) -> int:
        return max(self.tokens) + 1

    @classmethod
    def train(
        cls,
        corpus: Callable[[], Iterable[Sequence[int]]],
        *,
        vocab_size: int = 30_000,
        min_frequency: int = 0,
        show_progress: bool = True,
        max_token_length: int | None = None,
    ) -> Self:
        base, num_training_sequences = scan_base_corpus(
            corpus,
            show_progress=show_progress,
        )
        base_tokens = private_use_tokens(base)
        tokenizer = train_tokenizers_bpe(
            text_corpus(corpus, base_tokens),
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            max_token_length=max_token_length,
            length=num_training_sequences,
        )
        tokens, merges = core_state_from_tokenizer(tokenizer, base_tokens)
        return cls(tokens, merges)

    def base_ids(self, token_id: int) -> tuple[int, ...]:
        if token_id in self.tokens:
            return self.tokens[token_id]
        raise KeyError(f"unknown token_id: {token_id}")

    def decode_with_counts(self, token_ids: Sequence[int]) -> tuple[list[int], list[int]]:
        if not token_ids:
            raise ValueError("token_ids must not be empty")

        base_ids: list[int] = []
        counts: list[int] = []
        for token_id in token_ids:
            ids = self.base_ids(token_id)
            base_ids.extend(ids)
            counts.append(len(ids))
        return base_ids, counts

    def repeat_interleave(
        self,
        x: torch.Tensor,
        token_ids: torch.Tensor,
        *,
        dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 0:
            raise ValueError("x must have at least one dimension")
        self.validate_token_ids(token_ids)
        if token_ids.dim() != 1:
            raise ValueError("token_ids must be a 1D tensor")

        dim = self.dim(dim, x.dim())
        if x.size(dim) != token_ids.numel():
            raise ValueError("x and token_ids must align on the sequence dimension")

        ids = [int(token_id) for token_id in token_ids.detach().cpu().tolist()]
        base_ids, counts = self.decode_with_counts(ids)
        repeats = torch.tensor(counts, dtype=torch.long, device=x.device)
        expanded_x = torch.repeat_interleave(x, repeats, dim=dim)
        expanded_ids = torch.tensor(base_ids, dtype=token_ids.dtype, device=token_ids.device)
        return expanded_x, expanded_ids

    def token_lengths(self) -> dict[int, int]:
        return {token_id: len(base_ids) for token_id, base_ids in self.tokens.items()}

    @staticmethod
    def validate_token_ids(token_ids: torch.Tensor) -> None:
        if (
            token_ids.dtype == torch.bool
            or torch.is_floating_point(token_ids)
            or torch.is_complex(token_ids)
        ):
            raise TypeError("token_ids must contain integer ids")

    @staticmethod
    def build_base_to_id(tokens: Mapping[int, tuple[int, ...]]) -> dict[int, int]:
        base_to_id: dict[int, int] = {}
        for token_id, base_ids in tokens.items():
            if len(base_ids) == 1:
                base_to_id[base_ids[0]] = token_id
        return base_to_id

    @staticmethod
    def dim(dim: int, ndim: int) -> int:
        if dim < 0:
            dim += ndim
        if dim < 0 or dim >= ndim:
            raise ValueError("dim is out of range for x")
        return dim


def private_use_char(index: int) -> str:
    for start, end in PRIVATE_USE_RANGES:
        size = end - start + 1
        if index < size:
            return chr(start + index)
        index -= size
    raise ValueError("private-use character index out of range")


def private_use_tokens(base_ids: Iterable[int]) -> dict[int, str]:
    return {
        base_id: private_use_char(index)
        for index, base_id in enumerate(sorted(base_ids))
    }


def progress(
    iterable: Iterable[_ProgressT],
    *,
    enabled: bool,
    desc: str,
) -> Iterable[_ProgressT]:
    if not enabled:
        return iterable
    from tqdm.auto import tqdm

    return tqdm(iterable, desc=desc)


def scan_base_corpus(
    corpus: Callable[[], Iterable[Sequence[int]]],
    *,
    show_progress: bool,
) -> tuple[set[int], int]:
    base: set[int] = set()
    num_sequences = 0
    for seq in progress(corpus(), enabled=show_progress, desc="CodecBPE alphabet"):
        ids = tuple(seq)
        num_sequences += 1
        if not ids:
            raise ValueError("corpus must not contain empty sequences")
        base.update(ids)

    if num_sequences == 0:
        raise ValueError("corpus must not be empty")
    if not base:
        raise ValueError("corpus must contain at least one frame")
    return base, num_sequences


def text_corpus(
    corpus: Callable[[], Iterable[Sequence[int]]],
    base_tokens: Mapping[int, str],
) -> Iterable[str]:
    for seq in corpus():
        ids = tuple(seq)
        if not ids:
            raise ValueError("corpus must not contain empty sequences")
        yield base_text(ids, base_tokens)


def base_text(base_ids: Sequence[int], base_tokens: Mapping[int, str]) -> str:
    pieces: list[str] = []
    for base_id in base_ids:
        if base_id not in base_tokens:
            raise KeyError(f"unknown frame id: {base_id}")
        pieces.append(base_tokens[base_id])
    return "".join(pieces)


def train_tokenizers_bpe(
    corpus: Iterable[str],
    *,
    vocab_size: int,
    min_frequency: int,
    show_progress: bool,
    max_token_length: int | None,
    length: int | None,
) -> Tokenizer:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer

    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        max_token_length=max_token_length,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer, length=length)
    return tokenizer


def core_state_from_tokenizer(
    tokenizer: Tokenizer,
    base_tokens: Mapping[int, str],
) -> tuple[dict[int, tuple[int, ...]], list[Merge]]:
    model = json.loads(tokenizer.to_str())["model"]
    char_to_base = {text: base_id for base_id, text in base_tokens.items()}
    vocab = {text: int(token_id) for text, token_id in model["vocab"].items()}
    tokens = {
        token_id: tuple(char_to_base[char] for char in text) for text, token_id in vocab.items()
    }
    merges: list[Merge] = []
    for left_text, right_text in model["merges"]:
        token_text = left_text + right_text
        merges.append((vocab[left_text], vocab[right_text], vocab[token_text]))
    return tokens, merges
