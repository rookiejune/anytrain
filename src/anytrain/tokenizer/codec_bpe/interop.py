from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

from .stats import Merge

if TYPE_CHECKING:
    from tokenizers import Tokenizer
    from tokenizers.models import BPE as TokenizersBPE


def _require_tokenizers_bpe() -> type[TokenizersBPE]:
    try:
        from tokenizers.models import BPE as TokenizersBPE
    except ImportError as error:
        raise ImportError("CodecBPE requires the `tokenizers` package") from error
    return TokenizersBPE


def _require_tokenizer() -> type[Tokenizer]:
    try:
        from tokenizers import Tokenizer
    except ImportError as error:
        raise ImportError("CodecBPE.tokenizer() requires the `tokenizers` package") from error
    return Tokenizer


def _train_tokenizers_bpe(
    corpus: Iterable[str],
    *,
    vocab_size: int,
    min_frequency: int,
    show_progress: bool,
    max_token_length: int | None,
    length: int | None,
) -> Tokenizer:
    try:
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
    except ImportError as error:
        raise ImportError("CodecBPE.train() requires the `tokenizers` package") from error

    tokenizer_type = _require_tokenizer()
    tokenizer = tokenizer_type(BPE())
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=show_progress,
        max_token_length=max_token_length,
    )
    tokenizer.train_from_iterator(corpus, trainer=trainer, length=length)
    return tokenizer


def _core_state_from_tokenizer(
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
        merges.append(
            Merge(
                left=vocab[left_text],
                right=vocab[right_text],
                token_id=vocab[token_text],
            )
        )
    return tokens, merges


