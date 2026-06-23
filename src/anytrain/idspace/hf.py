from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Self

from .layout import Modality, ModalityRange, TokenLayout, normalize_ids
from .tokenizer import MultiTokenizer


@dataclass(frozen=True, slots=True)
class HFTokenizerAdapter:
    tokenizer: Any
    layout: TokenLayout
    modality: Modality = Modality.TEXT

    @classmethod
    def from_pretrained(
        cls,
        path: str,
        *,
        modality: Modality = Modality.TEXT,
        **kwargs: Any,
    ) -> Self:
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError("HFTokenizerAdapter.from_pretrained() requires transformers.") from error
        tokenizer = AutoTokenizer.from_pretrained(path, **kwargs)
        return cls.from_tokenizer(tokenizer, modality=modality)

    @classmethod
    def from_tokenizer(cls, tokenizer: Any, *, modality: Modality = Modality.TEXT) -> Self:
        if not isinstance(modality, Modality):
            raise TypeError("modality must be a Modality.")
        vocab = _require_vocab(tokenizer)
        token_by_id = _invert_vocab(vocab)
        special_names = _special_names(tokenizer, vocab, token_by_id)

        special_items = list(special_names.items())
        special_token_ids = {name: old_id for old_id, name in special_items}
        layout = TokenLayout(
            special_token_ids,
            [ModalityRange(modality, 0, max(token_by_id) + 1)],
        )
        return cls(
            tokenizer=tokenizer,
            layout=layout,
            modality=modality,
        )

    def multi_tokenizer(self) -> MultiTokenizer:
        return MultiTokenizer({self.modality: self})

    def encode(self, value: Any) -> list[int]:
        old_ids = _encode_without_special(self.tokenizer, value)
        local_ids: list[int] = []
        for old_id in old_ids:
            if self.layout.is_special_token_id(old_id):
                raise ValueError(f"pretrained tokenizer produced special token id: {old_id}")
            if old_id < 0 or old_id >= self.layout.modality_range(self.modality).vocab_size:
                raise KeyError(f"unknown pretrained token id: {old_id}")
            local_ids.append(old_id)
        return local_ids

    def decode(self, ids: Sequence[int]) -> Any:
        local_ids = normalize_ids(ids, name="ids")
        modality_range = self.layout.modality_range(self.modality)
        for old_id in local_ids:
            if self.layout.is_special_token_id(old_id):
                raise ValueError(f"ids contains moved special token id: {old_id}.")
            if not modality_range.contains(old_id):
                raise KeyError(f"unknown pretrained token id: {old_id}")
        return _decode(self.tokenizer, local_ids)


def _require_vocab(tokenizer: Any) -> dict[str, int]:
    try:
        vocab = tokenizer.get_vocab()
    except AttributeError as error:
        raise TypeError("tokenizer must expose get_vocab().") from error
    if not isinstance(vocab, Mapping):
        raise TypeError("tokenizer.get_vocab() must return a mapping.")
    normalized: dict[str, int] = {}
    for token, token_id in vocab.items():
        if not isinstance(token, str):
            raise TypeError("tokenizer vocab keys must be strings.")
        if isinstance(token_id, bool) or not isinstance(token_id, int):
            raise TypeError("tokenizer vocab ids must be integers.")
        normalized[token] = token_id
    return normalized


def _invert_vocab(vocab: Mapping[str, int]) -> dict[int, str]:
    token_by_id: dict[int, str] = {}
    for token, token_id in vocab.items():
        if token_id in token_by_id:
            raise ValueError("tokenizer vocab ids must be unique.")
        token_by_id[token_id] = token
    return token_by_id


def _special_names(
    tokenizer: Any,
    vocab: Mapping[str, int],
    token_by_id: Mapping[int, str],
) -> dict[int, str]:
    explicit: dict[int, str] = {}
    try:
        special_map = tokenizer.special_tokens_map
    except AttributeError:
        special_map = {}
    if not isinstance(special_map, Mapping):
        raise TypeError("tokenizer.special_tokens_map must be a mapping.")

    for key, value in special_map.items():
        if key == "additional_special_tokens":
            if not isinstance(value, Sequence) or isinstance(value, str | bytes):
                raise TypeError("additional_special_tokens must be a sequence.")
            for index, token in enumerate(value):
                old_id = _special_token_id(vocab, token)
                explicit[old_id] = f"additional_{index}"
        else:
            old_id = _special_token_id(vocab, value)
            explicit[old_id] = key.removesuffix("_token")

    try:
        all_special_ids = tokenizer.all_special_ids
    except AttributeError:
        all_special_ids = []
    if not isinstance(all_special_ids, Sequence) or isinstance(all_special_ids, str | bytes):
        raise TypeError("tokenizer.all_special_ids must be a sequence.")

    for old_id in all_special_ids:
        if isinstance(old_id, bool) or not isinstance(old_id, int):
            raise TypeError("all_special_ids must contain integers.")
        if old_id not in token_by_id:
            raise ValueError(f"special token id {old_id} is not present in vocab.")
        if old_id not in explicit:
            explicit[old_id] = _fallback_special_name(token_by_id[old_id], old_id)

    names = list(explicit.values())
    if len(set(names)) != len(names):
        raise ValueError("special token names must be unique.")
    return dict(sorted(explicit.items(), key=lambda item: item[0]))


def _special_token_id(vocab: Mapping[str, int], value: Any) -> int:
    if not isinstance(value, str):
        raise TypeError("special token values must be strings.")
    if value not in vocab:
        raise ValueError(f"special token {value!r} is not present in vocab.")
    return vocab[value]


def _fallback_special_name(token: str, old_id: int) -> str:
    stripped = token.strip("<>[]/ ").lower()
    cleaned = "".join(ch if ch.isalnum() else "_" for ch in stripped).strip("_")
    return cleaned or f"special_{old_id}"


def _encode_without_special(tokenizer: Any, value: Any) -> list[int]:
    try:
        ids = tokenizer.encode(value, add_special_tokens=False)
    except TypeError:
        ids = tokenizer.encode(value)
    return normalize_ids(ids, name="encoded ids")


def _decode(tokenizer: Any, ids: Sequence[int]) -> Any:
    old_ids = normalize_ids(ids, name="ids")
    try:
        return tokenizer.decode(old_ids, skip_special_tokens=True)
    except TypeError:
        return tokenizer.decode(old_ids)


__all__ = [
    "HFTokenizerAdapter",
]
