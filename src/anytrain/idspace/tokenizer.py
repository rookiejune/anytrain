from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .layout import Modality, normalize_ids


@runtime_checkable
class SubTokenizer(Protocol):
    def encode(self, value: Any) -> Sequence[int]: ...

    def decode(self, ids: Sequence[int]) -> Any: ...


class MultiTokenizer:
    def __init__(self, tokenizers: Mapping[Modality, SubTokenizer]) -> None:
        self.tokenizers = _normalize_tokenizers(tokenizers)

    def encode(self, modality: Modality, value: Any) -> list[int]:
        tokenizer = self._tokenizer(modality)
        return normalize_ids(tokenizer.encode(value), name="encoded ids")

    def decode(self, modality: Modality, ids: Sequence[int]) -> Any:
        tokenizer = self._tokenizer(modality)
        local_ids = normalize_ids(ids, name="ids")
        return tokenizer.decode(local_ids)

    def _tokenizer(self, modality: Modality) -> SubTokenizer:
        if not isinstance(modality, Modality):
            raise TypeError("modality must be a Modality.")
        try:
            return self.tokenizers[modality]
        except KeyError as error:
            raise KeyError(f"unknown tokenizer modality {modality!r}.") from error


def _normalize_tokenizers(tokenizers: Mapping[Modality, SubTokenizer]) -> dict[Modality, SubTokenizer]:
    if not isinstance(tokenizers, Mapping):
        raise TypeError("tokenizers must be a mapping of Modality to SubTokenizer.")
    normalized: dict[Modality, SubTokenizer] = {}
    for modality, tokenizer in tokenizers.items():
        if not isinstance(modality, Modality):
            raise TypeError("tokenizer keys must be Modality values.")
        if not isinstance(tokenizer, SubTokenizer):
            raise TypeError(f"tokenizers[{modality!r}] must implement SubTokenizer.")
        if modality in normalized:
            raise ValueError("tokenizer modalities must be unique.")
        normalized[modality] = tokenizer
    return normalized


__all__ = [
    "MultiTokenizer",
    "SubTokenizer",
]
