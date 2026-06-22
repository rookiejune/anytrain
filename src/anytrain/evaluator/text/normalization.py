from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

type TextInput = str | Sequence[str]


@dataclass(frozen=True)
class TextNormalizationConfig:
    strip: bool = True
    collapse_whitespace: bool = True
    lowercase: bool = False


def coerce_text_batch(
    text: TextInput,
    *,
    name: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if isinstance(text, str):
        return (text,)
    if isinstance(text, bytes | bytearray) or not isinstance(text, Sequence):
        raise TypeError(f"{name} must be a string or a sequence of strings.")

    values = tuple(text)
    if not allow_empty and not values:
        raise ValueError(f"{name} must contain at least one text.")

    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"{name}[{index}] must be a string.")
    return values


def normalize_text(text: str, config: TextNormalizationConfig) -> str:
    if config.strip:
        text = text.strip()
    if config.collapse_whitespace:
        text = " ".join(text.split())
    if config.lowercase:
        text = text.lower()
    return text


def normalize_text_batch(
    text: TextInput,
    *,
    name: str,
    config: TextNormalizationConfig,
) -> tuple[str, ...]:
    return tuple(normalize_text(value, config) for value in coerce_text_batch(text, name=name))
