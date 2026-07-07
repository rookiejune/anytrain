from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from os import PathLike
from typing import Protocol, TypedDict, Unpack, overload

import torch
from torch import Tensor

AudioReference = str | PathLike[str]


class TTSKwargs(TypedDict, total=False):
    speaker: str | None
    language: str | None
    sample_rate: int | None
    max_new_tokens: int | None
    temperature: float | None
    top_p: float | None
    seed: int | None
    extra: Mapping[str, object]


_TTS_OPTION_KEYS = frozenset(TTSKwargs.__annotations__)


@dataclass(init=False)
class TTSOptions:
    speaker: str | None = None
    language: str | None = None
    sample_rate: int | None = None
    max_new_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    seed: int | None = None
    extra: Mapping[str, object] = field(default_factory=dict)
    _set_fields: frozenset[str] = field(default_factory=frozenset, repr=False, compare=False)

    def __init__(self, **kwargs: Unpack[TTSKwargs]) -> None:
        unknown = set(kwargs) - _TTS_OPTION_KEYS
        if unknown:
            names = ", ".join(sorted(unknown))
            raise TypeError(f"unknown TTS option keys: {names}.")

        self.speaker = kwargs.get("speaker")
        self.language = kwargs.get("language")
        self.sample_rate = kwargs.get("sample_rate")
        self.max_new_tokens = kwargs.get("max_new_tokens")
        self.temperature = kwargs.get("temperature")
        self.top_p = kwargs.get("top_p")
        self.seed = kwargs.get("seed")
        self.extra = kwargs.get("extra", {})
        self._set_fields = frozenset(kwargs)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.sample_rate is not None:
            _validate_positive_int(self.sample_rate, "sample_rate")
        if self.max_new_tokens is not None:
            _validate_positive_int(self.max_new_tokens, "max_new_tokens")
        if self.temperature is not None and self.temperature <= 0:
            raise ValueError("temperature must be positive when set.")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1] when set.")
        if self.seed is not None and (
            isinstance(self.seed, bool) or not isinstance(self.seed, int)
        ):
            raise TypeError("seed must be an integer when set.")
        _validate_extra(self.extra)

    def merged(self, override: TTSOptions | None) -> TTSOptions:
        if override is None:
            return self
        return self._new(
            speaker=override.speaker if override._has("speaker") else self.speaker,
            language=override.language if override._has("language") else self.language,
            sample_rate=override.sample_rate
            if override._has("sample_rate")
            else self.sample_rate,
            max_new_tokens=override.max_new_tokens
            if override._has("max_new_tokens")
            else self.max_new_tokens,
            temperature=override.temperature
            if override._has("temperature")
            else self.temperature,
            top_p=override.top_p if override._has("top_p") else self.top_p,
            seed=override.seed if override._has("seed") else self.seed,
            extra=override.extra if override._has("extra") else self.extra,
            set_fields=self._set_fields | override._set_fields,
        )

    def _has(self, name: str) -> bool:
        return name in self._set_fields

    @classmethod
    def _new(
        cls,
        *,
        speaker: str | None,
        language: str | None,
        sample_rate: int | None,
        max_new_tokens: int | None,
        temperature: float | None,
        top_p: float | None,
        seed: int | None,
        extra: Mapping[str, object],
        set_fields: frozenset[str],
    ) -> TTSOptions:
        options = cls(
            speaker=speaker,
            language=language,
            sample_rate=sample_rate,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            extra=extra,
        )
        options._set_fields = set_fields
        return options


@dataclass(eq=False)
class TTSTokens:
    input_ids: Tensor
    attention_mask: Tensor | None = None
    extra_tensors: Mapping[str, Tensor] = field(default_factory=dict)
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.input_ids, Tensor):
            raise TypeError("input_ids must be a torch.Tensor.")
        if self.input_ids.ndim < 1:
            raise ValueError("input_ids must have at least one dimension.")
        if self.attention_mask is not None:
            if not isinstance(self.attention_mask, Tensor):
                raise TypeError("attention_mask must be a torch.Tensor when set.")
            if self.attention_mask.shape != self.input_ids.shape:
                raise ValueError("attention_mask must match input_ids shape.")
        for key, value in self.extra_tensors.items():
            if not isinstance(key, str) or not key:
                raise ValueError("extra_tensors keys must be non-empty strings.")
            if not isinstance(value, Tensor):
                raise TypeError(f"extra_tensors[{key!r}] must be a torch.Tensor.")

    def to(self, device: torch.device | str) -> TTSTokens:
        resolved = torch.device(device)
        return TTSTokens(
            input_ids=self.input_ids.to(resolved),
            attention_mask=None
            if self.attention_mask is None
            else self.attention_mask.to(resolved),
            extra_tensors={key: value.to(resolved) for key, value in self.extra_tensors.items()},
            meta=self.meta,
        )

    def model_inputs(self) -> dict[str, Tensor]:
        inputs = {"input_ids": self.input_ids, **dict(self.extra_tensors)}
        if self.attention_mask is not None:
            inputs["attention_mask"] = self.attention_mask
        return inputs


@dataclass(eq=False)
class TTSGeneration:
    value: object
    sample_rate: int | None = None
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sample_rate is not None:
            _validate_positive_int(self.sample_rate, "sample_rate")


@dataclass(eq=False)
class TTSOutput:
    waveform: Tensor
    sample_rate: int
    duration: float
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.waveform, Tensor):
            raise TypeError("waveform must be a torch.Tensor.")
        if self.waveform.ndim != 2:
            raise ValueError("waveform must have shape [channels, time].")
        if self.waveform.shape[0] <= 0 or self.waveform.shape[-1] <= 0:
            raise ValueError("waveform must not be empty.")
        _validate_positive_int(self.sample_rate, "sample_rate")
        if self.duration <= 0:
            raise ValueError("duration must be positive.")


class TTSBackend(Protocol):
    name: str
    sample_rate: int

    def config_hash(self) -> str: ...

    @overload
    def synthesize(
        self,
        text: str,
        options: TTSOptions | None = None,
        reference_audio_path: AudioReference | None = None,
    ) -> TTSOutput: ...

    @overload
    def synthesize(
        self,
        text: Sequence[str],
        options: TTSOptions | None = None,
        reference_audio_paths: Sequence[AudioReference] | None = None,
    ) -> list[TTSOutput]: ...

    def synthesize(
        self,
        text: str | Sequence[str],
        options: TTSOptions | None = None,
        reference_audio_path: AudioReference | None = None,
        reference_audio_paths: Sequence[AudioReference] | None = None,
    ) -> TTSOutput | list[TTSOutput]: ...


def validate_text(text: str) -> str:
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if not text.strip():
        raise ValueError("text must not be empty.")
    return text


def waveform_duration(waveform: Tensor, sample_rate: int) -> float:
    _validate_positive_int(sample_rate, "sample_rate")
    if waveform.ndim != 2:
        raise ValueError("waveform must have shape [channels, time].")
    return float(waveform.shape[-1]) / float(sample_rate)


def _validate_positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_extra(extra: Mapping[str, object]) -> None:
    if not isinstance(extra, Mapping):
        raise TypeError("extra must be a mapping.")
    for key in extra:
        if not isinstance(key, str) or not key:
            raise ValueError("extra keys must be non-empty strings.")
