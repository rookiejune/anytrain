from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, TypedDict, Unpack, cast

import torch
from torch import Tensor, nn

from anytrain.tts import (
    TTSGeneration,
    TTSOptions,
    TTSOutput,
    TTSTokens,
    validate_text,
    waveform_duration,
)

from ._deps import load_transformers_auto_model_class

DEFAULT_MODEL = "moss-tts"
DEFAULT_SAMPLE_RATE = 24000


class MossLoadKwargs(TypedDict, total=False):
    attn_implementation: object
    device_map: object
    dtype: object
    low_cpu_mem_usage: bool
    torch_dtype: object
    use_safetensors: bool


class MossRuntimeKwargs(TypedDict, total=False):
    audio_tokenizer_pretrained_name_or_path: str
    audio_temperature: float
    audio_top_p: float
    do_sample: bool
    max_new_frames: int
    mode: str
    output_audio_path: str | Path
    prompt_audio_path: str | Path
    reference_audio_path: str | Path
    text_temperature: float
    text_top_p: float


class _DecodeModel(Protocol):
    def decode(self, generation: object, **kwargs: object) -> object: ...


class _InferenceModel(Protocol):
    def inference(self, text: str, output_audio_path: Path, **kwargs: object) -> object: ...


class _AttentionModel(Protocol):
    def _set_attention_implementation(self, value: str) -> None: ...


@dataclass(frozen=True)
class MossTTSConfig:
    model: str = DEFAULT_MODEL
    sample_rate: int = DEFAULT_SAMPLE_RATE
    speaker: str | None = None
    language: str | None = None
    revision: str | None = None
    trust_remote_code: bool = False
    load_kwargs: Mapping[str, object] = field(default_factory=dict)
    runtime_kwargs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model:
            raise ValueError("model must be non-empty.")
        if isinstance(self.sample_rate, bool) or not isinstance(self.sample_rate, int):
            raise TypeError("sample_rate must be an integer.")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        _validate_kwargs(self.load_kwargs, "load_kwargs")
        _validate_kwargs(self.runtime_kwargs, "runtime_kwargs")

    def options(self) -> TTSOptions:
        return TTSOptions(
            speaker=self.speaker,
            language=self.language,
            sample_rate=self.sample_rate,
            extra=self.runtime_kwargs,
        )

    def hash_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "sample_rate": self.sample_rate,
            "speaker": self.speaker,
            "language": self.language,
            "revision": self.revision,
            "trust_remote_code": self.trust_remote_code,
            "load_kwargs": _hashable_mapping(self.load_kwargs),
            "runtime_kwargs": _hashable_mapping(self.runtime_kwargs),
        }


@dataclass
class MossTTS:
    model: Any
    tokenizer: Any | None = None
    config: MossTTSConfig = field(default_factory=MossTTSConfig)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    model_loaded_from: str = "custom"

    name: str = "moss-tts"

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)
        if isinstance(self.model, nn.Module):
            self.model = self.model.to(self.device)

    @classmethod
    def from_pretrained(
        cls,
        model: str | Path = DEFAULT_MODEL,
        *,
        cache_dir: str | Path | None = None,
        device: str | torch.device | None = None,
        local_files_only: bool = False,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        speaker: str | None = None,
        language: str | None = None,
        revision: str | None = None,
        trust_remote_code: bool = False,
        runtime_kwargs: Mapping[str, object] | None = None,
        **load_kwargs: Unpack[MossLoadKwargs],
    ) -> MossTTS:
        config = MossTTSConfig(
            model=str(model),
            sample_rate=sample_rate,
            speaker=speaker,
            language=language,
            revision=revision,
            trust_remote_code=trust_remote_code,
            load_kwargs=load_kwargs,
            runtime_kwargs={} if runtime_kwargs is None else dict(runtime_kwargs),
        )
        pretrained_kwargs = _pretrained_kwargs(
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            revision=revision,
            trust_remote_code=trust_remote_code,
            load_kwargs=load_kwargs,
        )
        try:
            auto_model_cls = load_transformers_auto_model_class()
        except ImportError as exc:
            raise ImportError(
                "`anytrain.tts.moss` requires `transformers` to load remote-code "
                "MOSS-TTS checkpoints. Install Moss TTS dependencies with "
                "`pip install anytrain[moss-tts]`."
            ) from exc
        try:
            loaded = auto_model_cls.from_pretrained(str(model), **pretrained_kwargs)
        except Exception as exc:
            raise RuntimeError(f"failed to load Hugging Face MOSS-TTS checkpoint {model!s}") from exc
        _set_attention_implementation(loaded, load_kwargs.get("attn_implementation", "sdpa"))
        resolved_device = _resolve_device(device)
        if isinstance(loaded, nn.Module):
            loaded = loaded.to(resolved_device)
        return cls(
            model=loaded,
            config=config,
            device=resolved_device,
            model_loaded_from="transformers",
        )

    @property
    def sample_rate(self) -> int:
        return self.config.sample_rate

    def config_hash(self) -> str:
        encoded = json.dumps(self.config.hash_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @torch.no_grad()
    def tokenize(
        self,
        text: str,
        options: TTSOptions | None = None,
        **runtime_kwargs: Unpack[MossRuntimeKwargs],
    ) -> TTSTokens:
        text = validate_text(text)
        resolved = self._options(options, runtime_kwargs)
        kwargs = _conditioning_kwargs(resolved)
        if self.tokenizer is not None:
            raw = self.tokenizer(text, return_tensors="pt", **kwargs)
        else:
            raw = self.model.tokenize(text, **kwargs)
        return _normalize_tokens(raw).to(self.device)

    @torch.no_grad()
    def generate(
        self,
        tokens: TTSTokens,
        options: TTSOptions | None = None,
        **runtime_kwargs: Unpack[MossRuntimeKwargs],
    ) -> TTSGeneration:
        resolved = self._options(options, runtime_kwargs)
        model_inputs = tokens.to(self.device).model_inputs()
        with _seeded_rng(resolved.seed, self.device):
            raw = self.model.generate(**model_inputs, **_generation_kwargs(resolved))
        if isinstance(raw, TTSGeneration):
            return raw
        return TTSGeneration(
            value=raw,
            sample_rate=resolved.sample_rate or self.sample_rate,
            meta={"backend": self.name},
        )

    @torch.no_grad()
    def decode(
        self,
        generation: TTSGeneration,
        options: TTSOptions | None = None,
        **runtime_kwargs: Unpack[MossRuntimeKwargs],
    ) -> TTSOutput:
        resolved = self._options(options, runtime_kwargs)
        if isinstance(generation.value, TTSOutput) or (
            isinstance(generation.value, Mapping) and _has_waveform(generation.value)
        ):
            raw = generation.value
        else:
            decoder = _decode_model(self.model)
            raw = generation.value if decoder is None else decoder.decode(
                generation.value,
                **_conditioning_kwargs(resolved),
            )
        return _normalize_output(
            raw,
            sample_rate=resolved.sample_rate or generation.sample_rate or self.sample_rate,
        )

    @torch.no_grad()
    def synthesize(
        self,
        text: str,
        options: TTSOptions | None = None,
        **runtime_kwargs: Unpack[MossRuntimeKwargs],
    ) -> TTSOutput:
        text = validate_text(text)
        inference = _inference_model(self.model)
        if inference is not None:
            return self._synthesize_with_inference(inference, text, options, runtime_kwargs)
        tokens = self.tokenize(text, options, **runtime_kwargs)
        generation = self.generate(tokens, options, **runtime_kwargs)
        return self.decode(generation, options, **runtime_kwargs)

    def _options(
        self,
        options: TTSOptions | None,
        runtime_kwargs: Mapping[str, object] | None = None,
    ) -> TTSOptions:
        config_options = self.config.options()
        resolved = config_options.merged(options)
        if runtime_kwargs:
            resolved = resolved.merged(TTSOptions(extra={**resolved.extra, **runtime_kwargs}))
        return resolved

    def _synthesize_with_inference(
        self,
        inference: _InferenceModel,
        text: str,
        options: TTSOptions | None,
        runtime_kwargs: Mapping[str, object],
    ) -> TTSOutput:
        resolved = self._options(options, runtime_kwargs)
        output_path, cleanup = _inference_output_path(resolved)
        kwargs = _inference_kwargs(resolved, self.device)
        try:
            with _seeded_rng(resolved.seed, self.device):
                raw = inference.inference(
                    text=text,
                    output_audio_path=output_path,
                    **kwargs,
                )
            return _normalize_inference_output(
                raw,
                sample_rate=resolved.sample_rate or self.sample_rate,
                output_audio_path=output_path,
            )
        finally:
            if cleanup:
                Path(output_path).unlink(missing_ok=True)


def _pretrained_kwargs(
    *,
    cache_dir: str | Path | None,
    local_files_only: bool,
    revision: str | None,
    trust_remote_code: bool,
    load_kwargs: Mapping[str, object],
) -> dict[str, object]:
    kwargs: dict[str, object] = {"local_files_only": local_files_only}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if revision is not None:
        kwargs["revision"] = revision
    if trust_remote_code:
        kwargs["trust_remote_code"] = trust_remote_code
    kwargs.update(load_kwargs)
    return kwargs


def _conditioning_kwargs(options: TTSOptions) -> dict[str, object]:
    kwargs = dict(options.extra)
    if options.speaker is not None:
        kwargs["speaker"] = options.speaker
    if options.language is not None:
        kwargs["language"] = options.language
    if options.sample_rate is not None:
        kwargs["sample_rate"] = options.sample_rate
    return kwargs


def _generation_kwargs(options: TTSOptions) -> dict[str, object]:
    kwargs = _conditioning_kwargs(options)
    if options.max_new_tokens is not None:
        kwargs["max_new_tokens"] = options.max_new_tokens
    if options.temperature is not None:
        kwargs["temperature"] = options.temperature
    if options.top_p is not None:
        kwargs["top_p"] = options.top_p
    return kwargs


def _inference_output_path(options: TTSOptions) -> tuple[Path, bool]:
    value = options.extra.get("output_audio_path")
    if value is not None:
        return Path(value), False
    with tempfile.NamedTemporaryFile(
        prefix="anytrain-moss-tts-",
        suffix=".wav",
        delete=False,
    ) as handle:
        return Path(handle.name), True


def _inference_kwargs(options: TTSOptions, device: torch.device) -> dict[str, object]:
    kwargs = dict(options.extra)
    kwargs.pop("output_audio_path", None)
    has_prompt_audio = kwargs.get("prompt_audio_path") is not None or kwargs.get("reference_audio_path") is not None
    kwargs.setdefault("mode", "voice_clone" if has_prompt_audio else "continuation")
    kwargs.setdefault("device", device)
    if options.max_new_tokens is not None:
        kwargs.setdefault("max_new_frames", options.max_new_tokens)
    if options.temperature is not None:
        kwargs.setdefault("do_sample", True)
        kwargs.setdefault("text_temperature", options.temperature)
        kwargs.setdefault("audio_temperature", options.temperature)
    if options.top_p is not None:
        kwargs.setdefault("text_top_p", options.top_p)
        kwargs.setdefault("audio_top_p", options.top_p)
    return kwargs


def _set_attention_implementation(model: object, value: object) -> None:
    attention = _attention_model(model)
    if attention is not None:
        attention._set_attention_implementation(str(value))


@contextmanager
def _seeded_rng(seed: int | None, device: torch.device) -> Iterator[None]:
    if seed is None:
        yield
        return
    cpu_state = torch.random.get_rng_state()
    cuda_states = _cuda_rng_states()
    try:
        torch.manual_seed(seed)
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        yield
    finally:
        torch.random.set_rng_state(cpu_state)
        _restore_cuda_rng_states(cuda_states)


def _cuda_rng_states() -> list[Tensor]:
    if not torch.cuda.is_available():
        return []
    return torch.cuda.get_rng_state_all()


def _restore_cuda_rng_states(states: Sequence[Tensor]) -> None:
    if states and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(states))


def _decode_model(model: object) -> _DecodeModel | None:
    decode = getattr(model, "decode", None)
    return cast(_DecodeModel, model) if callable(decode) else None


def _inference_model(model: object) -> _InferenceModel | None:
    inference = getattr(model, "inference", None)
    return cast(_InferenceModel, model) if callable(inference) else None


def _attention_model(model: object) -> _AttentionModel | None:
    setter = getattr(model, "_set_attention_implementation", None)
    return cast(_AttentionModel, model) if callable(setter) else None


def _validate_kwargs(value: Mapping[str, object], name: str) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping.")
    for key in value:
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings.")


def _hashable_mapping(value: Mapping[str, object]) -> dict[str, object]:
    return {key: _hashable_value(item) for key, item in sorted(value.items())}


def _hashable_value(value: object) -> object:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _hashable_value(item)
            for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
        }
    if isinstance(value, tuple | list):
        return [_hashable_value(item) for item in value]
    return str(value)


def _normalize_inference_output(
    value: object,
    *,
    sample_rate: int,
    output_audio_path: Path,
) -> TTSOutput:
    if isinstance(value, TTSOutput):
        return value
    if isinstance(value, Tensor):
        return _normalize_output(value, sample_rate=sample_rate)
    if isinstance(value, Mapping):
        if _has_waveform(value):
            raw = dict(value)
            raw["meta"] = _inference_meta(value, output_audio_path=output_audio_path)
            return _normalize_output(raw, sample_rate=sample_rate)
        if output_audio_path.exists():
            output_sample_rate = int(value.get("sample_rate", sample_rate))
            waveform, loaded_sample_rate = _load_audio_output(output_audio_path)
            if loaded_sample_rate != output_sample_rate:
                output_sample_rate = loaded_sample_rate
            return TTSOutput(
                waveform=waveform,
                sample_rate=output_sample_rate,
                duration=waveform_duration(waveform, output_sample_rate),
                meta=_inference_meta(value, output_audio_path=output_audio_path),
            )
    return _normalize_output(value, sample_rate=sample_rate)


def _inference_meta(value: Mapping[object, object], *, output_audio_path: Path) -> dict[str, object]:
    raw_meta = value.get("meta", {})
    if raw_meta is None:
        meta: dict[str, object] = {}
    elif isinstance(raw_meta, Mapping):
        meta = {str(key): item for key, item in raw_meta.items()}
    else:
        raise TypeError("output meta must be a mapping when set.")
    meta.setdefault("backend", DEFAULT_MODEL)
    audio_path = value.get("audio_path", output_audio_path)
    meta["audio_path"] = str(audio_path)
    audio_token_ids = value.get("audio_token_ids")
    if isinstance(audio_token_ids, Tensor) and audio_token_ids.ndim >= 1:
        meta["audio_token_frames"] = int(audio_token_ids.shape[0])
    return meta


def _load_audio_output(path: Path) -> tuple[Tensor, int]:
    try:
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "MOSS-TTS inference did not return a waveform; install `torchaudio` to load "
            f"the generated file at {path}."
        ) from exc
    waveform, sample_rate = torchaudio.load(str(path))
    return _normalize_waveform(waveform), int(sample_rate)


def _normalize_tokens(value: object) -> TTSTokens:
    if isinstance(value, TTSTokens):
        return value
    if isinstance(value, Tensor):
        return TTSTokens(input_ids=value)
    if isinstance(value, Mapping):
        tensors = _tensor_mapping(value)
        try:
            input_ids = tensors.pop("input_ids")
        except KeyError as exc:
            raise KeyError("tokenizer output must include `input_ids`.") from exc
        attention_mask = tensors.pop("attention_mask", None)
        return TTSTokens(
            input_ids=input_ids,
            attention_mask=attention_mask,
            extra_tensors=tensors,
        )
    raise TypeError("tokenizer output must be TTSTokens, a Tensor, or a mapping of tensors.")


def _normalize_output(value: object, *, sample_rate: int) -> TTSOutput:
    if isinstance(value, TTSOutput):
        return value
    if isinstance(value, Tensor):
        waveform = _normalize_waveform(value)
        return TTSOutput(
            waveform=waveform,
            sample_rate=sample_rate,
            duration=waveform_duration(waveform, sample_rate),
        )
    if isinstance(value, Mapping):
        waveform = _normalize_waveform(_waveform_value(value))
        output_sample_rate = int(value.get("sample_rate", sample_rate))
        duration = float(value.get("duration", waveform_duration(waveform, output_sample_rate)))
        meta = value.get("meta", {})
        if not isinstance(meta, Mapping):
            raise TypeError("output meta must be a mapping when set.")
        return TTSOutput(
            waveform=waveform,
            sample_rate=output_sample_rate,
            duration=duration,
            meta=meta,
        )
    raise TypeError("decoder output must be TTSOutput, a Tensor, or a mapping.")


def _normalize_waveform(waveform: Tensor) -> Tensor:
    if not isinstance(waveform, Tensor):
        raise TypeError("waveform must be a torch.Tensor.")
    wave = waveform.detach()
    wave = wave.float() if not torch.is_floating_point(wave) else wave.to(dtype=torch.float32)
    if wave.ndim == 1:
        wave = wave.unsqueeze(0)
    if wave.ndim != 2:
        raise ValueError("waveform must have shape [time] or [channels, time].")
    return wave.contiguous()


def _tensor_mapping(value: Mapping[object, object]) -> dict[str, Tensor]:
    tensors: dict[str, Tensor] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("tokenizer output keys must be non-empty strings.")
        if not isinstance(item, Tensor):
            raise TypeError(f"tokenizer output {key!r} must be a torch.Tensor.")
        tensors[key] = item
    return tensors


def _waveform_value(value: Mapping[object, object]) -> Tensor:
    for key in ("waveform", "audio"):
        item = value.get(key)
        if isinstance(item, Tensor):
            return item
    raise KeyError("decoder output mapping must include `waveform` or `audio`.")


def _has_waveform(value: Mapping[object, object]) -> bool:
    return isinstance(value.get("waveform"), Tensor) or isinstance(value.get("audio"), Tensor)


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
