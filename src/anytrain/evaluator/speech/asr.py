from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ...env import whisper_root
from ..abc import EvaluatorABC, MetricDict
from ..text import TextComparisonEvaluator, TextInput
from ..text.normalization import coerce_text_batch
from ._torch import freeze_model
from .audio import load_wave_batch, resample_wave, validate_sample_rate

_DEFAULT_MODEL_NAME = "large-v3"
_SUPPORTED_MODEL_NAMES = (
    "tiny.en",
    "tiny",
    "base.en",
    "base",
    "small.en",
    "small",
    "medium.en",
    "medium",
    "large-v1",
    "large-v2",
    "large-v3",
    "large",
    "large-v3-turbo",
    "turbo",
)
_SUPPORTED_MODEL_NAME_SET = frozenset(_SUPPORTED_MODEL_NAMES)


class _OpenAIWhisperBackend:
    target_sample_rate = 16000

    def __init__(
        self,
        *,
        model_name: str = _DEFAULT_MODEL_NAME,
        device: Any | None = None,
        download_root: str | Path | None = None,
        load_options: Mapping[str, Any] | None = None,
    ) -> None:
        self.model_name = self._validate_model_name(model_name)
        self.device = device
        root = Path(download_root).expanduser() if download_root is not None else whisper_root()
        self.download_root = str(root)
        self.model: Any | None = None
        self.load_options = self._validate_load_options(load_options)

    def transcribe(
        self,
        audio: Any,
        sample_rate: int,
        **decode_options: Any,
    ) -> TextInput:
        wave, sample_rate = load_wave_batch(audio, sample_rate)
        wave = resample_wave(wave, sample_rate, self.target_sample_rate)
        model = self._prepare_model(self._load_model())

        with torch.inference_mode():
            predictions = [
                self._extract_text(model.transcribe(sample.cpu().numpy(), **decode_options))
                for sample in wave
            ]
        return predictions[0] if len(predictions) == 1 else predictions

    def _load_model(self) -> Any:
        if self.model is not None:
            return self.model

        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "WhisperASREvaluator requires the `openai-whisper` package. "
                "Install speech dependencies with `pip install anytrain[speech]`."
            ) from exc

        load_options = dict(self.load_options)
        if self.device is not None:
            load_options["device"] = self.device
        if self.download_root is not None:
            load_options["download_root"] = self.download_root
        self.model = whisper.load_model(self.model_name, **load_options)
        return self.model

    def _prepare_model(self, model: Any) -> Any:
        model = freeze_model(model, device=self.device)
        self.model = model
        return model

    def _extract_text(self, result: object) -> str:
        if isinstance(result, str):
            return result
        if isinstance(result, Mapping):
            text = result.get("text")
            if isinstance(text, str):
                return text
            raise TypeError("Whisper transcribe result must contain a string `text` value.")
        raise TypeError("Whisper transcribe result must be a string or mapping.")

    def _validate_model_name(self, model_name: str) -> str:
        if not isinstance(model_name, str):
            raise TypeError("model_name must be a string.")
        if model_name not in _SUPPORTED_MODEL_NAME_SET:
            names = ", ".join(_SUPPORTED_MODEL_NAMES)
            raise ValueError(f"model_name must be one of: {names}.")
        return model_name

    def _validate_load_options(
        self,
        load_options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if load_options is None:
            return {}
        if not isinstance(load_options, Mapping):
            raise TypeError("load_options must be a mapping.")
        return dict(load_options)


class WhisperASREvaluator(EvaluatorABC):
    required_metric_keys = ("bleu", "wer", "chrf")
    default_model_name = _DEFAULT_MODEL_NAME
    supported_model_names = _SUPPORTED_MODEL_NAMES

    def __init__(
        self,
        *,
        text_evaluator: TextComparisonEvaluator | None = None,
        decode_options: Mapping[str, Any] | None = None,
        model_name: str = default_model_name,
        device: Any | None = None,
        download_root: str | Path | None = None,
        load_options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.device = device
        self._backend = _OpenAIWhisperBackend(
            model_name=model_name,
            device=device,
            download_root=download_root,
            load_options=load_options,
        )
        self.download_root = self._backend.download_root
        self.text_evaluator = self._resolve_text_evaluator(text_evaluator)
        self.decode_options = self._validate_decode_options(decode_options)

    def evaluate(
        self,
        audio: Any,
        sample_rate: int,
        reference_text: TextInput | None = None,
        *,
        target_text: TextInput | None = None,
        **decode_options: Any,
    ) -> MetricDict:
        reference_text = self._resolve_reference_text(reference_text, target_text)
        prediction_text = self.transcribe(audio, sample_rate, **decode_options)
        predictions = self._normalize_text_input(prediction_text, label="prediction_text")
        references = self._normalize_text_input(reference_text, label="reference_text")

        if len(predictions) != len(references):
            raise ValueError(
                "prediction/reference text counts must match: "
                f"got {len(predictions)} predictions and {len(references)} references."
            )

        metrics = self.text_evaluator.evaluate(predictions, references)
        return self._extract_required_metrics(metrics)

    def transcribe(self, audio: Any, sample_rate: int, **decode_options: Any) -> TextInput:
        sample_rate = validate_sample_rate(sample_rate)
        options = dict(self.decode_options)
        options.update(decode_options)
        output = self._backend.transcribe(audio, sample_rate, **options)
        if isinstance(output, str):
            return output
        return self._normalize_text_input(output, label="prediction_text")

    def _resolve_text_evaluator(
        self,
        text_evaluator: TextComparisonEvaluator | None,
    ) -> TextComparisonEvaluator:
        if text_evaluator is None:
            return TextComparisonEvaluator()
        if not isinstance(text_evaluator, TextComparisonEvaluator):
            raise TypeError("text_evaluator must be a TextComparisonEvaluator.")
        return text_evaluator

    def _resolve_reference_text(
        self,
        reference_text: TextInput | None,
        target_text: TextInput | None,
    ) -> TextInput:
        if reference_text is not None and target_text is not None:
            raise ValueError("Provide only one of reference_text or target_text.")
        resolved = reference_text if reference_text is not None else target_text
        if resolved is None:
            raise ValueError(
                "reference_text is required because WhisperASREvaluator.evaluate() "
                "returns only numeric metrics, not transcription text."
            )
        return resolved

    def _extract_required_metrics(self, metrics: object) -> MetricDict:
        if not isinstance(metrics, dict):
            raise TypeError("text_evaluator.evaluate(...) must return a metric dict.")

        missing = [key for key in self.required_metric_keys if key not in metrics]
        if missing:
            missing_names = ", ".join(missing)
            raise ValueError(
                "text_evaluator.evaluate(...) must return bleu, wer, and chrf metrics; "
                f"missing: {missing_names}."
            )

        return {key: metrics[key] for key in self.required_metric_keys}

    def _normalize_text_input(self, value: TextInput, *, label: str) -> list[str]:
        return list(coerce_text_batch(value, name=label, allow_empty=False))

    def _validate_decode_options(
        self,
        decode_options: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        if decode_options is None:
            return {}
        if not isinstance(decode_options, Mapping):
            raise TypeError("decode_options must be a mapping.")
        return dict(decode_options)
