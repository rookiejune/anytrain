from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from anytrain.evaluator.abc import EvaluatorABC, MetricDict
from anytrain.evaluator.text import TextInput
from anytrain.evaluator.text.normalization import coerce_text_batch


@runtime_checkable
class WhisperASRBackendProtocol(Protocol):
    def transcribe(
        self,
        audio: Any,
        sample_rate: int,
        **decode_options: Any,
    ) -> TextInput:
        raise NotImplementedError


@runtime_checkable
class TextMetricEvaluatorProtocol(Protocol):
    def evaluate(self, prediction_text: TextInput, reference_text: TextInput) -> MetricDict:
        raise NotImplementedError


class WhisperASREvaluator(EvaluatorABC):
    required_metric_keys = ("bleu", "wer", "chrf")

    def __init__(
        self,
        *,
        backend: WhisperASRBackendProtocol | None = None,
        text_evaluator: TextMetricEvaluatorProtocol | None = None,
        decode_options: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.backend = self._validate_backend(backend)
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
        sample_rate = self._validate_sample_rate(sample_rate)
        options = dict(self.decode_options)
        options.update(decode_options)
        output = self.backend.transcribe(audio, sample_rate, **options)
        if isinstance(output, str):
            return output
        return self._normalize_text_input(output, label="prediction_text")

    def _validate_backend(
        self,
        backend: WhisperASRBackendProtocol | None,
    ) -> WhisperASRBackendProtocol:
        if backend is None:
            raise ValueError(
                "WhisperASREvaluator requires an explicit backend implementing "
                "transcribe(audio, sample_rate, **decode_options). anytrain does not "
                "load or download Whisper models."
            )
        if not isinstance(backend, WhisperASRBackendProtocol):
            raise TypeError(
                "WhisperASREvaluator backend must implement "
                "transcribe(audio, sample_rate, **decode_options)."
            )
        return backend

    def _resolve_text_evaluator(
        self,
        text_evaluator: TextMetricEvaluatorProtocol | None,
    ) -> TextMetricEvaluatorProtocol:
        if text_evaluator is None:
            text_evaluator = self._load_default_text_evaluator()
        if not isinstance(text_evaluator, TextMetricEvaluatorProtocol):
            raise TypeError(
                "text_evaluator must implement evaluate(prediction_text, reference_text)."
            )
        return text_evaluator

    def _load_default_text_evaluator(self) -> TextMetricEvaluatorProtocol:
        try:
            from anytrain.evaluator.text import TextComparisonEvaluator
        except ImportError as exc:
            raise ImportError(
                "WhisperASREvaluator requires text_evaluator=... or an available "
                "anytrain.evaluator.text.TextComparisonEvaluator for BLEU/WER/chrF metrics."
            ) from exc

        return TextComparisonEvaluator()

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

    def _validate_sample_rate(self, sample_rate: int) -> int:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an integer.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        return sample_rate
