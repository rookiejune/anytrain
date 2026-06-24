from __future__ import annotations

from typing import Any

from ..abc import EvaluatorABC, MetricDict
from ..text import TextInput
from .asr import WhisperASREvaluator
from .utmos import UTMOSEvaluator


class SpeechEvaluator(EvaluatorABC):
    def __init__(
        self,
        *,
        asr: WhisperASREvaluator | None = None,
        utmos: UTMOSEvaluator | None = None,
    ) -> None:
        super().__init__()
        self.asr = self._resolve_asr(asr)
        self.utmos = self._resolve_utmos(utmos)

    def evaluate(
        self,
        audio: Any,
        sample_rate: int,
        reference_text: TextInput | None = None,
        *,
        target_text: TextInput | None = None,
        **decode_options: Any,
    ) -> MetricDict:
        metrics = self.asr(
            audio,
            sample_rate,
            reference_text=reference_text,
            target_text=target_text,
            **decode_options,
        )
        for name, value in self.utmos(audio, sample_rate).items():
            if name in metrics:
                raise ValueError(f"Duplicate speech metric key {name!r}.")
            metrics[name] = value
        return metrics

    def _resolve_asr(self, asr: WhisperASREvaluator | None) -> WhisperASREvaluator:
        if asr is None:
            return WhisperASREvaluator()
        if not isinstance(asr, WhisperASREvaluator):
            raise TypeError("asr must be a WhisperASREvaluator.")
        return asr

    def _resolve_utmos(self, utmos: UTMOSEvaluator | None) -> UTMOSEvaluator:
        if utmos is None:
            return UTMOSEvaluator()
        if not isinstance(utmos, UTMOSEvaluator):
            raise TypeError("utmos must be a UTMOSEvaluator.")
        return utmos
