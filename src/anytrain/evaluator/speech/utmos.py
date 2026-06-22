from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from anytrain.evaluator.abc import EvaluatorABC, MetricDict


@runtime_checkable
class UTMOSBackendProtocol(Protocol):
    def score(self, audio: Any, sample_rate: int) -> float | Sequence[float]:
        raise NotImplementedError


class UTMOSEvaluator(EvaluatorABC):
    def __init__(self, *, backend: UTMOSBackendProtocol | None = None) -> None:
        super().__init__()
        self.backend = self._validate_backend(backend)

    def evaluate(self, audio: Any, sample_rate: int) -> MetricDict:
        sample_rate = self._validate_sample_rate(sample_rate)
        score = self.backend.score(audio, sample_rate)
        scores = self._normalize_scores(score)
        return {"utmos": sum(scores) / len(scores)}

    def _validate_backend(self, backend: UTMOSBackendProtocol | None) -> UTMOSBackendProtocol:
        if backend is None:
            raise ValueError(
                "UTMOSEvaluator requires an explicit backend implementing "
                "score(audio, sample_rate). anytrain does not load or download UTMOS models."
            )
        if not isinstance(backend, UTMOSBackendProtocol):
            raise TypeError("UTMOSEvaluator backend must implement score(audio, sample_rate).")
        return backend

    def _normalize_scores(self, score: float | Sequence[float]) -> list[float]:
        if self._is_number(score):
            return [float(score)]

        if isinstance(score, bytes | bytearray | str) or not isinstance(score, Sequence):
            raise TypeError("UTMOS backend score must be a float or a sequence of floats.")

        scores = list(score)
        if not scores:
            raise ValueError("UTMOS backend score sequence must contain at least one value.")

        for index, value in enumerate(scores):
            if not self._is_number(value):
                raise TypeError(f"UTMOS backend score[{index}] must be a float.")
        return [float(value) for value in scores]

    def _validate_sample_rate(self, sample_rate: int) -> int:
        if isinstance(sample_rate, bool) or not isinstance(sample_rate, int):
            raise TypeError("sample_rate must be an integer.")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive.")
        return sample_rate

    def _is_number(self, value: object) -> bool:
        return not isinstance(value, bool) and isinstance(value, float | int)
