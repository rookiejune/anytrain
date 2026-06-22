from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from torch import nn

from .abc import EvaluatorABC, MetricDict


class EvaluatorGroup(nn.Module):
    metric_key_separator = EvaluatorABC.metric_key_separator

    def __init__(
        self,
        metrics: Mapping[str, EvaluatorABC],
    ) -> None:
        super().__init__()
        self.metrics = cast(dict[str, EvaluatorABC], nn.ModuleDict(self._validate_metrics(metrics)))

    def forward(self, *args: Any, **kwargs: Any) -> MetricDict:
        output: MetricDict = {}
        for name, metric in self.metrics.items():
            metrics = metric(*args, **kwargs)
            self._merge_metric_output(output, name, metrics)
        return output

    def update(self, *args: Any, **kwargs: Any) -> None:
        for metric in self.metrics.values():
            metric.update(*args, **kwargs)

    def compute(self) -> MetricDict:
        output: MetricDict = {}
        for name, metric in self.metrics.items():
            metrics = metric.compute()
            self._merge_metric_output(output, name, metrics)
        return output

    def reset(self) -> None:
        for metric in self.metrics.values():
            metric.reset()

    def _validate_metrics(
        self,
        metrics: Mapping[str, EvaluatorABC],
    ) -> dict[str, EvaluatorABC]:
        if not isinstance(metrics, Mapping):
            raise TypeError("metrics must be a mapping of names to evaluators.")

        if not metrics:
            raise ValueError("metrics must contain at least one evaluator.")

        validated: dict[str, EvaluatorABC] = {}
        for raw_name, metric in metrics.items():
            name = self._validate_name(raw_name, "metric name")
            if not isinstance(metric, EvaluatorABC):
                raise TypeError(f"Metric {name!r} must inherit EvaluatorABC.")
            validated[name] = metric
        return validated

    def _merge_metric_output(
        self,
        output: MetricDict,
        metric_name: str,
        metrics: object,
    ) -> None:
        validated = self._validate_metric_dict(metrics)
        for key, value in validated.items():
            name = self.metric_key_separator.join([metric_name, key])
            if name in output:
                raise ValueError(f"Duplicate metric key {name!r}.")
            output[name] = value

    def _validate_metric_dict(self, metrics: object) -> MetricDict:
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict of string keys to metric values.")
        if not metrics:
            raise ValueError("metrics must contain at least one value.")
        validated: MetricDict = {}
        for key, value in metrics.items():
            key = self._validate_name(key, "metric key")
            validated[key] = value
        return validated

    def _validate_name(self, name: str, label: str) -> str:
        if not isinstance(name, str):
            raise TypeError(f"{label} must be a string.")
        if self.metric_key_separator in name:
            raise ValueError(f"{label} must not contain separator {self.metric_key_separator!r}.")
        return name
