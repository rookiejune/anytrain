from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
from torch import nn

type MetricValue = float | torch.Tensor
type MetricDict = dict[str, MetricValue]


class EvaluatorABC(nn.Module, ABC):
    metric_key_separator = "/"

    def __init__(self) -> None:
        super().__init__()
        self.metric_values: dict[str, list[MetricValue]] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> MetricDict:
        return self._validate_metric_dict(self.evaluate(*args, **kwargs), detach=False)

    @abstractmethod
    def evaluate(self, *args: Any, **kwargs: Any) -> MetricDict:
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        metrics = self._validate_metric_dict(self.evaluate(*args, **kwargs), detach=True)
        for name, value in metrics.items():
            self.metric_values.setdefault(name, []).append(value)

    def compute(self) -> MetricDict:
        if not self.metric_values:
            raise ValueError("No metric values have been recorded.")
        return {
            name: self._mean_metric_values(name, values)
            for name, values in self.metric_values.items()
        }

    def reset(self) -> None:
        self.metric_values.clear()

    def _validate_metric_dict(self, metrics: object, *, detach: bool) -> MetricDict:
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dict of string keys to metric values.")
        if not metrics:
            raise ValueError("metrics must contain at least one value.")
        return {
            self._validate_metric_name(name): self._validate_metric_value(
                value,
                name=str(name),
                detach=detach,
            )
            for name, value in metrics.items()
        }

    def _validate_metric_value(self, value: object, *, name: str, detach: bool) -> MetricValue:
        if isinstance(value, bool):
            raise TypeError(f"Metric value {name!r} must be a float or 0-d tensor.")
        if isinstance(value, float):
            return value
        if isinstance(value, torch.Tensor):
            if value.ndim != 0:
                raise ValueError(f"Metric value {name!r} must be a 0-d tensor.")
            return value.detach() if detach else value
        raise TypeError(f"Metric value {name!r} must be a float or 0-d tensor.")

    def _validate_metric_name(self, name: object) -> str:
        if not isinstance(name, str):
            raise TypeError("metric key must be a string.")
        if not name:
            raise ValueError("metric key must not be empty.")
        if self.metric_key_separator in name:
            raise ValueError(
                f"metric key must not contain separator {self.metric_key_separator!r}."
            )
        return name

    def _mean_metric_values(self, name: str, values: list[MetricValue]) -> MetricValue:
        if not values:
            raise ValueError(f"Metric {name!r} has no recorded values.")

        tensors = [value for value in values if isinstance(value, torch.Tensor)]
        if not tensors:
            return sum(float(value) for value in values) / len(values)

        reference = tensors[0]
        normalized = [
            value.to(device=reference.device, dtype=reference.dtype)
            if isinstance(value, torch.Tensor)
            else torch.tensor(float(value), device=reference.device, dtype=reference.dtype)
            for value in values
        ]
        return torch.stack(normalized).mean()
