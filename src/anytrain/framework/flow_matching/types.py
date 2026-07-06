from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor, nn


class Source(Protocol):
    def sample_like(self, x_1: Tensor) -> Tensor: ...


class TimeSampler(Protocol):
    def sample(self, batch_size: int, device: torch.device) -> Tensor: ...


ModelExtras = Mapping[str, object]
ModelCaller = Callable[[nn.Module, Tensor, Tensor, ModelExtras], Tensor]
FlowLossFn = Callable[[Tensor, Tensor, ModelExtras], Tensor]


@dataclass(eq=False)
class FlowSampleOutput:
    final: Tensor
    states: Tensor | None = None
    time_grid: Tensor | None = None


@dataclass(eq=False)
class ContinuousTrainingSample:
    x_t: Tensor
    t: Tensor
    velocity: Tensor


def default_call_model(
    model: nn.Module,
    x_t: Tensor,
    t: Tensor,
    extras: ModelExtras,
) -> Tensor:
    return model(x_t, t, **extras)


__all__ = [
    "ContinuousTrainingSample",
    "FlowSampleOutput",
    "FlowLossFn",
    "ModelCaller",
    "ModelExtras",
    "Source",
    "TimeSampler",
    "default_call_model",
]
