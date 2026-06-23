from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

DEFAULT_TIME_EPS = 1e-3


@dataclass(frozen=True)
class LogitNormalTimeSampler:
    mean: float = 0.0
    std: float = 1.0
    t_min: float = 0.0
    t_max: float = 1.0

    def __post_init__(self) -> None:
        if self.std <= 0:
            raise ValueError(f"std must be positive, got {self.std}.")
        if self.t_min < 0:
            raise ValueError(f"t_min must be non-negative, got {self.t_min}.")
        if self.t_max > 1:
            raise ValueError(f"t_max must be at most 1, got {self.t_max}.")
        if self.t_max <= self.t_min:
            raise ValueError(
                f"t_max must be greater than t_min, got t_min={self.t_min}, t_max={self.t_max}."
            )

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        z = torch.randn(batch_size, device=device) * self.std + self.mean
        t = torch.sigmoid(z)
        return t * (self.t_max - self.t_min) + self.t_min


@dataclass(frozen=True)
class UniformTimeSampler:
    t_min: float = 0.0
    t_max: float = 1.0

    def __post_init__(self) -> None:
        if self.t_min < 0:
            raise ValueError(f"t_min must be non-negative, got {self.t_min}.")
        if self.t_max > 1:
            raise ValueError(f"t_max must be at most 1, got {self.t_max}.")
        if self.t_max <= self.t_min:
            raise ValueError(
                f"t_max must be greater than t_min, got t_min={self.t_min}, t_max={self.t_max}."
            )

    def sample(self, batch_size: int, device: torch.device) -> Tensor:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")

        t = torch.rand(batch_size, device=device)
        return t * (self.t_max - self.t_min) + self.t_min


__all__ = ["DEFAULT_TIME_EPS", "LogitNormalTimeSampler", "UniformTimeSampler"]
