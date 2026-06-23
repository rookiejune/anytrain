from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class GaussianSource:
    def sample_like(self, x_1: Tensor) -> Tensor:
        return torch.randn_like(x_1)


@dataclass(frozen=True)
class UniformSource:
    low: float = 0.0
    high: float = 1.0

    def __post_init__(self) -> None:
        if self.high <= self.low:
            raise ValueError(f"high must be greater than low, got low={self.low}, high={self.high}.")

    def sample_like(self, x_1: Tensor) -> Tensor:
        return torch.empty_like(x_1).uniform_(self.low, self.high)


@dataclass(frozen=True)
class UniformTokenSource:
    vocab_size: int

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}.")

    def sample_like(self, x_1: Tensor) -> Tensor:
        return torch.randint(
            low=0,
            high=self.vocab_size,
            size=x_1.shape,
            device=x_1.device,
            dtype=torch.long,
        )


@dataclass(frozen=True)
class MaskTokenSource:
    mask_id: int

    def __post_init__(self) -> None:
        if self.mask_id < 0:
            raise ValueError(f"mask_id must be non-negative, got {self.mask_id}.")

    def sample_like(self, x_1: Tensor) -> Tensor:
        return torch.full_like(x_1, fill_value=self.mask_id, dtype=torch.long)


__all__ = [
    "GaussianSource",
    "MaskTokenSource",
    "UniformSource",
    "UniformTokenSource",
]
