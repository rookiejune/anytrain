from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._deps import (
    CondOTProbPath,
    MixtureDiscreteProbPath,
    MixturePathGeneralizedKL,
    PolynomialConvexScheduler,
    ProbPath,
)
from .source import GaussianSource, UniformTokenSource
from .time import LogitNormalTimeSampler
from .types import ModelCaller, Source, TimeSampler, default_call_model


def _require_batch(x: Tensor, name: str) -> None:
    if x.ndim == 0:
        raise ValueError(f"{name} must include a batch dimension.")
    if x.shape[0] <= 0:
        raise ValueError(f"{name} batch size must be positive.")


class ContinuousVelocityObjective(nn.Module):
    def __init__(
        self,
        *,
        path: ProbPath | None = None,
        source: Source | None = None,
        time_sampler: TimeSampler | None = None,
        call_model: ModelCaller = default_call_model,
    ):
        super().__init__()
        self.path = CondOTProbPath() if path is None else path
        self.source = GaussianSource() if source is None else source
        self.time_sampler = LogitNormalTimeSampler() if time_sampler is None else time_sampler
        self.call_model = call_model

    def forward(
        self,
        model: nn.Module,
        x_1: Tensor,
        x_0: Tensor | None = None,
        **model_extras: object,
    ) -> Tensor:
        _require_batch(x_1, "x_1")
        if x_0 is None:
            x_0 = self.source.sample_like(x_1)
        if x_0.shape != x_1.shape:
            raise ValueError(f"x_0 and x_1 must have the same shape, got {x_0.shape} and {x_1.shape}.")

        t = self.time_sampler.sample(x_1.shape[0], x_1.device)
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        prediction = self.call_model(model, path_sample.x_t, path_sample.t, model_extras)
        return F.mse_loss(prediction, path_sample.dx_t)


class DiscreteGeneralizedKLObjective(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        path: ProbPath | None = None,
        source: Source | None = None,
        time_sampler: TimeSampler | None = None,
        call_model: ModelCaller = default_call_model,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")

        self.vocab_size = vocab_size
        self.path = (
            MixtureDiscreteProbPath(PolynomialConvexScheduler(n=2.0)) if path is None else path
        )
        self.source = UniformTokenSource(vocab_size) if source is None else source
        self.time_sampler = LogitNormalTimeSampler() if time_sampler is None else time_sampler
        self.call_model = call_model
        self.loss_fn = MixturePathGeneralizedKL(self.path)

    def forward(
        self,
        model: nn.Module,
        x_1: Tensor,
        x_0: Tensor | None = None,
        **model_extras: object,
    ) -> Tensor:
        _require_batch(x_1, "x_1")
        if x_1.dtype != torch.long:
            x_1 = x_1.long()
        x_0 = self.source.sample_like(x_1) if x_0 is None else x_0.long()
        if x_0.shape != x_1.shape:
            raise ValueError(f"x_0 and x_1 must have the same shape, got {x_0.shape} and {x_1.shape}.")

        t = self.time_sampler.sample(x_1.shape[0], x_1.device)
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        logits = self.call_model(model, path_sample.x_t, path_sample.t, model_extras)
        return self.loss_fn.forward(
            logits=logits,
            x_1=x_1,
            x_t=path_sample.x_t,
            t=path_sample.t,
        )


__all__ = [
    "ContinuousVelocityObjective",
    "DiscreteGeneralizedKLObjective",
]
