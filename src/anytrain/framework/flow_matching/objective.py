from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ._deps import (
    CondOTProbPath,
    MixtureDiscreteProbPath,
    MixturePathGeneralizedKL,
    PolynomialConvexScheduler,
)
from .source import GaussianSource, UniformTokenSource
from .time import DEFAULT_TIME_EPS, LogitNormalTimeSampler
from .types import FlowLossFn, ModelCaller, ModelExtras, Source, TimeSampler, default_call_model


def _require_batch(x: Tensor, name: str) -> None:
    if x.ndim == 0:
        raise ValueError(f"{name} must include a batch dimension.")
    if x.shape[0] <= 0:
        raise ValueError(f"{name} batch size must be positive.")


def mse_velocity_loss(
    prediction: Tensor,
    target: Tensor,
    extras: ModelExtras,
) -> Tensor:
    del extras
    return F.mse_loss(prediction, target)


def _require_scalar_loss(loss: Tensor) -> Tensor:
    if loss.ndim != 0:
        raise ValueError("flow matching loss_fn must return a scalar tensor.")
    return loss


class ContinuousVelocityObjective(nn.Module):
    def __init__(
        self,
        *,
        path: CondOTProbPath | None = None,
        source: Source | None = None,
        time_sampler: TimeSampler | None = None,
        call_model: ModelCaller = default_call_model,
        loss_fn: FlowLossFn = mse_velocity_loss,
    ):
        super().__init__()
        self.path = CondOTProbPath() if path is None else path
        self.source = GaussianSource() if source is None else source
        self.time_sampler = LogitNormalTimeSampler() if time_sampler is None else time_sampler
        self.call_model = call_model
        self.loss_fn = loss_fn

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
        return _require_scalar_loss(
            self.loss_fn(prediction, path_sample.dx_t, model_extras)
        )


class DiscreteGeneralizedKLObjective(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        path: MixtureDiscreteProbPath | None = None,
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
        self.time_sampler = (
            LogitNormalTimeSampler(t_max=1.0 - DEFAULT_TIME_EPS)
            if time_sampler is None
            else time_sampler
        )
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
            raise TypeError(f"x_1 must have dtype torch.long, got {x_1.dtype}.")
        if x_0 is None:
            x_0 = self.source.sample_like(x_1)
        elif x_0.dtype != torch.long:
            raise TypeError(f"x_0 must have dtype torch.long, got {x_0.dtype}.")
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
    "mse_velocity_loss",
]
