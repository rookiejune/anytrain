from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from ._deps import MixturePathGeneralizedKL
from .continuous import ContinuousFlowRuntime
from .discrete import DiscreteFlowRuntime
from .types import FlowLossFn, ModelExtras


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
        runtime: ContinuousFlowRuntime,
        *,
        loss_fn: FlowLossFn = mse_velocity_loss,
    ):
        super().__init__()
        self.runtime = runtime
        self.loss_fn = loss_fn

    def forward(
        self,
        model: nn.Module,
        x_1: Tensor,
        x_0: Tensor | None = None,
        **model_extras: object,
    ) -> Tensor:
        sample = self.runtime.training_sample(x_1, x_0=x_0)
        prediction = self.runtime.call_model(model, sample.x_t, sample.t, model_extras)
        return _require_scalar_loss(self.loss_fn(prediction, sample.velocity, model_extras))


class DiscreteGeneralizedKLObjective(nn.Module):
    def __init__(
        self,
        runtime: DiscreteFlowRuntime,
    ):
        super().__init__()
        self.runtime = runtime
        self.loss_fn = MixturePathGeneralizedKL(runtime.path)

    def forward(
        self,
        model: nn.Module,
        x_1: Tensor,
        x_0: Tensor | None = None,
        **model_extras: object,
    ) -> Tensor:
        sample = self.runtime.training_sample(x_1, x_0=x_0)
        logits = self.runtime.call_model(model, sample.x_t, sample.t, model_extras)
        return self.loss_fn(
            logits=logits,
            x_1=x_1,
            x_t=sample.x_t,
            t=sample.t,
        )


__all__ = [
    "ContinuousVelocityObjective",
    "DiscreteGeneralizedKLObjective",
    "mse_velocity_loss",
]
