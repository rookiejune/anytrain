from __future__ import annotations

from torch import Tensor, nn

from ._deps import CondOTProbPath
from .objective import ContinuousVelocityObjective, mse_velocity_loss
from .sampler import ODESampler
from .source import GaussianSource
from .time import LogitNormalTimeSampler
from .types import (
    FlowLossFn,
    FlowSampleOutput,
    ModelCaller,
    Source,
    TimeSampler,
    default_call_model,
)


class ContinuousFlowMatcher(nn.Module):
    def __init__(
        self,
        *,
        path: CondOTProbPath | None = None,
        source: Source | None = None,
        time_sampler: TimeSampler | None = None,
        sampler: ODESampler | None = None,
        call_model: ModelCaller = default_call_model,
        loss_fn: FlowLossFn = mse_velocity_loss,
    ):
        super().__init__()
        self.source = GaussianSource() if source is None else source
        self.time_sampler = LogitNormalTimeSampler() if time_sampler is None else time_sampler
        self.objective = ContinuousVelocityObjective(
            path=path,
            source=self.source,
            time_sampler=self.time_sampler,
            call_model=call_model,
            loss_fn=loss_fn,
        )
        self.sampler = ODESampler(call_model=call_model) if sampler is None else sampler

    def loss(
        self,
        model: nn.Module,
        x_1: Tensor,
        x_0: Tensor | None = None,
        **model_extras: object,
    ) -> Tensor:
        return self.objective(model, x_1, x_0=x_0, **model_extras)

    def sample(
        self,
        model: nn.Module,
        x_0: Tensor,
        **model_extras: object,
    ) -> FlowSampleOutput:
        return self.sampler.sample(model, x_0, **model_extras)


__all__ = ["ContinuousFlowMatcher"]
