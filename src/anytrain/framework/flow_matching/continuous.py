from __future__ import annotations

from torch import Tensor, nn

from ._deps import CondOTProbPath
from ._runtime import require_batch, require_pair
from .sampler import ODESampler
from .source import GaussianSource
from .time import UniformTimeSampler
from .types import (
    ContinuousTrainingSample,
    FlowSampleOutput,
    ModelCaller,
    Source,
    TimeSampler,
    default_call_model,
)


class ContinuousFlowRuntime:
    def __init__(
        self,
        *,
        path: CondOTProbPath | None = None,
        source: Source | None = None,
        time_sampler: TimeSampler | None = None,
        sampler: ODESampler | None = None,
        call_model: ModelCaller = default_call_model,
    ):
        self.path = CondOTProbPath() if path is None else path
        self.source = GaussianSource() if source is None else source
        self.time_sampler = UniformTimeSampler() if time_sampler is None else time_sampler
        self.sampler = ODESampler() if sampler is None else sampler
        self.call_model = call_model

    def source_like(self, x_1: Tensor) -> Tensor:
        return self.source.sample_like(x_1)

    def training_sample(
        self,
        x_1: Tensor,
        *,
        x_0: Tensor | None = None,
    ) -> ContinuousTrainingSample:
        require_batch(x_1, "x_1")
        if x_0 is None:
            x_0 = self.source_like(x_1)
        require_pair(x_0, x_1)
        if x_0.dtype != x_1.dtype:
            raise TypeError("x_0 and x_1 must have the same dtype.")
        t = self.time_sampler.sample(x_1.shape[0], x_1.device).to(dtype=x_1.dtype)
        sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        return ContinuousTrainingSample(
            x_t=sample.x_t,
            t=sample.t,
            velocity=sample.dx_t,
        )

    def sample(
        self,
        model: nn.Module,
        x_0: Tensor,
        *,
        time_grid: Tensor | None = None,
        **model_extras: object,
    ) -> FlowSampleOutput:
        return self.sampler.sample(
            model,
            x_0,
            time_grid=time_grid,
            call_model=self.call_model,
            **model_extras,
        )


__all__ = ["ContinuousFlowRuntime"]
