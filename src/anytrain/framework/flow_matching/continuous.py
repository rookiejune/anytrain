from __future__ import annotations

from torch import Tensor, nn

from ._deps import CondOTProbPath
from .source import GaussianSource
from .time import LogitNormalTimeSampler
from .types import (
    ContinuousTrainingSample,
    Source,
    TimeSampler,
)


class ContinuousFlowRuntime(nn.Module):
    def __init__(
        self,
        *,
        path: CondOTProbPath | None = None,
        source: Source | None = None,
        time_sampler: TimeSampler | None = None,
    ):
        super().__init__()
        self.path = CondOTProbPath() if path is None else path
        self.source = GaussianSource() if source is None else source
        self.time_sampler = LogitNormalTimeSampler() if time_sampler is None else time_sampler

    def source_like(self, x_1: Tensor) -> Tensor:
        return self.source.sample_like(x_1)

    def training_sample(
        self,
        x_1: Tensor,
        *,
        x_0: Tensor | None = None,
    ) -> ContinuousTrainingSample:
        _require_batch(x_1, "x_1")
        if x_0 is None:
            x_0 = self.source_like(x_1)
        if x_0.shape != x_1.shape:
            raise ValueError(
                f"x_0 and x_1 must have the same shape, got {x_0.shape} and {x_1.shape}."
            )
        t = self.time_sampler.sample(x_1.shape[0], x_1.device)
        sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        return ContinuousTrainingSample(
            x_t=sample.x_t,
            t=sample.t,
            velocity=sample.dx_t,
        )


def _require_batch(x: Tensor, name: str) -> None:
    if x.ndim == 0:
        raise ValueError(f"{name} must include a batch dimension.")
    if x.shape[0] <= 0:
        raise ValueError(f"{name} batch size must be positive.")


__all__ = ["ContinuousFlowRuntime"]
