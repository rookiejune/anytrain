from __future__ import annotations

from torch import Tensor, nn

from ._deps import MixtureDiscreteProbPath
from .objective import DiscreteGeneralizedKLObjective
from .sampler import DiscreteEulerSampler
from .source import UniformTokenSource
from .time import DEFAULT_TIME_EPS, LogitNormalTimeSampler
from .types import FlowSampleOutput, ModelCaller, Source, TimeSampler, default_call_model


class DiscreteFlowMatcher(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        path: MixtureDiscreteProbPath | None = None,
        source: Source | None = None,
        time_sampler: TimeSampler | None = None,
        sampler: DiscreteEulerSampler | None = None,
        call_model: ModelCaller = default_call_model,
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {vocab_size}.")

        self.vocab_size = vocab_size
        self.source = UniformTokenSource(vocab_size) if source is None else source
        self.time_sampler = (
            LogitNormalTimeSampler(t_max=1.0 - DEFAULT_TIME_EPS)
            if time_sampler is None
            else time_sampler
        )
        self.objective = DiscreteGeneralizedKLObjective(
            vocab_size,
            path=path,
            source=self.source,
            time_sampler=self.time_sampler,
            call_model=call_model,
        )
        self.sampler = (
            DiscreteEulerSampler(vocab_size, path=self.objective.path, call_model=call_model)
            if sampler is None
            else sampler
        )

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


__all__ = ["DiscreteFlowMatcher"]
