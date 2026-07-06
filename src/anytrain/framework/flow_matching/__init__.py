"""Composable wrappers around Facebook flow_matching."""

from .continuous import ContinuousFlowRuntime
from .discrete import DiscreteFlowMatcher
from .objective import (
    ContinuousVelocityObjective,
    DiscreteGeneralizedKLObjective,
    mse_velocity_loss,
)
from .sampler import DiscreteEulerSampler, ODESampler
from .source import GaussianSource, MaskTokenSource, UniformSource, UniformTokenSource
from .time import DEFAULT_TIME_EPS, LogitNormalTimeSampler, UniformTimeSampler
from .types import (
    ContinuousTrainingSample,
    FlowLossFn,
    FlowSampleOutput,
    ModelCaller,
    ModelExtras,
    Source,
    TimeSampler,
    default_call_model,
)

__all__ = [
    "ContinuousFlowRuntime",
    "ContinuousTrainingSample",
    "ContinuousVelocityObjective",
    "DEFAULT_TIME_EPS",
    "DiscreteEulerSampler",
    "DiscreteFlowMatcher",
    "DiscreteGeneralizedKLObjective",
    "FlowSampleOutput",
    "FlowLossFn",
    "GaussianSource",
    "MaskTokenSource",
    "ModelCaller",
    "ModelExtras",
    "ODESampler",
    "LogitNormalTimeSampler",
    "Source",
    "TimeSampler",
    "UniformSource",
    "UniformTimeSampler",
    "UniformTokenSource",
    "default_call_model",
    "mse_velocity_loss",
]
