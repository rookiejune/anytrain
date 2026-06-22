from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto


class MuonAdjustLRFn(StrEnum):
    ORIGINAL = auto()
    MATCH_RMS_ADAMW = auto()


@dataclass(frozen=True)
class AdamWConfig:
    lr: float
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    fused: bool | None = None

    def __post_init__(self) -> None:
        _validate_non_negative_float(self.lr, name="lr")
        _validate_non_negative_float(self.weight_decay, name="weight_decay")
        _validate_betas(self.betas)
        _validate_positive_float(self.eps, name="eps")
        if self.fused is not None and not isinstance(self.fused, bool):
            raise TypeError("fused must be a bool or None.")


@dataclass(frozen=True)
class MuonConfig:
    lr: float
    weight_decay: float = 0.1
    momentum: float = 0.95
    nesterov: bool = True
    ns_coefficients: tuple[float, float, float] = (3.4445, -4.775, 2.0315)
    eps: float = 1e-7
    ns_steps: int = 5
    adjust_lr_fn: MuonAdjustLRFn | str = MuonAdjustLRFn.MATCH_RMS_ADAMW

    def __post_init__(self) -> None:
        _validate_non_negative_float(self.lr, name="lr")
        _validate_non_negative_float(self.weight_decay, name="weight_decay")
        _validate_non_negative_float(self.momentum, name="momentum")
        if not isinstance(self.nesterov, bool):
            raise TypeError("nesterov must be a bool.")
        _validate_ns_coefficients(self.ns_coefficients)
        _validate_positive_float(self.eps, name="eps")
        _validate_positive_int(self.ns_steps, name="ns_steps")
        object.__setattr__(self, "adjust_lr_fn", _normalize_muon_adjust_lr_fn(self.adjust_lr_fn))


@dataclass(frozen=True)
class MuonAdamWConfig:
    muon: MuonConfig
    adamw: AdamWConfig

    def __post_init__(self) -> None:
        if not isinstance(self.muon, MuonConfig):
            raise TypeError("muon must be a MuonConfig.")
        if not isinstance(self.adamw, AdamWConfig):
            raise TypeError("adamw must be an AdamWConfig.")


def _validate_non_negative_float(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a float.")
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


def _validate_positive_float(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a float.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_positive_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_betas(betas: tuple[float, float]) -> None:
    if not isinstance(betas, tuple) or len(betas) != 2:
        raise TypeError("betas must be a tuple of two floats.")
    for index, beta in enumerate(betas):
        if isinstance(beta, bool) or not isinstance(beta, int | float):
            raise TypeError(f"betas[{index}] must be a float.")
        if not 0 <= beta < 1:
            raise ValueError(f"betas[{index}] must satisfy 0 <= beta < 1.")


def _validate_ns_coefficients(coefficients: tuple[float, float, float]) -> None:
    if not isinstance(coefficients, tuple) or len(coefficients) != 3:
        raise TypeError("ns_coefficients must be a tuple of three floats.")
    for index, coefficient in enumerate(coefficients):
        if isinstance(coefficient, bool) or not isinstance(coefficient, int | float):
            raise TypeError(f"ns_coefficients[{index}] must be a float.")


def _normalize_muon_adjust_lr_fn(adjust_lr_fn: MuonAdjustLRFn | str) -> MuonAdjustLRFn:
    if isinstance(adjust_lr_fn, MuonAdjustLRFn):
        return adjust_lr_fn
    if not isinstance(adjust_lr_fn, str):
        raise TypeError("adjust_lr_fn must be a string or MuonAdjustLRFn.")
    try:
        return MuonAdjustLRFn(adjust_lr_fn)
    except ValueError as error:
        raise ValueError("adjust_lr_fn must be original or match_rms_adamw.") from error
