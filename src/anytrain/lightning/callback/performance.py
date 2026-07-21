from __future__ import annotations

import time
import warnings
from collections import deque
from typing import Any

import torch
from lightning import pytorch as pl

from anytrain.perf import (
    PeakFlops,
    count_parameters,
    infer_peak_flops,
    model_flops_utilization,
)


class PerformanceCallback(pl.Callback):
    def __init__(
        self,
        *,
        model_flops_per_step: float | None = None,
        compute_dtype: torch.dtype | str | None = None,
        hardware_peak_flops: float | None = None,
        log_every_n_steps: int = 100,
        warmup_steps: int = 20,
        measure_window_steps: int = 100,
        sync_cuda: bool = True,
    ) -> None:
        _require_positive_int(log_every_n_steps, "log_every_n_steps")
        _require_non_negative_int(warmup_steps, "warmup_steps")
        _require_positive_int(measure_window_steps, "measure_window_steps")
        if model_flops_per_step is not None:
            _require_positive(model_flops_per_step, "model_flops_per_step")
        if hardware_peak_flops is not None:
            _require_positive(hardware_peak_flops, "hardware_peak_flops")

        self.model_flops_per_step = model_flops_per_step
        self.compute_dtype = compute_dtype
        self.hardware_peak_flops = hardware_peak_flops
        self.log_every_n_steps = log_every_n_steps
        self.warmup_steps = warmup_steps
        self.measure_window_steps = measure_window_steps
        self.sync_cuda = sync_cuda
        self.hardware: PeakFlops | None = None
        self._step_started_at: float | None = None
        self._step_times: deque[float] = deque(maxlen=measure_window_steps)

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.hardware = _infer_hardware(
            trainer=trainer,
            pl_module=pl_module,
            compute_dtype=self.compute_dtype,
            hardware_peak_flops=self.hardware_peak_flops,
        )

        metrics = {
            "perf/model_params": float(count_parameters(pl_module)),
            "perf/model_trainable_params": float(count_parameters(pl_module, trainable_only=True)),
        }
        if self.model_flops_per_step is not None:
            metrics["perf/model_flops_per_step"] = float(self.model_flops_per_step)
        if self.hardware is not None:
            metrics["perf/hardware_peak_flops"] = float(self.hardware.flops)

        pl_module.log_dict(metrics, on_step=True, on_epoch=False, logger=True)
        _log_metadata(trainer, _metadata(self.hardware))

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del trainer, pl_module, batch, batch_idx
        _sync_cuda(self.sync_cuda)
        self._step_started_at = time.perf_counter()

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del outputs, batch, batch_idx
        if self._step_started_at is None:
            return

        _sync_cuda(self.sync_cuda)
        step_time = time.perf_counter() - self._step_started_at
        self._step_started_at = None
        self._step_times.append(step_time)

        global_step = int(trainer.global_step)
        if global_step < self.warmup_steps:
            return
        if global_step % self.log_every_n_steps != 0:
            return

        mean_step_time = sum(self._step_times) / len(self._step_times)
        metrics = {
            "perf/step_time": float(step_time),
            "perf/step_time_window": float(mean_step_time),
        }
        if self.model_flops_per_step is not None:
            metrics["perf/model_flops_per_step"] = float(self.model_flops_per_step)
        if self.hardware is not None and self.model_flops_per_step is not None:
            metrics["perf/mfu"] = model_flops_utilization(
                model_flops_per_step=float(self.model_flops_per_step),
                step_time=mean_step_time,
                hardware_peak_flops=self.hardware.flops,
            )

        pl_module.log_dict(metrics, on_step=True, on_epoch=False, logger=True, sync_dist=True)


def _infer_hardware(
    *,
    trainer: pl.Trainer,
    pl_module: pl.LightningModule,
    compute_dtype: torch.dtype | str | None,
    hardware_peak_flops: float | None,
) -> PeakFlops | None:
    dtype = compute_dtype
    if dtype is None:
        dtype = _dtype_from_trainer(trainer)
    device = _device(pl_module)
    try:
        peak = infer_peak_flops(
            dtype=dtype,
            device=device,
            hardware_peak_flops=hardware_peak_flops,
        )
    except ValueError as exc:
        warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
        return None
    if peak is None:
        warnings.warn(
            "Cannot infer hardware peak FLOPs for the current device and compute dtype; "
            "set hardware_peak_flops to enable perf/mfu.",
            RuntimeWarning,
            stacklevel=2,
        )
    return peak


def _dtype_from_trainer(trainer: pl.Trainer) -> torch.dtype | str:
    precision = str(trainer.precision).lower()
    if "bf16" in precision:
        return torch.bfloat16
    if "16" in precision:
        return torch.float16
    if "64" in precision:
        return torch.float64
    if torch.cuda.is_available() and torch.backends.cuda.matmul.allow_tf32:
        return "tf32"
    return torch.float32


def _device(pl_module: pl.LightningModule) -> torch.device | None:
    try:
        return pl_module.device
    except (AttributeError, RuntimeError):
        return None


def _sync_cuda(sync_cuda: bool) -> None:
    if sync_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()


def _metadata(hardware: PeakFlops | None) -> dict[str, str | float]:
    if hardware is None:
        return {"perf/hardware_peak_flops_source": "unresolved"}
    return {
        "perf/hardware_peak_flops_source": hardware.source,
        "perf/hardware_device_name": hardware.device_name,
        "perf/hardware_compute_dtype": hardware.dtype,
        "perf/hardware_peak_flops": hardware.flops,
    }


def _log_metadata(trainer: pl.Trainer, metadata: dict[str, str | float]) -> None:
    for logger in trainer.loggers:
        log_hyperparams = getattr(logger, "log_hyperparams", None)
        if log_hyperparams is not None:
            log_hyperparams(metadata)


def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_positive_int(value: int, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _require_non_negative_int(value: int, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")


__all__ = [
    "PerformanceCallback",
]
