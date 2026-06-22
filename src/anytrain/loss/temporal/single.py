from __future__ import annotations

import torch
from torch import Tensor

from ..abc import LossABC


def _sum_channels_and_time(signal: Tensor) -> Tensor:
    if signal.ndim < 2:
        raise ValueError("audio tensors must include a time dimension.")
    return signal.square().sum(dim=(-2, -1))


def global_sdr(estimate: Tensor, reference: Tensor, *, eps: float = 1e-8) -> Tensor:
    if estimate.shape != reference.shape:
        raise ValueError("estimate and reference must have the same shape.")

    error = estimate - reference
    numerator = _sum_channels_and_time(reference) + eps
    denominator = _sum_channels_and_time(error) + eps
    return 10 * torch.log10(numerator / denominator)


def scale_invariant_signal(
    estimate: Tensor,
    reference: Tensor,
    *,
    eps: float = 1e-8,
    min_energy: float = 1e-12,
    fallback_scale: float = 1.0,
) -> Tensor:
    if estimate.shape != reference.shape:
        raise ValueError("estimate and reference must have the same shape.")

    reference_energy = (reference * reference).sum(dim=(-2, -1))
    estimate_energy = (estimate * estimate).sum(dim=(-2, -1))
    projection = (estimate * reference).sum(dim=(-2, -1))
    scale = projection / (reference_energy + eps)
    fallback = torch.full_like(scale, fallback_scale)
    scale = torch.where(estimate_energy < min_energy, fallback, scale)
    return scale[..., None, None] * reference


def si_snr(estimate: Tensor, reference: Tensor, *, eps: float = 1e-8) -> Tensor:
    target = scale_invariant_signal(estimate, reference, eps=eps)
    return global_sdr(estimate, target, eps=eps)


class SDRLoss(LossABC):
    def __init__(
        self,
        *,
        scale_invariant: bool = True,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.scale_invariant = scale_invariant
        self.eps = eps

    def compute_loss(self, input: Tensor, target: Tensor) -> Tensor:
        score = (
            si_snr(input, target, eps=self.eps)
            if self.scale_invariant
            else global_sdr(input, target, eps=self.eps)
        )
        return -score.mean()
