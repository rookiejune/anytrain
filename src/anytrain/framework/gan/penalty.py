from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ._output import split
from .types import Reduction


class _GradientPenalty(nn.Module):
    def __init__(self, *, reduction: Reduction | str = Reduction.Mean) -> None:
        super().__init__()
        self.reduction = Reduction(reduction)

    def forward(self, discriminator: nn.Module, fake: Tensor, real: Tensor) -> Tensor:
        if fake.shape != real.shape:
            raise ValueError("fake and real tensors must have the same shape for gradient penalty.")
        if fake.ndim < 2:
            raise ValueError("fake and real tensors must include batch and feature dimensions.")

        alpha_shape = (real.shape[0],) + (1,) * (real.ndim - 1)
        alpha = torch.rand(alpha_shape, device=real.device, dtype=real.dtype)
        interpolated = alpha * real + (1 - alpha) * fake
        interpolated.requires_grad_(True)

        _, logits = split(discriminator(interpolated))
        scores = _sample_scores(logits, reduction=self.reduction)
        if scores.shape != (real.shape[0],):
            raise ValueError("gradient penalty score reduction must return one score per item.")

        grad = torch.autograd.grad(
            outputs=scores,
            inputs=interpolated,
            grad_outputs=torch.ones_like(scores),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]
        grad_norm = grad.reshape(grad.shape[0], -1).norm(2, dim=1)
        return ((grad_norm - 1) ** 2).mean()


def _sample_scores(logits: Sequence[Tensor], *, reduction: Reduction) -> Tensor:
    if not logits:
        raise ValueError("at least one discriminator branch is required.")

    branch_scores: list[Tensor] = []
    batch_size: int | None = None
    for index, branch_logits in enumerate(logits):
        if branch_logits.ndim == 0:
            raise ValueError(f"discriminator branch {index} logits must include a batch dimension.")
        if batch_size is None:
            batch_size = branch_logits.shape[0]
        if branch_logits.shape[0] != batch_size:
            raise ValueError("all discriminator branches must share the same batch size.")
        branch_scores.append(branch_logits.reshape(branch_logits.shape[0], -1).mean(dim=1))

    stacked = torch.stack(branch_scores)
    if reduction == Reduction.Mean:
        return stacked.mean(dim=0)
    if reduction == Reduction.Sum:
        return stacked.sum(dim=0)
    raise ValueError(f"Unsupported GAN reduction {reduction!r}.")
