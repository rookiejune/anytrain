from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ._output import Features, validate_matching_features
from .types import Reduction


class _FeatureMatching(nn.Module):
    def __init__(
        self,
        *,
        reduction: Reduction | str = Reduction.Mean,
        detach_real: bool = True,
        loss_fn: Callable[[Tensor, Tensor], Tensor] = F.l1_loss,
    ) -> None:
        super().__init__()
        self.reduction = Reduction(reduction)
        self.detach_real = detach_real
        self.loss_fn = loss_fn

    def forward(
        self,
        fake: Features,
        real: Features,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        validate_matching_features(fake, real, require_features=True)
        losses: list[Tensor] = []
        for fake_branch, real_branch in zip(fake, real, strict=True):
            for fake_feature, real_feature in zip(fake_branch, real_branch, strict=True):
                target = real_feature.detach() if self.detach_real else real_feature
                loss = self.loss_fn(fake_feature, target)
                if loss.ndim != 0:
                    raise ValueError("feature matching loss_fn must return a scalar tensor.")
                losses.append(loss)

        total = self._reduce(losses)
        return total, {"feature": total}

    def _reduce(self, losses: list[Tensor]) -> Tensor:
        if not losses:
            raise ValueError("feature matching requires at least one feature map.")
        stacked = torch.stack(losses)
        match self.reduction:
            case Reduction.Mean:
                return stacked.mean()
            case Reduction.Sum:
                return stacked.sum()
