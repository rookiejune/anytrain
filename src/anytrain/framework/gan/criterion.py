from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .types import GAN, Reduction, _gan


class _LogitCriterion(nn.Module):
    def __init__(
        self,
        gan: GAN | str = GAN.Hinge,
        *,
        reduction: Reduction | str = Reduction.Mean,
    ) -> None:
        super().__init__()
        self.gan = _gan(gan)
        self.reduction = Reduction(reduction)

    def discriminator_loss(
        self,
        real: Sequence[Tensor],
        fake: Sequence[Tensor],
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if len(real) != len(fake):
            raise ValueError("real and fake logits must have the same branch count.")

        real_loss = self._reduce([self.real_loss(logits) for logits in real])
        fake_loss = self._reduce([self.fake_loss(logits) for logits in fake])
        return real_loss + fake_loss, {"real": real_loss, "fake": fake_loss}

    def generator_loss(self, fake: Sequence[Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        adv_loss = self._reduce([self.adv_loss(logits) for logits in fake])
        return adv_loss, {"adv": adv_loss}

    def real_loss(self, real_logits: Tensor) -> Tensor:
        if self.gan == GAN.Hinge:
            return F.relu(1 - real_logits).mean()
        if self.gan == GAN.LSGAN:
            return torch.mean((1 - real_logits) ** 2)
        if self.gan == GAN.WGAN:
            return -real_logits.mean()
        raise ValueError(f"Unsupported GAN type {self.gan!r}.")

    def fake_loss(self, fake_logits: Tensor) -> Tensor:
        if self.gan == GAN.Hinge:
            return F.relu(1 + fake_logits).mean()
        if self.gan == GAN.LSGAN:
            return (fake_logits**2).mean()
        if self.gan == GAN.WGAN:
            return fake_logits.mean()
        raise ValueError(f"Unsupported GAN type {self.gan!r}.")

    def adv_loss(self, fake_logits: Tensor) -> Tensor:
        if self.gan in {GAN.Hinge, GAN.WGAN}:
            return -fake_logits.mean()
        if self.gan == GAN.LSGAN:
            return torch.mean((1 - fake_logits) ** 2)
        raise ValueError(f"Unsupported GAN type {self.gan!r}.")

    def _reduce(self, losses: list[Tensor]) -> Tensor:
        if not losses:
            raise ValueError("at least one discriminator branch is required.")
        stacked = torch.stack(losses)
        if self.reduction == Reduction.Mean:
            return stacked.mean()
        if self.reduction == Reduction.Sum:
            return stacked.sum()
        raise ValueError(f"Unsupported GAN reduction {self.reduction!r}.")
