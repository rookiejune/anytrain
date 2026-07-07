from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from ...loss.abc import LossDetails, LossDetailValue
from ._output import Features, split
from .criterion import _LogitCriterion
from .feature import _FeatureMatching
from .penalty import _GradientPenalty
from .types import GAN, Preset, Reduction, resolve_gan, resolve_preset


class Loss(nn.Module):
    def __init__(
        self,
        discriminator: nn.Module,
        *,
        gan: GAN | str = GAN.Hinge,
        reduction: Reduction | str = Reduction.Mean,
        feature_weight: float = 0.0,
        gp_weight: float | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(discriminator, nn.Module):
            raise TypeError("discriminator must be a torch.nn.Module.")
        if feature_weight < 0:
            raise ValueError("feature_weight must be non-negative.")
        if gp_weight is not None and gp_weight < 0:
            raise ValueError("gp_weight must be non-negative.")

        self.gan = resolve_gan(gan)
        self.reduction = Reduction(reduction)
        self.discriminator = discriminator
        self._criterion = _LogitCriterion(self.gan, reduction=self.reduction)
        self.feature_weight = float(feature_weight)
        self._feature = (
            _FeatureMatching(reduction=self.reduction) if self.feature_weight > 0 else None
        )
        self.gp_weight = (
            10.0 if gp_weight is None and self.gan == GAN.WGAN else float(gp_weight or 0.0)
        )
        self._gp = _GradientPenalty(reduction=self.reduction) if self.gp_weight > 0 else None

    @classmethod
    def from_preset(
        cls,
        preset: Preset,
        *,
        gan: GAN | str = GAN.Hinge,
        reduction: Reduction | str = Reduction.Mean,
        feature_weight: float = 0.0,
        gp_weight: float | None = None,
        **kwargs,
    ) -> Loss:
        resolved = resolve_preset(preset)
        if resolved == Preset.DAC:
            from .audio import DACDiscriminator

            discriminator = DACDiscriminator(**kwargs)
        else:
            raise NotImplementedError(f"Unsupported GAN preset: {resolved.value}")
        return cls(
            discriminator,
            gan=gan,
            reduction=reduction,
            feature_weight=feature_weight,
            gp_weight=gp_weight,
        )

    def discriminator_loss(self, fake: Tensor, real: Tensor) -> tuple[Tensor, LossDetails]:
        _, fake_logits = self._discriminate(fake.detach())
        _, real_logits = self._discriminate(real)
        total, details = self._criterion.discriminator_loss(real_logits, fake_logits)

        if self._gp is not None and self.gp_weight > 0:
            gp = self._gp(self.discriminator, fake.detach(), real.detach())
            total = total + total.new_tensor(self.gp_weight) * gp
            details = {**details, "gp": gp}
        return total, _detach_details(details)

    def generator_loss(
        self, fake: Tensor, real: Tensor | None = None
    ) -> tuple[Tensor, LossDetails]:
        fake_features, fake_logits = self._discriminate(fake)
        total, details = self._criterion.generator_loss(fake_logits)

        if self.feature_weight > 0:
            if real is None:
                raise ValueError("real must be provided when feature matching is enabled.")
            if self._feature is None:
                raise RuntimeError("feature_loss is not configured.")
            with torch.no_grad():
                real_features, _ = self._discriminate(real)
            feature, _ = self._feature(fake_features, real_features)
            total = total + total.new_tensor(self.feature_weight) * feature
            details = {
                **details,
                "feature": feature,
                "feature_weight": self.feature_weight,
            }
        return total, _detach_details(details)

    def _discriminate(self, x: Tensor) -> tuple[Features, list[Tensor]]:
        return split(self.discriminator(x))


def _detach_details(details: Mapping[str, LossDetailValue]) -> LossDetails:
    detached: LossDetails = {}
    for name, value in details.items():
        if not isinstance(name, str):
            raise TypeError("loss detail key must be a string.")
        if not name:
            raise ValueError("loss detail key must not be empty.")
        if "/" in name:
            raise ValueError("loss detail key must not contain '/'.")
        if isinstance(value, bool):
            raise TypeError(f"Loss detail value {name!r} must be a float or 0-d tensor.")
        if isinstance(value, float):
            detached[name] = value
            continue
        if isinstance(value, Tensor):
            if value.ndim != 0:
                raise ValueError(f"Loss detail value {name!r} must be a 0-d tensor.")
            detached[name] = value.detach()
            continue
        raise TypeError(f"Loss detail value {name!r} must be a float or 0-d tensor.")
    return detached
