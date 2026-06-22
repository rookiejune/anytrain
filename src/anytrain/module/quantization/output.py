from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass(eq=False)
class QuantizationLoss:
    commitment: Tensor
    codebook: Tensor

    @property
    def total(self) -> Tensor:
        return self.commitment + self.codebook


@dataclass(eq=False)
class QuantizeOutput:
    quantized_latents: Tensor
    indices: Tensor
    codebook_vectors: Tensor | None = None
    latents: Tensor | None = None
    loss: QuantizationLoss | None = None
    active_codebook_mask: Tensor | None = None
