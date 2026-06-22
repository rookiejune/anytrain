from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .output import QuantizationLoss, QuantizeOutput
from .projection import make_projection


@dataclass
class VQConfig:
    input_dim: int
    codebook_size: int
    codebook_dim: int | None = None
    normalize_latents: bool = True
    weight_norm: bool = False
    scale_grad_by_freq: bool = False
    use_ema: bool = False
    decay: float = 0.99
    eps: float = 1e-5
    projection_bias: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")
        if self.codebook_size <= 0:
            raise ValueError(f"codebook_size must be positive, got {self.codebook_size}.")
        if self.codebook_dim is None:
            self.codebook_dim = self.input_dim
        if self.codebook_dim <= 0:
            raise ValueError(f"codebook_dim must be positive, got {self.codebook_dim}.")
        if not 0 <= self.decay < 1:
            raise ValueError(f"decay must be in [0, 1), got {self.decay}.")
        if self.eps <= 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")


class EmbeddingVectorQuantizer(nn.Module):
    config: VQConfig
    project_in: nn.Module
    project_out: nn.Module
    codebook: nn.Embedding
    _ema_counts: Tensor
    _ema_sums: Tensor

    def __init__(self, config: VQConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.codebook_size = config.codebook_size
        self.codebook_dim = config.codebook_dim
        self.num_codebooks = 1

        self.project_in = make_projection(
            config.input_dim,
            config.codebook_dim,
            bias=config.projection_bias,
            weight_norm=config.weight_norm,
        )
        self.project_out = make_projection(
            config.codebook_dim,
            config.input_dim,
            bias=config.projection_bias,
            weight_norm=config.weight_norm,
        )
        self.codebook = nn.Embedding(
            config.codebook_size,
            config.codebook_dim,
            scale_grad_by_freq=config.scale_grad_by_freq,
        )
        if config.use_ema:
            self._ema_counts = nn.Buffer(torch.ones(config.codebook_size))
            self._ema_sums = nn.Buffer(torch.empty(config.codebook_size, config.codebook_dim))
            self.codebook.weight.requires_grad_(False)
        self.reset_parameters()

    @property
    def use_ema(self) -> bool:
        return self.config.use_ema

    @property
    def ema_counts(self) -> Tensor:
        if not self.use_ema:
            raise AttributeError("ema_counts is only available when use_ema=True.")
        return self._ema_counts

    def reset_parameters(self) -> None:
        with torch.no_grad():
            nn.init.normal_(self.codebook.weight, mean=0.0, std=1.0)
            if self.use_ema:
                self._ema_sums.copy_(self.codebook.weight)
                self._ema_counts.fill_(1)

    def forward(self, latents: Tensor) -> QuantizeOutput:
        return self.quantize(latents)

    def quantize(self, latents: Tensor) -> QuantizeOutput:
        self._validate_input_latents(latents)
        projected_latents = self.project_in(latents)
        codebook_vectors, indices = self._nearest_codebook_vectors(projected_latents)

        loss = None
        if self.training:
            if self.use_ema:
                self._update_ema(projected_latents, indices)
            else:
                loss = QuantizationLoss(
                    commitment=F.mse_loss(projected_latents, codebook_vectors.detach()),
                    codebook=F.mse_loss(codebook_vectors, projected_latents.detach()),
                )

        straight_through = projected_latents + (codebook_vectors - projected_latents).detach()
        quantized_latents = self.project_out(straight_through)
        return QuantizeOutput(
            quantized_latents=quantized_latents,
            indices=indices,
            codebook_vectors=codebook_vectors,
            latents=projected_latents,
            loss=loss,
        )

    def latents_to_codebook_vectors(self, latents: Tensor) -> Tensor:
        self._validate_input_latents(latents)
        codebook_vectors, _ = self._nearest_codebook_vectors(self.project_in(latents))
        return codebook_vectors

    def codebook_vectors_to_indices(self, codebook_vectors: Tensor) -> Tensor:
        self._validate_codebook_vectors(codebook_vectors)
        _, indices = self._nearest_codebook_vectors(codebook_vectors)
        return indices

    def indices_to_codebook_vectors(self, indices: Tensor) -> Tensor:
        self._validate_indices(indices)
        return self.codebook(indices)

    def project_codebook_vectors(self, codebook_vectors: Tensor) -> Tensor:
        self._validate_codebook_vectors(codebook_vectors)
        return self.project_out(codebook_vectors)

    @classmethod
    def from_kwargs(
        cls,
        input_dim: int,
        codebook_size: int,
        codebook_dim: int | None = None,
        normalize_latents: bool = True,
        weight_norm: bool = False,
        scale_grad_by_freq: bool = False,
        use_ema: bool = False,
        decay: float = 0.99,
        eps: float = 1e-5,
        projection_bias: bool = True,
    ) -> EmbeddingVectorQuantizer:
        return cls(
            VQConfig(
                input_dim=input_dim,
                codebook_size=codebook_size,
                codebook_dim=codebook_dim,
                normalize_latents=normalize_latents,
                weight_norm=weight_norm,
                scale_grad_by_freq=scale_grad_by_freq,
                use_ema=use_ema,
                decay=decay,
                eps=eps,
                projection_bias=projection_bias,
            )
        )

    def _nearest_codebook_vectors(self, projected_latents: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_codebook_vectors(projected_latents, name="projected_latents")
        leading_shape = projected_latents.shape[:-1]
        flat_latents = projected_latents.reshape(-1, self.codebook_dim)
        codebook = self.codebook.weight

        if self.config.normalize_latents:
            flat_lookup = F.normalize(flat_latents, dim=-1)
            codebook_lookup = F.normalize(codebook, dim=-1)
            indices = (flat_lookup @ codebook_lookup.t()).argmax(dim=-1)
        else:
            distances = (
                flat_latents.pow(2).sum(dim=-1, keepdim=True)
                - 2 * flat_latents @ codebook.t()
                + codebook.pow(2).sum(dim=-1)
            )
            indices = distances.argmin(dim=-1)

        codebook_vectors = self.codebook(indices).reshape(*leading_shape, self.codebook_dim)
        return codebook_vectors, indices.reshape(*leading_shape)

    @torch.no_grad()
    def _update_ema(self, projected_latents: Tensor, indices: Tensor) -> None:
        flat_latents = projected_latents.detach().reshape(-1, self.codebook_dim)
        flat_indices = indices.reshape(-1)
        one_hot = F.one_hot(flat_indices, self.codebook_size).to(dtype=flat_latents.dtype)
        counts = one_hot.sum(dim=0)
        sums = one_hot.t() @ flat_latents

        self._ema_counts.mul_(self.config.decay).add_(counts, alpha=1 - self.config.decay)
        self._ema_sums.mul_(self.config.decay).add_(sums, alpha=1 - self.config.decay)

        total_count = self._ema_counts.sum()
        smoothed_counts = (
            (self._ema_counts + self.config.eps)
            / (total_count + self.codebook_size * self.config.eps)
            * total_count
        )
        self.codebook.weight.copy_(
            self._ema_sums / smoothed_counts.clamp_min(self.config.eps)[:, None]
        )

    def _validate_input_latents(self, latents: Tensor) -> None:
        if latents.ndim == 0:
            raise ValueError("latents must have at least one dimension.")
        if latents.shape[-1] != self.input_dim:
            raise ValueError(
                f"expected latents last dimension to be input_dim={self.input_dim}, "
                f"got {latents.shape[-1]}."
            )
        if latents.numel() == 0:
            raise ValueError("latents must contain at least one vector.")

    def _validate_codebook_vectors(
        self,
        codebook_vectors: Tensor,
        *,
        name: str = "codebook_vectors",
    ) -> None:
        if codebook_vectors.ndim == 0:
            raise ValueError(f"{name} must have at least one dimension.")
        if codebook_vectors.shape[-1] != self.codebook_dim:
            raise ValueError(
                f"{name} must end with codebook_dim={self.codebook_dim}, "
                f"got {tuple(codebook_vectors.shape)}."
            )
        if codebook_vectors.numel() == 0:
            raise ValueError(f"{name} must contain at least one vector.")

    def _validate_indices(self, indices: Tensor) -> None:
        if torch.is_floating_point(indices) or torch.is_complex(indices):
            raise TypeError("indices must be an integer tensor.")
        if indices.numel() == 0:
            raise ValueError("indices must contain at least one value.")
        min_index = int(indices.min().item())
        max_index = int(indices.max().item())
        if min_index < 0 or max_index >= self.codebook_size:
            raise ValueError(
                "indices must be in [0, codebook_size - 1]: "
                f"got min={min_index}, max={max_index}, codebook_size={self.codebook_size}."
            )
