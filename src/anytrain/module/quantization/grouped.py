from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .output import QuantizationLoss, QuantizeOutput
from .projection import make_projection


@dataclass
class GVQConfig:
    input_dim: int
    group_sizes: tuple[int, ...]
    codebook_dim: int | None = None
    group_dims: tuple[int, ...] | None = None
    normalize_latents: bool = True
    weight_norm: bool = False
    projection_bias: bool = True

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}.")
        if not self.group_sizes:
            raise ValueError("group_sizes must contain at least one value.")
        for group_size in self.group_sizes:
            if group_size <= 0:
                raise ValueError(f"each group size must be positive, got {group_size}.")

        if self.codebook_dim is None:
            self.codebook_dim = self.input_dim
        if self.codebook_dim <= 0:
            raise ValueError(f"codebook_dim must be positive, got {self.codebook_dim}.")

        if self.group_dims is None:
            self.group_dims = _balanced_group_dims(self.codebook_dim, len(self.group_sizes))
        else:
            self.group_dims = tuple(self.group_dims)

        if len(self.group_dims) != len(self.group_sizes):
            raise ValueError(
                "group_dims must have the same length as group_sizes: "
                f"got {len(self.group_dims)} and {len(self.group_sizes)}."
            )
        if sum(self.group_dims) != self.codebook_dim:
            raise ValueError(
                f"sum(group_dims) must equal codebook_dim={self.codebook_dim}, "
                f"got {sum(self.group_dims)}."
            )
        for group_dim in self.group_dims:
            if group_dim <= 0:
                raise ValueError(f"each group dim must be positive, got {group_dim}.")

        self.group_sizes = tuple(int(size) for size in self.group_sizes)
        self.group_dims = tuple(int(dim) for dim in self.group_dims)


class GroupedVectorQuantizer(nn.Module):
    config: GVQConfig
    project_in: nn.Module
    project_out: nn.Module
    codebooks: nn.ParameterList
    _basis: Tensor

    def __init__(self, config: GVQConfig) -> None:
        super().__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.codebook_dim = config.codebook_dim
        self.group_sizes = config.group_sizes
        self.group_dims = config.group_dims
        self.num_groups = len(config.group_sizes)
        self.num_codebooks = 1
        self.codebook_size = reduce(mul, config.group_sizes, 1)

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
        self.codebooks = nn.ParameterList(
            [nn.Parameter(torch.empty(size, dim)) for size, dim in zip(config.group_sizes, config.group_dims, strict=True)]
        )
        self._basis = nn.Buffer(torch.cumprod(torch.tensor([1, *config.group_sizes[:-1]]), dim=0))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            for codebook in self.codebooks:
                nn.init.normal_(codebook, mean=0.0, std=1.0)

    def forward(self, latents: Tensor) -> QuantizeOutput:
        return self.quantize(latents)

    def quantize(self, latents: Tensor) -> QuantizeOutput:
        self._validate_input_latents(latents)
        projected_latents = self.project_in(latents)
        codebook_vectors, group_indices = self._nearest_group_vectors(projected_latents)
        indices = self.group_indices_to_indices(group_indices)
        loss = None
        if self.training:
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
        codebook_vectors, _ = self._nearest_group_vectors(self.project_in(latents))
        return codebook_vectors

    def codebook_vectors_to_indices(self, codebook_vectors: Tensor) -> Tensor:
        self._validate_codebook_vectors(codebook_vectors)
        _, group_indices = self._nearest_group_vectors(codebook_vectors)
        return self.group_indices_to_indices(group_indices)

    def indices_to_codebook_vectors(self, indices: Tensor) -> Tensor:
        group_indices = self.indices_to_group_indices(indices)
        return self.group_indices_to_codebook_vectors(group_indices)

    def project_codebook_vectors(self, codebook_vectors: Tensor) -> Tensor:
        self._validate_codebook_vectors(codebook_vectors)
        return self.project_out(codebook_vectors)

    def indices_to_group_indices(self, indices: Tensor) -> Tensor:
        self._validate_indices(indices)
        return (indices.unsqueeze(-1) // self._basis) % torch.tensor(
            self.group_sizes,
            dtype=indices.dtype,
            device=indices.device,
        )

    def group_indices_to_indices(self, group_indices: Tensor) -> Tensor:
        self._validate_group_indices(group_indices)
        return (group_indices * self._basis.to(device=group_indices.device)).sum(dim=-1)

    def group_indices_to_codebook_vectors(self, group_indices: Tensor) -> Tensor:
        self._validate_group_indices(group_indices)
        vectors = []
        for group_index, codebook in enumerate(self.codebooks):
            vectors.append(codebook[group_indices[..., group_index]])
        return torch.cat(vectors, dim=-1)

    @classmethod
    def from_kwargs(
        cls,
        input_dim: int,
        group_sizes: tuple[int, ...] | list[int],
        codebook_dim: int | None = None,
        group_dims: tuple[int, ...] | list[int] | None = None,
        normalize_latents: bool = True,
        weight_norm: bool = False,
        projection_bias: bool = True,
    ) -> GroupedVectorQuantizer:
        return cls(
            GVQConfig(
                input_dim=input_dim,
                group_sizes=tuple(group_sizes),
                codebook_dim=codebook_dim,
                group_dims=None if group_dims is None else tuple(group_dims),
                normalize_latents=normalize_latents,
                weight_norm=weight_norm,
                projection_bias=projection_bias,
            )
        )

    def _nearest_group_vectors(self, projected_latents: Tensor) -> tuple[Tensor, Tensor]:
        self._validate_codebook_vectors(projected_latents, name="projected_latents")
        parts = projected_latents.split(self.group_dims, dim=-1)
        vector_parts = []
        index_parts = []
        for part, codebook in zip(parts, self.codebooks, strict=True):
            indices = self._nearest_indices(part, codebook)
            vector_parts.append(codebook[indices])
            index_parts.append(indices)
        return torch.cat(vector_parts, dim=-1), torch.stack(index_parts, dim=-1)

    def _nearest_indices(self, latents: Tensor, codebook: Tensor) -> Tensor:
        leading_shape = latents.shape[:-1]
        flat_latents = latents.reshape(-1, latents.shape[-1])
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
        return indices.reshape(*leading_shape)

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

    def _validate_group_indices(self, group_indices: Tensor) -> None:
        if group_indices.ndim == 0:
            raise ValueError(f"group_indices must end with num_groups={self.num_groups}.")
        if group_indices.shape[-1] != self.num_groups:
            raise ValueError(
                f"group_indices must end with num_groups={self.num_groups}, "
                f"got {tuple(group_indices.shape)}."
            )
        if torch.is_floating_point(group_indices) or torch.is_complex(group_indices):
            raise TypeError("group_indices must be an integer tensor.")
        if group_indices.numel() == 0:
            raise ValueError("group_indices must contain at least one value.")
        for group_index, group_size in enumerate(self.group_sizes):
            values = group_indices[..., group_index]
            min_value = int(values.min().item())
            max_value = int(values.max().item())
            if min_value < 0 or max_value >= group_size:
                raise ValueError(
                    f"group_indices[..., {group_index}] must be in [0, {group_size - 1}], "
                    f"got min={min_value}, max={max_value}."
                )


def _balanced_group_dims(codebook_dim: int, num_groups: int) -> tuple[int, ...]:
    base = codebook_dim // num_groups
    remainder = codebook_dim % num_groups
    return tuple(base + (1 if index < remainder else 0) for index in range(num_groups))

