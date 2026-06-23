from __future__ import annotations

from torch import Tensor
from torch.nn import functional as F


def nearest_codebook_indices(
    latents: Tensor,
    codebook: Tensor,
    *,
    normalize: bool,
) -> Tensor:
    if latents.ndim == 0:
        raise ValueError("latents must have at least one dimension.")
    if codebook.ndim != 2:
        raise ValueError(f"codebook must have shape (size, dim), got {tuple(codebook.shape)}.")
    if latents.shape[-1] != codebook.shape[-1]:
        raise ValueError(
            f"latents must end with codebook dim={codebook.shape[-1]}, "
            f"got {tuple(latents.shape)}."
        )

    leading_shape = latents.shape[:-1]
    flat_latents = latents.reshape(-1, codebook.shape[-1])
    if normalize:
        lookup = F.normalize(flat_latents, dim=-1)
        codebook_lookup = F.normalize(codebook, dim=-1)
        indices = F.linear(lookup, codebook_lookup).argmax(dim=-1)
    else:
        distances = (
            flat_latents.square().sum(dim=-1, keepdim=True)
            - 2 * flat_latents @ codebook.t()
            + codebook.square().sum(dim=-1)
        )
        indices = distances.argmin(dim=-1)
    return indices.reshape(*leading_shape)
