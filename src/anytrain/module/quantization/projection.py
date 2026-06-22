from __future__ import annotations

from torch import nn
from torch.nn.utils import parametrizations


def make_projection(
    in_features: int,
    out_features: int,
    *,
    bias: bool = True,
    weight_norm: bool = False,
) -> nn.Module:
    if in_features <= 0:
        raise ValueError(f"in_features must be positive, got {in_features}.")
    if out_features <= 0:
        raise ValueError(f"out_features must be positive, got {out_features}.")

    if in_features == out_features:
        return nn.Identity()

    projection: nn.Module = nn.Linear(in_features, out_features, bias=bias)
    if weight_norm:
        projection = parametrizations.weight_norm(projection)
    return projection
