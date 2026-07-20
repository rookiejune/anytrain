from __future__ import annotations

import torch
from torch import nn

__all__ = ["register_buffer"]


def register_buffer(
    module: nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    persistent: bool = True,
) -> None:
    """Register a tensor buffer across PyTorch versions.

    PyTorch 2.5 added ``nn.Buffer``, which auto-registers on assignment. Older
    versions require the explicit ``Module.register_buffer`` API.
    """
    buffer_type = getattr(nn, "Buffer", None)
    if buffer_type is None:
        module.register_buffer(name, tensor, persistent=persistent)
    else:
        setattr(module, name, buffer_type(tensor, persistent=persistent))
