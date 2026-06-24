from __future__ import annotations

from typing import Any

import torch
from torch import nn


def freeze_model(model: Any, *, device: Any | None = None) -> Any:
    if not isinstance(model, nn.Module):
        return model

    if device is not None:
        model = model.to(torch.device(device))
    model.requires_grad_(False)
    model.eval()
    return model
