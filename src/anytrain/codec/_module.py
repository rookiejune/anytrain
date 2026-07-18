from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from torch import nn


class DeviceModule(nn.Module):
    _device: torch.Tensor

    def _init_device(self, device: torch.device) -> None:
        self._device = nn.Buffer(
            torch.empty(0, dtype=torch.uint8, device=device),
            persistent=False,
        )
        self.to(device)

    @property
    def device(self) -> torch.device:
        return self._device.device

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        if assign:
            self._raise_assign_error()
        return super().load_state_dict(state_dict, strict=strict, assign=False)

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        if local_metadata.get("assign_to_params_buffers", False):
            self._raise_assign_error()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _raise_assign_error() -> None:
        raise ValueError(
            "Codec wrappers do not support load_state_dict(assign=True) because it can "
            "separate backend tensors from the configured device."
        )
