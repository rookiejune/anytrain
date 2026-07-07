from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn

LossDetailValue = float | torch.Tensor
LossDetails = dict[str, LossDetailValue]
LossResult = torch.Tensor | tuple[torch.Tensor, LossDetails]


class LossABC(nn.Module, ABC):
    detail_key_separator = "/"

    def forward(self, *args: Any, **kwargs: Any) -> LossResult:
        return self._validate_loss_result(self.compute_loss(*args, **kwargs))

    @abstractmethod
    def compute_loss(self, *args: Any, **kwargs: Any) -> LossResult:
        raise NotImplementedError

    def _validate_loss_result(self, result: object) -> LossResult:
        if isinstance(result, torch.Tensor):
            return self._validate_loss_tensor(result)

        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError("loss result tuple must contain exactly loss and details.")
            loss, details = result
            return (
                self._validate_loss_tensor(loss),
                self._validate_loss_details(details, detach=True),
            )

        raise TypeError("loss result must be a scalar tensor or (scalar tensor, details).")

    def _validate_loss_tensor(self, loss: object) -> torch.Tensor:
        if not isinstance(loss, torch.Tensor):
            raise TypeError("loss must be a torch.Tensor.")
        if loss.ndim != 0:
            raise ValueError("loss must be a scalar tensor.")
        return loss

    def _validate_loss_details(
        self,
        details: object,
        *,
        detach: bool,
    ) -> LossDetails:
        if not isinstance(details, Mapping):
            raise TypeError("loss details must be a mapping of string keys to detail values.")

        validated: LossDetails = {}
        for raw_name, raw_value in details.items():
            name = self._validate_detail_name(raw_name)
            validated[name] = self._validate_detail_value(raw_value, name=name, detach=detach)
        return validated

    def _validate_detail_name(self, name: object) -> str:
        if not isinstance(name, str):
            raise TypeError("loss detail key must be a string.")
        if not name:
            raise ValueError("loss detail key must not be empty.")
        if self.detail_key_separator in name:
            raise ValueError(
                f"loss detail key must not contain separator {self.detail_key_separator!r}."
            )
        return name

    def _validate_detail_value(
        self,
        value: object,
        *,
        name: str,
        detach: bool,
    ) -> LossDetailValue:
        if isinstance(value, bool):
            raise TypeError(f"Loss detail value {name!r} must be a float or 0-d tensor.")
        if isinstance(value, float):
            return value
        if isinstance(value, torch.Tensor):
            if value.ndim != 0:
                raise ValueError(f"Loss detail value {name!r} must be a 0-d tensor.")
            return value.detach() if detach else value
        raise TypeError(f"Loss detail value {name!r} must be a float or 0-d tensor.")
