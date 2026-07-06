from __future__ import annotations

from collections.abc import Sequence

import torch


def int_sequence(ids: Sequence[int], *, name: str) -> list[int]:
    if not isinstance(ids, Sequence) or isinstance(ids, str | bytes):
        raise TypeError(f"{name} must be a sequence of integer ids.")
    normalized: list[int] = []
    for index, token_id in enumerate(ids):
        validate_int(token_id, name=f"{name}[{index}]")
        normalized.append(token_id)
    return normalized


def id_sequence(ids: Sequence[int], *, name: str) -> list[int]:
    values = int_sequence(ids, name=name)
    for index, token_id in enumerate(values):
        if token_id < 0:
            raise ValueError(f"{name}[{index}] must be non-negative.")
    return values


def validate_id_tensor(ids: torch.Tensor, *, name: str) -> None:
    if not isinstance(ids, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if ids.dtype == torch.bool or torch.is_floating_point(ids) or torch.is_complex(ids):
        raise TypeError(f"{name} must contain integer ids.")


def validate_int(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")


def validate_positive_int(value: int, *, name: str) -> None:
    validate_int(value, name=name)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def validate_non_negative_int(value: int, *, name: str) -> None:
    validate_int(value, name=name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
