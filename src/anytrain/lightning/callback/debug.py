from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Literal

import torch
from lightning import pytorch as pl

_DEBUG_ENV = "ANYTRAIN_DEBUG"
_DEBUG_ENV_VALUE = "True"


@dataclass(frozen=True)
class _NonfiniteIssue:
    kind: Literal["parameter", "gradient"]
    name: str
    index: tuple[int, ...]
    value: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class DebugCallback(pl.Callback):
    def __init__(self) -> None:
        if os.environ.get(_DEBUG_ENV) != _DEBUG_ENV_VALUE:
            raise RuntimeError(
                f"{self.__class__.__name__} requires {_DEBUG_ENV}={_DEBUG_ENV_VALUE}. "
                "Remove the callback or enable the debug environment variable explicitly."
            )

    def on_after_backward(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        issue = _find_first_nonfinite(pl_module)
        if issue is None:
            return

        message = _format_issue(trainer, issue)
        print(message, file=sys.stderr, flush=True)
        raise RuntimeError(message)


def _find_first_nonfinite(pl_module: pl.LightningModule) -> _NonfiniteIssue | None:
    for name, parameter in pl_module.named_parameters():
        issue = _find_nonfinite_tensor(name, "parameter", parameter)
        if issue is not None:
            return issue

        if parameter.grad is None:
            continue

        issue = _find_nonfinite_tensor(name, "gradient", parameter.grad)
        if issue is not None:
            return issue

    return None


def _find_nonfinite_tensor(
    name: str,
    kind: Literal["parameter", "gradient"],
    tensor: torch.Tensor,
) -> _NonfiniteIssue | None:
    data = tensor.detach()
    mask = ~torch.isfinite(data)
    if not bool(mask.any().item()):
        return None

    index = tuple(int(item) for item in mask.nonzero(as_tuple=False)[0].tolist())
    value = data[index].item()
    return _NonfiniteIssue(
        kind=kind,
        name=name,
        index=index,
        value=repr(value),
        shape=tuple(data.shape),
        dtype=data.dtype,
        device=data.device,
    )


def _format_issue(trainer: pl.Trainer, issue: _NonfiniteIssue) -> str:
    return (
        f"Non-finite {issue.kind} detected after backward "
        f"(epoch={trainer.current_epoch}, global_step={trainer.global_step}, "
        f"rank={trainer.global_rank}). "
        f"name={issue.name!r}, index={issue.index}, value={issue.value}, "
        f"shape={issue.shape}, dtype={issue.dtype}, device={issue.device}."
    )


__all__ = [
    "DebugCallback",
]
