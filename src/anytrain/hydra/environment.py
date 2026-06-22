from __future__ import annotations

import torch
from omegaconf import DictConfig


def configure_environment(cfg: DictConfig | None) -> None:
    if cfg is None:
        return

    precision = cfg.get("torch_matmul_precision")
    if precision is not None:
        torch.set_float32_matmul_precision(str(precision))

    seed = cfg.get("seed")
    if seed is not None:
        from lightning import pytorch as pl

        pl.seed_everything(int(seed), workers=bool(cfg.get("seed_workers", True)))
