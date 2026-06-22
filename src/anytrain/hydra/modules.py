from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from .trainer import create_trainer


@dataclass(frozen=True)
class TrainModules:
    pl_module: Any
    data_module: Any
    trainer: Any


def instantiate_train_modules(cfg: DictConfig) -> TrainModules:
    return TrainModules(
        pl_module=instantiate(cfg.get("pl_module"), _convert_="object"),
        data_module=instantiate(cfg.get("data_module"), _convert_="object"),
        trainer=create_trainer(cfg.get("trainer"), experiment=cfg.get("experiment")),
    )
