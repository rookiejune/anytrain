from __future__ import annotations

from typing import Any

from omegaconf import DictConfig

from .instantiate import instantiate_callbacks, to_object
from .paths import logger_fields, logger_root


def create_trainer(cfg: DictConfig | None, *, experiment: Any = None):
    from lightning import pytorch as pl

    trainer_kwargs = to_object(cfg) or {}
    fields = logger_fields(experiment)
    trainer_kwargs.setdefault("default_root_dir", logger_root(**fields))
    if cfg is not None and "callbacks" in cfg:
        trainer_kwargs["callbacks"] = instantiate_callbacks(cfg.get("callbacks"))

    logger = trainer_kwargs.get("logger")
    if isinstance(logger, dict):
        raise ValueError(
            "`trainer.logger` no longer accepts a config object; use `true`/`false` or pass "
            "a logger instance in Python."
        )
    return pl.Trainer(**trainer_kwargs)
