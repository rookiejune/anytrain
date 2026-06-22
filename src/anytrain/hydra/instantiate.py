from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf


def to_object(cfg: Any) -> Any:
    if cfg is None:
        return None
    return OmegaConf.to_object(cfg) if isinstance(cfg, DictConfig) else cfg


def instantiate_callbacks(callbacks: Any) -> Any:
    if isinstance(callbacks, (DictConfig, ListConfig)) or has_hydra_target(callbacks):
        return instantiate(callbacks, _convert_="object")
    return callbacks


def has_hydra_target(value: Any) -> bool:
    if isinstance(value, Mapping):
        return "_target_" in value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return any(has_hydra_target(item) for item in value)
    return False
