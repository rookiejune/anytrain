from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from omegaconf import DictConfig

PL_MODULE_DEPENDENCY_KEYS = frozenset(
    {
        "model",
        "models",
        "loss",
        "loss_fn",
        "losses",
        "evaluator",
        "evaluators",
        "plotter",
        "plotters",
        "framework",
        "optimizer",
        "optimizers",
        "scheduler",
        "schedulers",
    }
)
FIT_CONFIG_KEYS = frozenset({"ckpt_path"})


def validate_train_config(cfg: DictConfig) -> None:
    if not isinstance(cfg, DictConfig):
        raise TypeError("cfg must be an OmegaConf DictConfig.")

    _validate_pl_module(cfg.get("pl_module"))
    _validate_optional_hydra_object_slot(cfg.get("data_module"), name="data_module")
    _validate_trainer_config(cfg.get("trainer"))
    _reject_top_level_pl_module_dependencies(cfg)
    _validate_fit_config(cfg.get("fit"))


def trainer_fit_kwargs(cfg: Any) -> dict[str, Any]:
    if cfg is None:
        return {}
    return {"ckpt_path": cfg.get("ckpt_path")}


def _validate_pl_module(value: Any) -> None:
    if value is None:
        raise ValueError(
            "`pl_module` with `_target_` must be provided. Put model, loss, optimizer, "
            "scheduler, evaluator, and other training components under `pl_module` as "
            "explicit constructor arguments."
        )
    if not _is_hydra_object_config(value):
        raise ValueError("`pl_module` must be a Hydra object config containing `_target_`.")


def _validate_optional_hydra_object_slot(value: Any, *, name: str) -> None:
    if value is None:
        return
    if not _is_hydra_object_config(value):
        raise ValueError(f"`{name}` must be null or a Hydra object config containing `_target_`.")


def _validate_trainer_config(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, (DictConfig, Mapping)):
        raise TypeError("`trainer` must be a mapping of Lightning Trainer keyword arguments.")
    if "_target_" in value:
        raise ValueError(
            "`trainer` is configured as Lightning Trainer keyword arguments; remove `_target_`. "
            "anytrain.hydra creates `lightning.pytorch.Trainer` for this slot."
        )


def _reject_top_level_pl_module_dependencies(cfg: DictConfig) -> None:
    invalid = sorted(key for key in PL_MODULE_DEPENDENCY_KEYS if cfg.get(key) is not None)
    if not invalid:
        return
    formatted = ", ".join(f"`{key}`" for key in invalid)
    raise ValueError(
        f"{formatted} are not top-level anytrain.hydra fields. Put these components under "
        "`pl_module` as explicit constructor arguments, or create them inside the "
        "LightningModule."
    )


def _validate_fit_config(value: Any) -> None:
    if value is None:
        return
    if not isinstance(value, (DictConfig, Mapping)):
        raise TypeError("`fit` must be a mapping of Trainer.fit keyword arguments.")
    unknown = sorted(key for key in value if key not in FIT_CONFIG_KEYS)
    if unknown:
        supported = ", ".join(f"`{key}`" for key in sorted(FIT_CONFIG_KEYS))
        formatted = ", ".join(f"`{key}`" for key in unknown)
        raise ValueError(f"`fit` contains unsupported fields: {formatted}. Supported: {supported}.")


def _is_hydra_object_config(value: Any) -> bool:
    return isinstance(value, (DictConfig, Mapping)) and "_target_" in value
