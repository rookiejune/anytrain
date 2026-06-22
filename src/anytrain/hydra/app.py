from __future__ import annotations

import hydra
from omegaconf import DictConfig, OmegaConf

from .config import trainer_fit_kwargs, validate_train_config
from .environment import configure_environment
from .modules import instantiate_train_modules


def run_train(cfg: DictConfig):
    validate_train_config(cfg)

    if cfg.get("print_config", True):
        print(OmegaConf.to_yaml(cfg))

    configure_environment(cfg.get("environment"))
    modules = instantiate_train_modules(cfg)
    modules.trainer.fit(
        modules.pl_module,
        datamodule=modules.data_module,
        **trainer_fit_kwargs(cfg.get("fit")),
    )
    return modules.trainer, modules.pl_module


@hydra.main(version_base="1.3", config_path=None, config_name=None)
def main(cfg: DictConfig) -> None:
    run_train(cfg)


if __name__ == "__main__":
    main()
