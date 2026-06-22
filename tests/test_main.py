import unittest

from omegaconf import OmegaConf

from anytrain.hydra import run_train, validate_train_config


class HydraAppTest(unittest.TestCase):
    def _base_config(self):
        return OmegaConf.create(
            {
                "environment": {
                    "seed": 0,
                    "seed_workers": True,
                    "torch_matmul_precision": "medium",
                },
                "experiment": {
                    "save_dir": "outputs",
                    "name": "anytrain",
                    "version": "unit",
                },
                "pl_module": {
                    "_target_": "examples.tiny_regression.TinyRegressionModule",
                    "model": {
                        "_target_": "torch.nn.Linear",
                        "in_features": 4,
                        "out_features": 1,
                    },
                    "optimizer": {
                        "_target_": "torch.optim.Adam",
                        "_partial_": True,
                        "lr": 0.01,
                    },
                },
                "data_module": {
                    "_target_": "examples.tiny_regression.TinyRegressionDataModule",
                    "num_samples": 8,
                    "input_dim": 4,
                    "batch_size": 4,
                },
                "trainer": {
                    "max_epochs": 1,
                    "accelerator": "cpu",
                    "logger": False,
                    "enable_checkpointing": False,
                    "enable_model_summary": False,
                    "enable_progress_bar": False,
                },
                "fit": {"ckpt_path": None},
                "print_config": False,
            }
        )

    def test_run_train_from_hydra_style_config(self):
        cfg = self._base_config()

        trainer, module = run_train(cfg)

        self.assertEqual(trainer.current_epoch, 1)
        self.assertEqual(str(trainer.default_root_dir), "outputs/anytrain/unit")
        self.assertEqual(module.model.in_features, 4)
        self.assertIsNotNone(module.optimizer)

    def test_validate_train_config_rejects_top_level_dependencies(self):
        cfg = self._base_config()
        cfg.optimizer = {"_target_": "torch.optim.Adam"}

        with self.assertRaisesRegex(ValueError, "not top-level anytrain.hydra fields"):
            validate_train_config(cfg)

    def test_validate_train_config_rejects_missing_pl_module_target(self):
        cfg = self._base_config()
        cfg.pl_module = {"model": {"_target_": "torch.nn.Linear"}}

        with self.assertRaisesRegex(ValueError, "pl_module"):
            validate_train_config(cfg)

    def test_validate_train_config_rejects_trainer_target(self):
        cfg = self._base_config()
        cfg.trainer._target_ = "lightning.pytorch.Trainer"

        with self.assertRaisesRegex(ValueError, "remove `_target_`"):
            validate_train_config(cfg)

    def test_validate_train_config_rejects_unknown_fit_fields(self):
        cfg = self._base_config()
        cfg.fit.validate = True

        with self.assertRaisesRegex(ValueError, "unsupported fields"):
            validate_train_config(cfg)


if __name__ == "__main__":
    unittest.main()
