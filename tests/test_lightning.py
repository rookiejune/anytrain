import unittest
from types import SimpleNamespace


class _LogModule:
    def __init__(self):
        self.log_dict_calls = []

    def log_dict(self, values, **kwargs):
        self.log_dict_calls.append((values, kwargs))


class LightningTest(unittest.TestCase):
    def test_stop_on_nonfinite_loss_callback_raises(self):
        import torch

        from anytrain.lightning import StopOnNonfiniteLossCallback

        callback = StopOnNonfiniteLossCallback()
        trainer = SimpleNamespace(current_epoch=0, global_step=0)

        with self.assertRaisesRegex(RuntimeError, "Non-finite loss"):
            callback.on_before_backward(trainer, None, torch.tensor(float("nan")))

    def test_prefixed_log_dict(self):
        from anytrain.lightning import prefixed_log_dict

        self.assertEqual(
            prefixed_log_dict("train/loss", {"recon": 1.0, "/adv/": 0.5}),
            {"train/loss/recon": 1.0, "train/loss/adv": 0.5},
        )

    def test_lightning_log_mixin_logs_prefixed_dict(self):
        from anytrain.lightning import LightningLogMixin

        class Module(LightningLogMixin, _LogModule):
            pass

        module = Module()
        module.log_prefixed_dict("val", {"loss": 2.0}, sync_dist=True)

        self.assertEqual(module.log_dict_calls, [({"val/loss": 2.0}, {"sync_dist": True})])

    def test_lightning_log_mixin_requires_media_logger(self):
        import torch

        from anytrain.lightning import LightningLogMixin

        class Module(LightningLogMixin):
            global_step = 3
            trainer = SimpleNamespace(loggers=[], is_global_zero=True)

        with self.assertRaisesRegex(RuntimeError, "no configured logger"):
            Module().log_audio("sample", torch.zeros(1, 8), sample_rate=16000)

    def test_lightning_log_mixin_skips_media_on_nonzero_rank_by_default(self):
        import torch
        from lightning.pytorch.loggers import TensorBoardLogger

        from anytrain.lightning import LightningLogMixin

        class Experiment:
            def __init__(self):
                self.audio = []
                self.figures = []

            def add_audio(self, tag, audio, *, global_step, sample_rate):
                self.audio.append((tag, audio, global_step, sample_rate))

            def add_figure(self, tag, figure, *, global_step):
                self.figures.append((tag, figure, global_step))

        class Logger(TensorBoardLogger):
            def __init__(self):
                self._experiment = Experiment()

            @property
            def experiment(self):
                return self._experiment

        logger = Logger()

        class Module(LightningLogMixin):
            global_step = 3
            trainer = SimpleNamespace(
                loggers=[logger],
                is_global_zero=False,
                world_size=4,
                global_rank=2,
            )

        module = Module()
        module.log_audio("sample", torch.zeros(1, 8), sample_rate=16000)
        module.log_figure("figure", object())

        self.assertEqual(logger.experiment.audio, [])
        self.assertEqual(logger.experiment.figures, [])

    def test_lightning_log_mixin_prefixes_rank_when_logging_all_ranks(self):
        import torch
        from lightning.pytorch.loggers import TensorBoardLogger

        from anytrain.lightning import LightningLogMixin

        class Experiment:
            def __init__(self):
                self.audio = []
                self.figures = []

            def add_audio(self, tag, audio, *, global_step, sample_rate):
                self.audio.append((tag, audio, global_step, sample_rate))

            def add_figure(self, tag, figure, *, global_step):
                self.figures.append((tag, figure, global_step))

        class Logger(TensorBoardLogger):
            def __init__(self):
                self._experiment = Experiment()

            @property
            def experiment(self):
                return self._experiment

        logger = Logger()
        figure = object()

        class Module(LightningLogMixin):
            global_step = 3
            trainer = SimpleNamespace(
                loggers=[logger],
                is_global_zero=False,
                world_size=4,
                global_rank=2,
            )

        module = Module()
        module.log_audio("sample", torch.zeros(1, 8), sample_rate=16000, rank_mode="all")
        module.log_figure("figure", figure, rank_mode="all")

        self.assertEqual(logger.experiment.audio[0][0], "rank=2/sample")
        self.assertEqual(logger.experiment.audio[0][2:], (3, 16000))
        self.assertEqual(logger.experiment.figures, [("rank=2/figure", figure, 3)])

    def test_create_trainer_instantiates_callback_config(self):
        from omegaconf import OmegaConf

        from anytrain.hydra import create_trainer
        from anytrain.lightning import StopOnNonfiniteLossCallback

        cfg = OmegaConf.create(
            {
                "logger": False,
                "enable_checkpointing": False,
                "enable_progress_bar": False,
                "callbacks": [
                    {"_target_": "anytrain.lightning.StopOnNonfiniteLossCallback"},
                ],
            }
        )
        trainer = create_trainer(cfg, experiment={"save_dir": "outputs"})

        self.assertTrue(
            any(isinstance(callback, StopOnNonfiniteLossCallback) for callback in trainer.callbacks)
        )


if __name__ == "__main__":
    unittest.main()
