import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace


class _LogModule:
    def __init__(self):
        self.log_dict_calls = []

    def log_dict(self, values, **kwargs):
        self.log_dict_calls.append((values, kwargs))


class _CheckpointStrategy:
    def __init__(self):
        self.barrier_count = 0

    def reduce_boolean_decision(self, decision, *, all=False):
        return decision

    def barrier(self, *args, **kwargs):
        self.barrier_count += 1


class _CheckpointTrainer:
    def __init__(self, payload: bytes = b"checkpoint"):
        self.global_step = 7
        self.is_global_zero = True
        self.loggers = []
        self.strategy = _CheckpointStrategy()
        self.saved_paths = []
        self.payload = payload

    def save_checkpoint(self, filepath, weights_only):
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.payload)
        self.saved_paths.append((path, weights_only))


class LightningTest(unittest.TestCase):
    def test_debug_callback_reports_first_gradient_after_backward(self):
        import torch

        from anytrain.lightning import DebugCallback

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.clean = torch.nn.Linear(1, 1, bias=False)
                self.bad = torch.nn.Linear(1, 1, bias=False)

        module = Module()
        module.clean.weight.grad = torch.ones_like(module.clean.weight)
        module.bad.weight.grad = torch.full_like(module.bad.weight, float("inf"))
        trainer = SimpleNamespace(current_epoch=0, global_step=0, global_rank=0)

        callback = DebugCallback()

        stderr = StringIO()
        with redirect_stderr(stderr), self.assertRaisesRegex(RuntimeError, "Non-finite gradient"):
            callback.on_after_backward(trainer, module)

        self.assertIn("bad.weight", stderr.getvalue())

    def test_debug_callback_checks_frozen_parameters_once_at_train_start(self):
        import torch

        from anytrain.lightning import DebugCallback

        module = torch.nn.Linear(1, 1, bias=False)
        module.weight.requires_grad_(False)
        module.weight.fill_(float("nan"))
        trainer = SimpleNamespace(current_epoch=0, global_step=0, global_rank=0)

        with self.assertRaisesRegex(RuntimeError, "at train start"):
            DebugCallback().on_train_start(trainer, module)

    def test_debug_callback_uses_aggregate_fast_path_for_finite_tensors(self):
        from unittest.mock import patch

        import torch

        from anytrain.lightning import DebugCallback

        module = torch.nn.Sequential(*(torch.nn.Linear(2, 2, bias=False) for _ in range(8)))
        for parameter in module.parameters():
            parameter.grad = torch.ones_like(parameter)
        trainer = SimpleNamespace(current_epoch=0, global_step=0, global_rank=0)

        with (
            patch(
                "anytrain.lightning.callback.debug._all_finite",
                return_value=True,
            ) as aggregate,
            patch("anytrain.lightning.callback.debug._find_nonfinite_tensor") as locate,
        ):
            DebugCallback().on_after_backward(trainer, module)

        aggregate.assert_called_once()
        locate.assert_not_called()

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

    def test_debug_callback_can_be_passed_to_trainer(self):
        from lightning import pytorch as pl

        from anytrain.lightning import DebugCallback

        callback = DebugCallback()
        trainer = pl.Trainer(
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            callbacks=[callback],
        )

        self.assertTrue(any(item is callback for item in trainer.callbacks))

    def test_model_checkpoint_matches_lightning_checkpoint_interface(self):
        import inspect

        from lightning.pytorch.callbacks import ModelCheckpoint as LightningModelCheckpoint

        from anytrain.lightning import ModelCheckpoint

        original = inspect.signature(LightningModelCheckpoint.__init__)
        custom = inspect.signature(ModelCheckpoint.__init__)

        self.assertEqual(list(custom.parameters)[:-1], list(original.parameters))
        self.assertEqual(custom.parameters["async_save"].default, True)
        self.assertEqual(custom.parameters["async_save"].kind, inspect.Parameter.KEYWORD_ONLY)

    def test_model_checkpoint_can_be_pickled_before_async_work(self):
        import pickle

        from anytrain.lightning import ModelCheckpoint

        callback = ModelCheckpoint(async_save=True)
        restored = pickle.loads(pickle.dumps(callback))

        self.assertTrue(restored.async_save)
        restored.wait_async_saves()

    def test_model_checkpoint_async_save_copies_from_local_tmp(self):
        from tempfile import TemporaryDirectory

        from anytrain.lightning import ModelCheckpoint

        callback = ModelCheckpoint(async_save=True)
        try:
            with TemporaryDirectory() as tmp_dir:
                target = Path(tmp_dir) / "nfs" / "model.ckpt"
                trainer = _CheckpointTrainer(payload=b"saved")

                callback._save_checkpoint(trainer, str(target))
                callback.wait_async_saves()

                self.assertEqual(target.read_bytes(), b"saved")
                self.assertNotEqual(trainer.saved_paths[0][0], target)
                self.assertEqual(trainer.saved_paths[0][0].suffix, ".ckpt")
                self.assertFalse(trainer.saved_paths[0][1])
                self.assertEqual(callback._last_checkpoint_saved, str(target))
        finally:
            callback._close_async_storage()

    def test_model_checkpoint_sync_opt_out_uses_target_path(self):
        from tempfile import TemporaryDirectory

        from anytrain.lightning import ModelCheckpoint

        with TemporaryDirectory() as tmp_dir:
            target = Path(tmp_dir) / "model.ckpt"
            trainer = _CheckpointTrainer(payload=b"sync")
            callback = ModelCheckpoint(async_save=False)

            callback._save_checkpoint(trainer, str(target))

            self.assertEqual(target.read_bytes(), b"sync")
            self.assertEqual(trainer.saved_paths, [(target, False)])

    def test_model_checkpoint_async_remove_is_ordered_after_copy(self):
        from tempfile import TemporaryDirectory

        from anytrain.lightning import ModelCheckpoint

        callback = ModelCheckpoint(async_save=True)
        try:
            with TemporaryDirectory() as tmp_dir:
                source = Path(tmp_dir) / "source.ckpt"
                source.write_bytes(b"old")
                target = Path(tmp_dir) / "target.ckpt"

                callback._submit_async_copy(source, target)
                callback._submit_async_remove(target)
                callback.wait_async_saves()

                self.assertFalse(target.exists())
        finally:
            callback._close_async_storage()


if __name__ == "__main__":
    unittest.main()
