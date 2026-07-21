import unittest
from types import SimpleNamespace
from unittest.mock import patch


class PerfTest(unittest.TestCase):
    def test_count_parameters_respects_trainable_only(self):
        import torch

        from anytrain.perf import count_parameters

        module = torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Linear(4, 2))
        module[1].weight.requires_grad_(False)
        module[1].bias.requires_grad_(False)

        self.assertEqual(count_parameters(module), 26)
        self.assertEqual(count_parameters(module, trainable_only=True), 16)

    def test_training_flops_from_forward_uses_explicit_backward_multiplier(self):
        from anytrain.perf import training_flops_from_forward

        self.assertEqual(training_flops_from_forward(10.0), 30.0)
        self.assertEqual(training_flops_from_forward(10.0, backward_multiplier=1.5), 25.0)

    def test_model_flops_utilization(self):
        from anytrain.perf import model_flops_utilization

        self.assertEqual(
            model_flops_utilization(
                model_flops_per_step=100.0,
                step_time=2.0,
                hardware_peak_flops=200.0,
            ),
            0.25,
        )

    def test_infer_peak_flops_uses_override(self):
        from anytrain.perf import infer_peak_flops

        peak = infer_peak_flops(
            dtype="bf16",
            device_name="Custom GPU",
            hardware_peak_flops=123.0,
        )

        self.assertIsNotNone(peak)
        assert peak is not None
        self.assertEqual(peak.flops, 123.0)
        self.assertEqual(peak.source, "override")
        self.assertEqual(peak.device_name, "Custom GPU")
        self.assertEqual(peak.dtype, "bfloat16")

    def test_infer_peak_flops_matches_known_gpu(self):
        from anytrain.perf import infer_peak_flops

        peak = infer_peak_flops(dtype="bf16", device_name="NVIDIA A100-PCIE-40GB")

        self.assertIsNotNone(peak)
        assert peak is not None
        self.assertEqual(peak.source, "auto")
        self.assertEqual(peak.flops, 312.0e12)

    def test_profile_forward_flops_reports_linear_ops(self):
        import torch

        from anytrain.perf import profile_forward_flops

        module = torch.nn.Linear(4, 3)
        flops = profile_forward_flops(module, args=(torch.ones(2, 4),))

        self.assertGreater(flops, 0)


class _Logger:
    def __init__(self):
        self.hyperparams = []

    def log_hyperparams(self, values):
        self.hyperparams.append(values)


class _Module:
    def __init__(self):
        import torch

        self.linear = torch.nn.Linear(2, 1)
        self.log_dict_calls = []
        self.device = torch.device("cpu")

    def parameters(self):
        return self.linear.parameters()

    def log_dict(self, values, **kwargs):
        self.log_dict_calls.append((values, kwargs))


class PerformanceCallbackTest(unittest.TestCase):
    def test_callback_logs_static_metrics_and_mfu(self):
        from anytrain.lightning import PerformanceCallback

        logger = _Logger()
        trainer = SimpleNamespace(
            precision="bf16-mixed",
            loggers=[logger],
            global_step=0,
        )
        module = _Module()
        callback = PerformanceCallback(
            model_flops_per_step=100.0,
            hardware_peak_flops=200.0,
            warmup_steps=0,
            log_every_n_steps=1,
            measure_window_steps=2,
            sync_cuda=False,
        )

        callback.on_train_start(trainer, module)
        self.assertEqual(module.log_dict_calls[0][0]["perf/model_flops_per_step"], 100.0)
        self.assertEqual(module.log_dict_calls[0][0]["perf/hardware_peak_flops"], 200.0)
        self.assertEqual(logger.hyperparams[0]["perf/hardware_peak_flops_source"], "override")

        with patch("time.perf_counter", side_effect=[1.0, 3.0]):
            callback.on_train_batch_start(trainer, module, batch=None, batch_idx=0)
            callback.on_train_batch_end(trainer, module, outputs=None, batch=None, batch_idx=0)

        metrics = module.log_dict_calls[-1][0]
        self.assertEqual(metrics["perf/step_time"], 2.0)
        self.assertEqual(metrics["perf/mfu"], 0.25)


if __name__ == "__main__":
    unittest.main()
