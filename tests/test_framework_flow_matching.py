import sys
import unittest
from importlib import import_module, util

import torch
from torch import nn

FLOW_MATCHING_AVAILABLE = util.find_spec("flow_matching") is not None


class FlowMatchingImportTest(unittest.TestCase):
    def test_framework_import_does_not_require_flow_matching_extra(self):
        framework = import_module("anytrain.framework")

        self.assertEqual(framework.__all__, [])

    def test_flow_matching_import_requires_optional_dependency_when_missing(self):
        if FLOW_MATCHING_AVAILABLE:
            self.skipTest("flow_matching is installed")

        sys.modules.pop("anytrain.framework.flow_matching", None)
        with self.assertRaisesRegex(ImportError, r"anytrain\[flow\]"):
            import_module("anytrain.framework.flow_matching")


@unittest.skipUnless(FLOW_MATCHING_AVAILABLE, "flow_matching extra is not installed")
class FlowMatchingComponentTest(unittest.TestCase):
    def test_sources_and_time_sampler(self):
        from anytrain.framework.flow_matching import (
            GaussianSource,
            LogitNormalTimeSampler,
            MaskTokenSource,
            UniformSource,
            UniformTimeSampler,
            UniformTokenSource,
        )

        x = torch.zeros(2, 3)

        self.assertEqual(GaussianSource().sample_like(x).shape, x.shape)
        self.assertEqual(UniformSource(low=-1, high=1).sample_like(x).shape, x.shape)
        self.assertTrue((UniformTokenSource(5).sample_like(x) < 5).all())
        self.assertTrue(
            torch.equal(MaskTokenSource(7).sample_like(x), torch.full_like(x, 7).long())
        )

        logit = LogitNormalTimeSampler()
        t = logit.sample(1024, x.device)
        self.assertEqual(t.shape, (1024,))
        self.assertTrue((t >= logit.t_min).all())
        self.assertTrue((t <= logit.t_max).all())

        t = UniformTimeSampler(t_min=0.2, t_max=0.3).sample(4, x.device)
        self.assertEqual(t.shape, (4,))
        self.assertTrue((t >= 0.2).all())
        self.assertTrue((t <= 0.3).all())

    def test_default_time_sampler_is_logit_normal(self):
        from anytrain.framework.flow_matching import (
            ContinuousFlowRuntime,
            DiscreteFlowMatcher,
            LogitNormalTimeSampler,
        )

        self.assertIsInstance(ContinuousFlowRuntime().time_sampler, LogitNormalTimeSampler)
        self.assertIsInstance(DiscreteFlowMatcher(6).time_sampler, LogitNormalTimeSampler)
        self.assertEqual(ContinuousFlowRuntime().time_sampler.t_min, 0.0)
        self.assertEqual(ContinuousFlowRuntime().time_sampler.t_max, 1.0)
        self.assertEqual(DiscreteFlowMatcher(6).time_sampler.t_min, 0.0)
        self.assertLess(DiscreteFlowMatcher(6).time_sampler.t_max, 1.0)

    def test_continuous_loss_backward_and_sample_shape(self):
        from anytrain.framework.flow_matching import (
            ContinuousFlowRuntime,
            ContinuousVelocityObjective,
            ODESampler,
        )

        class ToyVelocity(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Linear(2, 2)

            def forward(self, x_t, t, condition=None):
                del condition
                scale = t.view(t.shape[0], *([1] * (x_t.ndim - 1)))
                return self.proj(x_t) + scale

        model = ToyVelocity()
        runtime = ContinuousFlowRuntime()
        objective = ContinuousVelocityObjective(runtime=runtime)
        sampler = ODESampler()
        x_1 = torch.randn(4, 3, 2)

        loss = objective(model, x_1, condition=torch.zeros(4, 1))
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.proj.weight.grad)

        output = sampler.sample(model, torch.randn(4, 3, 2), condition=torch.zeros(4, 1))
        self.assertEqual(output.final.shape, x_1.shape)
        self.assertIsNotNone(output.states)
        self.assertIsNotNone(output.time_grid)

    def test_continuous_loss_accepts_custom_masked_loss(self):
        from anytrain.framework.flow_matching import ContinuousVelocityObjective

        class ZeroVelocity(nn.Module):
            def forward(self, x_t, t, mask=None):
                del t, mask
                return torch.zeros_like(x_t)

        def masked_loss(prediction, target, extras):
            mask = extras["mask"].to(device=prediction.device, dtype=prediction.dtype)
            weights = mask.unsqueeze(-1)
            return ((prediction - target).square() * weights).sum() / (
                weights.sum() * prediction.size(-1)
            )

        objective = ContinuousVelocityObjective(loss_fn=masked_loss)
        x_1 = torch.tensor(
            [
                [[1.0, 2.0], [10.0, 20.0]],
                [[3.0, 4.0], [30.0, 40.0]],
            ]
        )
        x_0 = torch.zeros_like(x_1)
        mask = torch.tensor([[True, False], [True, False]])

        loss = objective(ZeroVelocity(), x_1, x_0=x_0, mask=mask)

        expected = torch.tensor([1.0, 2.0, 3.0, 4.0]).square().mean()
        self.assertTrue(torch.equal(loss, expected))

    def test_continuous_custom_loss_must_return_scalar(self):
        from anytrain.framework.flow_matching import ContinuousVelocityObjective

        class ZeroVelocity(nn.Module):
            def forward(self, x_t, t):
                del t
                return torch.zeros_like(x_t)

        def vector_loss(prediction, target, extras):
            del extras
            return (prediction - target).square().mean(dim=-1)

        objective = ContinuousVelocityObjective(loss_fn=vector_loss)

        with self.assertRaisesRegex(ValueError, "scalar"):
            objective(ZeroVelocity(), torch.randn(2, 3, 4))

    def test_ode_sampler_expands_scalar_time_to_batch(self):
        from anytrain.framework.flow_matching import ODESampler

        class RecordingModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.t_shape = None

            def forward(self, x_t, t):
                self.t_shape = tuple(t.shape)
                return x_t

        class FakeSolver:
            def __init__(self, model):
                self.model = model

            def sample(self, x_init, **kwargs):
                del kwargs
                return self.model(x=x_init, t=torch.tensor(0.5, device=x_init.device))

        model = RecordingModel()
        sampler = ODESampler(solver_factory=FakeSolver, return_intermediates=False)
        output = sampler.sample(model, torch.zeros(4, 3))

        self.assertEqual(model.t_shape, (4,))
        self.assertEqual(output.final.shape, (4, 3))

    def test_discrete_loss_backward_and_sample_shape(self):
        from anytrain.framework.flow_matching import DiscreteFlowMatcher

        class ToyTokenFlow(nn.Module):
            def __init__(self, vocab_size: int):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, 8)
                self.head = nn.Linear(8, vocab_size)

            def forward(self, x_t, t, condition=None):
                del condition
                scale = t.view(t.shape[0], *([1] * (x_t.ndim - 1)), 1)
                return self.head(self.embedding(x_t.long()) + scale)

        vocab_size = 6
        model = ToyTokenFlow(vocab_size)
        matcher = DiscreteFlowMatcher(vocab_size)
        x_1 = torch.randint(0, vocab_size, (4, 5))

        loss = matcher.loss(model, x_1, condition=torch.zeros(4, 1))
        loss.backward()

        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.head.weight.grad)

        output = matcher.sample(model, torch.randint(0, vocab_size, (4, 5)))
        self.assertEqual(output.final.shape, x_1.shape)
        self.assertIsNotNone(output.states)
        self.assertIsNotNone(output.time_grid)

    def test_discrete_inputs_must_be_long(self):
        from anytrain.framework.flow_matching import (
            DiscreteEulerSampler,
            DiscreteFlowMatcher,
        )

        matcher = DiscreteFlowMatcher(6)

        with self.assertRaisesRegex(TypeError, "x_1"):
            matcher.loss(nn.Identity(), torch.randn(2, 3))
        with self.assertRaisesRegex(TypeError, "x_0"):
            matcher.loss(nn.Identity(), torch.zeros(2, 3, dtype=torch.long), x_0=torch.randn(2, 3))
        with self.assertRaisesRegex(TypeError, "x_0"):
            DiscreteEulerSampler(6).sample(nn.Identity(), torch.randn(2, 3))


if __name__ == "__main__":
    unittest.main()
