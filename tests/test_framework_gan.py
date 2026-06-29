import sys
import unittest

import torch
from anytrain.framework.gan import GAN, Loss, Preset, Reduction
from torch import nn


class IdentityDiscriminator(nn.Module):
    def forward(self, x: torch.Tensor):
        return [[x]]


class ScaleBranchDiscriminator(nn.Module):
    def forward(self, x: torch.Tensor):
        return [[x], [2 * x]]


class TensorDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x: torch.Tensor):
        return [[self.linear(x).squeeze(-1)]]


class RawTensorDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class FeatureDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1, bias=False)

    def forward(self, x: torch.Tensor):
        feature = x.unsqueeze(1)
        logits = self.linear(x).squeeze(-1)
        return [[feature, logits]]


class LogitOnlyOrFeatureDiscriminator(nn.Module):
    def forward(self, x: torch.Tensor):
        logits = x.mean(dim=-1)
        if x.mean().item() < 0:
            return [[logits]]
        return [[x.unsqueeze(1), logits]]


class LossTest(unittest.TestCase):
    def test_audio_preset_is_lazy_on_root_import(self):
        self.assertNotIn("anytrain.framework.gan.audio", sys.modules)

    def test_hinge_values(self):
        loss_fn = Loss(IdentityDiscriminator(), gan=GAN.Hinge)
        real = torch.tensor([2.0, 0.0])
        fake = torch.tensor([-2.0, 0.0])

        d_loss, d_details = loss_fn.discriminator_loss(fake, real)
        g_loss, g_details = loss_fn.generator_loss(fake)

        self.assertTrue(torch.isclose(d_loss, torch.tensor(1.0)))
        self.assertTrue(torch.isclose(d_details["real"], torch.tensor(0.5)))
        self.assertTrue(torch.isclose(d_details["fake"], torch.tensor(0.5)))
        self.assertTrue(torch.isclose(g_loss, torch.tensor(1.0)))
        self.assertTrue(torch.isclose(g_details["adv"], torch.tensor(1.0)))

    def test_lsgan_and_wgan_values(self):
        logits = torch.tensor([-1.0, 1.0])

        lsgan = Loss(IdentityDiscriminator(), gan="lsgan")
        d_loss, d_details = lsgan.discriminator_loss(logits, logits)
        g_loss, g_details = lsgan.generator_loss(logits)
        self.assertTrue(torch.isclose(d_loss, torch.tensor(3.0)))
        self.assertTrue(torch.isclose(d_details["fake"], torch.tensor(1.0)))
        self.assertTrue(torch.isclose(d_details["real"], torch.tensor(2.0)))
        self.assertTrue(torch.isclose(g_loss, torch.tensor(2.0)))
        self.assertTrue(torch.isclose(g_details["adv"], torch.tensor(2.0)))

        wgan = Loss(IdentityDiscriminator(), gan="wgan", gp_weight=0.0)
        d_loss, d_details = wgan.discriminator_loss(logits, logits)
        g_loss, g_details = wgan.generator_loss(logits)
        self.assertTrue(torch.isclose(d_loss, torch.tensor(0.0)))
        self.assertTrue(torch.isclose(d_details["fake"], torch.tensor(0.0)))
        self.assertTrue(torch.isclose(d_details["real"], torch.tensor(0.0)))
        self.assertTrue(torch.isclose(g_loss, torch.tensor(0.0)))
        self.assertTrue(torch.isclose(g_details["adv"], torch.tensor(0.0)))

    def test_default_reduction_is_mean(self):
        loss_fn = Loss(ScaleBranchDiscriminator(), gan=GAN.LSGAN)
        fake = torch.zeros(2)
        real = torch.ones(2)

        d_loss, details = loss_fn.discriminator_loss(fake, real)

        self.assertTrue(torch.isclose(d_loss, torch.tensor(0.5)))
        self.assertTrue(torch.isclose(details["real"], torch.tensor(0.5)))
        self.assertTrue(torch.isclose(details["fake"], torch.tensor(0.0)))

    def test_sum_reduction(self):
        loss_fn = Loss(ScaleBranchDiscriminator(), gan=GAN.LSGAN, reduction=Reduction.Sum)
        fake = torch.zeros(2)
        real = torch.ones(2)

        d_loss, details = loss_fn.discriminator_loss(fake, real)

        self.assertTrue(torch.isclose(d_loss, torch.tensor(1.0)))
        self.assertTrue(torch.isclose(details["real"], torch.tensor(1.0)))
        self.assertTrue(torch.isclose(details["fake"], torch.tensor(0.0)))

    def test_raw_tensor_output_is_rejected(self):
        loss_fn = Loss(RawTensorDiscriminator())
        fake = torch.randn(2, 4)
        real = torch.randn(2, 4)

        with self.assertRaisesRegex(TypeError, r"\[\[logits\]\]"):
            loss_fn.discriminator_loss(fake, real)

    def test_discriminator_loss_detaches_fake(self):
        discriminator = TensorDiscriminator()
        loss_fn = Loss(discriminator)
        fake = torch.randn(3, 4, requires_grad=True)
        real = torch.randn(3, 4)

        loss, details = loss_fn.discriminator_loss(fake, real)
        loss.backward()

        self.assertIn("real", details)
        self.assertIn("fake", details)
        self.assertIsNone(fake.grad)
        self.assertIsNotNone(discriminator.linear.weight.grad)
        self.assertFalse(details["real"].requires_grad)

    def test_generator_loss_keeps_fake_grad(self):
        discriminator = TensorDiscriminator()
        loss_fn = Loss(discriminator)
        fake = torch.randn(3, 4, requires_grad=True)

        loss, details = loss_fn.generator_loss(fake)
        loss.backward()

        self.assertIn("adv", details)
        self.assertIsNotNone(fake.grad)

    def test_generator_feature_matching_requires_real(self):
        loss_fn = Loss(FeatureDiscriminator(), feature_weight=1.0)
        fake = torch.randn(2, 4)

        with self.assertRaisesRegex(ValueError, "real"):
            loss_fn.generator_loss(fake)

    def test_generator_feature_matching_requires_features(self):
        loss_fn = Loss(TensorDiscriminator(), feature_weight=1.0)
        fake = torch.randn(2, 4)
        real = torch.randn(2, 4)

        with self.assertRaisesRegex(ValueError, "feature maps"):
            loss_fn.generator_loss(fake, real)

    def test_generator_feature_matching_adds_weighted_feature_and_detaches_real(self):
        loss_fn = Loss(FeatureDiscriminator(), feature_weight=0.5)
        fake = torch.randn(2, 4, requires_grad=True)
        real = torch.randn(2, 4, requires_grad=True)

        loss, details = loss_fn.generator_loss(fake, real)
        loss.backward()

        self.assertTrue(loss.requires_grad)
        self.assertIn("feature", details)
        self.assertEqual(details["feature_weight"], 0.5)
        self.assertIsNotNone(fake.grad)
        self.assertIsNone(real.grad)

    def test_feature_matching_rejects_mismatched_feature_count(self):
        loss_fn = Loss(LogitOnlyOrFeatureDiscriminator(), feature_weight=1.0)
        fake = -torch.ones(2, 4)
        real = torch.ones(2, 4)

        with self.assertRaisesRegex(ValueError, "feature count"):
            loss_fn.generator_loss(fake, real)

    def test_wgan_uses_default_gradient_penalty(self):
        discriminator = TensorDiscriminator()
        loss_fn = Loss(discriminator, gan=GAN.WGAN)
        fake = torch.randn(3, 4)
        real = torch.randn(3, 4)

        loss, details = loss_fn.discriminator_loss(fake, real)
        loss.backward()

        self.assertIn("gp", details)
        self.assertEqual(details["gp"].ndim, 0)
        self.assertIsNotNone(discriminator.linear.weight.grad)

    def test_non_wgan_does_not_use_default_gradient_penalty(self):
        loss_fn = Loss(TensorDiscriminator(), gan=GAN.Hinge)
        fake = torch.randn(3, 4)
        real = torch.randn(3, 4)

        _, details = loss_fn.discriminator_loss(fake, real)

        self.assertNotIn("gp", details)

    def test_invalid_reduction_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "valid Reduction"):
            Loss(TensorDiscriminator(), reduction="none")

    def test_from_preset_dac_returns_loss_with_feature_discriminator(self):
        dac_kwargs = {
            "in_channels": 1,
            "periods": (2,),
            "n_ffts": (16,),
            "bands": (0.25, 0.5, 0.75),
            "mpd_dim": 2,
            "mrd_dim": 2,
        }
        loss_fn = Loss.from_preset(
            Preset.DAC,
            feature_weight=0.1,
            **dac_kwargs,
        )
        fake = torch.randn(2, 1, 64, requires_grad=True)
        real = torch.randn(2, 1, 64)

        d_loss, d_details = loss_fn.discriminator_loss(fake, real)
        g_loss, g_details = loss_fn.generator_loss(fake, real)
        (d_loss + g_loss).backward()

        self.assertIn("real", d_details)
        self.assertIn("fake", d_details)
        self.assertIn("adv", g_details)
        self.assertIn("feature", g_details)
        self.assertIsNotNone(fake.grad)

    def test_dac_discriminator_output_contract(self):
        from anytrain.framework.gan.audio import DACDiscriminator

        discriminator = DACDiscriminator(
            in_channels=1,
            periods=(2,),
            n_ffts=(16,),
            bands=(0.25, 0.5, 0.75),
            mpd_dim=2,
            mrd_dim=2,
        )
        output = discriminator(torch.randn(2, 1, 64))

        self.assertEqual(len(output), 2)
        self.assertTrue(all(len(branch) > 1 for branch in output))
        self.assertTrue(
            all(isinstance(tensor, torch.Tensor) for branch in output for tensor in branch)
        )

    def test_from_preset_requires_preset_enum(self):
        with self.assertRaisesRegex(TypeError, "Preset"):
            Loss.from_preset("dac")


if __name__ == "__main__":
    unittest.main()
