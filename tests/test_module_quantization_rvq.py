import unittest

import torch

from anytrain._compat import strict_zip
from anytrain.module.quantization import (
    ResidualVectorQuantizer,
    RVQConfig,
    VQConfig,
)


class ResidualVectorQuantizerTest(unittest.TestCase):
    def test_forward_shapes(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=8, num_codebooks=3, codebook_size=16, codebook_dim=8)
        )
        x = torch.randn(2, 5, 8)

        output = quantizer(x)

        self.assertEqual(output.quantized_latents.shape, x.shape)
        self.assertEqual(output.indices.shape, (2, 5, 3))
        self.assertEqual(output.codebook_vectors.shape, (2, 5, 3, 8))
        self.assertEqual(output.latents.shape, (2, 5, 3, 8))
        self.assertEqual(output.active_codebook_mask.shape, (2, 5, 3))
        self.assertIsNotNone(output.loss)

    def test_num_active_codebooks_controls_output_width(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=8, num_codebooks=4, codebook_size=16)
        )
        quantizer.eval()

        output = quantizer(torch.randn(2, 8), num_active_codebooks=2)

        self.assertEqual(output.indices.shape, (2, 2))
        self.assertEqual(output.codebook_vectors.shape, (2, 2, 8))
        self.assertTrue(output.active_codebook_mask.all())

    def test_indices_round_trip(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        indices = torch.tensor([[[0, 1], [2, 3]]])

        vectors = quantizer.indices_to_codebook_vectors(indices)
        round_trip = quantizer.codebook_vectors_to_indices(vectors)

        self.assertTrue(torch.equal(round_trip, indices))

    def test_project_codebook_vectors_sums_projected_books(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        indices = torch.tensor([[0, 1], [2, 3]])
        vectors = quantizer.indices_to_codebook_vectors(indices)

        projected = quantizer.project_codebook_vectors(vectors)

        self.assertEqual(projected.shape, (2, 4))

    def test_latents_to_codebook_vectors_has_no_training_side_effects(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(
                input_dim=4,
                num_codebooks=3,
                codebook_size=8,
                use_ema=True,
                dropout=1.0,
            )
        )
        before_counts = [book.ema_counts.clone() for book in quantizer.quantizers]

        vectors = quantizer.latents_to_codebook_vectors(torch.randn(16, 4))

        self.assertEqual(vectors.shape, (16, 3, 4))
        for book, before in strict_zip(quantizer.quantizers, before_counts):
            self.assertTrue(torch.equal(book.ema_counts, before))

    def test_dropout_marks_inactive_codebooks_during_training(self):
        torch.manual_seed(0)
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=4, codebook_size=8, dropout=1.0)
        )

        output = quantizer(torch.randn(64, 4))

        self.assertTrue((output.indices == -1).any())
        self.assertFalse(output.active_codebook_mask.all())

    def test_dropout_has_no_effect_with_one_active_codebook(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=4, codebook_size=8, dropout=1.0)
        )

        output = quantizer(torch.randn(8, 4), num_active_codebooks=1)

        self.assertEqual(output.indices.shape, (8, 1))
        self.assertTrue((output.indices >= 0).all())
        self.assertTrue(output.active_codebook_mask.all())

    def test_eval_disables_dropout(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=4, codebook_size=8, dropout=1.0)
        )
        quantizer.eval()

        output = quantizer(torch.randn(8, 4))

        self.assertTrue((output.indices >= 0).all())
        self.assertTrue(output.active_codebook_mask.all())

    def test_backward_reaches_input(self):
        quantizer = ResidualVectorQuantizer(
            RVQConfig.from_kwargs(input_dim=4, num_codebooks=2, codebook_size=8)
        )
        x = torch.randn(3, 4, requires_grad=True)

        output = quantizer(x)
        assert output.loss is not None
        (output.quantized_latents.sum() + output.loss.total).backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_invalid_config_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "at least one"):
            RVQConfig(vq_configs=[])
        with self.assertRaisesRegex(ValueError, "uniform codebook_dim"):
            RVQConfig(
                vq_configs=[
                    VQConfig(input_dim=4, codebook_size=8, codebook_dim=2),
                    VQConfig(input_dim=4, codebook_size=8, codebook_dim=3),
                ]
            )


if __name__ == "__main__":
    unittest.main()
