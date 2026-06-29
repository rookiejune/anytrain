import unittest
from dataclasses import fields

import torch
from anytrain.module.quantization import (
    EmbeddingVectorQuantizer,
    QuantizationLoss,
    VQConfig,
)


class EmbeddingVectorQuantizerTest(unittest.TestCase):
    def test_config_is_plain_data(self):
        config = VQConfig(input_dim=8, codebook_size=16, codebook_dim=4)

        field_names = {field.name for field in fields(config)}

        self.assertIn("normalize_latents", field_names)
        self.assertEqual(config.codebook_dim, 4)

    def test_forward_shapes(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=8, codebook_size=16, codebook_dim=4)
        )
        x = torch.randn(3, 5, 8)

        output = quantizer(x)

        self.assertEqual(output.quantized_latents.shape, x.shape)
        self.assertEqual(output.indices.shape, (3, 5))
        self.assertEqual(output.codebook_vectors.shape, (3, 5, 4))
        self.assertEqual(output.latents.shape, (3, 5, 4))
        self.assertIsInstance(output.loss, QuantizationLoss)
        self.assertEqual(output.loss.commitment.ndim, 0)
        self.assertEqual(output.loss.codebook.ndim, 0)

    def test_eval_returns_no_loss(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=8, codebook_size=16, codebook_dim=4)
        )
        quantizer.eval()

        output = quantizer(torch.randn(2, 8))

        self.assertIsNone(output.loss)
        self.assertEqual(output.indices.shape, (2,))

    def test_indices_round_trip(self):
        quantizer = EmbeddingVectorQuantizer(VQConfig(input_dim=4, codebook_size=8))
        indices = torch.tensor([[0, 1, 2], [3, 4, 5]])

        vectors = quantizer.indices_to_codebook_vectors(indices)
        round_trip = quantizer.codebook_vectors_to_indices(vectors)

        self.assertTrue(torch.equal(round_trip, indices))

    def test_backward_reaches_input_and_codebook_without_ema(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=8, codebook_size=16, codebook_dim=4)
        )
        x = torch.randn(3, 8, requires_grad=True)

        output = quantizer(x)
        assert output.loss is not None
        (output.quantized_latents.sum() + output.loss.total).backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertIsNotNone(quantizer.codebook.weight.grad)
        self.assertTrue(torch.isfinite(quantizer.codebook.weight.grad).all())

    def test_ema_updates_without_codebook_grad(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=8, codebook_size=16, codebook_dim=4, use_ema=True, decay=0.5)
        )
        before_counts = quantizer.ema_counts.clone()

        output = quantizer(torch.randn(6, 8))

        self.assertIsNone(output.loss)
        self.assertFalse(torch.allclose(quantizer.ema_counts, before_counts))
        self.assertFalse(quantizer.codebook.weight.requires_grad)

    def test_non_normalized_lookup(self):
        quantizer = EmbeddingVectorQuantizer(
            VQConfig(input_dim=2, codebook_size=2, normalize_latents=False)
        )
        with torch.no_grad():
            quantizer.codebook.weight.copy_(torch.tensor([[0.0, 0.0], [10.0, 0.0]]))

        output = quantizer(torch.tensor([[9.0, 0.0], [1.0, 0.0]]))

        self.assertTrue(torch.equal(output.indices, torch.tensor([1, 0])))

    def test_invalid_config_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "codebook_size"):
            VQConfig(input_dim=4, codebook_size=0)
        with self.assertRaisesRegex(ValueError, "decay"):
            VQConfig(input_dim=4, codebook_size=8, decay=1.0)


if __name__ == "__main__":
    unittest.main()
