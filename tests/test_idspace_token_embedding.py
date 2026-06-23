import unittest

import torch
import torch.nn.functional as F

from anytrain.idspace import Modality, ModalityRange, TokenEmbedding, TokenLayout


class TokenEmbeddingTest(unittest.TestCase):
    def test_forward_routes_special_and_modality_ids(self):
        layout = TokenLayout(
            {"pad": 0, "bos": 1},
            [ModalityRange(Modality.TEXT, 2, 2), ModalityRange(Modality.AUDIO, 4, 1)],
        )
        embed = TokenEmbedding(layout, 2)
        with torch.no_grad():
            embed.special_embeddings["pad"].copy_(torch.tensor([1.0, 0.0]))
            embed.special_embeddings["bos"].copy_(torch.tensor([2.0, 0.0]))
            embed.modality_embeddings[Modality.TEXT].weight.copy_(
                torch.tensor([[0.0, 3.0], [0.0, 4.0]])
            )
            embed.modality_embeddings[Modality.AUDIO].weight.copy_(torch.tensor([[5.0, 5.0]]))

        y = embed(torch.tensor([[0, 2, 4, 1, 3]]))

        expected = torch.tensor(
            [[[1.0, 0.0], [0.0, 3.0], [5.0, 5.0], [2.0, 0.0], [0.0, 4.0]]]
        )
        self.assertTrue(torch.equal(y, expected))

    def test_head_matches_compact_weight_without_registering_params(self):
        layout = TokenLayout({"pad": 0, "eos": 3}, [ModalityRange(Modality.TEXT, 0, 5)])
        embed = TokenEmbedding(layout, 2)
        with torch.no_grad():
            embed.special_embeddings["pad"].copy_(torch.tensor([10.0, 0.0]))
            embed.special_embeddings["eos"].copy_(torch.tensor([30.0, 0.0]))
            embed.modality_embeddings[Modality.TEXT].weight.copy_(
                torch.tensor(
                    [
                        [1.0, 0.0],
                        [2.0, 0.0],
                        [3.0, 0.0],
                        [4.0, 0.0],
                        [5.0, 0.0],
                    ]
            )
        )
        head = embed.as_head()
        parent = torch.nn.Module()
        parent.embed = embed
        parent.head = head
        x = torch.tensor([[[1.0, 0.0]]])

        self.assertEqual(list(parent.head.parameters()), [])
        self.assertEqual(head.global_ids, (0, 3, 1, 2, 4))
        self.assertEqual(head.vocab_size, 5)
        self.assertEqual(head.to_head_ids([0, 1, 2, 3, 4]), [0, 2, 3, 1, 4])
        self.assertEqual(head.to_global_ids([0, 1, 2, 3, 4]), [0, 3, 1, 2, 4])
        self.assertTrue(torch.equal(head.to_head_ids(torch.tensor([0, 3, 4])), torch.tensor([0, 1, 4])))
        self.assertTrue(
            torch.equal(head.to_global_ids(torch.tensor([0, 1, 4])), torch.tensor([0, 3, 4]))
        )
        self.assertTrue(
            torch.allclose(
                head(x),
                F.linear(x, torch.stack([embed.dense_weight()[i] for i in head.global_ids])),
            )
        )

    def test_head_can_select_subset_and_skip_specials(self):
        layout = TokenLayout(
            {"pad": 0, "bos": 1, "eos": 2},
            [ModalityRange(Modality.TEXT, 3, 2), ModalityRange(Modality.AUDIO, 5, 2)],
        )
        embed = TokenEmbedding(layout, 2)
        head = embed.as_head(special_tokens=False, modalities=[Modality.AUDIO])

        self.assertEqual(head.global_ids, (5, 6))
        self.assertEqual(head.to_head_ids([5, 6]), [0, 1])
        self.assertEqual(head.to_global_ids([0, 1]), [5, 6])
        self.assertTrue(
            torch.allclose(
                head(torch.tensor([[[1.0, 0.0]]], dtype=embed.dense_weight().dtype)),
                F.linear(
                    torch.tensor([[[1.0, 0.0]]], dtype=embed.dense_weight().dtype),
                    embed.modality_embeddings[Modality.AUDIO].weight,
                ),
            )
        )

    def test_init_defaults(self):
        layout = TokenLayout({"pad": 0}, [ModalityRange(Modality.TEXT, 1, 3)])

        embed = TokenEmbedding(layout, 4, dtype=torch.float64)

        self.assertEqual(embed.dim, 4)
        self.assertEqual(set(embed.special_embeddings), {"pad"})
        self.assertEqual(tuple(embed.special_embeddings["pad"].shape), (4,))
        self.assertEqual(embed.special_embeddings["pad"].dtype, torch.float64)
        self.assertEqual(embed.modality_embeddings[Modality.TEXT].num_embeddings, 3)
        self.assertEqual(embed.modality_embeddings[Modality.TEXT].embedding_dim, 4)
        self.assertEqual(embed.modality_embeddings[Modality.TEXT].weight.dtype, torch.float64)

    def test_init_accepts_explicit_weights(self):
        layout = TokenLayout({"pad": 0}, [ModalityRange(Modality.TEXT, 1, 2)])
        special_embeddings = torch.nn.ParameterDict(
            {"pad": torch.nn.Parameter(torch.tensor([1.0, 2.0]))}
        )
        text = torch.nn.Embedding(2, 2)

        embed = TokenEmbedding(
            layout,
            2,
            special_embeddings=special_embeddings,
            modality_embeddings={Modality.TEXT: text},
        )

        self.assertIs(embed.special_embeddings, special_embeddings)
        self.assertIs(embed.modality_embeddings[Modality.TEXT], text)

    def test_init_infers_dim_from_explicit_weights(self):
        layout = TokenLayout({"pad": 0}, [ModalityRange(Modality.TEXT, 1, 2)])
        special_embeddings = torch.nn.ParameterDict(
            {"pad": torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))}
        )
        text = torch.nn.Embedding(2, 3)

        embed = TokenEmbedding(
            layout,
            None,
            special_embeddings=special_embeddings,
            modality_embeddings={Modality.TEXT: text},
        )

        self.assertEqual(embed.dim, 3)
        self.assertIs(embed.special_embeddings, special_embeddings)
        self.assertIs(embed.modality_embeddings[Modality.TEXT], text)

    def test_init_accepts_partial_explicit_weights(self):
        layout = TokenLayout(
            {"pad": 0, "eos": 1},
            [ModalityRange(Modality.TEXT, 2, 2), ModalityRange(Modality.AUDIO, 4, 3)],
        )
        special_embeddings = torch.nn.ParameterDict(
            {"pad": torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))}
        )
        text = torch.nn.Embedding(2, 3)

        embed = TokenEmbedding(
            layout,
            None,
            special_embeddings=special_embeddings,
            modality_embeddings={Modality.TEXT: text},
        )

        self.assertEqual(embed.dim, 3)
        self.assertIs(embed.special_embeddings, special_embeddings)
        self.assertIs(embed.modality_embeddings[Modality.TEXT], text)
        self.assertIn("eos", embed.special_embeddings)
        self.assertEqual(tuple(embed.special_embeddings["eos"].shape), (3,))
        self.assertEqual(embed.modality_embeddings[Modality.AUDIO].num_embeddings, 3)
        self.assertEqual(embed.modality_embeddings[Modality.AUDIO].embedding_dim, 3)

    def test_init_requires_dim_without_explicit_weights(self):
        layout = TokenLayout({"pad": 0}, [ModalityRange(Modality.TEXT, 1, 2)])

        with self.assertRaisesRegex(ValueError, "dim must be provided"):
            TokenEmbedding(layout, None)

    def test_init_rejects_unknown_special_embedding(self):
        layout = TokenLayout({"pad": 0}, [ModalityRange(Modality.TEXT, 1, 2)])

        with self.assertRaisesRegex(ValueError, "special_embeddings keys"):
            TokenEmbedding(
                layout,
                2,
                special_embeddings=torch.nn.ParameterDict(
                    {"eos": torch.nn.Parameter(torch.zeros(2))}
                ),
            )

    def test_init_rejects_unknown_modality_embedding(self):
        layout = TokenLayout({"pad": 0}, [ModalityRange(Modality.TEXT, 1, 2)])

        with self.assertRaisesRegex(ValueError, "modality_embeddings keys"):
            TokenEmbedding(
                layout,
                2,
                modality_embeddings={Modality.AUDIO: torch.nn.Embedding(2, 2)},
            )

    def test_init_rejects_mismatched_modality_embedding(self):
        layout = TokenLayout({"pad": 0}, [ModalityRange(Modality.TEXT, 1, 2)])

        with self.assertRaisesRegex(ValueError, "num_embeddings"):
            TokenEmbedding(
                layout,
                2,
                modality_embeddings={Modality.TEXT: torch.nn.Embedding(3, 2)},
            )


if __name__ == "__main__":
    unittest.main()
