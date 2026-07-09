import unittest
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from anytrain.idspace import (
    IdSpace,
    IdSpaceEmbedding,
    Modality,
    ModalityBlock,
)


class IdSpaceEmbeddingTest(unittest.TestCase):
    def test_forward_routes_special_and_modality_ids(self):
        space = IdSpace(
            {"pad": 0, "bos": 1},
            [ModalityBlock(Modality.TEXT, 2, 2), ModalityBlock(Modality.AUDIO, 4, 1)],
        )
        embed = IdSpaceEmbedding(space, 2)
        with torch.no_grad():
            embed.special_embeddings["pad"].copy_(torch.tensor([1.0, 0.0]))
            embed.special_embeddings["bos"].copy_(torch.tensor([2.0, 0.0]))
            embed.modality_embeddings[Modality.TEXT].weight.copy_(
                torch.tensor([[0.0, 3.0], [0.0, 4.0]])
            )
            embed.modality_embeddings[Modality.AUDIO].weight.copy_(torch.tensor([[5.0, 5.0]]))

        y = embed(torch.tensor([[0, 2, 4, 1, 3]]))

        expected = torch.tensor([[[1.0, 0.0], [0.0, 3.0], [5.0, 5.0], [2.0, 0.0], [0.0, 4.0]]])
        self.assertTrue(torch.equal(y, expected))

    def test_forward_rejects_non_tensor_ids(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.TEXT, 1, 2)])
        embed = IdSpaceEmbedding(space, 2)

        with self.assertRaisesRegex(TypeError, "torch.Tensor"):
            embed([0, 1])

    def test_head_view_selects_specials_and_modality_blocks(self):
        space = IdSpace({"pad": 0, "eos": 3}, [ModalityBlock(Modality.TEXT, 0, 5)])
        embed = IdSpaceEmbedding(
            space,
            2,
            special_embeddings=torch.nn.ParameterDict(
                {
                    "pad": torch.nn.Parameter(torch.tensor([10.0, 0.0])),
                    "eos": torch.nn.Parameter(torch.tensor([30.0, 0.0])),
                }
            ),
        )
        with torch.no_grad():
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
        head = embed.head_view()
        parent = torch.nn.Module()
        parent.embed = embed
        parent.head = head
        x = torch.tensor([[[1.0, 0.0]]])

        self.assertEqual(list(parent.head.parameters()), [])
        self.assertEqual(head.global_ids, (0, 3, 1, 2, 4))
        self.assertEqual(head.vocab_size, 5)
        self.assertEqual(head.to_head_ids([0, 1, 2, 3, 4]), [0, 2, 3, 1, 4])
        self.assertEqual(head.to_global_ids([0, 1, 2, 3, 4]), [0, 3, 1, 2, 4])
        self.assertTrue(
            torch.equal(head.to_head_ids(torch.tensor([0, 3, 4])), torch.tensor([0, 1, 4]))
        )
        self.assertTrue(
            torch.equal(head.to_global_ids(torch.tensor([0, 1, 4])), torch.tensor([0, 3, 4]))
        )
        self.assertTrue(
            torch.allclose(
                head(x),
                F.linear(x, torch.stack([embed.weight[i] for i in head.global_ids])),
            )
        )

    def test_head_view_can_select_subset_and_skip_specials(self):
        space = IdSpace(
            {"pad": 0, "bos": 1, "eos": 2},
            [ModalityBlock(Modality.TEXT, 3, 2), ModalityBlock(Modality.AUDIO, 5, 2)],
        )
        embed = IdSpaceEmbedding(space, 2)
        head = embed.head_view(special_tokens=False, modalities=[Modality.AUDIO])

        self.assertEqual(head.global_ids, (5, 6))
        self.assertEqual(head.to_head_ids([5, 6]), [0, 1])
        self.assertEqual(head.to_global_ids([0, 1]), [5, 6])
        self.assertTrue(
            torch.allclose(
                head(torch.tensor([[[1.0, 0.0]]], dtype=embed.weight.dtype)),
                F.linear(
                    torch.tensor([[[1.0, 0.0]]], dtype=embed.weight.dtype),
                    embed.modality_embeddings[Modality.AUDIO].weight,
                ),
            )
        )

    def test_head_view_uses_embedding_like_weight_property(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.AUDIO, 1, 2)])
        audio = ProjectedEmbedding(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        )
        embed = IdSpaceEmbedding(
            space,
            2,
            modality_embeddings={Modality.AUDIO: audio},
        )
        head = embed.head_view(special_tokens=False, modalities=[Modality.AUDIO])
        x = torch.tensor([[1.0, 1.0]])

        self.assertIs(embed.modality_embeddings[Modality.AUDIO], audio)
        self.assertTrue(torch.equal(embed.weight[1:3], audio.weight))
        self.assertTrue(torch.equal(embed(torch.tensor([1, 2])), audio(torch.tensor([0, 1]))))
        self.assertTrue(torch.equal(head(x), F.linear(x, audio.weight)))

    def test_weight_is_dense_global_tensor(self):
        space = IdSpace(
            {"pad": 0},
            [ModalityBlock(Modality.TEXT, 1, 2), ModalityBlock(Modality.AUDIO, 3, 1)],
        )
        embed = IdSpaceEmbedding(space, 2)
        with torch.no_grad():
            embed.special_embeddings["pad"].copy_(torch.tensor([1.0, 0.0]))
            embed.modality_embeddings[Modality.TEXT].weight.copy_(
                torch.tensor([[2.0, 0.0], [3.0, 0.0]])
            )
            embed.modality_embeddings[Modality.AUDIO].weight.copy_(torch.tensor([[4.0, 0.0]]))

        weight = embed.weight
        loss = F.linear(torch.tensor([[1.0, 0.0]]), weight).sum()
        loss.backward()

        self.assertEqual(embed.num_embeddings, 4)
        self.assertEqual(embed.vocab_size, 4)
        self.assertEqual(embed.embedding_dim, 2)
        self.assertEqual(embed.dim, 2)
        self.assertTrue(
            torch.equal(
                weight,
                torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]),
            )
        )
        self.assertIsNotNone(embed.special_embeddings["pad"].grad)
        self.assertIsNotNone(embed.modality_embeddings[Modality.TEXT].weight.grad)
        self.assertIsNotNone(embed.modality_embeddings[Modality.AUDIO].weight.grad)

    def test_cache_weights_reuses_modality_weight_inside_context(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.AUDIO, 1, 2)])
        audio = CountingEmbedding(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([[1.0, 0.0], [0.0, 2.0]]),
        )
        embed = IdSpaceEmbedding(
            space,
            2,
            modality_embeddings={Modality.AUDIO: audio},
        )
        head = embed.head_view(special_tokens=False, modalities=[Modality.AUDIO])
        x = torch.tensor([[1.0, 1.0]])
        expected_weight = audio.uncounted_weight
        expected_logits = F.linear(x, expected_weight)
        audio.weight_calls = 0

        with embed.cache_weights():
            self.assertTrue(torch.equal(embed.weight[1:3], expected_weight))
            self.assertTrue(torch.equal(head(x), expected_logits))
            self.assertEqual(audio.weight_calls, 1)

        self.assertTrue(torch.equal(head(x), expected_logits))
        self.assertEqual(audio.weight_calls, 2)

    def test_init_defaults(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.TEXT, 1, 3)])

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            embed = IdSpaceEmbedding(space, 4)

        self.assertEqual(embed.dim, 4)
        self.assertEqual(embed.embedding_dim, 4)
        self.assertEqual(embed.num_embeddings, 4)
        self.assertEqual(set(embed.special_embeddings), {"pad"})
        self.assertEqual(tuple(embed.special_embeddings["pad"].shape), (4,))
        self.assertEqual(embed.modality_embeddings[Modality.TEXT].num_embeddings, 3)
        self.assertEqual(embed.modality_embeddings[Modality.TEXT].embedding_dim, 4)

    def test_adapters_project_lookup_rows_only(self):
        space = IdSpace(
            {"pad": 0},
            [ModalityBlock(Modality.TEXT, 1, 2), ModalityBlock(Modality.AUDIO, 3, 2)],
        )
        text = nn.Embedding(2, 4)
        audio = nn.Embedding(2, 2)
        adapter = nn.Linear(2, 4, bias=False)
        with torch.no_grad():
            text.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))
            audio.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
            adapter.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 3.0], [0.0, 0.0], [0.0, 0.0]]))
            pad = nn.Parameter(torch.tensor([9.0, 9.0, 9.0, 9.0]))

        embed = IdSpaceEmbedding(
            space,
            4,
            special_embeddings=nn.ParameterDict({"pad": pad}),
            modality_embeddings={Modality.TEXT: text, Modality.AUDIO: audio},
            adapters={Modality.AUDIO: adapter},
        )

        y = embed(torch.tensor([0, 1, 3, 4]))
        expected = torch.stack(
            [
                pad,
                text.weight[0],
                adapter(audio.weight[0]),
                adapter(audio.weight[1]),
            ]
        )
        self.assertTrue(torch.allclose(y, expected))
        self.assertTrue(torch.equal(embed.modality_embeddings[Modality.AUDIO].weight, audio.weight))
        with self.assertRaisesRegex(ValueError, "weight requires all modality embedding dims"):
            _ = embed.weight

        head = embed.head_view(special_tokens=False, modalities=[Modality.AUDIO])
        self.assertEqual(head.dim, 2)
        x = torch.tensor([[1.0, 0.0]])
        self.assertTrue(torch.allclose(head(x), F.linear(x, audio.weight)))

    def test_head_view_rejects_mixed_native_dims(self):
        space = IdSpace(
            {"pad": 0},
            [ModalityBlock(Modality.TEXT, 1, 1), ModalityBlock(Modality.AUDIO, 2, 1)],
        )
        embed = IdSpaceEmbedding(
            space,
            4,
            special_embeddings=nn.ParameterDict({"pad": nn.Parameter(torch.zeros(4))}),
            modality_embeddings={
                Modality.TEXT: nn.Embedding(1, 4),
                Modality.AUDIO: nn.Embedding(1, 2),
            },
            adapters={Modality.AUDIO: nn.Linear(2, 4, bias=False)},
        )

        with self.assertRaisesRegex(ValueError, "share the same native embedding dim"):
            embed.head_view(special_tokens=False)

    def test_adapter_without_explicit_embedding_is_rejected(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.AUDIO, 1, 2)])

        with self.assertRaisesRegex(ValueError, "requires an explicit modality embedding"):
            IdSpaceEmbedding(
                space,
                4,
                special_embeddings=nn.ParameterDict({"pad": nn.Parameter(torch.zeros(4))}),
                adapters={Modality.AUDIO: nn.Linear(2, 4, bias=False)},
            )

    def test_adapter_output_dim_must_match_dim(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.AUDIO, 1, 2)])

        with self.assertRaisesRegex(ValueError, "output dim must match dim"):
            IdSpaceEmbedding(
                space,
                4,
                special_embeddings=nn.ParameterDict({"pad": nn.Parameter(torch.zeros(4))}),
                modality_embeddings={Modality.AUDIO: nn.Embedding(2, 2)},
                adapters={Modality.AUDIO: nn.Linear(2, 3, bias=False)},
            )

    def test_init_warns_when_using_default_initialization(self):
        space = IdSpace(
            {"pad": 0},
            [ModalityBlock(Modality.TEXT, 1, 2), ModalityBlock(Modality.AUDIO, 3, 2)],
        )

        with self.assertWarnsRegex(UserWarning, "PyTorch default initialization") as special:
            IdSpaceEmbedding(
                space,
                2,
                modality_embeddings={Modality.TEXT: torch.nn.Embedding(2, 2)},
            )

        self.assertIn("special embeddings", str(special.warning))

        with self.assertWarnsRegex(UserWarning, "PyTorch default initialization") as modality:
            IdSpaceEmbedding(
                space,
                2,
                special_embeddings=torch.nn.ParameterDict(
                    {"pad": torch.nn.Parameter(torch.zeros(2))}
                ),
                modality_embeddings={Modality.TEXT: torch.nn.Embedding(2, 2)},
            )

        self.assertIn("modality embeddings", str(modality.warning))

    def test_init_does_not_warn_for_explicit_embeddings(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.TEXT, 1, 2)])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            IdSpaceEmbedding(
                space,
                2,
                special_embeddings=torch.nn.ParameterDict(
                    {"pad": torch.nn.Parameter(torch.zeros(2))}
                ),
                modality_embeddings={Modality.TEXT: torch.nn.Embedding(2, 2)},
            )

        self.assertEqual(caught, [])

    def test_init_accepts_explicit_weights(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.TEXT, 1, 2)])
        special_embeddings = torch.nn.ParameterDict(
            {"pad": torch.nn.Parameter(torch.tensor([1.0, 2.0]))}
        )
        text = torch.nn.Embedding(2, 2)

        embed = IdSpaceEmbedding(
            space,
            2,
            special_embeddings=special_embeddings,
            modality_embeddings={Modality.TEXT: text},
        )

        self.assertIs(embed.special_embeddings, special_embeddings)
        self.assertIs(embed.modality_embeddings[Modality.TEXT], text)

    def test_init_infers_dim_from_explicit_weights(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.TEXT, 1, 2)])
        special_embeddings = torch.nn.ParameterDict(
            {"pad": torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))}
        )
        text = torch.nn.Embedding(2, 3)

        embed = IdSpaceEmbedding(
            space,
            special_embeddings=special_embeddings,
            modality_embeddings={Modality.TEXT: text},
        )

        self.assertEqual(embed.dim, 3)
        self.assertIs(embed.special_embeddings, special_embeddings)
        self.assertIs(embed.modality_embeddings[Modality.TEXT], text)

    def test_init_accepts_partial_explicit_weights(self):
        space = IdSpace(
            {"pad": 0, "eos": 1},
            [ModalityBlock(Modality.TEXT, 2, 2), ModalityBlock(Modality.AUDIO, 4, 3)],
        )
        special_embeddings = torch.nn.ParameterDict(
            {"pad": torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))}
        )
        text = torch.nn.Embedding(2, 3)

        embed = IdSpaceEmbedding(
            space,
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

    def test_missing_special_embedding_falls_back_to_modality_block(self):
        space = IdSpace(
            {"bos": 0, "extra": 3},
            [ModalityBlock(Modality.TEXT, 0, 2), ModalityBlock(Modality.AUDIO, 3, 1)],
        )
        special_embeddings = torch.nn.ParameterDict(
            {"extra": torch.nn.Parameter(torch.tensor([3.0, 0.0]))}
        )
        text = torch.nn.Embedding(2, 2)
        with torch.no_grad():
            text.weight.copy_(torch.tensor([[1.0, 0.0], [2.0, 0.0]]))

        embed = IdSpaceEmbedding(
            space,
            special_embeddings=special_embeddings,
            modality_embeddings={Modality.TEXT: text},
        )

        self.assertEqual(set(embed.special_embeddings), {"extra"})
        self.assertIs(embed.modality_embeddings[Modality.TEXT], text)
        self.assertTrue(
            torch.equal(
                embed(torch.tensor([0, 1, 3])), torch.tensor([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
            )
        )
        self.assertTrue(torch.equal(embed.weight[0], text.weight[0]))

        head = embed.head_view(special_tokens=["bos"], modalities=[Modality.TEXT])
        x = torch.tensor([[1.0, 0.0]])
        self.assertEqual(head.global_ids, (0, 1))
        self.assertTrue(torch.allclose(head(x), F.linear(x, text.weight[:2])))

    def test_special_outside_modality_gets_default_parameter(self):
        space = IdSpace(
            {"pad": 0, "bos": 10},
            [ModalityBlock(Modality.TEXT, 1, 2)],
        )

        with self.assertWarnsRegex(UserWarning, "special embeddings"):
            embed = IdSpaceEmbedding(space, 2)

        self.assertEqual(set(embed.special_embeddings), {"pad", "bos"})
        self.assertEqual(tuple(embed.special_embeddings["pad"].shape), (2,))
        self.assertEqual(tuple(embed.special_embeddings["bos"].shape), (2,))

    def test_init_requires_dim_without_explicit_weights(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.TEXT, 1, 2)])

        with self.assertRaisesRegex(ValueError, "dim must be provided"):
            IdSpaceEmbedding(space)

    def test_init_rejects_unknown_special_embedding(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.TEXT, 1, 2)])

        with self.assertRaisesRegex(ValueError, "special_embeddings keys"):
            IdSpaceEmbedding(
                space,
                2,
                special_embeddings=torch.nn.ParameterDict(
                    {"eos": torch.nn.Parameter(torch.zeros(2))}
                ),
            )

    def test_init_rejects_unknown_modality_embedding(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.TEXT, 1, 2)])

        with self.assertRaisesRegex(ValueError, "modality_embeddings keys"):
            IdSpaceEmbedding(
                space,
                2,
                modality_embeddings={Modality.AUDIO: torch.nn.Embedding(2, 2)},
            )

    def test_init_rejects_mismatched_modality_embedding(self):
        space = IdSpace({"pad": 0}, [ModalityBlock(Modality.TEXT, 1, 2)])

        with self.assertRaisesRegex(ValueError, "num_embeddings"):
            IdSpaceEmbedding(
                space,
                2,
                modality_embeddings={Modality.TEXT: torch.nn.Embedding(3, 2)},
            )


class ProjectedEmbedding(nn.Module):
    def __init__(self, base_weight: torch.Tensor, shift_weight: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("base_weight", base_weight)
        self.shift = nn.Linear(shift_weight.size(1), shift_weight.size(1), bias=False)
        with torch.no_grad():
            self.shift.weight.copy_(shift_weight)
        self.num_embeddings = base_weight.size(0)
        self.embedding_dim = base_weight.size(1)

    @property
    def weight(self) -> torch.Tensor:
        return self.base_weight + self.shift(self.base_weight)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.weight[ids]


class CountingEmbedding(ProjectedEmbedding):
    def __init__(self, base_weight: torch.Tensor, shift_weight: torch.Tensor) -> None:
        super().__init__(base_weight, shift_weight)
        self.weight_calls = 0

    @property
    def weight(self) -> torch.Tensor:
        self.weight_calls += 1
        return super().weight

    @property
    def uncounted_weight(self) -> torch.Tensor:
        return super().weight


if __name__ == "__main__":
    unittest.main()
