import unittest

import torch

from anytrain.idspace import HFTokenizerAdapter, Modality, MultiTokenizer, TokenEmbedding


class FakeHFTokenizer:
    def __init__(self) -> None:
        self.vocab = {
            "<pad>": 0,
            "hello": 1,
            "world": 2,
            "<bos>": 3,
            "!": 4,
            "<eos>": 5,
        }
        self.tokens = {token_id: token for token, token_id in self.vocab.items()}
        self.special_tokens_map = {
            "pad_token": "<pad>",
            "bos_token": "<bos>",
            "eos_token": "<eos>",
        }
        self.all_special_ids = [0, 3, 5]

    def get_vocab(self) -> dict[str, int]:
        return dict(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.vocab[token] for token in text.split()]
        if add_special_tokens:
            return [self.vocab["<bos>"], *ids, self.vocab["<eos>"]]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        special_ids = set(self.all_special_ids)
        for token_id in ids:
            if skip_special_tokens and token_id in special_ids:
                continue
            tokens.append(self.tokens[token_id])
        return " ".join(tokens)


class HFTokenizerAdapterTest(unittest.TestCase):
    def test_from_tokenizer_splits_special_and_regular_tokens(self):
        adapter = HFTokenizerAdapter.from_tokenizer(FakeHFTokenizer())

        self.assertEqual(adapter.layout.special_token_ids, {"pad": 0, "bos": 3, "eos": 5})
        self.assertEqual(adapter.layout.all_special_ids, (0, 3, 5))
        self.assertEqual(adapter.layout.modality_range(Modality.TEXT).vocab_size, 6)
        self.assertEqual(adapter.layout.to_global(Modality.TEXT, [1, 2, 4]), [1, 2, 4])

    def test_multi_tokenizer_uses_regular_text_as_modality(self):
        adapter = HFTokenizerAdapter.from_tokenizer(FakeHFTokenizer())
        tokenizer = MultiTokenizer({Modality.TEXT: adapter})

        ids = tokenizer.encode(Modality.TEXT, "hello world")

        self.assertEqual(ids, [1, 2])
        self.assertEqual(tokenizer.decode(Modality.TEXT, ids), "hello world")

    def test_multi_tokenizer_accepts_hf_style_tokenizer(self):
        tokenizer = MultiTokenizer({Modality.TEXT: FakeHFTokenizer()})

        ids = tokenizer.encode(Modality.TEXT, "hello world")

        self.assertEqual(ids, [1, 2])
        self.assertEqual(tokenizer.decode(Modality.TEXT, ids), "hello world")

    def test_adapter_builds_multi_tokenizer(self):
        adapter = HFTokenizerAdapter.from_tokenizer(FakeHFTokenizer())
        tokenizer = adapter.multi_tokenizer()

        self.assertEqual(tokenizer.encode(Modality.TEXT, "hello world"), [1, 2])

    def test_rejects_hf_special_inside_text_modality(self):
        adapter = HFTokenizerAdapter.from_tokenizer(FakeHFTokenizer())
        tokenizer = adapter.multi_tokenizer()

        with self.assertRaisesRegex(ValueError, "special"):
            tokenizer.encode(Modality.TEXT, "<pad>")

        with self.assertRaisesRegex(ValueError, "special"):
            adapter.layout.to_global(Modality.TEXT, [0])
        with self.assertRaisesRegex(ValueError, "special"):
            adapter.decode([0])

    def test_weight_migration_copies_special_rows_and_keeps_text_ids(self):
        adapter = HFTokenizerAdapter.from_tokenizer(FakeHFTokenizer())
        embed = TokenEmbedding(adapter.layout, 2)
        weight = torch.tensor(
            [
                [10.0, 0.0],
                [1.0, 0.0],
                [2.0, 0.0],
                [30.0, 0.0],
                [4.0, 0.0],
                [50.0, 0.0],
            ]
        )

        with torch.no_grad():
            for name, token_id in adapter.layout.special_token_ids.items():
                embed.special_embeddings[name].copy_(weight[token_id])
            embed.modality_embeddings[Modality.TEXT].weight.copy_(weight)

        special_weight = torch.stack(
            [
                embed.special_embeddings["pad"],
                embed.special_embeddings["bos"],
                embed.special_embeddings["eos"],
            ]
        )
        self.assertTrue(
            torch.equal(
                special_weight,
                torch.tensor([[10.0, 0.0], [30.0, 0.0], [50.0, 0.0]]),
            )
        )
        self.assertTrue(torch.equal(embed.modality_embeddings[Modality.TEXT].weight, weight))


if __name__ == "__main__":
    unittest.main()
