import tempfile
import unittest

import torch

from anytrain.tokenizer.int_bpe import IntBPE

try:
    import tokenizers
except ImportError:
    tokenizers = None


def state():
    return {
        "tokens": {
            "1": [1],
            "2": [2],
            "3": [3],
            "4": [1, 2],
            "5": [1, 2, 3],
        },
        "merges": [
            {"left": 1, "right": 2, "token_id": 4},
            {"left": 4, "right": 3, "token_id": 5},
        ],
        "strict": True,
    }


@unittest.skipIf(tokenizers is None, "tokenizers is not installed")
class BPETest(unittest.TestCase):
    def test_from_dict_accepts_int_tokens_and_merges(self):
        bpe = IntBPE.from_dict(state())

        self.assertEqual(bpe.encode_units([1, 2, 3]), [5])
        self.assertEqual(bpe.expand_ids([5]), [1, 2, 3])

    def test_from_pretrained_round_trip(self):
        bpe = IntBPE.from_dict(state())

        with tempfile.TemporaryDirectory() as tmp:
            bpe.save_pretrained(tmp)
            loaded = IntBPE.from_pretrained(tmp)

        self.assertEqual(loaded.to_dict(), bpe.to_dict())
        self.assertEqual(loaded.encode_units([1, 2, 3]), [5])

    def test_from_pretrained_accepts_state_file(self):
        bpe = IntBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)

        with tempfile.TemporaryDirectory() as tmp:
            out = bpe.save_pretrained(tmp)
            loaded = IntBPE.from_pretrained(out / "int_bpe.json")

        self.assertEqual(loaded.to_dict(), bpe.to_dict())

    def test_uninitialized_instance_fails_clearly(self):
        bpe = IntBPE()

        with self.assertRaisesRegex(ValueError, "not initialized"):
            bpe.encode_units([1, 2, 3])

    def test_train_merges_int_sequences_and_round_trips(self):
        bpe = IntBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)

        self.assertEqual(bpe.tokens[4], (1, 2))
        self.assertEqual(bpe.tokens[5], (1, 2, 3))
        self.assertEqual(bpe.encode_units([1, 2, 3]), [5])

        token_ids = bpe.encode_units([1, 2, 1, 2, 3])

        self.assertEqual(token_ids, [4, 5])
        self.assertEqual(bpe.expand_ids(token_ids), [1, 2, 1, 2, 3])

    def test_expand_with_counts_returns_unit_ids_and_lengths(self):
        bpe = IntBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)

        unit_ids, counts = bpe.expand_with_counts([4, 5])

        self.assertEqual(unit_ids, [1, 2, 1, 2, 3])
        self.assertEqual(counts, [2, 3])

    def test_repeat_interleave_uses_expansion_counts(self):
        bpe = IntBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)
        x = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        token_ids = torch.tensor([4, 5])

        expanded_x, unit_ids = bpe.repeat_interleave(x, token_ids, dim=0)

        expected_x = torch.tensor(
            [
                [10.0, 11.0],
                [10.0, 11.0],
                [20.0, 21.0],
                [20.0, 21.0],
                [20.0, 21.0],
            ]
        )
        self.assertTrue(torch.equal(expanded_x, expected_x))
        self.assertTrue(torch.equal(unit_ids, torch.tensor([1, 2, 1, 2, 3])))

    def test_strict_rejects_empty_inputs_and_unknown_ids(self):
        bpe = IntBPE.train([[1, 2, 3]], num_merges=1)

        with self.assertRaisesRegex(ValueError, "empty"):
            IntBPE.train([])
        with self.assertRaisesRegex(ValueError, "empty"):
            IntBPE.train([[]])
        with self.assertRaisesRegex(ValueError, "empty"):
            bpe.encode_units([])
        with self.assertRaisesRegex(KeyError, "unknown unit"):
            bpe.encode_units([9])
        with self.assertRaisesRegex(KeyError, "unknown token_id"):
            bpe.expand_ids([9])

    def test_non_strict_keeps_unknown_ids_atomic(self):
        bpe = IntBPE.train([[1, 2, 3]], num_merges=1, strict=False)

        self.assertEqual(bpe.encode_units([]), [])
        self.assertEqual(bpe.encode_units([9]), [9])
        self.assertEqual(bpe.expand_ids([9]), [9])

    def test_repeat_interleave_rejects_misaligned_inputs(self):
        bpe = IntBPE.train([[1, 2, 3]], num_merges=1)

        with self.assertRaisesRegex(ValueError, "1D"):
            bpe.repeat_interleave(torch.randn(2, 3), torch.tensor([[1, 2]]), dim=0)
        with self.assertRaisesRegex(ValueError, "align"):
            bpe.repeat_interleave(torch.randn(2, 3), torch.tensor([1]), dim=0)
        with self.assertRaisesRegex(ValueError, "out of range"):
            bpe.repeat_interleave(torch.randn(2, 3), torch.tensor([1, 2]), dim=2)

    def test_int_bpe_holds_tokenizers_model(self):
        bpe = IntBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)

        self.assertEqual(bpe.tokens[4], (1, 2))
        self.assertEqual(bpe.model.id_to_token(4), bpe.token_text(4))
        self.assertEqual(bpe.model.id_to_token(5), bpe.token_text(5))

    def test_int_bpe_encode_matches_core(self):
        bpe = IntBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)
        units = [1, 2, 1, 2, 3]

        self.assertEqual(bpe.encode_units(units), bpe.core.encode_units(units))
        self.assertEqual(bpe.expand_ids(bpe.encode_units(units)), units)

        tokenizer = bpe.tokenizer()
        encoded = tokenizer.encode(bpe.units_text(units))

        self.assertEqual(encoded.ids, bpe.core.encode_units(units))

    def test_eval_reports_expected_compression(self):
        bpe = IntBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)

        stats = bpe.eval([[1, 2, 1, 2, 3], [1, 2, 3]])

        self.assertEqual(stats.num_sequences, 2)
        self.assertEqual(stats.original_tokens, 8)
        self.assertEqual(stats.encoded_tokens, 3)
        self.assertEqual(stats.mean_original_length, 4.0)
        self.assertEqual(stats.mean_encoded_length, 1.5)
        self.assertAlmostEqual(stats.compression_ratio, 3 / 8)
        self.assertAlmostEqual(stats.compression_factor, 8 / 3)
        self.assertAlmostEqual(stats.compression_gain, 5 / 8)

    def test_eval_rejects_empty_corpus(self):
        bpe = IntBPE.train([[1, 2, 3]], num_merges=1)

        with self.assertRaisesRegex(ValueError, "corpus"):
            bpe.eval([])


if __name__ == "__main__":
    unittest.main()
