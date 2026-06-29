import io
import tempfile
import unittest
from contextlib import redirect_stderr

import torch

from anytrain.tokenizer.codec_bpe import CodecBPE

try:
    import tokenizers
except ImportError:
    tokenizers = None


def state():
    return {
        "tokens": {
            "0": [1],
            "1": [2],
            "2": [3],
            "3": [1, 2],
            "4": [1, 2, 3],
        },
        "merges": [
            {"left": 0, "right": 1, "token_id": 3},
            {"left": 3, "right": 2, "token_id": 4},
        ],
        "strict": True,
    }


@unittest.skipIf(tokenizers is None, "tokenizers is not installed")
class BPETest(unittest.TestCase):
    def test_from_dict_accepts_int_tokens_and_merges(self):
        bpe = CodecBPE.from_dict(state())

        self.assertEqual(bpe.encode_units([1, 2, 3]), [4])
        self.assertEqual(bpe.expand_ids([4]), [1, 2, 3])

    def test_from_pretrained_round_trip(self):
        bpe = CodecBPE.from_dict(state())

        with tempfile.TemporaryDirectory() as tmp:
            bpe.save_pretrained(tmp)
            loaded = CodecBPE.from_pretrained(tmp)

        self.assertEqual(loaded.to_dict(), bpe.to_dict())
        self.assertEqual(loaded.encode_units([1, 2, 3]), [4])
        self.assertEqual(loaded.vocab_size, 5)

    def test_from_pretrained_accepts_state_file(self):
        bpe = CodecBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)

        with tempfile.TemporaryDirectory() as tmp:
            out = bpe.save_pretrained(tmp)
            loaded = CodecBPE.from_pretrained(out / "codec_bpe.json")

        self.assertEqual(loaded.to_dict(), bpe.to_dict())

    def test_uninitialized_instance_fails_clearly(self):
        bpe = CodecBPE()

        with self.assertRaisesRegex(ValueError, "not initialized"):
            bpe.encode_units([1, 2, 3])

    def test_train_merges_int_sequences_and_round_trips(self):
        bpe = CodecBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)

        self.assertEqual(bpe.vocab_size, 5)
        self.assertEqual(bpe.tokens[3], (1, 2))
        self.assertEqual(bpe.tokens[4], (1, 2, 3))
        self.assertEqual(bpe.encode_units([1, 2, 3]), [4])

        token_ids = bpe.encode_units([1, 2, 1, 2, 3])

        self.assertEqual(token_ids, [3, 4])
        self.assertEqual(bpe.expand_ids(token_ids), [1, 2, 1, 2, 3])

    def test_train_supports_longcat_scale_unit_vocab(self):
        units = list(range(8192))

        bpe = CodecBPE.train([units], vocab_size=8192)

        encoded = bpe.encode_units([0, 4096, 8191])
        self.assertEqual(encoded, [0, 4096, 8191])
        self.assertEqual(bpe.expand_ids(encoded), [0, 4096, 8191])
        self.assertEqual(len(bpe.token_text(8191)), 1)

    def test_train_interprets_vocab_size_as_compact_size(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            CodecBPE.train([[100, 101, 102]], vocab_size=2)

        bpe = CodecBPE.train([[100, 101, 100, 101]], vocab_size=4)

        self.assertEqual(bpe.vocab_size, 3)
        self.assertEqual(bpe.tokens[0], (100,))
        self.assertEqual(bpe.tokens[1], (101,))
        self.assertEqual(bpe.tokens[2], (100, 101))
        self.assertEqual(bpe.encode_units([100, 101]), [2])

    def test_train_compacts_sparse_unit_ids(self):
        bpe = CodecBPE.train([[100, 101, 100, 101]], vocab_size=3)

        self.assertEqual(bpe.vocab_size, 3)
        self.assertEqual(bpe.encode_units([100, 101, 100, 101]), [2, 2])

    def test_train_merges_tuple_units_and_round_trips(self):
        corpus = [
            [(1, 4), (2, 7), (1, 4), (2, 7), (3, 8)],
            [(1, 4), (2, 7), (3, 8)],
        ]

        bpe = CodecBPE.train(corpus, num_merges=2)

        self.assertEqual(bpe.vocab_size, 5)
        self.assertEqual(bpe.tokens[0], ((1, 4),))
        self.assertEqual(bpe.tokens[3], ((1, 4), (2, 7)))
        self.assertEqual(bpe.tokens[4], ((1, 4), (2, 7), (3, 8)))
        self.assertEqual(bpe.encode_units([(1, 4), (2, 7), (3, 8)]), [4])

        token_ids = bpe.encode_units(corpus[0])

        self.assertEqual(token_ids, [3, 4])
        self.assertEqual(bpe.expand_ids(token_ids), corpus[0])

    def test_tuple_units_round_trip_through_pretrained_state(self):
        bpe = CodecBPE.train([[(1, 4), (2, 7), (1, 4), (2, 7)]], num_merges=1)

        with tempfile.TemporaryDirectory() as tmp:
            bpe.save_pretrained(tmp)
            loaded = CodecBPE.from_pretrained(tmp)

        self.assertEqual(loaded.to_dict(), bpe.to_dict())
        self.assertEqual(loaded.encode_units([(1, 4), (2, 7)]), [2])
        self.assertEqual(loaded.expand_ids([2]), [(1, 4), (2, 7)])

    def test_tuple_units_must_have_fixed_length(self):
        with self.assertRaisesRegex(ValueError, "fixed-length"):
            CodecBPE.train([[(1, 4), (2, 7, 9)]])

        bpe = CodecBPE.train([[(1, 4), (2, 7)]])
        with self.assertRaisesRegex(ValueError, "unit shape"):
            bpe.encode_units([(1, 4, 9)])

    def test_train_progress_keeps_same_result(self):
        corpus = [[1, 2, 1, 2, 3], [1, 2, 3]]

        plain = CodecBPE.train(corpus, num_merges=2)
        progress_output = io.StringIO()
        with redirect_stderr(progress_output):
            with_progress = CodecBPE.train(corpus, num_merges=2, progress=True)

        self.assertEqual(with_progress.to_dict(), plain.to_dict())
        self.assertIn("CodecBPE corpus", progress_output.getvalue())
        self.assertIn("CodecBPE merges", progress_output.getvalue())

    def test_expand_with_counts_returns_unit_ids_and_lengths(self):
        bpe = CodecBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)

        unit_ids, counts = bpe.expand_with_counts([3, 4])

        self.assertEqual(unit_ids, [1, 2, 1, 2, 3])
        self.assertEqual(counts, [2, 3])

    def test_repeat_interleave_uses_expansion_counts(self):
        bpe = CodecBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)
        x = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        token_ids = torch.tensor([3, 4])

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

    def test_repeat_interleave_expands_tuple_units(self):
        bpe = CodecBPE.train(
            [
                [(1, 4), (2, 7), (1, 4), (2, 7), (3, 8)],
                [(1, 4), (2, 7), (3, 8)],
            ],
            num_merges=2,
        )
        x = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        token_ids = torch.tensor([3, 4])

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
        self.assertTrue(
            torch.equal(
                unit_ids,
                torch.tensor(
                    [
                        [1, 4],
                        [2, 7],
                        [1, 4],
                        [2, 7],
                        [3, 8],
                    ]
                ),
            )
        )

    def test_repeat_interleave_pads_batched_expansion(self):
        bpe = CodecBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)
        x = torch.tensor(
            [
                [[10.0, 11.0], [20.0, 21.0], [0.0, 0.0]],
                [[30.0, 31.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        token_ids = torch.tensor([[3, 4, 0], [4, 0, 0]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        expanded_x, unit_ids, expanded_mask = bpe.repeat_interleave(
            x,
            token_ids,
            mask,
            dim=1,
        )

        expected_x = torch.tensor(
            [
                [
                    [10.0, 11.0],
                    [10.0, 11.0],
                    [20.0, 21.0],
                    [20.0, 21.0],
                    [20.0, 21.0],
                ],
                [
                    [30.0, 31.0],
                    [30.0, 31.0],
                    [30.0, 31.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ],
            ]
        )
        self.assertTrue(torch.equal(expanded_x, expected_x))
        self.assertTrue(torch.equal(unit_ids, torch.tensor([[1, 2, 1, 2, 3], [1, 2, 3, 0, 0]])))
        self.assertTrue(
            torch.equal(
                expanded_mask,
                torch.tensor(
                    [
                        [True, True, True, True, True],
                        [True, True, True, False, False],
                    ]
                ),
            )
        )

    def test_repeat_interleave_pads_batched_tuple_expansion(self):
        bpe = CodecBPE.train(
            [
                [(1, 4), (2, 7), (1, 4), (2, 7), (3, 8)],
                [(1, 4), (2, 7), (3, 8)],
            ],
            num_merges=2,
        )
        x = torch.tensor(
            [
                [[10.0, 11.0], [20.0, 21.0], [0.0, 0.0]],
                [[30.0, 31.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        token_ids = torch.tensor([[3, 4, 0], [4, 0, 0]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        expanded_x, unit_ids, expanded_mask = bpe.repeat_interleave(
            x,
            token_ids,
            mask,
            dim=1,
        )

        self.assertEqual(tuple(expanded_x.shape), (2, 5, 2))
        self.assertTrue(
            torch.equal(
                unit_ids,
                torch.tensor(
                    [
                        [[1, 4], [2, 7], [1, 4], [2, 7], [3, 8]],
                        [[1, 4], [2, 7], [3, 8], [0, 0], [0, 0]],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                expanded_mask,
                torch.tensor(
                    [
                        [True, True, True, True, True],
                        [True, True, True, False, False],
                    ]
                ),
            )
        )

    def test_repeat_interleave_keeps_unpadded_batched_expansion(self):
        bpe = CodecBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)
        x = torch.tensor(
            [
                [[10.0, 11.0], [20.0, 21.0]],
                [[30.0, 31.0], [40.0, 41.0]],
            ]
        )
        token_ids = torch.tensor([[3, 4], [3, 4]])

        expanded_x, unit_ids, expanded_mask = bpe.repeat_interleave(x, token_ids, dim=1)

        self.assertEqual(tuple(expanded_x.shape), (2, 5, 2))
        self.assertTrue(torch.equal(unit_ids, torch.tensor([[1, 2, 1, 2, 3], [1, 2, 1, 2, 3]])))
        self.assertTrue(torch.equal(expanded_mask, torch.ones((2, 5), dtype=torch.bool)))

    def test_strict_rejects_empty_inputs_and_unknown_ids(self):
        bpe = CodecBPE.train([[1, 2, 3]], num_merges=1)

        with self.assertRaisesRegex(ValueError, "empty"):
            CodecBPE.train([])
        with self.assertRaisesRegex(ValueError, "empty"):
            CodecBPE.train([[]])
        with self.assertRaisesRegex(ValueError, "empty"):
            bpe.encode_units([])
        with self.assertRaisesRegex(KeyError, "unknown unit"):
            bpe.encode_units([9])
        with self.assertRaisesRegex(KeyError, "unknown token_id"):
            bpe.expand_ids([9])

    def test_train_rejects_negative_unit_ids(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            CodecBPE.train([[-1, 0, -1, 0]], num_merges=1)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            CodecBPE.from_dict(
                {
                    "tokens": {"-1": [-1]},
                    "merges": [],
                    "strict": True,
                }
            )

    def test_non_strict_keeps_unknown_ids_atomic(self):
        bpe = CodecBPE.train([[1, 2, 3]], num_merges=1, strict=False)

        self.assertEqual(bpe.encode_units([]), [])
        self.assertEqual(bpe.encode_units([9]), [9])
        self.assertEqual(bpe.expand_ids([9]), [9])

    def test_non_strict_rejects_unknown_unit_colliding_with_token_id(self):
        bpe = CodecBPE.train([[1, 2, 1, 2]], num_merges=1, strict=False)

        with self.assertRaisesRegex(ValueError, "collides"):
            bpe.encode_units([0])

        with self.assertRaisesRegex(ValueError, "non-negative"):
            bpe.encode_units([-1])

    def test_repeat_interleave_rejects_misaligned_inputs(self):
        bpe = CodecBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)

        with self.assertRaisesRegex(ValueError, "mask"):
            bpe.repeat_interleave(
                torch.randn(2, 3),
                torch.tensor([1, 2]),
                torch.tensor([True, True]),
                dim=0,
            )
        with self.assertRaisesRegex(ValueError, "non-batch"):
            bpe.repeat_interleave(torch.randn(2, 3), torch.tensor([[1, 2]]), dim=0)
        with self.assertRaisesRegex(ValueError, "align"):
            bpe.repeat_interleave(torch.randn(2, 3), torch.tensor([1]), dim=0)
        with self.assertRaisesRegex(ValueError, "out of range"):
            bpe.repeat_interleave(torch.randn(2, 3), torch.tensor([1, 2]), dim=2)
        with self.assertRaisesRegex(ValueError, "same shape"):
            bpe.repeat_interleave(
                torch.randn(2, 2, 3),
                torch.tensor([[1, 2], [1, 2]]),
                torch.tensor([[True, True]]),
                dim=1,
            )
        with self.assertRaisesRegex(ValueError, "padding values"):
            bpe.repeat_interleave(
                torch.randn(2, 2, 3),
                torch.tensor([[1, 0], [2, -1]]),
                torch.tensor([[True, False], [True, False]]),
                dim=1,
            )
        with self.assertRaisesRegex(ValueError, "cannot infer"):
            bpe.repeat_interleave(
                torch.randn(2, 2, 3),
                torch.tensor([[3, 1], [3, 3]]),
                dim=1,
            )

    def test_codec_bpe_holds_tokenizers_model(self):
        bpe = CodecBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)

        self.assertEqual(bpe.tokens[3], (1, 2))
        self.assertEqual(bpe.model.id_to_token(3), bpe.token_text(3))
        self.assertEqual(bpe.model.id_to_token(4), bpe.token_text(4))

    def test_codec_bpe_encode_matches_core(self):
        bpe = CodecBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)
        units = [1, 2, 1, 2, 3]

        self.assertEqual(bpe.encode_units(units), bpe.core.encode_units(units))
        self.assertEqual(bpe.expand_ids(bpe.encode_units(units)), units)

        tokenizer = bpe.tokenizer()
        encoded = tokenizer.encode(bpe.units_text(units))

        self.assertEqual(encoded.ids, bpe.core.encode_units(units))

    def test_eval_reports_expected_compression(self):
        bpe = CodecBPE.train([[1, 2, 1, 2, 3], [1, 2, 3]], num_merges=2)

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
        bpe = CodecBPE.train([[1, 2, 3]], num_merges=1)

        with self.assertRaisesRegex(ValueError, "corpus"):
            bpe.eval([])


if __name__ == "__main__":
    unittest.main()
