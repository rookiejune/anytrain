import tempfile
import unittest

import torch

from anytrain.tokenizer.codec_bpe import CodecBPE

try:
    import tokenizers
except ImportError:
    tokenizers = None


def state():
    return {
        "codebook_sizes": [16],
        "tokens": {
            "0": [[1]],
            "1": [[2]],
            "2": [[3]],
            "3": [[1], [2]],
            "4": [[1], [2], [3]],
        },
        "merges": [
            {"left": 0, "right": 1, "token_id": 3},
            {"left": 3, "right": 2, "token_id": 4},
        ],
        "strict": True,
    }


@unittest.skipIf(tokenizers is None, "tokenizers is not installed")
class BPETest(unittest.TestCase):
    def test_from_dict_accepts_frame_tokens_and_merges(self):
        bpe = CodecBPE.from_dict(state())

        self.assertEqual(bpe.encode_frames([[1], [2], [3]]), [4])
        self.assertEqual(bpe.expand_ids([4]), [(1,), (2,), (3,)])

    def test_from_pretrained_round_trip(self):
        bpe = CodecBPE.from_dict(state())

        with tempfile.TemporaryDirectory() as tmp:
            bpe.save_pretrained(tmp)
            loaded = CodecBPE.from_pretrained(tmp)

        self.assertEqual(loaded.to_dict(), bpe.to_dict())
        self.assertEqual(loaded.encode_frames([[1], [2], [3]]), [4])
        self.assertEqual(loaded.vocab_size, 5)
        self.assertEqual(loaded.codebook_sizes, (16,))

    def test_from_pretrained_accepts_state_file(self):
        bpe = CodecBPE.train(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            codebook_sizes=(16,),
            vocab_size=5,
        )

        with tempfile.TemporaryDirectory() as tmp:
            out = bpe.save_pretrained(tmp)
            loaded = CodecBPE.from_pretrained(out / "codec_bpe.json")

        self.assertEqual(loaded.to_dict(), bpe.to_dict())

    def test_uninitialized_instance_fails_clearly(self):
        bpe = CodecBPE()

        with self.assertRaisesRegex(ValueError, "not initialized"):
            bpe.encode_frames([[1], [2], [3]])

    def test_train_merges_single_codebook_frames_and_round_trips(self):
        bpe = CodecBPE.train(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            codebook_sizes=(16,),
            vocab_size=5,
        )

        self.assertEqual(bpe.vocab_size, 5)
        self.assertEqual(bpe.tokens[3], ((1,), (2,)))
        self.assertEqual(bpe.tokens[4], ((1,), (2,), (3,)))
        self.assertEqual(bpe.encode_frames([[1], [2], [3]]), [4])

        token_ids = bpe.encode_frames([[1], [2], [1], [2], [3]])

        self.assertEqual(token_ids, [3, 4])
        self.assertEqual(bpe.expand_ids(token_ids), [(1,), (2,), (1,), (2,), (3,)])

    def test_train_supports_longcat_scale_frame_vocab(self):
        frames = [[value] for value in range(8192)]

        bpe = CodecBPE.train([frames], codebook_sizes=(8192,), vocab_size=8192)

        encoded = bpe.encode_frames([[0], [4096], [8191]])
        self.assertEqual(encoded, [0, 4096, 8191])
        self.assertEqual(bpe.expand_ids(encoded), [(0,), (4096,), (8191,)])
        self.assertEqual(len(bpe.token_text(8191)), 1)

    def test_train_keeps_alphabet_when_vocab_size_is_smaller(self):
        bpe = CodecBPE.train(
            [[[100], [101], [102]]],
            codebook_sizes=(128,),
            vocab_size=2,
            show_progress=False,
        )

        self.assertEqual(bpe.vocab_size, 3)
        self.assertEqual(bpe.encode_frames([[100], [101], [102]]), [0, 1, 2])

        bpe = CodecBPE.train(
            [[[100], [101], [100], [101]]],
            codebook_sizes=(128,),
            vocab_size=3,
        )

        self.assertEqual(bpe.vocab_size, 3)
        self.assertEqual(bpe.tokens[0], ((100,),))
        self.assertEqual(bpe.tokens[1], ((101,),))
        self.assertEqual(bpe.tokens[2], ((100,), (101,)))
        self.assertEqual(bpe.encode_frames([[100], [101]]), [2])

    def test_train_compacts_sparse_frame_ids(self):
        bpe = CodecBPE.train(
            [[[100], [101], [100], [101]]],
            codebook_sizes=(128,),
            vocab_size=3,
        )

        self.assertEqual(bpe.vocab_size, 3)
        self.assertEqual(bpe.encode_frames([[100], [101], [100], [101]]), [2, 2])

    def test_train_respects_min_frequency(self):
        bpe = CodecBPE.train(
            [[[1], [2]]],
            codebook_sizes=(16,),
            vocab_size=3,
            min_frequency=2,
            show_progress=False,
        )

        self.assertEqual(bpe.vocab_size, 2)
        self.assertEqual(bpe.encode_frames([[1], [2]]), [0, 1])

    def test_train_respects_max_token_length(self):
        bpe = CodecBPE.train(
            [[[1], [2], [3]], [[1], [2], [3]]],
            codebook_sizes=(16,),
            vocab_size=5,
            max_token_length=2,
            show_progress=False,
        )

        self.assertEqual(bpe.vocab_size, 4)
        self.assertEqual(bpe.tokens[3], ((1,), (2,)))
        self.assertEqual(bpe.encode_frames([[1], [2], [3]]), [3, 2])

    def test_train_merges_multi_codebook_frames_and_round_trips(self):
        corpus = [
            [[1, 4], [2, 7], [1, 4], [2, 7], [3, 8]],
            [[1, 4], [2, 7], [3, 8]],
        ]

        bpe = CodecBPE.train(corpus, codebook_sizes=(4, 16), vocab_size=5)

        self.assertEqual(bpe.vocab_size, 5)
        self.assertEqual(bpe.tokens[0], ((1, 4),))
        self.assertEqual(bpe.tokens[3], ((1, 4), (2, 7)))
        self.assertEqual(bpe.tokens[4], ((1, 4), (2, 7), (3, 8)))
        self.assertEqual(bpe.encode_frames([[1, 4], [2, 7], [3, 8]]), [4])

        token_ids = bpe.encode_frames(corpus[0])

        self.assertEqual(token_ids, [3, 4])
        self.assertEqual(bpe.expand_ids(token_ids), [(1, 4), (2, 7), (1, 4), (2, 7), (3, 8)])

    def test_multi_codebook_frames_round_trip_through_pretrained_state(self):
        bpe = CodecBPE.train(
            [[[1, 4], [2, 7], [1, 4], [2, 7]]],
            codebook_sizes=(4, 16),
            vocab_size=3,
        )

        with tempfile.TemporaryDirectory() as tmp:
            bpe.save_pretrained(tmp)
            loaded = CodecBPE.from_pretrained(tmp)

        self.assertEqual(loaded.to_dict(), bpe.to_dict())
        self.assertEqual(loaded.encode_frames([[1, 4], [2, 7]]), [2])
        self.assertEqual(loaded.expand_ids([2]), [(1, 4), (2, 7)])

    def test_frames_must_match_codebooks(self):
        with self.assertRaisesRegex(ValueError, "number of codebooks"):
            CodecBPE.train([[[1, 4], [2, 7, 9]]], codebook_sizes=(4, 16))

        with self.assertRaisesRegex(ValueError, "codebook 1"):
            CodecBPE.train([[[1, 4], [2, 17]]], codebook_sizes=(4, 16))

        bpe = CodecBPE.train([[[1, 4], [2, 7]]], codebook_sizes=(4, 16))
        with self.assertRaisesRegex(ValueError, "number of codebooks"):
            bpe.encode_frames([[1, 4, 9]])

    def test_train_rejects_1d_inputs(self):
        with self.assertRaisesRegex(TypeError, "frames"):
            CodecBPE.train([[1, 2, 3]], codebook_sizes=(16,))

        bpe = CodecBPE.train([[[1], [2], [3]]], codebook_sizes=(16,))
        with self.assertRaisesRegex(TypeError, "frames"):
            bpe.encode_frames([1, 2, 3])

    def test_train_progress_keeps_same_result(self):
        corpus = [[[1], [2], [1], [2], [3]], [[1], [2], [3]]]

        plain = CodecBPE.train(corpus, codebook_sizes=(16,), vocab_size=5, show_progress=False)
        with_progress = CodecBPE.train(
            corpus,
            codebook_sizes=(16,),
            vocab_size=5,
            show_progress=True,
        )

        self.assertEqual(with_progress.to_dict(), plain.to_dict())

    def test_expand_with_counts_returns_frames_and_lengths(self):
        bpe = CodecBPE.train(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            codebook_sizes=(16,),
            vocab_size=5,
        )

        frames, counts = bpe.expand_with_counts([3, 4])

        self.assertEqual(frames, [(1,), (2,), (1,), (2,), (3,)])
        self.assertEqual(counts, [2, 3])

    def test_repeat_interleave_uses_expansion_counts(self):
        bpe = CodecBPE.train(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            codebook_sizes=(16,),
            vocab_size=5,
        )
        x = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        token_ids = torch.tensor([3, 4])

        expanded_x, frames = bpe.repeat_interleave(x, token_ids, dim=0)

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
        self.assertTrue(torch.equal(frames, torch.tensor([[1], [2], [1], [2], [3]])))

    def test_repeat_interleave_expands_multi_codebook_frames(self):
        bpe = CodecBPE.train(
            [
                [[1, 4], [2, 7], [1, 4], [2, 7], [3, 8]],
                [[1, 4], [2, 7], [3, 8]],
            ],
            codebook_sizes=(4, 16),
            vocab_size=5,
        )
        x = torch.tensor([[10.0, 11.0], [20.0, 21.0]])
        token_ids = torch.tensor([3, 4])

        expanded_x, frames = bpe.repeat_interleave(x, token_ids, dim=0)

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
                frames,
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
        bpe = CodecBPE.train(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            codebook_sizes=(16,),
            vocab_size=5,
        )
        x = torch.tensor(
            [
                [[10.0, 11.0], [20.0, 21.0], [0.0, 0.0]],
                [[30.0, 31.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        token_ids = torch.tensor([[3, 4, 0], [4, 0, 0]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        expanded_x, frames, expanded_mask = bpe.repeat_interleave(
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
        self.assertTrue(
            torch.equal(
                frames,
                torch.tensor([[[1], [2], [1], [2], [3]], [[1], [2], [3], [1], [1]]]),
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

    def test_repeat_interleave_pads_batched_multi_codebook_expansion(self):
        bpe = CodecBPE.train(
            [
                [[1, 4], [2, 7], [1, 4], [2, 7], [3, 8]],
                [[1, 4], [2, 7], [3, 8]],
            ],
            codebook_sizes=(4, 16),
            vocab_size=5,
        )
        x = torch.tensor(
            [
                [[10.0, 11.0], [20.0, 21.0], [0.0, 0.0]],
                [[30.0, 31.0], [0.0, 0.0], [0.0, 0.0]],
            ]
        )
        token_ids = torch.tensor([[3, 4, 0], [4, 0, 0]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        expanded_x, frames, expanded_mask = bpe.repeat_interleave(
            x,
            token_ids,
            mask,
            dim=1,
        )

        self.assertEqual(tuple(expanded_x.shape), (2, 5, 2))
        self.assertTrue(
            torch.equal(
                frames,
                torch.tensor(
                    [
                        [[1, 4], [2, 7], [1, 4], [2, 7], [3, 8]],
                        [[1, 4], [2, 7], [3, 8], [1, 4], [1, 4]],
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

    def test_repeat_interleave_decodes_multi_codebook_padding_frame(self):
        bpe = CodecBPE.train(
            [
                [[1, 4], [2, 7], [1, 4], [2, 7]],
                [[2, 7], [3, 8]],
            ],
            codebook_sizes=(4, 16),
            vocab_size=4,
        )
        x = torch.tensor(
            [
                [[10.0], [20.0], [0.0]],
                [[30.0], [0.0], [0.0]],
            ]
        )
        token_ids = torch.tensor([[3, 1, 1], [1, 1, 1]])
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

        _, frames, expanded_mask = bpe.repeat_interleave(x, token_ids, mask, dim=1)

        self.assertTrue(
            torch.equal(
                frames,
                torch.tensor(
                    [
                        [[1, 4], [2, 7], [2, 7]],
                        [[2, 7], [2, 7], [2, 7]],
                    ]
                ),
            )
        )
        self.assertTrue(
            torch.equal(
                expanded_mask,
                torch.tensor(
                    [
                        [True, True, True],
                        [True, False, False],
                    ]
                ),
            )
        )

    def test_repeat_interleave_keeps_unpadded_batched_expansion(self):
        bpe = CodecBPE.train(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            codebook_sizes=(16,),
            vocab_size=5,
        )
        x = torch.tensor(
            [
                [[10.0, 11.0], [20.0, 21.0]],
                [[30.0, 31.0], [40.0, 41.0]],
            ]
        )
        token_ids = torch.tensor([[3, 4], [3, 4]])

        expanded_x, frames, expanded_mask = bpe.repeat_interleave(x, token_ids, dim=1)

        self.assertEqual(tuple(expanded_x.shape), (2, 5, 2))
        self.assertTrue(
            torch.equal(
                frames,
                torch.tensor([[[1], [2], [1], [2], [3]], [[1], [2], [1], [2], [3]]]),
            )
        )
        self.assertTrue(torch.equal(expanded_mask, torch.ones((2, 5), dtype=torch.bool)))

    def test_strict_rejects_empty_inputs_and_unknown_ids(self):
        bpe = CodecBPE.train([[[1], [2], [3]]], codebook_sizes=(16,), vocab_size=4)

        with self.assertRaisesRegex(ValueError, "empty"):
            CodecBPE.train([], codebook_sizes=(16,))
        with self.assertRaisesRegex(ValueError, "empty"):
            CodecBPE.train([[]], codebook_sizes=(16,))
        with self.assertRaisesRegex(ValueError, "empty"):
            bpe.encode_frames([])
        with self.assertRaisesRegex(KeyError, "unknown frame"):
            bpe.encode_frames([[9]])
        with self.assertRaisesRegex(KeyError, "unknown token_id"):
            bpe.expand_ids([9])

    def test_train_rejects_negative_frame_ids(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            CodecBPE.train([[[-1], [0], [-1], [0]]], codebook_sizes=(16,), vocab_size=3)

        with self.assertRaisesRegex(ValueError, "non-negative"):
            CodecBPE.from_dict(
                {
                    "codebook_sizes": [16],
                    "tokens": {"-1": [[-1]]},
                    "merges": [],
                    "strict": True,
                }
            )

    def test_non_strict_allows_empty_input_but_not_unknown_frames(self):
        bpe = CodecBPE.train([[[1], [2], [3]]], codebook_sizes=(16,), vocab_size=4, strict=False)

        self.assertEqual(bpe.encode_frames([]), [])
        with self.assertRaisesRegex(KeyError, "unknown frame"):
            bpe.encode_frames([[9]])
        with self.assertRaisesRegex(KeyError, "unknown token_id"):
            bpe.expand_ids([9])

    def test_repeat_interleave_rejects_misaligned_inputs(self):
        bpe = CodecBPE.train(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            codebook_sizes=(16,),
            vocab_size=5,
        )

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
        bpe = CodecBPE.train(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            codebook_sizes=(16,),
            vocab_size=5,
        )

        self.assertEqual(bpe.tokens[3], ((1,), (2,)))
        self.assertEqual(bpe.model.id_to_token(3), bpe.token_text(3))
        self.assertEqual(bpe.model.id_to_token(4), bpe.token_text(4))

    def test_codec_bpe_encode_matches_core(self):
        bpe = CodecBPE.train(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            codebook_sizes=(16,),
            vocab_size=5,
        )
        frames = [[1], [2], [1], [2], [3]]
        base_ids = [1, 2, 1, 2, 3]

        self.assertEqual(bpe.encode_frames(frames), bpe.core.encode_ids(base_ids))
        self.assertEqual(bpe.expand_ids(bpe.encode_frames(frames)), [(1,), (2,), (1,), (2,), (3,)])

        tokenizer = bpe.tokenizer()
        encoded = tokenizer.encode(bpe.frames_text(frames))

        self.assertEqual(encoded.ids, bpe.core.encode_ids(base_ids))

    def test_eval_reports_expected_compression(self):
        bpe = CodecBPE.train(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            codebook_sizes=(16,),
            vocab_size=5,
        )

        stats = bpe.eval([[[1], [2], [1], [2], [3]], [[1], [2], [3]]])

        self.assertEqual(stats.num_sequences, 2)
        self.assertEqual(stats.original_tokens, 8)
        self.assertEqual(stats.encoded_tokens, 3)
        self.assertEqual(stats.mean_original_length, 4.0)
        self.assertEqual(stats.mean_encoded_length, 1.5)
        self.assertAlmostEqual(stats.compression_ratio, 3 / 8)
        self.assertAlmostEqual(stats.compression_factor, 8 / 3)
        self.assertAlmostEqual(stats.compression_gain, 5 / 8)

    def test_eval_rejects_empty_corpus(self):
        bpe = CodecBPE.train([[[1], [2], [3]]], codebook_sizes=(16,), vocab_size=4)

        with self.assertRaisesRegex(ValueError, "corpus"):
            bpe.eval([])


if __name__ == "__main__":
    unittest.main()
