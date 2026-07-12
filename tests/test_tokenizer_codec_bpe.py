import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    }


def load_state(value):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        (path / "codec_bpe.json").write_text(json.dumps(value), encoding="utf-8")
        return CodecBPE.from_pretrained(path)


def replay(corpus):
    return lambda: corpus


@unittest.skipIf(tokenizers is None, "tokenizers is not installed")
class BPETest(unittest.TestCase):
    def test_train_reports_missing_tokenizer_extra(self):
        with (
            patch.dict(sys.modules, {"tokenizers": None, "tokenizers.models": None}),
            self.assertRaisesRegex(ImportError, r"anytrain\[tokenizer\]"),
        ):
            CodecBPE.train(
                replay([[[1], [2]]]),
                codebook_sizes=(4,),
                show_progress=False,
            )

    def test_from_pretrained_accepts_frame_tokens_and_merges(self):
        bpe = load_state(state())

        self.assertEqual(bpe.encode([[1], [2], [3]]), [4])
        self.assertEqual(bpe.decode([4]), [(1,), (2,), (3,)])

    def test_from_pretrained_round_trip(self):
        bpe = load_state(state())

        with tempfile.TemporaryDirectory() as tmp:
            out = bpe.save_pretrained(tmp)
            loaded = CodecBPE.from_pretrained(tmp)

            self.assertTrue((out / "codec_bpe.json").exists())

        self.assertEqual(loaded.codebook_sizes, bpe.codebook_sizes)
        self.assertEqual(loaded.vocab_size, bpe.vocab_size)
        self.assertEqual(loaded.encode([[1], [2], [3]]), [4])
        self.assertEqual(loaded.decode([4]), bpe.decode([4]))

    def test_train_merges_single_codebook_frames_and_round_trips(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=5,
        )

        token_ids = bpe.encode([[1], [2], [1], [2], [3]])

        self.assertEqual(bpe.vocab_size, 5)
        self.assertEqual(token_ids, [3, 4])
        self.assertEqual(bpe.decode(token_ids), [(1,), (2,), (1,), (2,), (3,)])

    def test_train_supports_longcat_scale_frame_vocab(self):
        frames = [[value] for value in range(8192)]

        bpe = CodecBPE.train(replay([frames]), codebook_sizes=(8192,), vocab_size=8192)

        encoded = bpe.encode([[0], [4096], [8191]])
        self.assertEqual(encoded, [0, 4096, 8191])
        self.assertEqual(bpe.decode(encoded), [(0,), (4096,), (8191,)])

    def test_train_keeps_alphabet_when_vocab_size_is_smaller(self):
        bpe = CodecBPE.train(
            replay([[[100], [101], [102]]]),
            codebook_sizes=(128,),
            vocab_size=2,
            show_progress=False,
        )

        self.assertEqual(bpe.vocab_size, 3)
        self.assertEqual(bpe.encode([[100], [101], [102]]), [0, 1, 2])

    def test_train_respects_min_frequency(self):
        bpe = CodecBPE.train(
            replay([[[1], [2]]]),
            codebook_sizes=(16,),
            vocab_size=3,
            min_frequency=2,
            show_progress=False,
        )

        self.assertEqual(bpe.vocab_size, 2)
        self.assertEqual(bpe.encode([[1], [2]]), [0, 1])

    def test_train_respects_max_token_length(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=5,
            max_token_length=2,
            show_progress=False,
        )

        self.assertEqual(bpe.vocab_size, 4)
        self.assertEqual(bpe.encode([[1], [2], [3]]), [3, 2])

    def test_train_merges_multi_codebook_frames_and_round_trips(self):
        corpus = [
            [[1, 4], [2, 7], [1, 4], [2, 7], [3, 8]],
            [[1, 4], [2, 7], [3, 8]],
        ]

        bpe = CodecBPE.train(replay(corpus), codebook_sizes=(4, 16), vocab_size=5)

        self.assertEqual(bpe.vocab_size, 5)
        self.assertEqual(bpe.encode([[1, 4], [2, 7], [3, 8]]), [4])
        self.assertEqual(bpe.decode([4]), [(1, 4), (2, 7), (3, 8)])

    def test_train_progress_keeps_same_result(self):
        corpus = [[[1], [2], [1], [2], [3]], [[1], [2], [3]]]

        plain = CodecBPE.train(
            replay(corpus),
            codebook_sizes=(16,),
            vocab_size=5,
            show_progress=False,
        )
        with_progress = CodecBPE.train(
            replay(corpus),
            codebook_sizes=(16,),
            vocab_size=5,
            show_progress=True,
        )

        self.assertEqual(with_progress.vocab_size, plain.vocab_size)
        self.assertEqual(
            with_progress.encode([[1], [2], [1], [2], [3]]),
            plain.encode([[1], [2], [1], [2], [3]]),
        )

    def test_train_calls_corpus_twice(self):
        calls = 0

        def corpus():
            nonlocal calls
            calls += 1
            return iter([[[1], [2], [1], [2], [3]], [[1], [2], [3]]])

        bpe = CodecBPE.train(corpus, codebook_sizes=(16,), vocab_size=5, show_progress=False)

        self.assertEqual(calls, 2)
        self.assertEqual(bpe.encode([[1], [2], [3]]), [4])

    def test_rejects_invalid_frames(self):
        with self.assertRaisesRegex(ValueError, "number of codebooks"):
            CodecBPE.train(replay([[[1, 4], [2, 7, 9]]]), codebook_sizes=(4, 16))
        with self.assertRaisesRegex(ValueError, "codebook 1"):
            CodecBPE.train(replay([[[1, 4], [2, 17]]]), codebook_sizes=(4, 16))
        with self.assertRaisesRegex(TypeError, "use \\[id\\]"):
            CodecBPE.train(replay([[1, 2, 3]]), codebook_sizes=(16,))
        with self.assertRaisesRegex(TypeError, "integers"):
            CodecBPE.train(replay([[[1.5]]]), codebook_sizes=(16,))

    def test_rejects_empty_corpus_and_sequences(self):
        with self.assertRaisesRegex(ValueError, "corpus"):
            CodecBPE.train(replay([]), codebook_sizes=(16,))
        with self.assertRaisesRegex(ValueError, "empty sequences"):
            CodecBPE.train(replay([[]]), codebook_sizes=(16,))

    def test_encode_rejects_empty_and_unknown_frames(self):
        bpe = CodecBPE.train(replay([[[1], [2], [3]]]), codebook_sizes=(16,), vocab_size=4)

        with self.assertRaisesRegex(ValueError, "frames"):
            bpe.encode([])
        with self.assertRaisesRegex(KeyError, "unknown frame"):
            bpe.encode([[4]])

    def test_decode_rejects_empty_and_unknown_token_ids(self):
        bpe = CodecBPE.train(replay([[[1], [2], [3]]]), codebook_sizes=(16,), vocab_size=4)

        with self.assertRaisesRegex(ValueError, "token_ids"):
            bpe.decode([])
        with self.assertRaises(KeyError):
            bpe.decode([bpe.vocab_size])

    def test_eval_reports_expected_compression(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=5,
        )

        stats = bpe.evaluate(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            show_progress=False,
        )

        self.assertEqual(stats["num_sequences"], 2)
        self.assertEqual(stats["original_frames"], 8)
        self.assertEqual(stats["encoded_tokens"], 3)
        self.assertEqual(stats["mean_original_length"], 4.0)
        self.assertEqual(stats["mean_encoded_length"], 1.5)
        self.assertAlmostEqual(stats["compression_ratio"], 3 / 8)
        self.assertAlmostEqual(stats["compression_factor"], 8 / 3)
        self.assertAlmostEqual(stats["compression_gain"], 5 / 8)

    def test_eval_reports_token_distributions(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=5,
        )

        stats = bpe.evaluate(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            show_progress=False,
        )

        self.assertEqual(stats["encoded_tokens"], 3)
        self.assertEqual(stats["token_count_histogram"], {1: 1, 2: 1})
        self.assertEqual([top["count"] for top in stats["top_token_counts"]], [2, 1])
        self.assertEqual(stats["num_used_tokens"], 2)
        self.assertAlmostEqual(
            sum(top["frequency"] for top in stats["top_token_counts"]),
            1.0,
        )
        self.assertAlmostEqual(
            stats["vocab_coverage"], stats["num_used_tokens"] / bpe.vocab_size
        )
        self.assertGreater(stats["entropy"], 0.0)
        self.assertEqual(stats["used_token_length_counts"], (0, 0, 1, 2))
        self.assertEqual(stats["used_token_length_frequencies"], (0.0, 0.0, 1 / 3, 2 / 3))
        self.assertEqual(sum(stats["vocab_token_length_counts"]), bpe.vocab_size)
        self.assertEqual(stats["mean_used_token_length"], 8 / 3)
        self.assertEqual(stats["max_used_token_length"], 3)
        self.assertEqual(stats["used_token_length_quantiles"]["p50"], 3.0)

    def test_eval_limits_top_token_counts(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=5,
        )

        stats = bpe.evaluate(
            [[[1], [2], [1], [2], [3]], [[1], [2], [3]]],
            show_progress=False,
            top_k=1,
        )

        self.assertEqual(len(stats["top_token_counts"]), 1)
        self.assertEqual(stats["top_token_counts"][0]["count"], 2)

    def test_eval_rejects_invalid_top_k(self):
        bpe = CodecBPE.train(replay([[[1], [2], [3]]]), codebook_sizes=(16,), vocab_size=4)

        with self.assertRaisesRegex(ValueError, "top_k"):
            bpe.evaluate([[[1], [2], [3]]], show_progress=False, top_k=-1)

    def test_eval_progress_keeps_same_result(self):
        bpe = CodecBPE.train(
            replay([[[1], [2], [1], [2], [3]], [[1], [2], [3]]]),
            codebook_sizes=(16,),
            vocab_size=5,
            show_progress=False,
        )
        corpus = [[[1], [2], [1], [2], [3]], [[1], [2], [3]]]

        plain = bpe.evaluate(corpus, show_progress=False)
        with_progress = bpe.evaluate(corpus, show_progress=True)

        self.assertEqual(with_progress, plain)

    def test_eval_rejects_empty_corpus(self):
        bpe = CodecBPE.train(replay([[[1], [2], [3]]]), codebook_sizes=(16,), vocab_size=4)

        with self.assertRaisesRegex(ValueError, "corpus"):
            bpe.evaluate([], show_progress=False)


if __name__ == "__main__":
    unittest.main()
