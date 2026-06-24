import unittest
from unittest.mock import patch

from anytrain.evaluator import EvaluatorABC
from anytrain.evaluator.text import TextComparisonEvaluator
from anytrain.evaluator.text.scores import corpus_chrf_score, word_error_rate


class TextComparisonEvaluatorTest(unittest.TestCase):
    def test_text_evaluator_inherits_evaluator_abc(self):
        self.assertIsInstance(TextComparisonEvaluator(), EvaluatorABC)

    def test_single_string_returns_bleu_wer_chrf_metrics(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("the quick brown fox", "the quick brown fox")

        self.assertEqual(set(metrics), {"bleu", "wer", "chrf"})
        self.assertEqual(metrics["bleu"], 100.0)
        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["chrf"], 100.0)

    def test_batch_scores_are_corpus_scores(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator(
            ["the quick brown fox", "hello world"],
            ["the quick brown fox", "hello there"],
        )

        self.assertGreater(metrics["bleu"], 50.0)
        self.assertLess(metrics["bleu"], 100.0)
        self.assertAlmostEqual(metrics["wer"], 1.0 / 6.0)
        self.assertGreater(metrics["chrf"], 50.0)
        self.assertLess(metrics["chrf"], 100.0)

    def test_normalization_defaults_strip_and_collapse_whitespace(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("  hello \n\t world  ", "hello world")

        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["chrf"], 100.0)

    def test_normalization_removes_punctuation_by_default(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("Please call Stella", "Please call Stella.")

        self.assertEqual(metrics["bleu"], 100.0)
        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["chrf"], 100.0)

    def test_punctuation_removal_can_be_disabled(self):
        evaluator = TextComparisonEvaluator(remove_punctuation=False)

        metrics = evaluator("Please call Stella", "Please call Stella.")

        self.assertGreater(metrics["wer"], 0.0)
        self.assertLess(metrics["chrf"], 100.0)

    def test_smoothed_bleu_reports_short_partial_match(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("Please cool Stella", "Please call Stella.")

        self.assertGreater(metrics["bleu"], 0.0)
        self.assertLess(metrics["bleu"], 100.0)
        self.assertAlmostEqual(metrics["wer"], 1.0 / 3.0)

    def test_bleu_smoothing_can_be_disabled(self):
        evaluator = TextComparisonEvaluator(bleu_smoothing=False)

        metrics = evaluator("Please cool Stella", "Please call Stella.")

        self.assertEqual(metrics["bleu"], 0.0)

    def test_normalization_lowercase_is_explicit(self):
        evaluator = TextComparisonEvaluator(lowercase=True)

        metrics = evaluator("Hello WORLD", "hello world")

        self.assertEqual(metrics["wer"], 0.0)
        self.assertEqual(metrics["chrf"], 100.0)

    def test_normalization_preserves_case_by_default(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("Hello WORLD", "hello world")

        self.assertEqual(metrics["wer"], 1.0)
        self.assertLess(metrics["chrf"], 100.0)

    def test_batch_length_mismatch_raises(self):
        evaluator = TextComparisonEvaluator()

        with self.assertRaisesRegex(ValueError, "same batch length"):
            evaluator(["hello"], ["hello", "world"])

    def test_text_input_rejects_bytes(self):
        evaluator = TextComparisonEvaluator()

        with self.assertRaisesRegex(TypeError, "prediction_text"):
            evaluator(b"hello", "hello")

    def test_text_input_rejects_non_string_items(self):
        evaluator = TextComparisonEvaluator()

        with self.assertRaisesRegex(TypeError, r"prediction_text\[0\]"):
            evaluator([1], ["hello"])

    def test_empty_reference_wer_policy(self):
        evaluator = TextComparisonEvaluator()

        empty_metrics = evaluator("", "")
        non_empty_prediction_metrics = evaluator("hello", "")

        self.assertEqual(empty_metrics["wer"], 0.0)
        self.assertEqual(empty_metrics["bleu"], 100.0)
        self.assertEqual(empty_metrics["chrf"], 100.0)
        self.assertEqual(non_empty_prediction_metrics["wer"], 1.0)
        self.assertEqual(non_empty_prediction_metrics["bleu"], 0.0)
        self.assertEqual(non_empty_prediction_metrics["chrf"], 0.0)

    def test_metric_keys_are_stable(self):
        evaluator = TextComparisonEvaluator()

        metrics = evaluator("a b c d", "a b c d")

        self.assertEqual(list(metrics), ["bleu", "wer", "chrf"])

    def test_empty_batch_raises(self):
        evaluator = TextComparisonEvaluator()

        with self.assertRaisesRegex(ValueError, "at least one item"):
            evaluator([], [])

    def test_wer_warns_and_uses_fallback_when_jiwer_is_missing(self):
        real_import = __import__

        def fail_jiwer_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "jiwer":
                raise ImportError("missing jiwer")
            return real_import(name, globals, locals, fromlist, level)

        with (
            patch("builtins.__import__", side_effect=fail_jiwer_import),
            self.assertWarnsRegex(RuntimeWarning, "fallback"),
        ):
            score = word_error_rate(["hello world"], ["hello there"])

        self.assertEqual(score, 0.5)

    def test_chrf_fallback_matches_sacrebleu_for_corpus_examples(self):
        examples = [
            (["Please cool Stella"], ["Please call Stella"]),
            (["the quick brown fox", "hello world"], ["the quick brown fox", "hello there"]),
        ]

        real_import = __import__

        def fail_sacrebleu_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name.startswith("sacrebleu"):
                raise ImportError("missing sacrebleu")
            return real_import(name, globals, locals, fromlist, level)

        for predictions, targets in examples:
            expected = corpus_chrf_score(predictions, targets)
            with (
                patch("builtins.__import__", side_effect=fail_sacrebleu_import),
                self.assertWarnsRegex(RuntimeWarning, "fallback"),
            ):
                fallback = corpus_chrf_score(predictions, targets)

            self.assertAlmostEqual(fallback, expected)


if __name__ == "__main__":
    unittest.main()
