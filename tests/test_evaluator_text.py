import unittest

from anytrain.evaluator import EvaluatorABC
from anytrain.evaluator.text import TextComparisonEvaluator


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


if __name__ == "__main__":
    unittest.main()
