from __future__ import annotations

from collections.abc import Sequence

from anytrain.evaluator import EvaluatorABC, MetricDict

from .normalization import TextNormalizationConfig, normalize_text_batch
from .scores import corpus_bleu_score, corpus_chrf_score, word_error_rate


class TextComparisonEvaluator(EvaluatorABC):
    """Lightweight text comparison evaluator.

    BLEU, WER, and chrF are implemented with the Python standard library so the
    optional text evaluator works in the default install. The public interface is
    intentionally small and can later be backed by mature metric libraries.
    """

    def __init__(
        self,
        *,
        strip: bool = True,
        collapse_whitespace: bool = True,
        lowercase: bool = False,
    ) -> None:
        super().__init__()
        self.normalization = TextNormalizationConfig(
            strip=strip,
            collapse_whitespace=collapse_whitespace,
            lowercase=lowercase,
        )

    def evaluate(
        self,
        prediction_text: str | Sequence[str],
        target_text: str | Sequence[str],
    ) -> MetricDict:
        predictions = normalize_text_batch(
            prediction_text,
            name="prediction_text",
            config=self.normalization,
        )
        targets = normalize_text_batch(
            target_text,
            name="target_text",
            config=self.normalization,
        )
        if len(predictions) != len(targets):
            raise ValueError(
                "prediction_text and target_text must have the same batch length: "
                f"got {len(predictions)} and {len(targets)}."
            )

        return {
            "bleu": corpus_bleu_score(predictions, targets),
            "wer": word_error_rate(predictions, targets),
            "chrf": corpus_chrf_score(predictions, targets),
        }
