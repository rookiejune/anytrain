from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from math import exp, log


def corpus_bleu_score(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    max_order: int = 4,
) -> float:
    """Return a deterministic lightweight corpus BLEU score on the 0-100 scale."""
    _validate_paired_corpus(predictions, targets)
    _validate_positive_integer(max_order, name="max_order")

    prediction_length = 0
    target_length = 0
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order

    for prediction, target in zip(predictions, targets, strict=True):
        prediction_tokens = prediction.split()
        target_tokens = target.split()
        prediction_length += len(prediction_tokens)
        target_length += len(target_tokens)

        for order in range(1, max_order + 1):
            prediction_ngrams = _ngram_counts(prediction_tokens, order)
            target_ngrams = _ngram_counts(target_tokens, order)
            overlap = prediction_ngrams & target_ngrams
            matches_by_order[order - 1] += sum(overlap.values())
            possible_matches_by_order[order - 1] += max(len(prediction_tokens) - order + 1, 0)

    if prediction_length == 0:
        return 100.0 if target_length == 0 else 0.0
    if target_length == 0:
        return 0.0

    precisions = [
        matches / possible
        for matches, possible in zip(matches_by_order, possible_matches_by_order, strict=True)
        if possible > 0
    ]
    if not precisions or any(precision == 0.0 for precision in precisions):
        return 0.0

    brevity_penalty = 1.0
    if prediction_length < target_length:
        brevity_penalty = exp(1.0 - target_length / prediction_length)

    mean_log_precision = sum(log(precision) for precision in precisions) / len(precisions)
    return 100.0 * brevity_penalty * exp(mean_log_precision)


def word_error_rate(predictions: Sequence[str], targets: Sequence[str]) -> float:
    """Return corpus WER using word-level Levenshtein distance over reference tokens."""
    _validate_paired_corpus(predictions, targets)

    total_distance = 0
    total_reference_tokens = 0
    total_empty_reference_errors = 0
    total_empty_references = 0
    for prediction, target in zip(predictions, targets, strict=True):
        prediction_tokens = prediction.split()
        target_tokens = target.split()
        if not target_tokens:
            total_empty_references += 1
            if prediction_tokens:
                total_empty_reference_errors += 1
            continue

        total_distance += _levenshtein_distance(prediction_tokens, target_tokens)
        total_reference_tokens += len(target_tokens)

    if total_reference_tokens == 0:
        if total_empty_reference_errors == 0:
            return 0.0
        return float(total_empty_reference_errors / total_empty_references)
    return float((total_distance + total_empty_reference_errors) / total_reference_tokens)


def corpus_chrf_score(
    predictions: Sequence[str],
    targets: Sequence[str],
    *,
    max_order: int = 6,
    beta: float = 2.0,
) -> float:
    """Return a deterministic lightweight chrF score on the 0-100 scale."""
    _validate_paired_corpus(predictions, targets)
    _validate_positive_integer(max_order, name="max_order")
    _validate_positive_float(beta, name="beta")

    total_matches = 0
    total_prediction_ngrams = 0
    total_target_ngrams = 0
    for prediction, target in zip(predictions, targets, strict=True):
        if prediction == "" and target == "":
            total_matches += max_order
            total_prediction_ngrams += max_order
            total_target_ngrams += max_order
            continue

        for order in range(1, max_order + 1):
            prediction_ngrams = _char_ngram_counts(prediction, order)
            target_ngrams = _char_ngram_counts(target, order)
            total_matches += sum((prediction_ngrams & target_ngrams).values())
            total_prediction_ngrams += sum(prediction_ngrams.values())
            total_target_ngrams += sum(target_ngrams.values())

    if total_prediction_ngrams == 0 or total_target_ngrams == 0:
        return 0.0

    precision = total_matches / total_prediction_ngrams
    recall = total_matches / total_target_ngrams
    if precision == 0.0 or recall == 0.0:
        return 0.0

    beta_square = beta * beta
    return 100.0 * (1.0 + beta_square) * precision * recall / (beta_square * precision + recall)


def _validate_paired_corpus(predictions: Sequence[str], targets: Sequence[str]) -> None:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have the same length.")
    if not predictions:
        raise ValueError("predictions and targets must contain at least one item.")


def _validate_positive_integer(value: int, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _validate_positive_float(value: float, *, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, float | int):
        raise TypeError(f"{name} must be a float.")
    if value <= 0:
        raise ValueError(f"{name} must be positive.")


def _ngram_counts(tokens: Sequence[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


def _char_ngram_counts(text: str, order: int) -> Counter[str]:
    return Counter(text[index : index + order] for index in range(len(text) - order + 1))


def _levenshtein_distance(prediction_tokens: Sequence[str], target_tokens: Sequence[str]) -> int:
    previous = list(range(len(target_tokens) + 1))
    for prediction_index, prediction_token in enumerate(prediction_tokens, start=1):
        current = [prediction_index]
        for target_index, target_token in enumerate(target_tokens, start=1):
            substitution_cost = 0 if prediction_token == target_token else 1
            current.append(
                min(
                    previous[target_index] + 1,
                    current[target_index - 1] + 1,
                    previous[target_index - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]
