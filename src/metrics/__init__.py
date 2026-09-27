"""OCR evaluation and distance metric calculation modules."""

from .distance import (
    character_error_rate,
    exact_match_accuracy,
    levenshtein_distance,
    word_error_rate,
)
from .evaluator import EvaluationResult, MetricsEvaluator

__all__ = [
    "levenshtein_distance",
    "character_error_rate",
    "word_error_rate",
    "exact_match_accuracy",
    "MetricsEvaluator",
    "EvaluationResult",
]
