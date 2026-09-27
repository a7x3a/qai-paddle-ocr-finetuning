"""OCR accuracy metrics backed by rapidfuzz (C speed) so full-split scoring is fast.

Definitions used everywhere in this project (corpus level, not averaged per sample):

    CER = sum(levenshtein(reference, hypothesis)) / sum(len(reference))
    WER = sum(levenshtein(ref_words, hyp_words)) / sum(len(ref_words))
    exact_match   = mean(reference == hypothesis)
    similarity    = mean(1 - lev / max(len(ref), len(hyp), 1))   (secondary, per-sample)

Reference length is always the ground-truth length, so CER and WER are directly
comparable between models and never exceed 1.0 for pure insertions-only drift
in the denominator sense, which is the standard definition.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from rapidfuzz.distance import Levenshtein


@dataclass
class Accumulator:
    """Running corpus-level counters for one metric group."""

    samples: int = 0
    char_errors: int = 0
    char_total: int = 0
    word_errors: int = 0
    word_total: int = 0
    exact: int = 0
    similarity_sum: float = 0.0
    confidence_sum: float = 0.0
    empty_predictions: int = 0
    length_histogram: dict[str, int] = field(default_factory=dict)

    def add(self, reference: str, hypothesis: str, confidence: float = 0.0, bucket: str | None = None) -> None:
        distance = Levenshtein.distance(reference, hypothesis)
        ref_words = reference.split()
        hyp_words = hypothesis.split()
        word_distance = Levenshtein.distance(ref_words, hyp_words)

        self.samples += 1
        self.char_errors += distance
        self.char_total += max(len(reference), 1)
        self.word_errors += word_distance
        self.word_total += max(len(ref_words), 1)
        self.exact += int(reference == hypothesis)
        self.similarity_sum += 1.0 - distance / max(len(reference), len(hypothesis), 1)
        self.confidence_sum += confidence
        self.empty_predictions += int(not hypothesis.strip())
        if bucket is not None:
            self.length_histogram[bucket] = self.length_histogram.get(bucket, 0) + 1

    def merge(self, other: "Accumulator") -> None:
        self.samples += other.samples
        self.char_errors += other.char_errors
        self.char_total += other.char_total
        self.word_errors += other.word_errors
        self.word_total += other.word_total
        self.exact += other.exact
        self.similarity_sum += other.similarity_sum
        self.confidence_sum += other.confidence_sum
        self.empty_predictions += other.empty_predictions
        for key, value in other.length_histogram.items():
            self.length_histogram[key] = self.length_histogram.get(key, 0) + value

    def summary(self) -> dict:
        if not self.samples:
            return {"samples": 0}
        return {
            "samples": self.samples,
            "cer": self.char_errors / self.char_total,
            "wer": self.word_errors / self.word_total,
            "exact_match_accuracy": self.exact / self.samples,
            "character_accuracy": 1.0 - self.char_errors / self.char_total,
            "word_accuracy": 1.0 - self.word_errors / self.word_total,
            "similarity": self.similarity_sum / self.samples,
            "mean_confidence": self.confidence_sum / self.samples,
            "char_errors": self.char_errors,
            "char_total": self.char_total,
            "word_errors": self.word_errors,
            "word_total": self.word_total,
            "empty_predictions": self.empty_predictions,
            "empty_prediction_rate": self.empty_predictions / self.samples,
        }


class MetricSuite:
    """Overall metrics plus breakdowns by domain, script and label length."""

    def __init__(self) -> None:
        self.overall = Accumulator()
        self.by_domain: dict[str, Accumulator] = defaultdict(Accumulator)
        self.by_script: dict[str, Accumulator] = defaultdict(Accumulator)
        self.by_length: dict[str, Accumulator] = defaultdict(Accumulator)
        self.by_image: dict[str, Accumulator] = defaultdict(Accumulator)

    def add(
        self,
        reference: str,
        hypothesis: str,
        confidence: float = 0.0,
        domain: str = "unknown",
        script: str = "unknown",
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        confidence = float(confidence)
        self.overall.add(reference, hypothesis, confidence, _length_bucket(len(reference)))
        self.by_domain[domain].add(reference, hypothesis, confidence, _length_bucket(len(reference)))
        self.by_script[script].add(reference, hypothesis, confidence, _length_bucket(len(reference)))
        self.by_length[_length_bucket(len(reference))].add(reference, hypothesis, confidence)
        self.by_image[_image_bucket(width, height)].add(reference, hypothesis, confidence)

    def report(self) -> dict:
        return {
            "overall": self.overall.summary(),
            "by_domain": {key: value.summary() for key, value in sorted(self.by_domain.items())},
            "by_script": {key: value.summary() for key, value in sorted(self.by_script.items())},
            "by_label_length": {key: value.summary() for key, value in sorted(self.by_length.items())},
            "by_image_size": {key: value.summary() for key, value in sorted(self.by_image.items())},
        }


def _length_bucket(length: int) -> str:
    if length <= 1:
        return "1"
    if length <= 3:
        return "2-3"
    if length <= 6:
        return "4-6"
    if length <= 12:
        return "7-12"
    if length <= 24:
        return "13-24"
    return "25+"


def _image_bucket(width: int | None, height: int | None) -> str:
    if not width or not height:
        return "unknown"
    if width <= 32 and height <= 32:
        return "tiny(<=32x32)"
    if width <= 64 or height <= 64:
        return "small(<=64)"
    if width <= 160:
        return "medium(<=160)"
    return "large(>160)"


def compare(baseline: dict, candidate: dict) -> dict:
    """Compare two report payloads. Lower CER/WER is better, higher accuracy is better."""
    rows = {}
    for key, better in (
        ("cer", "lower"),
        ("wer", "lower"),
        ("exact_match_accuracy", "higher"),
        ("character_accuracy", "higher"),
        ("word_accuracy", "higher"),
        ("similarity", "higher"),
    ):
        base_value = baseline.get("overall", {}).get(key)
        new_value = candidate.get("overall", {}).get(key)
        if base_value is None or new_value is None:
            continue
        delta = new_value - base_value
        improved = delta < 0 if better == "lower" else delta > 0
        rows[key] = {
            "baseline": base_value,
            "candidate": new_value,
            "delta": delta,
            "verdict": "improved" if abs(delta) < 1e-12 else ("improved" if improved else "regressed"),
        }
    return rows
