"""Evaluation aggregator for OCR performance metrics and inference benchmarks.

Tracks sample-level metrics and aggregates macro/micro Character Error Rate (CER),
Word Error Rate (WER), exact match accuracy, mean latency, and throughput (FPS).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from .distance import (
    character_error_rate,
    exact_match_accuracy,
    levenshtein_distance,
    word_error_rate,
)


@dataclass(frozen=True)
class EvaluationResult:
    """Structured evaluation report containing aggregated OCR metrics."""

    total_samples: int
    exact_matches: int
    accuracy: float
    macro_cer: float
    micro_cer: float
    macro_wer: float
    micro_wer: float
    total_time_sec: float
    avg_latency_ms: float
    throughput_fps: float

    def to_dict(self) -> dict[str, Any]:
        """Convert result dataclass to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serialize result dataclass to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


class MetricsEvaluator:
    """Stateful evaluator accumulating predictions and computing aggregated metrics."""

    def __init__(self) -> None:
        """Initialize empty metric accumulators."""
        self.reset()

    def reset(self) -> None:
        """Clear all accumulated samples and latency stats."""
        self.references: list[str] = []
        self.hypotheses: list[str] = []
        self.latencies_ms: list[float] = []

    def update(
        self,
        reference: str,
        hypothesis: str,
        latency_ms: float = 0.0,
    ) -> None:
        """Record a single prediction against its reference ground truth.

        Args:
            reference: Target ground truth string.
            hypothesis: Model prediction string.
            latency_ms: Inference duration in milliseconds.
        """
        self.references.append(str(reference))
        self.hypotheses.append(str(hypothesis))
        self.latencies_ms.append(float(latency_ms))

    def update_batch(
        self,
        references: list[str],
        hypotheses: list[str],
        latencies_ms: Optional[list[float]] = None,
    ) -> None:
        """Record a batch of predictions against reference ground truths.

        Args:
            references: List of target ground truths.
            hypotheses: List of model predictions.
            latencies_ms: Optional list of per-sample latencies in milliseconds.

        Raises:
            ValueError: If lengths of references and hypotheses mismatch.
        """
        if len(references) != len(hypotheses):
            raise ValueError(
                f"Batch size mismatch: {len(references)} references vs {len(hypotheses)} hypotheses"
            )

        self.references.extend(references)
        self.hypotheses.extend(hypotheses)
        if latencies_ms is not None:
            self.latencies_ms.extend(latencies_ms)
        else:
            self.latencies_ms.extend([0.0] * len(references))

    def compute(self, total_elapsed_sec: Optional[float] = None) -> EvaluationResult:
        """Compute aggregated evaluation statistics across all stored samples.

        Args:
            total_elapsed_sec: Total batch wall-clock time in seconds. If None,
                calculated from accumulated per-sample latencies.

        Returns:
            EvaluationResult dataclass with macro/micro CER, WER, and latency metrics.
        """
        n_samples = len(self.references)
        if n_samples == 0:
            return EvaluationResult(
                total_samples=0,
                exact_matches=0,
                accuracy=0.0,
                macro_cer=0.0,
                micro_cer=0.0,
                macro_wer=0.0,
                micro_wer=0.0,
                total_time_sec=0.0,
                avg_latency_ms=0.0,
                throughput_fps=0.0,
            )

        exact_matches = 0
        sum_sample_cer = 0.0
        sum_sample_wer = 0.0

        total_char_dist = 0
        total_ref_chars = 0

        total_word_dist = 0
        total_ref_words = 0

        for ref, hyp in zip(self.references, self.hypotheses):
            # Exact match
            if ref == hyp:
                exact_matches += 1

            # Per-sample metrics for Macro calculation
            s_cer = character_error_rate(ref, hyp)
            s_wer = word_error_rate(ref, hyp)
            sum_sample_cer += s_cer
            sum_sample_wer += s_wer

            # Global counts for Micro calculation
            c_dist = levenshtein_distance(ref, hyp)
            total_char_dist += c_dist
            total_ref_chars += len(ref)

            ref_words = ref.strip().split()
            hyp_words = hyp.strip().split()
            w_dist = levenshtein_distance(ref_words, hyp_words)
            total_word_dist += w_dist
            total_ref_words += len(ref_words)

        # Micro metrics
        micro_cer = (
            float(total_char_dist) / float(total_ref_chars)
            if total_ref_chars > 0
            else (0.0 if total_char_dist == 0 else 1.0)
        )
        micro_wer = (
            float(total_word_dist) / float(total_ref_words)
            if total_ref_words > 0
            else (0.0 if total_word_dist == 0 else 1.0)
        )

        # Macro metrics
        macro_cer = sum_sample_cer / float(n_samples)
        macro_wer = sum_sample_wer / float(n_samples)
        accuracy = float(exact_matches) / float(n_samples)

        # Latency & Throughput calculation
        if total_elapsed_sec is not None and total_elapsed_sec > 0:
            total_sec = total_elapsed_sec
            avg_latency = (total_sec * 1000.0) / float(n_samples)
        else:
            total_ms = sum(self.latencies_ms)
            total_sec = total_ms / 1000.0
            avg_latency = total_ms / float(n_samples) if n_samples > 0 else 0.0

        fps = float(n_samples) / total_sec if total_sec > 0 else 0.0

        return EvaluationResult(
            total_samples=n_samples,
            exact_matches=exact_matches,
            accuracy=accuracy,
            macro_cer=macro_cer,
            micro_cer=micro_cer,
            macro_wer=macro_wer,
            micro_wer=micro_wer,
            total_time_sec=total_sec,
            avg_latency_ms=avg_latency,
            throughput_fps=fps,
        )
