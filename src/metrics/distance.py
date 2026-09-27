"""Algorithmic distance metrics for OCR sequence evaluation.

Implements high-performance, deterministic Levenshtein distance using 2-row dynamic programming
(Wagner-Fischer), Character Error Rate (CER), Word Error Rate (WER), and Exact Match Accuracy.
"""

from __future__ import annotations

from typing import Sequence, TypeVar

T = TypeVar("T")


def levenshtein_distance(seq1: Sequence[T], seq2: Sequence[T]) -> int:
    """Compute the Levenshtein edit distance between two arbitrary sequences.

    Uses an optimized Wagner-Fischer algorithm with two-row buffer allocation,
    guaranteeing O(min(|seq1|, |seq2|)) space complexity and O(|seq1| * |seq2|) time.

    Args:
        seq1: First sequence (e.g. reference characters or words).
        seq2: Second sequence (e.g. hypothesis characters or words).

    Returns:
        Minimum number of single-element insertions, deletions, or substitutions.
    """
    len1, len2 = len(seq1), len(seq2)

    # Fast-path identical sequences or empty sequences
    if seq1 == seq2:
        return 0
    if len1 == 0:
        return len2
    if len2 == 0:
        return len1

    # Ensure seq2 is the shorter sequence to minimize auxiliary buffer
    if len1 < len2:
        seq1, seq2 = seq2, seq1
        len1, len2 = len2, len1

    current_row = list(range(len2 + 1))

    for i in range(1, len1 + 1):
        prev_row = current_row
        current_row = [i] + [0] * len2
        elem1 = seq1[i - 1]

        for j in range(1, len2 + 1):
            elem2 = seq2[j - 1]
            cost = 0 if elem1 == elem2 else 1

            insert_cost = current_row[j - 1] + 1
            delete_cost = prev_row[j] + 1
            subst_cost = prev_row[j - 1] + cost

            if insert_cost <= delete_cost and insert_cost <= subst_cost:
                current_row[j] = insert_cost
            elif delete_cost <= subst_cost:
                current_row[j] = delete_cost
            else:
                current_row[j] = subst_cost

    return current_row[len2]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate (CER) between reference and hypothesis.

    Formula:
        CER = Levenshtein(reference, hypothesis) / max(1, len(reference))

    Edge Cases:
        - If both reference and hypothesis are empty, CER is 0.0.
        - If reference is empty and hypothesis has length L, CER is L.

    Args:
        reference: Ground truth transcription string.
        hypothesis: Model predicted string.

    Returns:
        Floating point CER value (>= 0.0).
    """
    ref_len = len(reference)
    hyp_len = len(hypothesis)

    if ref_len == 0 and hyp_len == 0:
        return 0.0
    if ref_len == 0:
        return float(hyp_len)

    dist = levenshtein_distance(reference, hypothesis)
    return float(dist) / float(ref_len)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate (WER) between reference and hypothesis.

    Formula:
        WER = Levenshtein(words(reference), words(hypothesis)) / max(1, len(words(reference)))

    Edge Cases:
        - If both reference and hypothesis have zero words, WER is 0.0.
        - If reference has zero words and hypothesis has W words, WER is W.

    Args:
        reference: Ground truth transcription string.
        hypothesis: Model predicted string.

    Returns:
        Floating point WER value (>= 0.0).
    """
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()

    ref_count = len(ref_words)
    hyp_count = len(hyp_words)

    if ref_count == 0 and hyp_count == 0:
        return 0.0
    if ref_count == 0:
        return float(hyp_count)

    dist = levenshtein_distance(ref_words, hyp_words)
    return float(dist) / float(ref_count)


def exact_match_accuracy(reference: str, hypothesis: str) -> float:
    """Calculate Exact Match Accuracy flag (1.0 for exact equality, 0.0 otherwise).

    Args:
        reference: Ground truth transcription string.
        hypothesis: Model predicted string.

    Returns:
        1.0 if identical else 0.0.
    """
    return 1.0 if reference == hypothesis else 0.0
