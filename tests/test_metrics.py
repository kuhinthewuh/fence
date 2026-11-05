"""Tests for statistical metrics in fence_evidence.metrics."""

from __future__ import annotations

import pytest

from fence_evidence.metrics import (
    audit_scores,
    compute_auroc,
    compute_brier_score,
    compute_confusion_and_rates,
    compute_reliability_bins,
)


def test_brier_score_perfect() -> None:
    scores = [1.0, 0.0, 1.0, 0.0]
    labels = [1, 0, 1, 0]
    assert compute_brier_score(scores, labels) == 0.0


def test_brier_score_worst() -> None:
    scores = [0.0, 1.0]
    labels = [1, 0]
    assert compute_brier_score(scores, labels) == 1.0


def test_brier_score_constant_half() -> None:
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [1, 0, 1, 0]
    assert compute_brier_score(scores, labels) == 0.25


def test_reliability_bins_and_ece() -> None:
    scores = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    labels = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    ece, bins = compute_reliability_bins(scores, labels, num_bins=10)
    assert 0.0 <= ece <= 1.0
    assert len(bins) == 10
    total_count = sum(b.count for b in bins)
    assert total_count == len(scores)


def test_auroc_perfect_separation() -> None:
    scores = [0.1, 0.2, 0.3, 0.8, 0.9, 0.95]
    labels = [0, 0, 0, 1, 1, 1]
    auroc = compute_auroc(scores, labels)
    assert auroc == 1.0


def test_auroc_inverted_separation() -> None:
    scores = [0.8, 0.9, 0.95, 0.1, 0.2, 0.3]
    labels = [0, 0, 0, 1, 1, 1]
    auroc = compute_auroc(scores, labels)
    assert auroc == 0.0


def test_auroc_tied_scores() -> None:
    # All scores tied at 0.5
    scores = [0.5, 0.5, 0.5, 0.5]
    labels = [0, 0, 1, 1]
    auroc = compute_auroc(scores, labels)
    assert auroc == 0.5

    # Partial ties
    scores = [0.2, 0.5, 0.5, 0.8]
    labels = [0, 0, 1, 1]
    auroc = compute_auroc(scores, labels)
    # Ranks: 0.2 -> 1, 0.5 -> 2.5, 0.5 -> 2.5, 0.8 -> 4
    # Label 1 indices: index 2 (rank 2.5) and index 3 (rank 4). Sum = 6.5.
    # U1 = 6.5 - 2*3/2 = 3.5. AUROC = 3.5 / 4 = 0.875.
    assert auroc == 0.875


def test_auroc_single_class() -> None:
    scores = [0.2, 0.5, 0.8]
    labels = [1, 1, 1]
    assert compute_auroc(scores, labels) == 0.5


def test_confusion_and_rates() -> None:
    scores = [0.9, 0.8, 0.4, 0.1]
    labels = [1, 0, 1, 0]
    tp, fp, tn, fn, precision, recall, f1 = compute_confusion_and_rates(scores, labels, threshold=0.5)
    assert tp == 1
    assert fp == 1
    assert tn == 1
    assert fn == 1
    assert precision == 0.5
    assert recall == 0.5
    assert f1 == 0.5


def test_audit_scores_full() -> None:
    scores = [0.95, 0.98, 0.99, 0.01, 0.02, 0.05] * 10
    labels = [1, 1, 1, 0, 0, 0] * 10
    res = audit_scores(scores, labels, threshold=0.5)
    assert res.sample_count == 60
    assert res.positive_count == 30
    assert res.negative_count == 30
    assert res.prevalence == 0.5
    assert res.auroc == 1.0
    assert len(res.warnings) == 0
    d = res.to_dict()
    assert d["sample_count"] == 60
