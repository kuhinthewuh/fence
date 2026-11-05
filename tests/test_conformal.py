"""Tests for class-conditional split conformal prediction in fence_evidence.conformal."""

from __future__ import annotations

import math
import pytest

from fence_evidence.conformal import (
    apply_conformal_gate,
    evaluate_conformal_gate,
    find_lexical_traps,
    fit_conformal_gate,
    predict_one,
)
from fence_evidence.core import Decision


def test_fit_conformal_gate_quantiles() -> None:
    # 19 class 0 examples with scores 0.05, 0.10, ..., 0.95
    # 19 class 1 examples with scores 0.05, 0.10, ..., 0.95
    scores0 = [i * 0.05 for i in range(1, 20)]
    scores1 = [i * 0.05 for i in range(1, 20)]

    scores = scores0 + scores1
    labels = [0] * len(scores0) + [1] * len(scores1)

    alpha = 0.10
    gate = fit_conformal_gate(scores, labels, alpha=alpha)

    assert gate.n0 == 19
    assert gate.n1 == 19
    # k = ceil((19 + 1) * 0.90) = ceil(18.0) = 18
    assert gate.rank0 == 18
    assert gate.rank1 == 18
    assert not gate.degenerate0
    assert not gate.degenerate1
    # 18th element in scores0 sorted: 18 * 0.05 = 0.90
    assert gate.q0 == 0.90
    # For class 1: nonconformity is 1 - s.
    # Scores are 0.05 to 0.95, so 1 - s is 0.05 to 0.95.
    # 18th element is 0.90
    assert gate.q1 == 0.90


def test_fit_conformal_gate_small_sample_infinite_quantile() -> None:
    # If n = 4, alpha = 0.10:
    # k = ceil((4 + 1) * 0.90) = ceil(4.5) = 5 > 4.
    # Must yield infinity rather than pretending a threshold is supported!
    scores = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]  # n0 = 4, n1 = 4
    gate = fit_conformal_gate(scores, labels, alpha=0.10)

    assert gate.n0 == 4
    assert gate.n1 == 4
    assert math.isinf(gate.q0)
    assert math.isinf(gate.q1)
    assert gate.degenerate0
    assert gate.degenerate1


def test_prediction_rule_decisions() -> None:
    # Gate with q0 = 0.40, q1 = 0.30
    # Include 0 <=> s <= 0.40
    # Include 1 <=> 1 - s <= 0.30 <=> s >= 0.70
    from fence_evidence.core import ConformalGate
    gate = ConformalGate(alpha=0.10, q0=0.40, q1=0.30, n0=50, n1=50, rank0=46, rank1=46)

    # Score = 0.85: s > 0.40 (not 0) and s >= 0.70 (is 1) -> set is [1] -> ACCEPT
    p_accept = predict_one(0.85, gate)
    assert p_accept.decision == Decision.ACCEPT
    assert p_accept.prediction_set == [1]
    assert p_accept.is_singleton

    # Score = 0.20: s <= 0.40 (is 0) and s < 0.70 (not 1) -> set is [0] -> REJECT
    p_reject = predict_one(0.20, gate)
    assert p_reject.decision == Decision.REJECT
    assert p_reject.prediction_set == [0]
    assert p_reject.is_singleton

    # Score = 0.55: s > 0.40 (not 0) and s < 0.70 (not 1) -> set is [] -> ABSTAIN
    p_empty = predict_one(0.55, gate)
    assert p_empty.decision == Decision.ABSTAIN
    assert p_empty.prediction_set == []
    assert not p_empty.is_singleton

    # Overlapping gate: q0 = 0.80, q1 = 0.80
    # Include 0 <=> s <= 0.80
    # Include 1 <=> s >= 0.20
    # Score 0.50: in both -> set [0, 1] -> ABSTAIN
    gate_overlap = ConformalGate(alpha=0.10, q0=0.80, q1=0.80, n0=50, n1=50, rank0=46, rank1=46)
    p_both = predict_one(0.50, gate_overlap)
    assert p_both.decision == Decision.ABSTAIN
    assert p_both.prediction_set == [0, 1]
    assert not p_both.is_singleton


def test_apply_conformal_gate_batch() -> None:
    from fence_evidence.core import ConformalGate
    gate = ConformalGate(alpha=0.10, q0=0.40, q1=0.30, n0=50, n1=50, rank0=46, rank1=46)
    preds = apply_conformal_gate([0.1, 0.5, 0.9], gate)
    assert len(preds) == 3
    assert preds[0].decision == Decision.REJECT
    assert preds[1].decision == Decision.ABSTAIN
    assert preds[2].decision == Decision.ACCEPT


def test_evaluate_conformal_gate_holdout() -> None:
    from fence_evidence.core import ConformalGate
    gate = ConformalGate(alpha=0.10, q0=0.50, q1=0.40, n0=100, n1=100, rank0=91, rank1=91)

    eval_scores = [0.1, 0.2, 0.8, 0.9] * 25  # 100 samples
    eval_labels = [0, 0, 1, 1] * 25
    eval_res = evaluate_conformal_gate(eval_scores, eval_labels, gate)

    assert eval_res.eval_count == 100
    assert eval_res.n0 == 50
    assert eval_res.n1 == 50
    assert eval_res.target_coverage == 0.90
    assert eval_res.class0_coverage == 1.0  # 0.1, 0.2 <= 0.50
    assert eval_res.class1_coverage == 1.0  # 0.8, 0.9 >= 0.60
    assert eval_res.false_sufficiency_rate == 0.0


def test_find_lexical_traps() -> None:
    scores = [0.95, 0.85, 0.40, 0.10, 0.88]
    labels = [1, 0, 0, 0, 0]  # indices 1 and 4 are traps (score >= 0.80 but label 0)
    items = ["doc_0", "doc_1", "doc_2", "doc_3", "doc_4"]
    queries = ["q0", "q1", "q2", "q3", "q4"]

    report = find_lexical_traps(scores, labels, threshold=0.80, item_ids=items, queries=queries)
    assert report.total_examined == 5
    assert report.insufficient_count == 4
    assert report.trap_count == 2
    assert report.traps[0].item_id == "doc_1"
    assert report.traps[1].item_id == "doc_4"
    assert report.traps[0].query == "q1"

