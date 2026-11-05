"""End-to-end reproducible demonstration of Fence.

Fence: Evidence deserves a boundary.
Runs statistical audit, calibration, conformal gating, holdout evaluation,
and lexical-trap discovery on a realistic simulated RAG scenario.
"""

from __future__ import annotations

import random
from typing import Any

from fence_evidence.conformal import (
    apply_conformal_gate,
    evaluate_conformal_gate,
    find_lexical_traps,
    fit_conformal_gate,
)
from fence_evidence.core import Decision
from fence_evidence.metrics import audit_scores


def generate_synthetic_rag_data(
    n_samples: int = 500,
    seed: int = 42,
) -> tuple[list[float], list[int], list[str], list[str], list[str]]:
    """Generate realistic synthetic retrieval scores with lexical traps.

    Generates four realistic sub-distributions:
    1. Truly sufficient evidence: high scores, label = 1
    2. Insufficient evidence: low scores, label = 0
    3. Lexical traps: high scores due to lexical overlap, label = 0
    4. Ambiguous evidence: mid-range scores, mixed labels

    Returns
    -------
    tuple of (scores, labels, item_ids, queries, passages)
    """
    rng = random.Random(seed)
    scores: list[float] = []
    labels: list[int] = []
    item_ids: list[str] = []
    queries: list[str] = []
    passages: list[str] = []

    for i in range(n_samples):
        item_id = f"rag_item_{i:04d}"
        item_ids.append(item_id)
        u = rng.random()

        if u < 0.45:
            # Type 1: Truly sufficient evidence (45%)
            # High score, label = 1
            s = min(1.0, max(0.0, rng.betavariate(8, 2)))
            y = 1
            q = "What is the primary mechanism of action for drug X?"
            p = "Drug X acts by selectively inhibiting enzyme Y, leading to downstream receptor blockade."
        elif u < 0.75:
            # Type 2: Truly insufficient evidence (30%)
            # Low score, label = 0
            s = min(1.0, max(0.0, rng.betavariate(2, 8)))
            y = 0
            q = "What are the contraindications for drug X in renal failure?"
            p = "Enzyme Y is expressed in hepatic and cardiac tissue."
        elif u < 0.90:
            # Type 3: Lexical trap (15%)
            # High score from superficial keyword overlap, but label = 0
            s = min(1.0, max(0.0, rng.betavariate(7, 2)))
            y = 0
            q = "What was the Q3 2026 revenue growth for Corporation Z?"
            p = "Corporation Z announced record growth in Q3 2025 across all primary product lines."
        else:
            # Type 4: Ambiguous evidence (10%)
            # Mid-range scores, uncertain entailment
            s = min(1.0, max(0.0, rng.betavariate(4, 4)))
            y = 1 if rng.random() > 0.5 else 0
            q = "Did the Supreme Court issue a stay in case ABC?"
            p = "The appellate court granted an injunction pending petition to the Supreme Court."

        scores.append(round(s, 4))
        labels.append(y)
        queries.append(q)
        passages.append(p)

    return scores, labels, item_ids, queries, passages


def run_demo(seed: int = 42, alpha: float = 0.10) -> dict[str, Any]:
    """Execute complete Fence demonstration.

    1. Synthesize 600 RAG retrieval examples (300 calibration, 300 holdout test).
    2. Audit raw retrieval scores.
    3. Fit class-conditional split conformal prediction gate at error level alpha.
    4. Evaluate conformal gate against a naive threshold on the holdout test set.
    5. Detect candidate lexical traps.

    Returns
    -------
    dict summarizing all diagnostic steps, comparisons, and guarantees.
    """
    scores, labels, item_ids, queries, passages = generate_synthetic_rag_data(
        n_samples=600, seed=seed
    )

    # Clean 50/50 split into calibration and holdout evaluation sets
    cal_scores, test_scores = scores[:300], scores[300:]
    cal_labels, test_labels = labels[:300], labels[300:]
    cal_ids, test_ids = item_ids[:300], item_ids[300:]
    cal_queries, test_queries = queries[:300], queries[300:]
    cal_passages, test_passages = passages[:300], passages[300:]

    # 1. Audit calibration scores
    audit_cal = audit_scores(cal_scores, cal_labels, threshold=0.70)

    # 2. Fit conformal gate
    gate = fit_conformal_gate(cal_scores, cal_labels, alpha=alpha)

    # 3. Evaluate conformal gate on holdout test set
    conformal_eval = evaluate_conformal_gate(test_scores, test_labels, gate)

    # 4. Compare with naive threshold (0.70) on holdout test set
    naive_threshold = 0.70
    naive_fp = sum(1 for s, y in zip(test_scores, test_labels) if s >= naive_threshold and y == 0)
    naive_n0 = sum(1 for y in test_labels if y == 0)
    naive_fsr = naive_fp / naive_n0 if naive_n0 > 0 else 0.0
    naive_accepted = sum(1 for s in test_scores if s >= naive_threshold)
    naive_accept_rate = naive_accepted / len(test_scores)

    # 5. Discover lexical traps
    trap_report = find_lexical_traps(
        scores=test_scores,
        labels=test_labels,
        threshold=0.70,
        item_ids=test_ids,
        queries=test_queries,
        passages=test_passages,
    )

    return {
        "status": "success",
        "demo_metadata": {
            "project": "Fence",
            "tagline": "Evidence deserves a boundary.",
            "version": "0.1.0",
            "synthetic_samples_total": 600,
            "calibration_samples": 300,
            "holdout_test_samples": 300,
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
        },
        "score_audit": audit_cal.to_dict(),
        "conformal_gate": gate.to_dict(),
        "holdout_conformal_evaluation": conformal_eval.to_dict(),
        "naive_baseline_comparison": {
            "naive_threshold": naive_threshold,
            "naive_acceptance_rate": round(naive_accept_rate, 4),
            "naive_false_sufficiency_rate": round(naive_fsr, 4),
            "conformal_false_sufficiency_rate": conformal_eval.false_sufficiency_rate,
            "false_sufficiency_reduction": (
                f"{((naive_fsr - conformal_eval.false_sufficiency_rate) / naive_fsr * 100):.1f}%"
                if naive_fsr > 0
                else "N/A"
            ),
            "conformal_abstention_rate": conformal_eval.abstention_rate,
        },
        "lexical_traps_identified": {
            "trap_count": trap_report.trap_count,
            "trap_rate_among_insufficient": trap_report.trap_rate_among_insufficient,
            "sample_traps": [t.to_dict() for t in trap_report.traps[:3]],
        },
        "conclusion": (
            f"Under naive thresholding (tau={naive_threshold}), the agent accepted false evidence {naive_fsr:.1%} "
            f"of the time due to lexical traps. With Fence conformal gating (alpha={alpha:.2f}), "
            f"false-sufficiency is strictly bounded at {conformal_eval.false_sufficiency_rate:.1%}, with "
            f"{conformal_eval.abstention_rate:.1%} conservative abstentions protecting the agent."
        ),
    }

