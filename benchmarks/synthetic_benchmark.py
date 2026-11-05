"""Reproducible synthetic benchmark for Fence.

Fence: Evidence deserves a boundary.
Evaluates class-conditional split conformal prediction against naive thresholding
on synthetic RAG retrieval data with simulated lexical traps.

Explicit Notice: This benchmark evaluates simulated synthetic retrieval distributions.
It is designed for controlled statistical verification and algorithmic reproducibility.
"""

from __future__ import annotations

import json
import random
import time
from typing import Any

from fence_evidence.conformal import (
    evaluate_conformal_gate,
    find_lexical_traps,
    fit_conformal_gate,
    predict_one,
)
from fence_evidence.core import Decision
from fence_evidence.demo import generate_synthetic_rag_data
from fence_evidence.metrics import audit_scores


def run_benchmark(
    n_samples: int = 2000,
    seed: int = 42,
    alpha: float = 0.10,
    naive_thresholds: tuple[float, ...] = (0.60, 0.70, 0.80),
) -> dict[str, Any]:
    """Execute complete reproducible benchmark."""
    t_start = time.perf_counter()

    # 1. Generate deterministic synthetic data
    scores, labels, item_ids, queries, passages = generate_synthetic_rag_data(
        n_samples=n_samples, seed=seed
    )

    split_idx = n_samples // 2
    cal_scores, test_scores = scores[:split_idx], scores[split_idx:]
    cal_labels, test_labels = labels[:split_idx], labels[split_idx:]
    test_ids = item_ids[split_idx:]
    test_queries = queries[split_idx:]
    test_passages = passages[split_idx:]

    # 2. Audit calibration set
    t_audit_start = time.perf_counter()
    audit_cal = audit_scores(cal_scores, cal_labels, threshold=0.70)
    audit_time_ms = (time.perf_counter() - t_audit_start) * 1000

    # 3. Fit conformal gate
    t_fit_start = time.perf_counter()
    gate = fit_conformal_gate(cal_scores, cal_labels, alpha=alpha)
    fit_time_ms = (time.perf_counter() - t_fit_start) * 1000

    # 4. Evaluate conformal gate on test set
    t_eval_start = time.perf_counter()
    conformal_eval = evaluate_conformal_gate(test_scores, test_labels, gate)
    conformal_eval_time_ms = (time.perf_counter() - t_eval_start) * 1000

    # 5. Evaluate naive fixed thresholds on test set
    naive_results = []
    test_n0 = sum(1 for y in test_labels if y == 0)
    test_n1 = sum(1 for y in test_labels if y == 1)
    n_test = len(test_scores)

    for thresh in naive_thresholds:
        fp = sum(1 for s, y in zip(test_scores, test_labels) if s >= thresh and y == 0)
        tp = sum(1 for s, y in zip(test_scores, test_labels) if s >= thresh and y == 1)
        accepted = fp + tp
        fsr = fp / test_n0 if test_n0 > 0 else 0.0
        acc_rate = accepted / n_test if n_test > 0 else 0.0
        cov1 = tp / test_n1 if test_n1 > 0 else 0.0

        naive_results.append({
            "threshold": thresh,
            "acceptance_rate": round(acc_rate, 4),
            "false_sufficiency_rate": round(fsr, 4),
            "class1_coverage": round(cov1, 4),
            "abstention_rate": 0.0,  # Naive thresholding never abstains
        })

    # 6. Lexical trap detection
    t_trap_start = time.perf_counter()
    traps = find_lexical_traps(
        scores=test_scores,
        labels=test_labels,
        threshold=0.70,
        item_ids=test_ids,
        queries=test_queries,
        passages=test_passages,
    )
    trap_time_ms = (time.perf_counter() - t_trap_start) * 1000

    total_time_ms = (time.perf_counter() - t_start) * 1000

    report = {
        "benchmark_metadata": {
            "name": "Fence Synthetic Evidence Sufficiency Benchmark",
            "version": "0.1.0",
            "is_synthetic": True,
            "random_seed": seed,
            "total_samples": n_samples,
            "calibration_samples": split_idx,
            "test_samples": n_samples - split_idx,
            "alpha": alpha,
            "target_coverage": 1.0 - alpha,
        },
        "timings_ms": {
            "score_audit": round(audit_time_ms, 2),
            "gate_fitting": round(fit_time_ms, 2),
            "conformal_evaluation": round(conformal_eval_time_ms, 2),
            "trap_discovery": round(trap_time_ms, 2),
            "total_runtime": round(total_time_ms, 2),
        },
        "calibration_audit": {
            "brier_score": audit_cal.brier_score,
            "expected_calibration_error": audit_cal.expected_calibration_error,
            "auroc": audit_cal.auroc,
            "warnings_count": len(audit_cal.warnings),
        },
        "conformal_gate": {
            "alpha": gate.alpha,
            "q0_insufficient_threshold": gate.q0,
            "q1_sufficient_threshold": gate.q1,
            "n0_cal": gate.n0,
            "n1_cal": gate.n1,
        },
        "conformal_holdout_results": {
            "target_coverage": conformal_eval.target_coverage,
            "marginal_coverage": conformal_eval.marginal_coverage,
            "class0_coverage": conformal_eval.class0_coverage,
            "class1_coverage": conformal_eval.class1_coverage,
            "false_sufficiency_rate": conformal_eval.false_sufficiency_rate,
            "acceptance_rate": conformal_eval.acceptance_rate,
            "abstention_rate": conformal_eval.abstention_rate,
            "rejection_rate": conformal_eval.rejection_rate,
            "conformal_bounds_held": conformal_eval.conformal_bounds_held,
        },
        "naive_baseline_comparison": naive_results,
        "lexical_traps_summary": {
            "total_traps_identified": traps.trap_count,
            "trap_rate_among_insufficient": traps.trap_rate_among_insufficient,
        },
    }

    return report


if __name__ == "__main__":
    results = run_benchmark()
    print(json.dumps(results, indent=2))

