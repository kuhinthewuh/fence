"""Example: End-to-End RAG Pipeline Statistical Audit with Fence.

Fence: Evidence deserves a boundary.
Demonstrates calibration diagnostics, conformal gating, and holdout evaluation.
"""

from __future__ import annotations

from fence_evidence import (
    audit_scores,
    evaluate_conformal_gate,
    find_lexical_traps,
    fit_conformal_gate,
)
from fence_evidence.calibration import IsotonicCalibrator
from fence_evidence.demo import generate_synthetic_rag_data


def main() -> None:
    print("=== Fence RAG Pipeline Evidence Audit ===")

    # 1. Obtain retrieval validation dataset (simulated RAG)
    scores, labels, item_ids, queries, passages = generate_synthetic_rag_data(
        n_samples=500, seed=123
    )
    cal_scores, test_scores = scores[:250], scores[250:]
    cal_labels, test_labels = labels[:250], labels[250:]
    test_ids = item_ids[250:]
    test_queries = queries[250:]
    test_passages = passages[250:]

    # 2. Score Audit
    audit = audit_scores(cal_scores, cal_labels, threshold=0.70)
    print("1. Calibration Audit:")
    print(f"   Brier Score: {audit.brier_score:.4f}")
    print(f"   Expected Calibration Error (ECE): {audit.expected_calibration_error:.4f}")
    print(f"   AUROC: {audit.auroc:.4f}")
    print(f"   Prevalence: {audit.prevalence:.2%}")
    print(f"   Warnings: {len(audit.warnings)}")
    for w in audit.warnings:
        print(f"     * {w}")
    print()

    # 3. Fit Class-Conditional Conformal Gate
    gate = fit_conformal_gate(cal_scores, cal_labels, alpha=0.10)
    print("2. Fitted Conformal Gate:")
    print(f"   Alpha: {gate.alpha:.2f} (Target coverage >= 90%)")
    print(f"   q0 (insufficient bound): {gate.q0:.4f}")
    print(f"   q1 (sufficient bound):   {gate.q1:.4f}")
    print()

    # 4. Holdout Evaluation
    eval_res = evaluate_conformal_gate(test_scores, test_labels, gate)
    print("3. Independent Holdout Evaluation:")
    print(f"   Marginal Coverage: {eval_res.marginal_coverage:.2%}")
    print(f"   Class 0 Coverage:  {eval_res.class0_coverage:.2%}")
    print(f"   Class 1 Coverage:  {eval_res.class1_coverage:.2%}")
    print(f"   False-Sufficiency Rate (FSR): {eval_res.false_sufficiency_rate:.2%} (bound: <= 10.0%)")
    print(f"   Abstention Rate:   {eval_res.abstention_rate:.2%}")
    print(f"   Acceptance Rate:   {eval_res.acceptance_rate:.2%}")
    print(f"   Conformal Bounds Held: {eval_res.conformal_bounds_held}")
    print()

    # 5. Discover Lexical Traps
    traps = find_lexical_traps(
        test_scores,
        test_labels,
        threshold=0.70,
        item_ids=test_ids,
        queries=test_queries,
        passages=test_passages,
    )
    print("4. Lexical Trap Discovery:")
    print(f"   Total Traps Identified: {traps.trap_count}")
    print(f"   Trap Rate Among Insufficient: {traps.trap_rate_among_insufficient:.2%}")
    if traps.traps:
        first = traps.traps[0]
        print(f"   Sample Trap: ID={first.item_id}, Score={first.score:.4f}")
        print(f"   Query: {first.query}")
        print(f"   Passage: {first.passage}")
    print()


if __name__ == "__main__":
    main()

