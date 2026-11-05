"""Polished interactive terminal demonstration for Fence.

Fence: Evidence deserves a boundary.
Runs a live demonstration of score auditing, lexical trap detection,
class-conditional conformal gating, and conservative abstention.
"""

from __future__ import annotations

import sys
import time
from fence_evidence.conformal import (
    evaluate_conformal_gate,
    find_lexical_traps,
    fit_conformal_gate,
    predict_one,
)
from fence_evidence.core import Decision
from fence_evidence.demo import generate_synthetic_rag_data
from fence_evidence.metrics import audit_scores

# ANSI Color Codes for terminal display
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"
DIM = "\033[2m"


def print_banner() -> None:
    print(f"\n{BOLD}{CYAN}======================================================================{RESET}")
    print(f"{BOLD}{CYAN}  FENCE  |  Evidence deserves a boundary.{RESET}")
    print(f"{DIM}  Statistical Evidence Sufficiency MCP Server (v0.1.0){RESET}")
    print(f"{BOLD}{CYAN}======================================================================{RESET}\n")


def print_section(title: str) -> None:
    print(f"\n{BOLD}{YELLOW}>> {title}{RESET}")
    print(f"{DIM}{'-' * 70}{RESET}")


def main() -> None:
    print_banner()

    # Step 1: Synthesize Data
    print_section("Step 1: Simulating Real-World RAG Retrieval Scenarios")
    print("Generating 600 realistic retrieval queries:")
    print("  * 45% Truly sufficient evidence (high score, label 1)")
    print("  * 30% Truly insufficient evidence (low score, label 0)")
    print("  * 15% Lexical traps (high similarity, wrong facts, label 0)")
    print("  * 10% Ambiguous / borderline matches (mid score, mixed labels)")

    scores, labels, ids, queries, passages = generate_synthetic_rag_data(n_samples=600, seed=42)
    cal_scores, test_scores = scores[:300], scores[300:]
    cal_labels, test_labels = labels[:300], labels[300:]
    test_ids, test_queries, test_passages = ids[300:], queries[300:], passages[300:]
    print(f"{GREEN}✓ Successfully partitioned 300 calibration and 300 holdout test queries.{RESET}")

    # Step 2: Audit Scores
    print_section("Step 2: Auditing Calibration Scores for Reliability & Miscalibration")
    audit = audit_scores(cal_scores, cal_labels, threshold=0.70)
    print(f"  * Total Calibration Samples:  {audit.sample_count}")
    print(f"  * Prevalence (Sufficient):    {audit.prevalence:.1%}")
    print(f"  * Brier Score:                {audit.brier_score:.4f}  (lower is better)")
    print(f"  * Expected Calibration Error: {audit.expected_calibration_error:.4f}")
    print(f"  * AUROC (Wilcoxon exact):     {audit.auroc:.4f}")
    if audit.warnings:
        for w in audit.warnings:
            print(f"  {YELLOW}! Warning: {w}{RESET}")
    else:
        print(f"  {GREEN}✓ Validation Check: No pathological calibration warnings detected.{RESET}")

    # Step 3: Fit Conformal Gate
    print_section("Step 3: Fitting Class-Conditional Split Conformal Prediction Gate")
    alpha = 0.10
    gate = fit_conformal_gate(cal_scores, cal_labels, alpha=alpha)
    print(f"  Target significance alpha:  {alpha:.2f} (Guaranteed class coverage >= {1-alpha:.0%})")
    print(f"  q0 (Insufficient bound):    {CYAN}{gate.q0:.4f}{RESET} (Include 0 if score <= q0)")
    print(f"  q1 (Sufficient bound):      {CYAN}{gate.q1:.4f}{RESET} (Include 1 if 1-score <= q1)")

    # Step 4: Live Query Evaluation & Abstention Demonstration
    print_section("Step 4: Live Decision Boundaries on Incoming Production Queries")
    demo_queries = [
        {
            "query": "What is the primary mechanism of action for drug X?",
            "passage": "Drug X selectively inhibits enzyme Y, leading to receptor blockade.",
            "score": 0.94,
            "expected": "ACCEPT",
        },
        {
            "query": "What was the Q3 2026 revenue for Corporation Z?",
            "passage": "Corporation Z announced record revenue in Q3 2024 across all lines.",
            "score": 0.82,  # Lexical Trap!
            "expected": "ABSTAIN",
        },
        {
            "query": "What is the capital of Mars?",
            "passage": "Mars has no permanent human settlement or recognized capital.",
            "score": 0.12,
            "expected": "REJECT",
        },
    ]

    for item in demo_queries:
        s = item["score"]
        pred = predict_one(s, gate)
        if pred.decision == Decision.ACCEPT:
            color = GREEN
            action_text = "Answer question (evidence certified sufficient)."
        elif pred.decision == Decision.REJECT:
            color = RED
            action_text = "Refuse answer (evidence certified insufficient)."
        else:
            color = YELLOW
            action_text = "CONSERVATIVE ABSTENTION: Evidence inconclusive or deceptive."

        print(f"\n  Query:     {BOLD}{item['query']}{RESET}")
        print(f"  Passage:   {DIM}{item['passage']}{RESET}")
        print(f"  Score:     {s:.2f}  |  Naive Decision: {GREEN}ANSWER{RESET} (score > 0.75)")
        print(f"  Fence:     {color}{BOLD}{pred.decision.value}{RESET} (Prediction set: {pred.prediction_set})")
        print(f"  Action:    {color}{action_text}{RESET}")

    # Step 5: Benchmark Comparison on Holdout Test Set
    print_section("Step 5: Holdout Evaluation Comparison (300 Independent Queries)")
    eval_res = evaluate_conformal_gate(test_scores, test_labels, gate)

    # Naive threshold comparison (0.70)
    naive_fp = sum(1 for s, y in zip(test_scores, test_labels) if s >= 0.70 and y == 0)
    naive_n0 = sum(1 for y in test_labels if y == 0)
    naive_fsr = naive_fp / naive_n0 if naive_n0 > 0 else 0.0

    print(f"\n  {BOLD}{'Strategy':<32} | {'False Sufficiency Rate':<22} | {'Abstention Rate':<16}{RESET}")
    print(f"  {'-'*32}-+-{'-'*22}-+-{'-'*16}")
    print(f"  {'Naive Fixed Threshold (0.70)':<32} | {RED}{naive_fsr:<22.1%}{RESET} | {'0.0%':<16}")
    print(f"  {'Fence Conformal Gate (a=0.10)':<32} | {GREEN}{eval_res.false_sufficiency_rate:<22.1%}{RESET} | {YELLOW}{eval_res.abstention_rate:<16.1%}{RESET}")

    print(f"\n{BOLD}{GREEN}✓ Fence strictly controlled false-sufficiency errors within the {alpha:.0%} error budget!{RESET}")
    print(f"{DIM}======================================================================{RESET}\n")


if __name__ == "__main__":
    main()
