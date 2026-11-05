"""Example: AI Agent Evidence Sufficiency Verification with Fence.

Fence: Evidence deserves a boundary.
Shows how an agent verifies retrieval scores before answering, preventing
hallucination or misleading answers on lexical traps.
"""

from __future__ import annotations

from fence_evidence import (
    Decision,
    apply_conformal_gate,
    fit_conformal_gate,
    predict_one,
)


def run_agent_workflow() -> None:
    print("=== Fence AI Agent Verification Example ===")

    # 1. Calibration history from verified prior retrieval queries
    cal_scores = [
        0.92, 0.88, 0.85, 0.79, 0.95, 0.81, 0.89, 0.74, 0.91, 0.86,  # y = 1 (sufficient)
        0.12, 0.25, 0.31, 0.18, 0.42, 0.29, 0.35, 0.21, 0.15, 0.38,  # y = 0 (insufficient)
        0.82, 0.77, 0.85,                                             # y = 0 (lexical traps!)
    ]
    cal_labels = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0,
    ]

    # 2. Fit conformal boundary with error rate alpha = 0.10 (90% coverage)
    gate = fit_conformal_gate(cal_scores, cal_labels, alpha=0.10)
    print(f"Fitted Conformal Gate (alpha={gate.alpha:.2f}):")
    print(f"  q0 (insufficient nonconformity quantile): {gate.q0:.4f}")
    print(f"  q1 (sufficient nonconformity quantile):   {gate.q1:.4f}")
    print()

    # 3. New incoming queries and retrieval results
    incoming_cases = [
        {
            "query": "What is the capital of France?",
            "retrieved_passage": "Paris is the capital and most populous city of France.",
            "retriever_score": 0.94,
        },
        {
            "query": "What is the efficacy of drug A in pediatric clinical trials?",
            "retrieved_passage": "Drug A was approved for adult clinical trials following Phase III results.",
            "retriever_score": 0.81,  # Lexical trap! High keywords, wrong population.
        },
        {
            "query": "What was the company's net income in fiscal year 2024?",
            "retrieved_passage": "Quarterly operating margins fluctuated throughout fiscal 2024.",
            "retriever_score": 0.52,  # Ambiguous / weak evidence
        },
        {
            "query": "Who won the 2026 World Cup?",
            "retrieved_passage": "Recipes for traditional sourdough bread.",
            "retriever_score": 0.08,  # Completely irrelevant
        },
    ]

    for case in incoming_cases:
        s = case["retriever_score"]
        pred = predict_one(s, gate)
        print(f"Query:     {case['query']}")
        print(f"Passage:   {case['retrieved_passage']}")
        print(f"Score:     {s:.2f}")
        print(f"Decision:  {pred.decision.value} (Prediction set: {pred.prediction_set})")
        print(f"Action:    ", end="")
        if pred.decision == Decision.ACCEPT:
            print("Generate answer using retrieved evidence (statistically certified).")
        elif pred.decision == Decision.REJECT:
            print("Refuse answer: evidence is statistically certified insufficient.")
        else:  # ABSTAIN
            print("Abstain or escalate retrieval: score is inconclusive/ambiguous.")
        print("-" * 60)


if __name__ == "__main__":
    run_agent_workflow()

