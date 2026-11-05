"""Walkthrough: Lexical Traps and Conservative Abstention with Fence.

Fence: Evidence deserves a boundary.
Shows side-by-side how naive thresholds fall into lexical traps,
while Fence's conformal boundary safely enforces conservative abstention.
"""

from __future__ import annotations

from fence_evidence import Decision, fit_conformal_gate, predict_one


def main() -> None:
    print("=" * 75)
    print("  FENCE WALKTHROUGH: The Lexical Trap Problem & Conservative Abstention")
    print("=" * 75)

    # 1. Representative calibration set from historical retrieval evaluations
    # Contains:
    # - True positives (scores 0.85 - 0.98, label 1)
    # - True negatives (scores 0.05 - 0.35, label 0)
    # - Historical lexical traps (scores 0.75 - 0.88, label 0)
    cal_scores = [
        # Sufficient evidence (y = 1)
        0.95, 0.91, 0.88, 0.86, 0.93, 0.89, 0.96, 0.84, 0.90, 0.87,
        # Insufficient evidence (y = 0)
        0.12, 0.22, 0.18, 0.31, 0.15, 0.28, 0.19, 0.25, 0.11, 0.34,
        # Historical lexical traps (y = 0)
        0.82, 0.79, 0.85, 0.81, 0.86,
    ]
    cal_labels = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
    ]

    # Fit conformal gate with alpha = 0.10 (90% target class coverage)
    alpha = 0.10
    gate = fit_conformal_gate(cal_scores, cal_labels, alpha=alpha)

    print(f"\n[Fitted Conformal Boundary (alpha={alpha:.2f})]")
    print(f"  * Insufficient Nonconformity Bound (q0): {gate.q0:.4f}")
    print(f"  * Sufficient Nonconformity Bound (q1):   {gate.q1:.4f}")
    print(f"  * Rule: An evidence score s is ACCEPTED iff s > {gate.q0:.4f} and s >= {1.0 - gate.q1:.4f}")
    print("-" * 75)

    # 2. Production query and candidate retrieved passages
    query = "What is the recommended pediatric dosage for Amoxicillin in acute otitis media?"

    cases = [
        {
            "name": "Case A: Legitimate Entailing Evidence",
            "passage": (
                "For pediatric acute otitis media, the recommended dose of amoxicillin is "
                "80 to 90 mg/kg/day divided into two doses."
            ),
            "score": 0.95,
            "ground_truth": "Sufficient (1)",
        },
        {
            "name": "Case B: The Lexical Trap (High similarity, wrong population)",
            "passage": (
                "For adult acute exacerbations of chronic bronchitis, the recommended dose "
                "of amoxicillin is 500 mg every 8 hours or 875 mg every 12 hours."
            ),
            "score": 0.84,  # High keyword overlap ("recommended dose of amoxicillin"), but adult bronchitis!
            "ground_truth": "Insufficient (0)",
        },
        {
            "name": "Case C: Borderline / Partial Evidence",
            "passage": (
                "Amoxicillin remains the first-line antimicrobial for acute otitis media in "
                "infants and children unless penicillin allergy is documented."
            ),
            "score": 0.72,
            "ground_truth": "Ambiguous (No dosage specified)",
        },
        {
            "name": "Case D: Irrelevant Passage",
            "passage": "Storage instructions: Reconstituted amoxicillin oral suspension should be kept refrigerated.",
            "score": 0.28,
            "ground_truth": "Insufficient (0)",
        },
    ]

    naive_threshold = 0.75

    for case in cases:
        score = case["score"]
        naive_accepted = score >= naive_threshold
        pred = predict_one(score, gate)

        print(f"\n{case['name']}")
        print(f"  Query:        {query}")
        print(f"  Passage:      {case['passage']}")
        print(f"  Retriever:    Score = {score:.2f} (Ground Truth: {case['ground_truth']})")

        # Compare decisions
        print("  Comparison:")
        if naive_accepted:
            naive_str = "ACCEPT (Answer query)"
            if "0" in case["ground_truth"] or "Ambiguous" in case["ground_truth"]:
                naive_str += "  <-- CATASTROPHIC HALLUCINATION ERROR!"
        else:
            naive_str = "REJECT (Refuse)"

        print(f"    * Naive Fixed Cutoff (>={naive_threshold}): {naive_str}")
        print(f"    * Fence Conformal Decision:     {pred.decision.value} (Set: {pred.prediction_set})")
        print(f"    * Safe Agent Action:            ", end="")

        if pred.decision == Decision.ACCEPT:
            print("Answer safely with certified evidence.")
        elif pred.decision == Decision.REJECT:
            print("Refuse: evidence is statistically rejected as insufficient.")
        else:
            print("ABSTAIN: Score falls within ambiguous zone; trigger retrieval escalation or ask user.")

    print("\n" + "=" * 75)
    print("  CONCLUSION: Fence safely abstains on Case B & C, preventing hallucinations,")
    print("  whereas a naive cutoff accepted misleading evidence with high confidence.")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
