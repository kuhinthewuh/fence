"""Class-conditional split conformal prediction engine for evidence sufficiency.

Fence: Evidence deserves a boundary.
Finite-sample statistical coverage, conservative abstention, and holdout evaluation.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from fence_evidence.core import (
    ConformalGate,
    Decision,
    EvaluationResult,
    GatePrediction,
    LexicalTrapItem,
    LexicalTrapReport,
    validate_alpha,
    validate_scores,
    validate_scores_and_labels,
)


def fit_conformal_gate(
    calibration_scores: Sequence[float],
    calibration_labels: Sequence[int],
    alpha: float = 0.10,
) -> ConformalGate:
    """Fit a class-conditional split conformal gate from calibration data.

    Under the exchangeability assumption, the class-conditional split conformal
    construction guarantees that for each class y in {0, 1}:
        P(y in Gamma^alpha(S) | Y = y) >= 1 - alpha

    Nonconformity scores:
        - Class 0 (insufficient): a0(s) = s
          (Higher scores are less conforming to insufficiency)
        - Class 1 (sufficient):   a1(s) = 1 - s
          (Lower scores are less conforming to sufficiency)

    Finite-sample quantile:
        k_y = ceil((n_y + 1) * (1 - alpha))
        If k_y > n_y, sample size is insufficient to guarantee coverage;
        q_y is set to infinity to prevent spurious exclusion.

    Parameters
    ----------
    calibration_scores:
        Calibration set scores in [0, 1].
    calibration_labels:
        Calibration set binary labels in {0, 1}.
    alpha:
        Significance level / allowable miscoverage rate (default: 0.10, for 90% coverage).

    Returns
    -------
    ConformalGate containing fitted thresholds, ranks, and sample counts.
    """
    clean_scores, clean_labels = validate_scores_and_labels(
        calibration_scores, calibration_labels, require_both_classes=False
    )
    val_alpha = validate_alpha(alpha)

    # Split nonconformity scores by class
    a0_list: list[float] = []
    a1_list: list[float] = []

    for s, y in zip(clean_scores, clean_labels):
        if y == 0:
            a0_list.append(s)  # a0(s) = s
        else:
            a1_list.append(1.0 - s)  # a1(s) = 1 - s

    n0 = len(a0_list)
    n1 = len(a1_list)

    # Class 0 quantile calculation
    k0 = math.ceil((n0 + 1) * (1.0 - val_alpha))
    if n0 == 0 or k0 > n0:
        q0 = float("inf")
        degenerate0 = True
    else:
        a0_list.sort()
        q0 = a0_list[k0 - 1]
        degenerate0 = False

    # Class 1 quantile calculation
    k1 = math.ceil((n1 + 1) * (1.0 - val_alpha))
    if n1 == 0 or k1 > n1:
        q1 = float("inf")
        degenerate1 = True
    else:
        a1_list.sort()
        q1 = a1_list[k1 - 1]
        degenerate1 = False

    guarantee_text = (
        f"Class-conditional coverage guarantee: Under exchangeability between calibration and future "
        f"observations, P(Y in Gamma | Y=y) >= {1.0 - val_alpha:.1%} for each class y in {{0, 1}}. "
        f"Requires exchangeable retrieval score distributions. Small sample sizes or distribution "
        f"shifts invalidate finite-sample bounds."
    )

    return ConformalGate(
        alpha=val_alpha,
        q0=round(q0, 6) if not math.isinf(q0) else float("inf"),
        q1=round(q1, 6) if not math.isinf(q1) else float("inf"),
        n0=n0,
        n1=n1,
        rank0=k0,
        rank1=k1,
        degenerate0=degenerate0,
        degenerate1=degenerate1,
        guarantee_text=guarantee_text,
    )


def predict_one(score: float, gate: ConformalGate) -> GatePrediction:
    """Evaluate a single evidence score through the conformal gate."""
    # Check membership for class 0: a0(s) <= q0  <=> s <= q0
    in_class0 = score <= gate.q0

    # Check membership for class 1: a1(s) <= q1  <=> 1 - s <= q1
    in_class1 = (1.0 - score) <= gate.q1

    prediction_set: list[int] = []
    if in_class0:
        prediction_set.append(0)
    if in_class1:
        prediction_set.append(1)

    # Conservative decision rule: ONLY accept when prediction set is strictly {1}
    if prediction_set == [1]:
        decision = Decision.ACCEPT
        is_singleton = True
        explanation = (
            f"Evidence statistically accepted: class 0 (insufficient) excluded at alpha={gate.alpha:.2f}; "
            f"score={score:.4f} meets sufficiency boundary (1-s <= {gate.q1:.4f} and s > {gate.q0:.4f})."
        )
    elif prediction_set == [0]:
        decision = Decision.REJECT
        is_singleton = True
        explanation = (
            f"Evidence statistically rejected: class 1 (sufficient) excluded at alpha={gate.alpha:.2f}; "
            f"score={score:.4f} fails sufficiency boundary (s <= {gate.q0:.4f} and 1-s > {gate.q1:.4f})."
        )
    elif prediction_set == [0, 1]:
        decision = Decision.ABSTAIN
        is_singleton = False
        explanation = (
            f"Conservative abstention: evidence is ambiguous/inconclusive. Both insufficient (0) and "
            f"sufficient (1) hypotheses conform to calibration data at alpha={gate.alpha:.2f} (score={score:.4f})."
        )
    else:  # Empty set []
        decision = Decision.ABSTAIN
        is_singleton = False
        explanation = (
            f"Conservative abstention: prediction set is empty. Nonconformity exceeds thresholds for both "
            f"classes (atypical score={score:.4f} at alpha={gate.alpha:.2f})."
        )

    return GatePrediction(
        score=score,
        prediction_set=prediction_set,
        decision=decision,
        is_singleton=is_singleton,
        explanation=explanation,
    )


def apply_conformal_gate(
    scores: Sequence[float],
    gate: ConformalGate,
) -> list[GatePrediction]:
    """Apply a fitted conformal gate to a sequence of evidence scores."""
    clean_scores = validate_scores(scores)
    return [predict_one(s, gate) for s in clean_scores]


def evaluate_conformal_gate(
    eval_scores: Sequence[float],
    eval_labels: Sequence[int],
    gate: ConformalGate,
) -> EvaluationResult:
    """Evaluate a fitted conformal gate on an independent holdout evaluation set.

    Strict holdout isolation: No gate parameters or thresholds are fitted or tuned
    on the evaluation observations.

    Parameters
    ----------
    eval_scores:
        Independent holdout evaluation scores.
    eval_labels:
        Independent holdout binary ground-truth labels.
    gate:
        Fitted ConformalGate.

    Returns
    -------
    EvaluationResult reporting class-conditional coverage, false-sufficiency rate,
    abstention rate, and confusion breakdown.
    """
    clean_scores, clean_labels = validate_scores_and_labels(
        eval_scores, eval_labels, require_both_classes=False
    )
    n = len(clean_scores)

    eval_n0 = 0
    eval_n1 = 0
    cov0_hits = 0
    cov1_hits = 0
    marginal_hits = 0
    false_sufficiency_count = 0  # y = 0, but decision is ACCEPT ({1})

    accepted_count = 0
    rejected_count = 0
    abstained_count = 0

    breakdown = {
        "set_[0]": 0,
        "set_[1]": 0,
        "set_[0,1]": 0,
        "set_empty": 0,
    }

    for s, y in zip(clean_scores, clean_labels):
        pred = predict_one(s, gate)
        p_set = pred.prediction_set

        if p_set == [0]:
            breakdown["set_[0]"] += 1
            rejected_count += 1
        elif p_set == [1]:
            breakdown["set_[1]"] += 1
            accepted_count += 1
        elif p_set == [0, 1]:
            breakdown["set_[0,1]"] += 1
            abstained_count += 1
        else:
            breakdown["set_empty"] += 1
            abstained_count += 1

        # Class coverage
        if y == 0:
            eval_n0 += 1
            if 0 in p_set:
                cov0_hits += 1
            if pred.decision == Decision.ACCEPT:
                false_sufficiency_count += 1
        else:
            eval_n1 += 1
            if 1 in p_set:
                cov1_hits += 1

        # Marginal coverage
        if y in p_set:
            marginal_hits += 1

    class0_coverage = cov0_hits / eval_n0 if eval_n0 > 0 else 1.0
    class1_coverage = cov1_hits / eval_n1 if eval_n1 > 0 else 1.0
    marginal_coverage = marginal_hits / n if n > 0 else 1.0
    false_sufficiency_rate = false_sufficiency_count / eval_n0 if eval_n0 > 0 else 0.0

    target_coverage = 1.0 - gate.alpha
    # Finite-sample binomial margin of error (95% confidence interval for empirical holdout coverage)
    tol0 = 1.96 * math.sqrt(target_coverage * (1.0 - target_coverage) / max(1, eval_n0)) if eval_n0 > 0 else 0.0
    tol1 = 1.96 * math.sqrt(target_coverage * (1.0 - target_coverage) / max(1, eval_n1)) if eval_n1 > 0 else 0.0
    tol_m = 1.96 * math.sqrt(target_coverage * (1.0 - target_coverage) / max(1, n)) if n > 0 else 0.0

    bounds_held = (
        (marginal_coverage >= (target_coverage - tol_m))
        and (class0_coverage >= (target_coverage - tol0))
        and (class1_coverage >= (target_coverage - tol1))
    )

    warnings: list[str] = []
    if eval_n0 == 0:
        warnings.append("Holdout set lacks class 0 examples; false-sufficiency rate cannot be evaluated.")
    if eval_n1 == 0:
        warnings.append("Holdout set lacks class 1 examples; sufficiency coverage cannot be evaluated.")
    if gate.degenerate0 or gate.degenerate1:
        warnings.append("Gate was fitted on a degenerate calibration sample (rank exceeded n_class).")

    return EvaluationResult(
        eval_count=n,
        n0=eval_n0,
        n1=eval_n1,
        alpha=gate.alpha,
        target_coverage=round(target_coverage, 4),
        marginal_coverage=round(marginal_coverage, 4),
        class0_coverage=round(class0_coverage, 4),
        class1_coverage=round(class1_coverage, 4),
        false_sufficiency_rate=round(false_sufficiency_rate, 4),
        acceptance_rate=round(accepted_count / n, 4) if n > 0 else 0.0,
        rejection_rate=round(rejected_count / n, 4) if n > 0 else 0.0,
        abstention_rate=round(abstained_count / n, 4) if n > 0 else 0.0,
        accepted_count=accepted_count,
        rejected_count=rejected_count,
        abstained_count=abstained_count,
        conformal_bounds_held=bounds_held,
        breakdown=breakdown,
        warnings=warnings,
    )


def find_lexical_traps(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float = 0.70,
    item_ids: Sequence[Any] | None = None,
    queries: Sequence[str] | None = None,
    passages: Sequence[str] | None = None,
) -> LexicalTrapReport:
    """Identify candidate false-sufficiency examples (lexical traps).

    Finds examples where the retrieval score is high (>= threshold) but the actual
    ground-truth label indicates insufficient evidence (y == 0).

    In real-world retrieval, high score with insufficient evidence typically indicates
    high lexical overlap or surface keyword matching without semantic entailment.

    Parameters
    ----------
    scores:
        Retrieval scores in [0, 1].
    labels:
        Binary ground-truth labels in {0, 1}.
    threshold:
        Score threshold above which an uncalibrated system would trust the retrieval (default: 0.70).
    item_ids:
        Optional identifiers for each example.
    queries:
        Optional query strings.
    passages:
        Optional retrieved passage snippets.

    Returns
    -------
    LexicalTrapReport with list of identified trap cases.
    """
    clean_scores, clean_labels = validate_scores_and_labels(scores, labels)
    n = len(clean_scores)

    ids = list(item_ids) if item_ids is not None else [f"doc_{i}" for i in range(n)]
    q_list = list(queries) if queries is not None else [None] * n
    p_list = list(passages) if passages is not None else [None] * n

    traps: list[LexicalTrapItem] = []
    insufficient_count = 0

    for idx in range(n):
        s = clean_scores[idx]
        y = clean_labels[idx]
        if y == 0:
            insufficient_count += 1
            if s >= threshold:
                traps.append(
                    LexicalTrapItem(
                        item_id=ids[idx] if idx < len(ids) else f"doc_{idx}",
                        score=round(s, 4),
                        label=y,
                        query=q_list[idx] if idx < len(q_list) else None,
                        passage=p_list[idx] if idx < len(p_list) else None,
                        note=(
                            f"Candidate lexical trap: high retrieval score ({s:.4f} >= {threshold:.2f}) "
                            f"despite insufficient evidence (label=0)."
                        ),
                    )
                )

    trap_rate = len(traps) / insufficient_count if insufficient_count > 0 else 0.0

    return LexicalTrapReport(
        total_examined=n,
        threshold=round(threshold, 4),
        trap_count=len(traps),
        insufficient_count=insufficient_count,
        trap_rate_among_insufficient=round(trap_rate, 4),
        traps=traps,
    )
