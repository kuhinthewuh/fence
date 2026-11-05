"""Statistical auditing metrics for retrieval and confidence scores.

Fence: Evidence deserves a boundary.
Deterministic evaluation of calibration, discrimination, and error rates.
"""

from __future__ import annotations

import math
from typing import Sequence

from fence_evidence.core import (
    AuditResult,
    ReliabilityBin,
    validate_scores_and_labels,
)


def compute_brier_score(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Calculate the Brier score (mean squared error of probability forecasts).

    Parameters
    ----------
    scores:
        Predicted sufficiency probabilities in [0, 1].
    labels:
        Binary ground-truth labels in {0, 1}.

    Returns
    -------
    float in [0, 1]. Lower is better (0 = perfect calibration and discrimination).
    """
    clean_scores, clean_labels = validate_scores_and_labels(scores, labels)
    n = len(clean_scores)
    total_sq_err = sum((s - y) ** 2 for s, y in zip(clean_scores, clean_labels))
    return float(total_sq_err / n)


def compute_reliability_bins(
    scores: Sequence[float],
    labels: Sequence[int],
    num_bins: int = 10,
) -> tuple[float, list[ReliabilityBin]]:
    """Partition predictions into bins and compute Expected Calibration Error (ECE).

    Parameters
    ----------
    scores:
        Predicted probabilities in [0, 1].
    labels:
        Binary ground-truth labels in {0, 1}.
    num_bins:
        Number of equal-width bins across [0, 1] (default: 10).

    Returns
    -------
    tuple of (ece, list of ReliabilityBin)
    """
    clean_scores, clean_labels = validate_scores_and_labels(scores, labels)
    if num_bins < 2:
        raise ValueError("num_bins must be at least 2")

    n = len(clean_scores)
    bin_width = 1.0 / num_bins

    # Bucket observations into bins
    bin_counts = [0] * num_bins
    bin_score_sums = [0.0] * num_bins
    bin_label_sums = [0] * num_bins

    for s, y in zip(clean_scores, clean_labels):
        if s >= 1.0:
            b_idx = num_bins - 1
        else:
            b_idx = int(s / bin_width)
            if b_idx >= num_bins:
                b_idx = num_bins - 1
        bin_counts[b_idx] += 1
        bin_score_sums[b_idx] += s
        bin_label_sums[b_idx] += y

    bins: list[ReliabilityBin] = []
    total_weighted_error = 0.0

    for i in range(num_bins):
        lower = i * bin_width
        upper = (i + 1) * bin_width if i < num_bins - 1 else 1.0
        count = bin_counts[i]
        if count > 0:
            mean_conf = bin_score_sums[i] / count
            emp_acc = bin_label_sums[i] / count
            cal_err = abs(emp_acc - mean_conf)
            total_weighted_error += (count / n) * cal_err
        else:
            mean_conf = (lower + upper) / 2.0
            emp_acc = 0.0
            cal_err = 0.0

        bins.append(
            ReliabilityBin(
                bin_index=i,
                lower_bound=round(lower, 4),
                upper_bound=round(upper, 4),
                count=count,
                mean_confidence=round(mean_conf, 4),
                empirical_accuracy=round(emp_acc, 4),
                calibration_error=round(cal_err, 4),
            )
        )

    ece = float(total_weighted_error)
    return ece, bins


def compute_auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Compute the Area Under the Receiver Operating Characteristic (AUROC).

    Uses the Wilcoxon-Mann-Whitney U statistic with exact fractional rank
    assignment for tied scores.

    Parameters
    ----------
    scores:
        Predicted confidence or retrieval scores in [0, 1].
    labels:
        Binary ground-truth labels in {0, 1}.

    Returns
    -------
    AUROC in [0, 1]. 0.5 represents chance, 1.0 represents perfect discrimination.
    """
    clean_scores, clean_labels = validate_scores_and_labels(scores, labels)
    n = len(clean_scores)
    n1 = sum(clean_labels)
    n0 = n - n1

    if n0 == 0 or n1 == 0:
        return 0.5  # Undefined when only one class is present; default to chance

    # Pair scores with their indices and labels
    # Sort primarily by score ascending
    indexed_data = sorted(
        [(clean_scores[i], clean_labels[i]) for i in range(n)],
        key=lambda item: item[0],
    )

    # Assign ranks with tie averaging (1-indexed ranks)
    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and indexed_data[j][0] == indexed_data[i][0]:
            j += 1
        # Indices i to j-1 are tied. Their 1-indexed ranks range from i+1 to j.
        # Average rank: ((i+1) + j) / 2.0
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    # Sum of ranks for the positive class (label == 1)
    sum_ranks_positive = sum(
        ranks[k] for k in range(n) if indexed_data[k][1] == 1
    )

    # Mann-Whitney U statistic for class 1
    u1 = sum_ranks_positive - (n1 * (n1 + 1)) / 2.0
    auroc = u1 / (n0 * n1)
    return float(max(0.0, min(1.0, auroc)))


def compute_confusion_and_rates(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float = 0.5,
) -> tuple[int, int, int, int, float, float, float]:
    """Calculate confusion matrix (TP, FP, TN, FN) and precision, recall, F1 at threshold."""
    clean_scores, clean_labels = validate_scores_and_labels(scores, labels)

    tp = 0
    fp = 0
    tn = 0
    fn = 0

    for s, y in zip(clean_scores, clean_labels):
        pred_pos = (s >= threshold)
        if pred_pos and y == 1:
            tp += 1
        elif pred_pos and y == 0:
            fp += 1
        elif not pred_pos and y == 0:
            tn += 1
        else:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return tp, fp, tn, fn, precision, recall, f1


def audit_scores(
    scores: Sequence[float],
    labels: Sequence[int],
    threshold: float = 0.5,
    num_bins: int = 10,
) -> AuditResult:
    """Execute complete statistical audit on retrieval / confidence scores.

    Calculates:
    - Sample counts and class prevalence
    - Brier score
    - Expected Calibration Error (ECE)
    - AUROC with exact handling of ties
    - Reliability bins
    - Precision, recall, and F1 at specified decision threshold
    - Diagnostics and warnings for small sample sizes or pathological distributions

    Parameters
    ----------
    scores:
        Predicted confidence or retrieval scores in [0, 1].
    labels:
        Binary ground-truth labels in {0, 1}.
    threshold:
        Decision cutoff threshold for classification metrics (default: 0.5).
    num_bins:
        Number of reliability bins for ECE (default: 10).

    Returns
    -------
    AuditResult with all diagnostic fields and warnings.
    """
    clean_scores, clean_labels = validate_scores_and_labels(scores, labels)
    n = len(clean_scores)
    n1 = sum(clean_labels)
    n0 = n - n1
    prevalence = n1 / n if n > 0 else 0.0

    warnings: list[str] = []
    if n < 30:
        warnings.append(
            f"Small sample warning: N={n} < 30. Calibration diagnostics and quantile estimation "
            f"carry high finite-sample variance."
        )
    if n0 == 0:
        warnings.append("Insufficient class 0 is absent in the sample; AUROC cannot be evaluated.")
    elif n0 < 10:
        warnings.append(f"Low negative sample count: n0={n0} < 10. False-sufficiency assessment is uncertain.")

    if n1 == 0:
        warnings.append("Sufficient class 1 is absent in the sample; AUROC cannot be evaluated.")
    elif n1 < 10:
        warnings.append(f"Low positive sample count: n1={n1} < 10. True-sufficiency assessment is uncertain.")

    brier = compute_brier_score(clean_scores, clean_labels)
    if brier > 0.25:
        warnings.append(
            f"High Brier score ({brier:.4f} > 0.25). Prediction accuracy is worse than an uninformative "
            f"constant 0.5 forecast."
        )

    ece, bins = compute_reliability_bins(clean_scores, clean_labels, num_bins=num_bins)
    if ece > 0.15:
        warnings.append(
            f"High Expected Calibration Error (ECE={ece:.4f} > 0.15). Raw scores cannot be trusted "
            f"as empirical probabilities without recalibration."
        )

    auroc = compute_auroc(clean_scores, clean_labels)
    if n0 > 0 and n1 > 0 and auroc < 0.60:
        warnings.append(
            f"Low AUROC ({auroc:.4f} < 0.60). Scores have poor discriminative separation between "
            f"sufficient and insufficient evidence."
        )

    tp, fp, tn, fn, precision, recall, f1 = compute_confusion_and_rates(
        clean_scores, clean_labels, threshold=threshold
    )

    return AuditResult(
        sample_count=n,
        positive_count=n1,
        negative_count=n0,
        prevalence=round(prevalence, 4),
        brier_score=round(brier, 4),
        expected_calibration_error=round(ece, 4),
        auroc=round(auroc, 4),
        threshold=round(threshold, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1_score=round(f1, 4),
        true_positives=tp,
        false_positives=fp,
        true_negatives=tn,
        false_negatives=fn,
        reliability_bins=bins,
        warnings=warnings,
    )

