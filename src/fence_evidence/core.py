"""Core data structures, enums, and validation functions for Fence.

Fence: Evidence deserves a boundary.
Statistical evidence-sufficiency verification and conservative abstention.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Sequence


class Decision(str, Enum):
    """Decision produced by the conformal evidence boundary."""

    ACCEPT = "ACCEPT"  # Prediction set is {1}: statistically certified sufficient
    REJECT = "REJECT"  # Prediction set is {0}: statistically certified insufficient
    ABSTAIN = "ABSTAIN"  # Prediction set is {0, 1} or empty: ambiguous or inconclusive


@dataclass(frozen=True)
class ReliabilityBin:
    """Bin summary for probability calibration diagnostics."""

    bin_index: int
    lower_bound: float
    upper_bound: float
    count: int
    mean_confidence: float
    empirical_accuracy: float
    calibration_error: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AuditResult:
    """Complete score auditing and reliability diagnostic report."""

    sample_count: int
    positive_count: int
    negative_count: int
    prevalence: float
    brier_score: float
    expected_calibration_error: float
    auroc: float
    threshold: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    reliability_bins: list[ReliabilityBin]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["reliability_bins"] = [b.to_dict() if hasattr(b, "to_dict") else b for b in self.reliability_bins]
        return d


@dataclass(frozen=True)
class ConformalGate:
    """Fitted class-conditional split conformal prediction gate."""

    alpha: float
    q0: float  # Nonconformity quantile for class 0 (insufficient)
    q1: float  # Nonconformity quantile for class 1 (sufficient)
    n0: int    # Number of calibration examples for class 0
    n1: int    # Number of calibration examples for class 1
    rank0: int
    rank1: int
    degenerate0: bool = False
    degenerate1: bool = False
    guarantee_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GatePrediction:
    """Individual conformal prediction decision for an evidence score."""

    score: float
    prediction_set: list[int]
    decision: Decision
    is_singleton: bool
    explanation: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "prediction_set": self.prediction_set,
            "decision": self.decision.value,
            "is_singleton": self.is_singleton,
            "explanation": self.explanation,
        }


@dataclass(frozen=True)
class EvaluationResult:
    """Holdout evaluation report for a fitted conformal gate."""

    eval_count: int
    n0: int
    n1: int
    alpha: float
    target_coverage: float
    marginal_coverage: float
    class0_coverage: float
    class1_coverage: float
    false_sufficiency_rate: float
    acceptance_rate: float
    rejection_rate: float
    abstention_rate: float
    accepted_count: int
    rejected_count: int
    abstained_count: int
    conformal_bounds_held: bool
    breakdown: dict[str, int]
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LexicalTrapItem:
    """Identified candidate false-sufficiency example / lexical trap."""

    item_id: str | int
    score: float
    label: int
    query: str | None = None
    passage: str | None = None
    note: str = "High retrieval score despite insufficient ground-truth evidence."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LexicalTrapReport:
    """Report detailing candidate lexical traps and false-sufficiency risks."""

    total_examined: int
    threshold: float
    trap_count: int
    insufficient_count: int
    trap_rate_among_insufficient: float
    traps: list[LexicalTrapItem]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_examined": self.total_examined,
            "threshold": self.threshold,
            "trap_count": self.trap_count,
            "insufficient_count": self.insufficient_count,
            "trap_rate_among_insufficient": self.trap_rate_among_insufficient,
            "traps": [t.to_dict() for t in self.traps],
        }


# ============================================================================
# Validation Functions
# ============================================================================

def validate_scores(scores: Sequence[float]) -> list[float]:
    """Validate a sequence of confidence or retrieval scores.

    Must be non-empty, finite floats in [0.0, 1.0].
    """
    if not isinstance(scores, (list, tuple)):
        try:
            scores = list(scores)
        except Exception as err:
            raise TypeError(f"scores must be an iterable of numbers, got {type(scores).__name__}") from err

    if len(scores) == 0:
        raise ValueError("scores sequence cannot be empty")

    clean_scores: list[float] = []
    for idx, s in enumerate(scores):
        if s is None:
            raise ValueError(f"scores[{idx}] is None; expected float in [0.0, 1.0]")
        try:
            val = float(s)
        except (ValueError, TypeError) as err:
            raise ValueError(f"scores[{idx}] = {s!r} cannot be converted to float") from err

        if math.isnan(val) or math.isinf(val):
            raise ValueError(f"scores[{idx}] = {val} is not finite (NaN or Infinity)")

        if not (0.0 <= val <= 1.0):
            raise ValueError(f"scores[{idx}] = {val} is out of bounds; expected [0.0, 1.0]")

        clean_scores.append(val)

    return clean_scores


def validate_labels(labels: Sequence[int]) -> list[int]:
    """Validate a sequence of ground-truth sufficiency labels.

    Must be non-empty integers in {0, 1}.
    """
    if not isinstance(labels, (list, tuple)):
        try:
            labels = list(labels)
        except Exception as err:
            raise TypeError(f"labels must be an iterable of integers, got {type(labels).__name__}") from err

    if len(labels) == 0:
        raise ValueError("labels sequence cannot be empty")

    clean_labels: list[int] = []
    for idx, y in enumerate(labels):
        if y is None:
            raise ValueError(f"labels[{idx}] is None; expected 0 or 1")
        try:
            val = int(y)
        except (ValueError, TypeError) as err:
            raise ValueError(f"labels[{idx}] = {y!r} cannot be converted to int") from err

        if val not in (0, 1):
            raise ValueError(f"labels[{idx}] = {val} is invalid; expected 0 (insufficient) or 1 (sufficient)")

        clean_labels.append(val)

    return clean_labels


def validate_scores_and_labels(
    scores: Sequence[float],
    labels: Sequence[int],
    require_both_classes: bool = False,
) -> tuple[list[float], list[int]]:
    """Validate aligned sequences of scores and binary ground truth labels.

    Parameters
    ----------
    scores:
        Sequence of floats in [0, 1].
    labels:
        Sequence of integers in {0, 1}.
    require_both_classes:
        If True, raises ValueError if either class 0 or class 1 is completely absent.

    Returns
    -------
    tuple of clean (scores, labels) as python lists.
    """
    clean_scores = validate_scores(scores)
    clean_labels = validate_labels(labels)

    if len(clean_scores) != len(clean_labels):
        raise ValueError(
            f"Length mismatch: received {len(clean_scores)} scores but {len(clean_labels)} labels"
        )

    n0 = sum(1 for y in clean_labels if y == 0)
    n1 = len(clean_labels) - n0

    if require_both_classes:
        if n0 == 0:
            raise ValueError("All calibration examples have label 1 (insufficient class 0 is missing)")
        if n1 == 0:
            raise ValueError("All calibration examples have label 0 (sufficient class 1 is missing)")

    return clean_scores, clean_labels


def validate_alpha(alpha: float) -> float:
    """Validate user significance level alpha in (0, 1)."""
    try:
        val = float(alpha)
    except (ValueError, TypeError) as err:
        raise ValueError(f"alpha must be float, got {alpha!r}") from err

    if math.isnan(val) or math.isinf(val):
        raise ValueError(f"alpha must be finite, got {val}")

    if not (0.0 < val < 1.0):
        raise ValueError(f"alpha must be strictly between 0 and 1 (exclusive), got {val}")

    return val

