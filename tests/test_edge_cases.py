"""Tests for input validation safeguards and pathological edge cases."""

from __future__ import annotations

import math
import pytest

from fence_evidence.core import (
    validate_alpha,
    validate_labels,
    validate_scores,
    validate_scores_and_labels,
)
from fence_evidence.metrics import audit_scores


def test_empty_scores() -> None:
    with pytest.raises(ValueError, match="scores sequence cannot be empty"):
        validate_scores([])


def test_empty_labels() -> None:
    with pytest.raises(ValueError, match="labels sequence cannot be empty"):
        validate_labels([])


def test_length_mismatch() -> None:
    with pytest.raises(ValueError, match="Length mismatch"):
        validate_scores_and_labels([0.5, 0.6], [1])


def test_nan_scores() -> None:
    with pytest.raises(ValueError, match="not finite"):
        validate_scores([0.5, float("nan")])


def test_inf_scores() -> None:
    with pytest.raises(ValueError, match="not finite"):
        validate_scores([0.5, float("inf")])
    with pytest.raises(ValueError, match="not finite"):
        validate_scores([0.5, float("-inf")])


def test_out_of_bounds_scores() -> None:
    with pytest.raises(ValueError, match="out of bounds"):
        validate_scores([-0.01])
    with pytest.raises(ValueError, match="out of bounds"):
        validate_scores([1.01])


def test_invalid_label_values() -> None:
    with pytest.raises(ValueError, match="invalid; expected 0"):
        validate_labels([0, 2])
    with pytest.raises(ValueError, match="invalid; expected 0"):
        validate_labels([-1, 1])


def test_invalid_alpha_values() -> None:
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        validate_alpha(0.0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        validate_alpha(1.0)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        validate_alpha(-0.1)
    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        validate_alpha(1.5)
    with pytest.raises(ValueError, match="must be finite"):
        validate_alpha(float("nan"))


def test_single_class_audit_warning() -> None:
    scores = [0.2, 0.4, 0.6, 0.8]
    labels = [1, 1, 1, 1]
    res = audit_scores(scores, labels)
    assert any("absent in the sample" in w for w in res.warnings)


def test_small_sample_audit_warning() -> None:
    scores = [0.8, 0.2]
    labels = [1, 0]
    res = audit_scores(scores, labels)
    assert any("Small sample warning" in w for w in res.warnings)

