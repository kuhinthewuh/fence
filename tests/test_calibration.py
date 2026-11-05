"""Tests for probability calibration in fence_evidence.calibration."""

from __future__ import annotations

import pytest

from fence_evidence.calibration import IsotonicCalibrator, PlattCalibrator
from fence_evidence.metrics import compute_brier_score


def test_isotonic_calibrator_monotonicity() -> None:
    # Uncalibrated noisy scores
    scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    labels = [0, 0, 1, 0, 1, 0, 1, 1, 1]

    calibrator = IsotonicCalibrator.fit(scores, labels)
    cal_scores = calibrator.predict(scores)

    # Monotonicity test: each predicted probability must be >= preceding
    for i in range(len(cal_scores) - 1):
        assert cal_scores[i] <= cal_scores[i + 1] + 1e-9


def test_isotonic_improves_brier_on_distorted_scores() -> None:
    # Severe score distortion: raw scores are heavily shifted high (e.g. cosine sim > 0.7 for label 0)
    scores = [0.70, 0.72, 0.75, 0.78, 0.85, 0.88, 0.92, 0.95]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]

    raw_brier = compute_brier_score(scores, labels)
    calibrator = IsotonicCalibrator.fit(scores, labels)
    cal_scores = calibrator.predict(scores)
    cal_brier = compute_brier_score(cal_scores, labels)

    assert cal_brier < raw_brier


def test_platt_calibrator_fit_and_predict() -> None:
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    labels = [0, 0, 0, 1, 1, 1]

    calibrator = PlattCalibrator.fit(scores, labels)
    preds = calibrator.predict(scores)

    assert len(preds) == len(scores)
    assert all(0.0 <= p <= 1.0 for p in preds)
    # Class 1 scores should produce higher predicted probability than Class 0 scores
    assert preds[0] < preds[-1]

