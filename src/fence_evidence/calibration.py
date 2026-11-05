"""Probability calibration algorithms for retrieval and confidence scores.

Fence: Evidence deserves a boundary.
Calibrate uncalibrated similarity or retriever scores to empirical probabilities.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from fence_evidence.core import validate_scores, validate_scores_and_labels


@dataclass
class IsotonicCalibrator:
    """Non-parametric isotonic regression calibrator using the Pool Adjacent Violators Algorithm (PAVA).

    Produces a monotonic non-decreasing piecewise-constant mapping from raw scores to probabilities.
    """

    thresholds: list[float]
    values: list[float]
    is_fitted: bool = False

    @classmethod
    def fit(cls, scores: Sequence[float], labels: Sequence[int]) -> IsotonicCalibrator:
        clean_scores, clean_labels = validate_scores_and_labels(scores, labels)
        n = len(clean_scores)

        # Sort pairs by score ascending
        pairs = sorted(zip(clean_scores, clean_labels), key=lambda x: x[0])

        # Initialize blocks: (weight, sum_labels, score_upper)
        # Block representation: weight w_i = 1, value = y_i, score = s_i
        blocks: list[list[float]] = []  # [weight, sum_y, score_min, score_max]
        for s, y in pairs:
            blocks.append([1.0, float(y), s, s])

        # Pool adjacent violators (PAVA)
        i = 0
        while i < len(blocks) - 1:
            mean_curr = blocks[i][1] / blocks[i][0]
            mean_next = blocks[i + 1][1] / blocks[i + 1][0]
            if mean_curr > mean_next:
                # Merge block i and block i+1
                merged = [
                    blocks[i][0] + blocks[i + 1][0],
                    blocks[i][1] + blocks[i + 1][1],
                    blocks[i][2],
                    blocks[i + 1][3],
                ]
                blocks[i] = merged
                del blocks[i + 1]
                if i > 0:
                    i -= 1  # check if previous block now violates monotonicity
            else:
                i += 1

        thresholds = [b[3] for b in blocks]
        values = [min(1.0, max(0.0, b[1] / b[0])) for b in blocks]

        return cls(thresholds=thresholds, values=values, is_fitted=True)

    def predict_one(self, score: float) -> float:
        if not self.is_fitted or not self.thresholds:
            return score
        # Linear or step search
        for t, v in zip(self.thresholds, self.values):
            if score <= t:
                return v
        return self.values[-1]

    def predict(self, scores: Sequence[float]) -> list[float]:
        clean_scores = validate_scores(scores)
        return [self.predict_one(s) for s in clean_scores]


@dataclass
class PlattCalibrator:
    """Parametric logistic calibration (Platt scaling).

    Fits a univariate sigmoid mapping: P(Y=1|s) = 1 / (1 + exp(-(a * s + b)))
    via maximum likelihood estimation.
    """

    a: float = 1.0
    b: float = 0.0
    is_fitted: bool = False

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        labels: Sequence[int],
        max_iter: int = 100,
        tol: float = 1e-6,
    ) -> PlattCalibrator:
        clean_scores, clean_labels = validate_scores_and_labels(scores, labels, require_both_classes=True)
        n = len(clean_scores)

        # Standard Newton-Raphson or gradient descent for logistic regression
        # Target smoothing to prevent overfitting (Platt 1999)
        n1 = sum(clean_labels)
        n0 = n - n1
        t1 = (n1 + 1.0) / (n1 + 2.0)
        t0 = 1.0 / (n0 + 2.0)
        targets = [t1 if y == 1 else t0 for y in clean_labels]

        a = 1.0
        b = 0.0

        for _ in range(max_iter):
            # Compute gradient and Hessian
            grad_a = 0.0
            grad_b = 0.0
            h_aa = 0.0
            h_ab = 0.0
            h_bb = 0.0

            for s, t in zip(clean_scores, targets):
                z = a * s + b
                # Clamped sigmoid
                if z >= 35.0:
                    p = 1.0
                elif z <= -35.0:
                    p = 0.0
                else:
                    p = 1.0 / (1.0 + math.exp(-z))

                diff = p - t
                weight = p * (1.0 - p)
                if weight < 1e-12:
                    weight = 1e-12

                grad_a += diff * s
                grad_b += diff
                h_aa += weight * s * s
                h_ab += weight * s
                h_bb += weight

            # Regularization
            h_aa += 1e-4
            h_bb += 1e-4

            det = h_aa * h_bb - h_ab * h_ab
            if abs(det) < 1e-12:
                break

            delta_a = (h_bb * grad_a - h_ab * grad_b) / det
            delta_b = (-h_ab * grad_a + h_aa * grad_b) / det

            a -= delta_a
            b -= delta_b

            if abs(delta_a) < tol and abs(delta_b) < tol:
                break

        return cls(a=float(a), b=float(b), is_fitted=True)

    def predict_one(self, score: float) -> float:
        if not self.is_fitted:
            return score
        z = self.a * score + self.b
        if z >= 35.0:
            return 1.0
        if z <= -35.0:
            return 0.0
        return 1.0 / (1.0 + math.exp(-z))

    def predict(self, scores: Sequence[float]) -> list[float]:
        clean_scores = validate_scores(scores)
        return [self.predict_one(s) for s in clean_scores]

