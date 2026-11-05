"""Fence: Evidence deserves a boundary.

An open-source statistical evidence-sufficiency MCP server that helps AI agents
determine whether retrieved evidence genuinely supports an answer using probability
calibration, class-conditional split conformal prediction, and conservative abstention.
"""

from __future__ import annotations

from fence_evidence.calibration import IsotonicCalibrator, PlattCalibrator
from fence_evidence.conformal import (
    apply_conformal_gate,
    evaluate_conformal_gate,
    find_lexical_traps,
    fit_conformal_gate,
    predict_one,
)
from fence_evidence.core import (
    AuditResult,
    ConformalGate,
    Decision,
    EvaluationResult,
    GatePrediction,
    LexicalTrapItem,
    LexicalTrapReport,
    ReliabilityBin,
)
from fence_evidence.demo import run_demo
from fence_evidence.metrics import (
    audit_scores,
    compute_auroc,
    compute_brier_score,
    compute_reliability_bins,
)

__version__ = "0.1.0"
__all__ = [
    "__version__",
    "AuditResult",
    "ConformalGate",
    "Decision",
    "EvaluationResult",
    "GatePrediction",
    "IsotonicCalibrator",
    "LexicalTrapItem",
    "LexicalTrapReport",
    "PlattCalibrator",
    "ReliabilityBin",
    "apply_conformal_gate",
    "audit_scores",
    "compute_auroc",
    "compute_brier_score",
    "compute_reliability_bins",
    "evaluate_conformal_gate",
    "find_lexical_traps",
    "fit_conformal_gate",
    "predict_one",
    "run_demo",
]

