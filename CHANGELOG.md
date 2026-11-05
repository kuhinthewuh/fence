# Changelog

All notable changes to **Fence** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-11-05

### Added
- **Initial Public Release of Fence** (`fence-evidence-mcp`).
- **Core Statistical Engine**:
  - Probability score auditing: Brier score, Expected Calibration Error (ECE), AUROC with exact Wilcoxon-Mann-Whitney tie handling.
  - Reliability binning and calibration diagnostics.
  - Class prevalence and classification metrics (precision, recall, F1) at configurable thresholds.
- **Class-Conditional Split Conformal Prediction**:
  - Separate nonconformity quantiles $q_0$ and $q_1$ for insufficient and sufficient classes.
  - Finite-sample quantile computation: $k_y = \lceil (n_y + 1)(1 - \alpha) \rceil$.
  - Conservative infinite-quantile fallback when calibration sample size is insufficient.
  - Strict prediction set decision rule: `ACCEPT` (set $\{1\}$), `REJECT` (set $\{0\}$), `ABSTAIN` (sets $\{0, 1\}$ or $\emptyset$).
- **Holdout Evaluation**:
  - Independent holdout evaluation of class-conditional coverage, false-sufficiency rates, acceptance, and abstention under strict holdout isolation.
- **Probability Calibration**:
  - Non-parametric isotonic regression via Pool Adjacent Violators Algorithm (PAVA).
  - Parametric logistic calibration (Platt scaling).
- **Lexical Trap Inspection**:
  - Detection and reporting of candidate false-sufficiency cases caused by superficial lexical overlap.
- **Model Context Protocol (MCP) Server**:
  - Built on official MCP Python SDK v2 (`mcp>=2.3.0,<3`) with stdio transport.
  - Tools: `audit_scores`, `fit_conformal_gate`, `apply_conformal_gate`, `evaluate_conformal_gate`, `find_lexical_traps`, `run_demo`.
  - Resource: `fence://methodology` (statistical foundations and operational guidance).
  - Prompt: `audit_rag_pipeline` (interactive agent auditing workflow).
  - CLI executable entry point: `fence-evidence-mcp`.
- **Reproducible Synthetic Benchmark**:
  - Deterministic evaluation comparing class-conditional conformal prediction against naive fixed thresholds on realistic RAG distributions.
- **Comprehensive Test Suite**:
  - Complete pytest coverage spanning metrics, conformal logic, edge cases, calibration, and MCP tool endpoints.

