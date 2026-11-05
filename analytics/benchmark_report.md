# Fence Benchmark Report: Evidence Sufficiency Verification
**Date Generated:** 2026-11-05 19:31:53 UTC  
**Software Version:** Fence v0.1.0  
**Data Mode:** Controlled Synthetic Simulation (`seed=42`)  

---

## 1. Executive Summary

| Metric | Fixed Threshold (0.70) | Fence Conformal Gate (alpha=0.10) | Relative Improvement |
| :--- | :---: | :---: | :---: |
| **False-Sufficiency Rate (FSR)** | **24.95%** | **9.90%** | **60.3% Error Reduction** |
| **Conservative Abstention Rate** | 0.00% | 37.20% | Principled Protection |
| **Class 0 Coverage (Insufficient)**| 75.05% | 90.10% | Exceeds 90% Target |
| **Latency Overhead** | ~0.00 ms | 0.005 ms | Negligible Overhead |

---

## 2. Benchmark Configuration
- **Total Instances:** 2000 (1000 calibration, 1000 independent holdout test)
- **Significance Level ($lpha$):** 0.1 (Target Coverage >= 90%)
- **Score Calibration Audit:**
  - Brier Score: `0.1635`
  - Expected Calibration Error (ECE): `0.1048`
  - AUROC (Wilcoxon Exact): `0.8337`

---

## 3. Conformal Prediction Gate
- **Class 0 Nonconformity Quantile ($q_0$):** `0.8555` (Include 0 if $s \le q_0$)
- **Class 1 Nonconformity Quantile ($q_1$):** `0.4159` (Include 1 if $1-s \le q_1$)
- **Calibration Counts:** $n_0 = 517$, $n_1 = 483$

---

## 4. Holdout Evaluation vs. Naive Baselines

| Method / Threshold | False-Sufficiency Rate | True Evidence Recall | Acceptance Rate | Abstention Rate | Conformal Bound Met? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Naive Fixed (tau = 0.60) | 32.76% | 86.95% | 58.50% | 0.00% | False |
| Naive Fixed (tau = 0.70) | 24.95% | 75.37% | 48.90% | 0.00% | False |
| Naive Fixed (tau = 0.80) | 15.43% | 51.79% | 32.70% | 0.00% | False |
| **Fence Conformal Gate** | **9.90%** | **88.00%** | **22.10%** | **37.20%** | **True** |

---

## 5. Lexical Trap Inspection
- **Total Candidate Traps Detected:** `131`
- **Trap Rate Among Insufficient Passages:** `24.95%`

---

## 6. Execution Latencies
- Score Audit: `2.39 ms`
- Gate Fitting: `0.4 ms`
- Holdout Evaluation: `2.54 ms`
- Total Benchmark Runtime: `11.97 ms`
