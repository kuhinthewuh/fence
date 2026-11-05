"""Tests for synthetic benchmark generation and reproducibility."""

from __future__ import annotations

from benchmarks.synthetic_benchmark import run_benchmark
from fence_evidence.demo import generate_synthetic_rag_data, run_demo


def test_synthetic_data_generator_reproducibility() -> None:
    scores1, labels1, ids1, q1, p1 = generate_synthetic_rag_data(n_samples=50, seed=42)
    scores2, labels2, ids2, q2, p2 = generate_synthetic_rag_data(n_samples=50, seed=42)

    assert scores1 == scores2
    assert labels1 == labels2
    assert ids1 == ids2
    assert q1 == q2
    assert p1 == p2


def test_synthetic_data_generator_diversity() -> None:
    scores, labels, ids, q, p = generate_synthetic_rag_data(n_samples=100, seed=42)
    assert 0 in labels
    assert 1 in labels
    assert min(scores) < 0.3
    assert max(scores) > 0.8


def test_run_demo_output_structure() -> None:
    out = run_demo(seed=42, alpha=0.10)
    assert out["status"] == "success"
    assert "conformal_gate" in out
    assert "holdout_conformal_evaluation" in out
    assert "naive_baseline_comparison" in out
    assert "lexical_traps_identified" in out

    # Verify conformal coverage bounds and error reduction relative to naive baseline
    eval_res = out["holdout_conformal_evaluation"]
    naive_cmp = out["naive_baseline_comparison"]
    assert eval_res["conformal_bounds_held"] is True
    assert eval_res["false_sufficiency_rate"] < naive_cmp["naive_false_sufficiency_rate"]


def test_run_benchmark_execution() -> None:
    res = run_benchmark(n_samples=200, seed=42, alpha=0.10)
    assert res["benchmark_metadata"]["is_synthetic"] is True
    assert res["conformal_holdout_results"]["conformal_bounds_held"] is True
    assert len(res["naive_baseline_comparison"]) == 3
