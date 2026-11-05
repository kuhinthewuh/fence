"""Tests for the Fence MCP server and its tool endpoints."""

from __future__ import annotations

import pytest

from fence_evidence.server import (
    app,
    apply_conformal_gate,
    audit_rag_pipeline_prompt,
    audit_scores,
    evaluate_conformal_gate,
    find_lexical_traps,
    fit_conformal_gate,
    get_methodology,
    run_demo,
)


def test_server_metadata() -> None:
    assert app.name == "Fence"
    assert app.version == "0.1.0"
    assert "evidence-sufficiency" in app.description.lower()


def test_tool_audit_scores() -> None:
    res = audit_scores(scores=[0.9, 0.8, 0.2, 0.1], labels=[1, 1, 0, 0])
    assert res["status"] == "success"
    assert res["result"]["sample_count"] == 4
    assert res["result"]["auroc"] == 1.0


def test_tool_audit_scores_error_handling() -> None:
    # Length mismatch
    res = audit_scores(scores=[0.9, 0.8], labels=[1])
    assert res["status"] == "error"
    assert "error_message" in res


def test_tool_fit_conformal_gate() -> None:
    res = fit_conformal_gate(
        scores=[0.9, 0.85, 0.8, 0.1, 0.2, 0.3] * 10,
        labels=[1, 1, 1, 0, 0, 0] * 10,
        alpha=0.10,
    )
    assert res["status"] == "success"
    gate = res["result"]
    assert gate["alpha"] == 0.10
    assert gate["n0"] == 30
    assert gate["n1"] == 30
    assert gate["q0"] <= 1.0
    assert gate["q1"] <= 1.0


def test_tool_apply_conformal_gate() -> None:
    res = apply_conformal_gate(
        scores=[0.1, 0.5, 0.9],
        q0=0.40,
        q1=0.30,
        alpha=0.10,
    )
    assert res["status"] == "success"
    assert res["count"] == 3
    preds = res["predictions"]
    assert preds[0]["decision"] == "REJECT"
    assert preds[1]["decision"] == "ABSTAIN"
    assert preds[2]["decision"] == "ACCEPT"


def test_tool_evaluate_conformal_gate() -> None:
    res = evaluate_conformal_gate(
        eval_scores=[0.1, 0.9],
        eval_labels=[0, 1],
        q0=0.30,
        q1=0.20,
        alpha=0.10,
    )
    assert res["status"] == "success"
    assert res["result"]["eval_count"] == 2


def test_tool_find_lexical_traps() -> None:
    res = find_lexical_traps(
        scores=[0.9, 0.2],
        labels=[0, 0],
        threshold=0.70,
        item_ids=["doc_1", "doc_2"],
    )
    assert res["status"] == "success"
    assert res["result"]["trap_count"] == 1
    assert res["result"]["traps"][0]["item_id"] == "doc_1"


def test_tool_run_demo() -> None:
    res = run_demo(seed=42, alpha=0.10)
    assert res["status"] == "success"
    assert "holdout_conformal_evaluation" in res["result"]


def test_resource_methodology() -> None:
    text = get_methodology()
    assert "# Fence Methodology" in text
    assert "Class-Conditional Split Conformal Prediction" in text
    assert "Conservative Decision Rule" in text


def test_prompt_audit_rag_pipeline() -> None:
    prompt_text = audit_rag_pipeline_prompt("retriever_v1", 0.95)
    assert "retriever_v1" in prompt_text
    assert "alpha = 0.05" in prompt_text

