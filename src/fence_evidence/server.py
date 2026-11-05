"""Model Context Protocol (MCP) server for Fence.

Fence: Evidence deserves a boundary.
Exposes statistical score auditing, class-conditional split conformal prediction,
conservative abstention, holdout evaluation, and lexical-trap inspection.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Any, Sequence

from mcp.server import MCPServer

from fence_evidence.conformal import (
    apply_conformal_gate as _apply_conformal_gate,
    evaluate_conformal_gate as _evaluate_conformal_gate,
    find_lexical_traps as _find_lexical_traps,
    fit_conformal_gate as _fit_conformal_gate,
)
from fence_evidence.core import ConformalGate
from fence_evidence.demo import run_demo as _run_demo
from fence_evidence.metrics import audit_scores as _audit_scores

# Configure logging strictly to sys.stderr so stdout remains clean for MCP JSON-RPC
logger = logging.getLogger("fence_evidence")
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [Fence] %(message)s",
)

# Initialize MCP Server v2
app = MCPServer(
    name="Fence",
    version="0.1.0",
    description=(
        "Statistical evidence-sufficiency MCP server. Implements probability calibration, "
        "class-conditional split conformal prediction, and conservative abstention to prevent "
        "AI agents from trusting misleading retrieval results."
    ),
)


# ============================================================================
# MCP Tools
# ============================================================================

@app.tool(
    name="audit_scores",
    description=(
        "Audit retrieval or confidence scores against binary ground truth labels. "
        "Calculates Brier score, Expected Calibration Error (ECE), AUROC with tie handling, "
        "class prevalence, reliability bins, and precision/recall/F1 diagnostics."
    ),
)
def audit_scores(
    scores: list[float],
    labels: list[int],
    threshold: float = 0.5,
    num_bins: int = 10,
) -> dict[str, Any]:
    """Execute statistical score audit."""
    try:
        res = _audit_scores(scores=scores, labels=labels, threshold=threshold, num_bins=num_bins)
        return {"status": "success", "result": res.to_dict()}
    except Exception as err:
        logger.error("audit_scores failed: %s", err)
        return {"status": "error", "error_message": str(err)}


@app.tool(
    name="fit_conformal_gate",
    description=(
        "Fit a class-conditional split conformal prediction gate from calibration data. "
        "Determines finite-sample nonconformity quantiles (q0 for insufficient class, q1 for "
        "sufficient class) guaranteeing 1 - alpha class-conditional coverage under exchangeability."
    ),
)
def fit_conformal_gate(
    scores: list[float],
    labels: list[int],
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Fit class-conditional conformal gate."""
    try:
        gate = _fit_conformal_gate(calibration_scores=scores, calibration_labels=labels, alpha=alpha)
        return {"status": "success", "result": gate.to_dict()}
    except Exception as err:
        logger.error("fit_conformal_gate failed: %s", err)
        return {"status": "error", "error_message": str(err)}


@app.tool(
    name="apply_conformal_gate",
    description=(
        "Apply fitted conformal gate thresholds (q0, q1) to a batch of retrieval scores. "
        "Returns individual prediction sets, statistical decisions (ACCEPT, REJECT, ABSTAIN), "
        "and natural-language explanations. Evidence is accepted ONLY when the set is strictly {1}."
    ),
)
def apply_conformal_gate(
    scores: list[float],
    q0: float,
    q1: float,
    alpha: float = 0.10,
    n0: int = 100,
    n1: int = 100,
) -> dict[str, Any]:
    """Apply conformal gate to scores."""
    try:
        gate = ConformalGate(
            alpha=alpha,
            q0=q0,
            q1=q1,
            n0=n0,
            n1=n1,
            rank0=0,
            rank1=0,
        )
        preds = _apply_conformal_gate(scores=scores, gate=gate)
        return {
            "status": "success",
            "count": len(preds),
            "predictions": [p.to_dict() for p in preds],
        }
    except Exception as err:
        logger.error("apply_conformal_gate failed: %s", err)
        return {"status": "error", "error_message": str(err)}


@app.tool(
    name="evaluate_conformal_gate",
    description=(
        "Evaluate a fitted conformal gate on an independent holdout evaluation set. "
        "Calculates empirical class-conditional coverage, false-sufficiency rate (FSR), "
        "acceptance rate, and abstention rate under strict holdout isolation."
    ),
)
def evaluate_conformal_gate(
    eval_scores: list[float],
    eval_labels: list[int],
    q0: float,
    q1: float,
    alpha: float = 0.10,
    n0: int = 100,
    n1: int = 100,
) -> dict[str, Any]:
    """Evaluate conformal gate on holdout set."""
    try:
        gate = ConformalGate(
            alpha=alpha,
            q0=q0,
            q1=q1,
            n0=n0,
            n1=n1,
            rank0=0,
            rank1=0,
        )
        eval_res = _evaluate_conformal_gate(eval_scores=eval_scores, eval_labels=eval_labels, gate=gate)
        return {"status": "success", "result": eval_res.to_dict()}
    except Exception as err:
        logger.error("evaluate_conformal_gate failed: %s", err)
        return {"status": "error", "error_message": str(err)}


@app.tool(
    name="find_lexical_traps",
    description=(
        "Inspect high-scoring retrieval examples where ground truth indicates insufficient evidence. "
        "Identifies candidate lexical traps (false-sufficiency cases) caused by surface keyword overlap."
    ),
)
def find_lexical_traps(
    scores: list[float],
    labels: list[int],
    threshold: float = 0.70,
    item_ids: list[str] | None = None,
    queries: list[str] | None = None,
    passages: list[str] | None = None,
) -> dict[str, Any]:
    """Identify candidate lexical traps."""
    try:
        report = _find_lexical_traps(
            scores=scores,
            labels=labels,
            threshold=threshold,
            item_ids=item_ids,
            queries=queries,
            passages=passages,
        )
        return {"status": "success", "result": report.to_dict()}
    except Exception as err:
        logger.error("find_lexical_traps failed: %s", err)
        return {"status": "error", "error_message": str(err)}


@app.tool(
    name="run_demo",
    description=(
        "Run an end-to-end reproducible synthetic demonstration of Fence. "
        "Simulates a RAG pipeline with lexical traps, fits a conformal gate, compares with naive thresholding, "
        "and demonstrates false-sufficiency error control and conservative abstention."
    ),
)
def run_demo(
    seed: int = 42,
    alpha: float = 0.10,
) -> dict[str, Any]:
    """Run reproducible demonstration."""
    try:
        demo_out = _run_demo(seed=seed, alpha=alpha)
        return {"status": "success", "result": demo_out}
    except Exception as err:
        logger.error("run_demo failed: %s", err)
        return {"status": "error", "error_message": str(err)}


# ============================================================================
# MCP Resources & Prompts
# ============================================================================

@app.resource(
    "fence://methodology",
    name="Statistical Evidence Sufficiency Methodology",
    mime_type="text/markdown",
    description="Mathematical foundation of class-conditional split conformal prediction and conservative abstention.",
)
def get_methodology() -> str:
    """Return markdown documentation of the statistical methodology."""
    return r"""# Fence Methodology: Evidence Deserves a Boundary

## 1. Problem Formulation
In Retrieval-Augmented Generation (RAG) systems, agents often rely on vector similarity or cross-encoder reranker scores to decide whether retrieved passages support an answer. However, uncalibrated scores suffer from:
1. **Lexical Traps:** High keyword overlap with irrelevant, dated, or contradictory information.
2. **Score Distortion:** Dense retrieval embeddings produce high cosine similarities even on unsupported answers.
3. **Overconfidence:** Fixed thresholds (e.g. `similarity > 0.75`) lack finite-sample error guarantees.

## 2. Class-Conditional Split Conformal Prediction
Fence implements class-conditional split conformal prediction:
- Label $Y = 1$: Sufficient evidence to support the claim.
- Label $Y = 0$: Insufficient evidence (unsupported, irrelevant, or contradictory).

### Nonconformity Measures
- For insufficient evidence ($Y=0$): $a_0(s) = s$ (high score is nonconforming).
- For sufficient evidence ($Y=1$): $a_1(s) = 1 - s$ (low score is nonconforming).

### Finite-Sample Quantiles
For class $y \in \{0, 1\}$ with $n_y$ calibration observations and error level $\alpha \in (0, 1)$:
$$k_y = \lceil (n_y + 1)(1 - \alpha) \rceil$$
$$q_y = a_{y,(k_y)}$$
If $k_y > n_y$, finite-sample coverage cannot be guaranteed with the available observations; $q_y$ defaults to $\infty$.

### Prediction Set
For a new score $s$:
$$\Gamma^\alpha(s) = \{ y \in \{0, 1\} : a_y(s) \le q_y \}$$
- $0 \in \Gamma^\alpha(s) \iff s \le q_0$
- $1 \in \Gamma^\alpha(s) \iff 1 - s \le q_1$

## 3. Conservative Decision Rule
- **ACCEPT:** $\Gamma^\alpha(s) = \{1\}$ ($s > q_0$ and $1 - s \le q_1$). Class 0 is statistically excluded at significance $\alpha$.
- **REJECT:** $\Gamma^\alpha(s) = \{0\}$ ($s \le q_0$ and $1 - s > q_1$). Class 1 is excluded.
- **ABSTAIN:** $\Gamma^\alpha(s) = \{0, 1\}$ or $\emptyset$. The evidence is ambiguous or inconclusive. The agent must conservatively abstain from answering or trigger retrieval escalation.

## 4. Guarantees and Limitations
- **Coverage Guarantee:** Under exchangeability between calibration and test distributions, $\mathbb{P}(Y \in \Gamma^\alpha(S) \mid Y=y) \ge 1 - \alpha$.
- **Exchangeability Assumption:** Assumes test queries and retrieval scores follow the same joint distribution as calibration data.
- **Not Semantic Truth:** Evaluates score reliability, not independent real-world truth.
"""


@app.prompt(
    name="audit_rag_pipeline",
    description="Guided prompt for an AI agent to audit a RAG pipeline's retrieval scores and establish conformal boundaries.",
)
def audit_rag_pipeline_prompt(retriever_name: str = "retriever", confidence_level: float = 0.90) -> str:
    """Return prompt template for pipeline audit."""
    alpha = round(1.0 - confidence_level, 2)
    return f"""You are auditing the evidence sufficiency of the RAG pipeline '{retriever_name}' using Fence.

Target confidence level: {confidence_level:.0%} (alpha = {alpha:.2f})

Follow this procedure:
1. Call `fence.audit_scores` with representative retrieval scores and binary sufficiency labels to inspect Brier score, ECE, and AUROC.
2. If lexical traps or high ECE are detected, run `fence.find_lexical_traps` to identify deceptive passages.
3. Fit the conformal boundary with `fence.fit_conformal_gate` at alpha={alpha:.2f}.
4. Evaluate on holdout data with `fence.evaluate_conformal_gate` to confirm empirical false-sufficiency rate is bounded below {alpha:.0%}.
5. Configure the agent runtime to call `fence.apply_conformal_gate`: answer ONLY when decision is ACCEPT ({1}); abstain when decision is ABSTAIN.
"""


# ============================================================================
# CLI Entry Point
# ============================================================================

def main() -> None:
    """CLI executable entry point for fence-evidence-mcp."""
    parser = argparse.ArgumentParser(
        prog="fence-evidence-mcp",
        description="Fence: Evidence deserves a boundary. Statistical evidence-sufficiency MCP server.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Fence 0.1.0",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run quick synthetic benchmark demo and print results to stderr before starting.",
    )
    args = parser.parse_args()

    if args.demo:
        sys.stderr.write("Running Fence synthetic benchmark demonstration...\n")
        demo_out = _run_demo()
        sys.stderr.write(f"Demo complete: {demo_out['conclusion']}\n\n")

    sys.stderr.write("Starting Fence MCP server over stdio...\n")
    app.run(transport="stdio")


if __name__ == "__main__":
    main()
