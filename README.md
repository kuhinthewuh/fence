# Fence
### Evidence deserves a boundary.

[![PyPI version](https://img.shields.io/pypi/v/fence-evidence-mcp.svg?color=blue)](https://pypi.org/project/fence-evidence-mcp/)
[![Python versions](https://img.shields.io/pypi/pyversions/fence-evidence-mcp.svg)](https://pypi.org/project/fence-evidence-mcp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![MCP v2](https://img.shields.io/badge/MCP-v2.3-purple.svg)](https://modelcontextprotocol.io/)

Your AI agent retrieved an answer. But does the evidence actually support it?

**Fence** gives AI agents a statistically grounded way to evaluate evidence sufficiency, identify misleading retrieval scores, calibrate decisions, and abstain when evidence is insufficient.

Unlike generic search wrappers or empirical heuristics, Fence operates as a deterministic statistical toolkit implementing **class-conditional split conformal prediction**, finite-sample validity bounds, probability calibration (isotonic PAVA and Platt scaling), and conservative abstention.

---

## The Problem: Why Raw Retrieval Scores Deceive Agents

Modern Retrieval-Augmented Generation (RAG) pipelines and autonomous agents evaluate retrieval passages using continuous confidence scores: vector cosine similarity, bi-encoder distances, or cross-encoder reranker logits.

In practice, these scores suffer from severe statistical failure modes:

1. **Lexical Traps:** High keyword overlap or semantic phrasing matches irrelevant, dated, or subtly contradictory texts (e.g. an annual report from 2024 answering a query about 2026).
2. **Score Distortion & Miscalibration:** Vector similarity scores cluster in narrow, uncalibrated ranges (e.g., $0.78$ to $0.88$). A score of $0.82$ does not represent an $82\%$ probability that the answer is entailed.
3. **Arbitrary Fixed Thresholds:** Hard-coded cutoffs like `if score > 0.75: answer()` have no formal error guarantees. In the presence of class imbalance or distribution shift, naive thresholds yield catastrophic false-sufficiency rates.
4. **Lack of Principled Abstention:** Standard agents either generate a hallucinated answer or drop the context entirely. They lack a mathematically principled middle ground: certifying when evidence is ambiguous and conservative abstention is mandatory.

---

## Technical Methodology

Fence provides rigorous statistical boundaries around retrieved evidence using class-conditional split conformal prediction.

```
                    +------------------------------------+
                    |   Retrieved Context & Raw Score    |
                    +-----------------+------------------+
                                      |
                                      v
                    +------------------------------------+
                    |  Probability Calibration (PAVA)    |
                    +-----------------+------------------+
                                      | Calibrated Score s
                                      v
          +------------------------------------------------------+
          | Class-Conditional Conformal Boundary Test (alpha)    |
          |                                                      |
          |  Insufficient Hyp (Y=0): a0(s) = s      <= q0?       |
          |  Sufficient Hyp   (Y=1): a1(s) = 1 - s  <= q1?       |
          +---------------------------+--------------------------+
                                      |
                 +--------------------+--------------------+
                 |                    |                    |
                 v                    v                    v
          Gamma = {1}          Gamma = {0, 1}       Gamma = {0}
         +-----------+        +--------------+     +-----------+
         |  ACCEPT   |        |   ABSTAIN    |     |  REJECT   |
         | Answer is |        | Ambiguous or |     | Evidence  |
         | certified |        | Inconclusive |     | fails     |
         +-----------+        +--------------+     +-----------+
```

### 1. Nonconformity Measures
Let $Y \in \{0, 1\}$ denote ground-truth evidence sufficiency ($1 = \text{sufficient}$, $0 = \text{insufficient}$), and let $s \in [0, 1]$ represent the retrieval score.
- For insufficient evidence ($Y=0$): $a_0(s) = s$ (high scores are nonconforming to insufficiency).
- For sufficient evidence ($Y=1$): $a_1(s) = 1 - s$ (low scores are nonconforming to sufficiency).

### 2. Finite-Sample Class-Conditional Quantiles
Given $n_0$ calibration examples of class 0 and $n_1$ calibration examples of class 1, with user significance level $\alpha \in (0, 1)$:
$$k_y = \left\lceil (n_y + 1)(1 - \alpha) \right\rceil \quad \text{for } y \in \{0, 1\}$$
$$q_y = a_{y,(k_y)}$$

> **Finite-Sample Safeguard:** If $k_y > n_y$, available calibration observations are insufficient to guarantee $1 - \alpha$ coverage. Rather than pretending a threshold is supported, Fence sets $q_y = \infty$, ensuring class $y$ is conservatively retained.

### 3. Prediction Sets and Decision Rule
For a test score $s$, the conformal prediction set $\Gamma^\alpha(s)$ contains all labels whose nonconformity does not exceed the calibrated threshold:
$$\Gamma^\alpha(s) = \{ y \in \{0, 1\} : a_y(s) \le q_y \}$$

- **ACCEPT:** $\Gamma^\alpha(s) = \{1\}$ ($s > q_0$ and $s \ge 1 - q_1$). The hypothesis of insufficiency ($0$) is statistically rejected at level $\alpha$. The agent may safely answer.
- **REJECT:** $\Gamma^\alpha(s) = \{0\}$ ($s \le q_0$ and $s < 1 - q_1$). Sufficiency ($1$) is rejected. The agent knows the evidence is insufficient.
- **ABSTAIN:** $\Gamma^\alpha(s) = \{0, 1\}$ or $\emptyset$. Both hypotheses conform to the calibration distribution. The evidence is ambiguous. The agent must conservatively abstain or request broader retrieval.

---

## Statistical Guarantees and Limitations

### Guarantees
- **Class-Conditional Coverage:** Under the exchangeability assumption between calibration observations and test queries, the probability that the true label is included in the prediction set is mathematically bounded:
  $$\mathbb{P}(Y \in \Gamma^\alpha(S) \mid Y = y) \ge 1 - \alpha \quad \text{for } y \in \{0, 1\}$$
- **Bounded False-Sufficiency Rate:** Because an answer is accepted only when $0 \notin \Gamma^\alpha(s)$, the false-sufficiency rate (accepting insufficient evidence) is strictly controlled:
  $$\text{FSR} = \mathbb{P}(\text{Decision} = \text{ACCEPT} \mid Y = 0) \le \alpha$$

### Limitations
1. **Exchangeability Assumption:** Conformal guarantees assume exchangeable retrieval distributions. Systematic domain shifts or unseen query distributions can breach coverage bounds.
2. **Sample Size Requirements:** At $\alpha = 0.10$, at least $9$ calibration samples per class are strictly required ($n \ge \frac{1}{\alpha} - 1$). Smaller sample sets yield infinite quantiles and conservative abstention.
3. **Score Informativeness:** Conformal prediction does not create information. If the underlying retriever has near-chance AUROC ($\approx 0.50$), prediction sets will predominantly be $\{0, 1\}$, resulting in high abstention rates.
4. **Not Semantic Proof:** Fence evaluates the statistical reliability of the scoring model. It does not perform symbolic fact checking or replace external knowledge verification.

---

## Synthetic Benchmark Results

The benchmark evaluates 2,000 synthetic RAG retrieval instances (1,000 calibration, 1,000 holdout test; seed=42) containing true positives, true negatives, borderline ambiguous cases, and simulated lexical traps (high lexical similarity with incorrect factual grounding).

Run the reproducible benchmark locally:
```bash
fence-evidence-mcp --demo
# or execute directly:
python benchmarks/synthetic_benchmark.py
```

### Empirical Results Summary

| Method / Threshold | False-Sufficiency Rate (FSR) | True Evidence Coverage | Acceptance Rate | Abstention Rate | Conformal Bound Held? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naive Fixed ($\tau = 0.60$)** | 32.76% | 86.95% | 58.50% | 0.00% | ❌ False (Severe Error) |
| **Naive Fixed ($\tau = 0.70$)** | 24.95% | 75.37% | 48.90% | 0.00% | ❌ False (Severe Error) |
| **Naive Fixed ($\tau = 0.80$)** | 15.43% | 51.79% | 32.70% | 0.00% | ❌ False (Compromised Recall) |
| **Fence Conformal Gate ($\alpha=0.10$)** | **9.90%** | **88.00%** | **22.10%** | **37.20%** | ✅ **True ($\le 10.0\%$)** |

**Benchmark Highlights:**
- Naive fixed thresholding accepted misleading lexical traps in up to **32.8%** of insufficient evidence cases.
- Fence class-conditional conformal gating bounded the empirical false-sufficiency rate to **9.90%** (meeting the $\le 10.0\%$ theoretical bound).
- **37.2%** conservative abstention safely shielded the agent from deceptive and borderline cases.
- Gate fitting executes in **0.41 ms**; full holdout evaluation executes in **2.48 ms**.

---

## Installation

Install Fence via pip:

```bash
pip install fence-evidence-mcp
```

Or install with development dependencies:

```bash
pip install "fence-evidence-mcp[dev]"
```

---

## Quickstart (Python API)

Fence is a zero-dependency statistical engine designed for standalone Python use or MCP integration.

```python
from fence_evidence import fit_conformal_gate, predict_one, Decision

# 1. Calibration scores and binary labels (1 = sufficient, 0 = insufficient)
cal_scores = [0.92, 0.88, 0.85, 0.79, 0.95, 0.12, 0.25, 0.31, 0.18, 0.42, 0.82, 0.77]
cal_labels = [1,    1,    1,    1,    1,    0,    0,    0,    0,    0,    0,    0]

# 2. Fit conformal boundary with alpha = 0.10 (90% target coverage)
gate = fit_conformal_gate(cal_scores, cal_labels, alpha=0.10)
print(f"Fitted Gate: q0={gate.q0:.4f}, q1={gate.q1:.4f}")

# 3. Predict on new retrieval scores
test_scores = [0.94, 0.81, 0.52, 0.15]
for score in test_scores:
    pred = predict_one(score, gate)
    print(f"Score: {score:.2f} -> Decision: {pred.decision.value} (Set: {pred.prediction_set})")
    if pred.decision == Decision.ACCEPT:
        print("  Action: Answer query with certified evidence.")
    elif pred.decision == Decision.REJECT:
        print("  Action: Reject evidence as insufficient.")
    else:
        print("  Action: Conservatively abstain or escalate retrieval.")
```

---

## MCP Server Configuration

Fence implements the official Model Context Protocol (MCP) Python SDK v2. Connect it to Claude Desktop, Cursor, Goose, or any MCP-compliant client.

### Claude Desktop Configuration
Add Fence to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "fence": {
      "command": "fence-evidence-mcp"
    }
  }
}
```

Or run via `uv`:
```json
{
  "mcpServers": {
    "fence": {
      "command": "uvx",
      "args": ["--from", "fence-evidence-mcp", "fence-evidence-mcp"]
    }
  }
}
```

### Antigravity / Custom MCP Client Configuration
```json
{
  "mcpServers": {
    "Fence": {
      "command": "python",
      "args": ["-m", "fence_evidence.server"]
    }
  }
}
```

---

## MCP Tool Documentation

| Tool | Parameters | Description |
| :--- | :--- | :--- |
| `audit_scores` | `scores: list[float]`, `labels: list[int]`, `threshold: float = 0.5`, `num_bins: int = 10` | Computes Brier score, Expected Calibration Error (ECE), AUROC with exact tie ranks, prevalence, reliability bins, and validation warnings. |
| `fit_conformal_gate` | `scores: list[float]`, `labels: list[int]`, `alpha: float = 0.10` | Fits class-conditional split conformal prediction quantiles ($q_0, q_1$) with finite-sample rank safeguards. |
| `apply_conformal_gate` | `scores: list[float]`, `q0: float`, `q1: float`, `alpha: float = 0.10` | Applies fitted boundary thresholds to scores, returning individual decisions (`ACCEPT`, `REJECT`, `ABSTAIN`) and prediction sets. |
| `evaluate_conformal_gate` | `eval_scores: list[float]`, `eval_labels: list[int]`, `q0: float`, `q1: float`, `alpha: float = 0.10` | Evaluates fitted gate on holdout data, reporting class coverage, false-sufficiency rates, and abstention rates. |
| `find_lexical_traps` | `scores: list[float]`, `labels: list[int]`, `threshold: float = 0.70`, `item_ids`, `queries`, `passages` | Identifies candidate false-sufficiency cases where high retrieval scores coincide with insufficient evidence. |
| `run_demo` | `seed: int = 42`, `alpha: float = 0.10` | Runs an end-to-end reproducible synthetic demonstration of calibration, gating, and naive baseline comparison. |

### MCP Resource
- `fence://methodology`: Markdown reference detailing mathematical formulations, quantile derivations, and operational deployment rules.

### MCP Prompt
- `audit_rag_pipeline`: Interactive agent auditing workflow for calibrating retrievers and establishing conformal boundaries.

---

## Examples

Run the bundled examples from the repository root:

```bash
# Agent verification workflow
python examples/agent_audit_example.py

# End-to-end RAG pipeline statistical audit
python examples/rag_pipeline_example.py
```

---

## Contributing

Contributions from machine learning researchers, retrieval engineers, and statistical practitioners are welcome:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/improved-bounds`).
3. Ensure all tests pass (`pytest -v`).
4. Format code and ensure type hints are preserved.
5. Submit a descriptive Pull Request.

---

## License

Fence is licensed under the [MIT License](LICENSE).
Copyright (c) 2026 Fence Contributors.

