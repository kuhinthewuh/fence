# Reddit Community Discussion Drafts for Fence

---

## 1. r/MachineLearning [P] (Project / Research Flair)

**Title:** [P] Fence: Statistical evidence sufficiency and conservative abstention for RAG via class-conditional split conformal prediction

**Post Body:**
```markdown
Hi r/MachineLearning,

I wanted to share an open-source project I’ve been developing called **Fence**: an evidence-sufficiency MCP server that helps AI agents decide whether retrieved context genuinely supports an answer before generating.

**GitHub:** https://github.com/fence-ai/fence
**PyPI:** `pip install fence-evidence-mcp`

### The Problem
In RAG pipelines, dense retrieval embeddings and rerankers frequently assign high similarity scores (e.g. 0.85+) to passages with high lexical overlap that are factually ungrounded, dated, or subtly contradictory ("lexical traps"). Setting heuristic cutoffs like `if score > 0.75: answer()` provides zero finite-sample error guarantees, while marginal conformal prediction averages error over easy negatives and fails on hard false positives.

### Methodology
Fence implements class-conditional split conformal prediction:
- Binary ground truth: $Y \in \{0, 1\}$ (1 = sufficient evidence, 0 = insufficient).
- Nonconformity scores:
  - For insufficient class 0: $a_0(s) = s$ (high score is nonconforming).
  - For sufficient class 1: $a_1(s) = 1 - s$ (low score is nonconforming).
- Class-specific finite-sample quantiles:
  $k_y = \lceil (n_y + 1)(1 - \alpha) \rceil$
  If $k_y > n_y$, sample size is insufficient to guarantee coverage; $q_y$ defaults to $\infty$ (conservative non-exclusion).
- Prediction set: $\Gamma^\alpha(s) = \{ y : a_y(s) \le q_y \}$.
- Decision rule:
  - Accept: strictly when $\Gamma^\alpha(s) == \{1\}$ ($s > q_0$ and $1 - s \le q_1$).
  - Reject: when $\Gamma^\alpha(s) == \{0\}$.
  - Abstain: when $\Gamma^\alpha(s) == \{0, 1\}$ or $\emptyset$ (inconclusive / ambiguous).

Under exchangeability, this guarantees $\mathbb{P}(Y \in \Gamma^\alpha(S) \mid Y=y) \ge 1 - \alpha$ for each class conditionally, strictly bounding the false-sufficiency rate (accepting insufficient evidence) below $\alpha$.

### Implementation Details
- Built purely on Python's standard library for the math engine (Brier score, ECE, Wilcoxon AUROC with exact fractional rank tie-handling, PAVA isotonic regression, Platt scaling).
- Zero heavy dependencies (no torch, no pandas, no scikit-learn).
- Built on the official Model Context Protocol Python SDK v2 (`mcp>=2.3.0,<3`) over stdio transport.

### Benchmark (2,000 synthetic queries, seed=42)
- Naive fixed threshold 0.70: False-Sufficiency Rate = 24.95%.
- Fence conformal gate ($\alpha = 0.10$): False-Sufficiency Rate = 9.90% (bound holds $\le 10.0\%$), with 37.2% conservative abstention.
- Gate fitting runtime: 0.41 ms.

I would love to get feedback from anyone working on retrieval auditing, uncertainty quantification, or conformal prediction in LLM systems.
```

---

## 2. r/LocalLLaMA (Practical / Local Agent Developers)

**Title:** Fence: Stop your local agents from hallucinating on bad retrieval matches using conformal boundaries (MCP server)

**Post Body:**
```markdown
Hey everyone,

If you run local RAG agents (via Ollama, vLLM, LM Studio, etc.), you've probably noticed that cross-encoders and vector search often give high similarity scores to passages that look relevant but don't actually answer the prompt.

We just released **Fence** as an open-source MCP server to fix this deterministically.

**TL;DR:**
Instead of hardcoding a similarity cutoff like `if score > 0.8: generate()`, Fence uses class-conditional conformal prediction to establish calibrated evidence boundaries:
- If evidence is certified sufficient: returns `ACCEPT` -> your agent answers.
- If evidence is certified insufficient: returns `REJECT` -> your agent rejects.
- If evidence is ambiguous or borderline: returns `ABSTAIN` -> your agent abstains or escalates retrieval.

### Quick Setup:
```bash
pip install fence-evidence-mcp
```

In Claude Desktop / Cursor / any MCP client:
```json
{
  "mcpServers": {
    "fence": {
      "command": "fence-evidence-mcp"
    }
  }
}
```

Or use it directly as a Python library:
```python
from fence_evidence import fit_conformal_gate, predict_one

gate = fit_conformal_gate(cal_scores, cal_labels, alpha=0.10)
pred = predict_one(score=0.82, gate=gate)
print(pred.decision) # ACCEPT, REJECT, or ABSTAIN
```

Zero heavy dependencies, pure standard library statistical implementation, and runs in sub-millisecond time.

Check out the repo: https://github.com/fence-ai/fence
```

---

## 3. r/Python (Python Engineering Focus)

**Title:** Fence: A lightweight, zero-dependency statistical evidence-sufficiency MCP server in Python 3.11+

**Post Body:**
```markdown
Hi r/Python,

I wanted to share **Fence** (`fence-evidence-mcp`), an open-source statistical toolkit and MCP server for auditing evidence sufficiency in AI agent pipelines.

PyPI: https://pypi.org/project/fence-evidence-mcp/
GitHub: https://github.com/fence-ai/fence

### Why we built it without numpy/scipy/scikit-learn
Most ML tooling pulls in massive dependency trees for basic statistical operations. For Fence, we implemented:
- Exact Wilcoxon-Mann-Whitney AUROC with fractional rank tie-handling in $O(N \log N)$ pure Python.
- Expected Calibration Error (ECE) and reliability binning.
- Pool Adjacent Violators Algorithm (PAVA) for isotonic regression.
- Platt scaling with Newton-Raphson optimization.
- Class-conditional split conformal prediction with finite-sample rank safeguards.

The entire core statistical package has zero external runtime dependencies beyond Python's standard library and the official `mcp>=2.3.0` protocol SDK.

Unit tests pass across Python 3.11, 3.12, and 3.13 on Linux, Windows, and macOS.

Feedback on the packaging, pure Python algorithmic implementation, or MCP integration is very welcome!
```
