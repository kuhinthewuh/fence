# Developer Community Outreach Drafts for Fence

---

## 1. Model Context Protocol (MCP) Official Discord / Community

**Channel:** `#servers` / `#showcase`

```markdown
👋 Hi everyone! We just open-sourced **Fence**, an MCP server for statistical evidence sufficiency and conservative abstention in RAG pipelines.

📦 **PyPI:** `pip install fence-evidence-mcp`
💻 **GitHub:** https://github.com/fence-ai/fence
🛠️ **Transport:** stdio via official MCP Python SDK v2 (`mcp>=2.3.0`)

**What it does:**
Prevents agents from hallucinating when retrieval scores are deceptive (e.g. lexical traps). Instead of trusting raw similarity scores, Fence uses class-conditional split conformal prediction:
- Evaluates scores and computes finite-sample nonconformity quantiles.
- Tools exposed: `audit_scores`, `fit_conformal_gate`, `apply_conformal_gate`, `evaluate_conformal_gate`, `find_lexical_traps`, `run_demo`.
- Includes a rich `fence://methodology` resource and an interactive `audit_rag_pipeline` prompt.
- Answers are accepted ONLY when the conformal prediction set is strictly `{1}`. On ambiguous scores, the server signals `ABSTAIN`.

Would love to get feedback on the MCP schema and agent integration workflows!
```

---

## 2. LlamaIndex Discord

**Channel:** `#showcase` / `#rag`

```markdown
Hey LlamaIndex community!

One persistent problem in production RAG is setting node post-processor similarity thresholds (e.g. `SimilarityPostprocessor(similarity_cutoff=0.7)`). Fixed cutoffs don't have statistical guarantees, and dense retrievers often assign 0.8+ scores to passages with surface keyword overlap that miss the actual answer.

We built an open-source statistical boundary toolkit and MCP server called **Fence**:
https://github.com/fence-ai/fence

You can use it either as an MCP server or directly in your Python post-processing pipeline:
```python
from fence_evidence import fit_conformal_gate, predict_one

# Fit conformal quantiles from your validation nodes
gate = fit_conformal_gate(cal_scores, cal_labels, alpha=0.10)

# Filter retrieved nodes
for node in response_nodes:
    pred = predict_one(node.score, gate)
    if pred.decision == "ACCEPT":
        # Statistically certified sufficient
        yield node
    elif pred.decision == "ABSTAIN":
        # Ambiguous: trigger fallback retrieval or escalate
```
Zero extra dependencies, pure standard library statistical implementation. Check it out and let us know what you think!
```

---

## 3. LangChain Forum / Community

**Topic:** Statistical evidence auditing for Agentic RAG with Fence

```markdown
Hi all!

When building multi-step agentic RAG workflows, one of the biggest risks is an agent taking actions or answering based on retrieval results that have high similarity but lack factual entailment.

We released **Fence** (`fence-evidence-mcp`), an open-source toolkit implementing class-conditional split conformal prediction and conservative abstention for retrieval scores.

Key features for LangChain agents:
- **Calibrated boundaries:** Fits finite-sample quantiles from historical evaluation data.
- **Conservative abstention:** Returns `ACCEPT`, `REJECT`, or `ABSTAIN`. Allows agents to cleanly route ambiguous cases to web search or human-in-the-loop escalation.
- **Standard MCP interface:** Can be called as standard MCP tools from any LangGraph or LangChain agent runtime.

GitHub: https://github.com/fence-ai/fence
PyPI: https://pypi.org/project/fence-evidence-mcp/
```

---

## 4. DSPy Community (Stanford NLP / Discord)

**Channel:** `#announcements` / `#general`

```markdown
Hey DSPy community!

In DSPy metric optimization and assertion tuning, continuous retrieval scores can be misleading due to calibration drift and lexical traps.

We built **Fence** (https://github.com/fence-ai/fence), a zero-dependency statistical toolkit that implements class-conditional split conformal prediction for retrieval scores. It allows you to wrap retrieval modules with finite-sample coverage guarantees ($1 - \alpha$), systematically bounding the false-sufficiency error rate under exchangeability.

It also exposes MCP tools for direct inspection during pipeline optimization. Let us know if you have ideas on DSPy teleprompter integrations!
```

