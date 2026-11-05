# MCP Registry & Catalog Submission Metadata

## 1. Official Model Context Protocol (MCP) Registry

- **Name:** Fence
- **Server ID:** `io.github.fence-ai.fence` (or `fence-evidence-mcp`)
- **Version:** `0.1.0`
- **Description:** Statistical evidence-sufficiency MCP server. Implements probability calibration, class-conditional split conformal prediction, and conservative abstention to prevent AI agents from trusting misleading retrieval results.
- **Repository:** `https://github.com/fence-ai/fence`
- **Homepage:** `https://pypi.org/project/fence-evidence-mcp/`
- **License:** MIT
- **Category:** AI Safety & Evaluation / RAG / Machine Learning
- **Transport:** stdio
- **Runtime:** Python (3.11+)
- **Execution Command:**
  ```json
  {
    "command": "uvx",
    "args": ["--from", "fence-evidence-mcp", "fence-evidence-mcp"]
  }
  ```
- **Tools Provided:**
  - `audit_scores`: Complete statistical audit (Brier, ECE, AUROC, reliability bins, warnings).
  - `fit_conformal_gate`: Fits finite-sample class-conditional conformal quantiles.
  - `apply_conformal_gate`: Evaluates scores into prediction sets and decisions (`ACCEPT`, `REJECT`, `ABSTAIN`).
  - `evaluate_conformal_gate`: Holdout evaluation under strict data isolation.
  - `find_lexical_traps`: Discovers candidate high-similarity false-sufficiency cases.
  - `run_demo`: Reproducible synthetic demonstration.
- **Resources Provided:**
  - `fence://methodology`: In-depth documentation of mathematical proofs and operational rules.
- **Prompts Provided:**
  - `audit_rag_pipeline`: Interactive guided workflow for auditing RAG pipelines.

---

## 2. Smithery.ai Registry Manifest (`smithery.yaml`)

```yaml
version: 1.0.0
name: Fence
description: Statistical evidence-sufficiency MCP server with class-conditional split conformal prediction and conservative abstention.
repository: https://github.com/fence-ai/fence
license: MIT
startCommand:
  type: stdio
  config:
    command: fence-evidence-mcp
```

---

## 3. Awesome-MCP-Servers PR Snippet

```markdown
### Evaluation & Safety
- [Fence](https://github.com/fence-ai/fence) - Statistical evidence-sufficiency MCP server that helps AI agents determine whether retrieved evidence genuinely supports an answer using probability calibration, class-conditional split conformal prediction, and conservative abstention.
```

---

## 4. PulseMCP Entry

- **Title:** Fence – Evidence deserves a boundary
- **Description:** Open-source statistical toolkit and MCP server for auditing evidence sufficiency in RAG pipelines. Prevents agents from acting on deceptive retrieval scores via finite-sample conformal prediction and conservative abstention.
- **Tags:** `mcp`, `rag`, `conformal-prediction`, `safety`, `evaluation`
