# Fence Release & Publishing Guide (v0.1.0)

## 1. Release Metadata
- **Product Name:** Fence
- **Tagline:** Evidence deserves a boundary.
- **PyPI Distribution:** `fence-evidence-mcp`
- **Import Package:** `fence_evidence`
- **Console Script:** `fence-evidence-mcp`
- **MCP Server Name:** `Fence`
- **Release Version:** `0.1.0`
- **Recommended Git Tag:** `v0.1.0`

---

## 2. GitHub Repository Setup

### Repository Details
- **Repository Name:** `fence` (or `fence-evidence-mcp`)
- **Description:** `Statistical evidence-sufficiency MCP server with class-conditional split conformal prediction and conservative abstention.`
- **Website:** `https://pypi.org/project/fence-evidence-mcp/`
- **Topics:**
  `mcp`, `model-context-protocol`, `conformal-prediction`, `rag`, `evidence-sufficiency`, `probability-calibration`, `abstention`, `ai-safety`, `ai-agents`, `python`

### Git Commit & Tag Commands
```bash
git init
git add .
git commit -m "feat: initial public release of Fence v0.1.0"
git tag -a v0.1.0 -m "Release v0.1.0: Statistical evidence-sufficiency MCP server"
git branch -M main
git remote add origin https://github.com/<your-org-or-username>/fence.git
git push -u origin main --tags
```

---

## 3. PyPI Publishing Instructions

The distribution packages have already been built and validated:
- `dist/fence_evidence_mcp-0.1.0-py3-none-any.whl` (Wheel)
- `dist/fence_evidence_mcp-0.1.0.tar.gz` (Source distribution)
- Validated with `twine check dist/*` (PASSED).

### Publish to TestPyPI (Optional Verification)
```bash
python -m twine upload --repository testpypi dist/*
```

### Publish to Production PyPI
When you are ready to release, execute the following command in your terminal and enter your PyPI API token (`pypi-...`):

```bash
python -m twine upload dist/*
```

*(Note: Never commit your PyPI token to version control or hardcode it in scripts).*

---

## 4. Technical Launch Announcement (LinkedIn / Twitter)

```markdown
🚀 Excited to announce the open-source release of Fence (v0.1.0): Evidence deserves a boundary.

When an AI agent retrieves evidence to answer a high-stakes query, how does it know whether that evidence actually supports the conclusion?

Most RAG pipelines rely on raw cosine similarity or cross-encoder logits with arbitrary heuristics like "if score > 0.75: answer()". In production, this fails catastrophically due to lexical traps—passages with dense keyword overlap that are factually ungrounded, outdated, or subtly contradictory.

Fence solves this by treating evidence sufficiency as a formal statistical boundary problem:
1. Probability Calibration: Recalibrates distorted retrieval scores via monotonic isotonic regression (PAVA) and Platt scaling.
2. Class-Conditional Split Conformal Prediction: Guarantees finite-sample coverage at user significance level alpha (e.g. 90% confidence) with asymmetric nonconformity scores for sufficient vs insufficient evidence.
3. Conservative Abstention: Accepts evidence ONLY when the conformal prediction set is strictly {1}. When evidence is ambiguous or borderline ({0, 1}), the agent conservatively abstains from answering.

In our synthetic benchmark (2,000 queries with simulated lexical traps):
- Naive fixed thresholding accepted misleading evidence in 25.0% to 32.8% of cases.
- Fence class-conditional conformal gating strictly bounded the false-sufficiency error rate to 9.90% (guaranteed <= 10.0% at alpha=0.10).
- 37.2% conservative abstentions protected the agent from hallucinatory answers.
- Zero external dependencies beyond standard library + official MCP Python SDK v2.

Install via PyPI:
`pip install fence-evidence-mcp`

Connect to Claude Desktop, Cursor, or your agent:
`fence-evidence-mcp`

GitHub: https://github.com/fence-ai/fence
PyPI: https://pypi.org/project/fence-evidence-mcp/

#AI #MachineLearning #ConformalPrediction #RAG #OpenSource #ModelContextProtocol #LLM #AISafety
```

---

## 5. MCP Directory Submissions

### 1. Model Context Protocol Official Directory / Smithery
- **Name:** Fence
- **Summary:** Statistical evidence-sufficiency MCP server with class-conditional split conformal prediction and conservative abstention.
- **Command:** `uvx fence-evidence-mcp` (or `fence-evidence-mcp`)
- **Capabilities:** Tools, Resources, Prompts
- **License:** MIT

### 2. Awesome MCP Servers (GitHub)
- **Entry:**
  `- [Fence](https://github.com/fence-ai/fence) - Statistical evidence-sufficiency MCP server using class-conditional split conformal prediction and conservative abstention to prevent agents from trusting misleading RAG retrieval results.`

