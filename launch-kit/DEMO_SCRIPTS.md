# Fence Demo Scripts & Storyboards

These scripts are designed for clean, concise video demonstrations (e.g., Loom, screen captures, X/LinkedIn videos, or terminal recordings using `vhs` / `asciinema`).

---

## 30-Second Demo: "The Lexical Trap Problem"

**Visual:** Terminal split screen or clean dark-mode terminal.

| Timestamp | Visual Action | Narration (Voiceover / Captions) |
| :--- | :--- | :--- |
| **0:00 - 0:08** | Terminal displays a RAG retrieval result: Query: *"What were Corporation Z's 2026 earnings?"* Retrieved Passage: *"Corporation Z announced record growth in Q3 2025 across all primary product lines."* Similarity Score: `0.88`. | "Your agent just retrieved a passage with an 88% similarity score. A standard fixed threshold accepts it immediately. But look closely: it's for 2025, not 2026. A classic lexical trap." |
| **0:08 - 0:18** | Run `fence-evidence-mcp --demo` or run `python scripts/terminal_demo.py`. The terminal outputs the Fence conformal boundary analysis with color-coded tables. | "Here is Fence. Instead of trusting raw similarity, Fence computes class-conditional conformal prediction sets." |
| **0:18 - 0:30** | Screen zooms into output: `Prediction Set: [0, 1]` -> `Decision: ABSTAIN`. Then on a certified query: `Prediction Set: [1]` -> `Decision: ACCEPT`. | "For ambiguous matches, Fence outputs ABSTAIN—protecting the agent from hallucinating. Evidence accepted ONLY when mathematically certified. Install via `pip install fence-evidence-mcp`." |

---

## 60-Second Demo: "Auditing and Gating a RAG Pipeline with Fence"

**Visual:** Full terminal with syntax highlighting, then opening Claude Desktop or an MCP client.

| Timestamp | Visual Action | Narration (Voiceover / Captions) |
| :--- | :--- | :--- |
| **0:00 - 0:12** | Command: `pip install fence-evidence-mcp`. Show `pyproject.toml` and clean zero-dependency tree. | "Today, most AI agents evaluate evidence using uncalibrated similarity scores. Fence is a new open-source statistical MCP server that establishes mathematical boundaries around retrieved context." |
| **0:12 - 0:25** | Terminal runs `python examples/rag_pipeline_example.py`. Terminal displays calibration audit: Brier Score (0.16), ECE (0.09), AUROC (0.83). | "Step 1: Fence audits your retrieval scoring. It computes Brier score, Expected Calibration Error, and exact Wilcoxon AUROC with tie handling to diagnose score reliability." |
| **0:25 - 0:42** | Terminal shows `fit_conformal_gate` outputting quantiles `q0=0.85`, `q1=0.41`. Then shows holdout evaluation comparing naive threshold (25% false sufficiency) against Fence (9.9% false sufficiency). | "Step 2: Fence fits class-conditional split conformal gates at your chosen error rate. On 1,000 holdout tests with lexical traps, fixed thresholds allowed 25% errors. Fence strictly bounded false sufficiency to 9.9%." |
| **0:42 - 0:55** | Switch to Claude Desktop or an MCP-connected agent. The agent calls tool `fence.apply_conformal_gate` on incoming queries, receiving `ACCEPT`, `REJECT`, or `ABSTAIN`. | "Step 3: Connect Fence to any MCP client. Your agent receives structured decisions. When evidence is ambiguous, it receives ABSTAIN and escalates retrieval instead of guessing." |
| **0:55 - 1:00** | GitHub repo and PyPI link displayed. | "Open source, zero heavy dependencies, sub-millisecond execution. Star the repo on GitHub and install with pip." |
