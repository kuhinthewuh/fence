# Fence Technical Positioning & Elevator Pitches

## 1. Core Brand Identity
- **Product Name:** Fence
- **Tagline:** Evidence deserves a boundary.
- **Short Definition:** An open-source statistical evidence-sufficiency MCP server that helps AI agents determine whether retrieved evidence genuinely supports an answer using probability calibration, class-conditional split conformal prediction, and conservative abstention.

---

## 2. Positioning Statement (What Fence Is and Is Not)

- **What Fence IS:**
  - A deterministic statistical toolkit for evidence sufficiency.
  - A mathematically principled framework providing finite-sample error control under exchangeability.
  - An MCP server that integrates directly into agent loops to provide conservative abstention.
  - A lightweight, zero-dependency Python implementation designed for high-performance agent infrastructure.

- **What Fence IS NOT:**
  - It is NOT an LLM wrapper or generic prompt-based evaluator.
  - It is NOT a web search aggregator or vector database.
  - It is NOT a black-box machine learning model that makes uncalibrated claims about external reality.
  - It does NOT claim unconditional guarantees against distribution shift.

---

## 3. Elevator Pitches by Time Horizon

### The 10-Second One-Liner
> "Fence gives AI agents a statistically grounded boundary to verify whether retrieved evidence genuinely supports an answer—and abstain when it doesn't."

### The 30-Second Pitch
> "In RAG pipelines, agents frequently hallucinate because vector similarity models assign high scores to passages with strong keyword overlap that don't actually answer the prompt. Arbitrary cutoffs like 'score > 0.75' have zero statistical guarantees. Fence is an open-source MCP server that uses class-conditional split conformal prediction to give agents calibrated evidence boundaries. It strictly bounds false-sufficiency errors at a chosen confidence level, and tells the agent to conservatively abstain whenever the evidence is ambiguous."

### The 60-Second Deep Pitch
> "When building autonomous agents that retrieve and reason over documents, false sufficiency is your highest-cost failure mode. If an agent acts on an invalid document, the downstream decision is compromised.
> Today's teams either trust raw cosine similarity—which suffers from severe miscalibration and lexical traps—or they call an LLM-as-a-judge, which adds seconds of latency and dollars of cost.
> Fence provides a deterministic alternative. Built purely on Python's standard library and the official Model Context Protocol v2, Fence calibrates retrieval scores and computes finite-sample nonconformity quantiles separately for sufficient and insufficient evidence.
> When a new document arrives, Fence constructs a conformal prediction set: if the set is strictly {1}, evidence is certified; if the set is {0}, it is rejected; if it's {0, 1}, Fence enforces conservative abstention. In our benchmarks, this cuts false-sufficiency errors from over 25% down to 9.9% while executing in sub-millisecond time."

