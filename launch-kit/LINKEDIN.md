# LinkedIn Launch & Thought Leadership Drafts for Fence

---

## Post 1: The Official Launch Announcement (High Reach / General AI Audience)

**Target Publish Time:** Day 1 (Morning)

```markdown
Your AI agent retrieved an answer with a 0.88 similarity score.
Does that evidence actually support the conclusion?

In production RAG systems, the answer is often no.

Today we're launching Fence (v0.1.0)—an open-source statistical evidence-sufficiency MCP server that provides mathematically grounded boundaries around retrieved context.

Tagline: Evidence deserves a boundary.

The problem:
Most RAG architectures rely on arbitrary thresholds like "if similarity > 0.75: answer()". In real pipelines, this causes severe hallucinations due to lexical traps—retrieved documents that share dense keyword overlap with the prompt but contain outdated, irrelevant, or subtly contradictory facts.

How Fence works:
Instead of treating vector similarity scores as ground truth, Fence applies class-conditional split conformal prediction:
1. Recalibrates uncalibrated similarity scores using monotonic isotonic regression (PAVA).
2. Calculates finite-sample nonconformity quantiles (q0 for insufficient evidence, q1 for sufficient evidence).
3. Evaluates evidence for incoming queries:
   - Accept: ONLY when the conformal prediction set is strictly {1} (evidence certified sufficient at significance alpha).
   - Reject: When the prediction set is {0}.
   - Abstain: When the prediction set is {0, 1} or empty (evidence is inconclusive or ambiguous).

In our synthetic benchmark (2,000 queries with simulated lexical traps):
- Naive fixed thresholding accepted misleading evidence in 25.0% to 32.8% of cases.
- Fence strictly bounded false-sufficiency error to 9.90% (guaranteed <= 10.0% at alpha=0.10).
- 37.2% conservative abstentions protected the agent from hallucinatory answers.
- Zero dependencies beyond standard Python library + official MCP Python SDK v2.

Install now via PyPI:
`pip install fence-evidence-mcp`

Connect to Claude Desktop, Cursor, or your agent:
`fence-evidence-mcp`

GitHub: https://github.com/fence-ai/fence
PyPI: https://pypi.org/project/fence-evidence-mcp/

#AI #MachineLearning #ConformalPrediction #RAG #OpenSource #ModelContextProtocol #AISafety #Python
```

---

## Post 2: Technical Deep-Dive — Lexical Traps and Class-Conditional Conformal Boundaries

**Target Publish Time:** Day 2 (Technical Audience)

```markdown
Why marginal conformal prediction fails for RAG evidence evaluation (and why class-conditional boundaries are required):

When applying conformal prediction to RAG, the naive approach is marginal split conformal prediction: compute nonconformity scores and guarantee 1 - alpha overall coverage.

The fatal flaw: Marginal coverage guarantees average performance across the entire dataset.
If 80% of your retrieved documents are completely irrelevant (easy negatives), a marginal conformal predictor can achieve 90% overall coverage while completely failing on the minority class of hard false positives (lexical traps).

In high-stakes agent workflows, false sufficiency has asymmetric cost. An agent answering confidently on invalid evidence is significantly worse than an agent abstaining.

Fence implements Class-Conditional Split Conformal Prediction:
- For class 0 (insufficient evidence): a0(s) = s
- For class 1 (sufficient evidence): a1(s) = 1 - s

We compute separate finite-sample quantiles:
k_y = ceil((n_y + 1) * (1 - alpha)) for y in {0, 1}.

If k_y > n_y, available calibration observations cannot satisfy the finite-sample bound. Rather than pretending a threshold is supported, Fence sets q_y = infinity—meaning class y is conservatively never excluded.

Incoming scores produce prediction sets Gamma(s):
- If s > q0 and 1 - s <= q1: Gamma(s) = {1} -> ACCEPT.
- If s <= q0 and 1 - s > q1: Gamma(s) = {0} -> REJECT.
- If both conditions hold: Gamma(s) = {0, 1} -> ABSTAIN.

This guarantees:
P(Y in Gamma(S) | Y = y) >= 1 - alpha for EACH class conditionally under exchangeability.

Try it in your pipeline:
`pip install fence-evidence-mcp`

Documentation and reproducible benchmark code: https://github.com/fence-ai/fence

#DataScience #MachineLearning #Statistics #ConformalPrediction #LLMs
```

---

## Post 3: Empirical Benchmark Breakdown (Data-Driven / Researcher Audience)

**Target Publish Time:** Day 3

```markdown
Can we mathematically eliminate false-sufficiency hallucinations in RAG?

Here are the actual numbers from Fence's 2,000-query reproducible benchmark:

We simulated a typical RAG distribution consisting of:
- Clean sufficient evidence (Beta(8, 2))
- Unambiguous insufficient evidence (Beta(2, 8))
- Lexical traps: high keyword similarity, wrong factual entailment (Beta(7, 2))
- Ambiguous borderline evidence (Beta(4, 4))

We evaluated 1,000 holdout test queries across standard fixed thresholds vs Fence conformal gating (alpha=0.10):

Results:
- Naive threshold 0.60: False-Sufficiency Rate = 32.76% (1 in 3 accepted answers were based on invalid evidence).
- Naive threshold 0.70: False-Sufficiency Rate = 24.95% (1 in 4 was invalid).
- Naive threshold 0.80: False-Sufficiency Rate = 15.43% (while true evidence coverage dropped to 51.79%).
- Fence Conformal Gate (alpha=0.10): False-Sufficiency Rate = 9.90% (conformal bound holds <= 10.0%) with 88.0% true evidence coverage and 37.2% conservative abstention.

Key takeaway: Fixed thresholds force a lose-lose choice between high hallucination rates and low recall. Conformal prediction with conservative abstention creates a principled third state: "inconclusive evidence".

Check out the reproducible benchmark runner:
`python benchmarks/synthetic_benchmark.py`

Code and research details: https://github.com/fence-ai/fence

#MachineLearning #RAG #AIResearch #StatisticalLearning
```

