# Frequently Asked Questions (FAQ) for Fence

---

### 1. What makes Fence different from simply using a threshold like `score > 0.8`?
A fixed threshold like `score > 0.8` has no statistical validity. If your retriever's score distribution shifts, or if the proportion of lexical traps increases, a fixed threshold can yield error rates over 30%.
Fence uses **class-conditional split conformal prediction**, which guarantees that under exchangeability, the error rate on insufficient evidence is mathematically bounded below your chosen significance level $\alpha$ (e.g. $\le 10\%$). Furthermore, Fence introduces **conservative abstention** when scores are ambiguous.

---

### 2. Why use class-conditional conformal prediction instead of standard conformal prediction?
Standard (marginal) split conformal prediction guarantees average coverage across all test points: $\mathbb{P}(Y \in \Gamma(X)) \ge 1 - \alpha$.
However, in retrieval systems, most negatives are easy (completely irrelevant documents with near-zero scores). Standard conformal prediction can achieve 90% overall coverage by getting all easy negatives right while failing on 100% of the hard false positives (lexical traps).
**Class-conditional** conformal prediction calculates separate quantiles for class 0 and class 1, guaranteeing $\ge 1 - \alpha$ coverage on each class independently.

---

### 3. What is the exchangeability assumption, and what happens if it is violated?
Conformal prediction guarantees finite-sample coverage under the assumption that calibration observations and test queries are **exchangeable** (drawn independently from the same joint distribution).
If your production queries differ fundamentally from your calibration set (e.g. calibrating on general Wikipedia and testing on specialized legal contracts), exchangeability is violated and empirical coverage may dip below $1 - \alpha$. To mitigate this, calibrate Fence on representative domain-specific queries.

---

### 4. What is a "lexical trap"?
A lexical trap occurs when a retrieved passage has high lexical or semantic embedding similarity to the query, but does not factually support the query.
*Example:*
- Query: *"What were the Q3 2026 earnings for Company X?"*
- Retrieved Passage: *"Company X reported record earnings in Q3 2024 across all major markets."*
- Raw Similarity Score: `0.85` (high keyword and entity overlap).
- True Sufficiency: `0` (wrong year, factual claim unsupported).

---

### 5. How many calibration samples does Fence need?
Because quantiles are calculated as $k = \lceil (n + 1)(1 - \alpha) \rceil$, at $\alpha = 0.10$ (90% confidence), Fence requires at least $n \ge \frac{1}{\alpha} - 1 = 9$ samples per class.
If fewer samples are provided, Fence's finite-sample safeguard sets the threshold to $\infty$, preventing the model from making unsupported claims and conservatively forcing abstention. In production, we recommend $\ge 100$ calibration samples per class for tight prediction intervals.

---

### 6. Does Fence require heavy ML libraries like PyTorch, Scikit-learn, or Pandas?
**No.** Fence's core statistical engine (Brier score, ECE, Wilcoxon AUROC with tie handling, PAVA isotonic regression, Platt scaling, and conformal gating) is implemented entirely using **Python's standard library**.
The only external dependency is the official Model Context Protocol SDK (`mcp>=2.3.0,<3`), ensuring ultra-fast installation and a tiny attack surface.

---

### 7. How fast is Fence?
Evaluating a single score takes less than **5 microseconds** ($\approx 0.005\text{ ms}$). Fitting a conformal gate on 1,000 calibration samples takes approximately **0.4 ms**. Fence adds virtually zero latency overhead to an agent's retrieval loop.

---

### 8. How do I connect Fence to Claude Desktop?
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
Or via `uvx`:
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

---

### 9. Can I use Fence without MCP as a pure Python library?
Yes! All functions and classes are directly importable from `fence_evidence`:
```python
from fence_evidence import fit_conformal_gate, predict_one, audit_scores
```
