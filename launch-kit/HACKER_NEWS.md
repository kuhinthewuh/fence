# Hacker News Submission Draft (Show HN)

## Title
Show HN: Fence – Statistical evidence sufficiency for AI agents via conformal prediction

## Link URL
`https://github.com/fence-ai/fence`

## Text / Opening Comment (by submitter immediately after posting)

```markdown
Hi HN! We built Fence, an open-source Model Context Protocol (MCP) server that provides statistical boundaries around retrieved evidence for AI agents.

Repo: https://github.com/fence-ai/fence
PyPI: https://pypi.org/project/fence-evidence-mcp/

The Problem:
In RAG pipelines, agents frequently hallucinate because vector similarity models assign high scores to passages with strong keyword overlap that don't actually entail the required answer ("lexical traps"). Most production systems resort to ad-hoc thresholds like "similarity > 0.75", which has no statistical validity, or they use LLM-as-a-judge evaluators that are slow, expensive, and subject to their own calibration errors.

Our Approach:
Fence treats evidence sufficiency as a class-conditional split conformal prediction problem:
1. Calibration & Auditing: We compute Brier scores, Expected Calibration Error (ECE), and exact Wilcoxon AUROC with fractional rank tie-handling to diagnose score reliability. If scores are uncalibrated, isotonic regression (PAVA) or Platt scaling is applied.
2. Class-Conditional Nonconformity:
   - For insufficient class (y=0): a0(s) = s
   - For sufficient class (y=1): a1(s) = 1 - s
   We calculate finite-sample quantiles k_y = ceil((n_y + 1) * (1 - alpha)). If k_y > n_y, the sample size is insufficient to guarantee coverage, so q_y defaults to infinity (conservative retention).
3. Conservative Abstention: An answer is ACCEPTED only if the prediction set is strictly {1} (meaning class 0 is statistically excluded at significance alpha). If the set is {0, 1}, the agent conservatively ABSTAINS.

Under exchangeability, this provides finite-sample class-conditional coverage guarantees: P(Y in Gamma(S) | Y=y) >= 1 - alpha.

Implementation:
- Written in pure Python 3.11+ using the standard library for all statistical operations. Zero heavy scientific packages (no numpy, scipy, or torch required for core math).
- Built on the official Model Context Protocol Python SDK v2 (`mcp>=2.3.0,<3`) running over stdio transport.
- Gate fitting runs in ~0.4 ms; test evaluation runs in ~2.5 ms.

In our synthetic benchmark (2,000 queries with simulated lexical traps):
- Naive fixed thresholding (0.70) had a 24.95% false-sufficiency error rate.
- Fence bounded false-sufficiency to 9.90% at alpha=0.10 (meeting the <= 10.0% theoretical guarantee), with 37.2% conservative abstention.

We'd love HN's thoughts on the statistical formulation, failure modes of the exchangeability assumption in production RAG, and MCP server developer experience!
```

