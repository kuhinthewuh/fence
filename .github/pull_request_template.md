## Description
Briefly describe the change, motivating context, and relevant issue links.

## Type of Change
- [ ] Bug fix (non-breaking change fixing an issue)
- [ ] Statistical / Algorithmic enhancement
- [ ] New feature / MCP tool addition
- [ ] Documentation update
- [ ] Performance optimization
- [ ] CI / Tooling improvement

## Statistical & Mathematical Verification
- [ ] Finite-sample coverage guarantees preserved
- [ ] Exchangeability assumptions noted where relevant
- [ ] No regression in Brier score, ECE, or AUROC calculation logic
- [ ] Degenerate/small-sample safeguards preserved (infinite quantile when $k > n$)

## Testing
- [ ] All pytest unit tests pass (`pytest -v`)
- [ ] New edge cases tested (NaN, infinite, empty, tied scores)
- [ ] Benchmark script verified (`python benchmarks/synthetic_benchmark.py`)

## Checklist
- [ ] My code adheres to PEP 8 standards and includes complete type hints
- [ ] No extraneous dependencies introduced
- [ ] Documentation updated where appropriate
- [ ] `CHANGELOG.md` updated under `[Unreleased]`

