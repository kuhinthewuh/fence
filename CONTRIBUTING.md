# Contributing to Fence

Thank you for your interest in contributing to **Fence**! We welcome contributions from machine learning researchers, statistical practitioners, retrieval engineers, and open-source developers.

## Our Philosophy

Fence is an open-source statistical toolkit designed for high-integrity evidence sufficiency auditing. We prioritize:
1. **Mathematical Correctness:** Finite-sample guarantees, probability calibration, and conservative abstention must be preserved without compromise.
2. **Minimal Dependencies:** The core statistical engine operates purely on the Python standard library. Do not add external dependencies unless strictly necessary.
3. **Research Integrity:** We do not fabricate benchmarks, exaggerate coverage claims, or represent heuristic filters as formal guarantees.

---

## Development Setup

### 1. Clone & Environment
```bash
git clone https://github.com/fence-ai/fence.git
cd fence
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install in editable mode with development dependencies:
pip install -e ".[dev]"
```

### 2. Running Tests
Run the test suite with pytest:
```bash
pytest -v
```

### 3. Running Benchmarks
Run the reproducible synthetic benchmark:
```bash
python benchmarks/synthetic_benchmark.py
```

### 4. Running the Local MCP Server
Run the MCP server directly over stdio:
```bash
fence-evidence-mcp
```
Or with the demo flag:
```bash
fence-evidence-mcp --demo
```

---

## Submitting a Pull Request

1. Create a feature branch: `git checkout -b feature/my-feature-name`.
2. Ensure code is typed and formatted cleanly.
3. Add unit tests for any new edge cases, math routines, or tools.
4. Verify that all 43+ tests pass (`pytest -v`).
5. Update `CHANGELOG.md` under `[Unreleased]`.
6. Open a Pull Request following the PR template.

