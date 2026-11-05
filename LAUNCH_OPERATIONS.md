# Fence: Autonomous Open-Source Growth Operations

**Product:** Fence  
**Tagline:** Evidence deserves a boundary.  
**PyPI Distribution:** `fence-evidence-mcp` (v0.1.0 published)  
**Import Package:** `fence_evidence`  
**CLI Executable:** `fence-evidence-mcp`  
**Target Deadline:** Sunday, November 8, 2026  
**Stretch Campaign Goal:** 5,000 genuine users (distinguished from package downloads)  

---

## 1. Project Status Dashboard (as of Nov 5, 2026)

| Dimension | Status | Notes |
| :--- | :---: | :--- |
| **PyPI Release** | ✅ LIVE | Version `0.1.0` published on PyPI (`fence-evidence-mcp`). |
| **Test Suite** | ✅ 43/43 PASSED | Full unit test coverage across metrics, conformal quantiles, calibration, edge cases, and server endpoints. |
| **Benchmark** | ✅ COMPLETED | 2,000 synthetic samples evaluated; False-Sufficiency Rate bounded at 9.90% vs 24.95% naive error. |
| **MCP v2 Support** | ✅ OPERATIONAL | Implements official Python MCP SDK v2 (`mcp>=2.3.0,<3`) with stdio transport. |
| **Git & Version Control** | 🟡 LOCAL READY | Git initialized locally on `main`; pending GitHub remote creation and initial push. |
| **Launch Materials** | ✅ COMPLETE | Full `launch-kit/` directory with LinkedIn, Reddit, Hacker News, MCP registry, FAQ, and demos. |
| **Automated Analytics** | ✅ OPERATIONAL | `scripts/growth_tracker.py` active with local historical snapshots. |

---

## 2. Daily Execution Checklist (Nov 5 – Nov 8, 2026)

### Day 1: Thursday, November 5, 2026 (Launch Foundation)
- [x] Complete codebase rename and brand integrity verification.
- [x] PyPI package published as `fence-evidence-mcp` (v0.1.0).
- [x] Validate build distribution and wheel integrity with Twine.
- [x] Set up CI/CD workflows (matrix tests, package validation, release OIDC).
- [x] Create complete `launch-kit/` (LinkedIn, Reddit, HN, outreach, demo scripts).
- [x] Build automated terminal demo (`scripts/terminal_demo.py`) and walkthrough.
- [x] Build automated growth analytics engine (`scripts/growth_tracker.py`).
- [ ] **USER ACTION:** Create remote GitHub repository `fence` (or `fence-evidence-mcp`) and push initial commit & tags.
- [ ] **USER ACTION:** Review and post Day 1 LinkedIn launch announcement from `launch-kit/LINKEDIN.md`.

### Day 2: Friday, November 6, 2026 (Developer Community Discovery)
- [ ] **USER ACTION:** Submit "Show HN: Fence" on Hacker News using `launch-kit/HACKER_NEWS.md`.
- [ ] **USER ACTION:** Post technical deep-dive to Reddit (r/MachineLearning, r/LocalLLaMA, r/Python) using `launch-kit/REDDIT.md`.
- [ ] Submit PR to Awesome-MCP-Servers (`https://github.com/punkpeye/awesome-mcp-servers`).
- [ ] Submit server to Smithery.ai catalog (`https://smithery.ai/`).
- [ ] Submit listing to PulseMCP (`https://www.pulsemcp.com/`).
- [ ] Run daily analytics snapshot: `python scripts/growth_tracker.py`.

### Day 3: Saturday, November 7, 2026 (Integration & Ecosystem Outreach)
- [ ] Share tailored integrations in LlamaIndex Discord (`launch-kit/COMMUNITY_OUTREACH.md`).
- [ ] Share in LangChain forum / community (`launch-kit/COMMUNITY_OUTREACH.md`).
- [ ] Post Day 3 technical benchmark breakdown on LinkedIn (`launch-kit/LINKEDIN.md`).
- [ ] Triage incoming GitHub issues and PRs.
- [ ] Run daily analytics snapshot: `python scripts/growth_tracker.py`.

### Day 4: Sunday, November 8, 2026 (Campaign Review & First User Milestone)
- [ ] Triage community feedback and survey responses (`launch-kit/USER_FEEDBACK_TEMPLATE.md`).
- [ ] Aggregate week-1 growth report and measure progress toward 5,000 genuine users.
- [ ] Release v0.1.1 patch if any community-reported edge cases arise.
- [ ] Publish retrospective on lessons learned in conformal evidence gating.

---

## 3. Prioritized Task Board

### P0 — Critical Blockers (Requires User Action)
1. **GitHub Remote Push:** Create the GitHub repository (e.g. `https://github.com/<user>/fence`) and execute initial push with `git push -u origin main --tags`.
2. **PyPI Trusted Publisher Configuration:** In PyPI project settings (`fence-evidence-mcp`), configure GitHub Actions Trusted Publisher for seamless future releases without API tokens.

### P1 — High-Impact Growth Improvements (Ready to Execute)
1. **Directory Submissions:** Submit to Official MCP Servers repo, Smithery, Awesome MCP, and PulseMCP (see `discovery/DIRECTORY_SUBMISSIONS.md`).
2. **Community Posts:** Publish the Show HN and Reddit threads using the tested drafts in `launch-kit/`.
3. **Loom / Video Recording:** Record a 30-second screen capture running `python scripts/terminal_demo.py` following `launch-kit/DEMO_SCRIPTS.md`.

### P2 — Useful Enhancements (Engineering)
1. **LlamaIndex Custom Node Post-Processor:** Package an optional helper `from fence_evidence.integrations.llamaindex import ConformalNodePostprocessor`.
2. **LangChain / LangGraph Runnable Tool:** Package an optional helper `from fence_evidence.integrations.langchain import FenceEvidenceGate`.
3. **Telemetry-Free Local Cache:** Allow saving fitted conformal gates to a lightweight JSON file (`fence_gate.json`) for instant agent restarts.

### P3 — Optional Polish
1. **Interactive HTML Widget:** Rich visual dashboard using KaTeX and SVG for live reliability curve rendering.
2. **Badges Update:** Connect GitHub Actions workflow badge and PyPI download badge once remote is live.

---

## 4. Manual Authorization Checkpoints

In adherence to safety and execution guidelines, the agent **never executes the following actions autonomously**; your explicit authorization is required:
1. Pushing commits or tags to a remote repository.
2. Posting or publishing on LinkedIn, Reddit, Hacker News, Discord, or X.
3. Submitting PRs or form submissions to external catalogs.
4. Publishing any new package version to PyPI.

---

## 5. Routine Operations Command Reference

```bash
# 1. Run all tests
pytest -v

# 2. Run terminal demonstration
python scripts/terminal_demo.py

# 3. Run walkthrough comparison
python examples/lexical_trap_walkthrough.py

# 4. Generate updated benchmark report
python scripts/run_benchmarks.py --out-md analytics/benchmark_report.md

# 5. Collect daily growth snapshot & update report
python scripts/growth_tracker.py

# 6. Build distribution packages
python -m build

# 7. Validate package integrity
python -m twine check --strict dist/*

# 8. Start MCP Server locally (stdio)
fence-evidence-mcp
```

