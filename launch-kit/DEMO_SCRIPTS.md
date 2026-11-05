# Fence Demo Scripts & Recording Instructions

This document provides recording scripts and storyboards for **Fence**, from short 30s clips to the full 100–120 second cinematic terminal demonstration.

---

## 1. Cinematic Terminal Demo (Full 100–120 Second Recording)

**Primary Executable:** `scripts/terminal_demo.py`  
**Target Duration:** 100–120 seconds  
**Narrative Arc:** An AI research assistant attempts to evaluate whether *Intervention A improves Outcome B* in a target study population. A high-similarity lexical trap lures naive thresholds into answering, but Fence's class-conditional conformal boundary rejects the evidence set and enforces conservative abstention. Following automated retrieval escalation, an authentic randomized controlled trial excerpt arrives, which Fence certifies as sufficient (`ACCEPT`).

### Exact Recording Commands

```bash
# Standard normal recording mode (~110 seconds):
python scripts/terminal_demo.py

# Explicit duration calibration (e.g. 120 seconds):
python scripts/terminal_demo.py --duration 120

# Accelerated development/CI mode (~5 seconds):
python scripts/terminal_demo.py --fast

# Monochromatic fallback (disables ANSI color escape codes):
python scripts/terminal_demo.py --no-color
```

### Full 7-Phase Narrative Storyboard

| Phase & Timestamp | Visual Action | On-Screen Elements | Narrative Focus |
| :--- | :--- | :--- | :--- |
| **PHASE 1: Initialization**<br>`(0:00 - 0:11)` | Fence branding banner prints.<br>Staged connection indicators pulse.<br>Active research query card appears. | `FENCE \| Evidence deserves a boundary.`<br>MCP Handshake: `fence-evidence-mcp (stdio)`<br>Query: *"Do the supplied research documents establish that Intervention A improves Outcome B..."* | Establish product identity, local MCP architecture, and the high-stakes evidence-verification task. |
| **PHASE 2: Retrieval**<br>`(0:11 - 0:33)` | Dense vector retrieval & neural reranking animation.<br>Three candidate document cards arrive sequentially with staged delays. | **EXC-01:** Observational Cohort (Sim: `0.84`)<br>**EXC-02:** Phase II Pilot (Sim: `0.72`)<br>**EXC-03:** Solvent Extraction (Sim: `0.19`)<br>`[3 of 3 candidate documents staged]` | Differentiate retriever similarity from logical entailment. Documents staged for audit. |
| **PHASE 3: Lexical Trap**<br>`(0:33 - 0:49)` | Yellow warning banner highlights `EXC-01`.<br>Text analysis shows observational confounding.<br>Naive threshold calculation shown. | `⚠️ LEXICAL TRAP IDENTIFIED IN [EXC-01]`<br>Raw Similarity: `0.84`<br>Naive Rule: `if sim >= 0.75 -> ANSWER()`<br>Result: **Catastrophic False-Sufficiency Error** | Reveal why raw similarity scores fail. High keyword overlap lures naive cutoffs into hallucinating. |
| **PHASE 4: Fence Evaluation**<br>`(0:49 - 0:71)` | Assistant invokes real Fence MCP tools.<br>Displays real probability calibration metrics.<br>Displays fitted conformal bounds & prediction sets. | **Audit:** Brier `0.1614`, ECE `0.0921`, AUROC `0.8397`<br>**Gate (alpha=0.10):** `q0=0.8504`, `q1=0.4952`<br>**Predictions:**<br>• EXC-01: Set `[0, 1]` -> `ABSTAIN`<br>• EXC-02: Set `[0, 1]` -> `ABSTAIN`<br>• EXC-03: Set `[0]` -> `REJECT` | Demonstrate actual statistical engine execution: calibration + class-conditional split conformal prediction. |
| **PHASE 5: Conservative Abstention**<br>`(0:71 - 0:85)` | Sufficiency rule evaluates: 0 of 3 certified.<br>Dramatic AI Assistant response card appears with principled abstention statement. | `✗ Evidence Sufficiency Gate: REJECTED`<br>AI Assistant: *"I don't have enough supporting evidence to answer this reliably... Conservatively abstaining to prevent hallucination."* | The central dramatic moment: Fence halts the agent before generating a false answer. |
| **PHASE 6: Retrieval Escalation**<br>`(0:85 - 1:02)` | Assistant escalates retrieval to RCT repository.<br>`EXC-04` card arrives.<br>Re-evaluation via Fence conformal gate. | **EXC-04:** Double-Blind RCT (N=840, Sim: `0.93`)<br>Evaluation: `0.93 > q0` and `0.07 <= q1`<br>Prediction Set: `{1}` -> **`ACCEPT`**<br>Assistant answers: *"Yes. Intervention A improves Outcome B (p < 0.001)..."* | Validates that when authentic evidence is retrieved, Fence certifies it with mathematical backing. |
| **PHASE 7: Benchmark Results**<br>`(1:02 - 1:15)` | Full 2,000-sample benchmark comparison table renders.<br>Closing banner with installation command. | Naive Cutoff (0.70): `24.9% FSR` (❌ False)<br>Fence Gate: `9.9% FSR`, `37.2% Abstention` (✅ True)<br>Latencies: `0.5 ms` gate fitting<br>`pip install fence-evidence-mcp` | Empirical verification proving theoretical error bound held on 1,000 holdout tests. |

---

## 2. Short-Form Video Demos (Loom, X, LinkedIn)

### 30-Second Demo: "The Lexical Trap Problem"
- **Visual:** Split screen: Left side shows naive threshold accepting `EXC-01` (`0.84 >= 0.75`). Right side shows Fence computing prediction set `[0, 1]` and emitting `ABSTAIN`.
- **Narration:**
  > "Your AI agent retrieved an answer with an 84% similarity score. A standard fixed threshold accepts it immediately. But look closely: it's an observational survey where confounding variables prevented causal claims—a classic lexical trap. Here is Fence. Instead of guessing, Fence uses class-conditional conformal prediction. For ambiguous matches, it enforces conservative abstention, protecting the agent from hallucinating. Install with `pip install fence-evidence-mcp`."

### 60-Second Demo: "Auditing & Gating RAG with Fence"
- **Visual:** Terminal running `python scripts/terminal_demo.py --duration 60`.
- **Narration:**
  > "Most AI agents evaluate evidence using uncalibrated similarity scores. Fence is a new open-source MCP server that establishes mathematical boundaries around retrieved context.
  > First, Fence audits your retrieval scoring—computing Brier score, Expected Calibration Error, and exact Wilcoxon AUROC with tie handling.
  > Next, it fits class-conditional split conformal prediction gates at your chosen error rate. In our benchmark with simulated lexical traps, fixed thresholds allowed 25% false sufficiency. Fence strictly bounded errors to 9.9%, with 37% conservative abstention.
  > When evidence is ambiguous, Fence instructs the agent to abstain rather than guess. When genuine RCT evidence is retrieved, Fence certifies the answer with 90% confidence.
  > Built purely in Python, zero heavy dependencies, sub-millisecond execution. Star us on GitHub and install via pip."

---

## 3. Recommended Recording Tools & Settings

- **OBS Studio:**
  - Resolution: 1920x1080 (1080p) or 2560x1440 (1440p)
  - Framerate: 60 fps
  - Terminal Font: JetBrains Mono, Fira Code, or SF Mono (Size: 16–18pt)
  - Color Theme: One Dark Pro or Tokyo Night
- **VHS (Charm.sh) Automated Terminal Recording:**
  ```tape
  Output demo.mp4
  Set FontSize 18
  Set Width 1280
  Set Height 800
  Set TypingSpeed 0ms
  Type "python scripts/terminal_demo.py --duration 100"
  Enter
  Sleep 105s
  ```
- **Asciinema (Interactive CLI Recording):**
  ```bash
  asciinema rec demo.cast -c "python scripts/terminal_demo.py --duration 100"
  ```
