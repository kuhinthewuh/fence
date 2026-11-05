# Early-Adopter Feedback & User Survey Template

We are conducting structured user interviews and collecting feedback from early adopters running Fence in production or experimental RAG systems.

---

## Early-Adopter Survey Form

**User / Organization:**
**Primary Use Case:** (e.g. Enterprise search, customer support agent, biomedical Q&A, legal discovery)
**Retriever Architecture:** (e.g. OpenAI embeddings + Pinecone, Cohere rerank, ColBERT, BM25 + dense hybrid)

### 1. Integration Experience
- **Installation Method:** (`pip install fence-evidence-mcp` / `uvx` / Git clone)
- **Time to First Working Decision:** (< 5 minutes / 15 minutes / > 30 minutes)
- **Did you use Fence as an MCP server or as a Python package?**
- **Were there any configuration or dependency friction points?**

### 2. Statistical Calibration & Gating
- **Approximate size of your calibration dataset:**
- **What significance level ($\alpha$) did you select?** (e.g. 0.05, 0.10, 0.20)
- **Did Fence identify lexical traps in your dataset? If so, what were the typical failure modes?**
- **What was your empirical abstention rate?**
- **Did the conservative abstention behavior improve overall agent trustworthiness?**

### 3. Agent Behavior & Routing
- **How does your agent handle `ABSTAIN` decisions?**
  - [ ] Escalates to secondary retrieval (e.g. broader search, web search)
  - [ ] Routes to human reviewer
  - [ ] Responds to user with "Evidence inconclusive"
  - [ ] Asks clarifying questions
- **What additional metrics or diagnostic tools would make Fence more valuable?**

### 4. General Rating
- **Overall satisfaction (1 - 5 stars):**
- **Would you recommend Fence to other agent developers?** (Yes / Maybe / No)
- **Freeform comments or feature requests:**

---

## Submitting Feedback
Please submit feedback by opening a GitHub Discussion at:
`https://github.com/fence-ai/fence/discussions`
Or submit via issue with label `feedback`.

