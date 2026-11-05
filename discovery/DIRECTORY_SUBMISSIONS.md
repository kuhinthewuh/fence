# Fence Open-Source Discovery & Directory Submission Guide

This guide compiles all targeted directory submissions, submission URLs, prerequisites, and copy-paste ready markdown/JSON payloads.

---

## 1. Official Model Context Protocol (MCP) Registry

- **Target URL:** `https://github.com/modelcontextprotocol/servers`
- **Method:** Pull Request adding entry to repository README / registry
- **Prerequisites:** Public GitHub repo, `mcp.json` manifest (already in repo root)
- **Submission Snippet:**
  ```markdown
  ### Evaluation & Safety
  - [Fence](https://github.com/fence-ai/fence) - Statistical evidence-sufficiency MCP server using class-conditional split conformal prediction and conservative abstention to prevent AI agents from trusting misleading RAG retrieval results.
  ```

---

## 2. Smithery.ai Registry

- **Target URL:** `https://smithery.ai/`
- **Method:** Automated import via GitHub repo URL or CLI `smithery login`
- **Prerequisites:** `smithery.yaml` (already in repo root)
- **Manifest:**
  ```yaml
  version: 1.0.0
  name: Fence
  description: Statistical evidence-sufficiency MCP server with class-conditional split conformal prediction and conservative abstention.
  repository: https://github.com/fence-ai/fence
  license: MIT
  startCommand:
    type: stdio
    config:
      command: fence-evidence-mcp
  ```

---

## 3. Awesome-MCP-Servers (GitHub)

- **Target URL:** `https://github.com/punkpeye/awesome-mcp-servers`
- **Method:** Pull Request adding entry under `Evaluation & Monitoring` or `Safety`
- **Prerequisites:** Working MCP server, PyPI package
- **PR Description:**
  ```markdown
  Add Fence: An open-source MCP server for statistical evidence sufficiency, probability calibration, and conservative abstention in RAG systems.
  ```
- **List Entry:**
  ```markdown
  - [Fence](https://github.com/fence-ai/fence) - Statistical evidence-sufficiency MCP server with class-conditional split conformal prediction and conservative abstention.
  ```

---

## 4. PulseMCP Catalog

- **Target URL:** `https://www.pulsemcp.com/`
- **Method:** Web submission form ("Submit a Server")
- **Fields:**
  - Name: `Fence`
  - URL: `https://github.com/fence-ai/fence`
  - Author: `Fence Contributors`
  - Short Bio: `Evidence deserves a boundary. Statistical evidence sufficiency for AI agents.`
  - Tags: `rag`, `conformal-prediction`, `safety`, `evaluation`, `mcp-v2`

---

## 5. PyPI Metadata Optimization

- **PyPI Page:** `https://pypi.org/project/fence-evidence-mcp/`
- **Current Keywords:**
  `mcp`, `model-context-protocol`, `conformal-prediction`, `rag`, `evidence-sufficiency`, `calibration`, `abstention`, `safety`, `evaluation`, `ai-agents`
- **Current Classifiers:**
  - `Development Status :: 4 - Beta`
  - `Intended Audience :: Developers`
  - `Intended Audience :: Science/Research`
  - `Programming Language :: Python :: 3.11`
  - `Programming Language :: Python :: 3.12`
  - `Programming Language :: Python :: 3.13`
  - `Topic :: Scientific/Engineering :: Artificial Intelligence`
  - `Topic :: Software Development :: Quality Assurance`

---

## 6. GitHub Repository Optimization (Upon Remote Setup)

- **Repository Name:** `fence` (or `fence-evidence-mcp`)
- **About / Description:**
  `Statistical evidence-sufficiency MCP server with class-conditional split conformal prediction and conservative abstention.`
- **Website URL:** `https://pypi.org/project/fence-evidence-mcp/`
- **Topics / Tags:**
  `mcp`, `model-context-protocol`, `conformal-prediction`, `rag`, `evidence-sufficiency`, `probability-calibration`, `abstention`, `ai-safety`, `ai-agents`, `python`

