# CraniMem: Cranial-Inspired Gated and Bounded Memory for Agentic Systems

This repository contains the implementation of CraniMem, a neurocognitively inspired memory framework for long-horizon LLM agents. CraniMem enables stable, goal-aware memory across extended interactions by combining selective encoding, bounded episodic storage, structured long-term knowledge graphs, and replay-based consolidation.


<img width="11310" height="5110" alt="Cranimem" src="https://github.com/user-attachments/assets/f9cd4d20-34ab-48b4-bda3-70686355363a" />


## Overview

Modern LLM agents require persistent memory beyond a single context window. Conventional approaches often treat memory as an external database with heuristic read and write rules, which can lead to unstable retention, noise accumulation, weak consolidation, and poor long-term consistency.

CraniMem models memory as a gated, multi-stage cognitive process inspired by biological systems. Incoming information is filtered for relevance, stored in a bounded episodic buffer for short-term continuity, selectively consolidated into a structured knowledge graph for durable recall, and retrieved through a dual-path mechanism that combines short-term and long-term context.

---

## Key Features

- Goal-conditioned input gating to filter irrelevant information  
- Utility tagging based on importance, surprise, and emotional salience  
- Bounded episodic buffer for recent interaction traces  
- Structured long-term semantic memory using a knowledge graph  
- Replay-based consolidation with pruning of low-utility items  
- Dual-path retrieval from episodic and semantic stores  
- Robustness to distractor noise in long interactions  

---

## Repository Structure

```plaintext
CraniMem/
├─ pyproject.toml          # Package metadata and build configuration
├─ README.md               # Documentation
├─ example.env             # Environment variable template
├─ cranimem/
│  ├─ __init__.py
│  ├─ config.py            # Settings and environment configuration
│  ├─ agent/
│  │  ├─ workflow.py       # LangGraph workflow assembly
│  │  ├─ nodes.py          # Core processing nodes (gating, retrieval, reasoning, memory, consolidation)
│  │  └─ state.py          # Agent state schema and episodic buffer logic
│  ├─ core/
│  │  ├─ embedding.py      # Embedding model loader
│  │  └─ graph_store.py    # Neo4j-backed semantic memory store
│  ├─ cognitive/
│  │  ├─ prompts.py        # Prompt templates
│  │  ├─ gating.py         # Two-stage gating mechanism
│  │  └─ memory_ops.py     # Memory maintenance utilities
│  └─ utils/
│     ├─ llm_factory.py    # LLM provider factory and caching
│     └─ json_utils.py     # JSON parsing utilities
├─ cranimem.egg-info/      # Build metadata
└─ dist/                   # Built distribution artifacts
````

---

## Installation

Install from PyPI:

```bash
pip install cranimem
```

Or install locally for development:

```bash
pip install -e .
```

---

## Configuration

Create a `.env` file based on `example.env`.

Example configuration:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your_api_key

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

CraniMem uses environment variables for:

* LLM provider configuration
* Embedding model settings
* Neo4j connection parameters
* Memory thresholds and limits

---

## Quick Start [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Ub6WGH-vYxhGqclxxLsq9v8W99MAvfN2?usp=sharing)


```python
from cranimem.agent.workflow import build_agent

agent = build_agent()

response = agent.run("Plan a trip to Japan next month.")
print(response)
```

CraniMem will automatically use configured providers and storage backends.

---

## Architecture (High Level)

CraniMem implements a gated, bounded, multi-stage memory pipeline:

1. Collection Phase
   Incoming inputs are filtered by relevance and assigned utility scores.

2. Short-Term Storage
   Accepted items are stored in a bounded episodic buffer.

3. Consolidation Phase
   High-utility traces are replayed into long-term memory while low-utility items are discarded.

4. Long-Term Storage
   Durable information is stored in a structured knowledge graph.

5. Retrieval Phase
   Context is assembled from both episodic and semantic memory to condition the LLM.

---

## Core Components

**Episodic Buffer**
Stores recent interaction traces with full fidelity for short-term continuity. The buffer is bounded to prevent unbounded growth.

**Knowledge Graph Store**
Maintains durable semantic information as structured entities and relationships, enabling multi-hop retrieval.

**Gating Module**
Filters inputs based on relevance to the current goal to prevent noise from entering memory.

**Consolidation Engine**
Periodically transfers high-utility experiences from the episodic buffer to long-term storage and prunes low-value items.

**Dual-Path Retrieval**
Combines short-range context from the episodic buffer with long-range knowledge from the graph.

---

## Use Cases

CraniMem is suitable for applications requiring persistent, coherent memory:

* Long-running conversational agents
* Research and analysis assistants
* Planning and workflow agents
* Multi-session copilots
* Personalized systems
* Multi-agent coordination

---

## Limitations

* Higher latency than simple retrieval systems
* Additional infrastructure requirements (e.g., graph database)
* Performance depends on gating and utility scoring quality
* Increased computational cost during consolidation

---




