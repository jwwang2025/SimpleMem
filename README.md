<div align="center">

<img alt="simplemem_logo" src="https://github.com/user-attachments/assets/6ea54ad1-e007-442c-99d7-1174b10d1fec" width="450">

<div align="center">

## Efficient Lifelong Memory for LLM Agents — Text Memory Core

<small>Store, compress, and retrieve long-term text memories with semantic lossless compression.</small>

</div>

<p><b>Works with any OpenAI-compatible API</b></p>

<p align="center">
  <a href="https://pypi.org/project/simplemem/">
    <img src="https://cdn.simpleicons.org/pypi/3775A9" width="48" height="48" alt="PyPI" />
  </a><br/>
  <sub>
    <a href="https://pypi.org/project/simplemem/"><b>PyPI Package</b></a>
  </sub>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.02553"><img src="https://img.shields.io/badge/arXiv-2601.02553-b31b1b?style=flat&labelColor=555" alt="arXiv"></a>
  <a href="https://github.com/aiming-lab/SimpleMem"><img src="https://img.shields.io/badge/github-SimpleMem-181717?style=flat&labelColor=555&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/aiming-lab/SimpleMem?style=flat&label=license&labelColor=555&color=2EA44F" alt="License"></a>
  <a href="https://pypi.org/project/simplemem/"><img src="https://img.shields.io/pypi/v/simplemem?style=flat&label=pypi&labelColor=555&color=3775A9&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/simplemem/"><img src="https://img.shields.io/pypi/pyversions/simplemem?style=flat&label=python&labelColor=555&color=3775A9&logo=python&logoColor=white" alt="Python"></a>
</p>

<br/>

[🚀 Quick Start](#-quick-start) • [🌟 Overview](#-overview) • [📦 Installation](#-installation) • [📝 Citation](#-citation)

</div>

</div>

<br/>

---

## 📑 Table of Contents

- [🚀 Quick Start](#-quick-start)
- [🌟 Overview](#-overview)
- [📦 Installation](#-installation)
- [⚙️ Configuration](#️-configuration)
- [🧠 How It Works](#-how-it-works)
- [📝 Citation](#-citation)

---

## 🚀 Quick Start

### 🧠 Understanding the Basic Workflow

At a high level, SimpleMem works as a long-term memory system for LLM-based agents. The workflow consists of three simple steps:

1. **Store information** – Dialogues or facts are processed and converted into structured, atomic memories.
2. **Index memory** – Stored memories are organized using semantic embeddings and structured metadata.
3. **Retrieve relevant memory** – When a query is made, SimpleMem retrieves the most relevant stored information based on meaning rather than keywords.

This design allows LLM agents to maintain context, recall past information efficiently, and avoid repeatedly processing redundant history.

### 🎓 Basic Usage

```python
from simplemem import SimpleMem

# Initialize memory system
mem = SimpleMem()

# Add dialogues
mem.add_dialogue(
    "Alice",
    "Bob, let's meet at Starbucks tomorrow at 2pm",
    "2025-11-15T14:30:00",
)
mem.add_dialogue(
    "Bob",
    "Sure, I'll bring the market analysis report",
    "2025-11-15T14:31:00",
)
mem.finalize()

# Query memories
answer = mem.ask("When and where will Alice and Bob meet?")
# → "16 November 2025 at 2:00 PM at Starbucks"
```

### ⚡ Parallel Processing

For large-scale dialogue processing, enable parallel mode:

```python
from simplemem import create

mem = create(
    clear_db=True,
    enable_parallel_processing=True,  # Parallel memory building
    max_parallel_workers=8,
    enable_parallel_retrieval=True,   # Parallel query execution
    max_retrieval_workers=4
)
```

> **💡 Pro Tip**: Parallel processing significantly reduces latency for batch operations!

---

## 🌟 Overview

**SimpleMem** is an efficient lifelong memory system for LLM agents, built on one principle: store *semantically lossless* memory at high information density, so an agent recalls more while spending far fewer tokens.

Most memory systems force a bad trade-off. They either passively accumulate raw interaction history (redundant, token-hungry) or run expensive reasoning loops to filter noise (slow, costly). SimpleMem instead compresses interactions through a three-stage pipeline:

| Stage | What it does |
|:--|:--|
| **1. Semantic Structured Compression** | Distills unstructured interactions into compact memory units (self-contained facts with resolved coreferences and absolute timestamps), each indexed through multiple complementary views for flexible retrieval. |
| **2. Online Semantic Synthesis** | Merges related context within a session into unified abstract representations, removing redundancy as memory is built rather than at query time. |
| **3. Intent-Aware Retrieval Planning** | Infers the search intent behind a query to decide *what* to retrieve and assemble a precise, compact context. |

On the LoCoMo benchmark this delivers a 26.4% average F1 gain over prior systems while cutting inference-time token consumption by roughly 30x.

---

## 📦 Installation

### 📝 Notes for First-Time Users

- Ensure you are using **Python 3.10+ in your active environment**.
- An OpenAI-compatible API key must be configured **before running any memory construction or retrieval**.
- When using non-OpenAI providers (e.g., Qwen or Azure OpenAI), verify both the model name and `OPENAI_BASE_URL` in your config.
- For large dialogue datasets, enabling parallel processing can significantly reduce memory construction time.

### 📋 Requirements

- 🐍 Python 3.10+
- 🔑 OpenAI-compatible API (OpenAI, Qwen, Azure OpenAI, etc.)

### 🛠️ Setup

```bash
# Clone repository
git clone https://github.com/aiming-lab/SimpleMem.git
cd SimpleMem

# Install dependencies
pip install -r requirements.txt

# — OR — install as an editable package
pip install -e .

# Configure API settings
cp config.py.example config.py
# Edit config.py with your API key and preferences
```

---

## ⚙️ Configuration

```python
# config.py
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = None  # or custom endpoint for Qwen/Azure

LLM_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # State-of-the-art retrieval
```

Because SimpleMem talks to any OpenAI-compatible endpoint, you can point it at any compatible provider by setting `OPENAI_BASE_URL`.

### Key Settings

| Setting | Default | Description |
|:--|:--|:--|
| `LLM_MODEL` | `gpt-4.1-mini` | LLM model for memory construction and answer generation |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model for semantic search |
| `SEMANTIC_TOP_K` | `25` | Max entries from semantic (vector) search |
| `KEYWORD_TOP_K` | `5` | Max entries from keyword (BM25) search |
| `STRUCTURED_TOP_K` | `5` | Max entries from structured metadata search |
| `ENABLE_PLANNING` | `True` | Enable multi-query planning for retrieval |
| `ENABLE_REFLECTION` | `True` | Enable reflection-based additional retrieval |
| `MAX_REFLECTION_ROUNDS` | `2` | Maximum reflection rounds |
| `ENABLE_PARALLEL_PROCESSING` | `True` | Enable parallel memory building |
| `ENABLE_PARALLEL_RETRIEVAL` | `True` | Enable parallel query execution |

---

## 🧠 How It Works

### Three-Stage Pipeline

#### 1. Semantic Structured Compression

Raw dialogues are processed and distilled into atomic memory entries. Each entry contains:
- **Lossless restatement** – Self-contained fact with resolved coreferences
- **Absolute timestamps** – All relative times converted to absolute
- **Structured metadata** – Persons, entities, locations, topics, keywords
- **Semantic embedding** – Vector representation for similarity search

#### 2. Online Semantic Synthesis

As memories are built, related context within a session is merged into unified abstract representations, removing redundancy proactively rather than at query time.

#### 3. Intent-Aware Retrieval Planning

When a question is asked, the system:
1. Analyzes the query intent
2. Plans multiple retrieval strategies (semantic, keyword, structured)
3. Reflects on whether the retrieved context is sufficient
4. Generates a concise, accurate answer

---

## 📝 Citation

If you use SimpleMem in your research, please cite:

```bibtex
@article{simplemem2026,
  title={SimpleMem: Efficient Lifelong Memory for LLM Agents},
  author={Liu, Jiaqi and Su, Yaofeng and Xia, Peng and Zhou, Yiyang and Han, Siwei and  Zheng, Zeyu and Xie, Cihang and Ding, Mingyu and Yao, Huaxiu},
  journal={arXiv preprint arXiv:2601.02553},
  year={2026},
  url={https://arxiv.org/abs/2601.02553}
}
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- 🔍 **Embedding Model**: [Qwen3-Embedding](https://github.com/QwenLM/Qwen) - State-of-the-art retrieval performance
- 🗄️ **Vector Database**: [LanceDB](https://lancedb.com/) - High-performance columnar storage
- 📊 **Benchmark**: [LoCoMo](https://github.com/snap-research/locomo) - Long-context memory evaluation framework
