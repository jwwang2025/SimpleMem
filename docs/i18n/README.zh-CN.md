<div align="center">

<img alt="SimpleMem 标志" src="https://github.com/user-attachments/assets/6ea54ad1-e007-442c-99d7-1174b10d1fec" width="450">

<div align="center">

## 面向 LLM 智能体的高效终身记忆 — 文本记忆核心

<small>通过语义无损压缩存储、压缩并检索长期文本记忆。</small>

</div>

<p><b>兼容任何 OpenAI 兼容的 API</b></p>

<p align="center">
  <a href="https://pypi.org/project/simplemem/">
    <img src="https://cdn.simpleicons.org/pypi/3775A9" width="48" height="48" alt="PyPI" />
  </a><br/>
  <sub>
    <a href="https://pypi.org/project/simplemem/"><b>PyPI 包</b></a>
  </sub>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2601.02553"><img src="https://img.shields.io/badge/arXiv-2601.02553-b31b1b?style=flat&labelColor=555" alt="arXiv"></a>
  <a href="https://github.com/aiming-lab/SimpleMem"><img src="https://img.shields.io/badge/github-SimpleMem-181717?style=flat&labelColor=555&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="../../LICENSE"><img src="https://img.shields.io/github/license/aiming-lab/SimpleMem?style=flat&label=license&labelColor=555&color=2EA44F" alt="License"></a>
  <a href="https://pypi.org/project/simplemem/"><img src="https://img.shields.io/pypi/v/simplemem?style=flat&label=pypi&labelColor=555&color=3775A9&logo=pypi&logoColor=white" alt="PyPI"></a>
  <a href="https://pypi.org/project/simplemem/"><img src="https://img.shields.io/pypi/pyversions/simplemem?style=flat&label=python&labelColor=555&color=3775A9&logo=python&logoColor=white" alt="Python"></a>
</p>

<br/>

[🚀 快速开始](#-快速开始) • [🌟 概述](#-概述) • [📦 安装](#-安装) • [📝 引用](#-引用)

</div>

</div>

<br/>

---

## 📑 目录

- [🚀 快速开始](#-快速开始)
- [🌟 概述](#-概述)
- [📦 安装](#-安装)
- [⚙️ 配置](#️-配置)
- [🧠 工作原理](#-工作原理)
- [📝 引用](#-引用)

---

## 🚀 快速开始

### 🧠 基本工作流程

简单来说，SimpleMem 是一个面向 LLM 智能体的长期记忆系统。工作流程包含三个简单步骤：

1. **存储信息** — 对话或事实被处理并转换为结构化的原子记忆
2. **建立索引** — 存储的记忆通过语义嵌入和结构化元数据进行组织
3. **检索相关记忆** — 当发起查询时，SimpleMem 基于语义（而非关键词）检索最相关的存储信息

这种设计使 LLM 智能体能够保持上下文、高效回忆过去的信息，并避免重复处理冗余历史。

### 🎓 基本用法

```python
from simplemem import SimpleMem

# 初始化记忆系统
mem = SimpleMem()

# 添加对话
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

# 查询记忆
answer = mem.ask("When and where will Alice and Bob meet?")
# → "2025 年 11 月 16 日下午 2:00，在星巴克"
```

### ⚡ 并行处理

对于大规模对话处理，可以启用并行模式：

```python
from simplemem import create

mem = create(
    clear_db=True,
    enable_parallel_processing=True,  # ⚡ 并行记忆构建
    max_parallel_workers=8,
    enable_parallel_retrieval=True,   # 🔍 并行查询执行
    max_retrieval_workers=4
)
```

> **💡 提示**：并行处理可显著降低批量操作的延迟！

---

## 🌟 概述

**SimpleMem** 是一个面向 LLM 智能体的高效终身记忆系统，基于一个原则构建：以**语义无损**的高信息密度存储记忆，让智能体在消耗更少 token 的同时记住更多信息。

大多数记忆系统都面临一个糟糕的权衡：它们要么被动累积原始交互历史（冗余、消耗大量 token），要么运行昂贵的推理循环来过滤噪声（慢、成本高）。SimpleMem 则通过一个三阶段流水线来压缩交互：

| 阶段 | 作用 |
|:--|:--|
| **1. 语义结构化压缩** | 将非结构化交互提炼为紧凑的记忆单元（具有已解析指代关系和绝对时间戳的自包含事实），每个单元通过多个互补视图进行索引，实现灵活检索。 |
| **2. 在线语义合成** | 在会话内将相关上下文合并为统一的抽象表示，在记忆构建时（而非查询时）就消除冗余。 |
| **3. 意图感知检索规划** | 推断查询背后的搜索意图，以决定检索什么并组装精确、紧凑的上下文。 |

在 LoCoMo 基准测试上，这一设计比之前的系统平均 F1 提升了 26.4%，同时将推理时的 token 消耗降低了约 30 倍。

---

## 📦 安装

### 📝 首次使用注意事项

- 确保你的活动环境中使用的是 **Python 3.10+**。
- 在运行任何记忆构建或检索之前，必须配置 **OpenAI 兼容的 API 密钥**。
- 使用非 OpenAI 提供商（例如 Qwen 或 Azure OpenAI）时，请在配置中验证模型名称和 `OPENAI_BASE_URL`。
- 对于大型对话数据集，启用并行处理可以显著减少记忆构建时间。

### 📋 系统要求

- 🐍 Python 3.10+
- 🔑 OpenAI 兼容 API（OpenAI、Qwen、Azure OpenAI 等）

### 🛠️ 设置步骤

```bash
# 克隆仓库
git clone https://github.com/aiming-lab/SimpleMem.git
cd SimpleMem

# 安装依赖
pip install -r requirements.txt

# — 或者 — 以可编辑包形式安装
pip install -e .

# 配置 API 设置
cp config.py.example config.py
# 编辑 config.py，填入你的 API 密钥和偏好设置
```

---

## ⚙️ 配置

```python
# config.py
OPENAI_API_KEY = "你的-api-密钥"
OPENAI_BASE_URL = None  # 或 Qwen/Azure 的自定义端点

LLM_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # 最先进的检索模型
```

由于 SimpleMem 可以与任何 OpenAI 兼容的端点通信，你可以通过设置 `OPENAI_BASE_URL` 指向任何兼容的提供商。

### 关键设置

| 设置 | 默认值 | 说明 |
|:--|:--|:--|
| `LLM_MODEL` | `gpt-4.1-mini` | 用于记忆构建和答案生成的 LLM 模型 |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | 用于语义搜索的嵌入模型 |
| `SEMANTIC_TOP_K` | `25` | 语义（向量）搜索返回的最大条目数 |
| `KEYWORD_TOP_K` | `5` | 关键词（BM25）搜索返回的最大条目数 |
| `STRUCTURED_TOP_K` | `5` | 结构化元数据搜索返回的最大条目数 |
| `ENABLE_PLANNING` | `True` | 启用多查询检索规划 |
| `ENABLE_REFLECTION` | `True` | 启用基于反思的补充检索 |
| `MAX_REFLECTION_ROUNDS` | `2` | 最大反思轮数 |
| `ENABLE_PARALLEL_PROCESSING` | `True` | 启用并行记忆构建 |
| `ENABLE_PARALLEL_RETRIEVAL` | `True` | 启用并行查询执行 |

---

## 🧠 工作原理

### 三阶段流水线

#### 1. 语义结构化压缩

原始对话被处理并提炼为原子记忆条目。每个条目包含：
- **无损重述** — 具有已解析指代关系的自包含事实
- **绝对时间戳** — 所有相对时间转换为绝对时间
- **结构化元数据** — 人物、实体、地点、主题、关键词
- **语义嵌入** — 用于相似度搜索的向量表示

#### 2. 在线语义合成

在构建记忆的过程中，会话内的相关上下文被合并为统一的抽象表示，主动消除冗余，而不是等到查询时才处理。

#### 3. 意图感知检索规划

当提出问题时，系统会：
1. 分析查询意图
2. 规划多种检索策略（语义、关键词、结构化）
3. 反思检索到的上下文是否足够
4. 生成简洁、准确的答案

---

## 📝 引用

如果你在研究中使用了 SimpleMem，请引用：

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

## 📄 许可证

本项目采用 **MIT 许可证** — 详见 [LICENSE](../../LICENSE) 文件。

---

## 🙏 致谢

- 🔍 **嵌入模型**：[Qwen3-Embedding](https://github.com/QwenLM/Qwen) — 最先进的检索性能
- 🗄️ **向量数据库**：[LanceDB](https://lancedb.com/) — 高性能列式存储
- 📊 **基准测试**：[LoCoMo](https://github.com/snap-research/locomo) — 长上下文记忆评估框架
