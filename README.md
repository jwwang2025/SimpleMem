<div align="center">

<img alt="simplemem_logo" src="https://github.com/user-attachments/assets/6ea54ad1-e007-442c-99d7-1174b10d1fec" width="450">

<div align="center">

## 面向 LLM 智能体的高效终身记忆 —— 文本记忆核心

<small>通过语义无损压缩，存储、压缩并检索长期文本记忆。</small>

</div>

<p><b>兼容任意 OpenAI 风格的 API</b></p>

<p align="center">
  <a href="https://pypi.org/project/simplemem/">
    <img src="https://cdn.simpleicons.org/pypi/3775A9" width="48" height="48" alt="PyPI" />
  </a><br/>
  <sub>
    <a href="https://pypi.org/project/simplemem/"><b>PyPI 软件包</b></a>
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

[🚀 快速开始](#-快速开始) • [🌟 概述](#-概述) • [📦 安装](#-安装) • [📝 引用](#-引用)

</div>

</div>

<br/>

> 🌐 语言：**中文** ｜ [English（上游官方 README）](https://github.com/aiming-lab/SimpleMem/blob/main/README.md)

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

### 🧠 理解基本工作流

从整体上看，SimpleMem 是一个面向基于 LLM 的智能体的长期记忆系统。整个工作流只包含三个简单步骤：

1. **存储信息** —— 对话或事实被处理并转换为结构化的原子记忆。
2. **记忆建索引** —— 已存储的记忆通过语义向量与结构化元数据进行组织。
3. **检索相关记忆** —— 当发起查询时，SimpleMem 依据语义而非关键词，检索最相关的已存储信息。

这一设计使 LLM 智能体能够保持上下文、高效回忆过往信息，并避免反复处理冗余历史。

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
# → "16 November 2025 at 2:00 PM at Starbucks"
```

### ⚡ 并行处理

对于大规模对话处理，可启用并行模式：

```python
from simplemem import create

mem = create(
    clear_db=True,
    enable_parallel_processing=True,  # 并行构建记忆
    max_parallel_workers=8,
    enable_parallel_retrieval=True,   # 并行执行查询
    max_retrieval_workers=4
)
```

> **💡 实用技巧**：并行处理可显著降低批量操作的延迟！

---

## 🌟 概述

**SimpleMem** 是一个面向 LLM 智能体的高效终身记忆系统，其构建遵循一个核心原则：以高信息密度存储*语义无损*的记忆，让智能体以更少的 token 回忆起更多内容。

大多数记忆系统都迫使使用者做出糟糕的取舍：要么被动堆积原始交互历史（冗余且耗费 token），要么运行昂贵的推理循环来过滤噪声（缓慢且成本高）。SimpleMem 则通过一个三阶段流水线对交互进行压缩：

| 阶段 | 作用 |
|:--|:--|
| **1. 语义结构化压缩** | 将非结构化交互提炼为紧凑的记忆单元（自包含的事实，已消解指代并附带绝对时间戳），每个单元通过多个互补视图建立索引，以支持灵活检索。 |
| **2. 在线语义综合** | 将会话内相关的上下文合并为统一的抽象表示，在记忆构建阶段（而非查询阶段）就消除冗余。 |
| **3. 意图感知的检索规划** | 推断查询背后的搜索意图，以决定*检索什么*，并组装出精确、紧凑的上下文。 |

在 LoCoMo 基准测试上，相比此前的系统，该方案平均 F1 提升 26.4%，同时将推理阶段的 token 消耗降低约 30 倍。

---

## 📦 安装

### 📝 首次使用须知

- 请确保当前活动环境使用的是 **Python 3.10 及以上版本**。
- 在运行任何记忆构建或检索之前，**必须先配置好** OpenAI 风格 API 的密钥。
- 使用非 OpenAI 提供方（如通义千问 Qwen 或 Azure OpenAI）时，请确认配置中的模型名称和 `OPENAI_BASE_URL` 均正确无误。
- 对于大型对话数据集，启用并行处理可显著缩短记忆构建时间。

### 📋 环境要求

- 🐍 Python 3.10+
- 🔑 OpenAI 兼容 API（OpenAI、通义千问 Qwen、Azure OpenAI 等）

### 🛠️ 安装步骤

```bash
# 克隆仓库
git clone https://github.com/aiming-lab/SimpleMem.git
cd SimpleMem

# 安装依赖
pip install -r requirements.txt

# —— 或者 —— 以可编辑模式安装为软件包
pip install -e .

# 配置 API 设置
cp config.py.example config.py
# 编辑 config.py，填入你的 API 密钥和偏好设置
```

---

## ⚙️ 配置

```python
# config.py
OPENAI_API_KEY = "your-api-key"
OPENAI_BASE_URL = None  # 或 Qwen/Azure 的自定义端点

LLM_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # 业界领先的检索效果
```

由于 SimpleMem 可与任意 OpenAI 兼容端点通信，你只需设置 `OPENAI_BASE_URL`，即可将其指向任何兼容的模型提供方。

### 关键设置

| 设置项 | 默认值 | 说明 |
|:--|:--|:--|
| `LLM_MODEL` | `gpt-4.1-mini` | 用于记忆构建与答案生成的 LLM 模型 |
| `EMBEDDING_MODEL` | `Qwen/Qwen3-Embedding-0.6B` | 用于语义搜索的向量模型 |
| `SEMANTIC_TOP_K` | `25` | 语义（向量）搜索返回的最大条目数 |
| `KEYWORD_TOP_K` | `5` | 关键词（BM25）搜索返回的最大条目数 |
| `STRUCTURED_TOP_K` | `5` | 结构化元数据搜索返回的最大条目数 |
| `ENABLE_PLANNING` | `True` | 是否启用检索的多查询规划 |
| `ENABLE_REFLECTION` | `True` | 是否启用基于反思的补充检索 |
| `MAX_REFLECTION_ROUNDS` | `2` | 最大反思轮数 |
| `ENABLE_PARALLEL_PROCESSING` | `True` | 是否启用并行记忆构建 |
| `ENABLE_PARALLEL_RETRIEVAL` | `True` | 是否启用并行查询执行 |

---

## 🧠 工作原理

### 三阶段流水线

#### 1. 语义结构化压缩

原始对话经过处理被提炼为原子记忆条目。每个条目包含：
- **无损重述** —— 已消解指代、自包含的事实
- **绝对时间戳** —— 所有相对时间均转换为绝对时间
- **结构化元数据** —— 人物、实体、地点、主题、关键词
- **语义向量** —— 用于相似度搜索的向量表示

#### 2. 在线语义综合

在记忆构建过程中，会话内相关的上下文会被合并为统一的抽象表示，从而主动（而非等到查询时）消除冗余。

#### 3. 意图感知的检索规划

当提出问题时，系统会：
1. 分析查询意图
2. 规划多种检索策略（语义、关键词、结构化）
3. 反思已检索到的上下文是否充分
4. 生成简洁、准确的答案

---

## 📝 引用

如果你的研究使用了 SimpleMem，请引用：

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

本项目基于 **MIT 许可证**开源，详情请参见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- 🔍 **向量模型**：[Qwen3-Embedding](https://github.com/QwenLM/Qwen) —— 业界领先的检索性能
- 🗄️ **向量数据库**：[LanceDB](https://lancedb.com/) —— 高性能列式存储
- 📊 **评测基准**：[LoCoMo](https://github.com/snap-research/locomo) —— 长上下文记忆评测框架
