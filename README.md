# KGPRAG: Advanced GraphRAG Engine for Multi-Hop QA

## 📅 更新日志 (Changelog)

*   **2025年12月22日**: 🚀 **F1 冲刺优化 (Precision & Recall Boost)**:
    *   **Recall 暴力提升**:
        *   📉 `TRUST_THRESHOLD`: **0.2 -> 0.01**。大幅降低剪枝门槛，确保在 HotpotQA 小空间检索中不漏掉任何微弱但关键的线索。
        *   📡 `DEFAULT_BEAM_WIDTH`: **5 -> 8**。增加搜索广度，支持并行追踪更多潜在推理路径。
        *   🛡️ `MIN_CANDIDATES_KEEP`: **3 -> 5**。在小空间模式下强制保留一半文档，防止全军覆没。
    *   **Precision 与格式规范化**:
        *   📝 **Answer-only 契约**: 强制 Prompt 输出格式为严格的 `Answer: <final answer>`，禁止输出推理过程和啰嗦句子，彻底解决 "Sem-Match 但 F1 低" 的冤案。
        *   🧹 **通用后处理 (Universal Post-processing)**: 实现了智能提取逻辑，自动截取最后一行 `Answer:` 并清洗符号，兼容所有问题类型。
        *   🔍 **增强型 Yes/No 检测**: 修复了 `_is_yes_no_question` 的漏检问题（新增 "same nationality/type" 等模式），确保正确应用格式化 Prompt。

*   **2025年12月21日**: 实施核心优化方案，包括：
    *   **自适应搜索域策略**: 根据搜索空间大小动态调整检索方式，解决 HotpotQA Distractor 模式下的拒答问题。
    *   **鲁棒图谱连通性**: 扩展 `HARD_EDGE_ENTITY_TYPES` (新增 Event, Product, Award, Concept 等类型) 并降低 `MIN_ENTITY_NAME_LENGTH`，修复图谱中的“断桥”，提升多跳推理能力。
    *   **多级容错机制**: 引入“豁免剪枝 (Force Keep)”和“Fallback 安全网”，提升系统在极端情况下的召回和稳定性。
## 🚀 核心创新与优化方案 (Key Innovations)

针对基线模型（如 KG2RAG）的不足，本项目实施了以下三大核心优化策略：

### 1. 自适应搜索域策略 (Context-Aware Retrieval)
传统的 RAG 往往采用“一刀切”的向量检索（ANN），在文档数较少但干扰性极强的场景（如 HotpotQA Distractor）下容易出现“灯下黑”。我们根据搜索空间大小动态调整策略：

*   **🌐 全开放空间 (Open Space)**: 当无文档限制时，采用高效的 **HNSW 向量索引** 进行全库检索（Top-K），保证大规模检索速度。
*   **🎯 受限小空间 (Constrained Small Space)**: 当候选文档少于阈值（如 20 个）时，启用 **"全量加载 + 重排 (Full Load + Rerank)"** 模式。
    *   **机制**: 跳过有损的向量检索，强制加载所有候选文档。
    *   **优势**: 利用 Cross-Encoder (FlagReranker) 的高精度进行细粒度排序，彻底消除检索召回阶段的漏报（False Negatives）。

### 2. 鲁棒的图谱连通性 (Robust Graph Connectivity)
为了修复图谱推理中的“断桥”现象，我们对图构建配置进行了精细调优：
*   **实体类型扩展**: 将硬边（Hard Edge）构建范围从仅 Person/Org/Loc 扩展至 **Event, Product, Award, Concept** 等 12 类，确保非实体类线索也能形成通路。
*   **短实体召回**: 将最小实体长度阈值从 4 降至 **2**，有效召回 "Wu", "Li", "AI" 等关键短实体。

### 3. 多级容错机制 (Tiered Resilience)
*   **豁免剪枝 (Force Keep)**: 在小空间模式下，即使所有文档的初始置信度（Trust Score）都低于阈值，系统也会强制保留 Top-K 文档进入图谱推理，防止过早被“误杀”。
*   **Fallback 安全网**: 如果图谱推理路径完全中断（返回空），系统会自动回退到基于 Rerank 的原始文档阅读模式，确保绝不出现“有文档却拒答 (I don't know)”的工程故障。

---
---


[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-green)](https://neo4j.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1-orange)](https://www.langchain.com/)

## 📖 项目简介 (Introduction)

**KGPRAG (Knowledge Graph-Powered Retrieval Augmented Generation)** 是一个面向多跳问答（Multi-Hop QA）的高级 GraphRAG 引擎。本项目针对 HotpotQA 等复杂数据集设计，旨在解决传统 RAG 在跨文档推理和长尾实体检索上的痛点。

本项目不仅实现了经典的 GraphRAG 架构（GLiNER 实体抽取 + REBEL 关系推理），更创新性地引入了 **"自适应搜索域策略 (Adaptive Search Scope Strategy)"**，在保证通用性的同时，显著提升了在限定搜索空间（如 Distractor Setting）下的召回率和推理精度。

---



## 🛠️ 项目架构与流程 (Workflow)

整个系统分为 **数据层 (Data Ingestion)** 和 **推理层 (Inference)** 两个解耦阶段。

### 第一阶段：全量图谱构建 (Offline Graph Construction)
**脚本**: `scripts/build_hotpot_global_kg.py`

此阶段负责将非结构化文本转化为结构化的知识图谱并存入 Neo4j 和 ChromaDB。
*   **Input**: `data/hotpot_dev_distractor_v1.json`
*   **Process**:
    1.  **Entity Extraction**: 使用 GLiNER 抽取实体。
    2.  **Relation Extraction**: 使用 REBEL 抽取关系三元组。
    3.  **Graph Indexing**: 构建 Chunk-Entity-Chunk 的多层图结构。
    4.  **Vector Indexing**: 生成 BGE-M3 嵌入向量。

> ⚠️ **注意**: 修改 `src/config.py` 后，**必须**重新运行此脚本以刷新图数据。

### 第二阶段：在线推理与评测 (Online Inference & Evaluation)
**脚本**: `evaluate.py` -> 调用 `src/retriever.py`

此阶段执行 QA 任务，并在 Distractor 数据集上计算 F1/EM 指标。
*   **Process**:
    1.  **Query Analysis**: 提取问题实体。
    2.  **Adaptive Retrieval**:
        *   检测 `doc_filter`（HotpotQA 提供的 10 个候选文档）。
        *   触发 **"小空间模式"**，全量加载 10 个文档。
    3.  **Graph Reasoning**:
        *   **Initial Scoring**: 使用 Reranker 打分，保留高置信度种子（Force Keep 生效）。
        *   **Graph Traversal**: 在 Neo4j 中游走，寻找 2-hop / 3-hop 的支撑证据（Bridge Documents）。
        *   **Path Pruning**: 基于多信号置信度（Rerank + PPR + Entity Overlap）剪枝。
    4.  **Fallback Check**: 若图搜索失败，触发兜底机制。
    5.  **Answer Generation**: 将筛选出的最佳路径和文档送入 LLM 生成答案。

---

## 🏃‍♂️ 快速开始 (Quick Start)

### 1. 环境准备
确保 Neo4j 服务已启动，且 `src/config.py` 中的数据库连接配置正确。

```bash
pip install -r requirements.txt
```

### 2. 构建全量图谱 (耗时较长)
这是为了应用最新的实体过滤规则（Config 优化），必须执行一次。

```bash
python scripts/build_hotpot_global_kg.py \
    --input data/hotpot_dev_distractor_v1.json \
    --persist_dir data/hotpotqa \
    --reset
```

### 3. 运行评测
执行推理引擎，验证优化效果。

```bash
python evaluate.py
```

### 4. 查看日志与结果
*   **日志**: `logs/kgprag_eval_*.log` (包含详细的推理路径和调试信息)
*   **结果**: `data/advanced_rag_results.jsonl` (包含预测答案和 F1 分数)

---

## ⚙️ 关键配置 (Configuration)

文件: `src/config.py`

| 参数 | 值 | 说明 |
| :--- |
| `SMALL_SPACE_THRESHOLD` | **20** | 小于此数量时触发全量加载 + Rerank 策略 |
| `HARD_EDGE_ENTITY_TYPES` | **12 Types** | 包含 Event, Product 等，确保图谱连通性 |
| `MIN_ENTITY_NAME_LENGTH` | **2** | 允许短实体（如人名缩写）建立连接 |
| `MIN_CANDIDATES_KEEP` | **5** | 小空间模式下强制保留的最少节点数 |
| `TRUST_THRESHOLD` | **0.01** | 节点进入下一跳的置信度门槛 |
| `DEFAULT_BEAM_WIDTH` | **8** | 搜索广度 |

---

## 📊 预期性能 (Expected Performance)

通过引入自适应搜索和图谱修复，本项目预期在 HotpotQA (Distractor) 上：
*   **Recall (召回率)**: 接近 100% (由全量加载保证)。
*   **F1 Score**: 显著超越 KG2RAG Baseline (0.663)，有望达到 0.70+。
*   **拒答率**: 大幅降低（由 Fallback 机制消除 False Negatives）。
