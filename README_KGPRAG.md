# KGPRAG 项目技术文档

## 1. 项目简介
本项目 (**KGPRAG**) 是一个基于**知识图谱增强检索生成 (GraphRAG)** 的问答系统，旨在解决多跳问答（Multi-hop QA）中的检索噪声和推理链断裂问题。
核心创新点在于**"低成本、高性能的构图策略"**与**"自适应多跳检索框架"**，通过结合 GLiNER（实体抽取）和 REBEL（关系抽取）构建三层知识图谱，并利用 Adaptive Multi-hop Search 算法进行精准推理。

---

## 2. 目录结构说明

```plaintext
F:\project\KGPRAG\
├── src/                     # 核心代码模块
│   ├── __init__.py
│   ├── config.py            # 全局配置文件 (模型路径、API Key、参数)
│   ├── engine.py            # 引擎入口，整合各模块 (支持内存/持久化模式)
│   ├── entity_extractor.py  # 实体与关系抽取 (GLiNER + REBEL)
│   ├── graph_builder.py     # 实时图谱构建逻辑
│   ├── graph_builder_offline.py # 离线批量图谱构建逻辑
│   ├── graph_store.py       # Neo4j 数据库操作接口
│   ├── vector_store.py      # ChromaDB 向量数据库操作接口 (内存模式)
│   ├── vector_store_persistent.py # ChromaDB 向量数据库操作接口 (持久化模式)
│   └── retriever.py         # 检索模块 (PPR + Adaptive Search + Hybrid Retrieval)
├── scripts/                 # 工具脚本
│   ├── build_hotpot_global_kg.py # 构建 HotpotQA 全局知识图谱
│   ├── build_index.py       # 通用离线建图脚本
│   └── query_index.py       # 通用在线查询脚本
├── util/                    # 工具库
│   ├── custom_logger.py     # 日志记录器
│   ├── hotpot_evaluate_v1.py# 评测脚本 (F1/EM 计算)
│   └── text_utils.py        # 文本处理工具
├── data/                    # 数据目录
├── evaluate.py              # 评测主程序
├── requirements.txt         # 依赖列表
└── logs/                    # 运行日志
```

---

## 3. 技术架构详解

### 3.1 离线建图流水线 (Offline Indexing Pipeline)

KGPRAG 采用**离线预计算**的方式构建大规模知识图谱，支持全量建图与增量更新。

#### 3.1.1 预处理与 Chunking
*   **输入**: JSON 格式文档列表 `[{"title": "...", "text": "..."}]`。
*   **Chunking**: 按固定大小（如 300 tokens）对长文档进行切分，同时保留文档标题作为上下文。
*   **去重**: 基于文档标题进行去重，避免重复索引。

#### 3.1.2 向量化 (Embedding)
*   **模型**: `BAAI/bge-m3`。
*   **策略**: 批量计算 Chunk 的 Embedding，用于后续的语义检索和图谱中的语义边构建。

#### 3.1.3 实体与关系抽取 (Information Extraction)
*   **实体抽取 (NER)**:
    *   **模型**: `urchade/gliner_medium-v2.1` (GLiNER)。
    *   **特点**: 使用 GLiNER 进行 Open-Schema NER，支持自定义实体类型（Person, Location, Organization, Event 等）。
    *   **优化**: 实现了批量推理 (`extract_entities_batch`)，大幅提升 GPU 利用率。
*   **关系抽取 (RE)**:
    *   **模型**: `Babelscape/rebel-large` (REBEL)。
    *   **特点**: 基于 Seq2Seq 生成 `(Head, Relation, Tail)` 三元组。
    *   **优化**: 实现了批量生成和解析逻辑。

#### 3.1.4 图谱构建 (Graph Construction)
构建了一个**三层混合图谱 (Hybrid Graph)**，存储于 **Neo4j**。

1.  **Level 1: 实体层 (Entity Layer)**
    *   **节点**: `Entity`。
    *   **边**: 
        *   `RELATION`: 由 REBEL 抽取的语义关系 (e.g., `(Obama)-[BORN_IN]->(Hawaii)`). 
        *   `MENTIONS`: Chunk 提及实体的关系 (e.g., `(Chunk1)-[MENTIONS]->(Obama)`). 
    *   **硬边构建 (Entity Bridge)**: 基于稀有实体共现建立 Chunk 间的硬连接 (`ENTITY_BRIDGE`)，用于连通多跳路径。

2.  **Level 2: 篇章层 (Passage Layer)**
    *   **节点**: `Chunk`, `Document`。
    *   **边**:
        *   `NEXT`: 文档内相邻 Chunk 的时序关系。
        *   `RELATED`: 基于 Embedding 相似度计算的语义边 (Semantic Edges)。仅连接相似度 > 0.7 的 Chunk 对。

3.  **Level 3: 摘要层 (Summary Layer)**
    *   **节点**: `Summary`。
    *   **构建**: 使用 RAPTOR 风格的自底向上聚类（或启发式摘要），构建层级摘要树。
    *   **边**: `HAS_SUMMARY` (Document -> Summary), `CONTAINS` (Summary -> Chunk/Summary)。


### 3.2 在线检索与推理 (Online Retrieval & Reasoning)

检索阶段采用了**自适应多跳检索 (Adaptive Multi-hop Search)** 框架，结合全局相关性与局部精准度。

#### 3.2.1 全局相关性计算 (Personalized PageRank)
*   **目的**: 在检索开始前，评估全局图中每个节点相对于 Query 的重要性。
*   **方法**: 
    1.  从 Query 中提取实体作为种子节点 (Seeds)。
    2.  在内存中构建由 `NEXT`, `RELATED`, `ENTITY_BRIDGE` 构成的稀疏图。
    3.  运行 **Personalized PageRank (PPR)** 算法。
*   **输出**: 全局每个 Chunk 的 PPR Score，作为后续检索的先验权重。

#### 3.2.2 混合初始检索 (Hybrid Initialization)
*   **Vector Search**: 使用 Query 向量在 ChromaDB 中检索 Top-K Chunks。
*   **Summary-Guided**: 从 Summary 树顶层向下检索，定位宏观相关的分支。
*   **Query-Entity**: 直接定位包含 Query 实体的 Chunk。

#### 3.2.3 自适应多跳扩展 (Adaptive Expansion)
采用 **Beam Search** 策略进行多跳推理：
1.  **当前节点**: 选择当前得分最高的节点。
2.  **扩展邻居**: 通过图谱扩展邻居节点 (Sequential, Semantic, Entity Bridge, Relation Path)。
3.  **混合检索补全 (Hybrid Retrieval)**: 如果图扩展邻居不足，使用当前上下文再次进行向量检索 (Vector Jump)，补充候选集。
4.  **打分与剪枝 (Scoring & Pruning)**: 计算**多信号可信度评分 (Multi-Signal Trust Score)**。

#### 3.2.4 多信号可信度评分 (Trust Score)
$$ Score = w_1 \cdot S_{Rerank} + w_2 \cdot S_{PPR} + w_3 \cdot S_{Entity} + w_4 \cdot S_{Path} + w_5 \cdot S_{Source} $$
*   **$S_{Rerank}$**: BGE-Reranker-v2-m3 计算的语义相关性 (权重最高)。
*   **$S_{PPR}$**: 全局 PPR 分数。
*   **$S_{Entity}$**: 节点文本对 Query 实体的覆盖率。
*   **$S_{Path}$**: 路径长度惩罚 (跳数越多分数越低)。
*   **$S_{Source}$**: 来源类型权重 (QueryEnt > EntBridge > Semantic > ...)。

#### 3.2.5 答案生成
*   收集所有选中节点的文本作为 Context。
*   调用 LLM (DeepSeek/GPT-4/Llama3) 生成最终答案，并附带推理路径。

---

## 4. 关键技术栈
*   **语言**: Python 3.10+
*   **图数据库**: Neo4j (存储图结构)
*   **向量数据库**: ChromaDB (存储 Embeddings)
*   **LLM 框架**: LangChain
*   **深度学习框架**: PyTorch, Hugging Face Transformers
*   **核心模型**:
    *   **LLM**: DeepSeek-V3 / Llama-3-8B
    *   **Embedding**: BAAI/bge-m3
    *   **Rerank**: BAAI/bge-reranker-v2-m3
    *   **NER**: urchade/gliner_medium-v2.1
    *   **RE**: Babelscape/rebel-large
*   **图算法库**: NetworkX

---

## 5. 快速开始

### 5.1 环境准备
```bash
# 安装依赖
pip install -r requirements.txt

# 启动 Neo4j (Docker)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest
```

### 5.2 离线建图
```bash
# 全量构建 HotpotQA 图谱
python scripts/build_hotpot_global_kg.py
```

### 5.3 运行评测
```bash
# 运行 HotpotQA 评测
python evaluate.py
```