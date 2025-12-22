# KGPRAG: 知识图谱增强 RAG 系统 (Knowledge Graph-Augmented RAG)

本文档详细介绍了 KGPRAG 系统的技术实现细节。该系统采用混合方法，结合了向量检索与三层知识图谱，以执行多跳推理并实现高精度的问答。

## 1. 系统架构 (System Architecture)

核心架构包含三个主要阶段：
1.  **图谱构建 (Graph Construction - Offline):** 将原始文档处理成丰富的知识图谱。
2.  **索引 (Indexing):** 存储向量嵌入（Embedding）以支持快速初始检索。
3.  **推理 (Inference - Online):** 执行自适应多跳检索与答案生成。

---

## 2. 源代码解析 (Source Code Analysis)

### `src/config.py` - 全局配置中心
*   **作用**: 集中管理所有系统参数，包括模型选择（LLM, Embedding, Reranker）、Neo4j 连接信息、检索参数（Beam Width, Hops）以及实体抽取的阈值和停用词。
*   **关键参数**:
    *   `DEFAULT_BEAM_WIDTH`: 控制每跳保留的候选路径数。
    *   `TRUST_THRESHOLD`: 剪枝的最低置信度阈值。
    *   `HARD_EDGE_ENTITY_TYPES`: 定义哪些实体类型用于构建硬边。

### `src/engine.py` - 核心引擎与编排器
*   **作用**: 系统的总指挥，协调各模块（检索器、图存储、向量存储）完成端到端的 RAG 流程。
*   **核心类 `AdvancedRAGEngine`**:
    *   `__init__`: 初始化所有组件，支持内存模式和持久化模式。
    *   `query`: 接收用户问题，调用 `retriever.search` 获取上下文，根据问题类型（Yes/No 或普通）选择 Prompt，调用 LLM 生成答案。
    *   `_post_process_answer`: 规范化 LLM 输出（如统一 Yes/No 格式）。

### `src/entity_extractor.py` - 实体与关系抽取
*   **作用**: 封装了 GLiNER 和 REBEL 模型，负责从文本中提取结构化信息。
*   **核心功能**:
    *   `extract_entities`: 使用 GLiNER 识别命名实体，并应用归一化和过滤规则。
    *   `extract_relations`: 使用 REBEL 生成 (Subject, Relation, Object) 三元组。
    *   `extract_query_entities`: 专门提取用户 Query 中的实体，作为检索的种子节点。

### `src/graph_builder_offline.py` - 离线图构建器
*   **作用**: 执行繁重的图谱构建任务，将文档集转化为知识图谱。
*   **核心逻辑**:
    *   **Pipeline**: 预处理 -> Embedding -> 实体/关系抽取 -> 摘要树生成 -> 语义边计算 -> 实体桥构建 -> 写入 Neo4j。
    *   **创新点**: 包含复杂的实体桥 (`:ENTITY_BRIDGE`) 构建算法，基于稀有实体共现来创建硬连接。

### `src/graph_store.py` - Neo4j 图数据库接口
*   **作用**: 封装所有与 Neo4j 的交互操作。
*   **核心功能**:
    *   `write_chunks/summaries/edges`: 批量写入各类节点和关系。
    *   `expand_node`: 检索中最重要的函数，给定一个节点，返回其所有类型的邻居（顺序、语义、实体桥、关系路径），支持多路召回。

### `src/retriever.py` - 多跳检索器
*   **作用**: 实现核心的检索算法。
*   **核心算法 `MultiHopRetriever`**:
    *   `search`: 执行自适应 Beam Search。根据文档数量自动切换“小空间全量模式”和“大空间检索模式”。
    *   `compute_trust_score`: 多维度打分函数，融合 Reranker 分数、PPR、实体覆盖率等信号。
    *   **混合策略**: 在图游走的同时，动态调用 `vector_store` 进行 Vector Jump，补充非连通的语义相关节点。

### `src/vector_store.py` - 向量数据库接口
*   **作用**: 封装 ChromaDB 的操作，管理 Embeddings。
*   **核心功能**:
    *   `similarity_search_with_score`: 基础的 KNN 检索。
    *   `hybrid_retrieval`: 使用“Query + 上下文”进行增强检索。
    *   `summary_guided_retrieval`: 自顶向下的检索策略，先找摘要再找 Chunk。

### `src/vector_store_persistent.py` - 持久化向量存储
*   **作用**: `VectorStore` 的持久化版本，用于加载预先构建好的 ChromaDB 索引，避免每次重启都重新 Embedding。

---

## 3. 详细技术实现 (Technical Implementation Details)

### A. 图谱构建 (`src/graph_builder_offline.py`)

系统构建了一个 **三层知识图谱 (3-Layer Knowledge Graph)** 以捕捉不同粒度的信息：
*   **第 1 层 (Tree):** 文档 (Document) -> 二级摘要 (L2) -> 一级摘要 (L1) -> 文本块 (Chunk)
*   **第 2 层 (Passage):** 文本块 <-> 文本块 (通过 顺序/语义/实体桥 连接)
*   **第 3 层 (Entity):** 文本块 -> 实体, 实体 -> 实体 (关系三元组)

**构建流程详解：**

1.  **文本预处理 (Text Preprocessing):** 将文档切分为固定大小的文本块 (Chunks)。
2.  **向量嵌入 (Vector Embedding):**
    *   模型: `BAAI/bge-m3` (默认)
    *   将每个文本块映射到高维向量空间。
3.  **实体抽取 (Entity Extraction - `GLiNER`):**
    *   模型: `urchade/gliner_medium-v2.1`
    *   从每个 Chunk 中抽取实体 (人名、地名、组织机构等)。
    *   **过滤机制:** 剔除停用词、过短实体及纯数字（保留年份）。
4.  **关系抽取 (Relation Extraction - `REBEL`):**
    *   模型: `Babelscape/rebel-large`
    *   对包含 >= 2 个实体的 Chunk，抽取结构化三元组 (主语, 关系, 宾语)。
5.  **摘要树构建 (Summary Tree Construction):**
    *   **聚类:** 使用 K-Means 对文档内的 Chunk 进行聚类。
    *   **摘要生成:** 为每个聚类生成 L1 摘要，为整篇文档生成 L2 摘要（支持启发式或 LLM 生成）。
    *   **链接:** 创建 `CONTAINS_SUMMARY` 和 `CONTAINS_CHUNK` 边，形成层级树状结构。
6.  **语义边计算 (Semantic Edge - `:RELATED`):**
    *   计算所有 Chunk 嵌入向量间的余弦相似度。
    *   将相似度 > 阈值 (默认 0.7) 的 Chunk 连接到其 Top-K (默认 3) 邻居。
7.  **实体桥构建 (Entity Bridge - `:ENTITY_BRIDGE`) [核心创新]:**
    *   **目的:** 连接共享稀有且关键实体的远距离 Chunk，实现图谱上的“瞬移”能力，解决多跳断链问题。
    *   **逻辑:**
        1.  **倒排索引:** 建立 `实体 -> Chunk列表` 的映射。
        2.  **频率过滤:** 仅考虑出现次数在 `MIN_ENTITY_OCCURRENCES` (2) 到 `MAX_ENTITY_OCCURRENCES` (50) 之间的实体。剔除“国家”、“人”等高频通用词，防止产生超级节点。
        3.  **连边策略:**
            *   **全连接:** 若实体出现的 Chunk 数 <= 20，则这些 Chunk 两两全连接。
            *   **采样连接:** 若 > 20，则进行随机采样连接，控制边密度。
    *   **存储:** 在 Neo4j 中直接存储为 Chunk 间的 `:ENTITY_BRIDGE` 关系。

---

### B. 检索流水线 (`src/retriever.py`)

`MultiHopRetriever` 实现了 **自适应 Beam Search (Adaptive Beam Search)** 策略，其中包含了核心创新机制：**可信子图选择 (Trusted Subgraph Selection)**。

**1. 可信子图选择机制 (Trusted Subgraph Selection)** [核心创新]
本系统并非简单地检索Top-K文档，而是在巨大的知识图谱中，动态“生长”并筛选出一棵**高信噪比的推理子图**。该机制通过多方式评判来识别噪声，确保证据链的纯净与鲁棒性。

*   **多方式评判 (Multi-Way Evaluation):** `compute_trust_score` 函数融合了 5 种异构信号来综合评判节点的可信度：
    1.  **语义相关性 (Semantic Relevance, 50%):** 利用 Cross-Encoder (`FlagReranker`) 进行深层语义匹配，这是最强的信号。
    2.  **拓扑重要性 (Topological Importance, 10%):** 利用 **PPR (Personalized PageRank)** 算法计算节点在局部图谱中的中心度。
    3.  **实体覆盖率 (Entity Coverage, 15%):** 计算节点包含多少查询实体，作为显式的关键词匹配信号。
    4.  **路径长度惩罚 (Path Penalty, 12%):** 优先选择推理路径较短的证据，防止过深推理引入噪声。
    5.  **来源类型先验 (Source Type Prior, 13%):** 对不同类型的边赋予不同的置信度（例如，硬性实体桥 `EntBridge` > 软性语义边 `Sem`）。

*   **噪声剔除与高可信保留 (Noise Elimination & Retention):**
    *   **动态剪枝 (Dynamic Pruning):** 在 Beam Search 的每一跳，系统会计算所有候选节点的 `Trust Score`。任何低于 `TRUST_THRESHOLD` 的节点被视为噪声直接剔除，防止错误路径扩散。
    *   **智能兜底 (Smart Fallback):** 在小样本空间（如 Distractor 设置）下，引入 `Force Keep` 机制，强制保留前 `MIN_CANDIDATES_KEEP` 个最佳候选，防止因阈值过高导致召回失败。

通过这种机制，系统最终返回的 `final_selected_nodes` 和 `best_path` 本质上就是从全量数据中提炼出的**“高可信推理子图”**。

**2. 自适应搜索空间策略 (Context-Aware Search Strategy)**
系统根据候选文档集的大小（如 HotpotQA Distractor vs. Full Wiki）动态调整策略：
1.  **小空间模式 (<= 20 文档):**
    *   策略: **全量加载 + 重排 (Full Load & Rerank)**。
    *   理由: 在小范围内，向量检索容易漏掉微弱的多跳线索。全量评分最安全。
2.  **大/开放空间模式:**
    *   策略: **向量检索 + 摘要引导**。
    *   理由: 利用 Embedding 快速定位初始候选，或通过摘要树下钻。

**3. Beam Search 循环 (Multi-Hop)**
1.  **启动:** 选取 Top-K (Beam Width) 个初始节点。
2.  **迭代 (最大跳数 Max Hops = 3):**
    *   **剪枝 (Pruning):** 保留 `Trust Score > TRUST_THRESHOLD` 的节点。(在小空间模式下，强制保留前 `MIN_CANDIDATES_KEEP` 名)。
    *   **扩展 (Expansion - `graph_store.expand_node`):**
        *   **顺序边:** 上下文 Chunk (`:NEXT`)。
        *   **语义边:** 相似 Chunk (`:RELATED`)。
        *   **实体桥 (Entity Bridge):** 基于稀有实体共现的硬边 (`:ENTITY_BRIDGE`)。**优先级最高**。
        *   **关系路径:** 通过 REBEL 三元组扩展 (`Chunk -> Entity -> Entity -> Chunk`)。
    *   **向量跳跃 (Vector Jump):** 如果图谱邻居不足，利用当前路径的上下文进行新一轮向量检索，寻找“隐形”关联。
    *   **重排 (Reranking):** 使用 `FlagReranker` 对所有新候选打分并更新 Frontier。

---

### C. 答案生成 (`src/engine.py`)

1.  **上下文组装:** 拼接最终 Beam 中排名最高的节点文本。
2.  **Prompt 工程:**
    *   **Yes/No 分类器:** 自动检测是否为是非题。
    *   **专用 Prompt:**
        *   *Yes/No 类:* 要求先进行简短推理，然后严格输出 "yes" 或 "no"。
        *   *标准类:* 强制要求极其简练的实体级答案 (日期、人名)，禁止输出完整句子，以最大化 F1 分数。
3.  **后处理:** 标准化答案格式 (例如 "Answer: yes" -> "yes") 以匹配评测标准。
