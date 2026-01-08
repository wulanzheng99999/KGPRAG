# KGPRAG 使用说明（README_KGPRAG）

本文档以当前代码为准，补充完整流程与使用方法，并重点细化 A. 图谱构建（`src/graph_builder_offline.py`）。

---

## 1. 项目定位与整体流程

KGPRAG 是面向多跳问答（Multi-Hop QA）的 GraphRAG 引擎，整体分为两阶段：

- **离线阶段：全量构图与索引**  
  将原始文档转成三层知识图谱 + 向量索引（Neo4j + Chroma）。
- **在线阶段：多跳检索与回答生成**  
  检索候选 -> 多跳扩展 -> 可信度评分 -> 生成答案。

---

## 2. 快速开始（HotpotQA / 通用文档）

### 2.1 环境准备
- Python 3.10+
- Neo4j 5.x（连接信息在 `src/config.py`）
- 依赖安装

```bash
pip install -r requirements.txt
```

> 模型路径支持环境变量覆盖：`EMBED_MODEL`、`RERANKER_MODEL`、`GLINER_MODEL`、`REBEL_MODEL`、`LOCAL_MODEL_DIR`。  
> LLM 使用 OpenAI 兼容接口，默认指向本地 Ollama（`OPENAI_BASE_URL`）。

### 2.2 构建 HotpotQA 全量图谱（推荐）

```bash
python scripts/build_hotpot_global_kg.py \
  --input data/hotpot_dev_distractor_v1.json \
  --persist_dir data/hotpotqa \
  --reset
```

可选参数：
- `--use_llm_summary`：用 LLM 生成摘要（默认启发式摘要）
- `--skip_existing`：断点续建

### 2.3 构建通用文档图谱

```bash
python scripts/build_index.py \
  --input data/documents.json \
  --persist_dir ./index \
  --reset
```

输入格式：

```json
[
  {"title": "Doc A", "text": "...."},
  {"title": "Doc B", "text": "...."}
]
```

### 2.4 评测（HotpotQA）

```bash
python evaluate.py
```

### 2.5 在线检索（交互/单次查询）

```bash
python scripts/query_index.py --persist_dir ./index --query "your question"
# 或
python scripts/query_index.py --persist_dir ./index --interactive
```

---

## 3. A. 图谱构建（`src/graph_builder_offline.py`）

### 3.1 输入与输出

- **输入**：`documents: List[{"title": str, "text": str}]`
- **输出**：
  - Neo4j 图数据（Chunk、Summary、Entity、关系边）
  - Chroma 向量索引（Chunk + Summary）
  - `doc_cache.json`（chunk_id -> {text, title}）
  - HotpotQA 专用映射（见 `scripts/build_hotpot_global_kg.py`）

### 3.2 三层图谱结构

- **Tree 层（层级摘要树）**
  - `Document -[:HAS_SUMMARY]-> Summary`
  - `Summary -[:CONTAINS]-> Summary/Chunk`
  - `Document -[:CONTAINS]-> Chunk`
- **Passage 层（段落/语义连接）**
  - `Chunk -[:NEXT]-> Chunk`（同文档内顺序）
  - `Chunk -[:RELATED]-> Chunk`（语义相似）
- **Entity 层（实体与关系）**
  - `Chunk -[:MENTIONS]-> Entity`
  - `Entity -[:RELATION]-> Entity`（REBEL）
  - `Chunk -[:ENTITY_BRIDGE]-> Chunk`（稀有实体桥）

### 3.3 构建步骤（与代码一致）

**Step 1/5：预处理文档**  
- 读取所有 `text/title`，形成 doc 列表。

**Step 2/5：文档级 Embedding（批量）**  
- 使用 `vector_store.embed_documents(texts)`。  
- 该结果在当前流程中不直接参与语义边/摘要（真正用于后续的是 Chunk embedding）。

**Step 3/5：文档级实体抽取（GLiNER）**  
- `extract_entities_batch` 批量处理，文本截断到 3000 字符。  
- 过滤停用词、过短实体与纯数字（保留年份）。

**Step 4/5：文档级关系抽取（REBEL）**  
- 仅对实体数 ≥2 的文档执行（早退优化）。  
- 文本截断到 512 字符，批量抽取三元组。

**Step 5/5：Chunk 组装（启用分块）**  
- 分块策略（滑窗 + 句边界回退）：

```text
chunk_size = 500
overlap = 100
cut at [. ! ? \n] within 50 chars (if possible)
```

- 对每个 Chunk：
  - 分配 `chunk_id`，写入 `doc_cache`
  - **实体过滤**：仅保留在当前 chunk 文本中出现的实体  
  - **关系过滤**：仅保留 source/target 都出现在当前 chunk 的关系  
  - 仅在**同一文档内部**串联 `prev_id`，用于 `NEXT` 边

**Step 5.1：Chunk Embedding 补算**  
- 对所有 Chunk 文本重新计算 embedding（语义边与摘要聚类都基于它）。

**Step 6：摘要树构建（Summary Tree）**  
- 按文档聚合 chunk  
- `n_clusters = max(1, min(len(chunks)//3, 5))`  
- 对每个簇生成 **L1 Summary**  
  - 默认启发式摘要（前 1~3 句 / 300 字）  
  - `--use_llm_summary` 时调用 LLM  
- 若 L1 Summary > 1，则生成 **L2 Summary**

**Step 7：写入向量库（Chroma）**  
- 写入所有 Chunk + Summary  
- `metadata` 中标注 `doc_id / title / type`

**Step 8：语义边构建（Top-K KNN）**  
参数（代码固定）：  
- `TOP_K_NEIGHBORS = 3`  
- `MIN_SIM_THRESHOLD = 0.5`  
- `BATCH_SIZE_KNN = 1000`  
流程：  
1) 归一化 embedding  
2) 批量点积计算相似度矩阵  
3) 为每个 Chunk 取 Top-K 且大于阈值  
4) 写入 `:RELATED`（Neo4j 中以无向边存储）

**Step 8.5：实体桥硬边（ENTITY_BRIDGE）**  
流程：  
- 倒排索引：entity -> chunk_ids  
- 频率过滤：`MIN_ENTITY_OCCURRENCES` ~ `MAX_ENTITY_OCCURRENCES`  
- 若出现次数 ≤ `MAX_CHUNKS_PER_ENTITY_FOR_FULL_CONNECT`：全连接  
- 否则采样连接（上限 `MAX_EDGES_PER_ENTITY`）

**Step 9：写入 Neo4j**  
- `write_chunks`：Chunk + Entity + RELATION + NEXT  
- `write_summaries`：Summary + CONTAINS/HAS_SUMMARY  
- `write_semantic_edges`：RELATED  
- `write_entity_bridge_edges`：ENTITY_BRIDGE

### 3.4 关键参数（来自 `src/config.py`）

- `BATCH_SIZE`：批量推理大小（影响 GLiNER/REBEL/Embedding）
- `HARD_EDGE_ENTITY_TYPES`：实体桥允许的类型
- `MIN_ENTITY_OCCURRENCES / MAX_ENTITY_OCCURRENCES`
- `MAX_CHUNKS_PER_ENTITY_FOR_FULL_CONNECT / MAX_EDGES_PER_ENTITY`
- `MIN_ENTITY_NAME_LENGTH`
- 分块策略固定：`chunk_size=500, overlap=100`
- 语义边策略固定：`K=3, min_sim=0.5`

### 3.5 构建产物（HotpotQA）

`scripts/build_hotpot_global_kg.py` 在构建完成后额外生成：

- `doc_cache.json`：`chunk_id -> {text, title}`
- `sample_doc_mapping.json`：`sample_id -> [chunk_ids]`
- `title_to_doc_id.json`：`title -> [chunk_ids]`

这些文件是 `evaluate.py` 的必需输入。

---

## 4. B. 在线检索与推理流程（概览）

在线阶段主要逻辑在 `src/retriever.py` + `src/graph_store.py`：

- **候选获取**  
  - 有 `doc_filter` 且数量小于 `SMALL_SPACE_THRESHOLD` → 全量加载 + rerank  
  - 否则向量检索
- **多跳扩展（GraphStore.expand_node）**
  - `Seq`：NEXT
  - `SemHigh`：RELATED ≥ 0.70  
  - `SemLow`：0.55 ≤ RELATED < 0.70  
  - `EntMention`：共享实体  
  - `RelPath`：REBEL 关系路径  
  - `EntBridge`：ENTITY_BRIDGE  
  - `QueryEnt`：查询实体模糊匹配
- **评分（compute_trust_score）**
  - Reranker + PPR + 实体覆盖 + 路径惩罚 + 来源类型权重  
  - SemHigh / SemLow 进一步用 `edge_score` 降权
- **多样性过滤（Diversity Filter V2）**
  - 以 `doc_title` 和 `source_type` 防止同质化

---

## 5. 使用注意事项

- **修改分块/实体/语义边参数后必须重建图谱**（`--reset`）。  
- 语义边计算为 O(N²)，CPU 计算时间长属正常；GPU 显存对该步骤影响有限。  
- HotpotQA 的评测必须与构建时的 `persist_dir` 一致，且映射文件齐全。
