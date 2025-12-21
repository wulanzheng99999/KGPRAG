"""
离线图构建模块：支持增量构建和无 LLM 摘要
"""
from __future__ import annotations

from typing import List, Dict, Set

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import (
    LLM_MODEL, BATCH_SIZE,
    HARD_EDGE_ENTITY_TYPES, MIN_ENTITY_OCCURRENCES, MAX_ENTITY_OCCURRENCES,
    MAX_CHUNKS_PER_ENTITY_FOR_FULL_CONNECT, MAX_EDGES_PER_ENTITY,
    MIN_ENTITY_NAME_LENGTH
)
from src.entity_extractor import EntityExtractor
from src.graph_store import GraphStore


class OfflineGraphBuilder:
    """离线三层图谱构建器"""
    
    def __init__(
        self, 
        entity_extractor: EntityExtractor, 
        graph_store: GraphStore, 
        vector_store,
        use_llm_summary: bool = False,
        logger = None
    ):
        self.entity_extractor = entity_extractor
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.use_llm_summary = use_llm_summary
        self.logger = logger
        
        if use_llm_summary:
            self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        else:
            self.llm = None
        
        self.doc_cache: Dict[str, Dict] = {}

    def log(self, message: str):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)
    
    def _summarize_text(self, text: str, hint: str) -> str:
        """
        摘要生成：可选 LLM 或纯启发式
        启发式摘要无需 LLM 调用，大幅降低索引成本
        """
        if not text:
            return ""
        
        # 启发式摘要（无 LLM 调用）
        if not self.use_llm_summary or len(text) < 500:
            clean_text = text.replace("\n", " ").strip()
            # 提取前几句作为摘要
            sentences = []
            for sep in ["。", ".", "！", "!", "？", "?"]:
                if sep in clean_text:
                    parts = clean_text.split(sep)
                    sentences = [p.strip() + sep for p in parts[:3] if p.strip()]
                    break
            
            if sentences:
                summary = "".join(sentences)
            else:
                summary = clean_text[:300]
            
            return f"[{hint}] {summary}"
        
        # LLM 摘要（可选）
        try:
            from langchain_core.messages import HumanMessage
            prompt = f"请用一句话总结以下内容，主题是'{hint}':\n{text[:2000]}"
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return f"[{hint}] {response.content}"
        except Exception as e:
            self.log(f"⚠️ Summary LLM Error: {e}")
            return f"[{hint}] " + text[:200] + "..."
    
    def build(
        self, 
        documents: List[Dict], 
        existing_chunk_ids: Set[str] = None,
        start_idx: int = 0
    ) -> Dict[str, Dict]:
        """
        三层图谱构建（支持增量）：
        Layer 1 (Tree): Document -> Summary(L2) -> Summary(L1) -> Chunk
        Layer 2 (Passage): Chunk <-> Chunk (NEXT, RELATED)
        Layer 3 (Entity): Chunk -> Entity, Entity -> Entity (RELATION)
        
        参数:
            documents: 文档列表 [{"title": str, "text": str}, ...]
            existing_chunk_ids: 已存在的 chunk ID 集合（用于增量构建）
            start_idx: 起始索引（用于增量构建）
        
        返回:
            doc_cache: {chunk_id: {"text": str, "title": str}, ...}
        """
        if not documents:
            return {}
        
        existing_chunk_ids = existing_chunk_ids or set()
        
        lc_docs: List[Document] = []
        ids: List[str] = []
        all_chunks: List[Dict] = []

        # Step 1: 批量预处理 (提取所有文本)
        texts = [doc.get("text", "") for doc in documents]
        titles = [doc.get("title", "") for doc in documents]
        
        self.log(f"  📝 Step 1/5: 预处理 {len(texts)} 个文档")
        
        # Step 2: 批量 Embedding (GPU 高效利用)
        self.log(f"  🔢 Step 2/5: 批量 Embedding...")
        embeddings = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="     Embedding"):
            batch_texts = texts[i : i + BATCH_SIZE]
            batch_embeddings = self.vector_store.embed_documents(batch_texts)
            embeddings.extend(batch_embeddings)
        
        # Step 3: 批量实体抽取 (GPU 高效利用)
        self.log(f"  🏷️ Step 3/5: 批量实体抽取...")
        all_entities = []
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc="     Entity Extraction"):
            batch_texts = texts[i : i + BATCH_SIZE]
            # extract_entities_batch 内部也支持批量，但为了进度条，我们这里手动分批调用
            # 注意：extract_entities_batch 本身会处理 list，所以这里调用它没问题
            batch_entities = self.entity_extractor.extract_entities_batch(batch_texts)
            all_entities.extend(batch_entities)
        
        # Step 4: 批量关系抽取 (GPU 高效利用) - 带早退优化
        self.log(f"  🔗 Step 4/5: 批量关系抽取...")
        # 只对实体数 >= 2 的文本进行关系抽取
        texts_for_rebel = []
        rebel_indices = []
        for i, ents in enumerate(all_entities):
            if len(ents) >= 2:  # 早退策略：实体少于2个则跳过REBEL
                texts_for_rebel.append(texts[i])
                rebel_indices.append(i)
        
        all_relations = [[] for _ in range(len(documents))]  # 初始化空列表
        if texts_for_rebel:
            self.log(f"     REBEL 处理 {len(texts_for_rebel)}/{len(documents)} 个文档 (早退优化)")
            rebel_results = []
            for i in tqdm(range(0, len(texts_for_rebel), BATCH_SIZE), desc="     Relation Extraction"):
                batch_texts = texts_for_rebel[i : i + BATCH_SIZE]
                batch_rels = self.entity_extractor.extract_relations_batch(batch_texts)
                rebel_results.extend(batch_rels)
            
            for idx, rels in zip(rebel_indices, rebel_results):
                all_relations[idx] = rels
        
        # Step 5: 组装 chunks
        self.log(f"  📦 Step 5/5: 组装图谱结构...")
        for i, doc in enumerate(tqdm(documents, desc="     Assembling Chunks")):
            cid = f"chunk_{start_idx + i}"
            
            # 跳过已存在的 chunk（增量模式）
            if cid in existing_chunk_ids:
                continue
            
            text = texts[i]
            title = titles[i]

            self.doc_cache[cid] = {"text": text, "title": title}
            
            gliner_ents = [{"name": k, "type": v} for k, v in all_entities[i].items()]

            all_chunks.append({
                "doc_title": title, 
                "chunk_id": cid, 
                "text": text, 
                "embedding": embeddings[i],
                "entities": gliner_ents,
                "rebel_rels": all_relations[i],
                "prev_id": f"chunk_{start_idx + i - 1}" if i > 0 else None,
            })

            ids.append(cid)
            lc_docs.append(Document(
                page_content=text, 
                metadata={"doc_id": cid, "title": title, "type": "chunk"}
            ))

        if not all_chunks:
            self.log("  📭 无新文档需要处理")
            return self.doc_cache

        # Step 6: 构建摘要树
        self.log(f"  🌳 构建摘要树...")
        docs_map: Dict[str, List[Dict]] = {}
        for c in all_chunks:
            docs_map.setdefault(c["doc_title"], []).append(c)

        summary_nodes_batch = []
        summary_rels_batch = []
        
        for title, chunks in tqdm(docs_map.items(), desc="     Building Summary Tree"):
            if not chunks:
                continue

            # Level 1 Summaries (聚类)
            vecs = np.array([c["embedding"] for c in chunks])
            n_clusters = max(1, min(len(chunks) // 3, 5))
            try:
                kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(vecs)
                labels = kmeans.labels_
            except Exception:
                labels = [0] * len(chunks)

            l1_summaries = []
            for c_idx in range(n_clusters):
                cluster_chunks = [chunks[j] for j, lbl in enumerate(labels) if lbl == c_idx]
                if not cluster_chunks:
                    continue
                
                cluster_text = " ".join([c["text"] for c in cluster_chunks])
                summary_text = self._summarize_text(cluster_text, hint=title)
                summary_id = f"summary_l1_{title}_{c_idx}"
                
                l1_summaries.append({"id": summary_id, "text": summary_text, "level": 1, "title": title})
                summary_nodes_batch.append({"id": summary_id, "text": summary_text, "level": 1, "doc_title": title})

                for chunk in cluster_chunks:
                    summary_rels_batch.append({
                        "source": summary_id, 
                        "target": chunk["chunk_id"], 
                        "type": "CONTAINS_CHUNK"
                    })
                
                ids.append(summary_id)
                lc_docs.append(Document(
                    page_content=summary_text, 
                    metadata={"doc_id": summary_id, "title": title, "type": "summary"}
                ))

            # Level 2 Summary (Root)
            if len(l1_summaries) > 1:
                l1_summary_text = "\n".join([s["text"] for s in l1_summaries])
                l2_summary_text = self._summarize_text(l1_summary_text, hint=f"Overall summary for {title}")
                l2_summary_id = f"summary_l2_{title}"

                summary_nodes_batch.append({
                    "id": l2_summary_id, 
                    "text": l2_summary_text, 
                    "level": 2, 
                    "doc_title": title
                })
                for l1_node in l1_summaries:
                    summary_rels_batch.append({
                        "source": l2_summary_id, 
                        "target": l1_node["id"], 
                        "type": "CONTAINS_SUMMARY"
                    })

                ids.append(l2_summary_id)
                lc_docs.append(Document(
                    page_content=l2_summary_text, 
                    metadata={"doc_id": l2_summary_id, "title": title, "type": "summary"}
                ))

        # Step 7: 写入向量库
        self.log(f"  💾 写入向量库: {len(lc_docs)} 个文档")
        self.vector_store.add_documents(lc_docs, ids=ids)

        # Step 8: 计算语义边 (Top-K KNN 而非阈值法)
        self.log(f"  🔗 计算语义边...")
        semantic_rels = []
        TOP_K_NEIGHBORS = 3  # 每个节点最多连接 K 个最相似的邻居
        MIN_SIM_THRESHOLD = 0.5  # 最低相似度阈值，避免噪声边
        
        if len(all_chunks) > 1:
            mat = np.array([c["embedding"] for c in all_chunks])
            sim_mat = cosine_similarity(mat)
            
            # 对每个节点，找 Top-K 最相似的邻居
            for i in tqdm(range(len(all_chunks)), desc="     Semantic Edges"):
                # 获取该节点与所有其他节点的相似度
                similarities = sim_mat[i]
                # 排除自己，获取 Top-K 索引
                top_k_indices = np.argsort(similarities)[::-1][1:TOP_K_NEIGHBORS+1]
                
                for j in top_k_indices:
                    score = float(similarities[j])
                    # 只添加超过最低阈值的边，且避免重复（i < j）
                    if score >= MIN_SIM_THRESHOLD and i < j:
                        semantic_rels.append({
                            "source": all_chunks[i]["chunk_id"],
                            "target": all_chunks[j]["chunk_id"],
                            "score": score,
                        })

        # Step 8.5: 计算实体共现硬边 (Entity Co-occurrence Hard Edges)
        self.log(f"  🔗 计算实体共现硬边...")
        hard_edges = []
        
        # Step 1: 建立倒排索引 (entity_name -> [chunk_id, ...])
        entity_to_chunks = {}  # entity_name (normalized) -> [chunk_id, ...]
        
        for chunk in all_chunks:
            for ent in chunk["entities"]:
                ent_name = ent["name"].lower()  # 归一化实体名
                ent_type = ent["type"]
                
                # 类型过滤：只考虑强类型实体
                if ent_type not in HARD_EDGE_ENTITY_TYPES:
                    continue
                
                # 长度过滤：避免短名误连
                if len(ent_name) < MIN_ENTITY_NAME_LENGTH:
                    continue
                
                entity_to_chunks.setdefault(ent_name, []).append(chunk["chunk_id"])
        
        # Step 2: 生成硬边（带采样策略防止边爆炸）
        entity_stats = {
            "total": len(entity_to_chunks),
            "filtered_low": 0,
            "filtered_high": 0,
            "used": 0,
            "full_connect": 0,
            "sampled": 0
        }
        
        import random
        random.seed(42)  # 可复现的随机采样
        
        for ent_name, chunk_ids in tqdm(entity_to_chunks.items(), desc="     Hard Edges"):
            # 去重 + 排序（保证可复现性，set 顺序在不同 Python 版本/运行中不一致）
            unique_chunk_ids = sorted(list(set(chunk_ids)))
            n = len(unique_chunk_ids)
            
            # 频率过滤
            if n < MIN_ENTITY_OCCURRENCES:
                entity_stats["filtered_low"] += 1
                continue
            if n > MAX_ENTITY_OCCURRENCES:
                entity_stats["filtered_high"] += 1
                continue
            
            entity_stats["used"] += 1
            
            # 决定连接策略
            if n <= MAX_CHUNKS_PER_ENTITY_FOR_FULL_CONNECT:
                # 全连接：当文档数 <= 20 时，所有文档两两连接
                entity_stats["full_connect"] += 1
                for i in range(n):
                    for j in range(i + 1, n):
                        hard_edges.append({
                            "source": unique_chunk_ids[i],
                            "target": unique_chunk_ids[j],
                            "score": 1.0,  # 硬边满分
                        })
            else:
                # 采样连接：当文档数 > 20 时，限制边数上限
                entity_stats["sampled"] += 1
                # 策略：每个 chunk 随机连接 k 个其他 chunks
                k = min(5, n - 1)  # 每个节点最多连接 5 个邻居
                sampled_pairs = set()
                
                for i in range(n):
                    # 为每个节点随机选择 k 个邻居
                    neighbors = random.sample([j for j in range(n) if j != i], k)
                    for j in neighbors:
                        pair = tuple(sorted([i, j]))
                        if pair not in sampled_pairs:
                            sampled_pairs.add(pair)
                            if len(sampled_pairs) >= MAX_EDGES_PER_ENTITY:
                                break
                    if len(sampled_pairs) >= MAX_EDGES_PER_ENTITY:
                        break
                
                # 添加采样得到的边
                for pair in sampled_pairs:
                    hard_edges.append({
                        "source": unique_chunk_ids[pair[0]],
                        "target": unique_chunk_ids[pair[1]],
                        "score": 1.0,
                    })
        
        self.log(f"  📊 实体统计: 总数={entity_stats['total']}, "
                 f"低频过滤={entity_stats['filtered_low']}, "
                 f"高频过滤={entity_stats['filtered_high']}, "
                 f"使用={entity_stats['used']} "
                 f"(全连接={entity_stats['full_connect']}, 采样={entity_stats['sampled']})")
        self.log(f"  🔗 生成 {len(hard_edges)} 条实体共现硬边 (ENTITY_BRIDGE)")

        # Step 9: 写入 Neo4j
        # 注意：语义边 (:RELATED) 和硬边 (:ENTITY_BRIDGE) 分开写入，避免 score 覆盖
        self.log(f"  💾 写入 Neo4j: {len(all_chunks)} chunks, {len(summary_nodes_batch)} summaries, "
                 f"{len(semantic_rels)} semantic edges, {len(hard_edges)} hard edges")
        self.graph_store.write_chunks(all_chunks)
        self.graph_store.write_summaries(summary_nodes_batch, summary_rels_batch)
        self.graph_store.write_semantic_edges(semantic_rels)
        self.graph_store.write_entity_bridge_edges(hard_edges)  # 硬边独立写入
        
        # 统计信息
        total_entities = sum(len(c["entities"]) for c in all_chunks)
        total_relations = sum(len(c["rebel_rels"]) for c in all_chunks)
        self.log(f"  📊 统计: {total_entities} 实体, {total_relations} 关系")
        
        return self.doc_cache
