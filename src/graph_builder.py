"""
图构建模块：三层图谱构建
"""
from __future__ import annotations

from typing import List, Dict

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.config import (
    LLM_MODEL,
    HARD_EDGE_ENTITY_TYPES, MIN_ENTITY_OCCURRENCES, MAX_ENTITY_OCCURRENCES,
    MAX_CHUNKS_PER_ENTITY_FOR_FULL_CONNECT, MAX_EDGES_PER_ENTITY,
    MIN_ENTITY_NAME_LENGTH
)
from src.entity_extractor import EntityExtractor
from src.graph_store import GraphStore
from src.vector_store import VectorStore


class GraphBuilder:
    """三层图谱构建器"""
    
    def __init__(self, entity_extractor: EntityExtractor, graph_store: GraphStore, vector_store: VectorStore):
        self.entity_extractor = entity_extractor
        self.graph_store = graph_store
        self.vector_store = vector_store
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        self.doc_cache: Dict[str, Dict] = {}
    
    def _summarize_text(self, text: str, hint: str) -> str:
        """LLM 增强摘要"""
        if not text:
            return ""
        
        if len(text) < 500:
            clean_text = text.replace("\n", " ").strip()
            return f"[{hint}] " + clean_text

        try:
            from langchain_core.messages import HumanMessage
            prompt = f"Please provide a concise summary of the following text, focusing on the topic '{hint}'.\n\nText:\n{text[:4000]}"
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return f"[{hint}] {response.content}"
        except Exception as e:
            print(f"⚠️ Summary LLM Error: {e}, falling back to heuristic.")
            clean_text = text.replace("\n", " ").strip()
            return f"[{hint}] " + clean_text[:300] + "..."
    
    def _split_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """简单的滑动窗口分块"""
        if not text: return []
        if len(text) <= chunk_size: return [text]
        chunks = []
        start = 0
        text_len = len(text)
        while start < text_len:
            end = min(start + chunk_size, text_len)
            if end < text_len:
                lookback = min(50, end - start)
                for i in range(lookback):
                    idx = end - i - 1
                    if text[idx] in ['.', '!', '?', '\n']:
                        end = idx + 1
                        break
            chunk = text[start:end].strip()
            if chunk: chunks.append(chunk)
            if end == text_len: break
            start = end - overlap
            if start >= end: start = end
        return chunks

    def build(self, documents: List[Dict]):
        """
        三层图谱构建：
        Layer 1 (Tree): Document -> Summary(L2) -> Summary(L1) -> Chunk
        Layer 2 (Passage): Chunk <-> Chunk (NEXT, RELATED)
        Layer 3 (Entity): Chunk -> Entity, Entity -> Entity (RELATION)
        """
        if not documents:
            return

        lc_docs: List[Document] = []
        ids: List[str] = []
        all_chunks: List[Dict] = []

        # Step 1: 批量预处理 (提取所有文本)
        texts = [doc.get("text", "") for doc in documents]
        titles = [doc.get("title", "") for doc in documents]
        
        # Step 2: 批量 Embedding (GPU 高效利用)
        embeddings = self.vector_store.embed_documents(texts)
        
        # Step 3: 批量实体抽取 (GPU 高效利用)
        all_entities = self.entity_extractor.extract_entities_batch(texts)
        
        # Step 4: 批量关系抽取 (GPU 高效利用) - 带早退优化
        # 只对实体数 >= 2 的文本进行关系抽取
        texts_for_rebel = []
        rebel_indices = []
        for i, ents in enumerate(all_entities):
            if len(ents) >= 2:  # 早退策略：实体少于2个则跳过REBEL
                texts_for_rebel.append(texts[i])
                rebel_indices.append(i)
        
        all_relations = [[] for _ in range(len(documents))]  # 初始化空列表
        if texts_for_rebel:
            rebel_results = self.entity_extractor.extract_relations_batch(texts_for_rebel)
            for idx, rels in zip(rebel_indices, rebel_results):
                all_relations[idx] = rels
        
        # Step 5: 组装 chunks
        global_chunk_idx = 0
        
        for i, doc in enumerate(documents):
            text = texts[i]
            title = titles[i]
            
            # 1. 切分
            text_chunks = self._split_text(text)
            
            doc_entities = all_entities[i]
            doc_relations = all_relations[i]
            
            prev_chunk_id = None
            
            for chunk_text in text_chunks:
                cid = f"chunk_{global_chunk_idx}"
                global_chunk_idx += 1
                
                self.doc_cache[cid] = {"text": chunk_text, "title": title}
                
                # 实体过滤
                chunk_ents = []
                chunk_text_lower = chunk_text.lower()
                for ent_name, ent_type in doc_entities.items():
                    if ent_name.lower() in chunk_text_lower:
                        chunk_ents.append({"name": ent_name, "type": ent_type})
                
                # 关系过滤
                chunk_rels = []
                for rel in doc_relations:
                    if rel["source"].lower() in chunk_text_lower and rel["target"].lower() in chunk_text_lower:
                        chunk_rels.append(rel)

                all_chunks.append({
                    "doc_title": title, 
                    "chunk_id": cid, 
                    "text": chunk_text, 
                    "embedding": None, # 稍后补算
                    "entities": chunk_ents,
                    "rebel_rels": chunk_rels,
                    "prev_id": prev_chunk_id, # 仅连接同文档上一块
                })
                
                prev_chunk_id = cid

                ids.append(cid)
                lc_docs.append(Document(
                    page_content=chunk_text, 
                    metadata={"doc_id": cid, "title": title, "type": "chunk"}
                ))
        
        # 补算 Chunk Embeddings
        chunk_texts = [c["text"] for c in all_chunks]
        chunk_embeddings = self.vector_store.embed_documents(chunk_texts)
        for i, c in enumerate(all_chunks):
            c["embedding"] = chunk_embeddings[i]

        # Step 2: 构建摘要树
        docs_map: Dict[str, List[Dict]] = {}
        for c in all_chunks:
            docs_map.setdefault(c["doc_title"], []).append(c)

        summary_nodes_batch = []
        summary_rels_batch = []
        
        for title, chunks in docs_map.items():
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

        # Step 3: 写入向量库
        self.vector_store.add_documents(lc_docs, ids=ids)

        # Step 4: 计算语义边 (Top-K KNN 而非阈值法) - 分批处理版
        semantic_rels = []
        TOP_K_NEIGHBORS = 3
        MIN_SIM_THRESHOLD = 0.5
        BATCH_SIZE_KNN = 500
        
        if len(all_chunks) > 1:
            try:
                embeddings = np.array([c["embedding"] for c in all_chunks], dtype=np.float32)
                norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / (norm + 1e-10)
                
                num_chunks = len(embeddings)
                print(f"     开始计算 {num_chunks} 个 Chunk 的语义邻居 (Batch Size: {BATCH_SIZE_KNN})...")
                import sys

                batch_count = 0
                for start_idx in range(0, num_chunks, BATCH_SIZE_KNN):
                    batch_count += 1
                    end_idx = min(start_idx + BATCH_SIZE_KNN, num_chunks)
                    
                    if batch_count % 10 == 0:
                        print(f"     [Progress] Processing batch {batch_count} ({start_idx}/{num_chunks})...", flush=True)

                    batch_embeddings = embeddings[start_idx:end_idx]
                    batch_sim = np.dot(batch_embeddings, embeddings.T)
                    
                    for i in range(len(batch_embeddings)):
                        global_idx = start_idx + i
                        sims = batch_sim[i]
                        sims[global_idx] = -1.0
                        
                        if len(sims) > TOP_K_NEIGHBORS:
                            k_indices = np.argpartition(sims, -TOP_K_NEIGHBORS)[-TOP_K_NEIGHBORS:]
                            k_indices = k_indices[np.argsort(sims[k_indices])[::-1]]
                        else:
                            k_indices = np.argsort(sims)[::-1]
                        
                        for target_idx in k_indices:
                            score = float(sims[target_idx])
                            if score >= MIN_SIM_THRESHOLD and global_idx < target_idx:
                                semantic_rels.append({
                                    "source": all_chunks[global_idx]["chunk_id"],
                                    "target": all_chunks[target_idx]["chunk_id"],
                                    "score": score,
                                })
                    del batch_sim
            except Exception as e:
                print(f"⚠️ Semantic Edge Calculation Failed: {e}")

        # Step 4.5: 计算实体共现硬边 (Entity Co-occurrence Hard Edges)
        hard_edges = []
        
        # 建立倒排索引
        entity_to_chunks = {}
        for chunk in all_chunks:
            for ent in chunk["entities"]:
                ent_name = ent["name"].lower()
                ent_type = ent["type"]
                if ent_type not in HARD_EDGE_ENTITY_TYPES:
                    continue
                if len(ent_name) < MIN_ENTITY_NAME_LENGTH:
                    continue
                entity_to_chunks.setdefault(ent_name, []).append(chunk["chunk_id"])
        
        # 生成硬边（带采样策略）
        import random
        random.seed(42)
        
        for ent_name, chunk_ids in entity_to_chunks.items():
            # 去重 + 排序（保证可复现性）
            unique_chunk_ids = sorted(list(set(chunk_ids)))
            n = len(unique_chunk_ids)
            
            if n < MIN_ENTITY_OCCURRENCES or n > MAX_ENTITY_OCCURRENCES:
                continue
            
            if n <= MAX_CHUNKS_PER_ENTITY_FOR_FULL_CONNECT:
                # 全连接
                for i in range(n):
                    for j in range(i + 1, n):
                        hard_edges.append({
                            "source": unique_chunk_ids[i],
                            "target": unique_chunk_ids[j],
                            "score": 1.0,
                        })
            else:
                # 采样连接
                k = min(5, n - 1)
                sampled_pairs = set()
                for i in range(n):
                    neighbors = random.sample([j for j in range(n) if j != i], k)
                    for j in neighbors:
                        pair = tuple(sorted([i, j]))
                        if pair not in sampled_pairs:
                            sampled_pairs.add(pair)
                            if len(sampled_pairs) >= MAX_EDGES_PER_ENTITY:
                                break
                    if len(sampled_pairs) >= MAX_EDGES_PER_ENTITY:
                        break
                for pair in sampled_pairs:
                    hard_edges.append({
                        "source": unique_chunk_ids[pair[0]],
                        "target": unique_chunk_ids[pair[1]],
                        "score": 1.0,
                    })

        # Step 5: 写入 Neo4j
        # 注意：语义边 (:RELATED) 和硬边 (:ENTITY_BRIDGE) 分开写入，避免 score 覆盖
        self.graph_store.write_chunks(all_chunks)
        self.graph_store.write_summaries(summary_nodes_batch, summary_rels_batch)
        self.graph_store.write_semantic_edges(semantic_rels)
        self.graph_store.write_entity_bridge_edges(hard_edges)  # 硬边独立写入
        
        return self.doc_cache
