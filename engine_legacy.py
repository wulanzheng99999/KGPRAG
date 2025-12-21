"""
Advanced GraphRAG Engine (生产版)
特性：
- 统一三层 KG：Document/Topic -> Chunk -> Entity
- 显式内存模式：使用 chromadb.EphemeralClient，避免持久化/文件锁
- 混合检索：Topic/Chunk 向量检索 + 图扩展（NEXT/RELATED/MENTIONS）
- 双模型抽取：GLiNER (实体) + REBEL (关系)
"""

from __future__ import annotations

import os
import uuid
from typing import List, Dict, Set

import numpy as np
import torch
import chromadb
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from neo4j import GraphDatabase
from gliner import GLiNER
from FlagEmbedding import FlagReranker
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import re

# ================= REBEL Model Configuration =================
REBEL_MODEL = os.environ.get("REBEL_MODEL", "Babelscape/rebel-large")

# ================= Configuration =================
# ！！！请在这里填入您的配置信息！！！
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-1408831cec78417d9a6024ac8e02dac4")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
GLINER_MODEL = os.environ.get("GLINER_MODEL", "urchade/gliner_medium-v2.1")

NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "9RP4s9YpWWSV:k3")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[System] Running on device: {DEVICE}")


class AdvancedRAGEngine:
    def __init__(self):
        print("🚀 初始化 AdvancedRAG 引擎 (双模型抽取: GLiNER + REBEL)...")

        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )

        # GLiNER: 实体抽取
        print(f"📦 加载实体模型: {GLINER_MODEL}")
        self.entity_model = GLiNER.from_pretrained(GLINER_MODEL)
        if DEVICE == "cuda":
            self.entity_model.to("cuda")
        
        # REBEL: 关系抽取
        print(f"📦 加载关系抽取模型: {REBEL_MODEL}")
        self.rebel_tokenizer = AutoTokenizer.from_pretrained(REBEL_MODEL)
        self.rebel_model = AutoModelForSeq2SeqLM.from_pretrained(REBEL_MODEL)
        if DEVICE == "cuda":
            self.rebel_model.to("cuda")
        self.rebel_model.eval()
        
        # Reranker
        print(f"📦 加载重排模型: {RERANKER_MODEL}")
        self.reranker = FlagReranker(RERANKER_MODEL, use_fp16=(DEVICE == "cuda"))

        self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        self.chroma_client = None
        self.vector_store = None
        self.doc_cache: Dict[str, Dict] = {}

        self.reset()

    def _normalize_entity(self, entity_text: str) -> str:
        """
        [思路B] 实体别名归一化
        去除冠词、标点，统一小写
        """
        # 去除开头的冠词
        normalized = re.sub(r'^(the|a|an)\s+', '', entity_text.lower())
        # 去除标点符号（保留字母数字空格）
        normalized = re.sub(r'[^\w\s]', '', normalized)
        return normalized.strip()

    def _extract_relations_rebel(self, text: str) -> List[Dict]:
        """
        使用 REBEL 模型抽取关系三元组 (head, relation, tail)
        REBEL 是专门训练的关系抽取模型，支持 200+ 种关系类型
        """
        relations = []
        try:
            # 截断过长文本避免 OOM
            text_truncated = text[:512]
            
            # Tokenize
            inputs = self.rebel_tokenizer(
                text_truncated, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            if DEVICE == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.rebel_model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=3,
                    num_return_sequences=1
                )
            
            # Decode
            decoded = self.rebel_tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
            
            # Parse REBEL output format: <triplet> head <subj> relation <obj> tail
            relations = self._parse_rebel_output(decoded)
            
        except Exception as e:
            print(f"⚠️ REBEL Extraction Error: {e}")
        
        return relations
    
    def _parse_rebel_output(self, text: str) -> List[Dict]:
        """
        解析 REBEL 输出格式
        格式: <triplet> head <subj> relation <obj> tail <triplet> ...
        """
        relations = []
        
        # 清理特殊 token
        text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
        
        # 按 <triplet> 分割
        triplets = text.split("<triplet>")
        
        for triplet in triplets:
            triplet = triplet.strip()
            if not triplet:
                continue
            
            try:
                # 提取 head
                if "<subj>" in triplet:
                    head = triplet.split("<subj>")[0].strip()
                    rest = triplet.split("<subj>")[1]
                else:
                    continue
                
                # 提取 relation 和 tail
                if "<obj>" in rest:
                    relation = rest.split("<obj>")[0].strip()
                    tail = rest.split("<obj>")[1].strip()
                else:
                    continue
                
                # 归一化实体名
                head_norm = self._normalize_entity(head)
                tail_norm = self._normalize_entity(tail)
                
                if head_norm and tail_norm and head_norm != tail_norm:
                    relations.append({
                        "source": head_norm,
                        "target": tail_norm,
                        "type": relation.upper().replace(" ", "_")
                    })
            except Exception:
                continue
        
        return relations

    def reset(self):
        """重置：清空图+向量库（纯内存 EphemeralClient）"""
        self.doc_cache = {}
        try:
            with self.driver.session() as session:
                session.run("MATCH (n) DETACH DELETE n")
                session.run("CREATE INDEX chunk_id_idx IF NOT EXISTS FOR (c:Chunk) ON (c.id)")
                session.run("CREATE INDEX topic_id_idx IF NOT EXISTS FOR (t:Topic) ON (t.id)")
        except Exception as e:
            print(f"⚠️ Neo4j Reset Error: {e}")

        try:
            self.chroma_client = chromadb.EphemeralClient()
            unique_col = f"rag_mem_{uuid.uuid4().hex}"
            self.vector_store = Chroma(
                client=self.chroma_client,
                collection_name=unique_col,
                embedding_function=self.embeddings,
            )
        except Exception as e:
            print(f"❌ Chroma Init Error: {e}")
            raise

    def _summarize_text(self, text: str, hint: str) -> str:
        """
        [优化策略] LLM 增强摘要 (LLM-based Summary)
        真正的摘要树构建需要语义压缩，而不仅仅是截断。
        """
        if not text: return ""
        
        # 简单的启发式快速过滤（如果文本太短，直接返回，节省 Token）
        if len(text) < 500:
            clean_text = text.replace("\n", " ").strip()
            return f"[{hint}] " + clean_text

        try:
            # 使用 LLM 生成摘要
            from langchain_core.messages import HumanMessage
            prompt = f"Please provide a concise summary of the following text, focusing on the topic '{hint}'.\n\nText:\n{text[:4000]}"
            response = self.llm.invoke([HumanMessage(content=prompt)])
            return f"[{hint}] {response.content}"
        except Exception as e:
            print(f"⚠️ Summary LLM Error: {e}, falling back to heuristic.")
            clean_text = text.replace("\n", " ").strip()
            return f"[{hint}] " + clean_text[:300] + "..."

    

    def ingest(self, documents: List[Dict]):
        """
        三层图谱构建（摘要树版）：
        Layer 1 (Tree): Document -> Summary(L2) -> Summary(L1) -> Chunk
        Layer 2 (Passage): Chunk <-> Chunk (NEXT, RELATED)
        Layer 3 (Entity): Chunk -> Entity
        """
        if not documents:
            return

        lc_docs: List[Document] = []
        ids: List[str] = []
        all_chunks: List[Dict] = []

        # Step 1: Chunk/Embedding/Entity Extraction (No LLM)
        for i, doc in enumerate(documents):
            # ... (this part is unchanged) ...
            cid = f"chunk_{i}"
            text = doc.get("text", "")
            title = doc.get("title", "")

            self.doc_cache[cid] = {"text": text, "title": title}
            embedding = self.embeddings.embed_query(text)

            try:
                # [思路A] 扩展实体类型 + 降低阈值
                labels = [
                    "Person", "Organization", "Location", "Event", "Product", "Concept",
                    "Work", "Facility", "Date", "Award", "Technology", "Sport", "Animal"
                ]
                ents = self.entity_model.predict_entities(text, labels, threshold=0.2)  # 降低阈值
                # [思路B] 实体归一化
                unique_ents = {self._normalize_entity(e["text"]): e["label"] for e in ents if self._normalize_entity(e["text"])}
            except Exception:
                unique_ents = {}
            
            # 使用 REBEL 抽取关系三元组
            rebel_rels = self._extract_relations_rebel(text)
            
            # 整理实体列表
            gliner_ents = [{"name": k, "type": v} for k, v in unique_ents.items()]

            all_chunks.append({
                "doc_title": title, "chunk_id": cid, "text": text, "embedding": embedding,
                "entities": gliner_ents,
                "rebel_rels": rebel_rels,  # REBEL 抽取的关系三元组
                "prev_id": f"chunk_{i-1}" if i > 0 else None,
            })

            ids.append(cid)
            lc_docs.append(Document(page_content=text, metadata={"doc_id": cid, "title": title, "type": "chunk"}))

        # Step 2: 构建摘要树 (Build Summary Tree)
        docs_map: Dict[str, List[Dict]] = {}
        for c in all_chunks:
            docs_map.setdefault(c["doc_title"], []).append(c)

        summary_nodes_batch = []
        summary_rels_batch = []
        
        for title, chunks in docs_map.items():
            if not chunks: continue

            # --- Level 1 Summaries ---
            vecs = np.array([c["embedding"] for c in chunks])
            n_clusters = max(1, min(len(chunks) // 3, 5)) # Control cluster count
            try:
                kmeans = KMeans(n_clusters=n_clusters, n_init="auto").fit(vecs)
                labels = kmeans.labels_
            except Exception:
                labels = [0] * len(chunks)

            l1_summaries = []
            for c_idx in range(n_clusters):
                cluster_chunks = [chunks[j] for j, lbl in enumerate(labels) if lbl == c_idx]
                if not cluster_chunks: continue
                
                cluster_text = " ".join([c["text"] for c in cluster_chunks])
                summary_text = self._summarize_text(cluster_text, hint=title)
                summary_id = f"summary_l1_{title}_{c_idx}"
                
                l1_summaries.append({"id": summary_id, "text": summary_text, "level": 1, "title": title})
                summary_nodes_batch.append({"id": summary_id, "text": summary_text, "level": 1, "doc_title": title})

                for chunk in cluster_chunks:
                    summary_rels_batch.append({"source": summary_id, "target": chunk["chunk_id"], "type": "CONTAINS_CHUNK"})
                
                ids.append(summary_id)
                lc_docs.append(Document(page_content=summary_text, metadata={"doc_id": summary_id, "title": title, "type": "summary"}))

            # --- Level 2 Summary (Root) ---
            if len(l1_summaries) > 1:
                l1_summary_text = "\n".join([s["text"] for s in l1_summaries])
                l2_summary_text = self._summarize_text(l1_summary_text, hint=f"Overall summary for {title}")
                l2_summary_id = f"summary_l2_{title}"

                summary_nodes_batch.append({"id": l2_summary_id, "text": l2_summary_text, "level": 2, "doc_title": title})
                for l1_node in l1_summaries:
                    summary_rels_batch.append({"source": l2_summary_id, "target": l1_node["id"], "type": "CONTAINS_SUMMARY"})

                ids.append(l2_summary_id)
                lc_docs.append(Document(page_content=l2_summary_text, metadata={"doc_id": l2_summary_id, "title": title, "type": "summary"}))

        # Step3: 写入向量库（内存）
        self.vector_store.add_documents(lc_docs, ids=ids)
        try:
            cnt = self.vector_store._collection.count()
            print(f"[DEBUG] Chroma count: {cnt}")
        except Exception as dbg:
            print(f"[DEBUG] Chroma count/get 失败: {dbg}")

        # Step4: 语义边 RELATED
        semantic_rels = []
        if len(all_chunks) > 1:
            mat = np.array([c["embedding"] for c in all_chunks])
            sim_mat = cosine_similarity(mat)
            rows, cols = np.where(sim_mat > 0.7)
            for r, c in zip(rows, cols):
                if r < c:
                    semantic_rels.append(
                        {
                            "source": all_chunks[r]["chunk_id"],
                            "target": all_chunks[c]["chunk_id"],
                            "score": float(sim_mat[r, c]),
                        }
                    )

        # Step5: 写 Neo4j
        try:
            with self.driver.session() as session:
                # 写入 Chunks 和 Entities
                session.run(
                    """
                    UNWIND $batch AS row
                    MERGE (d:Document {title: row.doc_title})
                    MERGE (c:Chunk {id: row.chunk_id}) SET c.text = row.text
                    MERGE (d)-[:CONTAINS]->(c)
                    
                    FOREACH (_ IN CASE WHEN row.prev_id IS NOT NULL THEN [1] ELSE [] END |
                        MERGE (p:Chunk {id: row.prev_id}) MERGE (p)-[:NEXT]->(c))
                    
                    FOREACH (e IN row.entities |
                        MERGE (ent:Entity {name: e.name}) SET ent.type = e.type
                        MERGE (c)-[:MENTIONS]->(ent))
                    """,
                    batch=all_chunks,
                )
                
                # 写入 REBEL 抽取的关系三元组
                session.run(
                    """
                    UNWIND $batch AS row
                    UNWIND row.rebel_rels AS rel
                    MERGE (source:Entity {name: rel.source})
                    MERGE (target:Entity {name: rel.target})
                    MERGE (source)-[r:RELATION {type: rel.type, source_chunk: row.chunk_id}]->(target)
                    """,
                    batch=all_chunks
                )

                # 写入 Summaries 和关系
                if summary_nodes_batch:
                    session.run(
                        """
                        UNWIND $batch AS row
                        MERGE (s:Summary {id: row.id}) SET s.text = row.text, s.level = row.level
                        MERGE (d:Document {title: row.doc_title}) MERGE (d)-[:HAS_SUMMARY]->(s)
                        """,
                        batch=summary_nodes_batch,
                    )
                    
                    # Split relationships to avoid MATCH inside FOREACH
                    rels_to_summary = [r for r in summary_rels_batch if r["type"] == "CONTAINS_SUMMARY"]
                    rels_to_chunk = [r for r in summary_rels_batch if r["type"] == "CONTAINS_CHUNK"]

                    if rels_to_summary:
                        session.run(
                            """
                            UNWIND $batch AS row
                            MATCH (source:Summary {id: row.source}), (target:Summary {id: row.target})
                            MERGE (source)-[:CONTAINS]->(target)
                            """,
                            batch=rels_to_summary
                        )

                    if rels_to_chunk:
                        session.run(
                            """
                            UNWIND $batch AS row
                            MATCH (source:Summary {id: row.source}), (target:Chunk {id: row.target})
                            MERGE (source)-[:CONTAINS]->(target)
                            """,
                            batch=rels_to_chunk
                        )

                # 写入语义边
                if semantic_rels:
                    session.run(
                        """
                        UNWIND $batch AS row
                        MATCH (a:Chunk {id: row.source}), (b:Chunk {id: row.target})
                        MERGE (a)-[r:RELATED]-(b) SET r.score = row.score
                        """,
                        batch=semantic_rels,
                    )
        except Exception as e:
            print(f"❌ Neo4j Error: {e}")

    def _extract_query_entities(self, query: str) -> List[str]:
        """
        [方案A + 思路B] 从用户问题中提取关键实体，用于引导多跳检索。
        使用扩展标签 + 归一化处理
        """
        try:
            # 扩展标签类型
            labels = [
                "Person", "Organization", "Location", "Event", "Product", "Concept",
                "Work", "Facility", "Date", "Award", "Technology", "Sport", "Animal"
            ]
            ents = self.entity_model.predict_entities(query, labels, threshold=0.2)
            # 归一化 + 去重
            entity_names = list({self._normalize_entity(e["text"]) for e in ents if self._normalize_entity(e["text"])})
            if entity_names:
                print(f"🎯 Query Entities (normalized): {entity_names}")
            return entity_names
        except Exception as e:
            print(f"⚠️ Query Entity Extraction Error: {e}")
            return []

    def _compute_trust_score(self, node: Dict, query_entities: List[str], reranker_score: float, hop_depth: int) -> float:
        """
        [方案C] 多信号可信度评分 (Multi-Signal Trustworthiness)
        融合多个信号计算节点的可信度：
        1. Reranker 语义相关性 (55%) - 主要信号，权重提高
        2. 实体覆盖率 (15%) - 辅助信号，权重降低避免过度惩罚
        3. 路径长度惩罚 (15%) - 越短越可信
        4. 来源类型加权 (15%) - QueryEnt > EntBridge > Sem > Seq
        """
        # 1. Reranker 分数归一化 (sigmoid 变换到 0-1)
        reranker_norm = 1 / (1 + np.exp(-reranker_score / 5))  # 除以5缓和曲线
        
        # 2. 实体覆盖率 (软化处理：即使没有实体匹配也给一个基础分)
        entity_coverage = 0.3  # 基础分，避免完全为0
        if query_entities:
            text_lower = node.get("text", "").lower()
            matched = sum(1 for ent in query_entities if ent in text_lower)
            entity_coverage = 0.3 + 0.7 * (matched / len(query_entities))  # 0.3-1.0 范围
        
        # 3. 路径长度惩罚 (hop_depth 越大惩罚越重，但缓和)
        path_penalty = 1.0 / (1 + 0.15 * hop_depth)  # hop=0: 1.0, hop=2: 0.77, hop=4: 0.625
        
        # 4. 来源类型加权
        source_weights = {
            "QueryEnt": 1.0,      # 直接命中问题实体
            "EntBridge": 0.90,   # 实体共现桥接
            "ActionPath": 0.85,  # 通过实体关系路径
            "SummaryDrill": 0.85, # 摘要下钻
            "Sem": 0.80,         # 语义相似
            "VectorJump": 0.75,  # 向量跳转
            "Seq": 0.65,         # 顺序扩展
        }
        # 从 title 中提取来源类型（如果是真实 title 则默认为 EntBridge）
        title = node.get("title", "")
        source_type = title if title in source_weights else "EntBridge"
        source_weight = source_weights.get(source_type, 0.75)
        
        # 加权融合
        trust_score = (
            0.55 * reranker_norm +
            0.15 * entity_coverage +
            0.15 * path_penalty +
            0.15 * source_weight
        )
        
        return trust_score

    def _summary_guided_retrieval(self, user_query: str, query_entities: List[str], top_k: int = 5) -> List[Dict]:
        """
        [方案E] Summary-Guided Top-Down Retrieval
        从 Summary Tree 顶层开始，通过摘要定位相关主题，然后下钻到具体 Chunk。
        这样可以利用摘要树的全局视角，避免一开始就陷入局部最优。
        """
        candidates = []
        try:
            # Step 1: 检索最相关的 Summary 节点
            summary_results = self.vector_store.similarity_search_with_score(
                user_query, 
                k=top_k,
                filter={"type": "summary"}  # 只检索摘要节点
            )
            
            if not summary_results:
                # 如果没有摘要节点，回退到普通检索
                return []
            
            # Step 2: 从每个相关 Summary 下钻到 Chunk
            for summary_doc, score in summary_results:
                summary_id = summary_doc.metadata.get("doc_id")
                children = self._get_summary_children(summary_id)
                
                for child in children:
                    # 如果子节点还是 Summary，继续下钻
                    if child["id"].startswith("summary_"):
                        grandchildren = self._get_summary_children(child["id"])
                        for gc in grandchildren:
                            if not gc["id"].startswith("summary_"):
                                candidates.append({
                                    "id": gc["id"],
                                    "text": gc["text"],
                                    "title": "SummaryDrill",
                                    "source_summary": summary_id
                                })
                    else:
                        candidates.append({
                            "id": child["id"],
                            "text": child["text"],
                            "title": "SummaryDrill",
                            "source_summary": summary_id
                        })
                
                if len(candidates) >= top_k * 2:
                    break
            
            if candidates:
                print(f"  🌳 Summary-Guided 下钻了 {len(candidates)} 个 Chunk")
                
        except Exception as e:
            # filter 可能不被支持，回退到普通方式
            print(f"⚠️ Summary-Guided Retrieval fallback: {e}")
        
        return candidates[:top_k]

    def query_adaptive_search(self, user_query: str, beam_width: int = 3, max_hops: int = 3) -> str:
        """
        自适应多跳检索 (Adaptive Multi-hop Search) - Best-First Search 版
        核心创新点:
        1. Global Best-First Strategy: 维护全局候选池，自然支持回溯 (Backtracking)。
        2. Context-Aware Reranking: 始终携带路径上下文。
        3. [方案C] Multi-Signal Trustworthiness: 多信号可信度评分。
        4. [方案A] Query-Guided Entity Linking: 从问题提取实体，引导多跳扩展。
        5. [方案D] Entity Relation Path: 通过 ACTION 边扩展。
        6. [方案E] Summary-Guided Top-Down: 摘要树引导检索。
        """
        print(f"🔍 Starting Adaptive Search (Max Hops: {max_hops}, Beam: {beam_width})")
        
        # [方案A] 提取问题中的实体，用于引导后续扩展
        query_entities = self._extract_query_entities(user_query)
        
        # --- 1. 初始化种子节点 ---
        initial_candidates = []
        
        # [方案E] Summary-Guided Top-Down Retrieval: 先从摘要树下钻
        summary_candidates = self._summary_guided_retrieval(user_query, query_entities, top_k=beam_width)
        for sc in summary_candidates:
            initial_candidates.append({
                "id": sc["id"],
                "text": sc["text"],
                "title": sc.get("title", "SummaryDrill"),
                "path_history": [],
                "context_str": "",
                "hop_depth": 0
            })
        
        # 普通向量检索补充
        try:
            seed_docs = self.vector_store.similarity_search_with_score(user_query, k=beam_width * 2)
            
            for doc, _ in seed_docs:
                d_id = doc.metadata.get("doc_id")
                d_type = doc.metadata.get("type")
                
                # 跳过已有的候选
                if any(c["id"] == d_id for c in initial_candidates):
                    continue
                
                # 摘要节点展开
                if d_type == "summary":
                    children = self._get_summary_children(d_id)
                    for child in children:
                        if not any(c["id"] == child["id"] for c in initial_candidates):
                            initial_candidates.append({
                                "id": child["id"], 
                                "text": child["text"], 
                                "title": child["title"],
                                "path_history": [],
                                "context_str": "",
                                "hop_depth": 0
                            })
                else:
                    initial_candidates.append({
                        "id": d_id, 
                        "text": doc.page_content, 
                        "title": doc.metadata.get("title", ""),
                        "path_history": [],
                        "context_str": "",
                        "hop_depth": 0
                    })
        except Exception as e:
            print(f"❌ Init Search Error: {e}")
            if not initial_candidates:
                return "I don't know."

        if not initial_candidates: return "I don't know."

        # --- 2. 全局优先队列 (Global Frontier) ---
        # 结构: List[Dict]
        # 我们在每一步都 Rerank 整个 Frontier (或者 Frontier 的 Top N)，然后选最好的扩展
        frontier = initial_candidates
        visited_ids = set()
        final_selected_nodes = {} # id -> node_data (去重后的最终证据)
        
        # 初始打分 [方案C] 使用多信号可信度评分
        pairs = [[user_query, c["text"]] for c in frontier]
        reranker_scores = self.reranker.compute_score(pairs)
        if isinstance(reranker_scores, float): reranker_scores = [reranker_scores]
        
        for i, node in enumerate(frontier):
            # [方案C] 多信号融合评分
            node["reranker_score"] = reranker_scores[i]
            node["trust_score"] = self._compute_trust_score(
                node, query_entities, reranker_scores[i], node.get("hop_depth", 0)
            )
            node["score"] = node["trust_score"]  # 使用可信度作为排序依据
            # 初始路径就是它自己
            node["path_history"] = [f"Start -> '{node['title']}'"]
            node["context_str"] = f"[{node['title']}] {node['text']}"

        # 按分数排序
        frontier.sort(key=lambda x: x["score"], reverse=True)
        frontier = frontier[:beam_width] # 只保留初始最好的几个

        # --- 3. 迭代扩展 (Best-First Loop) ---
        step = 0
        while step < max_hops and frontier:
            step += 1
            print(f"--- Step {step} (Frontier Size: {len(frontier)}) ---")
            
            # 取出当前最好的节点进行扩展 (Pop Best)
            # 注意：为了模拟 Beam Search 的宽度，我们这里可以一次取 Top 1 或 Top K 扩展
            # 这里采用：取 Top 1 进行扩展，然后将新节点加入 Frontier 再排序
            # 这样能最大程度体现 "Backtracking"：如果新扩展的节点分数烂，下次循环就会取原来第二好的
            
            current_best_node = frontier.pop(0) # 取出第一名
            
            if current_best_node["id"] in visited_ids:
                continue
            
            visited_ids.add(current_best_node["id"])
            
            # [方案C] 使用可信度阈值进行剪枝
            # 可信度范围 0-1，阈值降低到 0.2 避免过度剪枝
            trust_threshold = 0.2
            if current_best_node["score"] >= trust_threshold:
                final_selected_nodes[current_best_node["id"]] = current_best_node
                print(f"  ✅ Selected: {current_best_node['title']} (Trust: {current_best_node['score']:.3f}, Reranker: {current_best_node.get('reranker_score', 0):.2f})")
            else:
                print(f"  🗑️ Pruned: {current_best_node['title']} (Low Trust: {current_best_node['score']:.3f})")
                continue # 可信度太低，不扩展这条路了 (Pruning)

            # 扩展邻居 [方案A] 传入 query_entities 进行定向扩展
            neighbors_map = self._expand_node(current_best_node["id"], visited_ids, query_entities)
            
            # [方案B] Hybrid Retrieval: 如果图扩展结果不足，用累积上下文做向量检索补充
            if len(neighbors_map) < 3:
                hybrid_candidates = self._hybrid_vector_retrieval(
                    user_query, 
                    current_best_node["context_str"], 
                    visited_ids,
                    top_k=5
                )
                for hc in hybrid_candidates:
                    if hc["id"] not in neighbors_map:
                        neighbors_map[hc["id"]] = {"text": hc["text"], "title": hc["title"]}
            
            if not neighbors_map:
                continue

            # 准备新的一批候选项
            new_candidates = []
            rerank_pairs = []
            
            # 构造 Context-Aware Query
            # 使用当前节点累积的上下文
            current_context = current_best_node["context_str"][-1000:].replace("\n", " ")
            rerank_query = f"{user_query} [Context: {current_context}]"

            # --- [Scheme B] Hybrid Retrieval: 向量兜底扩展 ---
            # 利用有了上下文的新 Query，去全局向量库里再捞一把，跳出局部图限制
            try:
                # 检索 Top-K (数量与 beam_width 相当即可)
                vector_candidates = self.vector_store.similarity_search_with_score(rerank_query, k=beam_width)
                
                for v_doc, v_score in vector_candidates:
                    v_id = v_doc.metadata.get("doc_id")
                    if v_id in visited_ids: continue # 避免回头路
                    
                    # 格式化为标准节点
                    v_node = {
                        "id": v_id,
                        "text": v_doc.page_content,
                        "title": "VectorJump",  # 标记为向量跳转类型
                        # 记录这是一个跳跃步骤
                        "path_history": current_best_node["path_history"] + [f"-> [VectorJump] '{v_doc.metadata.get('title', '')}'"],
                        "context_str": current_best_node["context_str"] + f"\n[{v_doc.metadata.get('title', '')}] {v_doc.page_content}",
                        "hop_depth": current_best_node.get("hop_depth", 0) + 1,
                        "source_type": "VectorJump"
                    }
                    new_candidates.append(v_node)
            except Exception as e:
                print(f"⚠️ Vector Expansion Error: {e}")

            # --- Graph Expansion ---
            current_hop = current_best_node.get("hop_depth", 0) + 1
            for n_id, n_data in neighbors_map.items():
                if n_id in visited_ids: continue
                
                new_node = {
                    "id": n_id,
                    "text": n_data["text"],
                    "title": n_data["title"],
                    # 继承路径和上下文
                    "path_history": current_best_node["path_history"] + [f"-> '{n_data['title']}'"],
                    "context_str": current_best_node["context_str"] + f"\n[{n_data['title']}] {n_data['text']}",
                    "hop_depth": current_hop,
                    "source_type": n_data["title"]  # 记录来源类型用于可信度计算
                }
                new_candidates.append(new_node)

            # 统一构造 rerank_pairs
            rerank_pairs = []
            for node in new_candidates:
                rerank_pairs.append([rerank_query, node["text"]])

            if not new_candidates: continue

            # 批量打分 [方案C] 使用多信号可信度评分
            reranker_scores = self.reranker.compute_score(rerank_pairs)
            if isinstance(reranker_scores, float): reranker_scores = [reranker_scores]

            for i, node in enumerate(new_candidates):
                node["reranker_score"] = reranker_scores[i]
                node["trust_score"] = self._compute_trust_score(
                    node, query_entities, reranker_scores[i], node.get("hop_depth", 0)
                )
                node["score"] = node["trust_score"]  # 使用可信度作为排序依据
            
            # 将新节点加入 Frontier
            frontier.extend(new_candidates)
            
            # 重新排序 Frontier
            frontier.sort(key=lambda x: x["score"], reverse=True)
            
            # 保持 Frontier 大小适中，防止爆炸
            frontier = frontier[:beam_width * 2]

        # --- 4. 生成答案 ---
        if not final_selected_nodes:
            return "I don't know."

        # 整理最终上下文和推理路径
        # 按照被选中的顺序（分数高低）排列
        sorted_evidence = sorted(final_selected_nodes.values(), key=lambda x: x["score"], reverse=True)
        
        context_str = "\n\n".join([f"[{n['title']}] {n['text']}" for n in sorted_evidence])
        # 这里的 Path 可能有多条，我们展示得分最高的那条路径
        best_path_str = " -> ".join(sorted_evidence[0]["path_history"])

        prompt = f"""You are a precise QA system. Answer the question based on the provided context.

**Rules:**
1. Answer strictly with the Entity Name, Date, Location, or Phrase.
2. Be extremely concise. Avoid full sentences.
3. For Yes/No questions, output ONLY 'yes' or 'no'.
4. If the answer allows for a reasonable inference from the context, provide it. Only return 'I don't know' if the context is completely irrelevant.

**Reasoning Path:**
{best_path_str}

**Context:**
{context_str}

Question: {user_query}
Answer:"""
        return self.llm.invoke(prompt).content

    def _hybrid_vector_retrieval(self, user_query: str, context_str: str, visited_ids: Set[str], top_k: int = 5) -> List[Dict]:
        """
        [方案B] Hybrid Retrieval: 用累积上下文构造增强查询，回到向量库检索补充候选。
        当图扩展失败时，这个方法可以兜底。
        """
        candidates = []
        try:
            # 构造增强查询：原问题 + 当前上下文的关键信息
            # 截取上下文的前500字符作为补充信息
            context_snippet = context_str[:500].replace("\n", " ").strip()
            enhanced_query = f"{user_query} {context_snippet}"
            
            # 向量检索
            results = self.vector_store.similarity_search_with_score(enhanced_query, k=top_k * 2)
            
            for doc, score in results:
                d_id = doc.metadata.get("doc_id")
                d_type = doc.metadata.get("type")
                
                # 跳过已访问的和摘要节点
                if d_id in visited_ids or d_type == "summary":
                    continue
                
                candidates.append({
                    "id": d_id,
                    "text": doc.page_content,
                    "title": doc.metadata.get("title", "VecRetrieval")
                })
                
                if len(candidates) >= top_k:
                    break
            
            if candidates:
                print(f"  🔄 Hybrid Retrieval补充了 {len(candidates)} 个候选")
                
        except Exception as e:
            print(f"⚠️ Hybrid Retrieval Error: {e}")
        
        return candidates

    def _get_summary_children(self, summary_id: str) -> List[Dict]:
        """获取一个摘要节点的所有子节点（下一层摘要或文本块）"""
        children = []
        with self.driver.session() as s:
            # 查询所有被该摘要节点 CONTAINS 的 Summary 或 Chunk
            res = s.run(
                """
                MATCH (parent:Summary {id: $id})-[:CONTAINS]->(child)
                RETURN child.id AS id, child.text AS text, labels(child)[0] AS type
                """,
                id=summary_id,
            )
            for r in res:
                node_type = r["type"]
                title = "Summary" if node_type == "Summary" else self.doc_cache.get(r["id"], {}).get("title", "")
                children.append({"id": r["id"], "text": r["text"], "title": title})
        return children

    def _expand_node(self, c_id: str, visited: Set[str], query_entities: List[str] = None) -> Dict[str, Dict]:
        """
        扩展节点的邻居。
        [方案A] 新增 query_entities 参数，优先检索包含问题实体的 Chunk。
        [方案D] 通过 ACTION 边（实体关系路径）扩展。
        """
        data: Dict[str, Dict] = {}
        query_entities = query_entities or []
        
        # 基础扩展查询 (Sequential + Semantic + Co-occurrence + [Scheme D] Action Path)
        base_query = """
        MATCH (s:Chunk {id: $id})
        // 1. Sequential expansion
        OPTIONAL MATCH (s)-[:NEXT]-(n:Chunk) WHERE NOT n.id IN $vis
        // 2. Semantic similarity expansion
        OPTIONAL MATCH (s)-[r:RELATED]-(sem:Chunk) WHERE r.score > 0.7 AND NOT sem.id IN $vis
        // 3. Entity bridge expansion (co-occurrence)
        OPTIONAL MATCH (s)-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(b:Chunk) WHERE NOT b.id IN $vis
        // 4. [Scheme D] Spacy Action Path expansion
        OPTIONAL MATCH (s)-[:MENTIONS]->(:Entity)-[:ACTION]->(:Entity)<-[:MENTIONS]-(act:Chunk) WHERE NOT act.id IN $vis
        
        RETURN n.id, n.text, sem.id, sem.text, b.id, b.text, act.id, act.text LIMIT 20
        """
        with self.driver.session() as s:
            res = s.run(base_query, id=c_id, vis=list(visited))
            for r in res:
                if r["n.id"]:
                    data[r["n.id"]] = {"text": r["n.text"], "title": "Seq"}
                if r["sem.id"]:
                    data[r["sem.id"]] = {"text": r["sem.text"], "title": "Sem"}
                if r["b.id"]:
                    data[r["b.id"]] = {"text": r["b.text"], "title": "EntBridge"}
                if r["act.id"]:
                    data[r["act.id"]] = {"text": r["act.text"], "title": "ActionPath"}
        
        # [方案A + 思路E] Query-Guided Entity Linking with Fuzzy Matching
        # 使用 CONTAINS 模糊匹配，解决实体名不完全一致的问题
        if query_entities:
            for qe in query_entities:
                # 归一化查询实体
                qe_norm = self._normalize_entity(qe)
                if not qe_norm:
                    continue
                    
                # 模糊匹配：实体名包含查询词 或 查询词包含实体名
                entity_query = """
                MATCH (ent:Entity)<-[:MENTIONS]-(c:Chunk)
                WHERE (ent.name CONTAINS $entity OR $entity CONTAINS ent.name) 
                      AND NOT c.id IN $vis
                RETURN DISTINCT c.id AS id, c.text AS text LIMIT 5
                """
                try:
                    with self.driver.session() as s:
                        res = s.run(entity_query, entity=qe_norm, vis=list(visited))
                        for r in res:
                            if r["id"] and r["id"] not in data:
                                data[r["id"]] = {"text": r["text"], "title": "QueryEnt"}
                except Exception:
                    pass

        # 补充 title 信息
        for k in data:
            if k in self.doc_cache:
                data[k]["title"] = self.doc_cache[k].get("title", data[k]["title"])
        return data

    def close(self):
        if self.driver:
            self.driver.close()
