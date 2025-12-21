"""
检索模块：多跳检索与可信度评分
"""
from __future__ import annotations

from typing import List, Dict, Set, Tuple

import numpy as np
import networkx as nx
from FlagEmbedding import FlagReranker

from src.config import DEVICE, RERANKER_MODEL, DEFAULT_BEAM_WIDTH, DEFAULT_MAX_HOPS, TRUST_THRESHOLD
from src.entity_extractor import EntityExtractor
from src.graph_store import GraphStore
from src.vector_store import VectorStore


class MultiHopRetriever:
    """多跳检索器"""
    
    def __init__(self, entity_extractor: EntityExtractor, graph_store: GraphStore, vector_store: VectorStore):
        self.entity_extractor = entity_extractor
        self.graph_store = graph_store
        self.vector_store = vector_store
        
        print(f"📦 加载重排模型: {RERANKER_MODEL}")
        self.reranker = FlagReranker(RERANKER_MODEL, use_fp16=(DEVICE == "cuda"))
        
        # Reranker 缓存 (避免重复计算)
        self._reranker_cache: Dict[str, float] = {}
        
        # PPR 分数缓存 (每个问题重置)
        self._ppr_scores: Dict[str, float] = {}
    
    def _get_reranker_scores(self, pairs: List[List[str]]) -> List[float]:
        """带缓存的 Reranker 调用"""
        results = []
        uncached_pairs = []
        uncached_indices = []
        
        for i, pair in enumerate(pairs):
            cache_key = f"{pair[0][:100]}|||{pair[1][:100]}"  # 截断避免 key 过长
            if cache_key in self._reranker_cache:
                results.append(self._reranker_cache[cache_key])
            else:
                results.append(None)  # 占位
                uncached_pairs.append(pair)
                uncached_indices.append(i)
        
        # 批量计算未缓存的
        if uncached_pairs:
            scores = self.reranker.compute_score(uncached_pairs)
            if isinstance(scores, float):
                scores = [scores]
            
            for idx, score in zip(uncached_indices, scores):
                results[idx] = score
                cache_key = f"{pairs[idx][0][:100]}|||{pairs[idx][1][:100]}"
                self._reranker_cache[cache_key] = score
        
        return results
    
    def reset_cache(self):
        """重置缓存（每个问题开始时调用）"""
        self._reranker_cache = {}
        self._ppr_scores = {}
    
    def _build_graph_and_compute_ppr(self, query_entities: List[str], alpha: float = 0.85) -> Dict[str, float]:
        """
        构建内存图并计算 Personalized PageRank
        
        参数:
            query_entities: 查询中的实体列表（作为种子节点）
            alpha: 阻尼系数 (0.85 是经典值)
        
        返回:
            {chunk_id: ppr_score, ...}
        """
        # 如果没有查询实体，跳过 PPR
        if not query_entities:
            print("  ⏭️ PPR skipped: no query entities")
            return {}
        
        G = nx.Graph()
        
        try:
            with self.graph_store.driver.session() as session:
                # 1. 获取所有 Chunk 节点
                result = session.run("MATCH (c:Chunk) RETURN c.id AS id")
                for record in result:
                    G.add_node(record["id"])
                
                if len(G.nodes) == 0:
                    print("  ⏭️ PPR skipped: no chunks in graph")
                    return {}
                
                # 2. 获取所有边 (分开查询，避免 UNION 超时)
                # NEXT 边
                result = session.run("""
                    MATCH (c1:Chunk)-[:NEXT]-(c2:Chunk)
                    RETURN DISTINCT c1.id AS source, c2.id AS target
                """)
                for record in result:
                    if record["source"] in G.nodes and record["target"] in G.nodes:
                        G.add_edge(record["source"], record["target"])
                
                # RELATED 边
                result = session.run("""
                    MATCH (c1:Chunk)-[:RELATED]-(c2:Chunk)
                    RETURN DISTINCT c1.id AS source, c2.id AS target
                """)
                for record in result:
                    if record["source"] in G.nodes and record["target"] in G.nodes:
                        G.add_edge(record["source"], record["target"])
                
                # Entity Bridge 边 (限制数量避免超时)
                result = session.run("""
                    MATCH (c1:Chunk)-[:MENTIONS]->(:Entity)<-[:MENTIONS]-(c2:Chunk)
                    WHERE c1.id < c2.id
                    RETURN DISTINCT c1.id AS source, c2.id AS target
                    LIMIT 500
                """)
                for record in result:
                    if record["source"] in G.nodes and record["target"] in G.nodes:
                        G.add_edge(record["source"], record["target"])
                
                # 3. 构建 personalization 向量
                personalization = {}
                for entity in query_entities[:5]:  # 限制最多 5 个实体
                    result = session.run("""
                        MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)
                        WHERE e.name CONTAINS $entity OR $entity CONTAINS e.name
                        RETURN c.id AS chunk_id
                        LIMIT 10
                    """, entity=entity)
                    for record in result:
                        chunk_id = record["chunk_id"]
                        if chunk_id in G.nodes:
                            personalization[chunk_id] = personalization.get(chunk_id, 0) + 1.0
                
                # 归一化
                if personalization:
                    total = sum(personalization.values())
                    personalization = {k: v/total for k, v in personalization.items()}
                else:
                    print("  ⏭️ PPR skipped: no seed nodes found")
                    return {}
                
        except Exception as e:
            print(f"  ⚠️ PPR Graph Build Error: {e}")
            return {}
        
        # 4. 计算 PPR
        if len(G.edges) == 0:
            print("  ⏭️ PPR skipped: no edges in graph")
            return {}
        
        try:
            ppr_scores = nx.pagerank(G, alpha=alpha, personalization=personalization, max_iter=30, tol=1e-4)
            
            # 归一化到 [0, 1]
            if ppr_scores:
                max_score = max(ppr_scores.values())
                min_score = min(ppr_scores.values())
                if max_score > min_score:
                    ppr_scores = {k: (v - min_score) / (max_score - min_score) for k, v in ppr_scores.items()}
            
            print(f"  📊 PPR: {len(ppr_scores)} nodes, {len(G.edges)} edges, {len(personalization)} seeds")
            return ppr_scores
            
        except Exception as e:
            print(f"  ⚠️ PPR Compute Error: {e}")
            return {}
    
    def compute_trust_score(self, node: Dict, query_entities: List[str], reranker_score: float, hop_depth: int) -> float:
        """
        多信号可信度评分 (Multi-Signal Trustworthiness)
        融合多个信号计算节点的可信度：
        1. Reranker 语义相关性 (50%)
        2. PPR 拓扑重要性 (10%) - 仅当 PPR 有效时
        3. 实体覆盖率 (15%)
        4. 路径长度惩罚 (12%)
        5. 来源类型加权 (13%)
        """
        # 1. Reranker 分数归一化
        reranker_norm = 1 / (1 + np.exp(-reranker_score / 5))
        
        # 2. PPR 分数（仅当 PPR 有效时使用）
        node_id = node.get("id", "")
        ppr_available = bool(self._ppr_scores)  # PPR 是否有效
        ppr_score = self._ppr_scores.get(node_id, 0.0) if ppr_available else 0.0
        
        # 3. 实体覆盖率
        entity_coverage = 0.3  # 基础分
        if query_entities:
            text_lower = node.get("text", "").lower()
            matched = sum(1 for ent in query_entities if ent in text_lower)
            entity_coverage = 0.3 + 0.7 * (matched / len(query_entities))
        
        # 4. 路径长度惩罚
        path_penalty = 1.0 / (1 + 0.15 * hop_depth)
        
        # 5. 来源类型加权
        source_weights = {
            "QueryEnt": 1.0,
            "EntBridge": 0.90,
            "RelPath": 0.85,
            "SummaryDrill": 0.85,
            "Sem": 0.80,
            "VectorJump": 0.75,
            "Seq": 0.65,
        }
        title = node.get("title", "")
        source_type = title if title in source_weights else "EntBridge"
        source_weight = source_weights.get(source_type, 0.75)
        
        # 加权融合 (动态调整：PPR 有效时使用，否则将权重分配给其他信号)
        if ppr_available:
            trust_score = (
                0.50 * reranker_norm +
                0.10 * ppr_score +
                0.15 * entity_coverage +
                0.12 * path_penalty +
                0.13 * source_weight
            )
        else:
            # PPR 无效时，回退到原始权重
            trust_score = (
                0.55 * reranker_norm +
                0.15 * entity_coverage +
                0.15 * path_penalty +
                0.15 * source_weight
            )
        
        return trust_score
    
    def search(
        self, 
        user_query: str, 
        doc_cache: Dict, 
        beam_width: int = DEFAULT_BEAM_WIDTH, 
        max_hops: int = DEFAULT_MAX_HOPS,
        doc_filter: Set[str] = None
    ) -> Dict:
        """
        自适应多跳检索 (Adaptive Multi-hop Search)
        
        参数:
            user_query: 用户查询
            doc_cache: 文档缓存
            beam_width: Beam 宽度
            max_hops: 最大跳数
            doc_filter: 限制检索范围的文档 ID 集合（用于 HotpotQA-Dist 设置）
                        如果为 None，则不限制（用于 HotpotQA-Full 设置）
        
        返回: {"nodes": 选中的节点列表, "best_path": 最佳路径}
        """
        filter_info = f", Filter: {len(doc_filter)} docs" if doc_filter else ""
        print(f"🔍 Starting Adaptive Search (Max Hops: {max_hops}, Beam: {beam_width}{filter_info})")

        def get_doc_title(doc_id: str) -> str:
            if not doc_cache:
                return ""
            return doc_cache.get(doc_id, {}).get("title", "")
        
        # 重置缓存
        self.reset_cache()
        
        # 提取问题实体
        query_entities = self.entity_extractor.extract_query_entities(user_query)
        
        # 计算 PPR 分数（以 query_entities 为种子）
        self._ppr_scores = self._build_graph_and_compute_ppr(query_entities)
        
        # --- 1. 初始化种子节点 ---
        initial_candidates = []
        
        # [方案E] Summary-Guided Top-Down Retrieval
        summary_candidates = self.vector_store.summary_guided_retrieval(
            user_query, self.graph_store, top_k=beam_width
        )
        for sc in summary_candidates:
            # 应用文档过滤器 (HotpotQA-Dist 设置)
            if doc_filter and sc["id"] not in doc_filter:
                continue

            doc_title = get_doc_title(sc["id"])
            initial_candidates.append({
                "id": sc["id"],
                "text": sc["text"],
                "title": sc.get("title", "SummaryDrill"),
                "doc_title": doc_title,
                "path_history": [],
                "context_str": "",
                "hop_depth": 0,
                "path_doc_titles": [doc_title] if doc_title else []
            })
        
        # 普通向量检索补充
        try:
            seed_docs = self.vector_store.similarity_search_with_score(user_query, k=beam_width * 2)
            
            for doc, _ in seed_docs:
                d_id = doc.metadata.get("doc_id")
                d_type = doc.metadata.get("type")
                
                if any(c["id"] == d_id for c in initial_candidates):
                    continue
                
                # 应用文档过滤器 (HotpotQA-Dist 设置)
                if doc_filter and d_id not in doc_filter:
                    continue
                
                if d_type == "summary":
                    children = self.graph_store.get_summary_children(d_id)
                    for child in children:
                        # 应用文档过滤器
                        if doc_filter and child["id"] not in doc_filter:
                            continue
                        if not any(c["id"] == child["id"] for c in initial_candidates):
                            doc_title = get_doc_title(child["id"])
                            initial_candidates.append({
                                "id": child["id"], 
                                "text": child["text"], 
                                "title": doc_cache.get(child["id"], {}).get("title", ""),
                                "doc_title": doc_title,
                                "path_history": [],
                                "context_str": "",
                                "hop_depth": 0,
                                "path_doc_titles": [doc_title] if doc_title else []
                            })
                else:
                    doc_title = get_doc_title(d_id)
                    initial_candidates.append({
                        "id": d_id, 
                        "text": doc.page_content, 
                        "title": doc.metadata.get("title", ""),
                        "doc_title": doc_title,
                        "path_history": [],
                        "context_str": "",
                        "hop_depth": 0,
                        "path_doc_titles": [doc_title] if doc_title else []
                    })
        except Exception as e:
            print(f"❌ Init Search Error: {e}")
            if not initial_candidates:
                return {"nodes": [], "best_path": ""}

        if not initial_candidates:
            return {"nodes": [], "best_path": ""}

        # --- 2. 初始打分 ---
        frontier = initial_candidates
        visited_ids = set()
        final_selected_nodes = {}
        
        pairs = [[user_query, c["text"]] for c in frontier]
        reranker_scores = self._get_reranker_scores(pairs)
        
        for i, node in enumerate(frontier):
            node["reranker_score"] = reranker_scores[i]
            node["trust_score"] = self.compute_trust_score(
                node, query_entities, reranker_scores[i], node.get("hop_depth", 0)
            )
            node["score"] = node["trust_score"]
            node["path_history"] = [f"Start -> '{node['title']}'"]
            node["context_str"] = f"[{node['title']}] {node['text']}"
            if "path_doc_titles" not in node:
                doc_title = node.get("doc_title", "")
                node["path_doc_titles"] = [doc_title] if doc_title else []

        frontier.sort(key=lambda x: x["score"], reverse=True)
        frontier = frontier[:beam_width]

        # --- 3. 迭代扩展 ---
        step = 0
        while step < max_hops and frontier:
            step += 1
            print(f"--- Step {step} (Frontier Size: {len(frontier)}) ---")
            
            current_best_node = frontier.pop(0)
            
            if current_best_node["id"] in visited_ids:
                continue
            
            visited_ids.add(current_best_node["id"])
            
            # 可信度剪枝
            if current_best_node["score"] >= TRUST_THRESHOLD:
                final_selected_nodes[current_best_node["id"]] = current_best_node
                print(f"  ✅ Selected: {current_best_node['title']} (Trust: {current_best_node['score']:.3f})")
            else:
                print(f"  🗑️ Pruned: {current_best_node['title']} (Low Trust: {current_best_node['score']:.3f})")
                continue

            # 扩展邻居
            neighbors_map = self.graph_store.expand_node(
                current_best_node["id"], visited_ids, query_entities
            )
            
            # [方案B] Hybrid Retrieval
            if len(neighbors_map) < 3:
                hybrid_candidates = self.vector_store.hybrid_retrieval(
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

            # 准备新候选
            new_candidates = []
            current_context = current_best_node["context_str"][-1000:].replace("\n", " ")
            rerank_query = f"{user_query} [Context: {current_context}]"

            # Vector Jump
            try:
                vector_candidates = self.vector_store.similarity_search_with_score(rerank_query, k=beam_width)
                
                for v_doc, v_score in vector_candidates:
                    v_id = v_doc.metadata.get("doc_id")
                    if v_id in visited_ids:
                        continue
                    
                    # 应用文档过滤器 (HotpotQA-Dist 设置)
                    if doc_filter and v_id not in doc_filter:
                        continue
                    
                    doc_title = get_doc_title(v_id) or v_doc.metadata.get("title", "")
                    path_doc_titles = list(current_best_node.get("path_doc_titles", []))
                    if doc_title:
                        path_doc_titles.append(doc_title)

                    v_node = {
                        "id": v_id,
                        "text": v_doc.page_content,
                        "title": "VectorJump",
                        "doc_title": doc_title,
                        "path_history": current_best_node["path_history"] + [f"-> [VectorJump] '{v_doc.metadata.get('title', '')}'"],
                        "context_str": current_best_node["context_str"] + f"\n[{v_doc.metadata.get('title', '')}] {v_doc.page_content}",
                        "hop_depth": current_best_node.get("hop_depth", 0) + 1,
                        "path_doc_titles": path_doc_titles,
                    }
                    new_candidates.append(v_node)
            except Exception as e:
                print(f"⚠️ Vector Expansion Error: {e}")

            # Graph Expansion
            current_hop = current_best_node.get("hop_depth", 0) + 1
            for n_id, n_data in neighbors_map.items():
                if n_id in visited_ids:
                    continue
                
                # 应用文档过滤器 (HotpotQA-Dist 设置)
                if doc_filter and n_id not in doc_filter:
                    continue

                doc_title = get_doc_title(n_id)
                path_doc_titles = list(current_best_node.get("path_doc_titles", []))
                if doc_title:
                    path_doc_titles.append(doc_title)

                new_node = {
                    "id": n_id,
                    "text": n_data["text"],
                    "title": n_data["title"],
                    "doc_title": doc_title,
                    "path_history": current_best_node["path_history"] + [f"-> '{n_data['title']}'"],
                    "context_str": current_best_node["context_str"] + f"\n[{n_data['title']}] {n_data['text']}",
                    "hop_depth": current_hop,
                    "path_doc_titles": path_doc_titles,
                }
                new_candidates.append(new_node)

            if not new_candidates:
                continue

            # 限制候选数量，减少 Reranker 调用开销
            MAX_CANDIDATES_PER_HOP = beam_width * 3  # 最多 9 个候选
            if len(new_candidates) > MAX_CANDIDATES_PER_HOP:
                new_candidates = new_candidates[:MAX_CANDIDATES_PER_HOP]

            # 批量打分 (带缓存)
            rerank_pairs = [[rerank_query, node["text"]] for node in new_candidates]
            new_reranker_scores = self._get_reranker_scores(rerank_pairs)

            for i, node in enumerate(new_candidates):
                node["reranker_score"] = new_reranker_scores[i]
                node["trust_score"] = self.compute_trust_score(
                    node, query_entities, new_reranker_scores[i], node.get("hop_depth", 0)
                )
                node["score"] = node["trust_score"]
            
            frontier.extend(new_candidates)
            frontier.sort(key=lambda x: x["score"], reverse=True)
            frontier = frontier[:beam_width * 2]

        # --- 4. 返回结果 ---
        if not final_selected_nodes:
            return {"nodes": [], "best_path": ""}

        sorted_evidence = sorted(final_selected_nodes.values(), key=lambda x: x["score"], reverse=True)
        best_path_str = " -> ".join(sorted_evidence[0]["path_history"])
        
        return {
            "nodes": sorted_evidence,
            "best_path": best_path_str,
            "best_path_doc_titles": sorted_evidence[0].get("path_doc_titles", []),
        }
