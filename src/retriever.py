"""
检索模块：多跳检索与可信度评分
"""
from __future__ import annotations

from typing import List, Dict, Set, Tuple

import numpy as np
import networkx as nx
from FlagEmbedding import FlagReranker

from src.config import (
    DEVICE, RERANKER_MODEL, DEFAULT_BEAM_WIDTH, DEFAULT_MAX_HOPS, 
    TRUST_THRESHOLD, SMALL_SPACE_THRESHOLD, MIN_CANDIDATES_KEEP
)
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
            "Content": 0.95,     # Hop 0 初始节点 (置信度极高)
            "EntBridge": 0.90,
            "RelPath": 0.85,
            "EntMention": 0.85,  # 视为弱关联路径
            "SummaryDrill": 0.85,
            "SemHigh": 0.80,     # [V3] 高置信度语义边 (>=0.7)
            "Sem": 0.80,         # 兼容旧代码
            "VectorJump": 0.75,
            "SemLow": 0.60,      # [V3] 低置信度语义边 (0.55-0.7)，显著降权
            "Seq": 0.65,
        }
        
        # 优先使用 explicit source_type (由 search 注入)，否则回退到 title
        raw_type = node.get("source_type", node.get("title", ""))
        
        # 如果是已知类型，直接查表；否则默认为 "Content" (避免误判为高权重的 EntBridge)
        source_key = raw_type if raw_type in source_weights else "Content"
        source_weight = source_weights.get(source_key, 0.75)
        
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
        
        # [V3] 细粒度降权：利用 edge_score 进一步区分 SemHigh/SemLow
        edge_score = node.get("edge_score")
        if edge_score is not None:
            if source_key == "SemHigh":
                # SemHigh (0.7~1.0): 保持接近原始分 (0.97~1.0 multiplier)
                # edge_factor = 0.90 + 0.10 * edge_score (e.g. 0.7 -> 0.97, 1.0 -> 1.0)
                trust_score *= (0.90 + 0.10 * edge_score)
            elif source_key == "SemLow":
                # SemLow (0.55~0.7): 显著压低 (0.84~0.89 multiplier)
                # edge_factor = 0.65 + 0.35 * edge_score (e.g. 0.55 -> 0.84, 0.69 -> 0.89)
                trust_score *= (0.65 + 0.35 * edge_score)
        
        return trust_score
    
    def _apply_diversity_filter(self, candidates: List[Dict], beam_width: int, hop: int = 0, is_small_space: bool = False) -> List[Dict]:
        """
        Diversity Filter V2: Document + Source-Type Aware with Soft Penalty
        采用迭代贪心选择 (Iterative Greedy Selection) + 动态惩罚
        """
        if not candidates:
            return []
            
        # 1. 参数配置
        lambda_doc = 0.10   # 文档同质化惩罚
        lambda_type = 0.05  # 类型同质化惩罚
        
        # 硬限制 (Safety Guard)
        # 仅在小空间模式下放宽限制，大空间保持严格多样性
        hard_limit_doc = 2 if is_small_space else 1
        
        # 类型优先级配置 (Base Bonus)
        # 给关键类型一点初始加分，确保它们有机会被选中
        type_bonus = {
            "QueryEnt": 0.05,
            "EntBridge": 0.04 if hop > 0 else 0.0,
            "RelPath": 0.04 if hop > 0 else 0.0,
            "SummaryDrill": 0.03 if hop == 0 else 0.0,
            "VectorJump": 0.02
        }

        # 辅助函数：推断 Source Type
        def get_source_type(node):
            if "source_type" in node: return node["source_type"]
            # 兼容旧代码：从 title 推断
            t = node.get("title", "")
            if t in ["VectorJump", "EntBridge", "RelPath", "QueryEnt", "SummaryDrill", "Seq", 
                     "Sem", "SemHigh", "SemLow", "EntMention"]: # V3: SemHigh/Low
                return t
            return "Content" # 默认普通文档节点

        # 2. 准备候选池
        pool = []
        for node in candidates:
            stype = get_source_type(node)
            doc_title = node.get("doc_title", "") or f"__NO_TITLE_{node['id']}__"
            
            # 基础分 = 原始分 + 类型奖励
            base_score = node.get("score", 0.0) + type_bonus.get(stype, 0.0)
            
            pool.append({
                "node": node,
                "base_score": base_score,
                "type": stype,
                "doc": doc_title,
                "id": node["id"]
            })

        # 3. 迭代选择 (Iterative Selection)
        selected = []
        selected_ids = set()
        
        # 计数器
        counts_doc = {}
        counts_type = {}
        
        while len(selected) < beam_width and pool:
            best_idx = -1
            best_adjusted_score = -float('inf')
            
            # 遍历池中剩余候选，计算动态分数
            for i, item in enumerate(pool):
                # 硬限制检查
                if counts_doc.get(item["doc"], 0) >= hard_limit_doc:
                    continue
                
                # 动态惩罚：已选的同类/同文档越多，惩罚越大
                penalty = (lambda_doc * counts_doc.get(item["doc"], 0) + 
                           lambda_type * counts_type.get(item["type"], 0))
                           
                adj_score = item["base_score"] - penalty
                
                if adj_score > best_adjusted_score:
                    best_adjusted_score = adj_score
                    best_idx = i
            
            # 如果找不到合法的候选（都触发硬限制了），尝试放宽硬限制兜底
            if best_idx == -1:
                # 兜底策略：取剩余里 base_score 最高的，忽略硬限制
                # (为了填满 Beam，不至于空着)
                best_idx = -1
                best_base_score = -float('inf')
                for i, item in enumerate(pool):
                    if item["base_score"] > best_base_score:
                        best_base_score = item["base_score"]
                        best_idx = i
                
                if best_idx == -1: # 还是空的（pool为空）
                    break

            # 选中最佳
            chosen = pool.pop(best_idx)
            selected.append(chosen["node"])
            selected_ids.add(chosen["id"])
            
            # 更新计数
            counts_doc[chosen["doc"]] = counts_doc.get(chosen["doc"], 0) + 1
            counts_type[chosen["type"]] = counts_type.get(chosen["type"], 0) + 1
            
        # 4. 再次按分数排序返回 (保证下游处理顺序)
        selected.sort(key=lambda x: x["score"], reverse=True)
        return selected

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
        
        # --- 1. 自适应搜索域策略 (Context-Aware Retrieval Strategy) ---
        initial_candidates = []
        is_small_space = False  # 标记是否为小空间模式（用于后续 force_keep）
        
        if doc_filter is not None:
            filter_size = len(doc_filter)
            
            if filter_size <= SMALL_SPACE_THRESHOLD:
                # === 模式A: 受限小空间 (Constrained Small Space) ===
                # 策略：全量加载 + Rerank（精度优先）
                # 在 10 个文档中，向量检索容易因语义漂移丢失桥接文档
                # 直接加载全部，让 Cross-Encoder Reranker 精确排序
                is_small_space = True
                print(f"  📋 [小空间模式] 全量加载 doc_filter ({filter_size} docs)")
                
                for doc_id in doc_filter:
                    if doc_id in doc_cache:
                        doc_data = doc_cache[doc_id]
                        doc_title = doc_data.get("title", "")
                        doc_text = doc_data.get("text", "")
                        if doc_text:
                            initial_candidates.append({
                                "id": doc_id,
                                "text": doc_text,
                                "title": doc_title,
                                "doc_title": doc_title,
                                "path_history": [],
                                "context_str": "",
                                "hop_depth": 0,
                                "path_doc_titles": [doc_title] if doc_title else []
                            })
            else:
                # === 模式B: 受限大空间 (Constrained Large Space) ===
                # 策略：向量检索，但应用 doc_filter 过滤
                print(f"  🔎 [大空间模式] 向量检索 + doc_filter 过滤 ({filter_size} docs)")
                
                try:
                    # Summary-Guided 检索
                    summary_candidates = self.vector_store.summary_guided_retrieval(
                        user_query, self.graph_store, top_k=beam_width * 2
                    )
                    for sc in summary_candidates:
                        if sc["id"] in doc_filter:
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
                    seed_docs = self.vector_store.similarity_search_with_score(user_query, k=beam_width * 3)
                    for doc, _ in seed_docs:
                        d_id = doc.metadata.get("doc_id")
                        if d_id not in doc_filter:
                            continue
                        if any(c["id"] == d_id for c in initial_candidates):
                            continue
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
                    print(f"❌ Large Space Search Error: {e}")
        else:
            # === 模式C: 全开放空间 (Open Space - Fullwiki) ===
            # 策略：全库 ANN 检索（效率优先）
            print(f"  🌐 [全开放模式] 全库向量检索")
            
            # Summary-Guided Top-Down Retrieval
            summary_candidates = self.vector_store.summary_guided_retrieval(
                user_query, self.graph_store, top_k=beam_width
            )
            for sc in summary_candidates:
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
                    
                    if d_type == "summary":
                        children = self.graph_store.get_summary_children(d_id)
                        for child in children:
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
                print(f"❌ Open Space Search Error: {e}")
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
            # 确保有 source_type
            if "source_type" not in node:
                node["source_type"] = "Content"
                
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
        # 在小空间模式下，为了不错过任何线索，我们放宽 Beam Width
        if is_small_space:
            # 确保所有初始候选都进入图谱推理，但不超过 SMALL_SPACE_THRESHOLD 的上限
            effective_beam_width = min(len(frontier), SMALL_SPACE_THRESHOLD)
            # 初始多样性过滤 V2
            frontier = self._apply_diversity_filter(frontier, effective_beam_width, hop=0, is_small_space=True)
        else:
            # 初始多样性过滤 V2
            frontier = self._apply_diversity_filter(frontier, beam_width, hop=0, is_small_space=False)

        # --- 3. 迭代扩展 (Beam Search) ---
        # 改为标准的 Beam Search 结构：每跳扩展 Top-K 个节点
        NODES_TO_EXPAND = 3  # 每跳扩展的节点数 (Top-3)
        
        for hop in range(max_hops):
            print(f"--- Hop {hop+1} (Frontier Size: {len(frontier)}) ---")
            
            # 本跳产生的所有新候选
            candidates_for_next_hop = []
            
            # 取 Frontier 中尚未访问的前 K 个节点进行扩展
            # 注意：frontier 已经是按分数排序且多样性过滤过的
            nodes_to_process = []
            for node in frontier:
                if node["id"] not in visited_ids:
                    nodes_to_process.append(node)
                    if len(nodes_to_process) >= NODES_TO_EXPAND:
                        break
            
            if not nodes_to_process:
                print("  🛑 No more nodes to expand.")
                break
                
            for current_node in nodes_to_process:
                visited_ids.add(current_node["id"])
                
                # 可信度检查
                if current_node["score"] < TRUST_THRESHOLD and not (is_small_space and len(final_selected_nodes) < MIN_CANDIDATES_KEEP):
                    print(f"  🗑️ Pruned: {current_node['title']} (Low Trust: {current_node['score']:.3f})")
                    continue
                
                # 记录为已选节点
                final_selected_nodes[current_node["id"]] = current_node
                print(f"  ✅ Expanding: {current_node['title']} (Score: {current_node['score']:.3f})")
                
                # 扩展邻居
                neighbors_map = self.graph_store.expand_node(
                    current_node["id"], visited_ids, query_entities
                )
                
                # Hybrid Retrieval 补充
                if len(neighbors_map) < 3:
                    hybrid_candidates = self.vector_store.hybrid_retrieval(
                        user_query, 
                        current_node["context_str"], 
                        visited_ids,
                        top_k=5
                    )
                    for hc in hybrid_candidates:
                        if hc["id"] not in neighbors_map:
                            neighbors_map[hc["id"]] = {"text": hc["text"], "title": hc["title"]}
                
                if not neighbors_map:
                    continue

                # 准备当前节点的扩展候选
                expansion_candidates = []
                current_context = current_node["context_str"][-1000:].replace("\n", " ")
                rerank_query = f"{user_query} [Context: {current_context}]"
                
                # Vector Jump
                try:
                    vector_candidates = self.vector_store.similarity_search_with_score(rerank_query, k=beam_width)
                    for v_doc, _ in vector_candidates:
                        v_id = v_doc.metadata.get("doc_id")
                        if v_id in visited_ids: continue
                        if doc_filter and v_id not in doc_filter: continue
                        
                        doc_title = get_doc_title(v_id) or v_doc.metadata.get("title", "")
                        path_doc_titles = list(current_node.get("path_doc_titles", []))
                        if doc_title: path_doc_titles.append(doc_title)

                        expansion_candidates.append({
                            "id": v_id,
                            "text": v_doc.page_content,
                            "title": "VectorJump",
                            "source_type": "VectorJump", # Added source_type
                            "doc_title": doc_title,
                            "path_history": current_node["path_history"] + [f"-> [VectorJump] '{v_doc.metadata.get('title', '')}'"],
                            "context_str": current_node["context_str"] + f"\n[{v_doc.metadata.get('title', '')}] {v_doc.page_content}",
                            "hop_depth": current_node.get("hop_depth", 0) + 1,
                            "path_doc_titles": path_doc_titles,
                        })
                except Exception as e:
                    print(f"⚠️ Vector Expansion Error: {e}")

                # Graph Expansion
                next_hop_depth = current_node.get("hop_depth", 0) + 1
                for n_id, n_data in neighbors_map.items():
                    if n_id in visited_ids: continue
                    if doc_filter and n_id not in doc_filter: continue

                    doc_title = get_doc_title(n_id)
                    path_doc_titles = list(current_node.get("path_doc_titles", []))
                    if doc_title: path_doc_titles.append(doc_title)

                    # Determine source_type
                    # 修正：直接使用 title 作为 source_type，如果它是已知类型
                    # 这样可以保留 Sem, EntMention, QueryEnt 等类型
                    known_types = {
                        "EntBridge", "RelPath", "Sem", "SemHigh", "SemLow", # V3
                        "EntMention", "QueryEnt", "SummaryDrill", "Seq", "VectorJump"
                    }
                    stype = n_data["title"] if n_data["title"] in known_types else "Sem"

                    expansion_candidates.append({
                        "id": n_id,
                        "text": n_data["text"],
                        "title": n_data["title"],
                        "source_type": stype, # Added source_type
                        "edge_score": n_data.get("edge_score"), # [V3] 传递 edge_score
                        "doc_title": doc_title,
                        "path_history": current_node["path_history"] + [f"-> '{n_data['title']}'"],
                        "context_str": current_node["context_str"] + f"\n[{n_data['title']}] {n_data['text']}",
                        "hop_depth": next_hop_depth,
                        "path_doc_titles": path_doc_titles,
                    })
                
                if not expansion_candidates:
                    continue
                
                # 移除之前的预截断 (Top 2*Beam)，直接对所有候选打分
                # 这样可以避免在 Rerank 前丢失高质量的桥接节点
                
                # 打分
                rerank_pairs = [[rerank_query, node["text"]] for node in expansion_candidates]
                scores = self._get_reranker_scores(rerank_pairs)
                
                for i, node in enumerate(expansion_candidates):
                    node["reranker_score"] = scores[i]
                    node["trust_score"] = self.compute_trust_score(
                        node, query_entities, scores[i], node["hop_depth"]
                    )
                    node["score"] = node["trust_score"]
                
                candidates_for_next_hop.extend(expansion_candidates)

            # 本跳结束，汇总所有新候选，进行全局排序和多样性过滤
            if candidates_for_next_hop:
                # 使用多样性过滤器 V2 更新 Frontier
                # current hop index is 'hop', so next hop frontier is prepared for hop+1? 
                # Actually 'hop' variable in loop is 0, 1, 2.
                # When hop=0, we are preparing for Hop 1. So we pass hop+1.
                frontier = self._apply_diversity_filter(candidates_for_next_hop, beam_width, hop=hop+1, is_small_space=is_small_space)
            else:
                # 如果没有新候选，保持现有 Frontier（或者直接断掉？通常是断掉）
                # 这里为了保留上一层未扩展的节点作为备选，可以尝试合并？
                # 简单策略：如果没有新节点，就停止
                pass

        # --- 4. 返回结果 ---
        # Fallback 机制：如果图谱游走一无所获，但在受限空间内（如 Distractor），
        # 我们不能交白卷。必须把原始文档作为兜底证据返回。
        if not final_selected_nodes:
            if doc_filter and len(doc_filter) <= SMALL_SPACE_THRESHOLD:
                print(f"  ⚠️ Graph search failed. Fallback: Loading all {len(doc_filter)} docs from filter.")
                fallback_nodes = []
                for doc_id in doc_filter:
                    if doc_id in doc_cache:
                        d = doc_cache[doc_id]
                        fallback_nodes.append({
                            "id": doc_id,
                            "text": d.get("text", ""),
                            "title": d.get("title", ""),
                            "score": 0.5, # 赋予默认分值
                            "path_history": ["Fallback (Raw Doc)"]
                        })
                
                # 再次尝试 Rerank 排序，选出最好的
                if fallback_nodes:
                    pairs = [[user_query, n["text"]] for n in fallback_nodes]
                    scores = self._get_reranker_scores(pairs)
                    for i, node in enumerate(fallback_nodes):
                        node["score"] = scores[i]
                    
                    fallback_nodes.sort(key=lambda x: x["score"], reverse=True)
                    # 只取 Top-Beam 作为证据，避免过多噪音
                    final_selected_nodes = {n["id"]: n for n in fallback_nodes[:beam_width]}
                    
            if not final_selected_nodes:
                return {"nodes": [], "best_path": ""}

        sorted_evidence = sorted(final_selected_nodes.values(), key=lambda x: x["score"], reverse=True)
        best_path_str = " -> ".join(sorted_evidence[0]["path_history"])
        
        return {
            "nodes": sorted_evidence,
            "best_path": best_path_str,
            "best_path_doc_titles": sorted_evidence[0].get("path_doc_titles", []),
        }
