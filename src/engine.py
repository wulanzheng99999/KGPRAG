"""
Advanced GraphRAG Engine (模块化版本)
特性：
- 统一三层 KG：Document/Topic -> Chunk -> Entity
- 双模型抽取：GLiNER (实体) + REBEL (关系)
- 多跳检索：Best-First Search + 可信度评分
- 模块化架构：易于维护和扩展
- 支持离线建图 + 在线检索模式
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Optional, Set

from langchain_openai import ChatOpenAI

from src.config import LLM_MODEL, DEFAULT_BEAM_WIDTH, DEFAULT_MAX_HOPS
from src.entity_extractor import EntityExtractor
from src.graph_store import GraphStore
from src.retriever import MultiHopRetriever


class AdvancedRAGEngine:
    """
    高级 RAG 引擎 - 模块化版本
    
    支持两种模式：
    1. 内存模式 (persist_dir=None): 实时建图，数据不持久化
    2. 持久化模式 (persist_dir="./index"): 加载离线构建的索引
    """
    
    def __init__(
        self, 
        persist_dir: Optional[str] = None, 
        online_mode: bool = True,
        use_llm_summary: bool = False
    ):
        """
        参数:
            persist_dir: 持久化目录路径，None 则使用内存模式
            online_mode: True=仅加载索引用于检索，False=可构建索引
            use_llm_summary: 是否使用 LLM 生成摘要（默认 False，使用启发式摘要加速）
        """
        self.persist_dir = persist_dir
        self.online_mode = online_mode
        self.use_llm_summary = use_llm_summary
        
        if persist_dir:
            print(f"🚀 初始化 AdvancedRAG 引擎 (持久化模式: {persist_dir})...")
        else:
            print("🚀 初始化 AdvancedRAG 引擎 (内存模式)...")
        
        # 初始化各模块
        self.entity_extractor = EntityExtractor()
        self.graph_store = GraphStore()
        
        # 根据模式选择向量存储
        if persist_dir:
            from src.vector_store_persistent import PersistentVectorStore
            self.vector_store = PersistentVectorStore(persist_dir=persist_dir)
            self._load_doc_cache(persist_dir)
        else:
            from src.vector_store import VectorStore
            self.vector_store = VectorStore()
            self.doc_cache: Dict[str, Dict] = {}
        
        # 图构建器（仅非在线模式需要）
        if not online_mode:
            from src.graph_builder_offline import OfflineGraphBuilder
            self.graph_builder = OfflineGraphBuilder(
                self.entity_extractor, 
                self.graph_store, 
                self.vector_store,
                use_llm_summary=use_llm_summary
            )
        else:
            self.graph_builder = None
        
        self.retriever = MultiHopRetriever(
            self.entity_extractor,
            self.graph_store,
            self.vector_store
        )
        
        # LLM for answer generation
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
        
        print("✅ 引擎初始化完成")
    
    def _is_yes_no_question(self, query: str) -> bool:
        """
        检测是否为 Yes/No 问题
        用于选择专用 Prompt 模板，提高格式正确率
        """
        query_lower = query.lower().strip()
        
        # 1. 基础 Yes/No 模式
        yes_no_patterns = [
            query_lower.startswith("are "),
            query_lower.startswith("is "),
            query_lower.startswith("was "),
            query_lower.startswith("were "),
            query_lower.startswith("do "),
            query_lower.startswith("does "),
            query_lower.startswith("did "),
            query_lower.startswith("can "),
            query_lower.startswith("could "),
            query_lower.startswith("would "),
            query_lower.startswith("will "),
            query_lower.startswith("has "),
            query_lower.startswith("have "),
            query_lower.startswith("had "),
        ]
        if any(yes_no_patterns):
            return True

        # 2. 增强模式：检测 "same" / "both" 类型的比较问题
        # e.g., "Were Scott Derrickson and Ed Wood of the same nationality?"
        if "same" in query_lower and any(x in query_lower for x in ["nationality", "country", "type", "category", "genre", "year", "time"]):
            return True
        
        if "both" in query_lower and any(x in query_lower for x in ["are", "were", "born", "died", "from"]):
            return True

        return False
    
    def _post_process_answer(self, raw_answer: str, query: str) -> str:
        """
        通用答案后处理：强制提取 'Answer:' 后的内容
        """
        answer = raw_answer.strip()
        
        # --- 1. 通用提取逻辑：取最后一个 "Answer:" ---
        # 很多时候模型会在 Reasoning 之后输出 "Answer: xxx"，或者多次输出
        import re
        # 查找所有 "Answer:" (忽略大小写) 的位置
        # 使用正则找 "Answer:" 或 "Final Answer:" 或 "答案："
        markers = ["answer:", "final answer:", "答案："]
        
        last_answer_content = None
        
        # 简单策略：按行倒序查找
        lines = [l.strip() for l in answer.split('\n') if l.strip()]
        for line in reversed(lines):
            line_lower = line.lower()
            for marker in markers:
                if marker in line_lower:
                    # 找到了标记，提取标记后的内容
                    parts = line_lower.rsplit(marker, 1) # 只分割最后一次出现的 marker
                    if len(parts) >= 2:
                        # 注意：这里需要从原始 line 中提取，以保留大小写（虽然 HotpotQA 评测不敏感，但保留更好）
                        # 重新定位 marker 在原始 line 中的索引
                        idx = line.lower().rfind(marker)
                        candidate = line[idx + len(marker):].strip()
                        if candidate:
                            last_answer_content = candidate
                            break
            if last_answer_content:
                break
        
        # 如果没找到 Answer: 标记，则兜底取最后一行非空文本
        if not last_answer_content:
            if lines:
                last_answer_content = lines[-1]
            else:
                last_answer_content = answer

        # --- 2. 清洗提取出的答案 ---
        final_ans = last_answer_content.strip()
        
        # 去除首尾的标点和引号 (e.g., "Yes.", "1990", **1990**)
        final_ans = re.sub(r'^["\'\*`]+|["\'\*`\.\!]+$', '', final_ans)
        
        # 去除常见的废话前缀
        # e.g., "The answer is 1990" -> "1990"
        final_ans = re.sub(r'^(the answer is |it is |that is )', '', final_ans, flags=re.I).strip()
        
        # --- 3. Yes/No 标准化 (仅针对 Yes/No 问题) ---
        if self._is_yes_no_question(query):
            ans_lower = final_ans.lower()
            if ans_lower.startswith("yes"): return "yes"
            if ans_lower.startswith("no"): return "no"
            
            # 兜底检测（如果提取出的内容还包含其他词）
            if "yes" in ans_lower: return "yes"
            if "no" in ans_lower: return "no"
            
        return final_ans
    
    def _load_doc_cache(self, persist_dir: str):
        """加载持久化的 doc_cache"""
        cache_path = Path(persist_dir) / "doc_cache.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                self.doc_cache = json.load(f)
            print(f"📂 加载 {len(self.doc_cache)} 个文档缓存")
        else:
            self.doc_cache = {}
            print("⚠️ 未找到文档缓存，请先运行离线建图")
    
    def reset(self):
        """重置所有存储"""
        self.doc_cache = {}
        self.graph_store.reset()
        self.vector_store.reset()
    
    def load_precomputed_cache(self, cache_data: Dict):
        """
        加载预计算的缓存数据（跳过 GLiNER/REBEL 推理）
        
        这是 HotpotQA 评测的推荐方式：
        1. 先用 scripts/precompute_hotpot.py 预计算每个样本的图谱数据
        2. 评测时直接加载缓存，无需实时抽取
        
        参数:
            cache_data: 预计算脚本生成的缓存字典，包含:
                - chunks: 预计算的 chunk 数据（含 embedding、entities、relations）
                - summaries: 摘要节点
                - summary_rels: 摘要关系
                - semantic_edges: 语义边
                - doc_cache: 文档缓存
        """
        # 1. 加载 doc_cache
        self.doc_cache = cache_data.get("doc_cache", {})
        
        # 2. 写入 Neo4j 图存储
        chunks = cache_data.get("chunks", [])
        summaries = cache_data.get("summaries", [])
        summary_rels = cache_data.get("summary_rels", [])
        semantic_edges = cache_data.get("semantic_edges", [])
        
        self.graph_store.write_chunks(chunks)
        self.graph_store.write_summaries(summaries, summary_rels)
        self.graph_store.write_semantic_edges(semantic_edges)
        
        # 3. 写入向量存储
        from langchain_core.documents import Document
        lc_docs = []
        ids = []
        
        # 添加 chunks
        for chunk in chunks:
            lc_docs.append(Document(
                page_content=chunk["text"],
                metadata={
                    "doc_id": chunk["chunk_id"],
                    "title": chunk["doc_title"],
                    "type": "chunk"
                }
            ))
            ids.append(chunk["chunk_id"])
        
        # 添加 summaries
        for summary in summaries:
            lc_docs.append(Document(
                page_content=summary["text"],
                metadata={
                    "doc_id": summary["id"],
                    "title": summary["doc_title"],
                    "type": "summary"
                }
            ))
            ids.append(summary["id"])
        
        self.vector_store.add_documents(lc_docs, ids=ids)
    
    def ingest(self, documents: List[Dict]):
        """
        摄入文档，构建三层图谱
        documents: [{"title": str, "text": str}, ...]
        
        注意：持久化模式下建议使用离线建图脚本 scripts/build_index.py
        """
        if self.graph_builder is None:
            raise RuntimeError(
                "在线模式下不支持 ingest()，请使用离线建图脚本:\n"
                "python scripts/build_index.py --input data/documents.json --persist_dir ./index"
            )
        self.doc_cache = self.graph_builder.build(documents)
    
    def query(
        self, 
        user_query: str, 
        beam_width: int = DEFAULT_BEAM_WIDTH, 
        max_hops: int = DEFAULT_MAX_HOPS,
        doc_filter: Set[str] = None,
        return_debug: bool = False
    ) -> str:
        """
        查询接口
        
        参数:
            user_query: 用户查询
            beam_width: Beam 宽度
            max_hops: 最大跳数
            doc_filter: 限制检索范围的文档 ID 集合（用于 HotpotQA-Dist 设置）
            return_debug: 是否返回检索调试信息
        """
        # 多跳检索
        search_result = self.retriever.search(
            user_query, 
            self.doc_cache,
            beam_width=beam_width, 
            max_hops=max_hops,
            doc_filter=doc_filter
        )
        
        if not search_result["nodes"]:
            answer = "I don't know."
            if return_debug:
                return answer, {"search_result": search_result}
            return answer
        
        # 准备上下文
        sorted_evidence = search_result["nodes"]
        
        # [Phase 1 优化] 小空间模式使用更多证据节点
        is_small_space = (doc_filter is not None and len(doc_filter) <= 20)
        if is_small_space:
            from src.config import MAX_EVIDENCE_NODES_SMALL_SPACE
            max_evidence = MAX_EVIDENCE_NODES_SMALL_SPACE
        else:
            from src.config import MAX_EVIDENCE_NODES_FOR_LLM
            max_evidence = MAX_EVIDENCE_NODES_FOR_LLM
        
        # 限制证据节点数量（避免超长 Context）
        sorted_evidence = sorted_evidence[:max_evidence]
        
        context_str = "\n\n".join([f"[{n['title']}] {n['text']}" for n in sorted_evidence])
        best_path_str = search_result["best_path"]
        
        # 根据问题类型选择不同的 Prompt
        is_yes_no = self._is_yes_no_question(user_query)
        
        if is_yes_no:
            # Yes/No 专用 Prompt - 允许简短推理
            prompt = f"""You are a precise QA system. Answer the Yes/No question based on the context.

**INSTRUCTION**: 
- First, briefly reason about the answer (1-2 sentences max)
- Then, output your final answer as ONLY "yes" or "no" on a new line
- Format: 
  Reasoning: [brief reasoning]
  Answer: yes/no

**Context:**
{context_str}

**Question:** {user_query}
"""
        else:
            # 普通问题: v2.1 增强稳健版 (With Path & Anchor)
            # 1. 保留 Reasoning Path 以利用图谱优势
            # 2. 保留 Answer: 锚点以诱导直接输出
            prompt = f"""You are a precise QA system. Answer the question based on the Context.

**Output Format (STRICTLY REQUIRED):**
- Output ONLY ONE LINE: Answer: <your answer>
- <your answer> MUST be: a single entity name, date, number, or short phrase (max 5 words)
- DO NOT include any explanation, reasoning, or sentence structure

**Reasoning Path:**
{best_path_str}

**Context:**
{context_str}

**Question:** {user_query}
**Answer:**"""
        
        raw_answer = self.llm.invoke(prompt).content
        
        # 后处理答案
        answer = self._post_process_answer(raw_answer, user_query)
        if return_debug:
            return answer, {"search_result": search_result}
        return answer
    
    # 兼容旧接口
    def query_adaptive_search(
        self, 
        user_query: str, 
        beam_width: int = DEFAULT_BEAM_WIDTH, 
        max_hops: int = DEFAULT_MAX_HOPS,
        doc_filter: Set[str] = None,
        return_debug: bool = False
    ) -> str:
        """兼容旧接口"""
        return self.query(
            user_query,
            beam_width,
            max_hops,
            doc_filter,
            return_debug=return_debug
        )
    
    def close(self):
        """关闭连接"""
        self.graph_store.close()
