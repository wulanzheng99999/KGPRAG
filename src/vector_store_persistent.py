"""
持久化向量存储模块：支持离线建图和在线检索
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import List, Dict, Set

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from src.config import DEVICE, EMBED_MODEL


class PersistentVectorStore:
    """持久化 ChromaDB 向量存储管理器"""
    
    def __init__(self, persist_dir: str = "./index/chroma"):
        print(f"📦 加载嵌入模型: {EMBED_MODEL}")
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": DEVICE},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # 使用持久化客户端
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.store = Chroma(
            client=self.chroma_client,
            collection_name="kgprag_index",
            embedding_function=self.embeddings,
        )
        
        try:
            cnt = self.store._collection.count()
            print(f"📂 加载持久化向量库: {cnt} 个文档")
        except Exception:
            print("📂 初始化空向量库")
    
    def reset(self):
        """清空并重建集合"""
        try:
            self.chroma_client.delete_collection("kgprag_index")
            print("🗑️ 已清空向量库")
        except Exception:
            pass
        self.store = Chroma(
            client=self.chroma_client,
            collection_name="kgprag_index",
            embedding_function=self.embeddings,
        )
    
    def embed_query(self, text: str) -> List[float]:
        """获取文本的嵌入向量"""
        return self.embeddings.embed_query(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的嵌入向量（高效，GPU 利用率提升 5-10x）"""
        return self.embeddings.embed_documents(texts)
    
    def add_documents(self, documents: List[Document], ids: List[str]):
        """添加文档到向量库 (带分批处理)"""
        BATCH_SIZE = 10000  # 设置一个安全的批次大小
        total_docs = len(documents)
        
        for i in range(0, total_docs, BATCH_SIZE):
            batch_docs = documents[i : i + BATCH_SIZE]
            batch_ids = ids[i : i + BATCH_SIZE]
            
            try:
                self.store.add_documents(batch_docs, ids=batch_ids)
                print(f"[Persist] Progress: {min(i + BATCH_SIZE, total_docs)}/{total_docs}")
            except Exception as e:
                print(f"[Persist] Batch {i} failed: {e}")

        try:
            cnt = self.store._collection.count()
            print(f"[Persist] Total Chroma count: {cnt}")
        except Exception as e:
            print(f"[Persist] Chroma count failed: {e}")
    
    def similarity_search_with_score(self, query: str, k: int = 5, filter: Dict = None):
        """相似度搜索"""
        if filter:
            try:
                return self.store.similarity_search_with_score(query, k=k, filter=filter)
            except Exception:
                # filter 可能不被支持，回退
                return self.store.similarity_search_with_score(query, k=k)
        return self.store.similarity_search_with_score(query, k=k)
    
    def hybrid_retrieval(self, user_query: str, context_str: str, visited_ids: Set[str], top_k: int = 5) -> List[Dict]:
        """
        [方案B] Hybrid Retrieval: 用累积上下文构造增强查询
        """
        candidates = []
        try:
            context_snippet = context_str[:500].replace("\n", " ").strip()
            enhanced_query = f"{user_query} {context_snippet}"
            
            results = self.store.similarity_search_with_score(enhanced_query, k=top_k * 2)
            
            for doc, score in results:
                d_id = doc.metadata.get("doc_id")
                d_type = doc.metadata.get("type")
                
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
    
    def summary_guided_retrieval(self, user_query: str, graph_store, top_k: int = 5) -> List[Dict]:
        """
        [方案E] Summary-Guided Top-Down Retrieval
        从摘要树顶层下钻到具体 Chunk
        """
        candidates = []
        try:
            summary_results = self.similarity_search_with_score(
                user_query, 
                k=top_k,
                filter={"type": "summary"}
            )
            
            if not summary_results:
                return []
            
            for summary_doc, score in summary_results:
                summary_id = summary_doc.metadata.get("doc_id")
                children = graph_store.get_summary_children(summary_id)
                
                for child in children:
                    if child["id"].startswith("summary_"):
                        grandchildren = graph_store.get_summary_children(child["id"])
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
            print(f"⚠️ Summary-Guided Retrieval fallback: {e}")
        
        return candidates[:top_k]
    
    def get_existing_ids(self) -> Set[str]:
        """获取已存在的文档 ID 集合（用于增量更新）"""
        try:
            collection = self.store._collection
            result = collection.get(include=[])
            return set(result.get("ids", []))
        except Exception as e:
            print(f"⚠️ Get existing IDs error: {e}")
            return set()
