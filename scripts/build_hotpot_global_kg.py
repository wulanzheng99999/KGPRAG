"""
HotpotQA 全量建图脚本：提取所有唯一文档并构建全局 KG

对标 KG2RAG 论文的实验设置：
- 从 HotpotQA 数据集提取所有唯一文档
- 构建全量知识图谱（一次性离线构建）
- 评测时通过 doc_filter 限制检索范围

用法:
    python scripts/build_hotpot_global_kg.py --input data/hotpot_dev_distractor_v1.json --persist_dir ./index
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Set
from tqdm import tqdm

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.entity_extractor import EntityExtractor
from src.graph_store import GraphStore
from src.graph_builder_offline import OfflineGraphBuilder
from src.vector_store_persistent import PersistentVectorStore
from util.custom_logger import ExperimentLogger


def extract_unique_documents(data: List[Dict]) -> tuple:
    """
    从 HotpotQA 数据集中提取所有唯一文档
    """
    unique_docs: Dict[str, str] = {}  # title -> text
    sample_doc_mapping: Dict[str, List[str]] = {}  # sample_id -> [doc_ids]
    
    for sample in data:
        sample_id = sample["_id"]
        context = sample["context"]
        doc_ids = []
        
        for item in context:
            title = item[0]
            sentences = item[1]
            text = " ".join(sentences)
            
            if title not in unique_docs:
                unique_docs[title] = text
            
            doc_ids.append(title)
        
        sample_doc_mapping[sample_id] = doc_ids
    
    documents = [{"title": title, "text": text} for title, text in unique_docs.items()]
    title_to_doc_id = {doc["title"]: f"chunk_{i}" for i, doc in enumerate(documents)}
    
    sample_doc_mapping_with_chunk_ids = {}
    for sample_id, titles in sample_doc_mapping.items():
        sample_doc_mapping_with_chunk_ids[sample_id] = [title_to_doc_id[t] for t in titles]
    
    return documents, title_to_doc_id, sample_doc_mapping_with_chunk_ids


def main():
    parser = argparse.ArgumentParser(description="HotpotQA 全量建图脚本")
    parser.add_argument("--input", default="data/hotpot_dev_distractor_v1.json", help="HotpotQA JSON 文件路径")
    parser.add_argument("--persist_dir", default="data/hotpotqa", help="持久化目录")
    parser.add_argument("--reset", action="store_true", help="清空现有索引后重建")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已存在的文档 (断点续传)")
    parser.add_argument("--use_llm_summary", action="store_true", help="使用 LLM 生成摘要")
    args = parser.parse_args()
    
    # --- 自动选择持久化目录逻辑 ---
    if args.persist_dir == "data/hotpotqa":
        if "fullwiki" in args.input:
            args.persist_dir = "data/hotpotqa_fullwiki"
            print(f"⚠️ 检测到 FullWiki 数据集，自动将持久化目录切换为: {args.persist_dir}")
        elif "distractor" in args.input:
            args.persist_dir = "data/hotpotqa"

    # 初始化日志
    log_dir = project_root / "logs" / "kgs"
    logger = ExperimentLogger(log_dir=str(log_dir), experiment_name="build_kg")
    
    logger.info("=" * 70)
    logger.info("🚀 HotpotQA 全量建图工具 (对标 KG2RAG)")
    logger.info("=" * 70)
    logger.info(f"📂 输入文件: {args.input}")
    logger.info(f"📂 持久化目录: {args.persist_dir}")
    logger.info("=" * 70)
    
    # 1. 创建持久化目录
    persist_dir = Path(args.persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. 加载数据集
    logger.info(f"\n📄 加载数据集...")
    input_path = project_root / args.input
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logger.info(f"✅ 加载 {len(data)} 个 QA 样本")
    
    # 3. 提取所有唯一文档
    logger.info(f"\n📑 提取唯一文档...")
    documents, title_to_doc_id, sample_doc_mapping = extract_unique_documents(data)
    logger.info(f"✅ 提取 {len(documents)} 个唯一文档")
    
    # 4. 初始化模块
    logger.info(f"\n🔧 初始化模块...")
    entity_extractor = EntityExtractor()
    graph_store = GraphStore()
    vector_store = PersistentVectorStore(persist_dir=str(persist_dir))
    
    graph_builder = OfflineGraphBuilder(
        entity_extractor=entity_extractor,
        graph_store=graph_store,
        vector_store=vector_store,
        use_llm_summary=args.use_llm_summary,
        logger=logger
    )
    
    # 5. 重置或检查断点
    existing_chunk_ids = set()
    if args.reset:
        logger.info(f"\n🗑️ 清空现有索引...")
        graph_store.reset()
        vector_store.reset()
    elif args.skip_existing:
        logger.info(f"\n🔍 检查已存在文档...")
        existing_chunk_ids = graph_store.get_existing_chunk_ids()
    
    # 6. 构建图谱
    logger.info(f"\n🚀 开始构建全量图谱...")
    start_time = time.time()
    doc_cache = graph_builder.build(documents, existing_chunk_ids=existing_chunk_ids)
    build_time = time.time() - start_time
    
    # 7. 保存缓存和映射
    cache_path = persist_dir / "doc_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(doc_cache, f, ensure_ascii=False)
    
    # 构建并保存映射
    title_to_chunk_ids = {}
    for cid, doc_info in doc_cache.items():
        t = doc_info["title"]
        title_to_chunk_ids.setdefault(t, []).append(cid)
    
    sample_mapping = {}
    for item in tqdm(data, desc="Mapping Samples"):
        s_id = item['_id']
        s_chunk_ids = []
        for t in [doc_item[0] for doc_item in item['context']]:
            s_chunk_ids.extend(title_to_chunk_ids.get(t, []))
        sample_mapping[s_id] = s_chunk_ids

    with open(persist_dir / "sample_doc_mapping.json", "w", encoding="utf-8") as f:
        json.dump(sample_mapping, f, ensure_ascii=False)
    with open(persist_dir / "title_to_doc_id.json", "w", encoding="utf-8") as f:
        json.dump(title_to_chunk_ids, f, ensure_ascii=False)
    
    logger.info(f"\n✅ 全量建图完成! 耗时: {build_time:.1f}s")
    graph_store.close()


if __name__ == "__main__":
    main()