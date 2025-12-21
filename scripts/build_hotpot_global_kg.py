"""
HotpotQA 全量建图脚本：提取所有唯一文档并构建全局 KG

对标 KG2RAG 论文的实验设置：
- 从 HotpotQA 数据集中提取所有唯一文档
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
    
    返回:
        documents: 去重后的文档列表 [{"title": str, "text": str}, ...]
        title_to_doc_id: 标题到文档ID的映射 {title: doc_id}
        sample_doc_mapping: 每个样本对应的文档ID列表 {sample_id: [doc_id1, doc_id2, ...]}
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
            
            # 使用 title 作为唯一标识（HotpotQA 中 title 是唯一的）
            if title not in unique_docs:
                unique_docs[title] = text
            
            doc_ids.append(title)  # 使用 title 作为 doc_id
        
        sample_doc_mapping[sample_id] = doc_ids
    
    # 转换为文档列表
    documents = [{"title": title, "text": text} for title, text in unique_docs.items()]
    
    # 创建 title -> doc_id 映射（这里 doc_id 就是在 documents 列表中的索引对应的 chunk_id）
    title_to_doc_id = {doc["title"]: f"chunk_{i}" for i, doc in enumerate(documents)}
    
    # 更新 sample_doc_mapping 使用 chunk_id
    sample_doc_mapping_with_chunk_ids = {}
    for sample_id, titles in sample_doc_mapping.items():
        sample_doc_mapping_with_chunk_ids[sample_id] = [title_to_doc_id[t] for t in titles]
    
    return documents, title_to_doc_id, sample_doc_mapping_with_chunk_ids


def main():
    parser = argparse.ArgumentParser(description="HotpotQA 全量建图脚本")
    parser.add_argument("--input", default="data/hotpot_dev_distractor_v1.json", help="HotpotQA JSON 文件路径")
    parser.add_argument("--persist_dir", default="data/hotpotqa", help="持久化目录 (默认: data/hotpotqa)")
    parser.add_argument("--reset", action="store_true", help="清空现有索引后重建")
    parser.add_argument("--skip_existing", action="store_true", help="跳过已存在的文档 (断点续传)")
    parser.add_argument("--use_llm_summary", action="store_true", help="使用 LLM 生成摘要 (默认: 启发式摘要)")
    args = parser.parse_args()
    
    # 初始化日志
    log_dir = project_root / "logs" / "kgs"
    logger = ExperimentLogger(log_dir=str(log_dir), experiment_name="build_kg")
    
    logger.info("=" * 70)
    logger.info("🚀 HotpotQA 全量建图工具 (对标 KG2RAG)")
    logger.info("=" * 70)
    logger.info(f"📂 输入文件: {args.input}")
    logger.info(f"📂 持久化目录: {args.persist_dir}")
    logger.info(f"⚙️  使用 LLM 摘要: {args.use_llm_summary}")
    logger.info(f"⏩ 断点续传: {args.skip_existing}")
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
    logger.info(f"   (原始 {len(data) * 10} 个文档，去重率: {1 - len(documents) / (len(data) * 10):.1%})")
    
    # 4. 保存文档映射（评测时需要用）
    mapping_path = persist_dir / "sample_doc_mapping.json"
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump(sample_doc_mapping, f, ensure_ascii=False)
    logger.info(f"💾 保存样本-文档映射: {mapping_path}")
    
    title_mapping_path = persist_dir / "title_to_doc_id.json"
    with open(title_mapping_path, "w", encoding="utf-8") as f:
        json.dump(title_to_doc_id, f, ensure_ascii=False)
    logger.info(f"💾 保存标题-文档ID映射: {title_mapping_path}")
    
    # 5. 初始化模块
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
    logger.info("✅ 模块初始化完成")
    
    # 6. 重置（如果指定）
    existing_chunk_ids = set()
    if args.reset:
        logger.info(f"\n🗑️ 清空现有索引...")
        graph_store.reset()
        vector_store.reset()
        logger.info("✅ 索引已清空")
    elif args.skip_existing:
        logger.info(f"\n🔍 检查已存在文档...")
        existing_chunk_ids = graph_store.get_existing_chunk_ids()
        logger.info(f"✅ 发现 {len(existing_chunk_ids)} 个已存在 Chunk")
    
    # 7. 构建全量图谱
    logger.info(f"\n🚀 开始构建全量图谱...")
    logger.info(f"   共 {len(documents)} 个文档")
    start_time = time.time()
    
    doc_cache = graph_builder.build(documents, existing_chunk_ids=existing_chunk_ids)
    
    build_time = time.time() - start_time
    logger.info(f"✅ 图谱构建完成，耗时: {build_time:.1f}s")
    
    # 8. 保存 doc_cache
    cache_path = persist_dir / "doc_cache.json"
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(doc_cache, f, ensure_ascii=False)
    logger.info(f"💾 保存文档缓存: {cache_path}")
    
    # 9. 统计信息
    logger.info(f"\n{'=' * 70}")
    logger.info(f"✅ 全量建图完成!")
    logger.info(f"📊 统计:")
    logger.info(f"   - QA 样本数: {len(data)}")
    logger.info(f"   - 唯一文档数: {len(documents)}")
    logger.info(f"   - 文档缓存大小: {len(doc_cache)}")
    logger.info(f"   - 构建耗时: {build_time:.1f}s")
    logger.info(f"   - 平均每文档: {build_time / len(documents):.3f}s")
    logger.info(f"📂 持久化目录: {persist_dir.absolute()}")
    logger.info(f"\n📝 下一步:")
    logger.info(f"   运行评测: python evaluate.py")
    logger.info("=" * 70)
    
    # 10. 关闭连接
    graph_store.close()


if __name__ == "__main__":
    main()
