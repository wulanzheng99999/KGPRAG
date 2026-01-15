"""
离线建图脚本：一次性构建三层知识图谱并持久化

用法:
    # 全量构建（清空现有索引）
    python scripts/build_index.py --input data/documents.json --persist_dir ./index --reset
    
    # 增量构建（保留现有索引，只添加新文档）
    python scripts/build_index.py --input data/new_documents.json --persist_dir ./index
    
输入文件格式 (JSON):
    [
        {"title": "文档标题1", "text": "文档内容1"},
        {"title": "文档标题2", "text": "文档内容2"},
        ...
    ]
"""
import argparse
import json
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.entity_extractor import EntityExtractor
from src.graph_store import GraphStore
from src.graph_builder_offline import OfflineGraphBuilder
from src.vector_store_persistent import PersistentVectorStore
from src.config import NEO4J_URI, NEO4J_URI_FULLWIKI


def load_documents(input_path: str) -> list:
    """加载文档"""
    with open(input_path, "r", encoding="utf-8") as f:
        documents = json.load(f)
    
    # 验证格式
    if not isinstance(documents, list):
        raise ValueError("输入文件必须是 JSON 数组")
    
    for i, doc in enumerate(documents):
        if "text" not in doc:
            raise ValueError(f"文档 {i} 缺少 'text' 字段")
        if "title" not in doc:
            doc["title"] = f"Document_{i}"
    
    return documents


def main():
    parser = argparse.ArgumentParser(description="离线建图脚本")
    parser.add_argument("--input", required=True, help="输入文档 JSON 文件路径")
    parser.add_argument("--persist_dir", default="data/hotpotqa", help="持久化目录 (默认: data/hotpotqa)")
    parser.add_argument("--reset", action="store_true", help="清空现有索引后重建")
    parser.add_argument("--use_llm_summary", action="store_true", help="使用 LLM 生成摘要 (默认: 启发式摘要)")
    parser.add_argument("--batch_size", type=int, default=32, help="批处理大小 (默认: 32)")
    args = parser.parse_args()
    
    # --- 智能检测 FullWiki 模式 ---
    # 逻辑：如果输入文件包含 'fullwiki' 或者 persist_dir 包含 'fullwiki'，则切换到 7688 和 对应目录
    is_fullwiki = ("fullwiki" in args.input.lower()) or ("fullwiki" in args.persist_dir.lower())
    
    if is_fullwiki:
        # 如果用户没有强制指定其他目录（即仍是默认值），则自动切换到 fullwiki 目录
        if args.persist_dir == "data/hotpotqa": 
            args.persist_dir = "data/hotpotqa_fullwiki"
            
        neo4j_uri = NEO4J_URI_FULLWIKI
        print(f"⚠️  检测到 FullWiki 模式: 自动切换 Neo4j URI 为 {neo4j_uri} (7688端口)")
    else:
        neo4j_uri = NEO4J_URI
        print(f"ℹ️  标准模式: 使用 Neo4j URI {neo4j_uri} (7687端口)")

    print("=" * 60)
    print("🚀 KGPRAG 离线建图工具 (build_index.py)")
    print("=" * 60)
    print(f"📂 输入文件: {args.input}")
    print(f"📂 持久化目录: {args.persist_dir}")
    print(f"🌐 Neo4j URI: {neo4j_uri}")
    print("=" * 60)
    
    # 1. 加载文档
    print(f"\n📄 加载文档: {args.input}")
    documents = load_documents(args.input)
    print(f"   共 {len(documents)} 个文档")
    
    # 2. 初始化组件
    print("\n📦 初始化组件...")
    entity_extractor = EntityExtractor()
    
    # 显式传递 URI 和 allow_no_auth (fullwiki 通常不鉴权或者使用默认)
    graph_store = GraphStore(uri=neo4j_uri, allow_no_auth=is_fullwiki)
    vector_store = PersistentVectorStore(persist_dir=args.persist_dir)
    
    # 3. 处理 reset
    if args.reset:
        print("\n🗑️ 清空现有索引...")
        graph_store.reset()
        vector_store.reset()
        existing_ids = set()
    else:
        existing_ids = vector_store.get_existing_ids()
        print(f"\n📂 现有索引包含 {len(existing_ids)} 个文档")
    
    # 4. 构建图谱
    print("\n🔨 开始构建三层知识图谱...")
    builder = OfflineGraphBuilder(
        entity_extractor, 
        graph_store, 
        vector_store,
        use_llm_summary=args.use_llm_summary
    )
    
    doc_cache = builder.build(
        documents, 
        existing_chunk_ids=existing_ids,
        start_idx=len(existing_ids)
    )
    
    # 5. 保存 doc_cache
    persist_path = Path(args.persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)
    
    cache_path = persist_path / "doc_cache.json"
    
    # 增量模式：合并现有缓存
    if cache_path.exists() and not args.reset:
        with open(cache_path, "r", encoding="utf-8") as f:
            old_cache = json.load(f)
        old_cache.update(doc_cache)
        doc_cache = old_cache
    
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(doc_cache, f, ensure_ascii=False, indent=2)
    
    print("\n" + "=" * 60)
    print(f"✅ 索引构建完成!")
    print(f"   持久化目录: {args.persist_dir}")
    print(f"   文档缓存: {cache_path}")
    print(f"   总文档数: {len(doc_cache)}")
    print("=" * 60)
    
    # 显式关闭连接
    graph_store.close()


if __name__ == "__main__":
    main()