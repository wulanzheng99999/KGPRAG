"""
在线检索脚本：加载持久化索引进行查询

用法:
    python scripts/query_index.py --persist_dir ./index --query "你的问题"
    
    # 交互模式
    python scripts/query_index.py --persist_dir ./index --interactive
"""
import argparse
import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.engine import AdvancedRAGEngine


def main():
    parser = argparse.ArgumentParser(description="在线检索脚本")
    parser.add_argument("--persist_dir", default="./index", help="持久化目录 (默认: ./index)")
    parser.add_argument("--query", type=str, help="查询问题")
    parser.add_argument("--interactive", action="store_true", help="交互模式")
    parser.add_argument("--beam_width", type=int, default=3, help="Beam 宽度 (默认: 3)")
    parser.add_argument("--max_hops", type=int, default=3, help="最大跳数 (默认: 3)")
    args = parser.parse_args()
    
    # 检查索引是否存在
    persist_path = Path(args.persist_dir)
    if not persist_path.exists():
        print(f"❌ 索引目录不存在: {args.persist_dir}")
        print("请先运行离线建图: python scripts/build_index.py --input data/documents.json --persist_dir ./index")
        return
    
    # 加载引擎（在线模式）
    print("=" * 60)
    print("🔍 KGPRAG 在线检索")
    print("=" * 60)
    
    engine = AdvancedRAGEngine(persist_dir=args.persist_dir, online_mode=True)
    
    if args.interactive:
        # 交互模式
        print("\n📝 进入交互模式 (输入 'quit' 退出)")
        print("-" * 60)
        
        while True:
            try:
                query = input("\n🙋 问题: ").strip()
                if query.lower() in ["quit", "exit", "q"]:
                    print("👋 再见!")
                    break
                if not query:
                    continue
                
                print("\n🔄 检索中...")
                answer = engine.query(query, beam_width=args.beam_width, max_hops=args.max_hops)
                print(f"\n💡 答案: {answer}")
                
            except KeyboardInterrupt:
                print("\n👋 再见!")
                break
    
    elif args.query:
        # 单次查询
        print(f"\n🙋 问题: {args.query}")
        print("\n🔄 检索中...")
        answer = engine.query(args.query, beam_width=args.beam_width, max_hops=args.max_hops)
        print(f"\n💡 答案: {answer}")
    
    else:
        print("请指定 --query 或 --interactive")
        parser.print_help()
    
    engine.close()


if __name__ == "__main__":
    main()
