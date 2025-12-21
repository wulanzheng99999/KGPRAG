"""
Min-RAG Query Engine (DeepSeek 版)
实现功能：向量检索 -> 图谱关联扩展 -> 最终问答
"""

import os
from typing import List, Set

# 修复 Warning，使用新的包
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from neo4j import GraphDatabase

# ==============================================================================
# 🔴 配置 (与入库时保持一致)
# ==============================================================================
os.environ["OPENAI_API_KEY"] = "sk-7b097abdf68f4e91ad414703d6e20f7a"
os.environ["OPENAI_BASE_URL"] = "https://api.deepseek.com"
LLM_MODEL = "deepseek-chat"  # 问答也可以用 deepseek-reasoner (R1) 效果更好

CHROMA_DIR = "./chroma_db_deepseek"
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "9RP4s9YpWWSV:k3")
# ==============================================================================

def get_llm():
    return ChatOpenAI(model=LLM_MODEL, temperature=0.3)

def get_vector_store():
    print("[Init] 加载向量库...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/BGE-M3")
    return Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)

def get_neo4j_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

# === 核心函数 1: 向量检索 (Hop 1) ===
def dense_retrieval(vector_store, query: str, top_k=1):
    print(f"\n🔍 [Hop 1] 向量检索: '{query}'")
    docs = vector_store.similarity_search(query, k=top_k)

    results = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id")
        text = doc.page_content
        source = doc.metadata.get("source")
        print(f"   -> 找到: [{chunk_id}] 来自 {source}")
        results.append({"id": chunk_id, "text": text, "source": source})
    return results

# === 核心函数 2: 图谱扩展 (Hop 2) ===
def graph_expansion(driver, seed_chunks: List[dict]):
    """
    论文核心算法简化版：
    Seed Chunks -> 包含的实体 -> 这些实体连接的其他 Chunks
    """
    print(f"\n🕸️ [Hop 2] 图谱扩展 (Looking for bridges...)")
    if not seed_chunks:
        return []

    expanded_chunks = []
    seed_ids = [c["id"] for c in seed_chunks]

    with driver.session() as session:
        # Cypher 查询逻辑：
        # 1. 找到 Seed Chunk 提到的实体 (e)
        # 2. 找到也提到这些实体 (e) 的其他 Chunk (other_c)
        # 3. 返回这些 chunk 的文本
        query = """
        MATCH (seed:Chunk)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(other:Chunk)
        WHERE seed.id IN $seed_ids AND NOT other.id IN $seed_ids
        RETURN DISTINCT other.id, other.text, other.source, collect(e.name) as bridges
        LIMIT 3
        """
        result = session.run(query, seed_ids=seed_ids)

        for record in result:
            c_id = record["other.id"]
            text = record["other.text"]
            source = record["other.source"]
            bridges = record["bridges"]

            print(f"   -> 扩展到: [{c_id}] (桥梁实体: {bridges}) 来自 {source}")
            expanded_chunks.append({"id": c_id, "text": text, "source": source})

    if not expanded_chunks:
        print("   -> (无关联扩展内容)")

    return expanded_chunks

# === 核心函数 3: 生成回答 ===
def generate_answer(llm, query, context_chunks):
    print(f"\n🤖 [Gen] 正在思考...")

    # 拼装上下文
    context_str = ""
    for i, c in enumerate(context_chunks):
        context_str += f"--- 文档片段 {i+1} (来源: {c['source']}) ---\n{c['text']}\n\n"

    prompt = f"""
请基于以下参考信息回答用户问题。
如果参考信息中有矛盾或不同角度的描述，请综合分析。

参考信息：
{context_chunks}

用户问题：{query}

回答：
"""
    # 打印完整的 Prompt 方便调试 (可选)
    # print(f"--- Prompt ---\n{prompt}\n----------------")

    response = llm.invoke(prompt)
    return response.content

def main():
    vs = get_vector_store()
    driver = get_neo4j_driver()
    llm = get_llm()

    # === 测试用例 ===
    # 这是一个典型的“多跳”问题：
    # 1. 向量检索能搜到“Alpha CEO”是张三，发布了 SkyBrain。
    # 2. 但是“市场前景”在向量库里很难直接匹配（因为研报里没提张三的名字）。
    # 3. 图谱扩展通过 "SkyBrain" 实体，把研报抓取进来，从而回答“前景不佳”。
    user_query = "Alpha 公司 CEO 发布的最新芯片，在当前的行业分析中面临什么样的市场前景？"

    # 1. Hop 1
    seed_chunks = dense_retrieval(vs, user_query, top_k=1)

    # 2. Hop 2
    expanded_chunks = graph_expansion(driver, seed_chunks)

    # 3. 合并上下文
    all_context = seed_chunks + expanded_chunks

    # 4. 生成
    answer = generate_answer(llm, user_query, all_context)

    print("\n" + "="*50)
    print(f"用户提问: {user_query}")
    print("-" * 50)
    print(f"Min-RAG 回答:\n{answer}")
    print("="*50)

    driver.close()

if __name__ == "__main__":
    main()