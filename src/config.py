"""
配置文件：集中管理所有配置项
"""
import os
import torch

# ================= Device Configuration =================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LLM Configuration =================
# [Local Config - Ollama Llama 8B]
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "ollama")  # Ollama 兼容 OpenAI 协议但不校验 key
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "llama3:8b")

# # [Original Config - DeepSeek API]
# # OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "sk-1408831cec78417d9a6024ac8e02dac4")
# # OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
# # LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")

# # [旧配置 - Local vLLM Llama/Qwen]
# # OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "EMPTY")
# # OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")
# # LLM_MODEL = os.environ.get("LLM_MODEL", "meta-llama-3-8b-instruct")
# # 或 Qwen2.5-14B-Instruct

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = OPENAI_BASE_URL

# ================= Embedding & Reranker =================
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

# ================= Entity & Relation Extraction =================
GLINER_MODEL = os.environ.get("GLINER_MODEL", "urchade/gliner_medium-v2.1")
REBEL_MODEL = os.environ.get("REBEL_MODEL", "Babelscape/rebel-large")

# 实体抽取标签（思路A：扩展标签）
ENTITY_LABELS = [
    "Person", "Organization", "Location", "Event", "Product", "Concept",
    "Work", "Facility", "Date", "Award", "Technology", "Sport", "Animal"
]
ENTITY_THRESHOLD = 0.2  # 降低阈值提高召回

# ================= Neo4j Configuration =================
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
# NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "9RP4s9YpWWSV:k3")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "9RP4s9YpWWSV:k3")

# ================= Search Configuration =================
DEFAULT_BEAM_WIDTH = 8  # 恢复为 5，比较类问题需要同时追踪多个实体
DEFAULT_MAX_HOPS = 3    # 保持 3 跳
TRUST_THRESHOLD = 0.01   # 保持 0.2，避免过早剪枝

# === 自适应搜索域策略 (Context-Aware Retrieval Strategy) ===
# 当 doc_filter 存在且大小 <= 此阈值时，采用"全量加载+重排"策略
# HotpotQA Distractor = 10 个文档，设为 20 留有余量
SMALL_SPACE_THRESHOLD = 20

# 小空间模式下，强制保留的最小候选数（防止被 TRUST_THRESHOLD 全部剪枝）
# 在只有 10 个文档时，相对排名 > 绝对阈值
MIN_CANDIDATES_KEEP = 5

# ================= Performance Tuning =================
# 批处理大小 (根据 GPU 显存调整)
# - 8GB 显存: 建议 2-4
# - 16GB 显存: 建议 8-12
# - 24GB 显存: 建议 16+
BATCH_SIZE = 16  # 8GB 显存安全值
# 早退策略：实体数少于此值时跳过 REBEL
MIN_ENTITIES_FOR_REBEL = 2
# REBEL beam search 数量 (降低可提速，但可能损失质量)
REBEL_NUM_BEAMS = 3

# ================= Entity Filtering Configuration =================
# 高频低信息实体停用词表（用于过滤无意义实体）
ENTITY_STOPWORDS = {
    # 时间类
    "year", "month", "day", "date", "time", "century", "decade", "week", "hour",
    "minute", "second", "period", "era", "age",
    # 通用地点
    "city", "country", "world", "place", "area", "region", "state", "town", "village",
    "district", "province", "county", "zone", "territory",
    # 通用人物/组织
    "man", "woman", "person", "people", "group", "company", "team", "member",
    "individual", "individuals", "organization", "institution",
    # 抽象概念
    "life", "history", "part", "end", "beginning", "work", "series", "type", "kind",
    "name", "number", "way", "thing", "fact", "case", "point", "example",
    "system", "method", "process", "result", "effect", "cause",
}

# 实体过滤参数
MIN_ENTITY_LENGTH = 2  # 最小实体长度（字符数）

# ================= Hard Edge Configuration =================
# 实体共现硬边构建参数
HARD_EDGE_ENTITY_TYPES = {"Person", "Location", "Organization", "Work", "Event", "Product", "Facility", "Sport", "Animal", "Technology", "Award", "Concept"}  # 扩展类型，确保覆盖 HotpotQA 的各类桥接实体
MIN_ENTITY_OCCURRENCES = 2   # 实体最少出现次数（低于此值不建立硬边）
MAX_ENTITY_OCCURRENCES = 50  # 实体最多出现次数（超过此值视为通用词，不建立硬边）
MAX_CHUNKS_PER_ENTITY_FOR_FULL_CONNECT = 20  # 当实体出现次数 <= 此值时，进行全连接
MAX_EDGES_PER_ENTITY = 50    # 每个实体最多产生的边数（采样时使用）
MIN_ENTITY_NAME_LENGTH = 2   # 实体名最短长度（用于硬边，避免短名误连）

print(f"[Config] Running on device: {DEVICE}")
