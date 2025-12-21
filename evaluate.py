"""
HotpotQA 评估脚本（全量建图 + 限定检索范围模式）

对标 KG2RAG 论文的实验设置：
- 离线构建全量知识图谱（所有唯一文档）
- 评测时通过 doc_filter 限制检索范围到每个问题的 10 个文档

使用方法：
1. 先运行全量建图脚本：
   python scripts/build_hotpot_global_kg.py --input data/hotpot_dev_distractor_v1.json --persist_dir ./data/hotpotqa --reset
   
2. 再运行评测：
   python evaluate.py
"""
import json
import os
import time
import sys
import warnings

# --- 屏蔽烦人的 transformers 警告 ---
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- 解决 KMeans 警告 ---
os.environ["OMP_NUM_THREADS"] = "1"

from typing import List, Dict, Any
from dataclasses import dataclass

# 引入模块化引擎
from src.engine import AdvancedRAGEngine
# 引入日志工具
from util.custom_logger import ExperimentLogger
# 引入配置
from src.config import DEFAULT_BEAM_WIDTH, DEFAULT_MAX_HOPS
# 使用官方评测脚本统一逻辑
from util.hotpot_evaluate_v1 import f1_score as calculate_metrics, normalize_answer


@dataclass
class MinRAGInput:
    id: str
    query: str
    documents: List[Dict[str, str]]
    answer_ground_truth: str
    supporting_facts_ground_truth: List[List[Any]]


class HotpotQALoader:
    def __init__(self, file_path: str, logger=None):
        self.file_path = file_path
        self.logger = logger
        self.data = []

    def log(self, msg):
        if self.logger:
            self.logger.info(msg)
        else:
            print(msg)

    def load(self):
        self.log(f"📂 正在加载数据集: {self.file_path} ...")
        if not os.path.exists(self.file_path):
            self.log(f"❌ 错误: 找不到文件 {self.file_path}")
            return

        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.log(f"✅ 成功加载 {len(self.data)} 条数据。")

    def process_sample(self, raw_sample: Dict) -> MinRAGInput:
        """
        将 HotpotQA 的原始数据转换为 Engine 需要的格式。
        """
        documents = []
        for item in raw_sample['context']:
            title = item[0]
            sentences = item[1]
            text_content = ' '.join(sentences)
            documents.append({
                "title": title,
                "text": text_content
            })

        return MinRAGInput(
            id=raw_sample['_id'],
            query=raw_sample['question'],
            documents=documents,
            answer_ground_truth=raw_sample['answer'],
            supporting_facts_ground_truth=raw_sample['supporting_facts']
        )

    def get_batch(self, batch_size: int = 1):
        for i in range(0, len(self.data), batch_size):
            batch_raw = self.data[i: i + batch_size]
            yield [self.process_sample(sample) for sample in batch_raw]


# ==========================================
# 🚀 执行主程序
# ==========================================
if __name__ == "__main__":
    # 1. 初始化日志系统
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    logger = ExperimentLogger(log_dir=os.path.join(PROJECT_ROOT, "logs"), experiment_name="kgprag_eval")
    
    # 使用绝对路径引用数据
    DATA_FILE = os.path.join(PROJECT_ROOT, "data", "hotpot_dev_distractor_v1.json")
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", "advanced_rag_results.jsonl")
    # 全量图谱索引目录（需与 scripts/build_hotpot_global_kg.py 的 --persist_dir 保持一致）
    INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "hotpotqa")

    # 2. 检查索引目录
    logger.info("\n" + "=" * 80)
    logger.info("🚀 正在启动 Advanced GraphRAG 引擎 (全量建图 + 限定检索模式)...")
    logger.info(f"   📂 索引目录: {INDEX_DIR}")
    logger.info("   ✅ 对标 KG2RAG 论文实验设置")
    logger.info("   ✅ 全量图谱 + doc_filter 限制检索范围")
    logger.info("   ✅ 多信号可信度评分: Enabled")
    logger.info("=" * 80 + "\n")
    
    # 检查索引目录是否存在
    if not os.path.exists(INDEX_DIR) or not os.path.exists(os.path.join(INDEX_DIR, "doc_cache.json")):
        logger.error(f"❌ 索引目录不存在或索引未构建: {INDEX_DIR}")
        logger.error("请先运行全量建图脚本:")
        logger.error("  python scripts/build_hotpot_global_kg.py --input data/hotpot_dev_distractor_v1.json --persist_dir ./index --reset")
        exit(1)
    
    # 加载样本-文档映射
    mapping_path = os.path.join(INDEX_DIR, "sample_doc_mapping.json")
    if not os.path.exists(mapping_path):
        logger.error(f"❌ 样本-文档映射文件不存在: {mapping_path}")
        logger.error("请重新运行全量建图脚本")
        exit(1)
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        sample_doc_mapping = json.load(f)
    logger.info(f"📁 加载样本-文档映射: {len(sample_doc_mapping)} 个样本")

    try:
        # 使用持久化模式加载全量图谱
        engine = AdvancedRAGEngine(persist_dir=INDEX_DIR, online_mode=True)
    except Exception as e:
        logger.error(f"❌ 引擎初始化失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit()

    # 3. 加载数据
    loader = HotpotQALoader(DATA_FILE, logger=logger)
    loader.load()
    if not loader.data:
        logger.error("❌ 数据未加载，程序退出。")
        exit()

    # 4. 配置测试参数（Beam/Hops 使用 config 默认值）
    MAX_TESTS = 7405
    BEAM_WIDTH = DEFAULT_BEAM_WIDTH
    MAX_HOPS = DEFAULT_MAX_HOPS

    count = 0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    retrieval_hit_count = 0
    bridge_hit_count = 0
    path_hit_count = 0

    logger.info(f"\n🎯 开始评测 (跑 {MAX_TESTS} 条数据)")
    logger.info(f"⚙️ 参数: Beam Width={BEAM_WIDTH}, Hops={MAX_HOPS}")
    logger.info(f"💾 结果将实时保存到: {OUTPUT_FILE}\n")

    # 清空旧的结果文件
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:

        for batch in loader.get_batch(batch_size=1):
            for item in batch:
                count += 1
                logger.info(f"{'-' * 60}")
                logger.info(f"进度: [{count}/{MAX_TESTS}] | Query ID: {item.id}")
                logger.info(f"❓ 问题: {item.query}")

                start_time = time.time()
                try:
                    # --- Step A: 获取该样本的文档过滤器 (HotpotQA-Dist 设置) ---
                    doc_filter = None
                    if item.id in sample_doc_mapping:
                        doc_filter = set(sample_doc_mapping[item.id])
                        logger.info(f"📋 doc_filter: {len(doc_filter)} 个文档")
                    else:
                        logger.warning(f"⚠️ 样本 {item.id} 未找到文档映射，使用全库检索")

                    # --- Step B: 多跳推理（使用 doc_filter 限制检索范围）---
                    result = engine.query(
                        item.query,
                        beam_width=BEAM_WIDTH,
                        max_hops=MAX_HOPS,
                        doc_filter=doc_filter,
                        return_debug=True
                    )
                    if isinstance(result, tuple):
                        prediction, debug_info = result
                    else:
                        prediction, debug_info = result, {}

                    search_result = debug_info.get("search_result", {})
                    retrieved_nodes = search_result.get("nodes", [])
                    if item.supporting_facts_ground_truth:
                        supporting_titles = {sf[0] for sf in item.supporting_facts_ground_truth}
                    else:
                        supporting_titles = set()
                    retrieved_titles = {
                        n.get("doc_title") for n in retrieved_nodes if n.get("doc_title")
                    }
                    hit_titles = supporting_titles & retrieved_titles
                    support_needed = min(2, len(supporting_titles))

                    retrieval_hit = len(hit_titles) >= 1 if supporting_titles else False
                    bridge_hit = support_needed > 0 and len(hit_titles) >= support_needed

                    path_titles = set(search_result.get("best_path_doc_titles", []))
                    path_hit = support_needed > 0 and len(path_titles & supporting_titles) >= support_needed

                    if retrieval_hit:
                        retrieval_hit_count += 1
                    if bridge_hit:
                        bridge_hit_count += 1
                    if path_hit:
                        path_hit_count += 1

                    # --- Step D: 评估 ---
                    f1, precision, recall = calculate_metrics(prediction, item.answer_ground_truth)

                    # --- Step E: LLM 语义裁判 ---
                    semantic_consistency = "N/A"
                    norm_pred = normalize_answer(prediction)
                    
                    if f1 < 0.8 and prediction.strip() and "i don't know" not in norm_pred:
                        try:
                            judge_prompt = f"""
                            Act as an objective judge. compare the Prediction and the Ground Truth.
                            Are they referring to the same entity, person, time, or event? 
                            Or is the Prediction a valid valid subset/synonym of the Ground Truth?
                            
                            Prediction: "{prediction}"
                            Ground Truth: "{item.answer_ground_truth}"
                            
                            Return ONLY 'yes' or 'no'.
                            """
                            from langchain_core.messages import HumanMessage
                            judge_res = engine.llm.invoke([HumanMessage(content=judge_prompt)]).content.lower().strip()
                            
                            if "yes" in judge_res:
                                semantic_consistency = "Consistent"
                                status_icon = "✅ Sem-Match"
                            else:
                                semantic_consistency = "Different"
                        except Exception as e:
                            logger.error(f"Judge Error: {e}")

                    # 判定与图标
                    if semantic_consistency != "Consistent":
                        if "i don't know" in norm_pred:
                            status_icon = "⚪ IDK"
                        elif f1 >= 0.5:
                            status_icon = f"🎉 High F1 ({f1:.2f})"
                        elif f1 > 0:
                            status_icon = f"⚠️ Low F1 ({f1:.2f})"
                        else:
                            status_icon = "❌ MISS"

                    duration = time.time() - start_time

                    logger.info(f"🤖 预测: {prediction.strip()}")
                    logger.info(f"✅ 真值: {item.answer_ground_truth}")
                    if semantic_consistency == "Consistent":
                        logger.info(f"⚖️ 裁判: ✅ 语义一致 (虽然 F1={f1:.2f})")
                    logger.info(f"📊 指标: F1={f1:.2f} | P={precision:.2f} | R={recall:.2f} | {status_icon} (耗时: {duration:.2f}s)")
                    logger.info(
                        f"🔎 Hits: retrieval={int(retrieval_hit)} "
                        f"bridge={int(bridge_hit)} path={int(path_hit)} "
                        f"| support={len(supporting_titles)} hit_docs={len(hit_titles)}"
                    )

                    # 写入文件
                    record = {
                        "id": item.id,
                        "query": item.query,
                        "prediction": prediction,
                        "ground_truth": item.answer_ground_truth,
                        "metrics": {
                            "f1": f1,
                            "precision": precision,
                            "recall": recall,
                            "semantic": semantic_consistency
                        },
                        "duration": duration,
                        "retrieval": {
                            "support_titles": sorted(supporting_titles),
                            "retrieved_title_count": len(retrieved_titles),
                            "hit_titles": sorted(hit_titles),
                            "retrieval_hit": retrieval_hit,
                            "bridge_hit": bridge_hit,
                            "path_hit": path_hit,
                        },
                        "method": "Advanced_GraphRAG_Modular"
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()

                    # 累计
                    total_f1 += f1
                    total_precision += precision
                    total_recall += recall

                except Exception as e:
                    logger.error(f"❌ 处理出错: {e}")
                    import traceback
                    logger.error(traceback.format_exc())

                if count >= MAX_TESTS:
                    break

            if count >= MAX_TESTS:
                break

    # 5. 打印最终统计
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🏁 测试结束")
    if count > 0:
        logger.info(f"📈 Avg F1:        {total_f1 / count:.4f}")
        logger.info(f"📈 Avg Precision: {total_precision / count:.4f}")
        logger.info(f"📈 Avg Recall:    {total_recall / count:.4f}")
        logger.info(f"📌 Retrieval Hit: {retrieval_hit_count / count:.4f} ({retrieval_hit_count}/{count})")
        logger.info(f"📌 Bridge Hit:    {bridge_hit_count / count:.4f} ({bridge_hit_count}/{count})")
        logger.info(f"📌 Path Hit:      {path_hit_count / count:.4f} ({path_hit_count}/{count})")
    else:
        logger.info("没有处理任何数据。")
    logger.info(f"💾 详细结果已保存至: {OUTPUT_FILE}")
    logger.info(f"📝 完整日志已保存至: {logger.get_log_path()}")

    engine.close()
