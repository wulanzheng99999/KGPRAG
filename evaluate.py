"""
HotpotQA 评估脚本 (统一通用版)

功能：
1. 自动识别数据集模式 (FullWiki vs Distractor)
2. 自动切换索引目录 (hotpotqa_fullwiki vs hotpotqa)
3. 自动隔离日志和结果文件 (logs/fullwiki vs logs/distractor)
4. 统一的强制过滤逻辑 (基于 sample_doc_mapping)

用法:
   # 运行 Distractor 模式 (默认)
   python evaluate.py --input data/hotpot_dev_distractor_v1.json

   # 运行 FullWiki 模式 (自动切换)
   python evaluate.py --input data/hotpot_dev_fullwiki_v1.json
"""
import json
import os
import time
import sys
import warnings
import argparse
from pathlib import Path
from typing import List, Dict, Any
from dataclasses import dataclass

# --- 屏蔽烦人的 transformers 警告 ---
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# --- 解决 KMeans 警告 ---
os.environ["OMP_NUM_THREADS"] = "1"

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
    # 0. 参数解析
    parser = argparse.ArgumentParser(description="HotpotQA 统一评估脚本")
    parser.add_argument("--input", default="data/hotpot_dev_distractor_v1.json", help="输入数据集路径")
    parser.add_argument("--limit", type=int, default=None, help="仅测试前 N 条数据 (调试用)")
    args = parser.parse_args()

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_FILE = os.path.join(PROJECT_ROOT, args.input)

    # 1. 自动识别模式 & 设置隔离路径
    input_filename = os.path.basename(args.input).lower()
    if "fullwiki" in input_filename:
        mode = "fullwiki"
        index_dir_name = "hotpotqa_fullwiki"
    else:
        mode = "distractor"
        index_dir_name = "hotpotqa"

    # 构建路径
    INDEX_DIR = os.path.join(PROJECT_ROOT, "data", index_dir_name)
    LOG_BASE_DIR = os.path.join(PROJECT_ROOT, "logs", mode)
    OUTPUT_FILE = os.path.join(PROJECT_ROOT, "data", f"results_{mode}.jsonl")
    
    # 确保日志目录存在
    os.makedirs(LOG_BASE_DIR, exist_ok=True)

    # 2. 初始化日志
    file_prefix = f"{mode}_eval"
    logger = ExperimentLogger(log_dir=LOG_BASE_DIR, experiment_name=file_prefix)
    
    logger.info("\n" + "=" * 80)
    logger.info(f"🚀 启动 KGPRAG 评估 | 模式: {mode.upper()}")
    logger.info("=" * 80)
    logger.info(f"   📂 输入数据: {DATA_FILE}")
    logger.info(f"   📂 索引目录: {INDEX_DIR}")
    logger.info(f"   📂 日志目录: {LOG_BASE_DIR}")
    logger.info(f"   💾 结果输出: {OUTPUT_FILE}")
    logger.info("   🔒 过滤策略: FullWiki：doc_filter=None / Distractor：doc_filter启用")
    logger.info("=" * 80 + "\n")
    
    # 检查索引目录
    if not os.path.exists(INDEX_DIR) or not os.path.exists(os.path.join(INDEX_DIR, "doc_cache.json")):
        logger.error(f"❌ 索引目录不存在或未构建: {INDEX_DIR}")
        logger.error("请先运行对应的建图脚本 (scripts/build_hotpot_global_kg.py)")
        exit(1)
    
    # 3. 加载样本-文档映射 (用于 doc_filter)
    mapping_path = os.path.join(INDEX_DIR, "sample_doc_mapping.json")
    if not os.path.exists(mapping_path):
        logger.error(f"❌ 样本-文档映射文件不存在: {mapping_path}")
        logger.error("请重新运行建图脚本以生成映射文件")
        exit(1)
    
    with open(mapping_path, "r", encoding="utf-8") as f:
        sample_doc_mapping = json.load(f)
    logger.info(f"📁 加载文档映射: {len(sample_doc_mapping)} 条记录")

    # 4. 初始化引擎
    try:
        engine = AdvancedRAGEngine(persist_dir=INDEX_DIR, online_mode=True)
    except Exception as e:
        logger.error(f"❌ 引擎初始化失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        exit()

    # 5. 加载数据
    loader = HotpotQALoader(DATA_FILE, logger=logger)
    loader.load()
    if not loader.data:
        logger.error("❌ 数据为空，退出。")
        exit()

    # 6. 评测循环
    MAX_TESTS = args.limit if args.limit else len(loader.data)
    BEAM_WIDTH = DEFAULT_BEAM_WIDTH
    MAX_HOPS = DEFAULT_MAX_HOPS

    count = 0
    total_f1 = 0.0
    total_precision = 0.0
    total_recall = 0.0
    retrieval_hit_count = 0
    bridge_hit_count = 0
    path_hit_count = 0

    logger.info(f"\n🎯 开始评测 (计划跑 {MAX_TESTS} 条)")
    
    # 清空旧的结果文件
    if os.path.exists(OUTPUT_FILE):
        try:
            os.remove(OUTPUT_FILE)
        except OSError:
            pass

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for batch in loader.get_batch(batch_size=1):
            for item in batch:
                count += 1
                logger.info(f"{'-' * 60}")
                logger.info(f"进度: [{count}/{MAX_TESTS}] | Query ID: {item.id}")
                logger.info(f"❓ 问题: {item.query}")

                start_time = time.time()
                try:
                    # --- doc_filter 策略 ---
                    if mode == "fullwiki":
                        # FullWiki 模式：强制全库检索 (Open Space)
                        # 忽略 sample_doc_mapping 中的 context，因为它们可能是不完整的或误导性的
                        doc_filter_ids = None
                        logger.info("🌐 检索范围: 全库检索 (Open Space Mode)")
                    else:
                        # Distractor 模式：强制限定范围 (Closed Space)
                        doc_filter_ids = set(sample_doc_mapping.get(item.id, []))
                        if not doc_filter_ids:
                            logger.warning(f"⚠️ 样本 {item.id} 在映射表中未找到文档 ID！跳过此样本。")
                            continue
                        logger.info(f"📋 检索范围: 限定 {len(doc_filter_ids)} 个文档")

                    # --- 执行查询 ---
                    result = engine.query(
                        item.query,
                        beam_width=BEAM_WIDTH,
                        max_hops=MAX_HOPS,
                        doc_filter=doc_filter_ids,
                        return_debug=True
                    )

                    if isinstance(result, tuple):
                        prediction, debug_info = result
                    else:
                        prediction, debug_info = result, {}

                    # --- 统计命中率 ---
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

                    if retrieval_hit: retrieval_hit_count += 1
                    if bridge_hit: bridge_hit_count += 1
                    if path_hit: path_hit_count += 1

                    # --- 计算指标 ---
                    f1, precision, recall = calculate_metrics(prediction, item.answer_ground_truth)

                    # --- 语义裁判 (可选) ---
                    semantic_consistency = "N/A"
                    norm_pred = normalize_answer(prediction)
                    status_icon = "❌ MISS"
                    
                    # 简单的状态图标逻辑
                    if "i don't know" in norm_pred:
                        status_icon = "⚪ IDK"
                    elif f1 >= 0.5:
                        status_icon = f"🎉 High F1 ({f1:.2f})"
                    elif f1 > 0:
                        status_icon = f"⚠️ Low F1 ({f1:.2f})"

                    duration = time.time() - start_time

                    logger.info(f"🤖 预测: {prediction.strip()}")
                    logger.info(f"✅ 真值: {item.answer_ground_truth}")
                    logger.info(f"📊 指标: F1={f1:.2f} | P={precision:.2f} | R={recall:.2f} | {status_icon} (耗时: {duration:.2f}s)")
                    
                    # 写入结果
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
                        "mode": mode
                    }
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()

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

    # 7. 最终统计
    logger.info(f"\n{'=' * 60}")
    logger.info(f"🏁 {mode.upper()} 测试结束")
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
