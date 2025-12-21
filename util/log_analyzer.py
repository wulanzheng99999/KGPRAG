import re
import statistics
#日志分析工具
def parse_log(file_path, limit=2000):
    stats = {
        "total": 0,
        "idk": 0,
        "miss": 0,  # F1 = 0 and NOT IDK
        "low_f1": 0, # 0 < F1 < 1
        "perfect": 0, # F1 = 1
        "latency": [],
        "doc_filter_count": 0,
        "f1_scores": []
    }
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Split by query blocks (roughly)
    # We'll use regex to find per-query blocks
    # 进度: [1/7405] ... to ... ----------------
    
    # Extract prediction lines
    preds = re.findall(r'🤖 预测: (.*)', content)
    # Extract metrics lines
    metrics = re.findall(r'📊 指标: F1=([0-9\.]+) \| P=([0-9\.]+) \| R=([0-9\.]+) \| (.*?) \(耗时: ([0-9\.]+)s\)', content)
    # Extract doc_filter lines
    filters = re.findall(r'📋 doc_filter: (\d+) 个文档', content)
    
    stats["total"] = min(len(metrics), limit)
    stats["doc_filter_count"] = len(filters)
    
    for i in range(stats["total"]):
        pred_text = preds[i].strip()
        f1 = float(metrics[i][0])
        latency = float(metrics[i][4])
        status_text = metrics[i][3]
        
        stats["latency"].append(latency)
        stats["f1_scores"].append(f1)
        
        is_idk = "i don't know" in pred_text.lower() or "idk" in status_text.lower() or "⚪" in status_text
        
        if is_idk:
            stats["idk"] += 1
        elif f1 == 0.0:
            stats["miss"] += 1
        elif f1 == 1.0:
            stats["perfect"] += 1
        else:
            stats["low_f1"] += 1
            
    return stats

def print_comparison(new_log, old_log):
    new_stats = parse_log(new_log)
    old_stats = parse_log(old_log)
    
    print(f"{'Metric':<25} | {'New (12-20)':<15} | {'Old (12-19)':<15} | {'Diff':<10}")
    print("-" * 70)
    
    # Sample Count
    print(f"{'Sample Count':<25} | {new_stats['total']:<15} | {old_stats['total']:<15} |")
    
    # IDK Rate
    new_idk_pct = (new_stats['idk'] / new_stats['total']) * 100
    old_idk_pct = (old_stats['idk'] / old_stats['total']) * 100
    print(f"{'IDK Rate (Give up)':<25} | {new_stats['idk']} ({new_idk_pct:.1f}%)   | {old_stats['idk']} ({old_idk_pct:.1f}%)   | {new_idk_pct - old_idk_pct:+.1f}%")
    
    # Miss Rate (Wrong Answer)
    new_miss_pct = (new_stats['miss'] / new_stats['total']) * 100
    old_miss_pct = (old_stats['miss'] / old_stats['total']) * 100
    print(f"{'Miss Rate (Wrong Ans)':<25} | {new_stats['miss']} ({new_miss_pct:.1f}%)   | {old_stats['miss']} ({old_miss_pct:.1f}%)   | {new_miss_pct - old_miss_pct:+.1f}%")
    
    # Perfect Match
    new_perf_pct = (new_stats['perfect'] / new_stats['total']) * 100
    old_perf_pct = (old_stats['perfect'] / old_stats['total']) * 100
    print(f"{'Perfect Match (F1=1.0)':<25} | {new_stats['perfect']} ({new_perf_pct:.1f}%)   | {old_stats['perfect']} ({old_perf_pct:.1f}%)   | {new_perf_pct - old_perf_pct:+.1f}%")
    
    # Latency
    new_avg_lat = statistics.mean(new_stats['latency'])
    old_avg_lat = statistics.mean(old_stats['latency'])
    print(f"{'Avg Latency (s)':<25} | {new_avg_lat:.2f}s          | {old_avg_lat:.2f}s          | {new_avg_lat - old_avg_lat:+.2f}s")
    
    # Avg F1 (Check)
    new_f1 = statistics.mean(new_stats['f1_scores'])
    old_f1 = statistics.mean(old_stats['f1_scores'])
    print(f"{'Avg F1 (Calculated)':<25} | {new_f1:.4f}          | {old_f1:.4f}          | {new_f1 - old_f1:+.4f}")

if __name__ == "__main__":
    print_comparison("logs/kgprag_eval_20251220_045601.log", "logs/evaluate_2025-12-19_163652.log")