"""
文本处理工具函数
"""
import re


def normalize_entity(entity_text: str) -> str:
    """
    实体别名归一化
    去除冠词、标点，统一小写
    """
    if not entity_text:
        return ""
    # 去除开头的冠词
    normalized = re.sub(r'^(the|a|an)\s+', '', entity_text.lower())
    # 去除标点符号（保留字母数字空格）
    normalized = re.sub(r'[^\w\s]', '', normalized)
    return normalized.strip()
