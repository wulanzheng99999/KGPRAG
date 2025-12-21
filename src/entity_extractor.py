"""
实体与关系抽取模块
- GLiNER: 实体抽取
- REBEL: 关系抽取
"""
from __future__ import annotations

import re
from typing import List, Dict

import torch
import flair
from gliner import GLiNER
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.config import (
    DEVICE, GLINER_MODEL, REBEL_MODEL, 
    ENTITY_LABELS, ENTITY_THRESHOLD, BATCH_SIZE,
    ENTITY_STOPWORDS, MIN_ENTITY_LENGTH
)
from util.text_utils import normalize_entity


class EntityExtractor:
    """双模型实体与关系抽取器"""
    
    def __init__(self):
        print(f"📦 加载实体模型: {GLINER_MODEL}")
        self.entity_model = GLiNER.from_pretrained(GLINER_MODEL)
        if DEVICE == "cuda":
            self.entity_model.to("cuda")
        
        print(f"📦 加载关系抽取模型: {REBEL_MODEL}")
        self.rebel_tokenizer = AutoTokenizer.from_pretrained(REBEL_MODEL)
        self.rebel_model = AutoModelForSeq2SeqLM.from_pretrained(REBEL_MODEL)
        if DEVICE == "cuda":
            self.rebel_model.to("cuda")
        self.rebel_model.eval()
    
    @staticmethod
    def normalize_entity(entity_text: str) -> str:
        """委托给 util.text_utils"""
        return normalize_entity(entity_text)
    
    def _should_filter_entity(self, entity_text: str, entity_label: str = None, is_query: bool = False) -> bool:
        """
        判断实体是否应被过滤（在归一化之后判断）
        返回 True 表示应该过滤掉
        
        参数:
            entity_text: 原始实体文本
            entity_label: 实体类型（可选，用于未来扩展）
            is_query: 是否为查询场景（查询场景下放宽过滤，保留更多桥接实体）
        """
        # 先归一化
        normalized = self.normalize_entity(entity_text)
        
        # 1. 空值过滤（始终生效）
        if not normalized:
            return True
        
        # 2. 长度过滤（查询场景下放宽：允许长度 >= 1）
        min_len = 1 if is_query else MIN_ENTITY_LENGTH
        if len(normalized) < min_len:
            return True
        
        # 3. 停用词过滤（查询场景下跳过，保留所有用户意图词）
        if not is_query and normalized in ENTITY_STOPWORDS:
            return True
        
        # 4. 纯数字过滤（但保留4位年份格式，如 1990, 2000）
        # 查询场景下保留所有数字（用户可能问 "in 2015..."）
        if not is_query and normalized.isdigit():
            if len(normalized) != 4:  # 非年份的纯数字过滤掉
                return True
        
        return False
    
    def extract_entities(self, text: str) -> Dict[str, str]:
        """
        使用 GLiNER 抽取实体
        返回: {归一化实体名: 实体类型}
        """
        try:
            # 截断过长文本，防止 OOM
            text = text[:3000]
            
            with torch.no_grad():
                ents = self.entity_model.predict_entities(
                    text, ENTITY_LABELS, threshold=ENTITY_THRESHOLD
                )
            # 归一化 + 过滤 + 去重
            unique_ents = {
                self.normalize_entity(e["text"]): e["label"] 
                for e in ents 
                if self.normalize_entity(e["text"]) 
                and not self._should_filter_entity(e["text"], e["label"])
            }
            return unique_ents
        except Exception as e:
            print(f"⚠️ Entity Extraction Error: {e}")
            return {}
    
    def extract_query_entities(self, query: str) -> List[str]:
        """
        从用户问题中提取实体，用于引导多跳检索
        """
        try:
            ents = self.entity_model.predict_entities(
                query, ENTITY_LABELS, threshold=ENTITY_THRESHOLD
            )
            # 归一化 + 过滤 + 去重
            # 查询场景：放宽过滤，保留更多桥接实体（is_query=True）
            entity_names = list({
                self.normalize_entity(e["text"]) 
                for e in ents 
                if self.normalize_entity(e["text"])
                and not self._should_filter_entity(e["text"], e["label"], is_query=True)
            })
            if entity_names:
                print(f"🎯 Query Entities (normalized): {entity_names}")
            return entity_names
        except Exception as e:
            print(f"⚠️ Query Entity Extraction Error: {e}")
            return []
    
    def extract_relations(self, text: str) -> List[Dict]:
        """
        使用 REBEL 模型抽取关系三元组 (head, relation, tail)
        """
        relations = []
        try:
            # 截断过长文本避免 OOM
            text_truncated = text[:512]
            
            # Tokenize
            inputs = self.rebel_tokenizer(
                text_truncated, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            if DEVICE == "cuda":
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = self.rebel_model.generate(
                    **inputs,
                    max_length=256,
                    num_beams=3,
                    num_return_sequences=1
                )
            
            # Decode
            decoded = self.rebel_tokenizer.batch_decode(outputs, skip_special_tokens=False)[0]
            
            # Parse REBEL output
            relations = self._parse_rebel_output(decoded)
            
        except Exception as e:
            print(f"⚠️ REBEL Extraction Error: {e}")
        
        return relations
    
    def extract_entities_batch(self, texts: List[str]) -> List[Dict[str, str]]:
        """
        批量实体抽取 (高效，支持分批处理避免 OOM)
        返回: [{归一化实体名: 实体类型}, ...]
        """
        results = []
        try:
            # 分批处理，避免 OOM
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                
                # 防御性截断
                batch_texts = [t[:3000] for t in batch_texts]
                
                # GLiNER 支持批量预测
                with torch.no_grad():
                    all_ents = self.entity_model.batch_predict_entities(
                        batch_texts, ENTITY_LABELS, threshold=ENTITY_THRESHOLD
                    )
                for ents in all_ents:
                    unique_ents = {
                        self.normalize_entity(e["text"]): e["label"]
                        for e in ents
                        if self.normalize_entity(e["text"])
                        and not self._should_filter_entity(e["text"], e["label"])
                    }
                    results.append(unique_ents)
        except Exception as e:
            print(f"⚠️ Batch Entity Extraction Error: {e}, falling back to sequential")
            # 回退到串行处理
            results = []
            for text in texts:
                results.append(self.extract_entities(text))
        return results
    
    def extract_relations_batch(self, texts: List[str]) -> List[List[Dict]]:
        """
        批量关系抽取 (高效，支持分批处理避免 OOM)
        返回: [[{source, target, type}, ...], ...]
        """
        if not texts:
            return []
        
        results = []
        try:
            # 分批处理，避免 OOM
            for i in range(0, len(texts), BATCH_SIZE):
                batch_texts = texts[i:i + BATCH_SIZE]
                
                # 截断所有文本
                texts_truncated = [t[:512] for t in batch_texts]
                
                # 批量 Tokenize
                inputs = self.rebel_tokenizer(
                    texts_truncated,
                    return_tensors="pt",
                    max_length=512,
                    truncation=True,
                    padding=True  # 批处理需要 padding
                )
                if DEVICE == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                # 批量 Generate
                with torch.no_grad():
                    outputs = self.rebel_model.generate(
                        **inputs,
                        max_length=256,
                        num_beams=3,
                        num_return_sequences=1
                    )
                
                # 批量 Decode
                decoded_batch = self.rebel_tokenizer.batch_decode(outputs, skip_special_tokens=False)
                
                # 解析每个输出
                for decoded in decoded_batch:
                    relations = self._parse_rebel_output(decoded)
                    results.append(relations)
                
        except Exception as e:
            print(f"⚠️ Batch REBEL Error: {e}, falling back to sequential")
            # 回退到串行处理
            results = []
            for text in texts:
                results.append(self.extract_relations(text))
        
        return results
    
    def _parse_rebel_output(self, text: str) -> List[Dict]:
        """
        解析 REBEL 输出格式
        格式: <triplet> head <subj> relation <obj> tail <triplet> ...
        """
        relations = []
        
        # 清理特殊 token
        text = text.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
        
        # 按 <triplet> 分割
        triplets = text.split("<triplet>")
        
        for triplet in triplets:
            triplet = triplet.strip()
            if not triplet:
                continue
            
            try:
                # 提取 head
                if "<subj>" in triplet:
                    head = triplet.split("<subj>")[0].strip()
                    rest = triplet.split("<subj>")[1]
                else:
                    continue
                
                # 提取 relation 和 tail
                if "<obj>" in rest:
                    relation = rest.split("<obj>")[0].strip()
                    tail = rest.split("<obj>")[1].strip()
                else:
                    continue
                
                # 归一化实体名
                head_norm = self.normalize_entity(head)
                tail_norm = self.normalize_entity(tail)
                
                if head_norm and tail_norm and head_norm != tail_norm:
                    relations.append({
                        "source": head_norm,
                        "target": tail_norm,
                        "type": relation.upper().replace(" ", "_")
                    })
            except Exception:
                continue
        
        return relations
