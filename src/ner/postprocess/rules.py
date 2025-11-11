"""内置后处理规则

提供常用的后处理规则实现。
"""

from __future__ import annotations
from typing import List

from .pipeline import BaseRule, PredictionResult


# 规则名称常量
BIO_CONSISTENCY_RULE = 'bio_consistency'
CONFIDENCE_THRESHOLD_RULE = 'confidence_threshold'


class BIOConsistencyRule(BaseRule):
    """BIO标签一致性规则
    
    确保标签序列符合BIO规范：
    - I-X 必须跟在 B-X 或 I-X 后面
    - 孤立的 I-X 会被转换为 B-X 或 O
    - 类型不匹配的 I-X 会被修正
    
    示例：
        输入:  ['I-PER', 'I-PER', 'O', 'I-ORG']
        输出:  ['B-PER', 'I-PER', 'O', 'B-ORG']
    """
    
    name = BIO_CONSISTENCY_RULE
    description = "Ensure BIO tagging consistency"
    
    def __init__(self, fix_orphan_i: str = 'to_b'):
        """
        Args:
            fix_orphan_i: 如何处理孤立的I标签
                - 'to_b': 转换为B标签（默认）
                - 'to_o': 转换为O标签
        """
        self.fix_orphan_i = fix_orphan_i
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        """应用一致性规则"""
        labels = result.labels.copy()
        
        for i, label in enumerate(labels):
            if label.startswith('I-'):
                entity_type = label[2:]
                
                # 检查前一个标签
                if i == 0:
                    # 序列开头的I标签
                    labels[i] = self._fix_orphan(label, entity_type)
                else:
                    prev_label = labels[i - 1]
                    
                    if prev_label == 'O':
                        # I标签前面是O
                        labels[i] = self._fix_orphan(label, entity_type)
                    elif prev_label.startswith('B-') or prev_label.startswith('I-'):
                        prev_type = prev_label[2:]
                        if prev_type != entity_type:
                            # 类型不匹配
                            labels[i] = f'B-{entity_type}'
        
        return PredictionResult(
            tokens=result.tokens,
            labels=labels,
            confidences=result.confidences
        )
    
    def _fix_orphan(self, label: str, entity_type: str) -> str:
        """修正孤立的I标签"""
        if self.fix_orphan_i == 'to_b':
            return f'B-{entity_type}'
        elif self.fix_orphan_i == 'to_o':
            return 'O'
        else:
            return label


class ConfidenceThresholdRule(BaseRule):
    """置信度阈值规则
    
    将低于阈值的预测标签设置为'O'。
    
    示例：
        threshold=0.7
        输入:  labels=['B-PER', 'I-PER', 'B-ORG'], confidences=[0.9, 0.6, 0.8]
        输出:  labels=['B-PER', 'O', 'B-ORG'], confidences=[0.9, 0.6, 0.8]
    """
    
    name = CONFIDENCE_THRESHOLD_RULE
    description = "Filter predictions by confidence threshold"
    
    def __init__(self, threshold: float = 0.5, keep_entity_if_any_high: bool = False):
        """
        Args:
            threshold: 置信度阈值
            keep_entity_if_any_high: 如果实体中任何token高于阈值，保留整个实体
        """
        self.threshold = threshold
        self.keep_entity_if_any_high = keep_entity_if_any_high
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        """应用置信度阈值"""
        labels = result.labels.copy()
        
        if not self.keep_entity_if_any_high:
            # 简单模式：逐token过滤
            for i, (label, conf) in enumerate(zip(labels, result.confidences)):
                if label != 'O' and conf < self.threshold:
                    labels[i] = 'O'
        else:
            # 实体级过滤：如果实体中有任何token高于阈值，保留整个实体
            entities = self._extract_entity_spans(labels)
            low_confidence_entities = set()
            
            for start, end, entity_type in entities:
                max_conf = max(result.confidences[start:end])
                if max_conf < self.threshold:
                    low_confidence_entities.add((start, end))
            
            for i, label in enumerate(labels):
                if label != 'O':
                    # 检查这个token是否属于低置信度实体
                    for start, end in low_confidence_entities:
                        if start <= i < end:
                            labels[i] = 'O'
                            break
        
        return PredictionResult(
            tokens=result.tokens,
            labels=labels,
            confidences=result.confidences
        )
    
    def _extract_entity_spans(self, labels: List[str]) -> List[tuple]:
        """提取实体跨度"""
        entities = []
        current_entity = None
        
        for i, label in enumerate(labels):
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                entity_type = label[2:]
                current_entity = (i, i + 1, entity_type)
            elif label.startswith('I-') and current_entity:
                entity_type = label[2:]
                if current_entity[2] == entity_type:
                    current_entity = (current_entity[0], i + 1, entity_type)
                else:
                    entities.append(current_entity)
                    current_entity = (i, i + 1, entity_type)
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        if current_entity:
            entities.append(current_entity)
        
        return entities