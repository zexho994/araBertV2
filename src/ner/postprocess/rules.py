"""内置后处理规则

提供常用的后处理规则实现。
"""

from __future__ import annotations
from typing import List, Optional, Set, Dict
import re

from .pipeline import BaseRule, PredictionResult


# 规则名称常量
BIO_CONSISTENCY_RULE = 'bio_consistency'
CONFIDENCE_THRESHOLD_RULE = 'confidence_threshold'
ENTITY_BOUNDARY_RULE = 'entity_boundary'
ENTITY_WHITELIST_RULE = 'entity_whitelist'
ENTITY_BLACKLIST_RULE = 'entity_blacklist'
MIN_ENTITY_LENGTH_RULE = 'min_entity_length'
PATTERN_CORRECTION_RULE = 'pattern_correction'
MERGE_ADJACENT_RULE = 'merge_adjacent'


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


class EntityBoundaryRule(BaseRule):
    """实体边界规则
    
    修正实体边界问题：
    - 移除边界的标点符号
    - 移除边界的停用词
    - 修正空白token
    
    示例：
        输入:  tokens=['Mr.', 'John', 'Smith', ','], labels=['B-PER', 'I-PER', 'I-PER', 'I-PER']
        输出:  tokens=['Mr.', 'John', 'Smith', ','], labels=['O', 'B-PER', 'I-PER', 'O']
    """
    
    name = ENTITY_BOUNDARY_RULE
    description = "Fix entity boundary issues"
    
    # 默认要从边界移除的标点
    DEFAULT_BOUNDARY_PUNCT = {'.', ',', '!', '?', ':', ';', '-', '(', ')', '[', ']', '{', '}', '"', "'"}
    
    # 默认要从边界移除的停用词（可根据语言扩展）
    DEFAULT_BOUNDARY_STOPWORDS = {'the', 'a', 'an', 'and', 'or', 'of', 'in', 'on', 'at', 'to', 'for'}
    
    def __init__(
        self,
        remove_boundary_punct: bool = True,
        remove_boundary_stopwords: bool = False,
        custom_punct: Optional[Set[str]] = None,
        custom_stopwords: Optional[Set[str]] = None
    ):
        """
        Args:
            remove_boundary_punct: 是否移除边界标点
            remove_boundary_stopwords: 是否移除边界停用词
            custom_punct: 自定义标点集合
            custom_stopwords: 自定义停用词集合
        """
        self.remove_boundary_punct = remove_boundary_punct
        self.remove_boundary_stopwords = remove_boundary_stopwords
        self.boundary_punct = custom_punct or self.DEFAULT_BOUNDARY_PUNCT
        self.boundary_stopwords = custom_stopwords or self.DEFAULT_BOUNDARY_STOPWORDS
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        """应用边界规则"""
        labels = result.labels.copy()
        entities = self._extract_entity_spans(labels)
        
        for start, end, entity_type in entities:
            # 检查并修正起始边界
            while start < end:
                token = result.tokens[start].strip()
                should_remove = False
                
                if self.remove_boundary_punct and self._is_punct(token):
                    should_remove = True
                elif self.remove_boundary_stopwords and token.lower() in self.boundary_stopwords:
                    should_remove = True
                
                if should_remove:
                    labels[start] = 'O'
                    start += 1
                else:
                    break
            
            # 检查并修正结束边界
            while end > start:
                token = result.tokens[end - 1].strip()
                should_remove = False
                
                if self.remove_boundary_punct and self._is_punct(token):
                    should_remove = True
                elif self.remove_boundary_stopwords and token.lower() in self.boundary_stopwords:
                    should_remove = True
                
                if should_remove:
                    labels[end - 1] = 'O'
                    end -= 1
                else:
                    break
        
        return PredictionResult(
            tokens=result.tokens,
            labels=labels,
            confidences=result.confidences
        )
    
    def _is_punct(self, token: str) -> bool:
        """判断token是否为标点"""
        return token in self.boundary_punct or all(c in self.boundary_punct for c in token)
    
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


class MinEntityLengthRule(BaseRule):
    """最小实体长度规则
    
    移除长度小于阈值的实体。
    
    示例：
        min_length=2
        输入:  tokens=['A', 'B', 'C'], labels=['B-ORG', 'B-PER', 'I-PER']
        输出:  tokens=['A', 'B', 'C'], labels=['O', 'B-PER', 'I-PER']
    """
    
    name = MIN_ENTITY_LENGTH_RULE
    description = "Remove entities shorter than minimum length"
    
    def __init__(self, min_length: int = 2, min_length_by_type: Optional[Dict[str, int]] = None):
        """
        Args:
            min_length: 默认最小长度（token数）
            min_length_by_type: 按实体类型指定最小长度，如 {'PER': 2, 'ORG': 3}
        """
        self.min_length = min_length
        self.min_length_by_type = min_length_by_type or {}
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        """应用最小长度规则"""
        labels = result.labels.copy()
        entities = self._extract_entity_spans(labels)
        
        for start, end, entity_type in entities:
            entity_length = end - start
            min_len = self.min_length_by_type.get(entity_type, self.min_length)
            
            if entity_length < min_len:
                # 移除这个实体
                for i in range(start, end):
                    labels[i] = 'O'
        
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


class PatternCorrectionRule(BaseRule):
    """基于模式的修正规则
    
    根据正则表达式模式修正特定实体。
    
    示例用途：
    - 确保电话号码被标记为PHONE
    - 确保邮箱被标记为EMAIL
    - 修正特定格式的实体类型
    """
    
    name = PATTERN_CORRECTION_RULE
    description = "Correct entities based on regex patterns"
    
    def __init__(self, patterns: Dict[str, str]):
        """
        Args:
            patterns: 模式字典，格式为 {entity_type: regex_pattern}
                例如: {'PHONE': r'\+?\d{1,3}[-.\s]?\d{3,}', 'EMAIL': r'\S+@\S+\.\S+'}
        """
        self.patterns = {
            entity_type: re.compile(pattern)
            for entity_type, pattern in patterns.items()
        }
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        """应用模式修正"""
        labels = result.labels.copy()
        
        # 对每个token检查是否匹配模式
        for i, token in enumerate(result.tokens):
            for entity_type, pattern in self.patterns.items():
                if pattern.match(token):
                    # 如果当前是O或其他类型，修正为匹配的类型
                    if labels[i] == 'O':
                        labels[i] = f'B-{entity_type}'
                    elif not labels[i].endswith(f'-{entity_type}'):
                        # 修正类型
                        prefix = 'B-' if labels[i].startswith('B-') else 'I-'
                        labels[i] = f'{prefix}{entity_type}'
                    break
        
        return PredictionResult(
            tokens=result.tokens,
            labels=labels,
            confidences=result.confidences
        )


class MergeAdjacentRule(BaseRule):
    """合并相邻同类实体规则
    
    将相邻的同类型实体合并为一个实体。
    
    示例：
        输入:  labels=['B-PER', 'I-PER', 'O', 'B-PER', 'I-PER']
        如果中间的O是空格或连接符，可以合并为一个PER实体
    """
    
    name = MERGE_ADJACENT_RULE
    description = "Merge adjacent entities of the same type"
    
    def __init__(self, merge_across_tokens: Optional[Set[str]] = None, max_gap: int = 1):
        """
        Args:
            merge_across_tokens: 允许跨越的token集合（如空格、连接符）
            max_gap: 最大允许的间隔token数
        """
        self.merge_across_tokens = merge_across_tokens or {' ', '-', '/', '&'}
        self.max_gap = max_gap
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        """应用合并规则"""
        labels = result.labels.copy()
        i = 0
        
        while i < len(labels):
            if labels[i].startswith('B-'):
                entity_type = labels[i][2:]
                entity_end = self._find_entity_end(labels, i, entity_type)
                
                # 查找下一个同类型实体
                j = entity_end
                while j < len(labels) and j - entity_end <= self.max_gap:
                    if labels[j] == 'O' and result.tokens[j] in self.merge_across_tokens:
                        j += 1
                    elif labels[j] == f'B-{entity_type}':
                        # 找到同类型实体，合并
                        for k in range(entity_end, j):
                            labels[k] = f'I-{entity_type}'
                        labels[j] = f'I-{entity_type}'
                        entity_end = self._find_entity_end(labels, j, entity_type)
                        j = entity_end
                    else:
                        break
                
                i = entity_end
            else:
                i += 1
        
        return PredictionResult(
            tokens=result.tokens,
            labels=labels,
            confidences=result.confidences
        )
    
    def _find_entity_end(self, labels: List[str], start: int, entity_type: str) -> int:
        """找到实体的结束位置"""
        i = start + 1
        while i < len(labels) and labels[i] == f'I-{entity_type}':
            i += 1
        return i
