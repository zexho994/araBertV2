"""内置后处理规则

提供常用的后处理规则实现。
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .pipeline import BaseRule, PredictionResult


# 规则名称常量
BIO_CONSISTENCY_RULE = 'bio_consistency'
CONFIDENCE_THRESHOLD_RULE = 'confidence_threshold'
REGEX_FILTER_RULE = 'regex_filter'
BLACKLIST_FILTER_RULE = 'blacklist_filter'


def _extract_entity_spans(labels: List[str]) -> List[Tuple[int, int, str]]:
    """提取实体跨度"""
    entities: List[Tuple[int, int, str]] = []
    current_start: Optional[int] = None
    current_end: Optional[int] = None
    current_type: Optional[str] = None
    
    for i, label in enumerate(labels):
        if label.startswith('B-'):
            if (
                current_start is not None
                and current_end is not None
                and current_type is not None
            ):
                entities.append((current_start, current_end, current_type))
            current_type = label[2:]
            current_start = i
            current_end = i + 1
        elif label.startswith('I-') and current_start is not None:
            entity_type = label[2:]
            if current_type == entity_type:
                current_end = i + 1
            else:
                if (
                    current_start is not None
                    and current_end is not None
                    and current_type is not None
                ):
                    entities.append((current_start, current_end, current_type))
                current_type = entity_type
                current_start = i
                current_end = i + 1
        else:
            if (
                current_start is not None
                and current_end is not None
                and current_type is not None
            ):
                entities.append((current_start, current_end, current_type))
            current_start = None
            current_end = None
            current_type = None
    
    if (
        current_start is not None
        and current_end is not None
        and current_type is not None
    ):
        entities.append((current_start, current_end, current_type))
    
    return entities


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
            entities = _extract_entity_spans(labels)
            low_confidence_entities = set()
            
            for start, end, _ in entities:
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
    
class RegexFilterRule(BaseRule):
    """正则实体过滤规则
    
    将token合并后的实体文本与给定正则匹配的实体全部置为'O'。
    
    示例：
        patterns=[r'^https?://']
        输入:  tokens=['访', '问', 'https', '://', 'example', '.com'], labels=['O', 'O', 'B-URL', 'I-URL', 'I-URL', 'I-URL']
        输出:  labels=['O', 'O', 'O', 'O', 'O', 'O']
    """
    
    name = REGEX_FILTER_RULE
    description = "Filter entities whose merged text matches regex patterns"
    
    def __init__(self, patterns: List[str], join_with: str = '', flags: int = 0):
        """
        Args:
            patterns: 正则表达式列表
            join_with: 合并token时使用的连接符
            flags: 传递给re.compile的flags
        """
        if not patterns:
            raise ValueError("patterns must not be empty")
        self._patterns = [re.compile(pattern, flags) for pattern in patterns]
        self.join_with = join_with
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        labels = result.labels.copy()
        
        for start, end, _ in _extract_entity_spans(labels):
            entity_text = self.join_with.join(result.tokens[start:end])
            if any(pattern.search(entity_text) for pattern in self._patterns):
                for idx in range(start, end):
                    labels[idx] = 'O'
        
        return PredictionResult(
            tokens=result.tokens,
            labels=labels,
            confidences=result.confidences
        )


class BlacklistFilterRule(BaseRule):
    """实体黑名单过滤规则
    
    将合并后的实体文本与黑名单匹配的实体全部置为'O'。
    黑名单按实体类型维护，每种类型对应一个文本文件。
    """
    
    name = BLACKLIST_FILTER_RULE
    description = "Filter entities whose merged text matches blacklist entries"
    
    def __init__(
        self,
        blacklist_files: Dict[str, str],
        join_with: str = ' ',
        case_insensitive: bool = False,
        strip_whitespace: bool = True,
    ):
        """
        Args:
            blacklist_files: {实体类型: 黑名单文件路径} 映射
            join_with: 合并token时使用的连接符
            case_insensitive: 是否忽略大小写匹配
            strip_whitespace: 是否在匹配前去除首尾空白
        """
        if not blacklist_files:
            raise ValueError("blacklist_files must not be empty")
        
        self.join_with = join_with
        self.case_insensitive = case_insensitive
        self.strip_whitespace = strip_whitespace
        self._blacklists = self._load_blacklists(blacklist_files)
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        labels = result.labels.copy()
        
        for start, end, entity_type in _extract_entity_spans(labels):
            blacklist = self._blacklists.get(entity_type.upper())
            if not blacklist:
                continue
            
            entity_text = self.join_with.join(result.tokens[start:end])
            if self.strip_whitespace:
                entity_text = entity_text.strip()
            if self.case_insensitive:
                entity_text = entity_text.lower()
            
            if entity_text in blacklist:
                for idx in range(start, end):
                    labels[idx] = 'O'
        
        return PredictionResult(
            tokens=result.tokens,
            labels=labels,
            confidences=result.confidences
        )
    
    def _load_blacklists(self, blacklist_files: Dict[str, str]) -> Dict[str, Set[str]]:
        blacklists: Dict[str, Set[str]] = {}
        for entity_type, path_str in blacklist_files.items():
            normalized_type = entity_type.upper()
            file_path = Path(path_str).expanduser()
            if not file_path.is_absolute():
                file_path = (Path.cwd() / file_path).resolve()
            if not file_path.exists():
                raise FileNotFoundError(f"Blacklist file not found for '{entity_type}': {file_path}")
            
            entries: Set[str] = set()
            with file_path.open('r', encoding='utf-8') as f:
                for line in f:
                    entry = line.strip() if self.strip_whitespace else line.rstrip('\n')
                    if not entry:
                        continue
                    if self.case_insensitive:
                        entry = entry.lower()
                    entries.add(entry)
            
            if entries:
                blacklists[normalized_type] = entries
        
        return blacklists