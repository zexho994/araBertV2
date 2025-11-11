"""后处理Pipeline核心

提供规则基类和后处理器主类。
"""

from __future__ import annotations
from typing import List, Optional, Iterable, Tuple, Dict, Any
from dataclasses import dataclass


@dataclass
class PredictionResult:
    """预测结果的标准化数据结构
    
    用于在后处理规则之间传递数据。
    """
    tokens: List[str]
    labels: List[str]
    confidences: List[float]
    
    def __post_init__(self):
        """验证数据一致性"""
        if not (len(self.tokens) == len(self.labels) == len(self.confidences)):
            raise ValueError(
                f"Length mismatch: tokens={len(self.tokens)}, "
                f"labels={len(self.labels)}, confidences={len(self.confidences)}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'tokens': self.tokens,
            'labels': self.labels,
            'confidences': self.confidences
        }


class BaseRule:
    """后处理规则基类
    
    所有后处理规则都应继承此类并实现 apply 方法。
    
    规则可以：
    - 修改标签（如修正边界错误）
    - 调整置信度
    - 合并或拆分token（高级用法）
    - 添加或删除实体
    """
    
    name: str = "base_rule"
    description: str = "Base rule class"
    
    def apply(self, result: PredictionResult) -> PredictionResult:
        """应用规则
        
        Args:
            result: 预测结果
            
        Returns:
            PredictionResult: 处理后的结果
        """
        return result
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class Postprocessor:
    """后处理器主类
    
    管理规则链，按顺序应用所有规则。
    
    使用示例：
        postprocessor = Postprocessor([
            ConsistencyRule(),
            ConfidenceThresholdRule(threshold=0.5),
            EntityBoundaryRule()
        ])
        
        result = postprocessor.apply(tokens, labels, confidences)
    """
    
    def __init__(self, rules: Optional[Iterable[BaseRule]] = None):
        """初始化后处理器
        
        Args:
            rules: 规则列表
        """
        self.rules: List[BaseRule] = list(rules) if rules else []
    
    def __str__(self) -> str:
        rule_names = ', '.join(rule.name for rule in self.rules)
        return f"Postprocessor(rules=[{rule_names}])"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def add_rule(self, rule: BaseRule) -> None:
        """添加规则
        
        Args:
            rule: 要添加的规则
        """
        self.rules.append(rule)
    
    def apply(
        self,
        tokens: List[str],
        labels: List[str],
        confidences: List[float]
    ) -> Tuple[List[str], List[str], List[float]]:
        """应用所有规则
        
        Args:
            tokens: token列表
            labels: 标签列表
            confidences: 置信度列表
            
        Returns:
            Tuple[List[str], List[str], List[float]]: 处理后的 (tokens, labels, confidences)
        """
        if not tokens:
            return tokens, labels, confidences
        
        result = PredictionResult(
            tokens=list(tokens),
            labels=list(labels),
            confidences=list(confidences)
        )
        
        for rule in self.rules:
            result = rule.apply(result)
        
        return result.tokens, result.labels, result.confidences
    
    def apply_batch(
        self,
        batch_tokens: List[List[str]],
        batch_labels: List[List[str]],
        batch_confidences: List[List[float]]
    ) -> Tuple[List[List[str]], List[List[str]], List[List[float]]]:
        """批量应用规则
        
        Args:
            batch_tokens: batch的token列表
            batch_labels: batch的标签列表
            batch_confidences: batch的置信度列表
            
        Returns:
            处理后的batch数据
        """
        processed_tokens = []
        processed_labels = []
        processed_confidences = []
        
        for tokens, labels, confidences in zip(batch_tokens, batch_labels, batch_confidences):
            t, l, c = self.apply(tokens, labels, confidences)
            processed_tokens.append(t)
            processed_labels.append(l)
            processed_confidences.append(c)
        
        return processed_tokens, processed_labels, processed_confidences
