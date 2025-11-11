"""从配置构建后处理器

支持从字典配置或JSON文件加载后处理规则。
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import json
from pathlib import Path

from .pipeline import Postprocessor
from .rules import (
    BIOConsistencyRule,
    ConfidenceThresholdRule,

    # 规则名称常量
    BIO_CONSISTENCY_RULE,
    CONFIDENCE_THRESHOLD_RULE,
)


# 规则工厂映射
RULE_FACTORY = {
    BIO_CONSISTENCY_RULE: BIOConsistencyRule,
    CONFIDENCE_THRESHOLD_RULE: ConfidenceThresholdRule,
}


def build_postprocessor_from_config(config: Dict[str, Any]) -> Optional[Postprocessor]:
    """从配置字典构建后处理器
    Args:
        config: 配置字典（可以是完整的国家配置或只包含postprocess部分）
        
    Returns:
        Postprocessor: 构建好的后处理器，如果配置中没有postprocess则返回None
    """
    postprocess_config = config.get('postprocess', {})
    
    # 如果没有配置后处理规则，返回None
    if not postprocess_config:
        return None
    
    rules_config = postprocess_config.get('rules', [])
    
    # 如果rules为空，返回空的后处理器
    if not rules_config:
        return Postprocessor([])
    
    rules = []
    for rule_config in rules_config:
        rule_type = rule_config.get('type')
        rule_params = rule_config.get('params', {})
        
        if rule_type not in RULE_FACTORY:
            raise ValueError(f"Unknown rule type: {rule_type}. Available types: {list(RULE_FACTORY.keys())}")
        
        rule_class = RULE_FACTORY[rule_type]
        try:
            rule = rule_class(**rule_params)
            rules.append(rule)
        except TypeError as e:
            raise ValueError(f"Invalid parameters for rule '{rule_type}': {e}")
    
    return Postprocessor(rules)


def build_postprocessor_from_file(config_path: str) -> Postprocessor:
    """从JSON配置文件构建后处理器
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        Postprocessor: 构建好的后处理器
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return build_postprocessor_from_config(config)


def get_default_postprocessor() -> Postprocessor:
    """获取默认后处理器
    
    包含最常用的规则组合。
    
    Returns:
        Postprocessor: 默认后处理器
    """
    return Postprocessor([
        BIOConsistencyRule(fix_orphan_i='to_b'),
    ])
