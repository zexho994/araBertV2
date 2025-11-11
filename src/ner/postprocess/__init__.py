"""NER Post-processing Pipeline

Rule-based post-processing for NER predictions.

设计目标：
- 在模型预测后应用规则修正标签
- 支持规则链式组合
- 可从配置文件加载
- 与预处理器对称的设计

Public API:
- Postprocessor: 后处理器主类
- BaseRule: 规则基类
- build_postprocessor_from_config: 从配置构建后处理器
"""

from .pipeline import Postprocessor, BaseRule
from .builder import build_postprocessor_from_config

__all__ = [
    "Postprocessor",
    "BaseRule",
    "build_postprocessor_from_config",
]
