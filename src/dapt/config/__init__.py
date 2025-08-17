"""配置管理模块

负责多国家 DAPT 训练的配置管理，包括：
- 国家级配置（Country-specific）
- 训练参数（Training parameters）
- 模型设置（Model settings）
- 数据处理选项（Data processing options）

# TODO: 在 `ConfigManager` 中增加 schema 验证与默认值补全的统一入口，减少各调用点的冗余判断。
"""

from .manager import ConfigManager
from .validator import ConfigValidator
from .templates import ConfigTemplates

__all__ = [
    "ConfigManager",
    "ConfigValidator",
    "ConfigTemplates"
]