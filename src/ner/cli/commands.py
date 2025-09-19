"""NER 命令行（CLI）指令集合

提供 NER 系统的常用指令，包括：
- 训练（train）
- 评估（evaluate）
- 预测（predict）
- 配置管理（config）
- 数据处理（data）
- 状态查看（status）

设计说明：
- 通过 `BaseCommand` 统一约束命令的名称与描述、参数解析与执行。
- 全局配置 `global_config` 由外层主程序注入，通常包含：
  - `config_dir`: 配置目录
  - `model_dir`: 模型目录
  - `log_dir`: 日志目录

# TODO: 为 `global_config` 定义强类型（TypedDict/dataclass），并在入口层进行完整校验。

重构说明：
- 各个具体命令已移动到单独的文件中，此文件仅保留 BaseCommand 基类和导入声明
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any
from ..config import ConfigManager

class BaseCommand(ABC):
    """所有 NER CLI 子命令的抽象基类

    职责：
    - 提供统一的命令名称（`name`）与描述（`description`）属性
    - 定义参数解析接口 `setup_parser` 与执行接口 `execute`
    - 保存外层注入的 `global_config`

    # TODO: 支持注入统一的 logger，并在各子命令中复用。
    """

    global_config: Dict[str, Any]
    country_config: Dict[str, Any]
    
    def __init__(self):
        self.global_config = {}
        self.country_config = {}
        self.logger = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Command name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Command description"""
        pass
    
    @abstractmethod
    def setup_parser(self, parser):
        """Setup command-specific arguments"""
        pass
    
    @abstractmethod
    def execute(self, args) -> bool:
        """Execute the command"""
        pass
    
    def set_global_config(self, config: Dict[str, Any]):
        """Set global configuration"""
        self.global_config = config

    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration"""
        return self.global_config

    def get_country_config(self, country: str) -> Dict[str, Any]:
        """Get country configuration"""
        if not self.country_config:
            config_manager = ConfigManager(self.global_config.get('config_dir'))
            self.country_config = config_manager.load_country_config(country)
        return self.country_config
    
    def set_logger(self, logger):
        """Inject a shared logger instance"""
        self.logger = logger
    
    def get_help(self) -> str:
        """Get help text for this command"""
        return self.description