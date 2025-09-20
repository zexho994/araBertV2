"""NER CLI 基础类定义

包含所有 CLI 命令的抽象基类和公共功能。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

from ..config import ConfigManager

from ..config import GlobalConfig

class BaseCommand(ABC):
    """所有 NER CLI 子命令的抽象基类

    职责：
    - 提供统一的命令名称（`name`）与描述（`description`）属性
    - 定义参数解析接口 `setup_parser` 与执行接口 `execute`

    # TODO: 支持注入统一的 logger，并在各子命令中复用。
    """


    global_config: GlobalConfig
    country_config: Dict[str, Any]
    
    def __init__(self):
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
    
    def set_global_config(self, config: GlobalConfig):
        """Set global configuration"""
        self.global_config = config

    def get_global_config(self) -> GlobalConfig:
        """Get global configuration"""
        return self.global_config

    def get_country_config(self, country: str) -> Dict[str, Any]:
        """Get country configuration"""
        if not self.country_config:
            config_manager = ConfigManager(self.global_config.get_config_dir())
            self.country_config = config_manager.load_country_config(country)
        return self.country_config
    
    def set_logger(self, logger):
        """Inject a shared logger instance"""
        self.logger = logger
    
    def get_help(self) -> str:
        """Get help text for this command"""
        return self.description
