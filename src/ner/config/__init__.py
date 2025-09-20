"""NER Configuration Module

Configuration management for NER training and evaluation.
Handles country-specific configurations, validation, and templates.
"""

from .manager import ConfigManager
from .validator import ConfigValidator
from .schema import ConfigSchema

__all__ = [
    'ConfigManager',
    'ConfigValidator',
    'ConfigSchema'
]

class GlobalConfig:
    """Global configuration"""
    config_dir: str
    data_dir: str
    model_dir: str
    log_dir: str
    eval_dir: str
    verbose: bool

    def __init__(self, config_dir: str, data_dir: str, model_dir: str, verbose: bool = False):
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.log_dir = f"{data_dir}/logs"
        self.eval_dir = f"{data_dir}/eval"
        self.verbose = verbose

    def get_verbose(self) -> bool:
        """Get verbose configuration"""
        return self.verbose

    def set_verbose(self, verbose: bool):
        """Set verbose configuration"""
        self.verbose = verbose

    def get_log_dir(self) -> str:
        """Get log directory"""
        return self.log_dir

    def get_eval_dir(self) -> str:
        """Get eval directory"""
        return self.eval_dir

    def get_model_dir(self) -> str:
        """Get model directory"""
        return self.model_dir

    def get_config_dir(self) -> str:
        """Get config directory"""
        return self.config_dir

    def get_data_dir(self) -> str:
        """Get data directory"""
        return self.data_dir
