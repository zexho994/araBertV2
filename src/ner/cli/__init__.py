"""NER CLI Module

Command-line interface components for the NER system.
Provides CLI manager and command implementations.

说明：尽量保持包导入轻量，不在包级别导入任何会触发重依赖（如 torch/pandas）的模块。
`repl_predict` 子模块用于交互式预测
"""

# 仅暴露类型与管理器，避免在此处导入可能间接拉取重库的子模块
from .main import NERCLIManager
from .base import BaseCommand
from .evaluate_command import EvaluateCommand
from .train_command import TrainCommand
from .predict_command import PredictCommand
from .evaluate_predict_command import EvaluatePredictCommand
from .preprocess_command import PreprocessCommand
from .config_command import ConfigCommand
from .data_command import DataCommand
from .model_command import ModelCommand
from .status_command import StatusCommand

__all__ = [
    'NERCLIManager',
    'BaseCommand',
    'TrainCommand',
    'EvaluateCommand', 
    'PredictCommand',
    'EvaluatePredictCommand',
    'PreprocessCommand',
    'ConfigCommand',
    'DataCommand',
    'ModelCommand',
    'StatusCommand'
]