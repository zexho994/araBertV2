"""NER CLI Module

Command-line interface components for the NER system.
Provides CLI manager and command implementations.

说明：尽量保持包导入轻量，不在包级别导入任何会触发重依赖（如 torch/pandas）的模块。
`repl_predict` 子模块用于交互式预测
"""

# 仅暴露类型与管理器，避免在此处导入可能间接拉取重库的子模块
from .main import NERCLIManager
from .evaluate_command import EvaluateCommand
from .train_command import TrainCommand
from .predict_command import PredictCommand
from .preprocess_command import PreprocessCommand
from .data_command import DataCommand
from .model_command import ModelCommand

__all__ = [
    'NERCLIManager',
    'TrainCommand',
    'EvaluateCommand', 
    'PredictCommand',
    'PreprocessCommand',
    'DataCommand',
    'ModelCommand',
]