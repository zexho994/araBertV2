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
- 各个具体命令已移动到单独的文件中
- BaseCommand 基类定义在 base.py 中，避免代码重复
- 此文件作为所有命令的统一导入入口
"""

# Import base class from dedicated module
from .base import BaseCommand

# Import all command implementations
from .evaluate_command import EvaluateCommand
from .train_command import TrainCommand
from .predict_command import PredictCommand
from .evaluate_predict_command import EvaluatePredictCommand
from .preprocess_command import PreprocessCommand
from .config_command import ConfigCommand
from .data_command import DataCommand
from .model_command import ModelCommand
from .status_command import StatusCommand