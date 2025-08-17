"""DAPT 训练引擎模块

包含 DAPT（领域自适应预训练）的核心组件：
- `DAPTTrainingEngine`: 训练流程的核心编排器
- `TrainingScheduler`: 训练任务的调度与队列管理
- `TrainingMonitor`: 训练过程监控与系统/训练指标采集

# TODO: 提供统一的类型注解与公共异常类型，便于调用方做细粒度异常处理。
"""

from .trainer import DAPTTrainingEngine
from .scheduler import TrainingScheduler, TrainingJob, TrainingStatus
from .monitor import TrainingMonitor, SystemMetrics, TrainingMetrics

__all__ = [
    "DAPTTrainingEngine",
    "TrainingScheduler",
    "TrainingJob",
    "TrainingStatus",
    "TrainingMonitor",
    "SystemMetrics",
    "TrainingMetrics"
]