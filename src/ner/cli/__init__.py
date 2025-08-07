"""NER CLI Module

Command-line interface components for the NER system.
Provides CLI manager and command implementations.
"""

from .main import NERCLIManager
from .commands import (
    BaseCommand,
    TrainCommand,
    EvaluateCommand,
    PredictCommand,
    ConfigCommand,
    DataCommand,
    ModelCommand,
    StatusCommand
)

__all__ = [
    'NERCLIManager',
    'BaseCommand',
    'TrainCommand',
    'EvaluateCommand', 
    'PredictCommand',
    'ConfigCommand',
    'DataCommand',
    'ModelCommand',
    'StatusCommand'
]