"""CLI Module for DAPT Training

Provides command-line interface for DAPT operations including:
- Training management
- Configuration setup
- Model evaluation
- Data processing
"""

from .main import DAPTCLIManager
from .commands import (
    TrainCommand,
    EvaluateCommand,
    ConfigCommand,
    DataCommand,
    ModelCommand
)

__all__ = [
    "DAPTCLIManager",
    "TrainCommand",
    "EvaluateCommand", 
    "ConfigCommand",
    "DataCommand",
    "ModelCommand"
]