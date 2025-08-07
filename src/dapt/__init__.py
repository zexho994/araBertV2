"""DAPT (Domain Adaptive Pre-Training) Module for AraBERTv2

This module provides domain-adaptive pre-training capabilities for AraBERTv2,
specifically designed for training address parsing models for different countries.

Main Components:
- CLI: Command-line interface for DAPT operations
- Engine: Core training engine for DAPT
- Config: Configuration management for multi-country support
- Data: Data processing and validation pipelines
- Models: Model management and versioning
- Evaluation: Model evaluation and metrics
- Utils: Utility functions and helpers
"""

__version__ = "1.0.0"
__author__ = "AraBERTv2 Team"

from .cli import DAPTCLIManager
from .engine import DAPTTrainingEngine
from .config import ConfigManager
from .data import DataProcessor
from .models import ModelManager
from .evaluation import EvaluationManager

__all__ = [
    "DAPTCLIManager",
    "DAPTTrainingEngine", 
    "ConfigManager",
    "DataProcessor",
    "ModelManager",
    "EvaluationManager"
]