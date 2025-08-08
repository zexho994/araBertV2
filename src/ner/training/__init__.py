"""NER Training Module

Training pipeline and utilities for NER models.
Handles model training, validation, and checkpointing.
"""

from .trainer import NERTrainer

__all__ = [
    'NERTrainer'
]