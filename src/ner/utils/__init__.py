"""NER Utils Module

Utility functions and helpers for the NER system.
Provides logging, file operations, and common utilities.
"""

from .logger import NERLogger, setup_logging

__all__ = [
    'NERLogger',
    'setup_logging',
]