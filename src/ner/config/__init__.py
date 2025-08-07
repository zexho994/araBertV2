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