"""NER Preprocessing Pipeline

Composable, step-based preprocessing for text and token-level flows.

Public API (minimal, stable):
- Preprocessor
- BaseStep
- build_preprocessor_from_config
"""

from .pipeline import Preprocessor, BaseStep
from .builder import build_preprocessor_from_config

__all__ = [
    "Preprocessor",
    "BaseStep",
    "build_preprocessor_from_config",
]


