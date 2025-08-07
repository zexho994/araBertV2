"""NER Evaluation Module

Model evaluation and metrics calculation for NER models.
Handles performance evaluation, report generation, and metrics analysis.
"""

from .evaluator import NEREvaluator
from .metrics import NERMetrics
from .reporter import EvaluationReporter

__all__ = [
    'NEREvaluator',
    'NERMetrics',
    'EvaluationReporter'
]