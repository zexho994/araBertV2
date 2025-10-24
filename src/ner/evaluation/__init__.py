"""NER Evaluation Module

Model evaluation and metrics calculation for NER models.
Handles performance evaluation, report generation, and metrics analysis.
"""

from .evaluator import NEREvaluator
from .metrics import NERMetrics
from .report_generator import NERReportGenerator
from .compare_reports import ReportComparator, compare_reports

__all__ = [
    'NEREvaluator',
    'NERMetrics',
    'NERReportGenerator',
    'ReportComparator',
    'compare_reports'
]