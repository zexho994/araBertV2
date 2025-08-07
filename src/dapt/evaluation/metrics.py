"""DAPT Evaluation Metrics

Metrics calculation for DAPT training evaluation:
- Standard NLP metrics (accuracy, precision, recall, F1)
- Domain-specific metrics
- Arabic language specific metrics
- Custom evaluation metrics
- Metric aggregation and reporting
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
import re
from collections import Counter, defaultdict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error,
    roc_auc_score, average_precision_score
)
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine
import logging

@dataclass
class MetricResult:
    """Result of a metric calculation"""
    name: str
    value: float
    description: str
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'value': self.value,
            'description': self.description,
            'metadata': self.metadata or {}
        }

class BaseMetric(ABC):
    """Abstract base class for metrics"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def calculate(self, predictions: Any, ground_truth: Any, **kwargs) -> MetricResult:
        """Calculate the metric"""
        pass
    
    def validate_inputs(self, predictions: Any, ground_truth: Any) -> bool:
        """Validate input data"""
        return predictions is not None and ground_truth is not None

class AccuracyMetric(BaseMetric):
    """Accuracy metric for classification tasks"""
    
    def __init__(self):
        super().__init__("accuracy", "Classification accuracy")
    
    def calculate(self, predictions: List, ground_truth: List, **kwargs) -> MetricResult:
        if not self.validate_inputs(predictions, ground_truth):
            return MetricResult(self.name, 0.0, self.description)
        
        try:
            accuracy = accuracy_score(ground_truth, predictions)
            return MetricResult(
                self.name, 
                float(accuracy), 
                self.description,
                {'total_samples': len(predictions)}
            )
        except Exception as e:
            return MetricResult(self.name, 0.0, f"{self.description} (Error: {str(e)})")

class PrecisionRecallF1Metric(BaseMetric):
    """Precision, Recall, and F1 metrics"""
    
    def __init__(self, average: str = 'weighted'):
        super().__init__("precision_recall_f1", f"Precision, Recall, F1 ({average} average)")
        self.average = average
    
    def calculate(self, predictions: List, ground_truth: List, **kwargs) -> MetricResult:
        if not self.validate_inputs(predictions, ground_truth):
            return MetricResult(self.name, 0.0, self.description)
        
        try:
            precision, recall, f1, support = precision_recall_fscore_support(
                ground_truth, predictions, average=self.average, zero_division=0
            )
            
            return MetricResult(
                self.name,
                float(f1),  # Return F1 as main value
                self.description,
                {
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'support': int(support) if isinstance(support, (int, np.integer)) else support.tolist(),
                    'average': self.average
                }
            )
        except Exception as e:
            return MetricResult(self.name, 0.0, f"{self.description} (Error: {str(e)})")

class PerplexityMetric(BaseMetric):
    """Perplexity metric for language modeling"""
    
    def __init__(self):
        super().__init__("perplexity", "Language model perplexity")
    
    def calculate(self, predictions: torch.Tensor, ground_truth: torch.Tensor, **kwargs) -> MetricResult:
        if not self.validate_inputs(predictions, ground_truth):
            return MetricResult(self.name, float('inf'), self.description)
        
        try:
            # Calculate cross-entropy loss
            if isinstance(predictions, torch.Tensor) and isinstance(ground_truth, torch.Tensor):
                loss = F.cross_entropy(predictions.view(-1, predictions.size(-1)), ground_truth.view(-1))
                perplexity = torch.exp(loss).item()
            else:
                # Fallback for non-tensor inputs
                perplexity = float('inf')
            
            return MetricResult(
                self.name,
                float(perplexity),
                self.description,
                {'cross_entropy_loss': float(loss) if 'loss' in locals() else None}
            )
        except Exception as e:
            return MetricResult(self.name, float('inf'), f"{self.description} (Error: {str(e)})")

class BLEUMetric(BaseMetric):
    """BLEU score for text generation tasks"""
    
    def __init__(self, n_gram: int = 4):
        super().__init__("bleu", f"BLEU-{n_gram} score")
        self.n_gram = n_gram
    
    def calculate(self, predictions: List[str], ground_truth: List[str], **kwargs) -> MetricResult:
        if not self.validate_inputs(predictions, ground_truth):
            return MetricResult(self.name, 0.0, self.description)
        
        try:
            # Simple BLEU implementation
            total_score = 0.0
            count = 0
            
            for pred, ref in zip(predictions, ground_truth):
                score = self._calculate_bleu_sentence(pred, ref)
                total_score += score
                count += 1
            
            bleu_score = total_score / count if count > 0 else 0.0
            
            return MetricResult(
                self.name,
                float(bleu_score),
                self.description,
                {'n_gram': self.n_gram, 'sentence_count': count}
            )
        except Exception as e:
            return MetricResult(self.name, 0.0, f"{self.description} (Error: {str(e)})")
    
    def _calculate_bleu_sentence(self, prediction: str, reference: str) -> float:
        """Calculate BLEU score for a single sentence pair"""
        try:
            pred_tokens = prediction.split()
            ref_tokens = reference.split()
            
            if len(pred_tokens) == 0 or len(ref_tokens) == 0:
                return 0.0
            
            # Calculate n-gram precision
            precisions = []
            for n in range(1, min(self.n_gram + 1, len(pred_tokens) + 1)):
                pred_ngrams = self._get_ngrams(pred_tokens, n)
                ref_ngrams = self._get_ngrams(ref_tokens, n)
                
                if len(pred_ngrams) == 0:
                    precisions.append(0.0)
                else:
                    matches = sum(min(pred_ngrams[ngram], ref_ngrams[ngram]) 
                                for ngram in pred_ngrams if ngram in ref_ngrams)
                    precisions.append(matches / len(pred_ngrams))
            
            if not precisions or all(p == 0 for p in precisions):
                return 0.0
            
            # Geometric mean of precisions
            log_precisions = [np.log(p) if p > 0 else -np.inf for p in precisions]
            if any(p == -np.inf for p in log_precisions):
                return 0.0
            
            geometric_mean = np.exp(np.mean(log_precisions))
            
            # Brevity penalty
            bp = min(1.0, np.exp(1 - len(ref_tokens) / len(pred_tokens)))
            
            return bp * geometric_mean
            
        except Exception:
            return 0.0
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Get n-grams from tokens"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return Counter(ngrams)

class ArabicSpecificMetric(BaseMetric):
    """Arabic language specific metrics"""
    
    def __init__(self):
        super().__init__("arabic_specific", "Arabic language specific metrics")
    
    def calculate(self, predictions: List[str], ground_truth: List[str], **kwargs) -> MetricResult:
        if not self.validate_inputs(predictions, ground_truth):
            return MetricResult(self.name, 0.0, self.description)
        
        try:
            metrics = {
                'diacritic_accuracy': self._calculate_diacritic_accuracy(predictions, ground_truth),
                'root_accuracy': self._calculate_root_accuracy(predictions, ground_truth),
                'dialect_consistency': self._calculate_dialect_consistency(predictions, kwargs.get('country_code')),
                'arabic_character_ratio': self._calculate_arabic_character_ratio(predictions)
            }
            
            # Overall score as average of available metrics
            available_metrics = [v for v in metrics.values() if v is not None]
            overall_score = np.mean(available_metrics) if available_metrics else 0.0
            
            return MetricResult(
                self.name,
                float(overall_score),
                self.description,
                metrics
            )
        except Exception as e:
            return MetricResult(self.name, 0.0, f"{self.description} (Error: {str(e)})")
    
    def _calculate_diacritic_accuracy(self, predictions: List[str], ground_truth: List[str]) -> Optional[float]:
        """Calculate accuracy of diacritics (tashkeel)"""
        try:
            diacritic_chars = set('ًٌٍَُِّْ')
            
            total_diacritics = 0
            correct_diacritics = 0
            
            for pred, truth in zip(predictions, ground_truth):
                pred_diacritics = [c for c in pred if c in diacritic_chars]
                truth_diacritics = [c for c in truth if c in diacritic_chars]
                
                total_diacritics += len(truth_diacritics)
                
                # Count matching diacritics at same positions
                min_len = min(len(pred_diacritics), len(truth_diacritics))
                for i in range(min_len):
                    if pred_diacritics[i] == truth_diacritics[i]:
                        correct_diacritics += 1
            
            return correct_diacritics / total_diacritics if total_diacritics > 0 else None
            
        except Exception:
            return None
    
    def _calculate_root_accuracy(self, predictions: List[str], ground_truth: List[str]) -> Optional[float]:
        """Calculate accuracy of Arabic root extraction"""
        try:
            # Simplified root extraction (remove common prefixes/suffixes)
            common_prefixes = ['ال', 'و', 'ف', 'ب', 'ك', 'ل']
            common_suffixes = ['ة', 'ات', 'ان', 'ين', 'ون', 'ها', 'هم', 'هن']
            
            def extract_root(word: str) -> str:
                # Remove diacritics
                word = re.sub(r'[ًٌٍَُِّْ]', '', word)
                
                # Remove common prefixes
                for prefix in common_prefixes:
                    if word.startswith(prefix):
                        word = word[len(prefix):]
                        break
                
                # Remove common suffixes
                for suffix in common_suffixes:
                    if word.endswith(suffix):
                        word = word[:-len(suffix)]
                        break
                
                return word
            
            correct_roots = 0
            total_words = 0
            
            for pred, truth in zip(predictions, ground_truth):
                pred_words = pred.split()
                truth_words = truth.split()
                
                min_len = min(len(pred_words), len(truth_words))
                total_words += min_len
                
                for i in range(min_len):
                    pred_root = extract_root(pred_words[i])
                    truth_root = extract_root(truth_words[i])
                    
                    if pred_root == truth_root:
                        correct_roots += 1
            
            return correct_roots / total_words if total_words > 0 else None
            
        except Exception:
            return None
    
    def _calculate_dialect_consistency(self, predictions: List[str], country_code: Optional[str]) -> Optional[float]:
        """Calculate consistency with country-specific dialect"""
        try:
            if not country_code:
                return None
            
            # Define dialect-specific patterns (simplified)
            dialect_patterns = {
                'UAE': ['شلون', 'وايد', 'يالله'],
                'SAU': ['وش', 'كيف', 'الله يعطيك'],
                'EGY': ['ازيك', 'عامل', 'كده'],
                'MAR': ['كيفاش', 'بزاف', 'واخا'],
                'LBN': ['كيفك', 'شو', 'يلا'],
                'JOR': ['كيفك', 'شو', 'يلا']
            }
            
            patterns = dialect_patterns.get(country_code, [])
            if not patterns:
                return None
            
            total_texts = len(predictions)
            consistent_texts = 0
            
            for text in predictions:
                # Check if text contains dialect-specific patterns
                if any(pattern in text for pattern in patterns):
                    consistent_texts += 1
            
            return consistent_texts / total_texts if total_texts > 0 else 0.0
            
        except Exception:
            return None
    
    def _calculate_arabic_character_ratio(self, predictions: List[str]) -> Optional[float]:
        """Calculate ratio of Arabic characters in predictions"""
        try:
            arabic_range = range(0x0600, 0x06FF + 1)  # Arabic Unicode block
            
            total_chars = 0
            arabic_chars = 0
            
            for text in predictions:
                for char in text:
                    if not char.isspace():
                        total_chars += 1
                        if ord(char) in arabic_range:
                            arabic_chars += 1
            
            return arabic_chars / total_chars if total_chars > 0 else None
            
        except Exception:
            return None

class NERMetric(BaseMetric):
    """Named Entity Recognition specific metrics"""
    
    def __init__(self):
        super().__init__("ner_metrics", "NER-specific evaluation metrics")
    
    def calculate(self, predictions: List[List[Dict]], ground_truth: List[List[str]], **kwargs) -> MetricResult:
        if not self.validate_inputs(predictions, ground_truth):
            return MetricResult(self.name, 0.0, self.description)
        
        try:
            # Convert predictions to entity lists
            pred_entities = self._extract_entities_from_predictions(predictions)
            true_entities = self._extract_entities_from_labels(ground_truth)
            
            # Calculate entity-level metrics
            entity_metrics = self._calculate_entity_metrics(pred_entities, true_entities)
            
            # Calculate token-level metrics
            token_metrics = self._calculate_token_metrics(predictions, ground_truth)
            
            # Combine metrics
            combined_metrics = {
                **entity_metrics,
                **token_metrics,
                'entity_types': self._analyze_entity_types(pred_entities, true_entities)
            }
            
            # Overall F1 score as main metric
            overall_f1 = entity_metrics.get('entity_f1', 0.0)
            
            return MetricResult(
                self.name,
                float(overall_f1),
                self.description,
                combined_metrics
            )
        except Exception as e:
            return MetricResult(self.name, 0.0, f"{self.description} (Error: {str(e)})")
    
    def _extract_entities_from_predictions(self, predictions: List[List[Dict]]) -> List[List[Tuple]]:
        """Extract entities from model predictions"""
        entities = []
        
        for pred_list in predictions:
            sent_entities = []
            for entity in pred_list:
                if isinstance(entity, dict) and 'entity' in entity:
                    # Extract entity info
                    entity_type = entity['entity']
                    start = entity.get('start', 0)
                    end = entity.get('end', 0)
                    text = entity.get('word', '')
                    
                    sent_entities.append((entity_type, start, end, text))
            
            entities.append(sent_entities)
        
        return entities
    
    def _extract_entities_from_labels(self, ground_truth: List[List[str]]) -> List[List[Tuple]]:
        """Extract entities from ground truth labels"""
        entities = []
        
        for labels in ground_truth:
            if isinstance(labels, str):
                try:
                    labels = json.loads(labels)
                except:
                    labels = []
            
            sent_entities = []
            if isinstance(labels, list):
                for i, label in enumerate(labels):
                    if isinstance(label, dict):
                        entity_type = label.get('entity', label.get('type', 'UNKNOWN'))
                        start = label.get('start', i)
                        end = label.get('end', i + 1)
                        text = label.get('text', label.get('word', ''))
                        
                        sent_entities.append((entity_type, start, end, text))
                    elif isinstance(label, str) and label != 'O':
                        # Simple BIO format
                        entity_type = label.replace('B-', '').replace('I-', '')
                        sent_entities.append((entity_type, i, i + 1, ''))
            
            entities.append(sent_entities)
        
        return entities
    
    def _calculate_entity_metrics(self, pred_entities: List[List[Tuple]], 
                                 true_entities: List[List[Tuple]]) -> Dict[str, float]:
        """Calculate entity-level metrics"""
        try:
            # Flatten entity lists
            pred_flat = [entity for sent in pred_entities for entity in sent]
            true_flat = [entity for sent in true_entities for entity in sent]
            
            # Count matches (exact match on type and span)
            pred_set = set((e[0], e[1], e[2]) for e in pred_flat)  # (type, start, end)
            true_set = set((e[0], e[1], e[2]) for e in true_flat)
            
            tp = len(pred_set & true_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'entity_precision': precision,
                'entity_recall': recall,
                'entity_f1': f1,
                'true_positives': tp,
                'false_positives': fp,
                'false_negatives': fn
            }
            
        except Exception:
            return {
                'entity_precision': 0.0,
                'entity_recall': 0.0,
                'entity_f1': 0.0,
                'true_positives': 0,
                'false_positives': 0,
                'false_negatives': 0
            }
    
    def _calculate_token_metrics(self, predictions: List[List[Dict]], 
                                ground_truth: List[List[str]]) -> Dict[str, float]:
        """Calculate token-level metrics"""
        try:
            # This is a simplified implementation
            # In practice, you'd need to align tokens properly
            
            total_tokens = 0
            correct_tokens = 0
            
            for pred_list, true_list in zip(predictions, ground_truth):
                if isinstance(true_list, str):
                    try:
                        true_list = json.loads(true_list)
                    except:
                        true_list = []
                
                # Simple token counting
                total_tokens += len(true_list) if isinstance(true_list, list) else 0
                correct_tokens += min(len(pred_list), len(true_list) if isinstance(true_list, list) else 0)
            
            token_accuracy = correct_tokens / total_tokens if total_tokens > 0 else 0.0
            
            return {
                'token_accuracy': token_accuracy,
                'total_tokens': total_tokens,
                'correct_tokens': correct_tokens
            }
            
        except Exception:
            return {
                'token_accuracy': 0.0,
                'total_tokens': 0,
                'correct_tokens': 0
            }
    
    def _analyze_entity_types(self, pred_entities: List[List[Tuple]], 
                             true_entities: List[List[Tuple]]) -> Dict[str, Any]:
        """Analyze performance by entity type"""
        try:
            # Count entities by type
            pred_types = Counter()
            true_types = Counter()
            
            for sent in pred_entities:
                for entity in sent:
                    pred_types[entity[0]] += 1
            
            for sent in true_entities:
                for entity in sent:
                    true_types[entity[0]] += 1
            
            # Calculate per-type metrics
            type_metrics = {}
            all_types = set(pred_types.keys()) | set(true_types.keys())
            
            for entity_type in all_types:
                pred_count = pred_types[entity_type]
                true_count = true_types[entity_type]
                
                # Simple precision/recall calculation
                precision = pred_count / max(pred_count, 1) if pred_count > 0 else 0.0
                recall = min(pred_count, true_count) / max(true_count, 1) if true_count > 0 else 0.0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
                
                type_metrics[entity_type] = {
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'predicted_count': pred_count,
                    'true_count': true_count
                }
            
            return {
                'type_metrics': type_metrics,
                'total_types': len(all_types),
                'predicted_distribution': dict(pred_types),
                'true_distribution': dict(true_types)
            }
            
        except Exception:
            return {}

class MetricsCalculator:
    """Main metrics calculator for DAPT evaluation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("dapt_metrics")
        
        # Initialize metrics
        self.metrics = {
            'accuracy': AccuracyMetric(),
            'precision_recall_f1': PrecisionRecallF1Metric(),
            'precision_recall_f1_macro': PrecisionRecallF1Metric(average='macro'),
            'precision_recall_f1_micro': PrecisionRecallF1Metric(average='micro'),
            'perplexity': PerplexityMetric(),
            'bleu': BLEUMetric(),
            'arabic_specific': ArabicSpecificMetric(),
            'ner_metrics': NERMetric()
        }
        
        # Custom metrics from config
        self._load_custom_metrics()
    
    def _load_custom_metrics(self) -> None:
        """Load custom metrics from configuration"""
        try:
            custom_metrics = self.config.get('custom_metrics', {})
            
            for name, metric_config in custom_metrics.items():
                # This would allow loading custom metric implementations
                # For now, just log that custom metrics are configured
                self.logger.info(f"Custom metric configured: {name}")
                
        except Exception as e:
            self.logger.warning(f"Error loading custom metrics: {str(e)}")
    
    def calculate_metrics(self, predictions: Any, ground_truth: Any, 
                         task_type: str = "classification",
                         metric_names: Optional[List[str]] = None,
                         **kwargs) -> Dict[str, MetricResult]:
        """Calculate specified metrics"""
        try:
            if metric_names is None:
                metric_names = self._get_default_metrics_for_task(task_type)
            
            results = {}
            
            for metric_name in metric_names:
                if metric_name in self.metrics:
                    try:
                        result = self.metrics[metric_name].calculate(
                            predictions, ground_truth, **kwargs
                        )
                        results[metric_name] = result
                        
                    except Exception as e:
                        self.logger.error(f"Error calculating {metric_name}: {str(e)}")
                        results[metric_name] = MetricResult(
                            metric_name, 0.0, f"Error: {str(e)}"
                        )
                else:
                    self.logger.warning(f"Unknown metric: {metric_name}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in metrics calculation: {str(e)}")
            return {}
    
    def _get_default_metrics_for_task(self, task_type: str) -> List[str]:
        """Get default metrics for a task type"""
        task_metrics = {
            'classification': [
                'accuracy', 'precision_recall_f1', 'precision_recall_f1_macro'
            ],
            'ner': [
                'ner_metrics', 'arabic_specific'
            ],
            'language_modeling': [
                'perplexity', 'arabic_specific'
            ],
            'generation': [
                'bleu', 'arabic_specific'
            ]
        }
        
        return task_metrics.get(task_type, ['accuracy', 'precision_recall_f1'])
    
    def aggregate_metrics(self, metric_results: List[Dict[str, MetricResult]]) -> Dict[str, MetricResult]:
        """Aggregate metrics across multiple evaluations"""
        try:
            if not metric_results:
                return {}
            
            # Get all metric names
            all_metric_names = set()
            for results in metric_results:
                all_metric_names.update(results.keys())
            
            aggregated = {}
            
            for metric_name in all_metric_names:
                values = []
                descriptions = []
                metadata_list = []
                
                for results in metric_results:
                    if metric_name in results:
                        result = results[metric_name]
                        values.append(result.value)
                        descriptions.append(result.description)
                        if result.metadata:
                            metadata_list.append(result.metadata)
                
                if values:
                    # Calculate aggregated statistics
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    min_value = np.min(values)
                    max_value = np.max(values)
                    
                    aggregated_metadata = {
                        'mean': float(mean_value),
                        'std': float(std_value),
                        'min': float(min_value),
                        'max': float(max_value),
                        'count': len(values),
                        'individual_values': values
                    }
                    
                    # Merge individual metadata if available
                    if metadata_list:
                        aggregated_metadata['individual_metadata'] = metadata_list
                    
                    aggregated[metric_name] = MetricResult(
                        metric_name,
                        float(mean_value),
                        f"Aggregated {descriptions[0] if descriptions else metric_name}",
                        aggregated_metadata
                    )
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error aggregating metrics: {str(e)}")
            return {}
    
    def compare_metrics(self, baseline_results: Dict[str, MetricResult],
                       comparison_results: Dict[str, MetricResult]) -> Dict[str, Dict[str, float]]:
        """Compare two sets of metric results"""
        try:
            comparison = {}
            
            # Get common metrics
            common_metrics = set(baseline_results.keys()) & set(comparison_results.keys())
            
            for metric_name in common_metrics:
                baseline_value = baseline_results[metric_name].value
                comparison_value = comparison_results[metric_name].value
                
                # Calculate comparison statistics
                absolute_diff = comparison_value - baseline_value
                relative_diff = (absolute_diff / baseline_value * 100) if baseline_value != 0 else 0.0
                
                comparison[metric_name] = {
                    'baseline': baseline_value,
                    'comparison': comparison_value,
                    'absolute_difference': absolute_diff,
                    'relative_difference_percent': relative_diff,
                    'improvement': absolute_diff > 0
                }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing metrics: {str(e)}")
            return {}
    
    def get_metric_summary(self, metric_results: Dict[str, MetricResult]) -> Dict[str, Any]:
        """Get summary of metric results"""
        try:
            summary = {
                'total_metrics': len(metric_results),
                'metric_names': list(metric_results.keys()),
                'metric_values': {name: result.value for name, result in metric_results.items()},
                'best_metrics': {},
                'worst_metrics': {},
                'average_score': 0.0
            }
            
            if metric_results:
                # Find best and worst performing metrics
                sorted_metrics = sorted(
                    metric_results.items(),
                    key=lambda x: x[1].value,
                    reverse=True
                )
                
                summary['best_metrics'] = {
                    name: result.value for name, result in sorted_metrics[:3]
                }
                summary['worst_metrics'] = {
                    name: result.value for name, result in sorted_metrics[-3:]
                }
                
                # Calculate average score (excluding inf values)
                valid_values = [
                    result.value for result in metric_results.values()
                    if not np.isinf(result.value) and not np.isnan(result.value)
                ]
                
                if valid_values:
                    summary['average_score'] = float(np.mean(valid_values))
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error creating metric summary: {str(e)}")
            return {}
    
    def export_metrics(self, metric_results: Dict[str, MetricResult], 
                      output_path: str, format: str = 'json') -> bool:
        """Export metrics to file"""
        try:
            from pathlib import Path
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for export
            export_data = {
                'metrics': {name: result.to_dict() for name, result in metric_results.items()},
                'summary': self.get_metric_summary(metric_results),
                'export_time': pd.Timestamp.now().isoformat()
            }
            
            if format.lower() == 'json':
                with open(output_path.with_suffix('.json'), 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                # Create DataFrame for CSV export
                df_data = []
                for name, result in metric_results.items():
                    row = {
                        'metric_name': name,
                        'value': result.value,
                        'description': result.description
                    }
                    if result.metadata:
                        row.update(result.metadata)
                    df_data.append(row)
                
                df = pd.DataFrame(df_data)
                df.to_csv(output_path.with_suffix('.csv'), index=False, encoding='utf-8')
            
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Metrics exported to: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting metrics: {str(e)}")
            return False