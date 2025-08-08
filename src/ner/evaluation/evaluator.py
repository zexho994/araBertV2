"""NER Model Evaluation

Implements comprehensive evaluation functionality for NER models
including metrics calculation, performance analysis, and reporting.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from collections import defaultdict, Counter
import json
import os
from datetime import datetime
from sklearn.metrics import (
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)
from seqeval.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report as seq_classification_report
)
from seqeval.scheme import IOB2

class NERMetrics:
    """NER evaluation metrics calculator"""
    
    def __init__(self, label_list: List[str], ignore_labels: Optional[List[str]] = None):
        """
        Initialize metrics calculator
        
        Args:
            label_list: List of all possible labels
            ignore_labels: Labels to ignore in evaluation (default: ['O'])
        """
        self.label_list = label_list
        self.ignore_labels = ignore_labels or ['O']
        self.entity_types = self._extract_entity_types(label_list)
    
    def _extract_entity_types(self, label_list: List[str]) -> List[str]:
        """Extract entity types from BIO labels"""
        entity_types = set()
        for label in label_list:
            if label not in self.ignore_labels and '-' in label:
                entity_type = label.split('-', 1)[1]
                entity_types.add(entity_type)
        return sorted(list(entity_types))
    
    def compute_token_metrics(
        self, 
        y_true: List[List[str]], 
        y_pred: List[List[str]]
    ) -> Dict[str, float]:
        """Compute token-level metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Token-level metrics
        """
        # Flatten sequences
        true_flat = [label for seq in y_true for label in seq]
        pred_flat = [label for seq in y_pred for label in seq]
        
        # Filter out ignored labels
        filtered_true = []
        filtered_pred = []
        
        for true_label, pred_label in zip(true_flat, pred_flat):
            if true_label not in self.ignore_labels:
                filtered_true.append(true_label)
                filtered_pred.append(pred_label)
        
        # Calculate metrics
        if not filtered_true:
            return {
                'token_precision': 0.0,
                'token_recall': 0.0,
                'token_f1': 0.0,
                'token_accuracy': 0.0
            }
        
        # Get unique labels
        unique_labels = sorted(list(set(filtered_true + filtered_pred)))
        
        # Calculate precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            filtered_true, filtered_pred, labels=unique_labels, average='weighted', zero_division=0
        )
        
        # Calculate accuracy
        accuracy = sum(t == p for t, p in zip(filtered_true, filtered_pred)) / len(filtered_true)
        
        return {
            'token_precision': float(precision),
            'token_recall': float(recall),
            'token_f1': float(f1),
            'token_accuracy': float(accuracy)
        }
    
    def compute_entity_metrics(
        self, 
        y_true: List[List[str]], 
        y_pred: List[List[str]]
    ) -> Dict[str, float]:
        """Compute entity-level metrics using seqeval
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Entity-level metrics
        """
        try:
            # Use seqeval for entity-level evaluation
            precision = precision_score(y_true, y_pred, mode='strict', scheme=IOB2)
            recall = recall_score(y_true, y_pred, mode='strict', scheme=IOB2)
            f1 = f1_score(y_true, y_pred, mode='strict', scheme=IOB2)
            accuracy = accuracy_score(y_true, y_pred)
            
            return {
                'entity_precision': float(precision),
                'entity_recall': float(recall),
                'entity_f1': float(f1),
                'entity_accuracy': float(accuracy)
            }
        except Exception as e:
            print(f"Warning: Error computing entity metrics: {e}")
            return {
                'entity_precision': 0.0,
                'entity_recall': 0.0,
                'entity_f1': 0.0,
                'entity_accuracy': 0.0
            }
    
    def compute_per_entity_metrics(
        self, 
        y_true: List[List[str]], 
        y_pred: List[List[str]]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics per entity type
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Per-entity metrics
        """
        per_entity_metrics = {}
        
        for entity_type in self.entity_types:
            # Extract entities for this type
            true_entities = self._extract_entities_for_type(y_true, entity_type)
            pred_entities = self._extract_entities_for_type(y_pred, entity_type)
            
            # Calculate metrics
            tp = len(true_entities & pred_entities)
            fp = len(pred_entities - true_entities)
            fn = len(true_entities - pred_entities)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            per_entity_metrics[entity_type] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1),
                'support': len(true_entities)
            }
        
        return per_entity_metrics
    
    def _extract_entities_for_type(
        self, 
        sequences: List[List[str]], 
        entity_type: str
    ) -> set:
        """Extract entities of specific type from sequences"""
        entities = set()
        
        for seq_idx, sequence in enumerate(sequences):
            current_entity = None
            
            for token_idx, label in enumerate(sequence):
                if label == f'B-{entity_type}':
                    # Start of new entity
                    if current_entity:
                        entities.add(current_entity)
                    current_entity = (seq_idx, token_idx, token_idx + 1)
                
                elif label == f'I-{entity_type}' and current_entity:
                    # Continuation of entity
                    current_entity = (current_entity[0], current_entity[1], token_idx + 1)
                
                else:
                    # End of entity or outside
                    if current_entity:
                        entities.add(current_entity)
                        current_entity = None
            
            # Add last entity if exists
            if current_entity:
                entities.add(current_entity)
        
        return entities
    
    def compute_confusion_matrix(
        self, 
        y_true: List[List[str]], 
        y_pred: List[List[str]]
    ) -> Tuple[np.ndarray, List[str]]:
        """Compute confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix and label names
        """
        # Flatten sequences
        true_flat = [label for seq in y_true for label in seq]
        pred_flat = [label for seq in y_pred for label in seq]
        
        # Get unique labels
        unique_labels = sorted(list(set(true_flat + pred_flat)))
        
        # Compute confusion matrix
        cm = confusion_matrix(true_flat, pred_flat, labels=unique_labels)
        
        return cm, unique_labels

class NEREvaluator:
    """Main NER model evaluator"""
    
    def __init__(
        self, 
        model, 
        tokenizer, 
        label_list: List[str],
        device: Optional[torch.device] = None
    ):
        """
        Initialize evaluator
        
        Args:
            model: NER model to evaluate
            tokenizer: Tokenizer
            label_list: List of all possible labels
            device: Device to use for evaluation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.label_list = label_list
        self.device = device or torch.device('cpu')
        self.metrics_calculator = NERMetrics(label_list)
        
        # Create label mappings
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for i, label in enumerate(label_list)}
    
    def evaluate(
        self, 
        dataloader, 
        return_predictions: bool = False,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Evaluate model on dataset
        
        Args:
            dataloader: DataLoader for evaluation data
            return_predictions: Whether to return predictions
            confidence_threshold: Confidence threshold for predictions
            
        Returns:
            Evaluation results
        """
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_logits = []
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                
                # Accumulate loss
                if 'loss' in outputs and outputs['loss'] is not None:
                    total_loss += outputs['loss'].item()
                
                # Get predictions
                logits = outputs['logits']
                predictions = torch.argmax(logits, dim=-1)
                
                # Store results
                all_logits.extend(logits.cpu().numpy())
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(batch['labels'].cpu().numpy())
                
                num_batches += 1
        
        # Convert to sequences
        pred_sequences, true_sequences = self._convert_to_sequences(
            all_predictions, all_labels, batch.get('attention_mask')
        )
        
        # Compute metrics
        token_metrics = self.metrics_calculator.compute_token_metrics(
            true_sequences, pred_sequences
        )
        entity_metrics = self.metrics_calculator.compute_entity_metrics(
            true_sequences, pred_sequences
        )
        per_entity_metrics = self.metrics_calculator.compute_per_entity_metrics(
            true_sequences, pred_sequences
        )
        
        # Prepare results
        results = {
            'loss': total_loss / num_batches if num_batches > 0 else 0.0,
            'token_metrics': token_metrics,
            'entity_metrics': entity_metrics,
            'per_entity_metrics': per_entity_metrics,
            'num_samples': len(pred_sequences),
            'num_batches': num_batches
        }
        
        if return_predictions:
            results['predictions'] = pred_sequences
            results['true_labels'] = true_sequences
            results['logits'] = all_logits
        
        return results
    
    def _convert_to_sequences(
        self, 
        predictions: List[np.ndarray], 
        labels: List[np.ndarray],
        attention_masks: Optional[List[np.ndarray]] = None
    ) -> Tuple[List[List[str]], List[List[str]]]:
        """Convert predictions and labels to sequences
        
        Args:
            predictions: Model predictions
            labels: True labels
            attention_masks: Attention masks
            
        Returns:
            Prediction and label sequences
        """
        pred_sequences = []
        true_sequences = []
        
        for i, (pred, true) in enumerate(zip(predictions, labels)):
            # Get attention mask if available
            mask = attention_masks[i] if attention_masks else None
            
            pred_seq = []
            true_seq = []
            
            for j, (p, t) in enumerate(zip(pred, true)):
                # Skip if masked out
                if mask is not None and mask[j] == 0:
                    continue
                
                # Skip special tokens
                if t == -100:  # Ignore index
                    continue
                
                pred_label = self.id2label.get(p, 'O')
                true_label = self.id2label.get(t, 'O')
                
                pred_seq.append(pred_label)
                true_seq.append(true_label)
            
            if pred_seq and true_seq:
                pred_sequences.append(pred_seq)
                true_sequences.append(true_seq)
        
        return pred_sequences, true_sequences
    
    def evaluate_text(
        self, 
        texts: List[str], 
        true_labels: List[List[str]],
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """Evaluate model on raw texts
        
        Args:
            texts: Input texts
            true_labels: True labels for each text
            confidence_threshold: Confidence threshold
            
        Returns:
            Evaluation results
        """
        predictions = []
        
        for text in texts:
            # Check if model is wrapped (TransformersNERModelWrapper)
            if hasattr(self.model, 'tokenizer') and hasattr(self.model, 'id2label'):
                # Use wrapped model's predict method (doesn't need tokenizer parameter)
                result = self.model.predict(
                    text, 
                    confidence_threshold=confidence_threshold
                )
            else:
                # Use standard model predict method (BertNERModel)
                result = self.model.predict(
                    text, 
                    self.tokenizer, 
                    confidence_threshold=confidence_threshold,
                    device=self.device
                )
            predictions.append(result['labels'])
        
        # Compute metrics
        token_metrics = self.metrics_calculator.compute_token_metrics(
            true_labels, predictions
        )
        entity_metrics = self.metrics_calculator.compute_entity_metrics(
            true_labels, predictions
        )
        per_entity_metrics = self.metrics_calculator.compute_per_entity_metrics(
            true_labels, predictions
        )
        
        return {
            'token_metrics': token_metrics,
            'entity_metrics': entity_metrics,
            'per_entity_metrics': per_entity_metrics,
            'predictions': predictions,
            'true_labels': true_labels,
            'num_samples': len(texts)
        }
    
    def generate_classification_report(
        self, 
        y_true: List[List[str]], 
        y_pred: List[List[str]]
    ) -> str:
        """Generate detailed classification report
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report string
        """
        try:
            # Use seqeval for entity-level report
            report = seq_classification_report(y_true, y_pred, mode='strict', scheme=IOB2)
            return report
        except Exception as e:
            print(f"Warning: Error generating classification report: {e}")
            return "Classification report generation failed"

class EvaluationReporter:
    """Generate evaluation reports"""
    
    def __init__(self, output_dir: str = "evaluation_reports"):
        """
        Initialize reporter
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(
        self, 
        results: Dict[str, Any], 
        model_name: str = "ner_model",
        dataset_name: str = "evaluation_dataset"
    ) -> str:
        """Generate comprehensive evaluation report
        
        Args:
            results: Evaluation results
            model_name: Name of the model
            dataset_name: Name of the dataset
            
        Returns:
            Path to generated report
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"{model_name}_{dataset_name}_{timestamp}.json"
        report_path = os.path.join(self.output_dir, report_filename)
        
        # Prepare report data
        report_data = {
            'metadata': {
                'model_name': model_name,
                'dataset_name': dataset_name,
                'timestamp': timestamp,
                'evaluation_date': datetime.now().isoformat()
            },
            'results': results
        }
        
        # Save report
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        # Generate summary report
        summary_path = self._generate_summary_report(report_data, report_path)
        
        return report_path
    
    def _generate_summary_report(
        self, 
        report_data: Dict[str, Any], 
        json_report_path: str
    ) -> str:
        """Generate human-readable summary report
        
        Args:
            report_data: Report data
            json_report_path: Path to JSON report
            
        Returns:
            Path to summary report
        """
        summary_path = json_report_path.replace('.json', '_summary.txt')
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("NER Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Metadata
            metadata = report_data['metadata']
            f.write(f"Model: {metadata['model_name']}\n")
            f.write(f"Dataset: {metadata['dataset_name']}\n")
            f.write(f"Evaluation Date: {metadata['evaluation_date']}\n\n")
            
            # Results
            results = report_data['results']
            
            # Overall metrics
            if 'token_metrics' in results:
                f.write("Token-level Metrics:\n")
                f.write("-" * 20 + "\n")
                for metric, value in results['token_metrics'].items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("\n")
            
            if 'entity_metrics' in results:
                f.write("Entity-level Metrics:\n")
                f.write("-" * 20 + "\n")
                for metric, value in results['entity_metrics'].items():
                    f.write(f"{metric}: {value:.4f}\n")
                f.write("\n")
            
            # Per-entity metrics
            if 'per_entity_metrics' in results:
                f.write("Per-Entity Metrics:\n")
                f.write("-" * 20 + "\n")
                for entity, metrics in results['per_entity_metrics'].items():
                    f.write(f"\n{entity}:\n")
                    for metric, value in metrics.items():
                        f.write(f"  {metric}: {value:.4f}\n")
                f.write("\n")
            
            # Additional info
            if 'num_samples' in results:
                f.write(f"Number of samples: {results['num_samples']}\n")
            if 'loss' in results:
                f.write(f"Average loss: {results['loss']:.4f}\n")
        
        return summary_path
    
    def compare_models(
        self, 
        reports: List[str], 
        output_filename: str = "model_comparison.txt"
    ) -> str:
        """Compare multiple model evaluation reports
        
        Args:
            reports: List of report file paths
            output_filename: Output comparison file name
            
        Returns:
            Path to comparison report
        """
        comparison_path = os.path.join(self.output_dir, output_filename)
        
        # Load all reports
        report_data = []
        for report_path in reports:
            with open(report_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                report_data.append(data)
        
        # Generate comparison
        with open(comparison_path, 'w', encoding='utf-8') as f:
            f.write("Model Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            
            # Compare entity-level F1 scores
            f.write("Entity-level F1 Scores:\n")
            f.write("-" * 30 + "\n")
            for data in report_data:
                model_name = data['metadata']['model_name']
                f1_score = data['results'].get('entity_metrics', {}).get('entity_f1', 0.0)
                f.write(f"{model_name}: {f1_score:.4f}\n")
            f.write("\n")
            
            # Compare per-entity metrics
            f.write("Per-Entity F1 Comparison:\n")
            f.write("-" * 30 + "\n")
            
            # Get all entity types
            all_entities = set()
            for data in report_data:
                per_entity = data['results'].get('per_entity_metrics', {})
                all_entities.update(per_entity.keys())
            
            for entity in sorted(all_entities):
                f.write(f"\n{entity}:\n")
                for data in report_data:
                    model_name = data['metadata']['model_name']
                    per_entity = data['results'].get('per_entity_metrics', {})
                    f1_score = per_entity.get(entity, {}).get('f1', 0.0)
                    f.write(f"  {model_name}: {f1_score:.4f}\n")
        
        return comparison_path