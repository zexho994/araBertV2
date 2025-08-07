"""DAPT Evaluation Manager

Evaluation management for DAPT training:
- Coordinate evaluation workflows
- Manage evaluation datasets and metrics
- Generate evaluation reports
- Compare model performances
- Track evaluation history
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForTokenClassification,
    pipeline, Trainer, TrainingArguments
)
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    classification_report, confusion_matrix
)
import torch
from torch.utils.data import DataLoader, Dataset

@dataclass
class EvaluationResult:
    """Evaluation result data structure"""
    model_name: str
    dataset_name: str
    country_code: str
    evaluation_time: str
    metrics: Dict[str, float]
    detailed_metrics: Dict[str, Any]
    predictions: Optional[List[Any]] = None
    ground_truth: Optional[List[Any]] = None
    evaluation_config: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationResult':
        """Create from dictionary"""
        return cls(**data)

@dataclass
class EvaluationDataset:
    """Evaluation dataset information"""
    name: str
    path: str
    country_code: str
    task_type: str  # 'ner', 'classification', 'language_modeling'
    size: int
    description: str
    created_at: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EvaluationDataset':
        """Create from dictionary"""
        return cls(**data)

class EvaluationManager:
    """Main evaluation manager for DAPT training"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
        self.country_code = config["country"]["code"]
        
        # Evaluation paths
        # Use default eval_dir if not provided in global_config
        eval_dir_base = global_config.get("eval_dir", "data/dapt/evaluation")
        self.eval_dir = Path(eval_dir_base) / self.country_code
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_dir = self.eval_dir / "results"
        self.results_dir.mkdir(exist_ok=True)
        
        self.datasets_dir = self.eval_dir / "datasets"
        self.datasets_dir.mkdir(exist_ok=True)
        
        self.reports_dir = self.eval_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)
        
        # Evaluation configuration
        self.eval_config = config.get("evaluation", {})
        self.metrics_config = self.eval_config.get("metrics", {})
        
        # Registered datasets and models
        self.datasets: Dict[str, EvaluationDataset] = {}
        self.evaluation_history: List[EvaluationResult] = []
        
        # Logging (setup first)
        self.logger = self._setup_logging()
        
        # Load existing data (after logger is setup)
        self._load_datasets_registry()
        self._load_evaluation_history()
        
        self.logger.info(f"Evaluation Manager initialized for country: {self.country_code}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for evaluation manager"""
        logger = logging.getLogger(f"dapt_evaluation_{self.country_code}")
        logger.setLevel(getattr(logging, self.config["logging"]["level"]))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = self.config["logging"].get("log_file")
        if log_file:
            log_path = Path(self.global_config["log_dir"]) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def register_dataset(self, name: str, path: str, task_type: str,
                        description: str = "", metadata: Dict[str, Any] = None) -> bool:
        """Register an evaluation dataset"""
        try:
            dataset_path = Path(path)
            if not dataset_path.exists():
                self.logger.error(f"Dataset path does not exist: {path}")
                return False
            
            # Get dataset size
            try:
                if dataset_path.suffix.lower() == '.csv':
                    df = pd.read_csv(dataset_path)
                    size = len(df)
                elif dataset_path.suffix.lower() == '.json':
                    with open(dataset_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    size = len(data) if isinstance(data, list) else 1
                else:
                    size = 0
            except Exception:
                size = 0
            
            # Create dataset info
            dataset = EvaluationDataset(
                name=name,
                path=str(dataset_path.absolute()),
                country_code=self.country_code,
                task_type=task_type,
                size=size,
                description=description,
                created_at=datetime.now().isoformat(),
                metadata=metadata or {}
            )
            
            self.datasets[name] = dataset
            self._save_datasets_registry()
            
            self.logger.info(f"Dataset registered: {name} ({size} samples)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error registering dataset {name}: {str(e)}")
            return False
    
    def evaluate_model(self, model: AutoModel, tokenizer: AutoTokenizer,
                      model_name: str, dataset_name: str,
                      task_type: str = "ner",
                      batch_size: int = 16,
                      max_length: int = 512,
                      save_predictions: bool = True,
                      **kwargs) -> Optional[EvaluationResult]:
        """Evaluate a model on a dataset"""
        try:
            if dataset_name not in self.datasets:
                self.logger.error(f"Dataset not found: {dataset_name}")
                return None
            
            dataset_info = self.datasets[dataset_name]
            
            self.logger.info(f"Evaluating model {model_name} on dataset {dataset_name}")
            
            # Load evaluation data
            eval_data = self._load_evaluation_data(dataset_info.path, task_type)
            if eval_data is None:
                return None
            
            # Perform evaluation based on task type
            if task_type == "ner":
                result = self._evaluate_ner_model(
                    model, tokenizer, eval_data, model_name, dataset_name,
                    batch_size, max_length, **kwargs
                )
            elif task_type == "classification":
                result = self._evaluate_classification_model(
                    model, tokenizer, eval_data, model_name, dataset_name,
                    batch_size, max_length, **kwargs
                )
            elif task_type == "language_modeling":
                result = self._evaluate_lm_model(
                    model, tokenizer, eval_data, model_name, dataset_name,
                    batch_size, max_length, **kwargs
                )
            else:
                self.logger.error(f"Unsupported task type: {task_type}")
                return None
            
            if result:
                # Save evaluation result
                self._save_evaluation_result(result, save_predictions)
                self.evaluation_history.append(result)
                
                self.logger.info(f"Evaluation completed for {model_name} on {dataset_name}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {str(e)}")
            return None
    
    def _load_evaluation_data(self, dataset_path: str, task_type: str) -> Optional[Dict[str, Any]]:
        """Load evaluation data from file"""
        try:
            dataset_path = Path(dataset_path)
            
            if dataset_path.suffix.lower() == '.csv':
                df = pd.read_csv(dataset_path)
                
                if task_type == "ner":
                    # Expect columns: text, labels
                    if 'text' not in df.columns or 'labels' not in df.columns:
                        self.logger.error("NER dataset must have 'text' and 'labels' columns")
                        return None
                    
                    return {
                        'texts': df['text'].tolist(),
                        'labels': df['labels'].tolist()
                    }
                
                elif task_type == "classification":
                    # Expect columns: text, label
                    if 'text' not in df.columns or 'label' not in df.columns:
                        self.logger.error("Classification dataset must have 'text' and 'label' columns")
                        return None
                    
                    return {
                        'texts': df['text'].tolist(),
                        'labels': df['label'].tolist()
                    }
            
            elif dataset_path.suffix.lower() == '.json':
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    if task_type == "ner":
                        texts = [item.get('text', '') for item in data]
                        labels = [item.get('labels', []) for item in data]
                        return {'texts': texts, 'labels': labels}
                    
                    elif task_type == "classification":
                        texts = [item.get('text', '') for item in data]
                        labels = [item.get('label', '') for item in data]
                        return {'texts': texts, 'labels': labels}
            
            self.logger.error(f"Unsupported dataset format: {dataset_path.suffix}")
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading evaluation data: {str(e)}")
            return None
    
    def _evaluate_ner_model(self, model: AutoModel, tokenizer: AutoTokenizer,
                           eval_data: Dict[str, Any], model_name: str, dataset_name: str,
                           batch_size: int, max_length: int, **kwargs) -> Optional[EvaluationResult]:
        """Evaluate NER model"""
        try:
            # Create NER pipeline
            ner_pipeline = pipeline(
                "ner",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            texts = eval_data['texts']
            true_labels = eval_data['labels']
            
            # Get predictions
            predictions = []
            for text in texts:
                try:
                    pred = ner_pipeline(text)
                    predictions.append(pred)
                except Exception as e:
                    self.logger.warning(f"Error predicting for text: {str(e)}")
                    predictions.append([])
            
            # Calculate metrics
            metrics = self._calculate_ner_metrics(predictions, true_labels)
            
            # Create evaluation result
            result = EvaluationResult(
                model_name=model_name,
                dataset_name=dataset_name,
                country_code=self.country_code,
                evaluation_time=datetime.now().isoformat(),
                metrics=metrics['summary'],
                detailed_metrics=metrics['detailed'],
                predictions=predictions,
                ground_truth=true_labels,
                evaluation_config={
                    'task_type': 'ner',
                    'batch_size': batch_size,
                    'max_length': max_length,
                    **kwargs
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in NER evaluation: {str(e)}")
            return None
    
    def _evaluate_classification_model(self, model: AutoModel, tokenizer: AutoTokenizer,
                                     eval_data: Dict[str, Any], model_name: str, dataset_name: str,
                                     batch_size: int, max_length: int, **kwargs) -> Optional[EvaluationResult]:
        """Evaluate classification model"""
        try:
            # Create classification pipeline
            classifier = pipeline(
                "text-classification",
                model=model,
                tokenizer=tokenizer,
                device=0 if torch.cuda.is_available() else -1
            )
            
            texts = eval_data['texts']
            true_labels = eval_data['labels']
            
            # Get predictions
            predictions = []
            for text in texts:
                try:
                    pred = classifier(text)
                    # Get the label with highest score
                    if isinstance(pred, list) and len(pred) > 0:
                        predictions.append(pred[0]['label'])
                    else:
                        predictions.append('UNKNOWN')
                except Exception as e:
                    self.logger.warning(f"Error predicting for text: {str(e)}")
                    predictions.append('UNKNOWN')
            
            # Calculate metrics
            metrics = self._calculate_classification_metrics(predictions, true_labels)
            
            # Create evaluation result
            result = EvaluationResult(
                model_name=model_name,
                dataset_name=dataset_name,
                country_code=self.country_code,
                evaluation_time=datetime.now().isoformat(),
                metrics=metrics['summary'],
                detailed_metrics=metrics['detailed'],
                predictions=predictions,
                ground_truth=true_labels,
                evaluation_config={
                    'task_type': 'classification',
                    'batch_size': batch_size,
                    'max_length': max_length,
                    **kwargs
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in classification evaluation: {str(e)}")
            return None
    
    def _evaluate_lm_model(self, model: AutoModel, tokenizer: AutoTokenizer,
                          eval_data: Dict[str, Any], model_name: str, dataset_name: str,
                          batch_size: int, max_length: int, **kwargs) -> Optional[EvaluationResult]:
        """Evaluate language modeling"""
        try:
            # Calculate perplexity
            texts = eval_data['texts']
            
            total_loss = 0.0
            total_tokens = 0
            
            model.eval()
            with torch.no_grad():
                for text in texts:
                    try:
                        # Tokenize
                        inputs = tokenizer(
                            text,
                            return_tensors="pt",
                            max_length=max_length,
                            truncation=True,
                            padding=True
                        )
                        
                        # Forward pass
                        outputs = model(**inputs, labels=inputs["input_ids"])
                        loss = outputs.loss
                        
                        total_loss += loss.item() * inputs["input_ids"].numel()
                        total_tokens += inputs["input_ids"].numel()
                        
                    except Exception as e:
                        self.logger.warning(f"Error processing text for LM evaluation: {str(e)}")
                        continue
            
            # Calculate perplexity
            avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
            perplexity = np.exp(avg_loss)
            
            metrics = {
                'summary': {
                    'perplexity': perplexity,
                    'average_loss': avg_loss,
                    'total_tokens': total_tokens
                },
                'detailed': {
                    'total_loss': total_loss,
                    'total_tokens': total_tokens,
                    'texts_processed': len(texts)
                }
            }
            
            # Create evaluation result
            result = EvaluationResult(
                model_name=model_name,
                dataset_name=dataset_name,
                country_code=self.country_code,
                evaluation_time=datetime.now().isoformat(),
                metrics=metrics['summary'],
                detailed_metrics=metrics['detailed'],
                predictions=None,  # No predictions for LM
                ground_truth=None,
                evaluation_config={
                    'task_type': 'language_modeling',
                    'batch_size': batch_size,
                    'max_length': max_length,
                    **kwargs
                }
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in language modeling evaluation: {str(e)}")
            return None
    
    def _calculate_ner_metrics(self, predictions: List[List[Dict]], 
                              true_labels: List[List[str]]) -> Dict[str, Any]:
        """Calculate NER evaluation metrics"""
        try:
            # Flatten predictions and labels for entity-level evaluation
            pred_entities = []
            true_entities = []
            
            for pred_list, true_list in zip(predictions, true_labels):
                # Extract entities from predictions
                pred_ents = []
                for entity in pred_list:
                    if isinstance(entity, dict) and 'entity' in entity:
                        pred_ents.append(entity['entity'])
                
                # Convert true labels to entity list
                if isinstance(true_list, str):
                    try:
                        true_list = json.loads(true_list)
                    except:
                        true_list = []
                
                pred_entities.extend(pred_ents)
                true_entities.extend(true_list if isinstance(true_list, list) else [])
            
            # Calculate metrics
            if len(true_entities) > 0 and len(pred_entities) > 0:
                # Get unique labels
                all_labels = list(set(true_entities + pred_entities))
                
                # Calculate precision, recall, F1
                precision, recall, f1, support = precision_recall_fscore_support(
                    true_entities, pred_entities, labels=all_labels, average='weighted', zero_division=0
                )
                
                accuracy = accuracy_score(true_entities, pred_entities)
                
                # Detailed metrics per entity type
                detailed_report = classification_report(
                    true_entities, pred_entities, labels=all_labels, output_dict=True, zero_division=0
                )
                
                return {
                    'summary': {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    },
                    'detailed': {
                        'classification_report': detailed_report,
                        'entity_counts': {
                            'predicted': len(pred_entities),
                            'true': len(true_entities)
                        }
                    }
                }
            else:
                return {
                    'summary': {
                        'accuracy': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1_score': 0.0
                    },
                    'detailed': {
                        'classification_report': {},
                        'entity_counts': {
                            'predicted': len(pred_entities),
                            'true': len(true_entities)
                        }
                    }
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating NER metrics: {str(e)}")
            return {
                'summary': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
                'detailed': {}
            }
    
    def _calculate_classification_metrics(self, predictions: List[str], 
                                        true_labels: List[str]) -> Dict[str, Any]:
        """Calculate classification evaluation metrics"""
        try:
            # Calculate basic metrics
            accuracy = accuracy_score(true_labels, predictions)
            precision, recall, f1, support = precision_recall_fscore_support(
                true_labels, predictions, average='weighted', zero_division=0
            )
            
            # Detailed metrics
            unique_labels = list(set(true_labels + predictions))
            detailed_report = classification_report(
                true_labels, predictions, labels=unique_labels, output_dict=True, zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(true_labels, predictions, labels=unique_labels)
            
            return {
                'summary': {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                },
                'detailed': {
                    'classification_report': detailed_report,
                    'confusion_matrix': cm.tolist(),
                    'label_names': unique_labels,
                    'sample_count': len(predictions)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating classification metrics: {str(e)}")
            return {
                'summary': {'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0},
                'detailed': {}
            }
    
    def compare_models(self, model_names: List[str], dataset_name: str,
                      metric: str = 'f1_score') -> Optional[Dict[str, Any]]:
        """Compare multiple models on the same dataset"""
        try:
            # Find evaluation results for the specified models and dataset
            results = []
            for result in self.evaluation_history:
                if (result.model_name in model_names and 
                    result.dataset_name == dataset_name and
                    metric in result.metrics):
                    results.append(result)
            
            if not results:
                self.logger.warning(f"No evaluation results found for comparison")
                return None
            
            # Group by model name and get the latest result for each
            model_results = {}
            for result in results:
                model_name = result.model_name
                if (model_name not in model_results or 
                    result.evaluation_time > model_results[model_name].evaluation_time):
                    model_results[model_name] = result
            
            # Create comparison data
            comparison = {
                'dataset_name': dataset_name,
                'metric': metric,
                'comparison_time': datetime.now().isoformat(),
                'models': {}
            }
            
            for model_name, result in model_results.items():
                comparison['models'][model_name] = {
                    'metric_value': result.metrics.get(metric, 0.0),
                    'all_metrics': result.metrics,
                    'evaluation_time': result.evaluation_time
                }
            
            # Rank models by the specified metric
            ranked_models = sorted(
                comparison['models'].items(),
                key=lambda x: x[1]['metric_value'],
                reverse=True
            )
            
            comparison['ranking'] = [
                {'model_name': name, 'metric_value': data['metric_value']}
                for name, data in ranked_models
            ]
            
            # Save comparison report
            self._save_comparison_report(comparison)
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            return None
    
    def generate_evaluation_report(self, model_name: str = None, 
                                 dataset_name: str = None,
                                 include_plots: bool = True) -> Optional[str]:
        """Generate comprehensive evaluation report"""
        try:
            # Filter results based on criteria
            filtered_results = self.evaluation_history
            
            if model_name:
                filtered_results = [r for r in filtered_results if r.model_name == model_name]
            
            if dataset_name:
                filtered_results = [r for r in filtered_results if r.dataset_name == dataset_name]
            
            if not filtered_results:
                self.logger.warning("No evaluation results found for report generation")
                return None
            
            # Generate report
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_name = f"evaluation_report_{timestamp}"
            
            if model_name:
                report_name += f"_{model_name}"
            if dataset_name:
                report_name += f"_{dataset_name}"
            
            report_path = self.reports_dir / f"{report_name}.html"
            
            # Create HTML report
            html_content = self._create_html_report(filtered_results, include_plots)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            self.logger.info(f"Evaluation report generated: {report_path}")
            return str(report_path)
            
        except Exception as e:
            self.logger.error(f"Error generating evaluation report: {str(e)}")
            return None
    
    def _create_html_report(self, results: List[EvaluationResult], 
                           include_plots: bool = True) -> str:
        """Create HTML evaluation report"""
        try:
            # Basic HTML structure
            html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>DAPT Evaluation Report - {self.country_code}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .section {{ margin: 20px 0; }}
        .metric-table {{ border-collapse: collapse; width: 100%; }}
        .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        .metric-table th {{ background-color: #f2f2f2; }}
        .model-section {{ border: 1px solid #ccc; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .metric-value {{ font-weight: bold; color: #2e7d32; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>DAPT Evaluation Report</h1>
        <p><strong>Country:</strong> {self.country_code}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p><strong>Total Evaluations:</strong> {len(results)}</p>
    </div>
"""
            
            # Summary section
            html += "\n<div class='section'>\n<h2>Summary</h2>\n"
            
            # Model summary
            models = list(set(r.model_name for r in results))
            datasets = list(set(r.dataset_name for r in results))
            
            html += f"<p><strong>Models Evaluated:</strong> {', '.join(models)}</p>\n"
            html += f"<p><strong>Datasets Used:</strong> {', '.join(datasets)}</p>\n"
            html += "</div>\n"
            
            # Results by model
            html += "\n<div class='section'>\n<h2>Evaluation Results</h2>\n"
            
            for model_name in models:
                model_results = [r for r in results if r.model_name == model_name]
                
                html += f"\n<div class='model-section'>\n<h3>{model_name}</h3>\n"
                
                # Create metrics table for this model
                html += "<table class='metric-table'>\n"
                html += "<tr><th>Dataset</th><th>Task Type</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>Evaluation Time</th></tr>\n"
                
                for result in model_results:
                    metrics = result.metrics
                    task_type = result.evaluation_config.get('task_type', 'unknown') if result.evaluation_config else 'unknown'
                    
                    html += f"<tr>"
                    html += f"<td>{result.dataset_name}</td>"
                    html += f"<td>{task_type}</td>"
                    html += f"<td class='metric-value'>{metrics.get('accuracy', 0.0):.4f}</td>"
                    html += f"<td class='metric-value'>{metrics.get('precision', 0.0):.4f}</td>"
                    html += f"<td class='metric-value'>{metrics.get('recall', 0.0):.4f}</td>"
                    html += f"<td class='metric-value'>{metrics.get('f1_score', 0.0):.4f}</td>"
                    html += f"<td>{result.evaluation_time[:19]}</td>"
                    html += f"</tr>\n"
                
                html += "</table>\n</div>\n"
            
            html += "</div>\n"
            
            # Detailed metrics section
            html += "\n<div class='section'>\n<h2>Detailed Metrics</h2>\n"
            
            for result in results:
                html += f"\n<div class='model-section'>\n"
                html += f"<h4>{result.model_name} on {result.dataset_name}</h4>\n"
                
                # Display detailed metrics if available
                if result.detailed_metrics:
                    html += "<pre>" + json.dumps(result.detailed_metrics, indent=2) + "</pre>\n"
                
                html += "</div>\n"
            
            html += "</div>\n"
            
            # Close HTML
            html += "\n</body>\n</html>"
            
            return html
            
        except Exception as e:
            self.logger.error(f"Error creating HTML report: {str(e)}")
            return "<html><body><h1>Error generating report</h1></body></html>"
    
    def _save_evaluation_result(self, result: EvaluationResult, save_predictions: bool = True) -> None:
        """Save evaluation result to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{result.model_name}_{result.dataset_name}_{timestamp}.json"
            filepath = self.results_dir / filename
            
            # Prepare data for saving
            result_data = result.to_dict()
            
            # Optionally exclude predictions to save space
            if not save_predictions:
                result_data['predictions'] = None
                result_data['ground_truth'] = None
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Evaluation result saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving evaluation result: {str(e)}")
    
    def _save_comparison_report(self, comparison: Dict[str, Any]) -> None:
        """Save model comparison report"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comparison_{comparison['dataset_name']}_{timestamp}.json"
            filepath = self.reports_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Comparison report saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving comparison report: {str(e)}")
    
    def _load_datasets_registry(self) -> None:
        """Load datasets registry from file"""
        try:
            registry_file = self.eval_dir / "datasets_registry.json"
            if registry_file.exists():
                with open(registry_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.datasets = {
                    name: EvaluationDataset.from_dict(dataset_data)
                    for name, dataset_data in data.items()
                }
                
                self.logger.debug(f"Loaded {len(self.datasets)} datasets from registry")
            
        except Exception as e:
            self.logger.warning(f"Could not load datasets registry: {str(e)}")
    
    def _save_datasets_registry(self) -> None:
        """Save datasets registry to file"""
        try:
            registry_file = self.eval_dir / "datasets_registry.json"
            
            data = {
                name: dataset.to_dict()
                for name, dataset in self.datasets.items()
            }
            
            with open(registry_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug("Datasets registry saved")
            
        except Exception as e:
            self.logger.error(f"Error saving datasets registry: {str(e)}")
    
    def _load_evaluation_history(self) -> None:
        """Load evaluation history from files"""
        try:
            self.evaluation_history = []
            
            for result_file in self.results_dir.glob("*.json"):
                try:
                    with open(result_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    result = EvaluationResult.from_dict(data)
                    self.evaluation_history.append(result)
                    
                except Exception as e:
                    self.logger.warning(f"Could not load result from {result_file}: {str(e)}")
            
            # Sort by evaluation time
            self.evaluation_history.sort(key=lambda x: x.evaluation_time, reverse=True)
            
            self.logger.debug(f"Loaded {len(self.evaluation_history)} evaluation results")
            
        except Exception as e:
            self.logger.warning(f"Could not load evaluation history: {str(e)}")
    
    def save_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save evaluation results to specified file"""
        try:
            import os
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Results saved to: {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving results to {output_file}: {str(e)}")
            raise
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of all evaluations"""
        try:
            models = list(set(r.model_name for r in self.evaluation_history))
            datasets = list(set(r.dataset_name for r in self.evaluation_history))
            
            # Calculate average metrics by model
            model_averages = {}
            for model in models:
                model_results = [r for r in self.evaluation_history if r.model_name == model]
                
                if model_results:
                    avg_metrics = {}
                    for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                        values = [r.metrics.get(metric, 0.0) for r in model_results if metric in r.metrics]
                        avg_metrics[metric] = np.mean(values) if values else 0.0
                    
                    model_averages[model] = avg_metrics
            
            return {
                'total_evaluations': len(self.evaluation_history),
                'unique_models': len(models),
                'unique_datasets': len(datasets),
                'models': models,
                'datasets': list(self.datasets.keys()),
                'model_averages': model_averages,
                'latest_evaluation': self.evaluation_history[0].evaluation_time if self.evaluation_history else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting evaluation summary: {str(e)}")
            return {}