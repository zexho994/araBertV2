"""NER 模型评估模块

提供完整的评估能力，包括：
- 指标计算（token 级与实体级）
- 逐实体类别的精确率/召回率/F1 统计
- 混淆矩阵生成（token 级）
- 评估报告（JSON 与可读摘要）

设计要点：
- 基于 seqeval（IOB2 严格模式）计算实体级指标；
- 基于 sklearn 的加权平均计算 token 级指标；
- 提供 `NEREvaluator` 统一对接模型、tokenizer 与数据加载器；
- `EvaluationReporter` 负责结果持久化与摘要生成。

# TODO: 支持可配置的标签方案（IOB/IOB2/IOBES），以及微/宏/加权等汇总策略。
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
    """NER 评估指标计算器

    职责：
    - 计算 token 级指标（加权 precision/recall/F1 与 accuracy）
    - 计算实体级指标（基于 seqeval 严格模式 IOB2）
    - 计算逐实体类别指标（基于实体跨度集合的集合运算）

    # TODO: `ignore_labels` 默认忽略 'O'；应允许按需扩展/关闭忽略机制。
    """
    
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
        """从 BIO/IOB2 标签中抽取实体类型集合"""
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
        """计算 token 级指标
        
        Args:
            y_true: 真实标签序列（二维）
            y_pred: 预测标签序列（二维）
            
        Returns:
            包含 token_precision/token_recall/token_f1/token_accuracy 的字典

        说明：
        - 先展平序列；
        - 过滤忽略标签（默认忽略 'O'）；
        - 使用 sklearn 的加权平均（weighted）汇总。

        # TODO: 支持 average 策略可配（micro/macro/weighted），并暴露 zero_division 策略。
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
        """使用 seqeval 计算实体级指标
        
        Args:
            y_true: 真实标签序列（二维）
            y_pred: 预测标签序列（二维）
            
        Returns:
            包含 entity_precision/entity_recall/entity_f1/entity_accuracy 的字典

        # TODO: 将 `mode` 与 `scheme` 参数暴露为可配置项（当前固定为 strict + IOB2）。
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
        """按实体类别统计指标
        
        Args:
            y_true: 真实标签序列
            y_pred: 预测标签序列
            
        Returns:
            每个实体类别对应的 precision/recall/f1/support 指标

        实现说明：
        - 通过 `_extract_entities_for_type` 将每类实体转为跨度集合（(seq_idx, start, end)）。
        - 使用集合运算计算 TP/FP/FN。

        # TODO: 支持忽略特定实体类别或做类别重加权；支持置信度阈值筛选（若模型输出概率）。
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
        """从序列中抽取指定类型的实体跨度集合

        约束：
        - 假设标签符合 IOB2 规范；若存在非法序列（如孤立 I-），当前实现会忽略不匹配片段。
        
        # TODO: 支持 IOBES/BILOU 等标注方案；对非法序列做更健壮的纠错策略。
        """
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
    """NER 主评估器

    职责：
    - 接收模型、tokenizer 与标签列表
    - 在 DataLoader 或文本集合上执行评估
    - 汇总并返回统一的评估结果结构

    # TODO: 支持半精度评估与多 GPU（DataParallel）的自动适配。
    """
    
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
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        """在数据集上评估模型
        
        Args:
            dataloader: 评估数据的 DataLoader
            return_predictions: 是否返回预测序列/标签/原始 logits
            confidence_threshold: 预测阈值（用于部分模型的文本预测接口）
            
        Returns:
            评估结果字典

        注意：
        - 将所有批次的 logits/预测/标签累积在内存中；大型数据集可能带来内存开销。
          # TODO: 支持流式聚合与按需保存。
        - attention_mask 的处理存在缺陷：当前实现仅传入“最后一个 batch 的 mask”。
          
          # ERROR: 需要在循环中累积所有批次的 attention_mask，并整体传入 `_convert_to_sequences`。
          # TODO: 在循环内 `all_attention_masks.append(batch['attention_mask'].cpu().numpy())`，
          #       并在调用 `_convert_to_sequences` 时传入拼接后的 masks。
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
        
        # Convert to sequences（将预测/标签转为字符串序列，并据 attention_mask 去除 padding/special tokens）
        pred_sequences, true_sequences = self._convert_to_sequences(
            all_predictions, all_labels, batch.get('attention_mask')
        )
        
        # Compute metrics（计算 token/实体级与逐实体指标）
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
        """将预测与标签张量转换为标签字符串序列
        
        Args:
            predictions: 模型预测（列表，元素为 [seq_len] 的 numpy 数组）
            labels: 真实标签（列表，元素为 [seq_len] 的 numpy 数组）
            attention_masks: 注意力 mask（可选，列表，对应每个样本）
            
        Returns:
            预测序列与真实序列（每个样本一条序列）

        注意：
        - 当 `attention_masks` 为 None 时，仅依据 -100 忽略索引过滤 padding；
        - 当提供 `attention_masks` 时，会进一步依据 mask==0 过滤。

        # TODO: 在上层 `evaluate` 中提供完整的 `attention_masks` 列表，而非单一批次的 mask。
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
        """对原始文本进行评估（无需 DataLoader）
        
        Args:
            texts: 输入文本列表
            true_labels: 对应的真实标签序列
            confidence_threshold: 预测置信度阈值
            
        Returns:
            评估结果字典

        # TODO: 若模型支持 batch 预测，考虑批量化以提升吞吐。
        """
        predictions = []
        
        for text in texts:
            if hasattr(self.model, 'tokenizer') and hasattr(self.model, 'id2label'):
                result = self.model.predict(
                    text, 
                    confidence_threshold=confidence_threshold
                )
            else:
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
        """生成实体级详细分类报告（基于 seqeval）
        
        Args:
            y_true: 真实标签序列
            y_pred: 预测标签序列
            
        Returns:
            分类报告字符串

        # TODO: 支持导出为 Markdown/HTML，并附带每类支持度与示例片段。
        """
        try:
            # Use seqeval for entity-level report
            report = seq_classification_report(y_true, y_pred, mode='strict', scheme=IOB2)
            return report
        except Exception as e:
            print(f"Warning: Error generating classification report: {e}")
            return "Classification report generation failed"

class EvaluationReporter:
    """评估报告生成器

    职责：
    - 持久化评估结果为 JSON
    - 生成简要的可读摘要文本

    # TODO: 支持 HTML/CSV 导出与可视化图表（如 PR 曲线、混淆矩阵热力图）。
    """
    
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
        """生成完整评估报告（JSON + 摘要）
        
        Args:
            results: 评估结果字典
            model_name: 模型名
            dataset_name: 数据集名
            
        Returns:
            生成的 JSON 报告文件路径

        # TODO: 元数据中补充模型版本、数据集版本、标签方案、评估配置等关键信息。
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
        """生成可读性较好的摘要报告（.txt）
        
        Args:
            report_data: 报告数据（元信息 + 评估结果）
            json_report_path: JSON 报告路径
            
        Returns:
            生成的摘要报告路径
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
        """对多个模型评估报告进行对比
        
        Args:
            reports: 报告文件路径列表（JSON）
            output_filename: 输出对比文件名
            
        Returns:
            生成的对比报告路径

        # TODO: 按关键指标排序（如 entity_f1），并输出表格化的对比；
        # TODO: 汇总逐实体指标差异，标注优势/短板类别。
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