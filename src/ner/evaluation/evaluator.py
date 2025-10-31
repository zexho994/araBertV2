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

from typing import Dict, List, Any, Optional

import torch
import torch.nn.functional as F
from seqeval.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from seqeval.scheme import IOB2
from sklearn.metrics import (
    precision_recall_fscore_support
)

from ..utils import NERLogger

class NERMetrics:
    """NER 评估指标计算器

    职责：
    - 计算 token 级指标（加权 precision/recall/F1 与 accuracy）
    - 计算实体级指标（基于 seqeval 严格模式 IOB2）
    - 计算逐实体类别指标（基于实体跨度集合的集合运算）

    """

    label_list = [] # 标签列表
    ignore_labels = ['O'] # 忽略的标签列表
    entity_types = [] # 实体类型列表
    
    def __init__(self, label_list: List[str], ignore_labels: Optional[List[str]] = None):
        """
        Initialize metrics calculator
        
        Args:
            label_list: 标签列表
            ignore_labels: 忽略的标签列表 (默认: ['O'])
        """
        
        if not label_list:
            raise ValueError("label_list is required")

        self.label_list = label_list

        # 如果 ignore_labels 不为空, 则设置忽略的标签列表
        if ignore_labels:
            self.ignore_labels = ignore_labels

        self.entity_types = self._extract_entity_types(label_list)
    
    def _extract_entity_types(self, label_list: List[str]) -> List[str]:
        """从 BIO 标签中抽取实体类型集合

        例如: 
            label_list = ['B-PER', 'I-PER', 'O', 'B-ORG', 'I-ORG']
            返回: ['PER', 'ORG']

        说明:
        - 从标签列表中抽取实体类型集合, 例如 'PER', 'ORG'
        - 忽略忽略的标签, 例如 'O'
        - 忽略标签中不包含 '-' 的标签, 例如 'O'
        - 返回的实体类型列表按字母顺序排序, 例如 ['PER', 'ORG']
        """
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
        # 展平序列
        true_flat = [label for seq in y_true for label in seq]
        pred_flat = [label for seq in y_pred for label in seq]
        
        # 过滤忽略标签
        filtered_true = []
        filtered_pred = []
        
        for true_label, pred_label in zip(true_flat, pred_flat):
            if true_label not in self.ignore_labels:
                filtered_true.append(true_label)
                filtered_pred.append(pred_label)
        
        if not filtered_true:
            return {
                'token_precision': 0.0,
                'token_recall': 0.0,
                'token_f1': 0.0,
                'token_accuracy': 0.0
            }
        
        # 获取唯一标签
        unique_labels = sorted(list(set(filtered_true + filtered_pred)))
        
        # 计算 precision, recall, f1
        precision, recall, f1, _ = precision_recall_fscore_support(
            filtered_true, filtered_pred, labels=unique_labels, average='weighted', zero_division=0
        )
        
        # 计算 accuracy
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
            # 使用 seqeval 计算实体级指标
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
            # 提取指定类型的实体跨度集合
            true_entities = self._extract_entities_for_type(y_true, entity_type)
            pred_entities = self._extract_entities_for_type(y_pred, entity_type)
            
            # 计算 TP/FP/FN
            tp = len(true_entities & pred_entities) # 真阳性, 预测正确的实体
            fp = len(pred_entities - true_entities) # 假阳性, 预测错误的实体
            fn = len(true_entities - pred_entities) # 假阴性, 漏掉的实体
            
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
        """
        entities = set()

        for seq_idx, sequence in enumerate(sequences):
            current_entity = None

            for token_idx, label in enumerate(sequence):
                if label == f'B-{entity_type}':
                    if current_entity:
                        entities.add(current_entity)
                    current_entity = (seq_idx, token_idx, token_idx + 1)

                elif label == f'I-{entity_type}' and current_entity is not None:
                    current_entity = (current_entity[0], current_entity[1], token_idx + 1)

                else:
                    if current_entity:
                        entities.add(current_entity)
                        current_entity = None

            if current_entity:
                entities.add(current_entity)

        return entities


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
        device: Optional[torch.device] = None,
        logger: Optional[NERLogger] = None
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
        self.logger = logger
        
        # Create label mappings
        self.label2id = {label: i for i, label in enumerate(label_list)}
        self.id2label = {i: label for i, label in enumerate(label_list)}
    
    def evaluate_predict(
        self, 
        texts: List[str], 
        true_labels: List[List[str]],
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """对原始文本进行评估
        
        Args:
            texts: 输入文本列表
            true_labels: 对应的真实标签序列
            confidence_threshold: 预测置信度阈值
            
        Returns:
            评估结果字典

        # TODO: 若模型支持 batch 预测，考虑批量化以提升吞吐。
        """
        predictions = []
        
        for text, true_seq in zip(texts, true_labels):
            # 使用真实 tokens 进行预测以避免与空白切词策略不一致
            words = text.split()
            # 若文本与标签长度不一致，优先使用标签长度截断文本
            if len(words) != len(true_seq):
                words = words[:len(true_seq)]
            result = self.model.predict_tokens(
                words,
                tokenizer=self.tokenizer,
                confidence_threshold=confidence_threshold,
                device=self.device
            )
            # 确保预测标签与真实标签长度完全一致
            pred_labels = result['labels']
            expected_length = len(true_seq)
            
            if len(pred_labels) < expected_length:
                # 若预测标签不足，用 'O' 填充
                pred_labels = pred_labels + ['O'] * (expected_length - len(pred_labels))
            elif len(pred_labels) > expected_length:
                # 若预测标签过长，截断
                pred_labels = pred_labels[:expected_length]
            
            predictions.append(pred_labels)
        
        token_metrics = self.metrics_calculator.compute_token_metrics(
            true_labels, predictions
        )
        entity_metrics  = self.metrics_calculator.compute_entity_metrics(
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

    def evaluate_dataloader(self, dataloader, confidence_threshold: Optional[float] = None) -> Dict[str, Any]:
        """基于 DataLoader 进行评估（与训练时对齐方式一致）

        Args:
            dataloader: 数据加载器
            confidence_threshold: 置信度阈值（可选）。如果提供，只有置信度高于阈值的预测才会被保留，否则设为 'O'
        
        返回：
            指标与可选的平均损失（若 batch 含 labels）
        """
        self.model.eval()

        total_loss: float = 0.0
        num_batches: int = 0

        y_true_sequences: List[List[str]] = []
        y_pred_sequences: List[List[str]] = []
        original_tokens_list: List[List[str]] = []  # 存储原始tokens，用于对齐验证

        # 获取 'O' 标签的 ID（用于置信度过滤）
        o_label_id = self.label2id.get('O', 0)

        with torch.no_grad():
            for batch in dataloader:
                # 保存原始tokens信息（如果batch中包含）
                batch_original_tokens = None
                if 'original_tokens' in batch:
                    # 如果是tensor，需要转换为list
                    if isinstance(batch['original_tokens'], torch.Tensor):
                        batch_original_tokens = batch['original_tokens'].tolist() if batch['original_tokens'].dim() == 0 else None
                    else:
                        batch_original_tokens = batch['original_tokens']
                
                # 将批次迁移到设备（但原始tokens不需要迁移）
                batch_for_model = {k: v.to(self.device) if isinstance(v, torch.Tensor) and k not in ['original_tokens', 'original_labels'] else v 
                                  for k, v in batch.items()}

                outputs = self.model(**{k: v for k, v in batch_for_model.items() if k not in ['original_tokens', 'original_labels']})
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits

                # 若存在损失则累积
                loss = outputs.get('loss', None) if isinstance(outputs, dict) else getattr(outputs, 'loss', None)
                if loss is not None:
                    total_loss += float(loss.item())
                    num_batches += 1

                # 根据是否使用置信度阈值选择不同的预测方式
                if confidence_threshold is not None:
                    # 计算概率分布
                    probabilities = F.softmax(logits, dim=-1)
                    # 获取最大概率和对应的预测
                    max_probs, predictions = torch.max(probabilities, dim=-1)
                    # 如果最大概率低于阈值，将预测设为 'O'
                    o_label_tensor = torch.full_like(predictions, o_label_id, device=self.device)
                    predictions = torch.where(max_probs >= confidence_threshold, predictions, o_label_tensor)
                else:
                    # 不使用置信度阈值，直接使用 argmax
                    predictions = torch.argmax(logits, dim=-1)

                batch_labels = batch_for_model.get('labels', None)
                if batch_labels is None:
                    # 若无标签，无法计算指标
                    continue

                # 逐样本解码（仅保留 labels != -100 的位置，这些对应word-level的标签）
                for i in range(batch_labels.size(0)):
                    mask_i = batch_labels[i] != -100
                    true_ids = batch_labels[i][mask_i].tolist()
                    pred_ids = predictions[i][mask_i].tolist()

                    true_seq = [self.id2label.get(int(tid), 'O') for tid in true_ids]
                    pred_seq = [self.id2label.get(int(pid), 'O') for pid in pred_ids]

                    y_true_sequences.append(true_seq)
                    y_pred_sequences.append(pred_seq)
                    
                    # 获取原始tokens（如果可用）
                    if batch_original_tokens is not None and isinstance(batch_original_tokens, list) and i < len(batch_original_tokens):
                        original_tokens_list.append(batch_original_tokens[i])
                    elif 'original_tokens' in batch:
                        # 从batch中直接获取
                        if isinstance(batch['original_tokens'], list) and i < len(batch['original_tokens']):
                            original_tokens_list.append(batch['original_tokens'][i])
                        else:
                            original_tokens_list.append([])
                    else:
                        # 尝试从dataset中获取（回退方案）
                        try:
                            dataset = dataloader.dataset
                            # 计算当前样本在dataset中的索引（需要考虑batch索引）
                            # 简单回退：使用已处理的样本数量
                            sample_idx = len(original_tokens_list)
                            if hasattr(dataset, 'processed_datasets') and sample_idx < len(dataset.processed_datasets):
                                orig_tokens = dataset.processed_datasets[sample_idx].get('original_tokens', [])
                                if isinstance(orig_tokens, list):
                                    original_tokens_list.append(orig_tokens)
                                else:
                                    original_tokens_list.append([])
                            else:
                                original_tokens_list.append([])
                        except Exception:
                            original_tokens_list.append([])

        token_metrics = self.metrics_calculator.compute_token_metrics(y_true_sequences, y_pred_sequences)
        entity_metrics = self.metrics_calculator.compute_entity_metrics(y_true_sequences, y_pred_sequences)
        per_entity_metrics = self.metrics_calculator.compute_per_entity_metrics(y_true_sequences, y_pred_sequences)

        results: Dict[str, Any] = {
            'token_metrics': token_metrics,
            'entity_metrics': entity_metrics,
            'per_entity_metrics': per_entity_metrics,
            'predictions': y_pred_sequences,
            'true_labels': y_true_sequences,
            'original_tokens': original_tokens_list,  # 添加原始tokens信息
            'num_samples': len(y_true_sequences)
        }

        if num_batches > 0:
            results['val_loss'] = total_loss / num_batches

        return results

    def print_evaluate_results(self, results):
        """打印评估结果
        
        Args:
            results: 评估结果
        
        Returns:
            None
        """
        self.logger.info("Evaluation Results:")

        # 打印 token 级指标
        self.logger.info("\nToken-level Metrics:")
        for metric, value in results.get('token_metrics', {}).items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        # 打印实体级指标
        self.logger.info("Entity-level Metrics:")
        for metric, value in results.get('entity_metrics', {}).items():
            self.logger.info(f"  {metric}: {value:.4f}")
        
        # 打印逐实体指标
        if 'per_entity_metrics' in results:
            self.logger.info("Per-Entity Metrics:")
            for entity, metrics in results['per_entity_metrics'].items():
                self.logger.info(f"  {entity}:")
                for metric, value in metrics.items():
                    self.logger.info(f"    {metric}: {value:.4f}")
        
        self.logger.info(f"Total samples evaluated: {results.get('num_samples', 0)}")
        if 'val_loss' in results:
            self.logger.info(f"Validation loss: {results['val_loss']:.4f}")