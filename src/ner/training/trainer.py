"""NER 训练器（Trainer）

实现 NER 模型训练的完整流程，包括：
- 数据准备（数据加载与 DataLoader 构建）
- 模型构建与设备选择
- 优化器与学习率调度器
- 训练与验证循环
- 检查点（checkpoint）保存与恢复
- 最终模型导出与训练元信息记录

使用约定（配置结构关键点）：
- `config['country']`: 包含 `code`/`name` 等基础信息
- `config['data']`: 包含 `train_file`/`val_file`/`max_length` 等
- `config['labels']`: 包含 `num_labels`/`label_names`/`label_mapping` 或 `entities`
- `config['model']`: 包含 `type`/`pretrained_model`/`dropout` 等
- `config['training']`: 包含 `epochs`/`batch_size`/`learning_rate` 等
- `config['output']`: 包含 `model_dir`/`results_dir`/`logs_dir` 等
- `config['hardware']`: 包含 `device`/`num_workers` 等
- `config['logging']`: 包含 `log_file` 等

# TODO: 提供统一的配置 Schema 校验（或复用上层 `ConfigManager` 的校验），并在构造时早失败。
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup, AutoConfig
from typing import Dict, Any, Optional, List
from pathlib import Path
from tqdm import tqdm
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from ..data import NERDataProcessor, NERDataLoader
from ..models import  BertNERModel
from ..evaluation.evaluator import NERMetrics as SeqevalNERMetrics
from ..utils import NERLogger

class NERTrainer:
    """Main trainer class for NER models"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any], logger: Optional[NERLogger] = None):
        """
        初始化 NER 训练器
        
        Args:
            config: 训练配置（国家/模型/数据/训练超参等）
            global_config: 全局配置（如日志/配置根路径等）
        """
        self.config = config
        self.global_config = global_config
        
        # 初始化日志（复用外层注入的 logger，若无则创建本地 logger）
        if logger is not None:
            self.logger = logger
        else:
            raise ValueError("Logger is required")

        # 初始化国家代码
        self.country_code = config.get('country', {}).get('code', None)
        if self.country_code is None:
            raise ValueError("Country code is required")
        
        # 初始化预训练模型名称
        self.pretrained_model_name = config.get('model', {}).get('pretrained_model', None) or global_config.get('pretrained_model', None)
        if self.pretrained_model_name is None:
            raise ValueError("Pretrained model name is required")
        
        # 初始化模型类型
        self.model_type = config.get('model', {}).get('type', None) or global_config.get('model_type', None)
        if self.model_type is None:
            raise ValueError("Model type is required")
        
        # 初始化 TensorBoard SummaryWriter
        logs_root = Path(config.get('output', {}).get('logs_dir', None)) or Path(global_config.get('log_dir', None))
        if logs_root is None:
            raise ValueError("Logs directory is required")
        run_name = f"{self.country_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.tensorboard_log_dir = logs_root / 'tensorboard' / run_name
        self.tensorboard_log_dir.mkdir(parents=True, exist_ok=True)
        self.tb_writer: Optional[SummaryWriter] = SummaryWriter(log_dir=str(self.tensorboard_log_dir))
        self.logger.info(f"TensorBoard logs will be written to: {self.tensorboard_log_dir}")
        
        # 设置设备, device 是训练器中重要的参数，决定了模型在哪个设备上运行
        self.device = self._setup_device()
        
        # 模型
        self.model = None
        # 优化器
        self.optimizer = None
        # 学习率调度器
        self.scheduler = None
        # 数据加载器
        self.data_loaders = {}
        # 全局步数
        self.global_step: int = 0
        
        # Training metrics
        self.training_history = {
            'train_loss': [], # 训练损失
            'val_loss': [], # 验证损失
            'val_f1': [], # 验证F1分数
            'val_precision': [], # 验证精度
            'val_recall': [], # 验证召回率
            'learning_rates': [] # 学习率
        }
        
        # Checkpointing
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = config.get('training', {}).get('early_stopping_patience', 5)
        
        # 训练模型保存输出目录
        self.output_dir = Path(config.get('output', {}).get('model_dir', None)) or Path(global_config.get('model_dir', None))
        if self.output_dir is None:
            raise ValueError("Output directory is required")
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized trainer for {country_code} on device: {self.device.type}")
    
    def _setup_device(self) -> torch.device:
        """设置训练设备（GPU/CPU/指定 CUDA 设备）

        规则：
        - hardware.device == 'auto'：若可用则优先 CUDA，否则 CPU
        - hardware.device == 'cuda'：强制使用 GPU，否则回退 CPU 并告警
        - hardware.device == 'cpu'：强制使用 CPU
        - 支持形如 'cuda:0' 的特定设备

        # TODO: 支持混合精度（fp16/bf16）与 GradScaler，并在此处根据硬件能力初始化策略。
        """
        hardware_config = self.config.get('hardware', {})
        device_config = hardware_config.get('device', 'auto')
        
        if device_config == 'auto':
            # 自动选择：若可用则优先 GPU，否则 CPU
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.info(f"Using GPU (auto-selected): {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                self.logger.info("Using CPU (auto-selected, no GPU available)")
        elif device_config == 'cuda':
            # 强制使用 GPU
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.info(f"Using GPU (forced): {torch.cuda.get_device_name()}")
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device('cpu')
        elif device_config == 'cpu':
            # 强制使用 CPU
            device = torch.device('cpu')
            self.logger.info("Using CPU (forced)")
        else:
            # 处理特定设备，如 'cuda:0'
            if device_config.startswith('cuda:') and torch.cuda.is_available():
                device = torch.device(device_config)
                self.logger.info(f"Using specific GPU device: {device_config}")
            else:
                self.logger.warning(f"Invalid device config '{device_config}', falling back to auto")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.logger.info(f"Using {'GPU' if device.type == 'cuda' else 'CPU'} (fallback)")
        
        return device
    
    def prepare_data(self):
        """准备训练/验证数据

        步骤：
        - 加载训练与验证文件
        - 解析标签定义，支持两种格式：`entities` 或完整的 BIO `label_names`
        - 基于标签构建 `label2id`/`id2label`
        - 构建 `NERDataLoader` 并生成 DataLoader 集合

        注意：
        - 若配置由上游校验强制要求 `label_names/num_labels/label_mapping`，此处的 `entities` 分支与之存在冗余。
          两者需要在规范上统一，避免训练时与评估/导出阶段的标签不一致。
          
          # TODO: 统一标签来源：优先从 `labels.label_names` 读取；如不存在再基于 `entities` 派生 BIO 标签。
        """
        
        data_config = self.config['data'] or {}
        train_file_path = data_config.get('train_file', None)
        val_file_path = data_config.get('val_file', None)

        if not train_file_path:
            raise ValueError("No training data file provided in config")
        if not val_file_path:
            raise ValueError("No validation data file provided in config")
        if not os.path.exists(train_file_path):
            raise ValueError(f"Training data file {train_file_path} does not exist")
        if not os.path.exists(val_file_path):
            raise ValueError(f"Validation data file {val_file_path} does not exist")
        
        self.logger.info(f"Preparing training data from {train_file_path}, validation data from {val_file_path}")

        # 初始化数据处理器
        processor = NERDataProcessor(self.config, logger=self.logger)
        
        # 加载训练数据
        train_dataset = processor.load_data_file(train_file_path)
        self.logger.info(f"Loaded {len(train_dataset)} training dataset from {train_file_path}")

        # 加载验证数据
        val_dataset = processor.load_data_file(val_file_path)
        self.logger.info(f"Loaded {len(val_dataset)} validation dataset from {val_file_path}")
        
        # 初始化标签映射
        labels_config = self.config.get('labels', {})
        if not labels_config:
            raise ValueError("No labels provided in config")
        
        if 'label_names' in labels_config:
            # 使用 label_names 列表格式
            bio_labels = labels_config['label_names']
            if not bio_labels:
                raise ValueError("No label names provided in config")

            entities = set()
            for label in bio_labels:
                if label.startswith('B-') or label.startswith('I-'):
                    entity = label[2:]  # 去除 'B-' 或 'I-' 前缀
                    entities.add(entity)
                    self.logger.debug(f"Found entity: {entity} from label: {label}")
            entities = sorted(list(entities))  # 转换为排序列表以保持一致性
        else:
            raise ValueError("Configuration must contain either 'entities' or 'label_names' in labels section")
        
        # 初始化标签到ID的映射, ex. {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4}
        self.label2id = {label: idx for idx, label in enumerate(bio_labels)}
        # 初始化ID到标签的映射, ex. {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG'}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        self.num_labels = len(self.label2id)
        self.logger.info(f"Initialized label mappings for {len(entities)} entities, total BIO labels={len(bio_labels)}")
        
        # 初始化数据加载器
        ner_data_loader = NERDataLoader(
            tokenizer_name=self.pretrained_model_name,
            label2id=self.label2id,
            max_length=data_config.get('max_length', 512),
            logger=self.logger
        )
        
        # 构建数据加载器, 返回包含 'train'/'val'/'test' 的 DataLoader 字典
        self.data_loaders = ner_data_loader.prepare_loaders(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config.get('hardware', {}).get('num_workers', 0)
        )
        
        self.tokenizer = ner_data_loader.get_tokenizer()
        self.logger.info(f"Created data loaders with {self.num_labels} labels")
    
    def prepare_model(self):
        """准备模型（根据配置加载预训练模型并适配标签数）

        # TODO: 添加支持使用 CRF, 在预测与验证处适配解码流程。
        """
        self.logger.info("Preparing model...")
        
        model_config = self.config['model']

        self.logger.debug(f"Preparing {self.model_type} model...")
        if self.model_type == 'bert' or self.model_type == 'roberta':
            self.model = BertNERModel.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name,
                num_labels=self.num_labels,
                dropout=model_config.get('dropout', 0.1),
                label2id=self.label2id,
                id2label=self.id2label
            )
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
        
        # Move model to device
        self.model.to(self.device)
        
        self.logger.info(f"Initialized {model_config['type']} model with {self.num_labels} labels")
    
    def prepare_optimizer(self):
        """准备优化器与学习率调度器

        - 优化器：默认 AdamW
        - 调度器：linear（基于总步数与 warmup_ratio）或 cosine（按总步数退火）

        # TODO: 按参数类型做权重衰减分组（bias/LayerNorm 不衰减），提升优化效果。
        # TODO: 支持梯度累积（gradient_accumulation_steps）以增大等效 batch size。
        """
        training_config = self.config['training']
        
        # Prepare optimizer
        optimizer_name = training_config.get('optimizer', 'adamw')
        learning_rate = training_config['learning_rate']
        weight_decay = training_config.get('weight_decay', 0.01)

        self.logger.debug(f"Preparing {optimizer_name} optimizer with learning rate {learning_rate} and weight decay {weight_decay}")
        
        if optimizer_name.lower() == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Prepare scheduler
        scheduler_name = training_config.get('scheduler', 'linear')
        num_epochs = training_config['epochs']
        num_training_steps = len(self.data_loaders['train']) * num_epochs
        warmup_steps = int(num_training_steps * training_config.get('warmup_ratio', 0.1))
        
        if scheduler_name.lower() == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")
        
        self.logger.info(f"Initialized {optimizer_name} optimizer with {scheduler_name} scheduler")
    
    def train_epoch(self, epoch: int) -> float:
        """单轮训练
        
        Args:
            epoch: 当前轮次编号（从 0 开始）
            
        Returns:
            本轮平均训练损失

        # TODO: 支持 AMP 混合精度（torch.cuda.amp.autocast + GradScaler）降低显存/提升吞吐。
        # TODO: 支持梯度累积，在大 batch 受限的设备上稳定训练。
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.data_loaders['train'])
        
        progress_bar = tqdm(
            self.data_loaders['train'],
            desc=f"Epoch {epoch + 1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            outputs = self.model(**batch)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping（梯度裁剪，防止梯度爆炸）
            max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # Update parameters
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar（进度条显示当前 loss 与 lr）
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # TensorBoard: batch-level scalars
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('train/batch_loss', float(loss.item()), self.global_step)
                self.tb_writer.add_scalar('train/lr', float(current_lr), self.global_step)
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                self.logger.debug(
                    f"Epoch {epoch + 1}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                )
            
            # Increase global step after logging
            self.global_step += 1
        
        avg_loss = total_loss / num_batches
        
        # TensorBoard: epoch-level train loss
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('train/epoch_loss', float(avg_loss), epoch + 1)
        
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """验证评估
        
        Returns:
            验证指标字典（实体级与 token 级）

        说明：
        - 忽略标签中的 padding（-100）再进行评估。
        - 构建每样本的标签序列，交由 `SeqevalNERMetrics` 计算实体级与 token 级指标。

        # TODO: 若模型包含 CRF 层，应替换为 CRF 解码的预测结果。
        # TODO: 支持输出分类报告或混淆矩阵到文件。
        """
        if 'val' not in self.data_loaders:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        # Accumulate per-sequence labels for entity-level evaluation
        y_true_sequences = []
        y_pred_sequences = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loaders['val'], desc="Validating", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits

                # Get predictions
                predictions = torch.argmax(logits, dim=-1)

                # Build per-sample sequences (strip padding = -100) and convert to label strings
                batch_labels = batch['labels']
                for i in range(batch_labels.size(0)):
                    mask_i = batch_labels[i] != -100
                    true_ids = batch_labels[i][mask_i].tolist()
                    pred_ids = predictions[i][mask_i].tolist()
                    true_seq = [self.id2label.get(int(tid), 'O') for tid in true_ids]
                    pred_seq = [self.id2label.get(int(pid), 'O') for pid in pred_ids]
                    y_true_sequences.append(true_seq)
                    y_pred_sequences.append(pred_seq)

                total_loss += loss.item()
        
        # Calculate entity-level and token-level metrics using seqeval-based metrics
        label_list = [self.id2label[i] for i in range(len(self.id2label))]
        seq_metrics = SeqevalNERMetrics(label_list)

        token_metrics = seq_metrics.compute_token_metrics(y_true_sequences, y_pred_sequences)
        entity_metrics = seq_metrics.compute_entity_metrics(y_true_sequences, y_pred_sequences)

        # Aggregate metrics for trainer consumption (use entity-level by default)
        avg_loss = total_loss / len(self.data_loaders['val'])
        metrics: Dict[str, float] = {
            'precision': entity_metrics.get('entity_precision', 0.0),
            'recall': entity_metrics.get('entity_recall', 0.0),
            'f1': entity_metrics.get('entity_f1', 0.0),
            'val_loss': avg_loss,
            'token_precision': token_metrics.get('token_precision', 0.0),
            'token_recall': token_metrics.get('token_recall', 0.0),
            'token_f1': token_metrics.get('token_f1', 0.0),
            'token_accuracy': token_metrics.get('token_accuracy', 0.0),
        }

        # Optional: include per-entity breakdown if requested in config
        eval_cfg = self.config.get('evaluation', {})
        if eval_cfg.get('return_entity_level_metrics', True) or eval_cfg.get('classification_report', False):
            try:
                per_entity = seq_metrics.compute_per_entity_metrics(y_true_sequences, y_pred_sequences)
                # Flatten selected stats with prefix for easy logging/consumption
                for ent, stats in per_entity.items():
                    metrics[f'entity_{ent}_f1'] = stats.get('f1', 0.0)
                    metrics[f'entity_{ent}_precision'] = stats.get('precision', 0.0)
                    metrics[f'entity_{ent}_recall'] = stats.get('recall', 0.0)
            except Exception:
                pass

        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """保存训练检查点（checkpoint）
        
        Args:
            epoch: 当前轮次编号
            metrics: 当前验证指标
            is_best: 是否为当前最佳

        # TODO: 使用 `safetensors` 或分片保存以降低风险与单文件体积。
        # TODO: 控制检查点保留数量，仅保留最近 N 个与 `best`，节省磁盘空间。
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'label2id': self.label2id,
            'id2label': self.id2label,
            'training_history': self.training_history
        }
        
        # 保存常规检查点
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳检查点
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint with F1: {metrics.get('f1', 0):.4f}")
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
    
    def save_final_model(self):
        """保存最终训练完成的模型与元数据

        - 保存 HF 兼容的模型权重与 tokenizer
        - 基于 `base_model_name` 生成 config.json，并注入 NER 相关字段
        - 另存训练元信息（便于部署/对比/复现实验）

        # TODO: 采用 `safe_serialization=True`（如适用）提高健壮性。
        # TODO: 导出 `label_mapping` 与版本信息，便于推理侧复盘。
        """
        model_dir = self.output_dir / "best_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # Create a Transformers config based on the actual pretrained base to
        # avoid shape mismatches (e.g., vocab_size/model_type must match xlm-roberta-base)
        base_model_name = self.pretrained_model_name
        base_cfg = AutoConfig.from_pretrained(base_model_name)
        # Inject NER-specific fields
        base_cfg.id2label = self.id2label
        base_cfg.label2id = self.label2id
        base_cfg.num_labels = self.config['labels']['num_labels']
        # Optional classifier dropout if present in our config
        dropout_val = self.config['model'].get('dropout', None)
        if dropout_val is not None:
            setattr(base_cfg, 'classifier_dropout', dropout_val)
        # Persist config.json
        config_path = model_dir / "config.json"
        base_cfg.to_json_file(config_path)
        
        # Save training metadata separately
        metadata_path = model_dir / "training_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                'training_config': self.config,
                'training_history': self.training_history,
                'model_info': {
                    'country': self.country_code,
                    'pretrained_model': self.pretrained_model_name,
                    'num_labels': self.config['labels']['num_labels'],
                    'label_names': self.config['labels']['label_names']
                }
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved final model to {model_dir}")
        self.logger.info(f"Saved training metadata to {metadata_path}")
    
    def train(self):
        """主训练循环

        # TODO: 将关键指标写入 TensorBoard/CSV，以便可视化对比与复盘。
        # TODO: 在无验证集的情况下，支持按训练损失或学习率策略做早停的替代策略。
        """
        self.logger.info("Starting training...")
        start_time = time.time()
        
        # 准备数据
        self.prepare_data()
        # 准备模型
        self.prepare_model()
        # 准备优化器
        self.prepare_optimizer()
        
        # 在 TensorBoard 中记录超参数与配置摘要
        if self.tb_writer is not None:
            try:
                # 仅记录关键信息，避免日志过大
                hparams = {
                    'epochs': self.config['training'].get('epochs'),
                    'batch_size': self.config['training'].get('batch_size'),
                    'learning_rate': self.config['training'].get('learning_rate'),
                    'optimizer': self.config['training'].get('optimizer', 'adamw'),
                    'scheduler': self.config['training'].get('scheduler', 'linear'),
                    'max_length': self.config.get('data', {}).get('max_length', 512)
                }
                self.tb_writer.add_text('config/country', str(self.config.get('country', {})))
                self.tb_writer.add_text('config/model', str(self.config.get('model', {})))
                self.tb_writer.add_text('config/training', str(hparams))
            except Exception:
                pass
        
        # 训练轮次
        num_epochs = self.config['training']['epochs']
        
        # 训练循环
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 执行训练轮次，返回训练损失
            # loss 是训练损失，是训练轮次中每个批次损失的平均值
            train_loss = self.train_epoch(epoch)
            
            # 验证训练结果，返回验证指标
            val_metrics = self.validate()
            
            # 更新训练历史结果
            self.training_history['train_loss'].append(train_loss)
            if val_metrics:
                self.training_history['val_loss'].append(val_metrics.get('val_loss', 0))
                self.training_history['val_f1'].append(val_metrics.get('f1', 0))
                self.training_history['val_precision'].append(val_metrics.get('precision', 0))
                self.training_history['val_recall'].append(val_metrics.get('recall', 0))
            
            if self.scheduler:
                self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # TensorBoard: 验证轮次指标
            if self.tb_writer is not None and val_metrics:
                self.tb_writer.add_scalar('val/loss', float(val_metrics.get('val_loss', 0.0)), epoch + 1)
                self.tb_writer.add_scalar('val/f1', float(val_metrics.get('f1', 0.0)), epoch + 1)
                self.tb_writer.add_scalar('val/precision', float(val_metrics.get('precision', 0.0)), epoch + 1)
                self.tb_writer.add_scalar('val/recall', float(val_metrics.get('recall', 0.0)), epoch + 1)
                self.tb_writer.add_scalar('val/token_f1', float(val_metrics.get('token_f1', 0.0)), epoch + 1)
                self.tb_writer.add_scalar('val/token_precision', float(val_metrics.get('token_precision', 0.0)), epoch + 1)
                self.tb_writer.add_scalar('val/token_recall', float(val_metrics.get('token_recall', 0.0)), epoch + 1)
                self.tb_writer.add_scalar('val/token_accuracy', float(val_metrics.get('token_accuracy', 0.0)), epoch + 1)
            
            # 检查最佳模型
            current_f1 = val_metrics.get('f1', 0) if val_metrics else 0
            is_best = current_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = current_f1
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # 记录轮次结果
            epoch_time = time.time() - epoch_start_time
            log_msg = f"Epoch {epoch + 1}/{num_epochs} - "
            log_msg += f"Train Loss: {train_loss:.4f}, "
            log_msg += f"Time: {epoch_time:.2f}s"
            
            if val_metrics:
                log_msg += f", Val Loss: {val_metrics.get('val_loss', 0):.4f}"
                log_msg += f", Val F1: {val_metrics.get('f1', 0):.4f}"
                log_msg += f", Val Precision: {val_metrics.get('precision', 0):.4f}"
                log_msg += f", Val Recall: {val_metrics.get('recall', 0):.4f}"
            
            self.logger.info(log_msg)
            
            # Early stopping（基于验证集 F1 触发）
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_final_model()
        
        # Training summary
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(f"Best validation F1: {self.best_val_f1:.4f}")
        
        # 关闭 TensorBoard Writer
        if self.tb_writer is not None:
            try:
                self.tb_writer.flush()
                self.tb_writer.close()
            except Exception:
                pass
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """从检查点恢复训练
        
        Args:
            checkpoint_path: 检查点文件路径

        # TODO: 处理跨设备恢复（如 CPU 上加载 GPU 保存的状态），并提供自动映射策略。
        # TODO: 校验优化器/调度器状态与当前配置是否兼容，不兼容时给出告警与降级方案。
        """
        self.logger.info(f"Resuming training from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_history = checkpoint.get('training_history', self.training_history)
        self.best_val_f1 = checkpoint.get('metrics', {}).get('f1', 0)
        
        self.logger.info(f"Resumed from epoch {checkpoint['epoch'] + 1}")

class TrainingEngine:
    """高层训练引擎（封装配置加载与训练启动）

    责任：
    - 按国家代码加载配置
    - 应用外部覆盖配置
    - 构造 `NERTrainer` 并启动训练

    # TODO: 覆盖合并应为深度合并（deep merge），避免嵌套字段被整体覆盖。
    """
    
    def __init__(self, global_config: Dict[str, Any]):
        self.global_config = global_config
        self.logger = NERLogger(
            name="training_engine",
            log_dir=global_config.get('log_dir', 'data/ner/logs')
        )
    
    def train_model(self, country: str, config_override: Optional[Dict[str, Any]] = None) -> bool:
        """针对指定国家训练模型
        
        Args:
            country: 国家代码
            config_override: 配置覆盖（浅合并）
            
        Returns:
            训练是否成功
        """
        try:
            # Load configuration
            from ..config import ConfigManager
            
            config_manager = ConfigManager(self.global_config.get('config_dir'))
            config = config_manager.load_country_config(country)
            
            # Apply overrides
            if config_override:
                # TODO: 深度合并，避免覆盖嵌套结构；并记录被覆盖项用于追踪。
                config.update(config_override)
            
            # Initialize trainer
            trainer = NERTrainer(config, self.global_config)
            
            # Start training
            trainer.train()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed for {country}: {e}")
            return False

class TrainingCallbacks:
    """Training callbacks for monitoring and control"""
    
    def __init__(self):
        self.callbacks = []
    
    def add_callback(self, callback):
        """Add training callback"""
        self.callbacks.append(callback)
    
    def on_epoch_start(self, epoch: int, trainer):
        """Called at start of epoch"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_start'):
                callback.on_epoch_start(epoch, trainer)
    
    def on_epoch_end(self, epoch: int, trainer, metrics: Dict[str, float]):
        """Called at end of epoch"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, trainer, metrics)
    
    def on_training_start(self, trainer):
        """Called at start of training"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_start'):
                callback.on_training_start(trainer)
    
    def on_training_end(self, trainer):
        """Called at end of training"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_end'):
                callback.on_training_end(trainer)