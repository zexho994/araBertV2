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

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, AutoConfig

from ..data import NERDataProcessor, NERDataLoader
from ..evaluation.evaluator import NERMetrics as SeqevalNERMetrics
from ..models import BertNERModel
from ..utils import NERLogger

class NERTrainer:
    """Main trainer class for NER models"""
    config = {}
    global_config = {}
    country_code: str
    pretrained_model_name: str
    model_type: str
    logger: NERLogger

    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any], logger: NERLogger):
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
        self.country_code = self.config.get('country', {}).get('code', '')
        if self.country_code == '':
            raise ValueError("Country code is required")
        
        # 初始化预训练模型名称
        self.pretrained_model_name = self.config.get('model', {}).get('pretrained_model', '')
        if self.pretrained_model_name == '':
            raise ValueError("Pretrained model name is required")
        
        # 初始化模型类型
        self.model_type = self.config.get('model', {}).get('type', '')
        if self.model_type == '':
            raise ValueError("Model type is required")
        
        # 初始化 TensorBoard SummaryWriter
        logs_root = Path(self.config.get('output', {}).get('logs_dir', ''))
        if logs_root == '':
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
        self.output_dir = Path(self.config.get('output', {}).get('model_dir', ''))
        if self.output_dir == '':
            raise ValueError("Output directory is required")
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Initialized trainer for {self.country_code} on device: {self.device.type}")
    
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
        train_file_path = data_config.get('train_file', '')
        val_file_path = data_config.get('val_file', '')

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
        
        支持LoRA增量训练，可以基于之前的LoRA模型继续训练
        
        # TODO: 添加支持使用 CRF, 在预测与验证处适配解码流程。
        """
        self.logger.info("Preparing model...")
        
        model_config = self.config['model']
        training_config = self.config['training']
        
        # 检查是否启用LoRA
        lora_config = training_config.get('lora', {})
        use_lora = lora_config.get('enabled', False)
        
        # 检查是否有指定的LoRA基础模型路径（用于增量训练）
        lora_base_model_path = lora_config.get('base_adapter_path', None)
        
        self.logger.debug(f"Preparing {self.model_type} model with LoRA={use_lora}...")
        
        # 检查是否要加载现有模型（LoRA 或完整模型）
        if use_lora and lora_base_model_path:
            # LoRA 增量训练：使用 manager 加载现有 LoRA 模型
            try:
                from ..models import NERModelManager
                
                # 检查路径是否存在
                if not os.path.exists(lora_base_model_path):
                    self.logger.warning(f"指定的LoRA基础模型路径不存在: {lora_base_model_path}，将创建新的LoRA模型")
                    self._create_new_model(model_config)
                else:
                    self.logger.info(f"正在加载LoRA基础模型: {lora_base_model_path}")
                    
                    # 使用 manager 加载现有模型
                    model_manager = NERModelManager(logger=self.logger)
                    self.model = model_manager.load_model(lora_base_model_path)
                    
                    # 确保模型是可训练的
                    if hasattr(self.model, 'train'):
                        self.model.train()
                    
                    self.logger.info("成功加载LoRA基础模型，将在此基础上继续训练")
            except Exception as e:
                self.logger.error(f"加载LoRA基础模型失败: {str(e)}，将创建新的LoRA模型")
                self._create_new_model(model_config)
        elif self._is_local_model_path(self.pretrained_model_name):
            # 本地模型路径：使用 manager 加载完整模型
            try:
                from ..models import NERModelManager
                
                self.logger.info(f"检测到本地模型路径，使用 manager 加载: {self.pretrained_model_name}")
                
                # 使用 manager 加载现有模型
                model_manager = NERModelManager(logger=self.logger)
                self.model = model_manager.load_model(self.pretrained_model_name)
                
                # 确保模型是可训练的
                if hasattr(self.model, 'train'):
                    self.model.train()
                
                self.logger.info("成功加载本地模型，将在此基础上继续训练")
            except Exception as e:
                self.logger.error(f"加载本地模型失败: {str(e)}，将创建新模型")
                self._create_new_model(model_config)
        else:
            # 预训练模型或 HuggingFace 模型
            self._create_new_model(model_config)
        
        # Move model to device
        self.model.to(self.device)
        self.logger.info(f"Initialized {model_config['type']} model with {self.num_labels} labels")
    
    def _is_local_model_path(self, model_path: str) -> bool:
        """检查是否为本地模型路径
        
        参数：
            model_path: 模型路径
            
        返回：
            是否为本地路径
        """
        # 检查是否为本地路径（包含 / 或 \ 或 . 开头）
        return ('/' in model_path or '\\' in model_path or 
                model_path.startswith('.') or 
                os.path.exists(model_path))
    
    def _is_continued_training(self) -> bool:
        """检查是否为继续训练（加载了现有模型）
        
        返回：
            是否为继续训练
        """
        # 检查是否加载了本地模型（非 HuggingFace 模型）
        return self._is_local_model_path(self.pretrained_model_name)
    
    def _setup_continued_training_optimizer(self, base_learning_rate: float, weight_decay: float):
        """设置继续训练的优化器（分层学习率）
        
        参数：
            base_learning_rate: 基础学习率
            weight_decay: 权重衰减
        """
        # 分层学习率：BERT 层使用更小的学习率，分类头使用正常学习率
        bert_lr = base_learning_rate * 0.1  # BERT 层：原学习率的 1/10
        classifier_lr = base_learning_rate * 0.5  # 分类头：原学习率的 1/2
        
        self.logger.info(f"继续训练分层学习率设置:")
        self.logger.info(f"  - BERT 层: {base_learning_rate} -> {bert_lr}")
        self.logger.info(f"  - 分类头: {base_learning_rate} -> {classifier_lr}")
        
        # 分离 BERT 层和分类头参数
        bert_params = []
        classifier_params = []
        
        for name, param in self.model.named_parameters():
            if 'bert' in name:
                bert_params.append(param)
            else:
                classifier_params.append(param)
        
        # 创建参数组
        param_groups = [
            {'params': bert_params, 'lr': bert_lr, 'weight_decay': weight_decay},
            {'params': classifier_params, 'lr': classifier_lr, 'weight_decay': weight_decay}
        ]
        
        self.optimizer = AdamW(param_groups)
    
    def _create_new_model(self, model_config):
        """创建新的模型实例，支持LoRA配置"""
        if self.model_type == 'bert' or self.model_type == 'roberta':
            # 先创建基础模型
            base_model = BertNERModel.from_pretrained(
                pretrained_model_name_or_path=self.pretrained_model_name,
                num_labels=self.num_labels,
                dropout=model_config.get('dropout', 0.1),
                label2id=self.label2id,
                id2label=self.id2label
            )
            
            # 检查是否需要应用LoRA
            training_config = self.config['training']
            lora_config = training_config.get('lora', {})
            use_lora = lora_config.get('enabled', False)
            
            if use_lora:
                self.logger.info("Applying LoRA configuration to base model...")
                try:
                    from peft import LoraConfig, get_peft_model, TaskType
                    
                    # 创建LoRA配置
                    peft_config = LoraConfig(
                        task_type=TaskType.TOKEN_CLS,  # 用于Token分类任务
                        inference_mode=False,  # 训练模式
                        r=lora_config.get('r', 16),
                        lora_alpha=lora_config.get('alpha', 32),
                        lora_dropout=lora_config.get('dropout', 0.1),
                        target_modules=lora_config.get('target_modules', ["query", "key", "value", "dense"]),
                        bias=lora_config.get('bias', "none")
                    )
                    
                    # 应用LoRA到基础模型
                    self.model = get_peft_model(base_model, peft_config)
                    
                    # 打印可训练参数信息
                    self.model.print_trainable_parameters()
                    self.logger.info("Successfully applied LoRA configuration")
                    
                except ImportError as e:
                    self.logger.error("PEFT library not available. Cannot create LoRA model.")
                    raise ImportError("PEFT library is required for LoRA training. Install with: pip install peft") from e
                except Exception as e:
                    self.logger.error(f"Failed to apply LoRA configuration: {e}")
                    raise e
            else:
                # 不使用LoRA，直接使用基础模型
                self.model = base_model
                self.logger.info("Created standard (non-LoRA) model")
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
    
    def prepare_optimizer(self):
        """准备优化器与学习率调度器

        - 优化器：默认 AdamW
        - 调度器：linear（基于总步数与 warmup_ratio）或 cosine（按总步数退火）
        - 继续训练：自动调整学习率，避免破坏已学习的特征

        # TODO: 按参数类型做权重衰减分组（bias/LayerNorm 不衰减），提升优化效果。
        # TODO: 支持梯度累积（gradient_accumulation_steps）以增大等效 batch size。
        """
        training_config = self.config['training']
        
        # Prepare optimizer
        optimizer_name = training_config.get('optimizer', 'adamw')
        base_learning_rate = training_config['learning_rate']
        weight_decay = training_config.get('weight_decay', 0.01)
        
        if optimizer_name.lower() == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=base_learning_rate,
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
        self.model.train() # 设置模型为训练模式
        total_loss = 0.0
        num_batches = len(self.data_loaders['train'])
        
        # 进度条显示当前训练轮次、训练损失、学习率
        progress_bar = tqdm(
            self.data_loaders['train'], # 训练数据集
            desc=f"Epoch {epoch + 1}", # 进度条显示当前训练轮次
            leave=False # 进度条不显示
        )
        
        # 遍历训练数据集
        for batch_idx, batch in enumerate(progress_bar):

            # 将批次数据移动到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播, 返回损失
            outputs = self.model(**batch)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            
            # 反向传播, 计算梯度
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪，防止梯度爆炸
            max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # 更新参数
            self.optimizer.step()

            # 更新学习率
            if self.scheduler:
                self.scheduler.step()

            # 更新总损失
            total_loss += loss.item()
            
            # 更新进度条（进度条显示当前 loss 与 lr）
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # TensorBoard: 记录当前批次损失与学习率
            if self.tb_writer is not None:
                self.tb_writer.add_scalar('train/batch_loss', float(loss.item()), self.global_step)
                self.tb_writer.add_scalar('train/lr', float(current_lr), self.global_step)
            
            # 记录当前批次损失与学习率
            if batch_idx % 100 == 0:
                self.logger.debug(
                    f"Epoch {epoch + 1}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                )
            
            # 更新全局步数
            self.global_step += 1
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        
        # TensorBoard: 记录当前轮次损失
        if self.tb_writer is not None:
            self.tb_writer.add_scalar('train/epoch_loss', float(avg_loss), epoch + 1)
        
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """验证评估
        
        Returns:
            验证指标字典, 包含实体级与 token 级指标

        说明：
        - 忽略标签中的 padding（-100）再进行评估。
        - 构建每样本的标签序列，交由 `SeqevalNERMetrics` 计算实体级与 token 级指标。

        # TODO: 若模型包含 CRF 层，应替换为 CRF 解码的预测结果。
        # TODO: 支持输出分类报告或混淆矩阵到文件。
        """
        if 'val' not in self.data_loaders:
            return {}

        # 设置模型为评估模式
        self.model.eval()

        # 初始化总损失
        total_loss = 0.0

        # 累积每样本的标签序列，用于实体级评估
        y_true_sequences = []
        y_pred_sequences = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loaders['val'], desc="Validating", leave=False):

                # 将批次数据移动到设备
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # 前向传播, 返回损失与 logits
                outputs = self.model(**batch)

                # loss 是损失值, 是当前批次所有样本的平均损失
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss

                # logits 是模型输出, 是当前批次所有样本的预测结果
                # 例如: 
                # logits = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
                # 因为 logits 的第二个维度是 3, 所以预测结果是 2
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits

                # 获取预测结果, argmax 是取最大值的索引
                # dim=-1 是取最后一个维度, 即每个样本的预测结果
                # 预测结果是每个样本的预测标签, 是当前批次所有样本的预测结果
                # 例如: logits = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
                # predictions = [2, 2]
                predictions = torch.argmax(logits, dim=-1)

                # 构建每样本的标签序列, 并转换为标签字符串
                # 例如: batch_labels = [[1, 2, 3], [4, 5, 6]]
                batch_labels = batch['labels']

                # 遍历当前批次所有样本
                for i in range(batch_labels.size(0)):
                    # 例如: mask_i = [True, True, True]
                    mask_i = batch_labels[i] != -100
                    # 例如: true_ids = [1, 2, 3]
                    true_ids = batch_labels[i][mask_i].tolist()
                    # 例如: pred_ids = [2, 2, 2]
                    pred_ids = predictions[i][mask_i].tolist()
                    # 例如: true_seq = ['B-PER', 'I-PER', 'O']
                    true_seq = [self.id2label.get(int(tid), 'O') for tid in true_ids]
                    # 例如: pred_seq = ['B-PER', 'I-PER', 'O']
                    pred_seq = [self.id2label.get(int(pid), 'O') for pid in pred_ids]
                    # 例如: y_true_sequences = [['B-PER', 'I-PER', 'O'], ['B-PER', 'I-PER', 'O']]
                    y_true_sequences.append(true_seq)
                    # 例如: y_pred_sequences = [['B-PER', 'I-PER', 'O'], ['B-PER', 'I-PER', 'O']]
                    y_pred_sequences.append(pred_seq)

                # 更新总损失
                total_loss += loss.item()
        
        # 计算实体级与 token 级指标
        label_list = [self.id2label[i] for i in range(len(self.id2label))]

        # 创建 SeqevalNERMetrics 实例
        seq_metrics = SeqevalNERMetrics(label_list)

        # 计算 token 级指标
        token_metrics = seq_metrics.compute_token_metrics(y_true_sequences, y_pred_sequences)

        # 计算实体级指标
        entity_metrics = seq_metrics.compute_entity_metrics(y_true_sequences, y_pred_sequences)

        # 计算平均损失
        avg_loss = total_loss / len(self.data_loaders['val'])

        # 创建验证指标字典
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

        # 如果配置中请求了实体级指标, 则计算实体级指标
        eval_cfg = self.config.get('evaluation', {})
        if eval_cfg.get('return_entity_level_metrics', True) or eval_cfg.get('classification_report', False):
            try:
                per_entity = seq_metrics.compute_per_entity_metrics(y_true_sequences, y_pred_sequences)
                # 将实体级指标扁平化, 便于日志记录与消费
                for ent, stats in per_entity.items():
                    metrics[f'entity_{ent}_f1'] = stats.get('f1', 0.0)
                    metrics[f'entity_{ent}_precision'] = stats.get('precision', 0.0)
                    metrics[f'entity_{ent}_recall'] = stats.get('recall', 0.0)
            except Exception as e:
                self.logger.error(f"计算实体级指标时出错: {e}")
                pass

        return metrics
    
    def _save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
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
        - 支持LoRA模型保存

        # TODO: 采用 `safe_serialization=True`（如适用）提高健壮性。
        # TODO: 导出 `label_mapping` 与版本信息，便于推理侧复盘。
        """
        model_dir = self.output_dir / "best_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # 检查是否为LoRA模型
        training_config = self.config['training']
        lora_config = training_config.get('lora', {})
        use_lora = lora_config.get('enabled', False)
        
        # 保存模型
        if use_lora:
            self.logger.info("Saving LoRA model...")
            # 对于LoRA模型，让PEFT库处理保存逻辑
            self.model.save_pretrained(model_dir)
            
            # 保存tokenizer
            self.tokenizer.save_pretrained(model_dir)
            
            # 检查PEFT是否正确生成了adapter_config.json
            adapter_config_path = model_dir / "adapter_config.json"
            if adapter_config_path.exists():
                self.logger.info(f"PEFT successfully saved adapter_config.json")
            else:
                self.logger.warning("adapter_config.json not found after PEFT save_pretrained")
            
            # 保存LoRA配置信息（作为备份）
            lora_config_path = model_dir / "lora_config.json"
            with open(lora_config_path, 'w', encoding='utf-8') as f:
                json.dump(lora_config, f, ensure_ascii=False, indent=2)
                
            # 为LoRA模型生成基础模型配置（不覆盖PEFT文件）
            # 检查是否已存在config.json，如果没有则创建
            config_path = model_dir / "config.json"
            if not config_path.exists():
                self.logger.info("Creating base model config.json for LoRA model...")
                # 从基础模型路径获取配置，确保配置正确
                try:
                    if hasattr(self.model, 'base_model') and hasattr(self.model.base_model, 'config'):
                        # 从PEFT模型的base_model获取配置
                        base_cfg = self.model.base_model.config
                    else:
                        # 备用方案：从预训练模型路径加载
                        base_cfg = AutoConfig.from_pretrained(self.pretrained_model_name)
                    
                    # 注入NER相关字段
                    base_cfg.id2label = self.id2label
                    base_cfg.label2id = self.label2id
                    base_cfg.num_labels = self.config['labels']['num_labels']
                    
                    # 添加分类器dropout
                    dropout_val = self.config['model'].get('dropout', None)
                    if dropout_val is not None:
                        setattr(base_cfg, 'classifier_dropout', dropout_val)
                    
                    # 保存配置
                    base_cfg.to_json_file(config_path)
                    self.logger.info("Base model config.json created")
                except Exception as e:
                    self.logger.warning(f"Failed to create base config.json: {e}")
            else:
                self.logger.info("config.json already exists (created by PEFT), skipping base config creation")
                
            self.logger.info(f"Saved LoRA adapter weights to {model_dir}")
        else:
            # 常规模型保存
            self.logger.info("Saving full model...")
            self.model.save_pretrained(model_dir)
            
            # 保存tokenizer
            self.tokenizer.save_pretrained(model_dir)
            
            # 为常规模型创建或更新配置
            base_model_name = self.pretrained_model_name
            base_cfg = AutoConfig.from_pretrained(base_model_name)
            # 注入NER相关字段
            base_cfg.id2label = self.id2label
            base_cfg.label2id = self.label2id
            base_cfg.num_labels = self.config['labels']['num_labels']
            # 可选的分类器dropout
            dropout_val = self.config['model'].get('dropout', None)
            if dropout_val is not None:
                setattr(base_cfg, 'classifier_dropout', dropout_val)
            # 保存配置
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
            self.write_hyper_parameters_to_tensorboard()

        # 训练轮次
        num_epochs = self.config['training']['epochs']
        self.logger.info(f"Total training epochs: {num_epochs}")
        
        # 训练循环
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # 执行训练轮次，返回训练损失, train_loss 是训练损失，是训练轮次中每个批次损失的平均值
            train_loss = self.train_epoch(epoch)
            
            # 验证训练结果，返回验证指标
            val_metrics = self.validate()
            
            # 更新训练历史结果
            self.update_training_metric(train_loss, val_metrics)

            # TensorBoard: 验证轮次指标
            self.write_training_metric_tensorboard_(epoch, val_metrics)
            
            # 检查最佳模型
            current_f1 = val_metrics.get('f1', 0) if val_metrics else 0
            is_best = current_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = current_f1
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self._save_checkpoint(epoch, val_metrics, is_best)
            
            # 记录轮次结果
            self._print_train_log(epoch, epoch_start_time, num_epochs, train_loss, val_metrics)

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
            except Exception as e:
                self.logger.error(f"Error flushing/closing TensorBoard writer: {e}")
                pass

    def _print_train_log(self, epoch: int, epoch_start_time: float, num_epochs, train_loss: float,
                         val_metrics: dict[str, float]):
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

    def write_training_metric_tensorboard_(self, epoch, val_metrics):
        if self.tb_writer is not None and val_metrics:
            self.tb_writer.add_scalar('val/loss', float(val_metrics.get('val_loss', 0.0)), epoch + 1)
            self.tb_writer.add_scalar('val/f1', float(val_metrics.get('f1', 0.0)), epoch + 1)
            self.tb_writer.add_scalar('val/precision', float(val_metrics.get('precision', 0.0)), epoch + 1)
            self.tb_writer.add_scalar('val/recall', float(val_metrics.get('recall', 0.0)), epoch + 1)
            self.tb_writer.add_scalar('val/token_f1', float(val_metrics.get('token_f1', 0.0)), epoch + 1)
            self.tb_writer.add_scalar('val/token_precision', float(val_metrics.get('token_precision', 0.0)), epoch + 1)
            self.tb_writer.add_scalar('val/token_recall', float(val_metrics.get('token_recall', 0.0)), epoch + 1)
            self.tb_writer.add_scalar('val/token_accuracy', float(val_metrics.get('token_accuracy', 0.0)), epoch + 1)

    def update_training_metric(self, train_loss, val_metrics):
        self.training_history['train_loss'].append(train_loss)
        if val_metrics:
            self.training_history['val_loss'].append(val_metrics.get('val_loss', 0))
            self.training_history['val_f1'].append(val_metrics.get('f1', 0))
            self.training_history['val_precision'].append(val_metrics.get('precision', 0))
            self.training_history['val_recall'].append(val_metrics.get('recall', 0))
        if self.scheduler:
            self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])

    def write_hyper_parameters_to_tensorboard(self):
        """记录超参数与配置摘要到 TensorBoard
        """
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
        except Exception as e:
            self.logger.warning(f"Error while logging hyperparameters to TensorBoard. {e}")
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
            trainer = NERTrainer(config, self.global_config, self.logger)
            
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