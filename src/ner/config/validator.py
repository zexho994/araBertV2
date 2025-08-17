"""NER 配置校验器（Configuration Validator）

职责：
- 对 NER 系统使用的配置进行全面校验，保证结构完整与字段一致性
- 输出错误与警告，帮助在训练前尽早发现问题

设计说明：
- 采用分区校验：country/model/training/data/labels/evaluation/output/hardware/logging
- 提供跨区校验 `_validate_cross_sections`，用于检查区间之间的一致性

# TODO: 支持 Schema 驱动的自动校验（如 pydantic），并汇总多处错误为结构化报告
# TODO: 允许配置“严格/宽松”模式（将部分错误降级为警告，便于快速试跑）
"""

import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

class ConfigValidator:
    """NER 配置文件与设置的校验器

    说明：
    - 存储校验期间的错误与警告列表，便于调用方打印或记录
    - `validate_config` 为主入口，内部调用各分区校验方法

    注意（与实现的其他模块一致性）：
    - 训练器当前仅实现 `optimizer=adamw` 与 `scheduler in {linear, cosine}`；
      本校验器的可选项更宽（如 `adam`/`sgd`/`rmsprop`、`polynomial`/`constant`）。
      
      # ERROR: 校验器允许的取值超出了训练器实际支持范围，会导致通过校验但训练报错。
      # TODO: 将有效取值与训练器实现对齐，或在训练器增加对应支持。
    - `hardware.device` 允许 `mps`，但训练器未显式支持；
      
      # TODO: 若要支持 MPS（Apple Silicon），训练器需增加相应分支与能力检测。
    - `hardware.dataloader_num_workers` 与训练器中的 `hardware.num_workers` 键名不一致；
      
      # ERROR: 配置键不一致将导致工作线程设置失效。建议统一为 `num_workers`。
    - `data.max_length` 与 `model.max_length`：当前 DataLoader 读取 `data.max_length`，
      校验器却在 `model` 分区校验 `max_length`。建议统一归属 `data` 分区。
    - `labels` 分区强制 `label_names` 与 `label_mapping`，而训练器允许通过 `entities` 自动派生 BIO 标签；
      
      # TODO: 统一标签定义来源（推荐固定为 `label_names`/`label_mapping`），避免重复口径。
    """
    
    # Valid model types
    VALID_MODEL_TYPES = ['bert', 'distilbert', 'roberta', 'albert']
    
    # Valid optimizers
    VALID_OPTIMIZERS = ['adam', 'adamw', 'sgd', 'rmsprop']
    
    # Valid schedulers
    VALID_SCHEDULERS = ['linear', 'cosine', 'polynomial', 'constant']
    
    # Valid log levels
    VALID_LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    # Valid evaluation metrics
    VALID_METRICS = ['precision', 'recall', 'f1', 'accuracy', 'entity_f1']
    
    def __init__(self):
        """初始化配置校验器"""
        self.errors = []
        self.warnings = []
    
    def validate_config(self, config: Dict[str, Any], country: str = None) -> bool:
        """校验完整配置
        
        Args:
            config: 待校验的配置字典
            country: 国家代码（可选，仅用于上下文提示）
            
        Returns:
            若配置有效返回 True，否则 False（错误信息在 `get_errors`）
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Validate each section
        self._validate_country_section(config.get('country', {}), country)
        self._validate_model_section(config.get('model', {}))
        self._validate_training_section(config.get('training', {}))
        self._validate_data_section(config.get('data', {}))
        self._validate_labels_section(config.get('labels', {}))
        self._validate_evaluation_section(config.get('evaluation', {}))
        self._validate_output_section(config.get('output', {}))
        self._validate_hardware_section(config.get('hardware', {}))
        self._validate_logging_section(config.get('logging', {}))
        # Cross-section validation
        self._validate_cross_sections(config)
        
        return len(self.errors) == 0
    
    def get_errors(self) -> List[str]:
        """获取校验错误列表"""
        return self.errors.copy()
    
    def get_warnings(self) -> List[str]:
        """获取校验警告列表"""
        return self.warnings.copy()
    
    def _validate_country_section(self, country_config: Dict[str, Any], country: str = None):
        """校验 country 分区"""
        required_fields = ['code', 'name']
        
        for field in required_fields:
            if field not in country_config:
                self.errors.append(f"Country section missing required field: {field}")
        
        if 'code' in country_config:
            code = country_config['code']
            # Allow lowercase letters, digits, underscore, hyphen; length 2-20
            if not isinstance(code, str) or not re.match(r'^[a-z0-9_-]{2,20}$', code):
                self.errors.append("Country code must be 2-20 characters: lowercase letters, digits, '_' or '-'")
            
            if country and code != country:
                self.warnings.append(f"Country code '{code}' doesn't match expected '{country}'")
        
        if 'name' in country_config:
            name = country_config['name']
            if not isinstance(name, str) or len(name.strip()) == 0:
                self.errors.append("Country name must be a non-empty string")
    
    def _validate_model_section(self, model_config: Dict[str, Any]):
        """校验 model 分区

        注意：
        - `VALID_MODEL_TYPES` 未包含诸如 `xlm-roberta`/`deberta` 等常见类型。
          # TODO: 结合实际模型类型扩展列表，或放宽此处限制，仅由模型加载过程兜底。
        - `max_length` 更适合放在 `data` 分区（DataLoader 消费），此处的校验可能与实际使用不一致。
          # TODO: 迁移为校验 `data.max_length`，保留向后兼容。
        """
        required_fields = ['name', 'type', 'pretrained_model']
        
        for field in required_fields:
            if field not in model_config:
                self.errors.append(f"Model section missing required field: {field}")
        
        if 'type' in model_config:
            model_type = model_config['type']
            if model_type not in self.VALID_MODEL_TYPES:
                self.errors.append(
                    f"Invalid model type '{model_type}'. Valid types: {self.VALID_MODEL_TYPES}"
                )
        
        if 'pretrained_model' in model_config:
            pretrained = model_config['pretrained_model']
            if not isinstance(pretrained, str) or len(pretrained.strip()) == 0:
                self.errors.append("Pretrained model must be a non-empty string")
        
        # Validate optional fields
        if 'max_length' in model_config:
            max_length = model_config['max_length']
            if not isinstance(max_length, int) or max_length <= 0 or max_length > 2048:
                self.errors.append("Model max_length must be a positive integer <= 2048")
        
        if 'dropout' in model_config:
            dropout = model_config['dropout']
            if not isinstance(dropout, (int, float)) or dropout < 0 or dropout > 1:
                self.errors.append("Model dropout must be a number between 0 and 1")
    
    def _validate_training_section(self, training_config: Dict[str, Any]):
        """校验 training 分区

        # ERROR: `optimizer` 与 `scheduler` 的可选值需与训练器实现对齐。
        #   - 训练器仅支持 optimizer=adamw
        #   - 训练器仅支持 scheduler in {linear, cosine}
        # TODO: 若需保留更广泛的校验范围，训练器侧需要补齐支持，否则应在此处收紧可选值。
        """
        required_fields = ['epochs', 'batch_size', 'learning_rate']
        
        for field in required_fields:
            if field not in training_config:
                self.errors.append(f"Training section missing required field: {field}")
        
        if 'epochs' in training_config:
            epochs = training_config['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                self.errors.append("Training epochs must be a positive integer")
        
        if 'batch_size' in training_config:
            batch_size = training_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                self.errors.append("Training batch_size must be a positive integer")
        
        if 'learning_rate' in training_config:
            lr = training_config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.errors.append("Training learning_rate must be a positive number")
        
        # Validate optional fields
        if 'optimizer' in training_config:
            optimizer = training_config['optimizer']
            if optimizer not in self.VALID_OPTIMIZERS:
                self.errors.append(
                    f"Invalid optimizer '{optimizer}'. Valid optimizers: {self.VALID_OPTIMIZERS}"
                )
        
        if 'scheduler' in training_config:
            scheduler = training_config['scheduler']
            if scheduler not in self.VALID_SCHEDULERS:
                self.errors.append(
                    f"Invalid scheduler '{scheduler}'. Valid schedulers: {self.VALID_SCHEDULERS}"
                )
        
        if 'warmup_steps' in training_config:
            warmup = training_config['warmup_steps']
            if not isinstance(warmup, int) or warmup < 0:
                self.errors.append("Training warmup_steps must be a non-negative integer")
        
        if 'weight_decay' in training_config:
            weight_decay = training_config['weight_decay']
            if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
                self.errors.append("Training weight_decay must be a non-negative number")
    
    def _validate_data_section(self, data_config: Dict[str, Any]):
        """校验 data 分区

        注意：
        - 训练器允许 `val_file` 缺省（将跳过验证）；此处强制要求 `val_file` 会提高门槛。
          # TODO: 将 `val_file` 从必填改为可选，缺省时给出警告而非错误。
        - `max_length` 应位于 data 分区；可在此补充合法性校验。
        """
        required_fields = ['train_file', 'val_file']
        
        for field in required_fields:
            if field not in data_config:
                self.errors.append(f"Data section missing required field: {field}")
        
        # Validate file paths (basic validation)
        for file_field in ['train_file', 'val_file', 'test_file']:
            if file_field in data_config:
                file_path = data_config[file_field]
                if not isinstance(file_path, str) or len(file_path.strip()) == 0:
                    self.errors.append(f"Data {file_field} must be a non-empty string")
        
        # Validate preprocessing options
        if 'preprocessing' in data_config:
            preprocessing = data_config['preprocessing']
            if not isinstance(preprocessing, dict):
                self.errors.append("Data preprocessing must be a dictionary")
            else:
                if 'lowercase' in preprocessing:
                    if not isinstance(preprocessing['lowercase'], bool):
                        self.errors.append("Preprocessing lowercase must be a boolean")
                
                if 'remove_diacritics' in preprocessing:
                    if not isinstance(preprocessing['remove_diacritics'], bool):
                        self.errors.append("Preprocessing remove_diacritics must be a boolean")
    
    def _validate_labels_section(self, labels_config: Dict[str, Any]):
        """校验 labels 分区

        说明：
        - 强制要求 `num_labels`/`label_names`/`label_mapping`，保证训练与导出一致性。
        - 若要支持仅给定 `entities` 的最小化配置，应在上游模板生成时补齐所需字段。
        """
        required_fields = ['num_labels', 'label_names', 'label_mapping']
        
        for field in required_fields:
            if field not in labels_config:
                self.errors.append(f"Labels section missing required field: {field}")
                return
        
        num_labels = labels_config['num_labels']
        label_names = labels_config['label_names']
        label_mapping = labels_config['label_mapping']
        
        # Validate num_labels
        if not isinstance(num_labels, int) or num_labels <= 0:
            self.errors.append("Labels num_labels must be a positive integer")
            return
        
        # Validate label_names
        if not isinstance(label_names, list):
            self.errors.append("Labels label_names must be a list")
            return
        
        if len(label_names) != num_labels:
            self.errors.append(
                f"Number of label names ({len(label_names)}) doesn't match num_labels ({num_labels})"
            )
        
        # Check for duplicate labels
        if len(set(label_names)) != len(label_names):
            self.errors.append("Label names contain duplicates")
        
        # Validate label_mapping
        if not isinstance(label_mapping, dict):
            self.errors.append("Labels label_mapping must be a dictionary")
            return
        
        if len(label_mapping) != num_labels:
            self.errors.append(
                f"Number of label mappings ({len(label_mapping)}) doesn't match num_labels ({num_labels})"
            )
        
        # Validate mapping consistency
        for label_name in label_names:
            if label_name not in label_mapping:
                self.errors.append(f"Label '{label_name}' missing from label_mapping")
        
        # Validate mapping values are unique integers
        mapping_values = list(label_mapping.values())
        if not all(isinstance(v, int) for v in mapping_values):
            self.errors.append("All label mapping values must be integers")
        
        if len(set(mapping_values)) != len(mapping_values):
            self.errors.append("Label mapping values contain duplicates")
        
        # Check if mapping values are in valid range
        expected_values = set(range(num_labels))
        actual_values = set(mapping_values)
        if actual_values != expected_values:
            self.errors.append(
                f"Label mapping values must be exactly {{0, 1, ..., {num_labels-1}}}"
            )
    
    def _validate_evaluation_section(self, eval_config: Dict[str, Any]):
        """校验 evaluation 分区

        # TODO: 支持更多指标（如 per-entity 选择、micro/macro F1、置信度相关指标）。
        """
        if 'metrics' in eval_config:
            metrics = eval_config['metrics']
            if not isinstance(metrics, list):
                self.errors.append("Evaluation metrics must be a list")
            else:
                for metric in metrics:
                    if metric not in self.VALID_METRICS:
                        self.errors.append(
                            f"Invalid metric '{metric}'. Valid metrics: {self.VALID_METRICS}"
                        )
        
        if 'save_predictions' in eval_config:
            if not isinstance(eval_config['save_predictions'], bool):
                self.errors.append("Evaluation save_predictions must be a boolean")
    
    def _validate_output_section(self, output_config: Dict[str, Any]):
        """校验 output 分区

        # TODO: 校验 `results_dir`/`logs_dir` 等字段（若存在），并检测路径可写性（可选）。
        """
        if 'model_dir' in output_config:
            model_dir = output_config['model_dir']
            if not isinstance(model_dir, str) or len(model_dir.strip()) == 0:
                self.errors.append("Output model_dir must be a non-empty string")
        
        if 'save_steps' in output_config:
            save_steps = output_config['save_steps']
            if not isinstance(save_steps, int) or save_steps <= 0:
                self.errors.append("Output save_steps must be a positive integer")
    
    def _validate_hardware_section(self, hardware_config: Dict[str, Any]):
        """校验 hardware 分区

        注意：
        - 训练器当前不支持 `mps`；若检测到 `mps`，应提示潜在不兼容。
        - 键名不一致问题：此处使用 `dataloader_num_workers`，训练器读取 `num_workers`。
          
          # ERROR: 键名不一致导致并行度配置失效。建议统一键为 `num_workers`。
        """
        if 'device' in hardware_config:
            device = hardware_config['device']
            if device not in ['auto', 'cpu', 'cuda', 'mps']:
                self.errors.append("Hardware device must be 'auto', 'cpu', 'cuda', or 'mps'")
        
        if 'mixed_precision' in hardware_config:
            if not isinstance(hardware_config['mixed_precision'], bool):
                self.errors.append("Hardware mixed_precision must be a boolean")
        
        if 'dataloader_num_workers' in hardware_config:
            workers = hardware_config['dataloader_num_workers']
            if not isinstance(workers, int) or workers < 0:
                self.errors.append("Hardware dataloader_num_workers must be a non-negative integer")
    
    def _validate_logging_section(self, logging_config: Dict[str, Any]):
        """校验 logging 分区

        # TODO: 校验 log_file 的目录是否存在或可创建；校验与 `output.logs_dir` 的一致性。
        """
        if 'level' in logging_config:
            level = logging_config['level']
            if level not in self.VALID_LOG_LEVELS:
                self.errors.append(
                    f"Invalid log level '{level}'. Valid levels: {self.VALID_LOG_LEVELS}"
                )
        
        if 'log_file' in logging_config:
            log_file = logging_config['log_file']
            if not isinstance(log_file, str) or len(log_file.strip()) == 0:
                self.errors.append("Logging log_file must be a non-empty string")
        
        # Validate wandb config
        if 'wandb' in logging_config:
            wandb_config = logging_config['wandb']
            if not isinstance(wandb_config, dict):
                self.errors.append("Logging wandb must be a dictionary")
            else:
                if 'enabled' in wandb_config:
                    if not isinstance(wandb_config['enabled'], bool):
                        self.errors.append("Logging wandb enabled must be a boolean")
    
    def _validate_cross_sections(self, config: Dict[str, Any]):
        """跨分区一致性校验

        - 对内存占用的粗略估计，提示可能的风险
        - 检查 `output.model_dir` 与 `logging.log_file` 是否在相近目录

        # TODO: 补充更多约束：如标签与模型 `num_labels` 一致、数据文件路径存在性、
        #       warmup_ratio/steps 之间的互斥或联动关系等。
        """
        # Check if model max_length is compatible with training batch_size
        if 'model' in config and 'training' in config:
            model_config = config['model']
            training_config = config['training']
            
            if 'max_length' in model_config and 'batch_size' in training_config:
                max_length = model_config['max_length']
                batch_size = training_config['batch_size']
                
                # Warn if memory usage might be high
                estimated_memory = max_length * batch_size
                if estimated_memory > 100000:  # Arbitrary threshold
                    self.warnings.append(
                        f"High memory usage expected: max_length ({max_length}) * "
                        f"batch_size ({batch_size}) = {estimated_memory}"
                    )
        
        # Check if output directories are consistent
        if 'output' in config and 'logging' in config:
            output_config = config['output']
            logging_config = config['logging']
            
            if 'model_dir' in output_config and 'log_file' in logging_config:
                model_dir = Path(output_config['model_dir'])
                log_file = Path(logging_config['log_file'])
                
                # Warn if they're in completely different locations
                if not str(log_file).startswith(str(model_dir.parent)):
                    self.warnings.append(
                        "Model directory and log file are in different locations"
                    )