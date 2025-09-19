"""训练 NER 模型的命令

流程：
1) 加载配置（可选使用自定义配置文件）
2) 应用命令行覆盖项（epochs/batch-size/learning-rate/output-dir 等）
3) 校验配置
4) 初始化并启动训练（可选从 checkpoint 恢复）

注意：
- 覆盖项仅在用户显式传入时生效，避免默认值覆盖配置。
- 支持外层 CLI 注入 `data_path`/`val_data_path` 用于快速调试。

# TODO: 支持 `test_data_path` 以及自动识别数据格式（JSONL）。
"""

import json

from .base import BaseCommand
from ..config import ConfigValidator

class TrainCommand(BaseCommand):
    """训练 NER 模型的命令

    流程：
    1) 加载配置（可选使用自定义配置文件）
    2) 应用命令行覆盖项（epochs/batch-size/learning-rate/output-dir 等）
    3) 校验配置
    4) 初始化并启动训练（可选从 checkpoint 恢复）

    注意：
    - 覆盖项仅在用户显式传入时生效，避免默认值覆盖配置。
    - 支持外层 CLI 注入 `data_path`/`val_data_path` 用于快速调试。

    # TODO: 支持 `test_data_path` 以及自动识别数据格式（JSONL）。
    """
    
    @property
    def name(self) -> str:
        return "train"
    
    @property
    def description(self) -> str:
        return "Train NER model for a specific country"
    
    def setup_parser(self, parser):
        parser.add_argument(
            "--country",
            required=True,
            help="Country code for training (e.g., 'uae', 'saudi')"
        )
        parser.add_argument(
            "--config",
            help="Path to custom configuration file (optional)"
        )
        parser.add_argument(
            "--epochs",
            type=int,
            help="Number of training epochs (overrides config)"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Training batch size (overrides config)"
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            help="Learning rate (overrides config)"
        )
        parser.add_argument(
            "--output-dir",
            help="Output directory for model (overrides config)"
        )
        parser.add_argument(
            "--model",
            help="Pretrained model name or path (overrides config)"
        )
        parser.add_argument(
            "--resume",
            help="Resume training from checkpoint"
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate configuration without training"
        )
        # LoRA相关参数
        parser.add_argument(
            "--lora",
            action="store_true",
            help="Enable LoRA training mode"
        )
        parser.add_argument(
            "--lora-r",
            type=int,
            help="LoRA attention dimension (r)"
        )
        parser.add_argument(
            "--lora-alpha",
            type=int,
            help="LoRA alpha parameter"
        )
        parser.add_argument(
            "--lora-dropout",
            type=float,
            help="LoRA dropout rate"
        )
        parser.add_argument(
            "--lora-target-modules",
            help="Comma-separated list of target modules for LoRA"
        )
        parser.add_argument(
            "--lora-base-adapter",
            help="Path to base LoRA adapter for incremental training"
        )
    
    def execute(self, args) -> bool:
        try:
            # 优先使用命令行参数指定的配置文件
            config_arg = getattr(args, 'config', None)
            if config_arg:
                with open(config_arg, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = self.get_country_config(args.country)
            
            # 覆盖配置
            if getattr(args, 'epochs', None):
                config['training']['epochs'] = args.epochs
            if getattr(args, 'batch_size', None):
                config['training']['batch_size'] = args.batch_size
            if getattr(args, 'learning_rate', None):
                config['training']['learning_rate'] = args.learning_rate
            if getattr(args, 'output_dir', None):
                config['output']['model_dir'] = args.output_dir
            if getattr(args, 'model', None):
                config['model']['pretrained_model_name'] = args.model
                self.logger.info(f"Using pretrained model: {args.model}")
                
            # 处理LoRA参数
            if getattr(args, 'lora', False):
                # 确保training配置中有lora部分
                if 'lora' not in config['training']:
                    config['training']['lora'] = {}
                # 启用LoRA
                config['training']['lora']['enabled'] = True
                
                # 设置LoRA参数
                if getattr(args, 'lora_r', None):
                    config['training']['lora']['r'] = args.lora_r
                if getattr(args, 'lora_alpha', None):
                    config['training']['lora']['alpha'] = args.lora_alpha
                if getattr(args, 'lora_dropout', None):
                    config['training']['lora']['dropout'] = args.lora_dropout
                if getattr(args, 'lora_target_modules', None):
                    # 将逗号分隔的字符串转换为列表
                    target_modules = [m.strip() for m in args.lora_target_modules.split(',')]
                    config['training']['lora']['target_modules'] = target_modules
                # 设置LoRA增量训练参数
                if getattr(args, 'lora_base_adapter', None):
                    config['training']['lora']['train_base_adapter'] = args.lora_base_adapter
                    self.logger.info(f"LoRA incremental training enabled with base adapter: {args.lora_base_adapter}")
                self.logger.info(f"LoRA training enabled with parameters: {config['training']['lora']}")
                
            # 校验配置
            validator = ConfigValidator()
            if not validator.validate_config(config, args.country):
                self.logger.error("Configuration validation failed:")
                for error in validator.get_errors():
                    self.logger.error(f"  ERROR: {error}")
                for warning in validator.get_warnings():
                    self.logger.info(f"  WARNING: {warning}")
                return False
            
            # 快速校验
            if getattr(args, 'dry_run', False):
                self.logger.info("Configuration validation passed. Dry run completed.")
                return True
            
            # 导入训练器
            from ..training import NERTrainer
            
            # 初始化训练器
            trainer = NERTrainer(config, self.global_config, logger=self.logger)
            
            # 恢复训练
            if args.resume:
                trainer.resume_from_checkpoint(args.resume)
            
            # 启动训练
            trainer.train()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False
