"""NER 命令行（CLI）指令集合

提供 NER 系统的常用指令，包括：
- 训练（train）
- 评估（evaluate）
- 预测（predict）
- 配置管理（config）
- 数据处理（data）
- 状态查看（status）

设计说明：
- 通过 `BaseCommand` 统一约束命令的名称与描述、参数解析与执行。
- 全局配置 `global_config` 由外层主程序注入，通常包含：
  - `config_dir`: 配置目录
  - `model_dir`: 模型目录
  - `log_dir`: 日志目录

# TODO: 为 `global_config` 定义强类型（TypedDict/dataclass），并在入口层进行完整校验。
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

from ..config import ConfigManager, ConfigValidator

class BaseCommand(ABC):
    """所有 NER CLI 子命令的抽象基类

    职责：
    - 提供统一的命令名称（`name`）与描述（`description`）属性
    - 定义参数解析接口 `setup_parser` 与执行接口 `execute`
    - 保存外层注入的 `global_config`

    # TODO: 支持注入统一的 logger，并在各子命令中复用。
    """
    
    def __init__(self):
        self.global_config = {}
        self.logger = None
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Command name"""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Command description"""
        pass
    
    @abstractmethod
    def setup_parser(self, parser):
        """Setup command-specific arguments"""
        pass
    
    @abstractmethod
    def execute(self, args) -> bool:
        """Execute the command"""
        pass
    
    def set_global_config(self, config: Dict[str, Any]):
        """Set global configuration"""
        self.global_config = config
    
    def set_logger(self, logger):
        """Inject a shared logger instance"""
        self.logger = logger
    
    def get_help(self) -> str:
        """Get help text for this command"""
        return self.description

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

    # TODO: 支持 `test_data_path` 以及自动识别数据格式（JSON/JSONL）。
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
            "--resume",
            help="Resume training from checkpoint"
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Validate configuration without training"
        )
    
    def execute(self, args) -> bool:
        try:
            # 加载配置
            config_manager = ConfigManager(self.global_config.get('config_dir'))

            # 优先使用命令行参数指定的配置文件
            config_arg = getattr(args, 'config', None)
            if config_arg:
                with open(config_arg, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                config = config_manager.load_country_config(args.country)
            
            # 覆盖配置
            if getattr(args, 'epochs', None):
                config['training']['epochs'] = args.epochs
            if getattr(args, 'batch_size', None):
                config['training']['batch_size'] = args.batch_size
            if getattr(args, 'learning_rate', None):
                config['training']['learning_rate'] = args.learning_rate
            if getattr(args, 'output_dir', None):
                config['output']['model_dir'] = args.output_dir
                
            # 校验配置
            validator = ConfigValidator()
            if not validator.validate_config(config, args.country):
                print("Configuration validation failed:")
                for error in validator.get_errors():
                    print(f"  ERROR: {error}")
                for warning in validator.get_warnings():
                    print(f"  WARNING: {warning}")
                return False
            
            # 快速校验
            if getattr(args, 'dry_run', False):
                print("Configuration validation passed. Dry run completed.")
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
            print(f"Training failed: {e}")
            return False

class EvaluateCommand(BaseCommand):
    """评估 NER 模型的命令

    说明：
    - 当前实现默认从模型目录读取 tokenizer 与标签映射，从而进行文本级评估。
    - 支持 `--output-dir` 持久化评估指标 JSON；可选 `--detailed-report` 生成详细报告（未实现）。

    # TODO: 支持基于 DataLoader 的批量评估，并尊重 `--batch-size`。
    # TODO: 将 `--metrics` 与 `--detailed-report` 真正接入评估与报告逻辑（当前未使用）。
    """
    
    @property
    def name(self) -> str:
        return "evaluate"
    
    @property
    def description(self) -> str:
        return "Evaluate trained NER models"
    
    def setup_parser(self, parser):
        parser.add_argument(
            "--model-path", "-m",
            type=str,
            required=True,
            help="Path to trained model"
        )
        parser.add_argument(
            "--data-path", "-d",
            type=str,
            required=True,
            help="Path to evaluation data file"
        )
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            help="Output directory for evaluation results"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            help="Evaluation batch size"
        )
        parser.add_argument(
            "--country",
            help="Country code for configuration"
        )
        parser.add_argument(
            "--metrics",
            nargs="+",
            default=["precision", "recall", "f1", "accuracy"],
            help="Metrics to compute"
        )
        parser.add_argument(
            "--compare-with",
            help="Compare with another model"
        )
        parser.add_argument(
            "--detailed-report",
            action="store_true",
            help="Generate detailed evaluation report"
        )
    
    def execute(self, args) -> bool:
        try:
            # Import evaluator
            import torch
            from ..evaluation import NEREvaluator
            from ..models import NERModelManager
            from ..models.wrapper import TransformersNERModelWrapper
            from pathlib import Path
            
            # Load model
            model_manager = NERModelManager(self.global_config.get('model_dir'), logger=self.logger)
            model = model_manager.load_model(args.model_path)
            
            # Load tokenizer separately
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            
            # Get label list from model config first
            if hasattr(model, 'config') and hasattr(model.config, 'id2label'):
                id2label = model.config.id2label
                label_list = list(id2label.values())
            else:
                raise ValueError("Model configuration does not contain label mappings.")
            
            # Check if model has predict method, if not, wrap it
            if not hasattr(model, 'predict'):
                print("Model doesn't have predict method, wrapping with TransformersNERModelWrapper...")
                # Get label mappings
                if hasattr(model, 'config') and hasattr(model.config, 'label2id'):
                    label2id = model.config.label2id
                else:
                    label2id = {label: i for i, label in id2label.items()}
                
                # Wrap the model
                model = TransformersNERModelWrapper(model, tokenizer, id2label, label2id)
                print("Model successfully wrapped.")
            
            # Load configuration if country specified; otherwise try from model's training metadata
            config = None
            if hasattr(args, 'country') and args.country:
                config_manager = ConfigManager(self.global_config.get('config_dir'))
                config = config_manager.load_country_config(args.country)
            
            # Initialize evaluator with eval-specific logger under output.logs_dir
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)  # Move model to device
            eval_logs_dir = None
            if config and 'output' in config and 'logs_dir' in config['output']:
                eval_logs_dir = config['output']['logs_dir']
            else:
                eval_logs_dir = self.global_config.get('log_dir', 'data/ner/logs')
            from ..utils import NERLogger
            eval_logger = NERLogger(
                name=f"{(config or {}).get('country', {}).get('code', 'unknown')}_eval",
                log_dir=eval_logs_dir
            )
            evaluator = NEREvaluator(model, tokenizer, label_list, device, logger=eval_logger)
            
            import json
            texts = []
            true_labels = []
            
            with open(args.data_path, 'r', encoding='utf-8') as f:
                try:
                    data_list = json.load(f)
                    for data in data_list:
                        # Use pre-tokenized tokens if available, otherwise split text
                        if 'tokens' in data:
                            text = ' '.join(data['tokens'])  # Reconstruct text from tokens
                        else:
                            text = data.get('text', '')
                        texts.append(text)
                        true_labels.append(data.get('labels', []))
                except json.JSONDecodeError:
                    # If that fails, try JSONL format
                    f.seek(0)
                    for line in f:
                        line = line.strip()
                        if line:
                            data = json.loads(line)
                            # Use pre-tokenized tokens if available, otherwise split text
                            if 'tokens' in data:
                                text = ' '.join(data['tokens'])  # Reconstruct text from tokens
                            else:
                                text = data.get('text', '')
                            texts.append(text)
                            true_labels.append(data.get('labels', []))
            
            # Run evaluation with configurable confidence threshold
            confidence_threshold = getattr(args, 'confidence_threshold', 0.1)  # Lower default threshold
            results = evaluator.evaluate_text(
                texts=texts,
                true_labels=true_labels,
                confidence_threshold=confidence_threshold
            )
            
            # Print results
            print("Evaluation Results:")
            print("\nToken-level Metrics:")
            for metric, value in results.get('token_metrics', {}).items():
                print(f"  {metric}: {value:.4f}")
            
            print("\nEntity-level Metrics:")
            for metric, value in results.get('entity_metrics', {}).items():
                print(f"  {metric}: {value:.4f}")
            
            if 'per_entity_metrics' in results:
                print("\nPer-Entity Metrics:")
                for entity, metrics in results['per_entity_metrics'].items():
                    print(f"  {entity}:")
                    for metric, value in metrics.items():
                        print(f"    {metric}: {value:.4f}")
            
            print(f"\nTotal samples evaluated: {results.get('num_samples', 0)}")

            # Persist results if output directory is provided or available via config
            out_dir = None
            if getattr(args, 'output_dir', None):
                out_dir = Path(args.output_dir)
            elif config and 'output' in config and 'results_dir' in config['output']:
                out_dir = Path(config['output']['results_dir'])
            
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = out_dir / 'metrics.json'
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\nSaved evaluation metrics to: {metrics_path}")
            
            # Compare with another model if specified
            if hasattr(args, 'compare_with') and args.compare_with:
                # ERROR: 下段代码参数不匹配 `NEREvaluator` 的构造与 `evaluate` 的签名，无法按预期工作。
                # TODO: 若需支持模型对比，应：
                #   1) 同样加载 `other_tokenizer` 与 `other_label_list` 构造 `other_evaluator`
                #   2) 复用同一 `texts/true_labels` 调用 `evaluate_text`，再对比关键指标。
                other_model = model_manager.load_model(args.compare_with)
                # other_evaluator = NEREvaluator(other_model, other_tokenizer, other_label_list, device)
                # other_results = other_evaluator.evaluate_text(texts, true_labels, confidence_threshold)
                other_results = {}
                
                print(f"\nComparison with {args.compare_with}:")
                for metric in args.metrics:
                    if metric in results and metric in other_results:
                        diff = results[metric] - other_results[metric]
                        print(f"  {metric}: {diff:+.4f}")
            
            return True
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return False

class PredictCommand(BaseCommand):
    """使用已训练模型进行预测的命令

    支持：
    - 单条文本预测（--text）
    - 文件批量预测（--file，每行一条文本）
    - 多种输出格式（json/text/conll）

    # TODO: 支持输入 JSON/JSONL（含 tokens/labels）并保持结构化输出。
    """
    
    @property
    def name(self) -> str:
        return "predict"
    
    @property
    def description(self) -> str:
        return "Use trained model for prediction"
    
    def setup_parser(self, parser):
        parser.add_argument(
            "--model-path",
            required=True,
            help="Model name or path for prediction"
        )
        parser.add_argument(
            "--text",
            help="Text to predict (single prediction)"
        )
        parser.add_argument(
            "--file",
            help="File containing texts to predict"
        )
        parser.add_argument(
            "--output-file",
            help="File to save predictions"
        )
        parser.add_argument(
            "--country",
            help="Country code for configuration"
        )
        parser.add_argument(
            "--output-format",
            choices=["json", "text", "conll"],
            default="json",
            help="Output format"
        )
        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.5,
            help="Confidence threshold for predictions"
        )
    
    def execute(self, args) -> bool:
        try:
            # Import required modules
            from ..models import NERModelManager
            
            # Load model
            model_manager = NERModelManager(self.global_config.get('model_dir'), logger=self.logger)
            model = model_manager.load_model(args.model_path)
            
            # Load tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)
            
            # Load configuration if country specified
            config = None
            if hasattr(args, 'country') and args.country:
                config_manager = ConfigManager(self.global_config.get('config_dir'))
                config = config_manager.load_country_config(args.country)
            
            # Single text prediction
            if args.text:
                prediction = model.predict(args.text, tokenizer=tokenizer, confidence_threshold=args.confidence_threshold)
                
                if args.output_format == "json":
                    print(json.dumps(prediction, indent=2, ensure_ascii=False))
                elif args.output_format == "text":
                    for token, label in zip(prediction['tokens'], prediction['labels']):
                        print(f"{token}\t{label}")
                elif args.output_format == "conll":
                    for token, label in zip(prediction['tokens'], prediction['labels']):
                        print(f"{token} {label}")
                
                return True
            
            # Batch prediction from file
            if args.file:
                predictions = []
                
                with open(args.file, 'r', encoding='utf-8') as f:
                    for line in f:
                        text = line.strip()
                        if text:
                            prediction = model.predict(text, tokenizer=tokenizer, confidence_threshold=args.confidence_threshold)
                            predictions.append(prediction)
                
                # Save or print predictions
                if args.output_file:
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        if args.output_format == "json":
                            json.dump(predictions, f, indent=2, ensure_ascii=False)
                        else:
                            for pred in predictions:
                                if args.output_format == "text":
                                    for token, label in zip(pred['tokens'], pred['labels']):
                                        f.write(f"{token}\t{label}\n")
                                    f.write("\n")
                                elif args.output_format == "conll":
                                    for token, label in zip(pred['tokens'], pred['labels']):
                                        f.write(f"{token} {label}\n")
                                    f.write("\n")
                else:
                    for pred in predictions:
                        print(json.dumps(pred, indent=2, ensure_ascii=False))
                        print("---")
                
                return True
            
            print("Please provide either --text or --file")
            return False
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            return False

class ConfigCommand(BaseCommand):
    """管理国家配置的命令

    支持：
    - 列表/展示/创建/校验/删除 配置

    # TODO: `create` 时支持指定输出根目录（覆盖模板中的 output.*），并提示创建的目录与样例文件。
    """
    
    @property
    def name(self) -> str:
        return "config"
    
    @property
    def description(self) -> str:
        return "Manage country configurations"
    
    def setup_parser(self, parser):
        subparsers = parser.add_subparsers(dest="config_action", help="Configuration actions")
        
        # List configurations
        list_parser = subparsers.add_parser("list", help="List available configurations")
        list_parser.add_argument("--templates", action="store_true", help="List templates instead")
        
        # Show configuration
        show_parser = subparsers.add_parser("show", help="Show configuration details")
        show_parser.add_argument("country", help="Country code to show")
        
        # Create configuration
        create_parser = subparsers.add_parser("create", help="Create new configuration")
        create_parser.add_argument("--country", required=True, help="Country code")
        create_parser.add_argument("--template", default="default", help="Template to use")
        create_parser.add_argument("--external-template", help="Path to external template file (overrides --template)")
        
        # Validate configuration
        validate_parser = subparsers.add_parser("validate", help="Validate configuration")
        validate_parser.add_argument("country", help="Country code to validate")
        
        # Delete configuration
        delete_parser = subparsers.add_parser("delete", help="Delete configuration")
        delete_parser.add_argument("country", help="Country code to delete")
        delete_parser.add_argument("--force", action="store_true", help="Force deletion without confirmation")
    
    def execute(self, args) -> bool:
        try:
            config_manager = ConfigManager(self.global_config.get('config_dir'))
            
            if args.config_action == "list":
                if args.templates:
                    templates = config_manager.list_templates()
                    print("Available templates:")
                    for template in templates:
                        print(f"  {template}")
                else:
                    countries = config_manager.list_countries()
                    print("Available country configurations:")
                    for country in countries:
                        print(f"  {country}")
                
            elif args.config_action == "show":
                config = config_manager.load_country_config(args.country)
                print(f"Configuration for {args.country}:")
                print(json.dumps(config, indent=2, ensure_ascii=False))
                
            elif args.config_action == "create":
                if config_manager.country_exists(args.country):
                    print(f"Configuration for '{args.country}' already exists")
                    return False
                
                if hasattr(args, 'external_template') and args.external_template:
                    config = config_manager.create_country_config(
                        args.country, 
                        args.template, 
                        external_template_path=args.external_template
                    )
                    print(f"Created configuration for '{args.country}' using external template '{args.external_template}'")
                else:
                    config = config_manager.create_country_config(args.country, args.template)
                    print(f"Created configuration for '{args.country}' using template '{args.template}'")

                
            elif args.config_action == "validate":
                config = config_manager.load_country_config(args.country)
                validator = ConfigValidator()
                
                if validator.validate_config(config, args.country):
                    print(f"Configuration for '{args.country}' is valid")
                    warnings = validator.get_warnings()
                    if warnings:
                        print("Warnings:")
                        for warning in warnings:
                            print(f"  WARNING: {warning}")
                else:
                    print(f"Configuration for '{args.country}' is invalid")
                    for error in validator.get_errors():
                        print(f"  ERROR: {error}")
                    return False
                
            elif args.config_action == "delete":
                if not config_manager.country_exists(args.country):
                    print(f"Configuration for '{args.country}' does not exist")
                    return False
                
                if not args.force:
                    response = input(f"Are you sure you want to delete configuration for '{args.country}'? (y/N): ")
                    if response.lower() != 'y':
                        print("Deletion cancelled")
                        return True
                
                config_manager.delete_country_config(args.country)
                print(f"Deleted configuration for '{args.country}'")
            
            else:
                print("Please specify a configuration action")
                return False
            
            return True
            
        except Exception as e:
            print(f"Configuration operation failed: {e}")
            return False

class DataCommand(BaseCommand):
    """数据处理相关命令

    支持：
    - 数据校验（validate）
    - 数据预处理（process）
    - 数据划分（split，暂未实现）

    # TODO: 实现 `split`，并支持自定义随机种子与分层抽样。
    """
    
    @property
    def name(self) -> str:
        return "data"
    
    @property
    def description(self) -> str:
        return "Process and validate training data"
    
    def setup_parser(self, parser):
        subparsers = parser.add_subparsers(dest="data_action", help="Data actions")
        
        # Validate data
        validate_parser = subparsers.add_parser("validate", help="Validate training data")
        validate_parser.add_argument("--country", required=True, help="Country code")
        validate_parser.add_argument("--input-file", required=True, help="Data file to validate")
        
        # Process data
        process_parser = subparsers.add_parser("process", help="Process and prepare data")
        process_parser.add_argument("--country", required=True, help="Country code")
        process_parser.add_argument("--input-file", required=True, help="Input data file")
        process_parser.add_argument("--output-file", required=True, help="Output processed file")
        
        # Split data
        split_parser = subparsers.add_parser("split", help="Split data into train/val/test")
        split_parser.add_argument("--input-file", required=True, help="Input data file")
        split_parser.add_argument("--train-ratio", type=float, default=0.8, help="Training data ratio")
        split_parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation data ratio")
        split_parser.add_argument("--test-ratio", type=float, default=0.1, help="Test data ratio")
        split_parser.add_argument("--output-dir", required=True, help="Output directory")
    
    def execute(self, args) -> bool:
        try:
            from ..data import NERDataProcessor
            
            if args.data_action == "validate":
                config_manager = ConfigManager(self.global_config.get('config_dir'))
                config = config_manager.load_country_config(args.country)
                
                processor = NERDataProcessor(config, logger=self.logger)
                is_valid = processor.validate_data_file(args.input_file)
                
                if is_valid:
                    print(f"Data file '{args.input_file}' is valid")
                else:
                    print(f"Data file '{args.input_file}' has validation errors")
                    return False
                
            elif args.data_action == "process":
                config_manager = ConfigManager(self.global_config.get('config_dir'))
                config = config_manager.load_country_config(args.country)
                
                processor = NERDataProcessor(config, logger=self.logger)
                processor.process_file(args.input_file, args.output_file)
                print(f"Processed data saved to '{args.output_file}'")
                
            elif args.data_action == "split":
                raise NotImplementedError("Data split not implemented")
                # # Validate ratios
                # total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
                # if abs(total_ratio - 1.0) > 0.001:
                #     print(f"Ratios must sum to 1.0, got {total_ratio}")
                #     return False
                
                # processor = NERDataProcessor()
                # processor.split_data(
                #     args.input_file,
                #     args.output_dir,
                #     train_ratio=args.train_ratio,
                #     val_ratio=args.val_ratio,
                #     test_ratio=args.test_ratio
                # )
                # print(f"Data split completed. Files saved to '{args.output_dir}'")
            
            else:
                print("Please specify a data action")
                return False
            
            return True
            
        except Exception as e:
            print(f"Data operation failed: {e}")
            return False

class ModelCommand(BaseCommand):
    """模型管理相关命令

    支持：
    - 列出/查询/删除 模型

    # TODO: 支持导出（export）与转换（onnx、safetensors 等），并完善信息展示。
    """
    
    @property
    def name(self) -> str:
        return "model"
    
    @property
    def description(self) -> str:
        return "Manage model operations"
    
    def setup_parser(self, parser):
        subparsers = parser.add_subparsers(dest="model_action", help="Model actions")
        
        # List models
        list_parser = subparsers.add_parser("list", help="List available models")
        list_parser.add_argument("--country", help="Filter by country")
        
        # Show model info
        info_parser = subparsers.add_parser("info", help="Show model information")
        info_parser.add_argument("model", help="Model name or path")
        
        # Delete model
        delete_parser = subparsers.add_parser("delete", help="Delete model")
        delete_parser.add_argument("model", help="Model name to delete")
        delete_parser.add_argument("--force", action="store_true", help="Force deletion")
    
    def execute(self, args) -> bool:
        try:
            from ..models import NERModelManager
            
            model_manager = NERModelManager(self.global_config.get('model_dir'), logger=self.logger)
            
            if args.model_action == "list":
                models = model_manager.list_models(country=args.country)
                print("Available models:")
                for model in models:
                    print(f"  {model}")
                
            elif args.model_action == "info":
                info = model_manager.get_model_info(args.model)
                print(f"Model information for '{args.model}':")
                print(json.dumps(info, indent=2, ensure_ascii=False))
                
            elif args.model_action == "delete":
                if not model_manager.model_exists(args.model):
                    print(f"Model '{args.model}' does not exist")
                    return False
                
                if not args.force:
                    response = input(f"Are you sure you want to delete model '{args.model}'? (y/N): ")
                    if response.lower() != 'y':
                        print("Deletion cancelled")
                        return True
                
                model_manager.delete_model(args.model)
                print(f"Deleted model '{args.model}'")
            
            else:
                print("Please specify a model action")
                return False
            
            return True
            
        except Exception as e:
            print(f"Model operation failed: {e}")
            return False

class StatusCommand(BaseCommand):
    """训练状态与日志查看命令

    功能：
    - 按国家或默认日志展示近期日志（tail）
    - 展示基本的目录状态与最近日志列表

    注意：
    - 日志文件命名约定（`{country}_training.log`）需与训练侧一致。
      
      # TODO: 与训练器的日志策略统一命名与路径；支持 CLI 选择具体日志文件。
    """
    
    @property
    def name(self) -> str:
        return "status"
    
    @property
    def description(self) -> str:
        return "Check training status and logs"
    
    def setup_parser(self, parser):
        parser.add_argument(
            "--training-id",
            help="Specific training ID to check"
        )
        parser.add_argument(
            "--country",
            help="Country to check status for"
        )
        parser.add_argument(
            "--logs",
            action="store_true",
            help="Show recent log entries"
        )
        parser.add_argument(
            "--tail",
            type=int,
            default=20,
            help="Number of log lines to show"
        )
    
    def execute(self, args) -> bool:
        try:
            log_dir = Path(self.global_config.get('log_dir', 'data/ner/logs'))
            
            if args.logs:
                # Show recent logs
                if args.country:
                    log_file = log_dir / f"{args.country}_training.log"
                else:
                    log_file = log_dir / "cli.log"
                
                if log_file.exists():
                    print(f"Recent logs from {log_file}:")
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines[-args.tail:]:
                            print(line.rstrip())
                else:
                    print(f"Log file {log_file} not found")
                    return False
            
            else:
                # Show general status
                print("NER System Status:")
                print(f"Log directory: {log_dir}")
                print(f"Model directory: {self.global_config.get('model_dir')}")
                print(f"Config directory: {self.global_config.get('config_dir')}")
                
                # List recent log files
                if log_dir.exists():
                    log_files = list(log_dir.glob("*.log"))
                    if log_files:
                        print("\nRecent log files:")
                        for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                            mtime = log_file.stat().st_mtime
                            import datetime
                            mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                            print(f"  {log_file.name} (modified: {mtime_str})")
            
            return True
            
        except Exception as e:
            print(f"Status check failed: {e}")
            return False