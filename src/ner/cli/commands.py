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

import json
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

from ..config import ConfigManager, ConfigValidator
from ..preprocess import build_preprocessor_from_config

class BaseCommand(ABC):
    """所有 NER CLI 子命令的抽象基类

    职责：
    - 提供统一的命令名称（`name`）与描述（`description`）属性
    - 定义参数解析接口 `setup_parser` 与执行接口 `execute`
    - 保存外层注入的 `global_config`

    # TODO: 支持注入统一的 logger，并在各子命令中复用。
    """

    global_config: Dict[str, Any]
    country_config: Dict[str, Any]
    
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

    def get_global_config(self) -> Dict[str, Any]:
        """Get global configuration"""
        if not self.global_config:
            self.global_config = ConfigManager(self.global_config.get('config_dir'))
        return self.global_config

    def get_country_config(self, country: str) -> Dict[str, Any]:
        """Get country configuration"""
        if not self.country_config:
            self.country_config = self.get_global_config().load_country_config(country)
        return self.country_config
    
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
            import torch
            from ..evaluation import NEREvaluator
            from ..models import NERModelManager
            from pathlib import Path
            from ..data import NERDataProcessor, NERDataLoader
            from transformers import AutoTokenizer
            
            # 加载模型
            model_manager = NERModelManager(logger=self.logger)
            model = model_manager.load_model(args.model_path)
            
            # 加载 tokenizer（保持与训练一致，优先从模型目录加载）
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)

            # 从模型配置中获取标签列表与映射（确保与模型训练时一致）
            if hasattr(model, 'config') and hasattr(model.config, 'id2label') and hasattr(model.config, 'label2id'):
                # id2label 可能是 {int: str} 或 {str: str}，统一按索引顺序取
                id2label = model.config.id2label
                model_label2id = model.config.label2id
                # 按键排序（数字键优先）；若是 str 键且可转 int，则按 int 排
                try:
                    label_list = [id2label[i] for i in range(len(id2label))]
                except Exception as e:
                    self.logger.error(f"加载标签列表与映射发生回退, ID2Label: {id2label}, Label2ID: {model_label2id}: {e}")
                    label_list = list(id2label.values())
            else:
                raise ValueError("Model configuration does not contain label mappings.")

            # 加载国家配置（用于数据与标签一致性）
            if not getattr(args, 'country', None):
                raise ValueError("--country is required for evaluate to ensure consistent data processing")
            config = self.get_country_config(args.country)

            # 准备数据（与训练流程一致）
            processor = NERDataProcessor(config, logger=self.logger)
            val_dataset = processor.load_data_file(args.data_path)

            # 构建与训练一致的 DataLoader（使用 is_split_into_words 对齐）
            ner_loader_builder = NERDataLoader(
                tokenizer_name=args.model_path,
                label2id=model_label2id,
                max_length=config.get('data', {}).get('max_length', 512),
                logger=self.logger
            )
            val_ds = ner_loader_builder.create_dataset_loader(val_dataset)
            batch_size = getattr(args, 'batch_size', None) or config.get('training', {}).get('batch_size', 16)
            val_loader = ner_loader_builder.create_dataloader(val_ds, batch_size=batch_size, shuffle=False)

            # 初始化评估器，迁移模型至设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            evaluator = NEREvaluator(model, tokenizer, label_list, device, logger=self.logger)

            # 运行基于 DataLoader 的评估
            results = evaluator.evaluate_dataloader(val_loader)
            
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

            # 持久化结果
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
                self.logger.info(f"\nSaved evaluation metrics to: {metrics_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return False

class EvaluatePredictCommand(BaseCommand):
    """基于 predict/evaluate() 的评估命令
    
    特点：
    - 使用 `model.predict()` + `NEREvaluator.evaluate()` 执行评估
    - 支持 `--confidence-threshold` 模拟线上推理阈值策略
    - 与训练解码不同处：低于阈值的词级预测将置为 'O'
    """
    
    @property
    def name(self) -> str:
        return "evaluate-predict"
    
    @property
    def description(self) -> str:
        return "Evaluate using model.predict() + evaluator.evaluate() (serving-like)"
    
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
            "--country",
            required=True,
            help="Country code for configuration"
        )
        parser.add_argument(
            "--output-dir", "-o",
            type=str,
            help="Output directory for evaluation results"
        )
        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.5,
            help="Confidence threshold used by model.predict()"
        )
        parser.add_argument(
            "--limit",
            type=int,
            help="Limit number of samples for quick evaluation"
        )
    
    def execute(self, args) -> bool:
        try:
            import torch
            from transformers import AutoTokenizer
            from ..evaluation import NEREvaluator
            from ..models import NERModelManager
            from ..config import ConfigManager
            from ..data import NERDataProcessor
            from pathlib import Path
            import json
            
            # 加载模型与 tokenizer
            model_manager = NERModelManager(logger=self.logger)
            model = model_manager.load_model(args.model_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
            
            # 解析标签列表（与模型一致）
            if hasattr(model, 'config') and hasattr(model.config, 'id2label') and hasattr(model.config, 'label2id'):
                id2label = model.config.id2label
                try:
                    label_list = [id2label[i] for i in range(len(id2label))]
                except Exception:
                    label_list = list(id2label.values())
            else:
                raise ValueError("Model configuration does not contain label mappings.")
            
            # 加载国家配置与数据
            config = self.get_country_config(args.country)
            processor = NERDataProcessor(config, logger=self.logger)
            dataset = processor.load_data_file(args.data_path)

            # 构建预处理器
            preprocessor = build_preprocessor_from_config(config)
            
            # 构造 texts 与 true_labels
            texts = []
            true_labels = []
            for ex in dataset[: (args.limit if getattr(args, 'limit', None) else None)]:
                tokens = ex.get('tokens')
                labels = ex.get('labels')
                if tokens is None or labels is None:
                    continue
                # 使用token-safe预处理，确保标签安全
                proc_tokens, proc_labels = preprocessor.apply_tokens(tokens, labels, allow_non_label_safe=False)
                if not proc_tokens or not proc_labels:
                    continue
                texts.append(' '.join(proc_tokens))
                true_labels.append(proc_labels)
            
            if not texts:
                raise ValueError("No valid examples found for evaluation")
            
            # 初始化评估器并执行评估
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            evaluator = NEREvaluator(model, tokenizer, label_list, device, logger=self.logger)
            results = evaluator.evaluate(texts, true_labels, confidence_threshold=getattr(args, 'confidence_threshold', 0.5))
            
            # 打印指标
            self.logger.info("Evaluation (predict) Results:")
            self.logger.info("\nToken-level Metrics:")
            for metric, value in results.get('token_metrics', {}).items():
                self.logger.info(f"  {metric}: {value:.4f}")
            self.logger.info("Entity-level Metrics:")
            for metric, value in results.get('entity_metrics', {}).items():
                self.logger.info(f"  {metric}: {value:.4f}")
            if 'per_entity_metrics' in results:
                self.logger.info("Per-Entity Metrics:")
                for entity, metrics in results['per_entity_metrics'].items():
                    self.logger.info(f"  {entity}:")
                    for metric, value in metrics.items():
                        self.logger.info(f"    {metric}: {value:.4f}")
            self.logger.info(f"Total samples evaluated: {results.get('num_samples', 0)}")
            
            # 持久化结果
            out_dir = None
            if getattr(args, 'output_dir', None):
                out_dir = Path(args.output_dir)
            elif config and 'output' in config and 'results_dir' in config['output']:
                out_dir = Path(config['output']['results_dir'])
            if out_dir is not None:
                out_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = out_dir / 'metrics_predict.json'
                with open(metrics_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                self.logger.info(f"\nSaved evaluation (predict) metrics to: {metrics_path}")
            
            return True
        except Exception as e:
            self.logger.error(f"Evaluation (predict) failed: {e}")
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
            required=False,
            help="Model name or path for prediction"
        )
        parser.add_argument(
            "--text",
            required=False,
            help="Text to predict (single prediction)"
        )
        parser.add_argument(
            "--file",
            required=False,
            help="File containing texts to predict"
        )
        parser.add_argument(
            "--output-file",
            required=False,
            help="File to save predictions"
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
            # 若既未提供 --text 也未提供 --file，则进入交互式 REPL
            if not getattr(args, 'text', None) and not getattr(args, 'file', None):
                from .repl_predict import PredictREPL
                repl = PredictREPL(logger=self.logger)
                # 继承一次性参数作为默认会话配置
                if getattr(args, 'output_format', None):
                    repl.output_format = args.output_format
                if getattr(args, 'confidence_threshold', None) is not None:
                    repl.confidence_threshold = args.confidence_threshold
                repl.run()
                return True

            # 执行一次性预测路径：需要提供 model_path
            if not getattr(args, 'model_path', None):
                self.logger.error("--model-path is required when using --text or --file")
                return False

            # 抑制 transformers 警告（仅在执行一次性预测时才导入重库）
            import logging
            logging.getLogger("transformers").setLevel(logging.ERROR)

            # 懒加载重依赖
            from ..models import NERModelManager
            from transformers import AutoTokenizer

            # 加载模型与分词器
            model_manager = NERModelManager(self.global_config.get('model_dir'), logger=self.logger)
            model = model_manager.load_model(args.model_path)
            tokenizer = AutoTokenizer.from_pretrained(args.model_path)

            # 构建预处理器
            preprocessor = None
            try:
                if self.get_country_config(args.country):
                    preprocessor = build_preprocessor_from_config(self.get_country_config(args.country))
                else:
                    # 兜底：使用空配置
                    preprocessor = build_preprocessor_from_config({})
            except Exception:
                preprocessor = None

            # 单条文本预测
            if args.text:
                input_text = args.text
                if preprocessor:
                    input_text = preprocessor.apply_text(input_text)
                prediction = model.predict(input_text, tokenizer=tokenizer, confidence_threshold=args.confidence_threshold)
                
                if args.output_format == "json":
                    self.logger.info(json.dumps(prediction, indent=2, ensure_ascii=False))
                elif args.output_format == "text":
                    for token, label in zip(prediction['tokens'], prediction['labels']):
                        self.logger.info(f"{token}\t{label}")
                elif args.output_format == "conll":
                    for token, label in zip(prediction['tokens'], prediction['labels']):
                        self.logger.info(f"{token} {label}")
                return True
            
            if args.file:
                predictions = []
                with open(args.file, 'r', encoding='utf-8') as f:
                    for line in f:
                        text = line.strip()
                        if text:
                            if preprocessor:
                                text = preprocessor.apply_text(text)
                            prediction = model.predict(text, tokenizer=tokenizer, confidence_threshold=args.confidence_threshold)
                            predictions.append(prediction)
                
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
                        self.logger.info(json.dumps(pred, indent=2, ensure_ascii=False))
                
                return True
            
            self.logger.error("Please provide either --text or --file")
            return False
            
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return False


class PreprocessCommand(BaseCommand):
    """预处理管道命令

    支持：
    - 一次性文本预处理（--text --country）
    - 交互式会话（REPL），参考 predict REPL：未提供 --text 时进入交互
    """

    @property
    def name(self) -> str:
        return "preprocess"

    @property
    def description(self) -> str:
        return "Run preprocessing pipeline on text or start interactive REPL"

    def setup_parser(self, parser):
        parser.add_argument(
            "--country",
            required=False,
            help="Country code for configuration (e.g., 'uae')"
        )
        parser.add_argument(
            "--text",
            required=False,
            help="Text to preprocess (single run). If omitted, starts REPL"
        )

    def execute(self, args) -> bool:
        try:
            # 交互式会话：未提供 --text 时进入 REPL
            if not getattr(args, 'text', None):
                from .repl_preprocess import PreprocessREPL
                repl = PreprocessREPL(logger=self.logger)
                # 可选：若传入了 --country，作为默认国家加载
                if getattr(args, 'country', None):
                    try:
                        from ..config import ConfigManager
                        cfg = ConfigManager()
                        config = cfg.load_country_config(args.country)
                        from ..preprocess import build_preprocessor_from_config
                        repl.preprocessor = build_preprocessor_from_config(config)
                        repl.country = args.country
                        repl._log_info(f"已加载国家配置: {args.country}")
                    except Exception as e:
                        repl._log_error(f"初始化国家配置失败: {e}")
                repl.run()
                return True

            # 一次性文本预处理路径：需要提供 country
            if not getattr(args, 'country', None):
                self.logger.error("--country is required when using --text for preprocess")
                return False

            # 加载国家配置与构建预处理器
            from ..config import ConfigManager
            config = ConfigManager().load_country_config(args.country)
            preprocessor = build_preprocessor_from_config(config)

            # 执行预处理
            output_text = preprocessor.apply_text(args.text)
            self.logger.info(output_text)
            return True

        except Exception as e:
            self.logger.error(f"Preprocess failed: {e}")
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
            if args.config_action == "list":
                if args.templates:
                    templates = self.get_global_config().list_templates()
                    self.logger.info("Available templates:")
                    for template in templates:
                        self.logger.info(f"  {template}")
                else:
                    countries = self.get_global_config().list_countries()
                    self.logger.info("Available country configurations:")
                    for country in countries:
                        self.logger.info(f"  {country}")
                
            elif args.config_action == "show":
                config = self.get_country_config(args.country)
                self.logger.info(f"Configuration for {args.country}:")
                self.logger.info(json.dumps(config, indent=2, ensure_ascii=False))
                
            elif args.config_action == "create":
                if self.get_global_config().country_exists(args.country):
                    self.logger.error(f"Configuration for '{args.country}' already exists")
                    return False

            elif args.config_action == "validate":
                config = self.get_country_config(args.country)
                validator = ConfigValidator()
                
                if validator.validate_config(config, args.country):
                    self.logger.info(f"Configuration for '{args.country}' is valid")
                    warnings = validator.get_warnings()
                    if warnings:
                        self.logger.info("Warnings:")
                        for warning in warnings:
                            self.logger.info(f"  WARNING: {warning}")
                else:
                    self.logger.error(f"Configuration for '{args.country}' is invalid")
                    for error in validator.get_errors():
                        self.logger.error(f"  ERROR: {error}")
                    return False
                
            elif args.config_action == "delete":
                if not self.get_global_config().country_exists(args.country):
                    self.logger.error(f"Configuration for '{args.country}' does not exist")
                    return False
                
                if not args.force:
                    response = input(f"Are you sure you want to delete configuration for '{args.country}'? (y/N): ")
                    if response.lower() != 'y':
                        self.logger.info("Deletion cancelled")
                        return True
                
                self.get_global_config().delete_country_config(args.country)
                self.logger.info(f"Deleted configuration for '{args.country}'")
            
            else:
                self.logger.error("Please specify a configuration action")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Configuration operation failed: {e}")
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
                config = self.get_country_config(args.country)
                
                processor = NERDataProcessor(config, logger=self.logger)
                is_valid = processor.validate_data_file(args.input_file)
                
                if is_valid:
                    self.logger.info(f"Data file '{args.input_file}' is valid")
                else:
                    self.logger.error(f"Data file '{args.input_file}' has validation errors")
                    return False
                
            elif args.data_action == "process":
                config = self.get_country_config(args.country)
                
                processor = NERDataProcessor(config, logger=self.logger)
                processor.process_file(args.input_file, args.output_file)
                self.logger.info(f"Processed data saved to '{args.output_file}'")
                
            elif args.data_action == "split":
                raise NotImplementedError("Data split not implemented")

            else:
                self.logger.error("Please specify a data action")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data operation failed: {e}")
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
                self.logger.info("Available models:")
                for model in models:
                    self.logger.info(f"  {model}")
                
            elif args.model_action == "info":
                info = model_manager.get_model_info(args.model)
                self.logger.info(f"Model information for '{args.model}':")
                self.logger.info(json.dumps(info, indent=2, ensure_ascii=False))
                
            elif args.model_action == "delete":
                if not model_manager.model_exists(args.model):
                    self.logger.error(f"Model '{args.model}' does not exist")
                    return False
                
                if not args.force:
                    response = input(f"Are you sure you want to delete model '{args.model}'? (y/N): ")
                    if response.lower() != 'y':
                        self.logger.info("Deletion cancelled")
                        return True
                
                model_manager.delete_model(args.model)
                self.logger.info(f"Deleted model '{args.model}'")
            
            else:
                self.logger.error("Please specify a model action")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Model operation failed: {e}")
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
                    self.logger.info(f"Recent logs from {log_file}:")
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line in lines[-args.tail:]:
                            self.logger.info(line.rstrip())
                else:
                    self.logger.error(f"Log file {log_file} not found")
                    return False
            
            else:
                # Show general status
                self.logger.info("NER System Status:")
                self.logger.info(f"Log directory: {log_dir}")
                self.logger.info(f"Model directory: {self.global_config.get('model_dir')}")
                self.logger.info(f"Config directory: {self.global_config.get('config_dir')}")
                
                # List recent log files
                if log_dir.exists():
                    log_files = list(log_dir.glob("*.log"))
                    if log_files:
                        self.logger.info("\nRecent log files:")
                        for log_file in sorted(log_files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                            mtime = log_file.stat().st_mtime
                            import datetime
                            mtime_str = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                            self.logger.info(f"  {log_file.name} (modified: {mtime_str})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Status check failed: {e}")
            return False