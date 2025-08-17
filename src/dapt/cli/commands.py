"""DAPT 命令行（CLI）指令集合

提供 DAPT（领域自适应预训练）的常用指令，包括但不限于：
- 训练管理（train）
- 模型评估（evaluate）
- 配置管理（config）
- 数据处理（data）
- 模型操作（model）
- 状态监控（status）

使用说明（高层设计）：
- 本模块通过抽象基类 `BaseCommand` 统一约束各子命令，确保具备参数解析与执行接口。
- 每个子命令在 `setup_parser` 中声明自己的参数；在 `execute` 中执行业务逻辑。
- 全局配置通过 `BaseCommand.global_config` 注入，通常包含如下关键目录：
  - `config_dir`: 配置文件根目录
  - `model_dir`: 模型保存根目录
  - `log_dir`: 日志根目录

注意事项：
- 如果全局配置未按要求提供上述键，将在运行期导致 KeyError。
- 子命令中有部分参数的默认值会覆盖配置文件中的值，这在易用性与可控性之间需要权衡。

# TODO: 为 `global_config` 定义强类型的数据结构（如 `TypedDict`/`dataclass`），并在入口处做完整校验。
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseCommand(ABC):
    """所有 CLI 子命令的基类

    责任：
    - 提供统一的名称与描述信息
    - 保存与注入全局配置 `global_config`
    - 约束子类实现参数解析 `setup_parser` 与执行 `execute`

    约定的 `global_config` 字段（由上层主程序注入）：
    - `config_dir`: 配置根目录（必须）
    - `model_dir`: 模型根目录（部分命令使用）
    - `log_dir`: 日志根目录（部分命令使用）

    # TODO: 在构造或 `set_global_config` 时进行键存在性与可访问性的校验，提前失败，提升可观测性。
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.global_config: Dict[str, Any] = {}
    
    def set_global_config(self, config: Dict[str, Any]) -> None:
        """Set global configuration"""
        self.global_config = config
    
    @abstractmethod
    def setup_parser(self, subparsers) -> None:
        """Setup argument parser for this command"""
        pass
    
    @abstractmethod
    def execute(self, args) -> bool:
        """Execute the command"""
        pass

class TrainCommand(BaseCommand):
    """启动 DAPT 训练的命令

    关键流程：
    1. 解析国家与配置路径，并加载配置
    2. 用命令行参数（如 epochs/batch_size/learning_rate）对配置进行覆盖
    3. 初始化训练引擎并执行训练

    风险与注意：
    - 当前实现中，即使用户未显式传参，也会用解析器的默认值覆盖配置文件中的值；这可能与用户期望不符。
      
      # TODO: 仅在用户显式传入参数时才覆盖配置（可通过将 argparse 默认值设为 None 实现）。
    - 依赖 `global_config` 中的 `config_dir`、`model_dir`；缺失将导致运行期错误。
    """
    
    def __init__(self):
        super().__init__("train", "Start DAPT training for a specific country")
    
    def setup_parser(self, subparsers) -> None:
        parser = subparsers.add_parser(
            self.name, 
            help=self.description,
            description="Start Domain Adaptive Pre-Training for a specific country model"
        )
        
        parser.add_argument(
            "--country", 
            required=True,
            help="Country code for training (e.g., uae, saudi, egypt)"
        )
        parser.add_argument(
            "--config", 
            help="Path to training configuration file"
        )
        parser.add_argument(
            "--epochs", 
            type=int, 
            default=10,
            help="Number of training epochs (default: 10)"
        )
        parser.add_argument(
            "--batch-size", 
            type=int, 
            default=16,
            help="Training batch size (default: 16)"
        )
        parser.add_argument(
            "--learning-rate", 
            type=float, 
            default=2e-5,
            help="Learning rate (default: 2e-5)"
        )
        parser.add_argument(
            "--output-dir", 
            help="Output directory for trained model"
        )
        parser.add_argument(
            "--resume", 
            help="Resume training from checkpoint"
        )
        parser.add_argument(
            "--dry-run", 
            action="store_true",
            help="Validate configuration without starting training"
        )
    
    def execute(self, args) -> bool:
        """执行训练命令"""
        from ..engine import DAPTTrainingEngine
        from ..config import ConfigManager
        
        print(f"Starting DAPT training for country: {args.country}")
        
        # Load configuration
        # 加载国家/自定义配置；优先使用 --config 指定文件，否则按国家模板加载
        # ERROR: 若 `self.global_config` 未包含 `config_dir` 键，此处将抛出 KeyError。
        config_manager = ConfigManager(self.global_config['config_dir'])
        
        if args.config:
            config = config_manager.load_config(args.config)
        else:
            config = config_manager.get_country_config(args.country)
        
        # Override config with command line arguments
        # 使用 CLI 参数覆盖配置文件中的训练超参
        # TODO: 仅当参数由用户显式传入时再覆盖，避免默认值意外覆盖配置文件。
        if hasattr(args, 'epochs'):
            config['training']['epochs'] = args.epochs
        if hasattr(args, 'batch_size'):
            config['training']['batch_size'] = args.batch_size
        if hasattr(args, 'learning_rate'):
            config['training']['learning_rate'] = args.learning_rate
        
        if args.dry_run:
            print("Configuration validation completed successfully")
            print(json.dumps(config, indent=2))
            return True
        
        # Initialize training engine
        # TODO: 支持更多训练相关参数（如随机种子、梯度累积、warmup 比例、混合精度等），并在配置中统一管理。
        engine = DAPTTrainingEngine(config, self.global_config)
        
        # Start training
        try:
            result = engine.train(
                output_dir=args.output_dir or f"{self.global_config['model_dir']}/{args.country}",
                resume_from=args.resume
            )
            print(f"Training completed successfully. Model saved to: {result['model_path']}")
            return True
        except Exception as e:
            print(f"Training failed: {str(e)}")
            return False

class EvaluateCommand(BaseCommand):
    """评估已训练模型的命令

    特点：
    - 针对 DAPT 模型提供简化评估流程
    - 在指定 `--eval_dir` 时会自动启用详细报告并将结果持久化

    # TODO: 支持递归统计模型体积（当前仅统计目录第一层文件大小）。
    """
    
    def __init__(self):
        super().__init__("evaluate", "Evaluate trained DAPT models")
    
    def setup_parser(self, subparsers) -> None:
        parser = subparsers.add_parser(
            self.name, 
            help=self.description,
            description="Evaluate performance of trained DAPT models"
        )
        
        parser.add_argument(
            "--model", 
            required=True,
            help="Model name or path to evaluate"
        )
        parser.add_argument(
            "--test-data", 
            help="Path to test data file"
        )
        parser.add_argument(
            "--country", 
            help="Country code for evaluation"
        )
        parser.add_argument(
            "--metrics", 
            nargs="+",
            default=["accuracy", "f1", "precision", "recall"],
            help="Metrics to calculate"
        )

        parser.add_argument(
            "--compare-with", 
            help="Compare with another model"
        )
        parser.add_argument(
            "--detailed-report", 
            action="store_true",
            help="Generate detailed evaluation report with additional metrics and analysis"
        )
        parser.add_argument(
            "--eval-dir", 
            help="Directory to save evaluation reports and results"
        )
    
    def execute(self, args) -> bool:
        """执行评估命令"""
        from ..evaluation import EvaluationManager
        from ..config import ConfigManager
        import os
        
        print(f"Evaluating model: {args.model}")
        
        # Handle eval_dir parameter and auto-enable detailed report
        eval_dir = None
        detailed_report_enabled = args.detailed_report
        if args.eval_dir:
            eval_dir = args.eval_dir
            detailed_report_enabled = True  # Auto-enable detailed report when eval_dir is specified
            # Create directory if it doesn't exist
            os.makedirs(eval_dir, exist_ok=True)
            print(f"Evaluation reports will be saved to: {eval_dir}")
            print("Detailed report automatically enabled when using --eval_dir")
        
        # Load country configuration if provided
        # 如果指定国家，则按国家维度加载配置；否则构造最小化配置用于 DAPT 评估
        if args.country:
            config_manager = ConfigManager(self.global_config['config_dir'])
            try:
                country_config = config_manager.get_country_config(args.country)
            except (FileNotFoundError, ValueError) as e:
                print(f"Failed to load configuration for country {args.country}: {str(e)}")
                return False
        else:
            # Create minimal config for DAPT evaluation
            country_config = {
                "country": {"code": "default"},
                "model": {"name": "default"},
                "data": {"path": "./"},
                "evaluation": {"metrics": {}},
                "logging": {
                    "level": "INFO",
                    "log_file": "evaluation.log",
                    "error_log_file": "evaluation_errors.log"
                }
            }
        
        evaluator = EvaluationManager(country_config, self.global_config)
        
        try:
            # For DAPT models, we'll use a simplified evaluation approach
            results = self._evaluate_dapt_model(
                evaluator=evaluator,
                model_path=args.model,
                test_data=args.test_data,
                country=args.country,
                metrics=args.metrics
            )
            
            if detailed_report_enabled:
                detailed_report = "\n=== Detailed Evaluation Report ===\n"
                detailed_report += f"Model: {args.model}\n"
                detailed_report += f"Country: {args.country or 'N/A'}\n"
                detailed_report += f"Test Data: {args.test_data or 'N/A'}\n"
                detailed_report += "\nEvaluation Results:\n"
                for metric, value in results.items():
                    if isinstance(value, float):
                        detailed_report += f"  {metric}: {value:.6f}\n"
                    else:
                        detailed_report += f"  {metric}: {value}\n"
                
                # Additional detailed metrics for DAPT models
                detailed_report += "\nDetailed Analysis:\n"
                detailed_report += f"  - Total metrics evaluated: {len(results)}\n"
                detailed_report += f"  - Metrics used: {', '.join(args.metrics)}\n"
                
                # Model information
                try:
                    if os.path.exists(args.model):
                        # TODO: 当前仅统计目录第一层文件大小，未递归子目录；
                        # 可考虑递归统计以获得更准确的模型体积。
                        model_size = sum(os.path.getsize(os.path.join(args.model, f)) 
                                       for f in os.listdir(args.model) if os.path.isfile(os.path.join(args.model, f)))
                        detailed_report += f"  - Model size: {model_size / (1024*1024):.2f} MB\n"
                except Exception:
                    pass
                
                print(detailed_report)
                
                # Save detailed report to eval_dir if specified
                if eval_dir:
                    report_file = os.path.join(eval_dir, "detailed_evaluation_report.txt")
                    with open(report_file, 'w', encoding='utf-8') as f:
                        f.write(detailed_report)
                    print(f"Detailed report saved to: {report_file}")
                    
            else:
                print("Evaluation Results:")
                for metric, value in results.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")
            
            # Save results to file when eval_dir is specified
            if eval_dir:
                output_file = os.path.join(eval_dir, "evaluation_results.json")
                # Always include detailed report when using eval_dir
                detailed_results = {
                    "model_path": args.model,
                    "country": args.country,
                    "test_data": args.test_data,
                    "metrics_used": args.metrics,
                    "results": results,
                    "detailed_report": True
                }
                evaluator.save_results(detailed_results, output_file)
                print(f"Results saved to: {output_file}")
            
            if args.compare_with:
                comparison = evaluator.compare_models(args.model, args.compare_with)
                print("\nModel Comparison:")
                print(json.dumps(comparison, indent=2))
            
            return True
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            return False
    
    def _evaluate_dapt_model(self, evaluator, model_path: str, test_data: str = None, 
                           country: str = None, metrics: list = None) -> dict:
        """以简化方式评估 DAPT 模型

        说明：
        - 载入 HF 配置/模型/分词器，返回模型结构与基础统计
        - 若提供 `test_data`，会进行基础文本统计，并在请求 `perplexity` 时尝试计算

        限制：
        - 非自回归语言模型的困惑度（perplexity）计算并不适用，且当前实现仅为占位示意。
        """
        import os
        from transformers import AutoModel, AutoTokenizer, AutoConfig
        
        try:
            # Load DAPT model
            print(f"Loading DAPT model from: {model_path}")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model path does not exist: {model_path}")
            
            # Load model components
            config = AutoConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModel.from_pretrained(model_path)
            
            # Basic model information
            results = {
                "model_path": model_path,
                "model_type": config.model_type if hasattr(config, 'model_type') else "unknown",
                "vocab_size": config.vocab_size if hasattr(config, 'vocab_size') else 0,
                "hidden_size": config.hidden_size if hasattr(config, 'hidden_size') else 0,
                "num_layers": config.num_hidden_layers if hasattr(config, 'num_hidden_layers') else 0,
                "num_attention_heads": config.num_attention_heads if hasattr(config, 'num_attention_heads') else 0
            }
            
            # Model size calculation
            try:
                model_size = sum(p.numel() for p in model.parameters())
                results["total_parameters"] = model_size
                results["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
            except Exception:
                results["total_parameters"] = 0
                results["trainable_parameters"] = 0
            
            # If test data is provided, perform basic text evaluation
            if test_data and os.path.exists(test_data):
                print(f"Evaluating on test data: {test_data}")
                text_results = self._evaluate_text_data(model, tokenizer, test_data)
                results.update(text_results)
            
            # Add basic metrics for DAPT
            if metrics:
                for metric in metrics:
                    if metric == "perplexity":
                        # For DAPT models, we can calculate perplexity if test data is available
                        if test_data and os.path.exists(test_data):
                            try:
                                ppl = self._calculate_perplexity(model, tokenizer, test_data)
                                results["perplexity"] = ppl
                            except Exception as e:
                                print(f"Warning: Could not calculate perplexity: {e}")
                                results["perplexity"] = float('inf')
                        else:
                            results["perplexity"] = "N/A (no test data)"
                    else:
                        # For other metrics, provide placeholder values
                        results[metric] = "N/A (DAPT model)"
            
            return results
            
        except Exception as e:
            raise Exception(f"Failed to evaluate DAPT model: {str(e)}")
    
    def _evaluate_text_data(self, model, tokenizer, test_data_path: str) -> dict:
        """在文本数据上进行基础统计评估

        说明：
        - 仅做轻量统计（样本数、平均长度、平均 token 数等）
        - 出于效率考虑，token 统计只抽样前 100 条样本

        # TODO: 允许通过参数控制采样大小，支持全量统计或分布指标（分位数）。
        """
        try:
            import json
            
            # Load test data
            with open(test_data_path, 'r', encoding='utf-8') as f:
                if test_data_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        texts = [item.get('text', str(item)) for item in data]
                    else:
                        texts = [str(data)]
                else:
                    # Assume plain text file
                    texts = f.read().strip().split('\n')
            
            # Basic statistics
            results = {
                "test_samples": len(texts),
                "avg_text_length": sum(len(text) for text in texts) / len(texts) if texts else 0,
                "total_tokens": 0
            }
            
            # Tokenization statistics
            total_tokens = 0
            for text in texts[:100]:  # 为效率限制在前 100 条样本
                tokens = tokenizer.encode(text, add_special_tokens=True)
                total_tokens += len(tokens)
            
            results["avg_tokens_per_sample"] = total_tokens / min(len(texts), 100)
            # TODO: 该 total_tokens 为抽样统计，而非全量；如需全量，需要遍历全部文本或改名以避免误解。
            results["total_tokens"] = total_tokens
            
            return results
            
        except Exception as e:
            print(f"Warning: Could not evaluate text data: {e}")
            return {"test_samples": 0, "avg_text_length": 0, "total_tokens": 0}
    
    def _calculate_perplexity(self, model, tokenizer, test_data_path: str) -> float:
        """计算 DAPT 模型的困惑度（perplexity）

        重要说明：
        - 当前实现仅为占位示例，并未基于语言模型的正确损失函数进行计算。
        - 对于非自回归结构（如 BERT 家族），困惑度指标并不直接适用。

        # ERROR: 下面的实现使用固定占位损失（`total_loss += 1.0`），并非真实的交叉熵损失，结果没有统计学意义。
        # TODO: 若需支持困惑度，应：
        #   1) 将模型切换为自回归语言模型（如 GPT 类），或
        #   2) 使用 `AutoModelForMaskedLM` 并基于 mask 目标计算近似指标，且明确定义口径。
        """
        try:
            import torch
            import json
            from torch.nn import CrossEntropyLoss
            
            # Load test data
            with open(test_data_path, 'r', encoding='utf-8') as f:
                if test_data_path.endswith('.json'):
                    data = json.load(f)
                    if isinstance(data, list):
                        texts = [item.get('text', str(item)) for item in data[:50]]  # Limit for efficiency
                    else:
                        texts = [str(data)]
                else:
                    texts = f.read().strip().split('\n')[:50]
            
            model.eval()
            total_loss = 0
            total_tokens = 0
            
            with torch.no_grad():
                for text in texts:
                    if not text.strip():
                        continue
                        
                    # Tokenize
                    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                    
                    # Get model outputs
                    outputs = model(**inputs)
                    
                    # For language modeling, we need the last hidden states
                    if hasattr(outputs, 'last_hidden_state'):
                        # Simple approximation for perplexity calculation
                        total_tokens += inputs['input_ids'].size(1)
                        # Use a simple loss approximation（占位实现，非真实损失）
                        total_loss += 1.0  # ERROR: 占位损失，需替换为基于 logits 的交叉熵损失
            
            if total_tokens > 0:
                avg_loss = total_loss / total_tokens
                perplexity = torch.exp(torch.tensor(avg_loss)).item()
                return min(perplexity, 1000.0)  # Cap at reasonable value
            else:
                return float('inf')
                
        except Exception as e:
            print(f"Warning: Perplexity calculation failed: {e}")
            return float('inf')

class ConfigCommand(BaseCommand):
    """管理国家级配置的命令

    功能：创建、列出、校验、展示配置。

    # TODO: `validate` 与 `show` 可支持按环境（dev/test/prod）或变体（模型/数据）进行细粒度过滤。
    """
    
    def __init__(self):
        super().__init__("config", "Manage country configurations")
    
    def setup_parser(self, subparsers) -> None:
        parser = subparsers.add_parser(
            self.name, 
            help=self.description,
            description="Create and manage country-specific configurations"
        )
        
        subcommands = parser.add_subparsers(dest="config_action", help="Configuration actions")
        
        # Create config
        create_parser = subcommands.add_parser("create", help="Create new country configuration")
        create_parser.add_argument("--country", required=True, help="Country code")
        create_parser.add_argument("--template", default="default", help="Configuration template")
        create_parser.add_argument("--force", action="store_true", help="Overwrite existing configuration")
        
        # List configs
        subcommands.add_parser("list", help="List available configurations")
        
        # Validate config
        validate_parser = subcommands.add_parser("validate", help="Validate configuration")
        validate_parser.add_argument("--country", help="Country code")
        validate_parser.add_argument("--file", help="Configuration file path")
        
        # Show config
        show_parser = subcommands.add_parser("show", help="Show configuration details")
        show_parser.add_argument("--country", required=True, help="Country code")
    
    def execute(self, args) -> bool:
        """执行配置相关命令"""
        from ..config import ConfigManager
        
        config_manager = ConfigManager(self.global_config['config_dir'])
        
        try:
            if args.config_action == "create":
                return self._create_config(config_manager, args)
            elif args.config_action == "list":
                return self._list_configs(config_manager)
            elif args.config_action == "validate":
                return self._validate_config(config_manager, args)
            elif args.config_action == "show":
                return self._show_config(config_manager, args)
            else:
                print("No action specified. Use --help for available actions.")
                return False
        except Exception as e:
            print(f"Configuration operation failed: {str(e)}")
            return False
    
    def _create_config(self, config_manager, args) -> bool:
        """Create new configuration"""
        success = config_manager.create_country_config(
            args.country, 
            template=args.template, 
            force=args.force
        )
        if success:
            print(f"Configuration created for country: {args.country}")
        return success
    
    def _list_configs(self, config_manager) -> bool:
        """List available configurations"""
        configs = config_manager.list_configs()
        print("Available configurations:")
        for config in configs:
            print(f"  - {config}")
        return True
    
    def _validate_config(self, config_manager, args) -> bool:
        """Validate configuration"""
        if args.country:
            valid = config_manager.validate_country_config(args.country)
        elif args.file:
            valid = config_manager.validate_config_file(args.file)
        else:
            print("Please specify either --country or --file")
            return False
        
        if valid:
            print("Configuration is valid")
        else:
            print("Configuration validation failed")
        return valid
    
    def _show_config(self, config_manager, args) -> bool:
        """Show configuration details"""
        config = config_manager.get_country_config(args.country)
        print(f"Configuration for {args.country}:")
        print(json.dumps(config, indent=2))
        return True

class DataCommand(BaseCommand):
    """数据处理相关命令

    子命令：
    - validate: 数据合法性校验
    - process: 预处理与切分
    - stats: 数据统计

    注意：
    - 当前实现中，`execute` 会无条件按 `args.country` 去加载国家配置；但 `validate/stats` 的 `--country` 是可选的，
      当未提供时将导致根据 `None` 尝试加载配置并失败。

      # ERROR: 参数定义与实现不一致。若 `--country` 为可选，需在未提供时使用最小化配置或调整为必填。
      # TODO: 若 `--country` 未提供，构造最小化配置以支持通用检查；或在解析器层将 `--country` 设为必选。
    """
    
    def __init__(self):
        super().__init__("data", "Process and validate training data")
    
    def setup_parser(self, subparsers) -> None:
        parser = subparsers.add_parser(
            self.name, 
            help=self.description,
            description="Process and validate training data for DAPT"
        )
        
        subcommands = parser.add_subparsers(dest="data_action", help="Data operations")
        
        # Validate data
        validate_parser = subcommands.add_parser("validate", help="Validate training data")
        validate_parser.add_argument("--input-file", required=True, help="Input data file")
        validate_parser.add_argument("--country", help="Country code for validation")
        validate_parser.add_argument("--format", choices=["csv", "json", "txt"], help="Data format")
        
        # Process data
        process_parser = subcommands.add_parser("process", help="Process and prepare training data")
        process_parser.add_argument("--input-file", required=True, help="Input data file")
        process_parser.add_argument("--output-file", help="Output file path")
        process_parser.add_argument("--country", required=True, help="Country code")
        process_parser.add_argument("--split-ratio", type=float, default=0.8, help="Train/validation split ratio")
        
        # Statistics
        stats_parser = subcommands.add_parser("stats", help="Show data statistics")
        stats_parser.add_argument("--input-file", required=True, help="Input data file")
        stats_parser.add_argument("--country", help="Country code")
    
    def execute(self, args) -> bool:
        """执行数据处理命令"""
        from ..data import DataProcessor
        from ..config import ConfigManager
        
        # Load country configuration
        config_manager = ConfigManager(self.global_config['config_dir'])
        try:
            country_config = config_manager.get_country_config(args.country)
        except (FileNotFoundError, ValueError) as e:
            print(f"Failed to load configuration for country {args.country}: {str(e)}")
            return False
        
        processor = DataProcessor(country_config, self.global_config)
        
        try:
            if args.data_action == "validate":
                return self._validate_data(processor, args)
            elif args.data_action == "process":
                return self._process_data(processor, args)
            elif args.data_action == "stats":
                return self._show_stats(processor, args)
            else:
                print("No action specified. Use --help for available actions.")
                return False
        except Exception as e:
            print(f"Data operation failed: {str(e)}")
            return False
    
    def _validate_data(self, processor, args) -> bool:
        """校验数据文件的结构与内容"""
        # Load data first if input file is specified
        if hasattr(args, 'input_file') and args.input_file:
            if not processor.load_data(args.input_file):
                print("Failed to load data file")
                return False
        
        # Validate the loaded data
        validation_result = processor.validate_data()
        
        if validation_result["valid"]:
            print("Data validation passed")
            if validation_result.get("warnings"):
                print("Warnings:")
                for warning in validation_result["warnings"]:
                    print(f"  - {warning}")
        else:
            print("Data validation failed")
            if validation_result.get("errors"):
                print("Errors:")
                for error in validation_result["errors"]:
                    print(f"  - {error}")
        
        return validation_result["valid"]
    
    def _process_data(self, processor, args) -> bool:
        """处理数据文件并进行切分/导出"""
        result = processor.process_data(
            input_file=args.input_file,
            output_file=args.output_file,
            country=args.country,
            split_ratio=args.split_ratio
        )
        if result:
            print(f"Data processed successfully. Output: {result['output_path']}")
            print(f"Training samples: {result['train_samples']}")
            print(f"Validation samples: {result['val_samples']}")
        return bool(result)
    
    def _show_stats(self, processor, args) -> bool:
        """展示数据统计信息（基础指标）"""
        stats = processor.get_data_stats(args.input_file, country=args.country)
        print("Data Statistics:")
        print(json.dumps(stats, indent=2))
        return True

class ModelCommand(BaseCommand):
    """模型管理相关命令

    功能：列出/导出/信息/删除 模型。

    # TODO: `delete` 操作在非交互环境中默认安全拒绝，可考虑支持 `--yes` 简化自动化流程。
    """
    
    def __init__(self):
        super().__init__("model", "Manage trained models")
    
    def setup_parser(self, subparsers) -> None:
        parser = subparsers.add_parser(
            self.name, 
            help=self.description,
            description="Manage and operate on trained DAPT models"
        )
        
        subcommands = parser.add_subparsers(dest="model_action", help="Model operations")
        
        # List models
        list_parser = subcommands.add_parser("list", help="List available models")
        list_parser.add_argument("--country", help="Filter by country")
        
        # Export model
        export_parser = subcommands.add_parser("export", help="Export model for deployment")
        export_parser.add_argument("--model", required=True, help="Model name or path")
        export_parser.add_argument("--format", choices=["pytorch", "onnx", "huggingface"], default="pytorch")
        export_parser.add_argument("--output-dir", help="Export directory")
        
        # Info
        info_parser = subcommands.add_parser("info", help="Show model information")
        info_parser.add_argument("--model", required=True, help="Model name or path")
        
        # Delete
        delete_parser = subcommands.add_parser("delete", help="Delete model")
        delete_parser.add_argument("--model", required=True, help="Model name or path")
        delete_parser.add_argument("--force", action="store_true", help="Force deletion without confirmation")
    
    def execute(self, args) -> bool:
        """执行模型相关命令"""
        from ..models import ModelManager
        from ..config import ConfigManager
        
        # Load country configuration if needed
        if hasattr(args, 'country') and args.country:
            config_manager = ConfigManager(self.global_config['config_dir'])
            try:
                country_config = config_manager.get_country_config(args.country)
                model_manager = ModelManager(country_config, self.global_config)
            except (FileNotFoundError, ValueError) as e:
                print(f"Failed to load configuration for country {args.country}: {str(e)}")
                return False
        else:
            # For operations that don't require country-specific config
            # Use a minimal config
            minimal_config = {"country": {"code": "default"}}
            model_manager = ModelManager(minimal_config, self.global_config)
        
        try:
            if args.model_action == "list":
                return self._list_models(model_manager, args)
            elif args.model_action == "export":
                return self._export_model(model_manager, args)
            elif args.model_action == "info":
                return self._show_model_info(model_manager, args)
            elif args.model_action == "delete":
                return self._delete_model(model_manager, args)
            else:
                print("No action specified. Use --help for available actions.")
                return False
        except Exception as e:
            print(f"Model operation failed: {str(e)}")
            return False
    
    def _list_models(self, model_manager, args) -> bool:
        """列出可用模型"""
        models = model_manager.list_models(country=args.country)
        print("Available models:")
        for model in models:
            print(f"  - {model['name']} ({model['country']}) - {model['created']}")
        return True
    
    def _export_model(self, model_manager, args) -> bool:
        """导出模型（pytorch/onnx/huggingface）"""
        result = model_manager.export_model(
            model_path=args.model,
            format=args.format,
            output_dir=args.output_dir
        )
        if result:
            print(f"Model exported to: {result['export_path']}")
        return bool(result)
    
    def _show_model_info(self, model_manager, args) -> bool:
        """展示模型元信息"""
        info = model_manager.get_model_info(args.model)
        print(f"Model Information for {args.model}:")
        print(json.dumps(info, indent=2))
        return True
    
    def _delete_model(self, model_manager, args) -> bool:
        """删除模型"""
        if not args.force:
            confirm = input(f"Are you sure you want to delete model '{args.model}'? (y/N): ")
            if confirm.lower() != 'y':
                print("Deletion cancelled")
                return True
        
        success = model_manager.delete_model(args.model)
        if success:
            print(f"Model '{args.model}' deleted successfully")
        return success

class StatusCommand(BaseCommand):
    """查看训练状态与日志的命令

    功能：
    - 列出训练会话（支持过滤活跃/按国家）
    - 查看指定会话的状态详情
    - 可选输出最近日志并控制 tail 行数

    # TODO: 支持流式实时追踪（类似 `tail -f`）。
    """
    
    def __init__(self):
        super().__init__("status", "Check training status and logs")
    
    def setup_parser(self, subparsers) -> None:
        parser = subparsers.add_parser(
            self.name, 
            help=self.description,
            description="Monitor training progress and view logs"
        )
        
        parser.add_argument(
            "--training-id", 
            help="Specific training ID to check"
        )
        parser.add_argument(
            "--country", 
            help="Filter by country"
        )
        parser.add_argument(
            "--active-only", 
            action="store_true",
            help="Show only active training sessions"
        )
        parser.add_argument(
            "--logs", 
            action="store_true",
            help="Show recent log entries"
        )
        parser.add_argument(
            "--tail", 
            type=int, 
            default=50,
            help="Number of log lines to show (default: 50)"
        )
    
    def execute(self, args) -> bool:
        """执行状态查询命令"""
        from ..utils import DAPTLogger
        
        logger = DAPTLogger(self.global_config['log_dir'])
        
        try:
            if args.training_id:
                status = logger.get_training_status(args.training_id)
                print(f"Training Status for {args.training_id}:")
                print(json.dumps(status, indent=2))
            else:
                sessions = logger.list_training_sessions(
                    country=args.country,
                    active_only=args.active_only
                )
                print("Training Sessions:")
                for session in sessions:
                    print(f"  - {session['id']} ({session['country']}) - {session['status']} - {session['started']}")
            
            if args.logs:
                logs = logger.get_recent_logs(
                    training_id=args.training_id,
                    tail=args.tail
                )
                print("\nRecent Logs:")
                for log_entry in logs:
                    print(f"  {log_entry['timestamp']} [{log_entry['level']}] {log_entry['message']}")
            
            return True
        except Exception as e:
            print(f"Status check failed: {str(e)}")
            return False