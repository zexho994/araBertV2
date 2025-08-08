"""NER CLI Commands

Implements all CLI commands for the NER system including training,
evaluation, prediction, configuration management, and more.
"""

import os
import json
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..config import ConfigManager, ConfigValidator
from ..utils import NERLogger

class BaseCommand(ABC):
    """Abstract base class for all NER CLI commands"""
    
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
    
    def get_help(self) -> str:
        """Get help text for this command"""
        return self.description

class TrainCommand(BaseCommand):
    """Command for training NER models"""
    
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
            # Load configuration
            config_manager = ConfigManager(self.global_config.get('config_dir'))

            config_arg = getattr(args, 'config', None)
            if config_arg:
                # Load custom configuration
                with open(config_arg, 'r', encoding='utf-8') as f:
                    config = json.load(f)
            else:
                # Load country configuration
                config = config_manager.load_country_config(args.country)
            
            # Override configuration with command line arguments
            if getattr(args, 'epochs', None):
                config['training']['epochs'] = args.epochs
            if getattr(args, 'batch_size', None):
                config['training']['batch_size'] = args.batch_size
            if getattr(args, 'learning_rate', None):
                config['training']['learning_rate'] = args.learning_rate
            if getattr(args, 'output_dir', None):
                config['output']['model_dir'] = args.output_dir
                
                # Optional data path overrides (when provided by outer CLI)
            if getattr(args, 'data_path', None):
                    config.setdefault('data', {})
                    config['data']['train_file'] = args.data_path
            if getattr(args, 'val_data_path', None):
                    config.setdefault('data', {})
                    config['data']['val_file'] = args.val_data_path
            
            # Validate configuration
            validator = ConfigValidator()
            if not validator.validate_config(config, args.country):
                print("Configuration validation failed:")
                for error in validator.get_errors():
                    print(f"  ERROR: {error}")
                for warning in validator.get_warnings():
                    print(f"  WARNING: {warning}")
                return False
            
            if getattr(args, 'dry_run', False):
                print("Configuration validation passed. Dry run completed.")
                return True
            
            # Import and initialize trainer
            from ..training import NERTrainer
            
            trainer = NERTrainer(config, self.global_config)
            
            if args.resume:
                trainer.resume_from_checkpoint(args.resume)
            
            # Start training
            trainer.train()
            
            return True
            
        except Exception as e:
            print(f"Training failed: {e}")
            return False

class EvaluateCommand(BaseCommand):
    """Command for evaluating NER models"""
    
    @property
    def name(self) -> str:
        return "evaluate"
    
    @property
    def description(self) -> str:
        return "Evaluate trained NER models"
    
    def setup_parser(self, parser):
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
        parser.add_argument(
            "--eval-dir",
            help="Directory to save evaluation results"
        )
    
    def execute(self, args) -> bool:
        try:
            # Import evaluator
            from ..evaluation import NEREvaluator
            from ..models import NERModelManager
            
            # Load model
            model_manager = NERModelManager(self.global_config.get('model_dir'))
            model = model_manager.load_model(args.model)
            
            # Load configuration if country specified
            config = None
            if args.country:
                config_manager = ConfigManager(self.global_config.get('config_dir'))
                config = config_manager.load_country_config(args.country)
            
            # Initialize evaluator
            evaluator = NEREvaluator(model, config, self.global_config)
            
            # Run evaluation
            results = evaluator.evaluate(
                test_data_path=args.test_data,
                metrics=args.metrics,
                detailed_report=args.detailed_report,
                output_dir=args.eval_dir or self.global_config.get('eval_dir')
            )
            
            # Print results
            print("Evaluation Results:")
            for metric, value in results.items():
                print(f"  {metric}: {value:.4f}")
            
            # Compare with another model if specified
            if args.compare_with:
                other_model = model_manager.load_model(args.compare_with)
                other_evaluator = NEREvaluator(other_model, config, self.global_config)
                other_results = other_evaluator.evaluate(
                    test_data_path=args.test_data,
                    metrics=args.metrics
                )
                
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
    """Command for making predictions with NER models"""
    
    @property
    def name(self) -> str:
        return "predict"
    
    @property
    def description(self) -> str:
        return "Use trained model for prediction"
    
    def setup_parser(self, parser):
        parser.add_argument(
            "--model",
            required=True,
            help="Model name or path for prediction"
        )
        parser.add_argument(
            "--text",
            help="Text to predict (single prediction)"
        )
        parser.add_argument(
            "--input-file",
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
            "--format",
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
            model_manager = NERModelManager(self.global_config.get('model_dir'))
            model = model_manager.load_model(args.model)
            
            # Load configuration if country specified
            config = None
            if args.country:
                config_manager = ConfigManager(self.global_config.get('config_dir'))
                config = config_manager.load_country_config(args.country)
            
            # Single text prediction
            if args.text:
                prediction = model.predict(args.text, confidence_threshold=args.confidence_threshold)
                
                if args.format == "json":
                    print(json.dumps(prediction, indent=2, ensure_ascii=False))
                elif args.format == "text":
                    for token, label in zip(prediction['tokens'], prediction['labels']):
                        print(f"{token}\t{label}")
                elif args.format == "conll":
                    for token, label in zip(prediction['tokens'], prediction['labels']):
                        print(f"{token} {label}")
                
                return True
            
            # Batch prediction from file
            if args.input_file:
                predictions = []
                
                with open(args.input_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        text = line.strip()
                        if text:
                            prediction = model.predict(text, confidence_threshold=args.confidence_threshold)
                            predictions.append(prediction)
                
                # Save or print predictions
                if args.output_file:
                    with open(args.output_file, 'w', encoding='utf-8') as f:
                        if args.format == "json":
                            json.dump(predictions, f, indent=2, ensure_ascii=False)
                        else:
                            for pred in predictions:
                                if args.format == "text":
                                    for token, label in zip(pred['tokens'], pred['labels']):
                                        f.write(f"{token}\t{label}\n")
                                    f.write("\n")
                                elif args.format == "conll":
                                    for token, label in zip(pred['tokens'], pred['labels']):
                                        f.write(f"{token} {label}\n")
                                    f.write("\n")
                else:
                    for pred in predictions:
                        print(json.dumps(pred, indent=2, ensure_ascii=False))
                        print("---")
                
                return True
            
            print("Please provide either --text or --input-file")
            return False
            
        except Exception as e:
            print(f"Prediction failed: {e}")
            return False

class ConfigCommand(BaseCommand):
    """Command for managing configurations"""
    
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
    """Command for data processing operations"""
    
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
                
                processor = NERDataProcessor(config)
                is_valid = processor.validate_data_file(args.input_file)
                
                if is_valid:
                    print(f"Data file '{args.input_file}' is valid")
                else:
                    print(f"Data file '{args.input_file}' has validation errors")
                    return False
                
            elif args.data_action == "process":
                config_manager = ConfigManager(self.global_config.get('config_dir'))
                config = config_manager.load_country_config(args.country)
                
                processor = NERDataProcessor(config)
                processor.process_file(args.input_file, args.output_file)
                print(f"Processed data saved to '{args.output_file}'")
                
            elif args.data_action == "split":
                # Validate ratios
                total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
                if abs(total_ratio - 1.0) > 0.001:
                    print(f"Ratios must sum to 1.0, got {total_ratio}")
                    return False
                
                processor = NERDataProcessor()
                processor.split_data(
                    args.input_file,
                    args.output_dir,
                    train_ratio=args.train_ratio,
                    val_ratio=args.val_ratio,
                    test_ratio=args.test_ratio
                )
                print(f"Data split completed. Files saved to '{args.output_dir}'")
            
            else:
                print("Please specify a data action")
                return False
            
            return True
            
        except Exception as e:
            print(f"Data operation failed: {e}")
            return False

class ModelCommand(BaseCommand):
    """Command for model management operations"""
    
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
            
            model_manager = NERModelManager(self.global_config.get('model_dir'))
            
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
    """Command for checking training status and logs"""
    
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