#!/usr/bin/env python3
"""NER CLI - Named Entity Recognition Command Line Interface

A comprehensive CLI tool for training, evaluating, and using NER models
with support for multiple languages and custom configurations.

Usage:
    python ner.py <command> [options]

Commands:
    train       Train a new NER model
    evaluate    Evaluate a trained model
    predict     Make predictions on text
    preprocess  Run preprocessing pipeline on text or start REPL
    config      Manage configurations
    data        Data processing utilities
    model       Model management utilities
    status      Show system status

Examples:
    # Train a model for UAE addresses
    python ner.py train --country uae --data-path ./data/uae_train.json
    
    # Evaluate a model
    python ner.py evaluate --model-path ./models/uae_model --data-path ./data/uae_test.json --country uae
    
    # Evaluate with detailed report
    python ner.py evaluate --model-path ./models/uae_model --data-path ./data/uae_test.json --country uae --detailed-report
    
    # Evaluate with confidence threshold (more aligned with production predictions)
    python ner.py evaluate --model-path ./models/uae_model --data-path ./data/uae_test.json --country uae --confidence-threshold 0.7
    
    # Evaluate with entity focus and detailed report (generates both Excel and JSONL files)
    python ner.py evaluate --model-path ./models/uae_model --data-path ./data/uae_test.json --country uae --detailed-report --entity "country,city"
    
    # Make predictions
    python ner.py predict --model-path ./models/uae_model --text "123 Sheikh Zayed Road, Dubai"
    or
    python ner.py predict # enter interactive mode
    
    # Batch prediction with detailed report (JSONL format supported)
    python ner.py predict --model-path ./models/uae_model --file ./input.jsonl --output-file predictions.json --detailed-report
    
    # Create a new country configuration
    python ner.py config create --country egypt --template address_ner
"""

from datetime import datetime
import sys
import argparse
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser"""
    parser = argparse.ArgumentParser(
        prog='ner',
        description='Named Entity Recognition CLI Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --country uae --data-path ./data/train.json
  %(prog)s evaluate --model-path ./models/uae_model --data-path ./data/test.json --country uae
  %(prog)s evaluate --model-path ./models/uae_model --data-path ./data/test.json --country uae --detailed-report
  %(prog)s evaluate --model-path ./models/uae_model --data-path ./data/test.json --country uae --confidence-threshold 0.7
  %(prog)s evaluate --model-path ./models/uae_model --data-path ./data/test.json --country uae --detailed-report --entity "country,city"
  %(prog)s predict --model-path ./models/uae_model --text "Dubai Marina"
  %(prog)s predict --model-path ./models/uae_model --file ./input.jsonl --output-file predictions.json --detailed-report
  %(prog)s config list
  %(prog)s data validate --data-path ./data/train.json
  %(prog)s model list
  %(prog)s status

For more information on each command, use:
  %(prog)s <command> --help
"""
    )
    
    # Global options
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO',
        help='Set logging level (default: INFO)'
    )
    
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path (default: logs/ner.log)'
    )
    
    parser.add_argument(
        '--config-dir',
        type=str,
        default='data/ner/configs',
        help='Configuration directory (default: data/ner/configs)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data/ner',
        help='Data directory (default: data/ner)'
    )
    
    parser.add_argument(
        '--model-dir',
        type=str,
        default='data/ner/models',
        help='Model directory (default: data/ner/models)'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='NER CLI 1.0.0'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        help='Available commands',
        metavar='<command>'
    )
    
    # Train command
    train_parser = subparsers.add_parser(
        'train',
        help='Train a new NER model',
        description='Train a new NER model with specified configuration'
    )
    train_parser.add_argument(
        '--country', '-c',
        type=str,
        required=True,
        help='Country configuration to use (e.g., uae, egypt)'
    )
    train_parser.add_argument(
        '--data-path', '-d',
        type=str,
        required=False,
        help='Path to training data file (optional; defaults to config data.train_file)'
    )
    train_parser.add_argument(
        '--val-data-path',
        type=str,
        help='Path to validation data file'
    )
    train_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for trained model'
    )
    train_parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs'
    )
    train_parser.add_argument(
        '--batch-size',
        type=int,
        help='Training batch size'
    )
    train_parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate'
    )
    train_parser.add_argument(
        '--resume',
        type=str,
        help='Resume training from checkpoint'
    )
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without training"
    )
    # LoRA相关参数
    train_parser.add_argument(
        "--lora",
        action="store_true",
        help="Enable LoRA training mode"
    )
    train_parser.add_argument(
        "--lora-r",
        type=int,
        help="LoRA attention dimension (r)"
    )
    train_parser.add_argument(
        "--lora-alpha",
        type=int,
        help="LoRA alpha parameter"
    )
    train_parser.add_argument(
        "--lora-dropout",
        type=float,
        help="LoRA dropout rate"
    )
    train_parser.add_argument(
        "--lora-target-modules",
        help="Comma-separated list of target modules for LoRA"
    )
    train_parser.add_argument(
        "--lora-base-adapter",
        help="Path to base LoRA adapter for incremental training"
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate a trained model',
        description='Evaluate a trained NER model on test data'
    )
    eval_parser.add_argument(
        '--model-path', '-m',
        type=str,
        required=True,
        help='Path to trained model'
    )
    eval_parser.add_argument(
        '--data-path', '-d',
        type=str,
        required=True,
        help='Path to evaluation data file'
    )
    eval_parser.add_argument(
        '--country',
        type=str,
        help='Country configuration to use (used to load results_dir from config)'
    )
    eval_parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for evaluation results'
    )
    eval_parser.add_argument(
        '--batch-size',
        type=int,
        help='Evaluation batch size'
    )
    eval_parser.add_argument(
        '--detailed-report', '--dr',
        action='store_true',
        help='Generate detailed evaluation report (Excel format)'
    )
    eval_parser.add_argument(
        '--entity', '-e',
        type=str,
        help='Comma-separated list of entity types to focus on (e.g., "country,city"). If specified, will create a special section for samples where all specified entities have issues and generate a separate JSONL file with problematic data.'
    )
    eval_parser.add_argument(
        '--confidence-threshold','--ct',
        type=float,
        default=0.5,
        help='Confidence threshold for predictions (default: 0.5)'
    )

    # Predict command
    predict_parser = subparsers.add_parser(
        'predict',
        help='Make predictions on text',
        description='Make NER predictions on input text'
    )
    predict_parser.add_argument(
        '--model-path', '-m',
        type=str,
        required=False,
        help='Path to trained model'
    )
    predict_group = predict_parser.add_mutually_exclusive_group(required=False)
    predict_group.add_argument(
        '--text', '-t',
        type=str,
        required=False,
        help='Text to analyze'
    )
    predict_group.add_argument(
        '--file', '-f',
        type=str,
        required=False,
        help='File containing text to analyze'
    )
    predict_parser.add_argument(
        '--output-format',
        choices=['json', 'text', 'conll'],
        default='json',
        help='Output format (default: json)'
    )
    predict_parser.add_argument(
        '--confidence-threshold',
        type=float,
        default=0.5,
        help='Confidence threshold for predictions (default: 0.5)'
    )
    predict_parser.add_argument(
        '--output-file', '-o',
        type=str,
        required=False,
        help='File to save predictions'
    )
    predict_parser.add_argument(
        '--detailed-report', '--dr',
        action='store_true',
        help='Generate detailed prediction report (Excel format) for batch prediction'
    )

    #----------preprocess command----------

    # Preprocess command
    preprocess_parser = subparsers.add_parser(
        'preprocess',
        help='Run preprocessing pipeline on text or start REPL',
        description='Apply country-specific preprocessing pipeline to text'
    )
    preprocess_parser.add_argument(
        '--country',
        type=str,
        required=False,
        help='Country configuration to use (e.g., uae)'
    )
    preprocess_group = preprocess_parser.add_mutually_exclusive_group(required=False)
    preprocess_group.add_argument(
        '--text', '-t',
        type=str,
        required=False,
        help='Text to preprocess (if omitted, starts interactive REPL)'
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Manage configurations',
        description='Manage NER configurations'
    )
    config_subparsers = config_parser.add_subparsers(
        dest='config_action',
        help='Configuration actions'
    )
    
    # Config list
    config_list_parser = config_subparsers.add_parser(
        'list',
        help='List available configurations'
    )
    config_list_parser.add_argument(
        '--templates',
        action='store_true',
        help='List available templates instead of country configurations'
    )
    
    # Config show
    config_show_parser = config_subparsers.add_parser(
        'show',
        help='Show configuration details'
    )
    config_show_parser.add_argument(
        'country',
        type=str,
        help='Country configuration to show'
    )
    
    # Config create
    config_create_parser = config_subparsers.add_parser(
        'create',
        help='Create new configuration'
    )
    config_create_parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Country name for new configuration'
    )
    config_create_parser.add_argument(
        '--template',
        type=str,
        default='default',
        help='Template to use (default: default)'
    )
    config_create_parser.add_argument(
        '--external-template',
        type=str,
        help='Path to external template file (overrides --template)'
    )
    
    # Config validate
    config_validate_parser = config_subparsers.add_parser(
        'validate',
        help='Validate configuration'
    )
    config_validate_parser.add_argument(
        'country',
        type=str,
        help='Country configuration to validate'
    )
    
    #----------data command----------
    
    # Data command
    data_parser = subparsers.add_parser(
        'data',
        help='Data processing utilities',
        description='Data processing and validation utilities'
    )
    data_subparsers = data_parser.add_subparsers(
        dest='data_action',
        help='Data actions'
    )
    
    # Data validate
    data_validate_parser = data_subparsers.add_parser(
        'validate',
        help='Validate data format'
    )
    data_validate_parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to data file'
    )
    data_validate_parser.add_argument(
        '--format',
        choices=['json', 'conll', 'csv'],
        help='Data format (auto-detected if not specified)'
    )
    
    # Data convert
    data_convert_parser = data_subparsers.add_parser(
        'convert',
        help='Convert CSV to JSONL format or validate CSV data'
    )
    data_convert_parser.add_argument(
        '--input-path',
        type=str,
        required=True,
        help='Input CSV file path'
    )
    data_convert_parser.add_argument(
        '--output-path',
        type=str,
        help='Output JSONL file path (not required when using -ov)'
    )
    data_convert_parser.add_argument(
        '--country',
        type=str,
        required=True,
        help='Country code for configuration'
    )
    data_convert_parser.add_argument(
        '--text-column',
        type=str,
        default='formatted_address',
        help='Text column name (default: formatted_address)'
    )
    data_convert_parser.add_argument(
        '--validation-mode',
        choices=['strict', 'lenient'],
        default='strict',
        help='Validation mode (default: strict)'
    )
    data_convert_parser.add_argument(
        '-ov', '--only-validate',
        action='store_true',
        help='Only validate CSV data without conversion'
    )
    data_convert_parser.add_argument(
        '-fix', '--fix',
        action='store_true',
        help='Automatically process anomalies (equivalent to auto_process=True)'
    )
    
    # Data split
    data_split_parser = data_subparsers.add_parser(
        'split',
        help='Split data into train/val/test sets'
    )
    data_split_parser.add_argument(
        '--data-path',
        type=str,
        required=True,
        help='Path to data file'
    )
    data_split_parser.add_argument(
        '--output-dir',
        type=str,
        required=True,
        help='Output directory for split files'
    )
    data_split_parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.8,
        help='Training set ratio (default: 0.8)'
    )
    data_split_parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.1,
        help='Validation set ratio (default: 0.1)'
    )
    
    # Data merge/compare
    data_merge_parser = data_subparsers.add_parser(
        'merge',
        help='Merge and compare two model evaluation reports'
    )
    data_merge_parser.add_argument(
        '--report-1',
        type=str,
        required=True,
        help='First model report path (xlsx or csv)'
    )
    data_merge_parser.add_argument(
        '--report-2',
        type=str,
        required=True,
        help='Second model report path (xlsx or csv)'
    )
    data_merge_parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output merged report path (xlsx)'
    )
    data_merge_parser.add_argument(
        '--model1-name',
        type=str,
        default='Model-1',
        help='First model name (default: Model-1)'
    )
    data_merge_parser.add_argument(
        '--model2-name',
        type=str,
        default='Model-2',
        help='Second model name (default: Model-2)'
    )

    #----------model command----------
    
    # Model command
    model_parser = subparsers.add_parser(
        'model',
        help='Model management utilities',
        description='Model management and analysis utilities'
    )
    model_subparsers = model_parser.add_subparsers(
        dest='model_action',
        help='Model actions'
    )
    
    # Model list
    model_subparsers.add_parser(
        'list',
        help='List available models'
    )
    
    # Model info
    model_info_parser = model_subparsers.add_parser(
        'info',
        help='Show model information'
    )
    model_info_parser.add_argument(
        'model_name',
        type=str,
        help='Model name'
    )
    
    # Model delete
    model_delete_parser = model_subparsers.add_parser(
        'delete',
        help='Delete a model'
    )
    model_delete_parser.add_argument(
        'model_name',
        type=str,
        help='Model name to delete'
    )
    model_delete_parser.add_argument(
        '--force',
        action='store_true',
        help='Force deletion without confirmation'
    )
    
    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show system status',
        description='Show NER system status and information'
    )
    status_parser.add_argument(
        '--detailed',
        action='store_true',
        help='Show detailed status information'
    )
    
    return parser

def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # Setup logging
    log_level = 'DEBUG' if args.verbose else args.log_level
    
    # 日志文件名, {country.code}
    if args.log_file:
        log_file = args.log_file
    elif hasattr(args, 'country') and args.country:
        log_file = f'data/ner/logs/{args.country}/ner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    else:
        log_file = f'data/ner/logs/ner_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    
    try:
        from ner.utils.logger import setup_logging
        from ner.cli.main import NERCLIManager
    except ImportError as e:
        print(f"Error importing NER modules: {e}")
        return 1

    logger = setup_logging(
        level=log_level,
        log_dir= Path(log_file).parent,
        log_to_console=True,
        log_to_file=True
    )
    
    try:
        cli_manager = NERCLIManager(
            config_dir=args.config_dir,
            data_dir=args.data_dir,
            model_dir=args.model_dir,
            logger=logger
        )
        
        # Execute command
        result = cli_manager.execute_command(args.command, args)
        
        if result:
            logger.info(f"Command '{args.command}' completed successfully")
        else:
            logger.error(f"Command '{args.command}' failed")
    
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        return 1
    
    finally:
        # Cleanup
        try:
            if 'cli_manager' in locals() and hasattr(cli_manager, 'shutdown'):
                cli_manager.shutdown()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

if __name__ == '__main__':
    sys.exit(main())