#!/usr/bin/env python3
"""DAPT CLI Main Entry Point

Command-line interface for Domain Adaptive Pre-Training (DAPT) operations.
Provides comprehensive tools for training, evaluating, and managing
address parsing models for different countries.

Usage:
    python dapt_cli.py <command> [options]
    
Commands:
    train       - Start DAPT training for a specific country
    evaluate    - Evaluate trained models
    config      - Manage country configurations
    data        - Process and validate training data
    model       - Manage model operations
    status      - Check training status and logs
    
Examples:
    python dapt_cli.py train --country uae --epochs 10
    python dapt_cli.py evaluate --model uae_v1.0
    python dapt_cli.py config create --country uae
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from dapt.cli import DAPTCLIManager
from dapt.utils import DAPTLogger

def main():
    """Main entry point for DAPT CLI"""
    parser = argparse.ArgumentParser(
        description="DAPT CLI - Domain Adaptive Pre-Training for AraBERTv2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s train --country uae --epochs 10 --batch-size 16
  %(prog)s evaluate --model uae_v1.0 --test-data uae_test.csv
  %(prog)s config create --country uae --template default
  %(prog)s data validate --country uae --input-file training_data.csv
  %(prog)s model list --country uae
  %(prog)s status --training-id train_20240101_001
        """
    )
    
    # Global options
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--config-dir", 
        default="data/dapt/configs",
        help="Directory containing configuration files (default: data/dapt/configs)"
    )
    parser.add_argument(
        "--data-dir", 
        default="data/dapt/training_data",
        help="Directory containing training data (default: data/dapt/training_data)"
    )
    parser.add_argument(
        "--model-dir", 
        default="data/dapt/models",
        help="Directory for model storage (default: data/dapt/models)"
    )
    parser.add_argument(
        "--log-dir", 
        default="data/dapt/logs",
        help="Directory for log files (default: data/dapt/logs)"
    )
    parser.add_argument(
        "--eval-dir", 
        default="data/dapt/evaluation",
        help="Directory for evaluation results (default: data/dapt/evaluation)"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command", 
        help="Available commands",
        metavar="COMMAND"
    )
    
    # Initialize CLI manager and add commands
    cli_manager = DAPTCLIManager()
    cli_manager.setup_commands(subparsers)
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging with proper configuration
    # Create a minimal config for CLI logging
    cli_config = {
        "country": {"code": "cli"},
        "logging": {
            "level": "DEBUG" if args.verbose else "INFO",
            "log_file": "cli.log",
            "error_log_file": "cli_errors.log",
            "wandb": {"enabled": False},
            "tensorboard": {"enabled": False}
        }
    }
    
    cli_global_config = {
        "log_dir": args.log_dir
    }
    
    logger = DAPTLogger(cli_config, cli_global_config)
    
    # Execute command
    if args.command is None:
        parser.print_help()
        return 1
    
    try:
        # Set global configuration
        cli_manager.set_global_config({
            'config_dir': args.config_dir,
            'data_dir': args.data_dir,
            'model_dir': args.model_dir,
            'log_dir': args.log_dir,
            'eval_dir': getattr(args, 'eval_dir', 'data/dapt/evaluation'),
            'reports_dir': args.reports_dir,
            'verbose': args.verbose
        })
        
        # Execute the command
        result = cli_manager.execute_command(args.command, args)
        return 0 if result else 1
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Error executing command '{args.command}': {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())