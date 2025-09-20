"""NER CLI Manager

Main CLI manager that coordinates all NER commands and provides
a unified interface for the command-line operations.
"""

from typing import Dict, Optional
from pathlib import Path

from ..config import GlobalConfig
from .evaluate_command import EvaluateCommand
from .train_command import TrainCommand
from .predict_command import PredictCommand
from .evaluate_predict_command import EvaluatePredictCommand
from .preprocess_command import PreprocessCommand
from .data_command import DataCommand
from .model_command import ModelCommand
from ..utils.logger import setup_logging


class NERCLIManager:
    """Manages NER CLI commands and global configuration"""

    global_config: GlobalConfig
    
    def __init__(self, config_dir: str = "data/ner/configs", data_dir: str = "data/ner", model_dir: str = "data/ner/models", logger = None):
        """Initialize the CLI manager
        
        Args:
            config_dir: Directory for configuration files
            data_dir: Directory for data files
            model_dir: Directory for model files
            logger: Logger instance
        """
        self.commands = {}
        self.logger = logger
        self.global_config = GlobalConfig(config_dir, data_dir, model_dir)
        
        # Register all available commands
        self._register_commands()
        
        # Pass global config to all commands immediately
        for command in self.commands.values():
            command.set_global_config(self.global_config)
            if self.logger is not None:
                # Inject shared logger if already provided
                if hasattr(command, "set_logger"):
                    command.set_logger(self.logger)
                else:
                    command.logger = self.logger
    
    def _register_commands(self):
        """Register all available CLI commands"""
        commands = [
            TrainCommand(),
            EvaluateCommand(),
            EvaluatePredictCommand(),
            PredictCommand(),
            PreprocessCommand(),
            DataCommand(),
            ModelCommand(),
        ]
        
        for command in commands:
            self.commands[command.name] = command
    
    def setup_commands(self, subparsers):
        """Setup command parsers for all registered commands
        
        Args:
            subparsers: Argparse subparsers object
        """
        for command_name, command in self.commands.items():
            # Create subparser for this command
            parser = subparsers.add_parser(
                command_name,
                help=command.description,
                description=command.description
            )
            
            # Let the command setup its specific arguments
            command.setup_parser(parser)
    
    
    def execute_command(self, command_name: str, args) -> bool:
        """Execute a specific command
        
        Args:
            command_name: Name of the command to execute
            args: Parsed command arguments
            
        Returns:
            True if command executed successfully, False otherwise
        """
        if command_name not in self.commands:
            self.logger.error(f"Unknown command: {command_name}")
            return False
        
        command = self.commands[command_name]
        
        try:
            self.logger.info(f"Executing command: {command_name}")
            result = command.execute(args)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error executing command '{command_name}': {str(e)}")
            return False
    
    def list_commands(self) -> Dict[str, str]:
        """Get a list of all available commands
        
        Returns:
            Dictionary mapping command names to descriptions
        """
        return {name: cmd.description for name, cmd in self.commands.items()}
    
    def get_command_help(self, command_name: str) -> Optional[str]:
        """Get help text for a specific command
        
        Args:
            command_name: Name of the command
            
        Returns:
            Help text for the command, or None if command doesn't exist
        """
        if command_name in self.commands:
            return self.commands[command_name].get_help()
        return None
    
    def _setup_logging(self):
        """Setup logging based on global configuration"""
        log_dir = self.global_config.get_log_dir()
        verbose = self.global_config.get_verbose()
        
        # Create log directory if it doesn't exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # Initialize and configure a shared logger instance
        self.logger = setup_logging(
            level='DEBUG' if verbose else 'INFO',
            log_dir=str(log_dir),
            log_to_file=True,
            log_to_console=True,
            logger_name='ner_cli'
        )
    
    
    def get_global_config(self) -> GlobalConfig:
        """Get current global configuration
        
        Returns:
            Copy of global configuration dictionary
        """
        return self.global_config
    
    def shutdown(self):
        """Cleanup and shutdown the CLI manager"""
        if self.logger:
            self.logger.info("NER CLI Manager shutting down")
        
        # Cleanup commands
        for command in self.commands.values():
            if hasattr(command, 'cleanup'):
                try:
                    command.cleanup()
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Error during command cleanup: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.shutdown()