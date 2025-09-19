"""NER CLI Manager

Main CLI manager that coordinates all NER commands and provides
a unified interface for the command-line operations.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from .evaluate_command import EvaluateCommand
from .train_command import TrainCommand
from .predict_command import PredictCommand
from .evaluate_predict_command import EvaluatePredictCommand
from .preprocess_command import PreprocessCommand
from .config_command import ConfigCommand
from .data_command import DataCommand
from .model_command import ModelCommand
from .status_command import StatusCommand
from ..utils.logger import setup_logging

class NERCLIManager:
    """Manages NER CLI commands and global configuration"""
    
    def __init__(self, config_dir: str = "data/ner/configs", data_dir: str = "data/ner", 
                 model_dir: str = "data/ner/models", logger = None):
        """Initialize the CLI manager
        
        Args:
            config_dir: Directory for configuration files
            data_dir: Directory for data files
            model_dir: Directory for model files
            logger: Logger instance
        """
        self.commands = {}
        self.config_dir = config_dir
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.logger = logger
        
        # Initialize global config with provided directories
        self.global_config = {
            'config_dir': config_dir,
            'data_dir': data_dir,
            'model_dir': model_dir,
            'log_dir': f"{data_dir}/logs",
            'eval_dir': f"{data_dir}/eval"
        }
        
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
            ConfigCommand(),
            DataCommand(),
            ModelCommand(),
            StatusCommand()
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
    
    def set_global_config(self, config: Dict[str, Any]):
        """Set global configuration for all commands
        
        Args:
            config: Global configuration dictionary
        """
        # Merge with existing global config
        self.global_config.update(config)
        
        # Setup logging with global config
        if self.logger is None:
            self._setup_logging()
        
        # Pass global config to all commands
        for command in self.commands.values():
            command.set_global_config(self.global_config)
            # Ensure shared logger is injected
            if hasattr(command, "set_logger"):
                command.set_logger(self.logger)
            else:
                command.logger = self.logger
    
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
            if self.global_config.get('verbose', False):
                import traceback
                self.logger.error(traceback.format_exc())
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
        log_dir = self.global_config.get('log_dir', 'data/ner/logs')
        verbose = self.global_config.get('verbose', False)
        
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
    
    def validate_global_config(self) -> bool:
        """Validate global configuration
        
        Returns:
            True if configuration is valid, False otherwise
        """
        required_dirs = ['config_dir', 'data_dir', 'model_dir', 'log_dir', 'eval_dir']
        
        for dir_key in required_dirs:
            if dir_key not in self.global_config:
                if self.logger:
                    self.logger.error(f"Missing required directory configuration: {dir_key}")
                return False
            
            dir_path = Path(self.global_config[dir_key])
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Cannot create directory {dir_path}: {e}")
                return False
        
        return True
    
    def get_global_config(self) -> Dict[str, Any]:
        """Get current global configuration
        
        Returns:
            Copy of global configuration dictionary
        """
        return self.global_config.copy()
    
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