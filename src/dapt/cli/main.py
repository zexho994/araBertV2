"""DAPT CLI Manager

Main CLI manager that coordinates all DAPT command-line operations.
Handles command registration, execution, and global configuration.
"""

from typing import Dict, Any
from pathlib import Path

class DAPTCLIManager:
    """Main CLI manager for DAPT operations"""
    
    def __init__(self):
        self.global_config: Dict[str, Any] = {}
        self.commands: Dict[str, Any] = {}
        
    def set_global_config(self, config: Dict[str, Any]) -> None:
        """Set global configuration for all commands"""
        self.global_config = config
        
        # Ensure directories exist
        for dir_key in ['config_dir', 'data_dir', 'model_dir', 'log_dir']:
            if dir_key in config:
                Path(config[dir_key]).mkdir(parents=True, exist_ok=True)
    
    def setup_commands(self, subparsers) -> None:
        """Setup all available CLI commands"""
        from .commands import (
            TrainCommand,
            EvaluateCommand,
            ConfigCommand,
            DataCommand,
            ModelCommand,
            StatusCommand
        )
        
        # Register commands
        commands = [
            TrainCommand(),
            EvaluateCommand(),
            ConfigCommand(),
            DataCommand(),
            ModelCommand(),
            StatusCommand()
        ]
        
        for command in commands:
            command.setup_parser(subparsers)
            self.commands[command.name] = command
    
    def execute_command(self, command_name: str, args) -> bool:
        """Execute a specific command"""
        if command_name not in self.commands:
            print(f"Unknown command: {command_name}")
            return False
        
        command = self.commands[command_name]
        
        # Pass global config to command
        command.set_global_config(self.global_config)
        
        try:
            return command.execute(args)
        except Exception as e:
            print(f"Error executing command '{command_name}': {str(e)}")
            if self.global_config.get('verbose', False):
                import traceback
                traceback.print_exc()
            return False
    
    def list_commands(self) -> Dict[str, str]:
        """List all available commands with descriptions"""
        return {name: cmd.description for name, cmd in self.commands.items()}