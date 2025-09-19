"""管理国家配置的命令

支持：
- 列表/展示/创建/校验/删除 配置

# TODO: `create` 时支持指定输出根目录（覆盖模板中的 output.*），并提示创建的目录与样例文件。
"""

import json
from abc import ABC, abstractmethod
from typing import Dict, Any

from ..config import ConfigManager, ConfigValidator


class BaseCommand(ABC):
    """所有 NER CLI 子命令的抽象基类"""

    global_config: Dict[str, Any]
    country_config: Dict[str, Any]
    
    def __init__(self):
        self.global_config = {}
        self.country_config = {}
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
        return self.global_config

    def get_country_config(self, country: str) -> Dict[str, Any]:
        """Get country configuration"""
        if not self.country_config:
            config_manager = ConfigManager(self.global_config.get('config_dir'))
            self.country_config = config_manager.load_country_config(country)
        return self.country_config
    
    def set_logger(self, logger):
        """Inject a shared logger instance"""
        self.logger = logger
    
    def get_help(self) -> str:
        """Get help text for this command"""
        return self.description

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
