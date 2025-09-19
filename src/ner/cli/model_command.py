"""模型管理相关命令

支持：
- 列出/查询/删除 模型
"""

import json
from .base import BaseCommand

class ModelCommand(BaseCommand):
    """模型管理相关命令

    支持：
    - 列出/查询/删除 模型
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
