"""训练状态与日志查看命令

功能：
- 按国家或默认日志展示近期日志（tail）
- 展示基本的目录状态与最近日志列表

注意：
- 日志文件命名约定（`{country}_training.log`）需与训练侧一致。
  
# TODO: 与训练器的日志策略统一命名与路径；支持 CLI 选择具体日志文件。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path

from ..config import ConfigManager

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
