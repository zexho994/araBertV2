"""预处理管道命令

支持：
- 一次性文本预处理（--text --country）
- 交互式会话（REPL），参考 predict REPL：未提供 --text 时进入交互
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from ..config import ConfigManager
from ..preprocess import build_preprocessor_from_config

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

class PreprocessCommand(BaseCommand):
    """预处理管道命令

    支持：
    - 一次性文本预处理（--text --country）
    - 交互式会话（REPL），参考 predict REPL：未提供 --text 时进入交互
    """

    @property
    def name(self) -> str:
        return "preprocess"

    @property
    def description(self) -> str:
        return "Run preprocessing pipeline on text or start interactive REPL"

    def setup_parser(self, parser):
        parser.add_argument(
            "--country",
            required=False,
            help="Country code for configuration (e.g., 'uae')"
        )
        parser.add_argument(
            "--text",
            required=False,
            help="Text to preprocess (single run). If omitted, starts REPL"
        )

    def execute(self, args) -> bool:
        try:
            # 交互式会话：未提供 --text 时进入 REPL
            if not getattr(args, 'text', None):
                from .repl_preprocess import PreprocessREPL
                repl = PreprocessREPL(logger=self.logger)
                # 可选：若传入了 --country，作为默认国家加载
                if getattr(args, 'country', None):
                    try:
                        config = ConfigManager().load_country_config(args.country)
                        repl.preprocessor = build_preprocessor_from_config(config)
                        repl.country = args.country
                        repl._log_info(f"已加载国家配置: {args.country}")
                    except Exception as e:
                        repl._log_error(f"初始化国家配置失败: {e}")
                repl.run()
                return True

            # 一次性文本预处理路径：需要提供 country
            if not getattr(args, 'country', None):
                self.logger.error("--country is required when using --text for preprocess")
                return False

            # 加载国家配置与构建预处理器
            config = ConfigManager().load_country_config(args.country)
            preprocessor = build_preprocessor_from_config(config)

            # 执行预处理
            output_text = preprocessor.apply_text(args.text)
            self.logger.info(output_text)
            return True

        except Exception as e:
            self.logger.error(f"Preprocess failed: {e}")
            return False
