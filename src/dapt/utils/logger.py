"""DAPT Logging System

Unified logging system for DAPT training:
- Structured logging with multiple handlers
- Country-specific log management
- Integration with external logging services
- Performance and error tracking
- Log rotation and archival
"""

import os
import sys
import logging
import logging.handlers
import json
import traceback
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import threading
from contextlib import contextmanager
import time
from enum import Enum

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

class LogLevel(Enum):
    """Log levels enumeration"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

@dataclass
class LogEntry:
    """Structured log entry"""
    timestamp: str
    level: str
    logger_name: str
    message: str
    country_code: str
    module: str
    function: str
    line_number: int
    thread_id: str
    process_id: int
    extra_data: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), ensure_ascii=False, default=str)

class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured logging"""
    
    def __init__(self, country_code: str = "unknown", include_extra: bool = True):
        super().__init__()
        self.country_code = country_code
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON"""
        try:
            # Extract caller information
            frame = sys._getframe()
            while frame:
                if frame.f_code.co_filename != __file__:
                    break
                frame = frame.f_back
            
            module_name = getattr(record, 'module', 'unknown')
            function_name = getattr(record, 'funcName', 'unknown')
            line_number = getattr(record, 'lineno', 0)
            
            # Create structured log entry
            log_entry = LogEntry(
                timestamp=datetime.fromtimestamp(record.created).isoformat(),
                level=record.levelname,
                logger_name=record.name,
                message=record.getMessage(),
                country_code=self.country_code,
                module=module_name,
                function=function_name,
                line_number=line_number,
                thread_id=str(threading.current_thread().ident),
                process_id=os.getpid(),
                extra_data=getattr(record, 'extra_data', {}) if self.include_extra else {}
            )
            
            return log_entry.to_json()
            
        except Exception as e:
            # Fallback to simple formatting if structured formatting fails
            return f"{datetime.now().isoformat()} - {record.levelname} - {record.getMessage()}"

class ColoredFormatter(logging.Formatter):
    """Colored formatter for console output"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def __init__(self, country_code: str = "unknown"):
        super().__init__()
        self.country_code = country_code
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console"""
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        formatted = (
            f"{color}[{timestamp}] "
            f"{record.levelname:8} "
            f"[{self.country_code.upper()}] "
            f"{record.name}: "
            f"{record.getMessage()}{reset}"
        )
        
        # Add exception info if present
        if record.exc_info:
            formatted += f"\n{reset}{self.formatException(record.exc_info)}"
        
        return formatted

class WandBHandler(logging.Handler):
    """Custom handler for Weights & Biases logging"""
    
    def __init__(self, project: str, entity: str = None, tags: List[str] = None):
        super().__init__()
        self.project = project
        self.entity = entity
        self.tags = tags or []
        self.wandb_initialized = False
        
        if WANDB_AVAILABLE:
            try:
                wandb.init(
                    project=self.project,
                    entity=self.entity,
                    tags=self.tags,
                    reinit=True
                )
                self.wandb_initialized = True
            except Exception as e:
                print(f"Failed to initialize wandb: {e}")
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record to wandb"""
        if not self.wandb_initialized or not WANDB_AVAILABLE:
            return
        
        try:
            log_data = {
                'level': record.levelname,
                'message': record.getMessage(),
                'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'logger': record.name
            }
            
            # Add extra data if available
            if hasattr(record, 'extra_data') and record.extra_data:
                log_data.update(record.extra_data)
            
            wandb.log(log_data)
            
        except Exception as e:
            print(f"Failed to log to wandb: {e}")

class TensorBoardHandler(logging.Handler):
    """Custom handler for TensorBoard logging"""
    
    def __init__(self, log_dir: str):
        super().__init__()
        self.log_dir = log_dir
        self.writer = None
        
        if TENSORBOARD_AVAILABLE:
            try:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir)
            except Exception as e:
                print(f"Failed to initialize TensorBoard writer: {e}")
    
    def emit(self, record: logging.LogRecord) -> None:
        """Emit log record to TensorBoard"""
        if not self.writer:
            return
        
        try:
            # Log metrics if available in extra_data
            if hasattr(record, 'extra_data') and record.extra_data:
                step = record.extra_data.get('step', 0)
                
                for key, value in record.extra_data.items():
                    if isinstance(value, (int, float)) and key != 'step':
                        self.writer.add_scalar(f"logs/{key}", value, step)
            
            # Log text messages
            self.writer.add_text(
                f"logs/{record.levelname}",
                record.getMessage(),
                global_step=int(time.time())
            )
            
        except Exception as e:
            print(f"Failed to log to TensorBoard: {e}")
    
    def close(self) -> None:
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()

class PerformanceTracker:
    """Track performance metrics for logging"""
    
    def __init__(self):
        self.start_times = {}
        self.metrics = {}
    
    def start_timer(self, name: str) -> None:
        """Start timing an operation"""
        self.start_times[name] = time.time()
    
    def end_timer(self, name: str) -> float:
        """End timing and return duration"""
        if name in self.start_times:
            duration = time.time() - self.start_times[name]
            self.metrics[f"{name}_duration"] = duration
            del self.start_times[name]
            return duration
        return 0.0
    
    def add_metric(self, name: str, value: Union[int, float]) -> None:
        """Add a performance metric"""
        self.metrics[name] = value
    
    def get_metrics(self) -> Dict[str, Union[int, float]]:
        """Get all metrics"""
        return self.metrics.copy()
    
    def clear_metrics(self) -> None:
        """Clear all metrics"""
        self.metrics.clear()
        self.start_times.clear()

class DAPTLogger:
    """Main DAPT logging system"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
        self.country_code = config["country"]["code"]
        
        # Setup logging directories
        self.log_dir = Path(global_config["log_dir"]) / self.country_code
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance tracker
        self.performance_tracker = PerformanceTracker()
        
        # Initialize loggers
        self.loggers = {}
        self.handlers = {}
        
        # Setup main logger
        self.logger = self._setup_main_logger()
        
        self.logger.info(f"DAPT Logger initialized for country: {self.country_code}")
    
    def _setup_main_logger(self) -> logging.Logger:
        """Setup the main logger with all handlers"""
        logger_name = f"dapt_{self.country_code}"
        logger = logging.getLogger(logger_name)
        
        # Set log level
        log_level = getattr(logging, self.config["logging"]["level"])
        logger.setLevel(log_level)
        
        # Clear existing handlers
        logger.handlers.clear()
        
        # Console handler with colors
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_formatter = ColoredFormatter(self.country_code)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        self.handlers['console'] = console_handler
        
        # File handler with rotation
        log_file = self.log_dir / self.config["logging"].get("log_file", "dapt.log")
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_formatter = StructuredFormatter(self.country_code)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        self.handlers['file'] = file_handler
        
        # Error file handler
        error_log_file = self.log_dir / self.config["logging"].get("error_log_file", "errors.log")
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        logger.addHandler(error_handler)
        self.handlers['error'] = error_handler
        
        # WandB handler
        wandb_config = self.config["logging"].get("wandb", {})
        if wandb_config.get("enabled", False) and WANDB_AVAILABLE:
            try:
                wandb_handler = WandBHandler(
                    project=wandb_config.get("project", f"dapt-{self.country_code}"),
                    entity=wandb_config.get("entity"),
                    tags=wandb_config.get("tags", [self.country_code])
                )
                wandb_handler.setLevel(logging.INFO)
                logger.addHandler(wandb_handler)
                self.handlers['wandb'] = wandb_handler
            except Exception as e:
                logger.warning(f"Failed to setup WandB handler: {e}")
        
        # TensorBoard handler
        tensorboard_config = self.config["logging"].get("tensorboard", {})
        if tensorboard_config.get("enabled", False) and TENSORBOARD_AVAILABLE:
            try:
                tb_log_dir = tensorboard_config.get("log_dir", str(self.log_dir / "tensorboard"))
                tensorboard_handler = TensorBoardHandler(tb_log_dir)
                tensorboard_handler.setLevel(logging.INFO)
                logger.addHandler(tensorboard_handler)
                self.handlers['tensorboard'] = tensorboard_handler
            except Exception as e:
                logger.warning(f"Failed to setup TensorBoard handler: {e}")
        
        # Prevent propagation to root logger
        logger.propagate = False
        
        return logger
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get or create a named logger"""
        full_name = f"dapt_{self.country_code}.{name}"
        
        if full_name not in self.loggers:
            logger = logging.getLogger(full_name)
            logger.setLevel(self.logger.level)
            
            # Add same handlers as main logger
            for handler in self.logger.handlers:
                logger.addHandler(handler)
            
            logger.propagate = False
            self.loggers[full_name] = logger
        
        return self.loggers[full_name]
    
    def log_with_extra(self, level: str, message: str, extra_data: Dict[str, Any] = None, logger_name: str = None) -> None:
        """Log message with extra structured data"""
        logger = self.get_logger(logger_name) if logger_name else self.logger
        
        # Create log record with extra data
        record = logging.LogRecord(
            name=logger.name,
            level=getattr(logging, level.upper()),
            pathname="",
            lineno=0,
            msg=message,
            args=(),
            exc_info=None
        )
        
        if extra_data:
            record.extra_data = extra_data
        
        logger.handle(record)
    
    def log_training_metrics(self, epoch: int, step: int, metrics: Dict[str, float], phase: str = "train") -> None:
        """Log training metrics"""
        extra_data = {
            'epoch': epoch,
            'step': step,
            'phase': phase,
            **metrics
        }
        
        message = f"[{phase.upper()}] Epoch {epoch}, Step {step}: {metrics}"
        self.log_with_extra("INFO", message, extra_data, "training")
    
    def log_evaluation_results(self, dataset_name: str, metrics: Dict[str, float], model_name: str = None) -> None:
        """Log evaluation results"""
        extra_data = {
            'dataset_name': dataset_name,
            'model_name': model_name,
            'evaluation_time': datetime.now().isoformat(),
            **metrics
        }
        
        message = f"Evaluation on {dataset_name}: {metrics}"
        self.log_with_extra("INFO", message, extra_data, "evaluation")
    
    def log_performance_metrics(self, operation: str, duration: float, additional_metrics: Dict[str, Any] = None) -> None:
        """Log performance metrics"""
        extra_data = {
            'operation': operation,
            'duration_seconds': duration,
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_metrics:
            extra_data.update(additional_metrics)
        
        message = f"Performance: {operation} completed in {duration:.2f}s"
        self.log_with_extra("INFO", message, extra_data, "performance")
    
    def log_error(self, error: Exception, context: str = None, extra_data: Dict[str, Any] = None) -> None:
        """Log error with full traceback"""
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
        
        if extra_data:
            error_data.update(extra_data)
        
        message = f"Error in {context}: {str(error)}" if context else f"Error: {str(error)}"
        self.log_with_extra("ERROR", message, error_data, "error")
    
    @contextmanager
    def timer(self, operation_name: str, log_result: bool = True):
        """Context manager for timing operations"""
        start_time = time.time()
        self.performance_tracker.start_timer(operation_name)
        
        try:
            yield
        finally:
            duration = self.performance_tracker.end_timer(operation_name)
            
            if log_result:
                self.log_performance_metrics(operation_name, duration)
    
    def get_log_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of logs from the last N hours"""
        try:
            log_file = self.log_dir / self.config["logging"].get("log_file", "dapt.log")
            
            if not log_file.exists():
                return {'error': 'Log file not found'}
            
            # Count log levels in recent logs
            cutoff_time = datetime.now() - timedelta(hours=hours)
            level_counts = {'DEBUG': 0, 'INFO': 0, 'WARNING': 0, 'ERROR': 0, 'CRITICAL': 0}
            total_logs = 0
            
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        log_time = datetime.fromisoformat(log_entry['timestamp'])
                        
                        if log_time >= cutoff_time:
                            level = log_entry.get('level', 'UNKNOWN')
                            if level in level_counts:
                                level_counts[level] += 1
                            total_logs += 1
                    except (json.JSONDecodeError, KeyError, ValueError):
                        continue
            
            return {
                'time_period_hours': hours,
                'total_logs': total_logs,
                'level_counts': level_counts,
                'log_file': str(log_file),
                'summary_generated': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {'error': f'Failed to generate log summary: {str(e)}'}
    
    def cleanup_old_logs(self, days: int = 30) -> None:
        """Clean up log files older than specified days"""
        try:
            cutoff_time = datetime.now() - timedelta(days=days)
            
            for log_file in self.log_dir.glob("*.log*"):
                try:
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_time:
                        log_file.unlink()
                        self.logger.info(f"Cleaned up old log file: {log_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up {log_file}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to cleanup old logs: {e}")
    
    def close(self) -> None:
        """Close all handlers and cleanup"""
        try:
            # Close TensorBoard writer
            if 'tensorboard' in self.handlers:
                self.handlers['tensorboard'].close()
            
            # Close all file handlers
            for handler in self.logger.handlers:
                if hasattr(handler, 'close'):
                    handler.close()
            
            self.logger.info("DAPT Logger closed")
            
        except Exception as e:
            print(f"Error closing logger: {e}")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

# Global logger instance
_global_logger = None

def get_logger(name: str = None) -> logging.Logger:
    """Get global logger instance"""
    global _global_logger
    
    if _global_logger is None:
        # Create a basic logger if no global logger is set
        logger = logging.getLogger(name or "dapt")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    return _global_logger.get_logger(name) if name else _global_logger.logger

def setup_global_logger(config: Dict[str, Any], global_config: Dict[str, Any]) -> DAPTLogger:
    """Setup global logger instance"""
    global _global_logger
    _global_logger = DAPTLogger(config, global_config)
    return _global_logger

def log_training_step(epoch: int, step: int, loss: float, metrics: Dict[str, float] = None) -> None:
    """Convenience function for logging training steps"""
    logger = get_logger("training")
    
    all_metrics = {'loss': loss}
    if metrics:
        all_metrics.update(metrics)
    
    message = f"Epoch {epoch}, Step {step}: Loss={loss:.4f}"
    if metrics:
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        message += f", {metrics_str}"
    
    logger.info(message)

def log_evaluation(dataset: str, metrics: Dict[str, float]) -> None:
    """Convenience function for logging evaluation results"""
    logger = get_logger("evaluation")
    
    metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    message = f"Evaluation on {dataset}: {metrics_str}"
    
    logger.info(message)