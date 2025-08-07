"""NER Logging Utilities

Provides logging configuration and utilities for the NER system.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
from pathlib import Path

class NERLogger:
    """NER-specific logger with enhanced functionality"""
    
    def __init__(
        self,
        name: str = "ner",
        level: str = "INFO",
        log_dir: Optional[str] = None,
        log_to_file: bool = True,
        log_to_console: bool = True,
        max_file_size: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5
    ):
        """
        Initialize NER logger
        
        Args:
            name: Logger name
            level: Logging level
            log_dir: Directory for log files
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
            max_file_size: Maximum log file size in bytes
            backup_count: Number of backup files to keep
        """
        self.name = name
        self.level = getattr(logging, level.upper())
        self.log_dir = log_dir or "logs"
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Setup handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # Console handler
        if self.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.log_to_file:
            # Create log directory
            os.makedirs(self.log_dir, exist_ok=True)
            
            # Create rotating file handler
            from logging.handlers import RotatingFileHandler
            
            log_file = os.path.join(
                self.log_dir, 
                f"{self.name}_{datetime.now().strftime('%Y%m%d')}.log"
            )
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=self.max_file_size,
                backupCount=self.backup_count
            )
            file_handler.setLevel(self.level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self.logger.critical(message, **kwargs)
    
    def log_training_step(
        self, 
        epoch: int, 
        step: int, 
        loss: float, 
        lr: float,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Log training step information
        
        Args:
            epoch: Current epoch
            step: Current step
            loss: Training loss
            lr: Learning rate
            metrics: Additional metrics
        """
        message = f"Epoch {epoch}, Step {step}: Loss={loss:.4f}, LR={lr:.6f}"
        
        if metrics:
            metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            message += f", {metric_str}"
        
        self.info(message)
    
    def log_evaluation(
        self, 
        dataset_name: str, 
        metrics: Dict[str, float],
        epoch: Optional[int] = None
    ):
        """Log evaluation results
        
        Args:
            dataset_name: Name of the dataset
            metrics: Evaluation metrics
            epoch: Current epoch (if applicable)
        """
        epoch_str = f"Epoch {epoch} - " if epoch is not None else ""
        metric_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
        
        message = f"{epoch_str}Evaluation on {dataset_name}: {metric_str}"
        self.info(message)
    
    def log_model_info(
        self, 
        model_name: str, 
        num_parameters: int,
        model_config: Optional[Dict[str, Any]] = None
    ):
        """Log model information
        
        Args:
            model_name: Name of the model
            num_parameters: Number of parameters
            model_config: Model configuration
        """
        message = f"Model: {model_name}, Parameters: {num_parameters:,}"
        
        if model_config:
            config_str = ", ".join([f"{k}={v}" for k, v in model_config.items()])
            message += f", Config: {config_str}"
        
        self.info(message)
    
    def log_data_info(
        self, 
        dataset_name: str, 
        num_samples: int,
        num_labels: int,
        label_distribution: Optional[Dict[str, int]] = None
    ):
        """Log dataset information
        
        Args:
            dataset_name: Name of the dataset
            num_samples: Number of samples
            num_labels: Number of unique labels
            label_distribution: Distribution of labels
        """
        message = f"Dataset: {dataset_name}, Samples: {num_samples:,}, Labels: {num_labels}"
        
        if label_distribution:
            # Show top 5 most common labels
            sorted_labels = sorted(label_distribution.items(), key=lambda x: x[1], reverse=True)
            top_labels = sorted_labels[:5]
            label_str = ", ".join([f"{label}:{count}" for label, count in top_labels])
            message += f", Top labels: {label_str}"
        
        self.info(message)
    
    def log_config(self, config: Dict[str, Any]):
        """Log configuration information
        
        Args:
            config: Configuration dictionary
        """
        self.info("Configuration:")
        for key, value in config.items():
            if isinstance(value, dict):
                self.info(f"  {key}:")
                for sub_key, sub_value in value.items():
                    self.info(f"    {sub_key}: {sub_value}")
            else:
                self.info(f"  {key}: {value}")
    
    def log_exception(self, exception: Exception, context: str = ""):
        """Log exception with context
        
        Args:
            exception: Exception to log
            context: Additional context
        """
        import traceback
        
        context_str = f" in {context}" if context else ""
        self.error(f"Exception{context_str}: {str(exception)}")
        self.error(f"Traceback: {traceback.format_exc()}")

def setup_logging(
    level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
    log_to_console: bool = True,
    logger_name: str = "ner"
) -> NERLogger:
    """Setup logging for NER system
    
    Args:
        level: Logging level
        log_dir: Directory for log files
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
        logger_name: Name of the logger
        
    Returns:
        Configured NER logger
    """
    return NERLogger(
        name=logger_name,
        level=level,
        log_dir=log_dir,
        log_to_file=log_to_file,
        log_to_console=log_to_console
    )

def get_logger(name: str = "ner") -> logging.Logger:
    """Get logger instance
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)

class TrainingLogger:
    """Specialized logger for training processes"""
    
    def __init__(self, logger: NERLogger, log_interval: int = 100):
        """
        Initialize training logger
        
        Args:
            logger: Base NER logger
            log_interval: Interval for logging training steps
        """
        self.logger = logger
        self.log_interval = log_interval
        self.step_count = 0
        self.epoch_losses = []
        self.best_metrics = {}
    
    def log_step(
        self, 
        epoch: int, 
        loss: float, 
        lr: float,
        metrics: Optional[Dict[str, float]] = None
    ):
        """Log training step
        
        Args:
            epoch: Current epoch
            loss: Training loss
            lr: Learning rate
            metrics: Additional metrics
        """
        self.step_count += 1
        self.epoch_losses.append(loss)
        
        if self.step_count % self.log_interval == 0:
            avg_loss = sum(self.epoch_losses[-self.log_interval:]) / min(len(self.epoch_losses), self.log_interval)
            self.logger.log_training_step(epoch, self.step_count, avg_loss, lr, metrics)
    
    def log_epoch_end(
        self, 
        epoch: int, 
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log end of epoch
        
        Args:
            epoch: Current epoch
            train_metrics: Training metrics
            val_metrics: Validation metrics
        """
        # Calculate average epoch loss
        avg_loss = sum(self.epoch_losses) / len(self.epoch_losses) if self.epoch_losses else 0.0
        
        # Log training metrics
        train_metrics['avg_loss'] = avg_loss
        self.logger.log_evaluation("Training", train_metrics, epoch)
        
        # Log validation metrics
        if val_metrics:
            self.logger.log_evaluation("Validation", val_metrics, epoch)
            
            # Check for best metrics
            for metric, value in val_metrics.items():
                if metric not in self.best_metrics or value > self.best_metrics[metric]:
                    self.best_metrics[metric] = value
                    self.logger.info(f"New best {metric}: {value:.4f}")
        
        # Reset epoch losses
        self.epoch_losses = []
    
    def log_training_complete(
        self, 
        total_epochs: int, 
        total_time: float,
        final_metrics: Dict[str, float]
    ):
        """Log training completion
        
        Args:
            total_epochs: Total number of epochs
            total_time: Total training time in seconds
            final_metrics: Final evaluation metrics
        """
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        self.logger.info(
            f"Training completed: {total_epochs} epochs in {hours:02d}:{minutes:02d}:{seconds:02d}"
        )
        
        # Log best metrics
        if self.best_metrics:
            best_str = ", ".join([f"{k}={v:.4f}" for k, v in self.best_metrics.items()])
            self.logger.info(f"Best metrics: {best_str}")
        
        # Log final metrics
        if final_metrics:
            final_str = ", ".join([f"{k}={v:.4f}" for k, v in final_metrics.items()])
            self.logger.info(f"Final metrics: {final_str}")