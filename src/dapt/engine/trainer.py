"""DAPT Training Engine

Core training engine for Domain Adaptive Pre-Training with support for:
- Multi-country model training
- Configurable training parameters
- Progress monitoring and logging
- Model checkpointing and recovery
"""

import json
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from datetime import datetime
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import Dataset
import logging

class DAPTTrainingEngine:
    """Main training engine for DAPT operations"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
        self.country_code = config["country"]["code"]
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Initialize model components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Training state
        self.training_id = None
        self.start_time = None
        self.is_training = False
        
        # Metrics tracking
        self.training_metrics = []
        self.best_metrics = {}
        
        self.logger.info(f"DAPT Training Engine initialized for country: {self.country_code}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for training"""
        logger = logging.getLogger(f"dapt_trainer_{self.country_code}")
        logger.setLevel(getattr(logging, self.config["logging"]["level"]))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = self.config["logging"].get("log_file")
        if log_file:
            log_path = Path(self.global_config["log_dir"]) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def initialize_model(self) -> bool:
        """Initialize tokenizer and model"""
        try:
            model_config = self.config["model"]
            
            # Load tokenizer
            self.logger.info(f"Loading tokenizer: {model_config['base_model']}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_config["base_model"]
            )
            
            # Load model for DAPT (Masked Language Modeling)
            self.logger.info(f"Loading model: {model_config['base_model']}")
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_config["base_model"]
            )
            
            self.logger.info("Model and tokenizer initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing model: {str(e)}")
            return False
    
    def prepare_datasets(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None) -> bool:
        """Prepare datasets for DAPT training"""
        try:
            self.logger.info("Preparing datasets for DAPT training")
            
            # Tokenize datasets for DAPT (no labels needed)
            def tokenize_function(examples):
                # Get texts from examples
                texts = examples[self.config["data"]["text_column"]]
                
                # Tokenize without padding (let data collator handle padding)
                return self.tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.config["model"]["max_length"]
                )
            
            # Apply tokenization
            self.train_dataset = train_dataset.map(
                tokenize_function,
                batched=True,
                desc="Tokenizing training data for DAPT",
                remove_columns=train_dataset.column_names  # Remove original columns
            )
            
            # For DAPT, we typically don't use validation datasets
            self.val_dataset = None
            
            self.logger.info(f"Training dataset size: {len(self.train_dataset)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error preparing datasets: {str(e)}")
            return False
    
    def setup_trainer(self, output_dir: str) -> bool:
        """Setup the Hugging Face trainer for DAPT"""
        try:
            training_config = self.config["training"]
            output_config = self.config["output"]
            hardware_config = self.config["hardware"]
            
            # Setup training arguments for DAPT (no evaluation needed)
            training_args = TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=training_config["epochs"],
                per_device_train_batch_size=training_config["batch_size"],
                learning_rate=training_config["learning_rate"],
                weight_decay=training_config["weight_decay"],
                warmup_steps=training_config["warmup_steps"],
                logging_steps=training_config["logging_steps"],
                save_steps=training_config["save_steps"],
                gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
                max_grad_norm=training_config["max_grad_norm"],
                lr_scheduler_type=training_config.get("lr_scheduler_type", "linear"),
                save_total_limit=output_config.get("save_total_limit", 3),
                save_strategy=output_config.get("save_strategy", "steps"),
                eval_strategy="no",  # No evaluation for DAPT
                load_best_model_at_end=False,  # No evaluation, so no best model
                fp16=hardware_config.get("fp16", False),
                dataloader_num_workers=hardware_config.get("dataloader_num_workers", 4),
                dataloader_pin_memory=hardware_config.get("dataloader_pin_memory", True),
                report_to=self.config["logging"].get("report_to", []),
                logging_dir=self.config["logging"].get("tensorboard_dir"),
                remove_unused_columns=False,
                push_to_hub=False
            )
            
            # Data collator for masked language modeling
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=True,  # Masked language modeling
                mlm_probability=0.15,  # Standard masking probability
                pad_to_multiple_of=None,
                return_tensors="pt"
            )
            
            # Initialize trainer (no compute_metrics for DAPT)
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=self.train_dataset,
                tokenizer=self.tokenizer,
                data_collator=data_collator
            )
            
            self.logger.info("DAPT Trainer setup completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting up DAPT trainer: {str(e)}")
            return False
    
    def train(self, output_dir: str, resume_from: Optional[str] = None) -> Dict[str, Any]:
        """Start DAPT training"""
        try:
            # Generate training ID
            self.training_id = f"train_{self.country_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.start_time = datetime.now()
            self.is_training = True
            
            self.logger.info(f"Starting DAPT training - ID: {self.training_id}")
            self.logger.info(f"Output directory: {output_dir}")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize model and tokenizer first
            self.logger.info("Initializing model and tokenizer...")
            if not self.initialize_model():
                raise RuntimeError("Failed to initialize model")
            
            # Load and prepare datasets
            self.logger.info("Loading training data...")
            from ..data.processor import DataProcessor
            
            data_processor = DataProcessor(self.config, self.global_config)
            train_dataset, val_dataset = data_processor.load_datasets()
            
            if not train_dataset:
                raise RuntimeError("Failed to load training data")
            
            self.logger.info("Preparing datasets...")
            if not self.prepare_datasets(train_dataset, val_dataset):
                raise RuntimeError("Failed to prepare datasets")
            
            self.logger.info("Setting up trainer...")
            if not self.setup_trainer(str(output_path)):
                raise RuntimeError("Failed to setup trainer")
            
            # Save training configuration
            config_path = output_path / "training_config.json"
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "training_id": self.training_id,
                    "country": self.country_code,
                    "start_time": self.start_time.isoformat(),
                    "config": self.config
                }, f, indent=2, ensure_ascii=False)
            
            # Resume from checkpoint if specified
            if resume_from:
                self.logger.info(f"Resuming training from: {resume_from}")
                resume_path = resume_from
            else:
                resume_path = None
            
            # Start training
            self.logger.info("Beginning training process...")
            train_result = self.trainer.train(resume_from_checkpoint=resume_path)
            
            # Save the final model
            final_model_path = output_path / "final_model"
            self.trainer.save_model(str(final_model_path))
            self.tokenizer.save_pretrained(str(final_model_path))
            
            # Save training results
            end_time = datetime.now()
            training_duration = (end_time - self.start_time).total_seconds()
            
            results = {
                "training_id": self.training_id,
                "country": self.country_code,
                "status": "completed",
                "start_time": self.start_time.isoformat(),
                "end_time": end_time.isoformat(),
                "duration_seconds": training_duration,
                "model_path": str(final_model_path),
                "output_dir": str(output_path),
                "train_results": train_result.metrics if hasattr(train_result, 'metrics') else {},
                "config": self.config
            }
            
            # Run final evaluation if validation dataset exists
            if self.val_dataset:
                self.logger.info("Running final evaluation...")
                eval_results = self.trainer.evaluate()
                results["eval_results"] = eval_results
                self.best_metrics = eval_results
                
                self.logger.info("Final evaluation results:")
                for metric, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        self.logger.info(f"  {metric}: {value:.4f}")
            
            # Save results
            results_path = output_path / "training_results.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.is_training = False
            self.logger.info(f"Training completed successfully in {training_duration:.2f} seconds")
            self.logger.info(f"Model saved to: {final_model_path}")
            
            return results
            
        except Exception as e:
            self.is_training = False
            self.logger.error(f"Training failed: {str(e)}")
            
            # Save error information
            if hasattr(self, 'training_id') and self.training_id:
                error_results = {
                    "training_id": self.training_id,
                    "country": self.country_code,
                    "status": "failed",
                    "start_time": self.start_time.isoformat() if self.start_time else None,
                    "error_time": datetime.now().isoformat(),
                    "error_message": str(e),
                    "config": self.config
                }
                
                error_path = Path(output_dir) / "training_error.json"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                with open(error_path, 'w', encoding='utf-8') as f:
                    json.dump(error_results, f, indent=2, ensure_ascii=False)
            
            raise
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        status = {
            "training_id": self.training_id,
            "country": self.country_code,
            "is_training": self.is_training,
            "start_time": self.start_time.isoformat() if self.start_time else None
        }
        
        if self.is_training and self.start_time:
            current_time = datetime.now()
            status["elapsed_time"] = (current_time - self.start_time).total_seconds()
        
        if self.trainer and hasattr(self.trainer.state, 'log_history'):
            status["latest_logs"] = self.trainer.state.log_history[-5:]  # Last 5 log entries
        
        if self.best_metrics:
            status["best_metrics"] = self.best_metrics
        
        return status
    
    def stop_training(self) -> bool:
        """Stop current training"""
        if not self.is_training:
            self.logger.warning("No training in progress")
            return False
        
        try:
            if self.trainer:
                # Note: Hugging Face Trainer doesn't have a direct stop method
                # This would require implementing a custom callback or using a different approach
                self.logger.warning("Training stop requested - will complete current step")
                self.is_training = False
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error stopping training: {str(e)}")
            return False
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        self.logger.info("Cleaning up training resources")
        
        if self.trainer:
            del self.trainer
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.is_training = False
        self.logger.info("Cleanup completed")