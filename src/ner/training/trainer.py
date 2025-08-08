"""NER Trainer

Implements training logic for NER models including training loop,
validation, checkpointing, and logging.
"""

import os
import json
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
from datetime import datetime

from ..data import NERDataProcessor, NERDataLoader
from ..models import NERModel, BertNERModel
from ..evaluation import NEREvaluator, NERMetrics
from ..utils import NERLogger

class NERTrainer:
    """Main trainer class for NER models"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        """
        Initialize NER Trainer
        
        Args:
            config: Training configuration
            global_config: Global configuration
        """
        self.config = config
        self.global_config = global_config
        
        # Initialize logger (ensure exists before any method uses it)
        self.logger = NERLogger(
            name=f"{config.get('country', {}).get('code', 'unknown')}_training",
            log_dir=global_config.get('log_dir', 'data/ner/logs')
        )
        
        # Setup device
        self.device = self._setup_device()
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.data_loaders = {}
        
        # Training metrics
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'learning_rates': []
        }
        
        # Checkpointing
        self.best_val_f1 = 0.0
        self.patience_counter = 0
        self.early_stopping_patience = config.get('training', {}).get('early_stopping_patience', 5)
        
        # Output directories
        self.output_dir = Path(config.get('output', {}).get('model_dir', 'data/ner/models'))
        self.checkpoint_dir = self.output_dir / 'checkpoints'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        country_display = config.get('country', {}).get('code') or config.get('country') or 'unknown'
        self.logger.info(f"Initialized trainer for {country_display} on device: {self.device}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device"""
        hardware_config = self.config.get('hardware', {})
        device_config = hardware_config.get('device', 'auto')
        
        if device_config == 'auto':
            # Auto-select: use GPU if available, otherwise CPU
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.info(f"Using GPU (auto-selected): {torch.cuda.get_device_name()}")
            else:
                device = torch.device('cpu')
                self.logger.info("Using CPU (auto-selected, no GPU available)")
        elif device_config == 'cuda':
            # Force GPU usage
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.logger.info(f"Using GPU (forced): {torch.cuda.get_device_name()}")
            else:
                self.logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device('cpu')
        elif device_config == 'cpu':
            # Force CPU usage
            device = torch.device('cpu')
            self.logger.info("Using CPU (forced)")
        else:
            # Handle specific device like 'cuda:0'
            if device_config.startswith('cuda:') and torch.cuda.is_available():
                device = torch.device(device_config)
                self.logger.info(f"Using specific GPU device: {device_config}")
            else:
                self.logger.warning(f"Invalid device config '{device_config}', falling back to auto")
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                self.logger.info(f"Using {'GPU' if device.type == 'cuda' else 'CPU'} (fallback)")
        
        return device
    
    def prepare_data(self):
        """Prepare training data"""
        self.logger.info("Preparing training data...")
        
        data_config = self.config['data']
        
        # Initialize data processor
        processor = NERDataProcessor(self.config)
        
        # Load training data
        train_examples = processor.load_data_file(data_config['train_file'])
        self.logger.info(f"Loaded {len(train_examples)} training examples")
        
        # Load validation data if available
        val_examples = None
        if 'val_file' in data_config and data_config['val_file']:
            val_examples = processor.load_data_file(data_config['val_file'])
            self.logger.info(f"Loaded {len(val_examples)} validation examples")
        
        # Create label mappings
        labels_config = self.config['labels']
        
        # Handle both 'entities' and 'label_names' formats
        if 'entities' in labels_config:
            # Direct entities list format
            entities = labels_config['entities']
            bio_labels = ['O'] + [f'B-{entity}' for entity in entities] + [f'I-{entity}' for entity in entities]
        elif 'label_names' in labels_config:
            # BIO label names format - extract entities from BIO tags
            label_names = labels_config['label_names']
            entities = set()
            for label in label_names:
                if label.startswith('B-') or label.startswith('I-'):
                    entity = label[2:]  # Remove 'B-' or 'I-' prefix
                    entities.add(entity)
            entities = sorted(list(entities))  # Convert to sorted list for consistency
            bio_labels = label_names  # Use existing label names
        else:
            raise ValueError("Configuration must contain either 'entities' or 'label_names' in labels section")
        
        label2id = {label: idx for idx, label in enumerate(bio_labels)}
        id2label = {idx: label for label, idx in label2id.items()}
        
        self.label2id = label2id
        self.id2label = id2label
        self.num_labels = len(bio_labels)
        
        # Initialize data loader
        data_loader = NERDataLoader(
            tokenizer_name=self.config['model']['pretrained_model'],
            label2id=label2id,
            max_length=data_config.get('max_length', 512)
        )
        
        # Create data loaders
        self.data_loaders = data_loader.prepare_data_loaders(
            train_examples=train_examples,
            val_examples=val_examples,
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config.get('hardware', {}).get('num_workers', 0)
        )
        
        self.tokenizer = data_loader.get_tokenizer()
        
        self.logger.info(f"Created data loaders with {self.num_labels} labels")
    
    def prepare_model(self):
        """Prepare model for training"""
        self.logger.info("Preparing model...")
        
        model_config = self.config['model']
        
        # Initialize model
        if model_config['type'] == 'bert':
            self.model = BertNERModel.from_pretrained(
                pretrained_model_name_or_path=model_config['pretrained_model'],
                num_labels=self.num_labels,
                dropout=model_config.get('dropout', 0.1)
            )
        else:
            raise ValueError(f"Unsupported model type: {model_config['type']}")
        
        # Move model to device
        self.model.to(self.device)
        
        self.logger.info(f"Initialized {model_config['type']} model with {self.num_labels} labels")
    
    def prepare_optimizer(self):
        """Prepare optimizer and scheduler"""
        training_config = self.config['training']
        
        # Prepare optimizer
        optimizer_name = training_config.get('optimizer', 'adamw')
        learning_rate = training_config['learning_rate']
        weight_decay = training_config.get('weight_decay', 0.01)
        
        if optimizer_name.lower() == 'adamw':
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Prepare scheduler
        scheduler_name = training_config.get('scheduler', 'linear')
        num_epochs = training_config['epochs']
        num_training_steps = len(self.data_loaders['train']) * num_epochs
        warmup_steps = int(num_training_steps * training_config.get('warmup_ratio', 0.1))
        
        if scheduler_name.lower() == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=num_training_steps
            )
        elif scheduler_name.lower() == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_training_steps
            )
        else:
            self.scheduler = None
        
        self.logger.info(f"Initialized {optimizer_name} optimizer with {scheduler_name} scheduler")
    
    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.data_loaders['train'])
        
        progress_bar = tqdm(
            self.data_loaders['train'],
            desc=f"Epoch {epoch + 1}",
            leave=False
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # Update parameters
            self.optimizer.step()
            if self.scheduler:
                self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{current_lr:.2e}"
            })
            
            # Log batch metrics
            if batch_idx % 100 == 0:
                self.logger.debug(
                    f"Epoch {epoch + 1}, Batch {batch_idx}/{num_batches}, "
                    f"Loss: {loss.item():.4f}, LR: {current_lr:.2e}"
                )
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> Dict[str, float]:
        """Validate model
        
        Returns:
            Validation metrics
        """
        if 'val' not in self.data_loaders:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.data_loaders['val'], desc="Validating", leave=False):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss'] if isinstance(outputs, dict) else outputs.loss
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs.logits
                
                # Get predictions
                predictions = torch.argmax(logits, dim=-1)
                
                # Collect predictions and labels
                mask = batch['labels'] != -100
                all_predictions.extend(predictions[mask].cpu().numpy())
                all_labels.extend(batch['labels'][mask].cpu().numpy())
                
                total_loss += loss.item()
        
        # Calculate metrics
        metrics_calculator = NERMetrics(self.id2label)
        metrics = metrics_calculator.compute_metrics(all_predictions, all_labels)
        
        avg_loss = total_loss / len(self.data_loaders['val'])
        metrics['val_loss'] = avg_loss
        
        return metrics
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save training checkpoint
        
        Args:
            epoch: Current epoch
            metrics: Current metrics
            is_best: Whether this is the best checkpoint
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
            'config': self.config,
            'label2id': self.label2id,
            'id2label': self.id2label,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint with F1: {metrics.get('f1', 0):.4f}")
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "latest_checkpoint.pt"
        torch.save(checkpoint, latest_path)
    
    def save_final_model(self):
        """Save final trained model"""
        model_dir = self.output_dir / f"{self.config['country']}_model"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # Save configuration and mappings
        config_path = model_dir / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump({
                'config': self.config,
                'label2id': self.label2id,
                'id2label': self.id2label,
                'training_history': self.training_history
            }, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Saved final model to {model_dir}")
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        start_time = time.time()
        
        # Prepare components
        self.prepare_data()
        self.prepare_model()
        self.prepare_optimizer()
        
        # Training configuration
        num_epochs = self.config['training']['epochs']
        
        # Training loop
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Train epoch
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update training history
            self.training_history['train_loss'].append(train_loss)
            if val_metrics:
                self.training_history['val_loss'].append(val_metrics.get('val_loss', 0))
                self.training_history['val_f1'].append(val_metrics.get('f1', 0))
                self.training_history['val_precision'].append(val_metrics.get('precision', 0))
                self.training_history['val_recall'].append(val_metrics.get('recall', 0))
            
            if self.scheduler:
                self.training_history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Check for best model
            current_f1 = val_metrics.get('f1', 0)
            is_best = current_f1 > self.best_val_f1
            if is_best:
                self.best_val_f1 = current_f1
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(epoch, val_metrics, is_best)
            
            # Log epoch results
            epoch_time = time.time() - epoch_start_time
            log_msg = f"Epoch {epoch + 1}/{num_epochs} - "
            log_msg += f"Train Loss: {train_loss:.4f}, "
            log_msg += f"Time: {epoch_time:.2f}s"
            
            if val_metrics:
                log_msg += f", Val Loss: {val_metrics.get('val_loss', 0):.4f}"
                log_msg += f", Val F1: {val_metrics.get('f1', 0):.4f}"
                log_msg += f", Val Precision: {val_metrics.get('precision', 0):.4f}"
                log_msg += f", Val Recall: {val_metrics.get('recall', 0):.4f}"
            
            self.logger.info(log_msg)
            
            # Early stopping
            if self.patience_counter >= self.early_stopping_patience:
                self.logger.info(f"Early stopping triggered after {epoch + 1} epochs")
                break
        
        # Save final model
        self.save_final_model()
        
        # Training summary
        total_time = time.time() - start_time
        self.logger.info(f"Training completed in {total_time:.2f}s")
        self.logger.info(f"Best validation F1: {self.best_val_f1:.4f}")
    
    def resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.logger.info(f"Resuming training from {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Restore state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.training_history = checkpoint.get('training_history', self.training_history)
        self.best_val_f1 = checkpoint.get('metrics', {}).get('f1', 0)
        
        self.logger.info(f"Resumed from epoch {checkpoint['epoch'] + 1}")

class TrainingEngine:
    """High-level training engine"""
    
    def __init__(self, global_config: Dict[str, Any]):
        self.global_config = global_config
        self.logger = NERLogger(
            name="training_engine",
            log_dir=global_config.get('log_dir', 'data/ner/logs')
        )
    
    def train_model(self, country: str, config_override: Optional[Dict[str, Any]] = None) -> bool:
        """Train model for specific country
        
        Args:
            country: Country code
            config_override: Configuration overrides
            
        Returns:
            True if training successful
        """
        try:
            # Load configuration
            from ..config import ConfigManager
            
            config_manager = ConfigManager(self.global_config.get('config_dir'))
            config = config_manager.load_country_config(country)
            
            # Apply overrides
            if config_override:
                config.update(config_override)
            
            # Initialize trainer
            trainer = NERTrainer(config, self.global_config)
            
            # Start training
            trainer.train()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed for {country}: {e}")
            return False

class TrainingCallbacks:
    """Training callbacks for monitoring and control"""
    
    def __init__(self):
        self.callbacks = []
    
    def add_callback(self, callback):
        """Add training callback"""
        self.callbacks.append(callback)
    
    def on_epoch_start(self, epoch: int, trainer):
        """Called at start of epoch"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_start'):
                callback.on_epoch_start(epoch, trainer)
    
    def on_epoch_end(self, epoch: int, trainer, metrics: Dict[str, float]):
        """Called at end of epoch"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_epoch_end'):
                callback.on_epoch_end(epoch, trainer, metrics)
    
    def on_training_start(self, trainer):
        """Called at start of training"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_start'):
                callback.on_training_start(trainer)
    
    def on_training_end(self, trainer):
        """Called at end of training"""
        for callback in self.callbacks:
            if hasattr(callback, 'on_training_end'):
                callback.on_training_end(trainer)