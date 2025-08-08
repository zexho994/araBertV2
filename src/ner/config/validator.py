"""Configuration Validator for NER System

Provides comprehensive validation for NER configuration files.
Ensures configuration integrity and consistency.
"""

import re
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

class ConfigValidator:
    """Validates NER configuration files and settings"""
    
    # Valid model types
    VALID_MODEL_TYPES = ['bert', 'distilbert', 'roberta', 'albert']
    
    # Valid optimizers
    VALID_OPTIMIZERS = ['adam', 'adamw', 'sgd', 'rmsprop']
    
    # Valid schedulers
    VALID_SCHEDULERS = ['linear', 'cosine', 'polynomial', 'constant']
    
    # Valid log levels
    VALID_LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    
    # Valid evaluation metrics
    VALID_METRICS = ['precision', 'recall', 'f1', 'accuracy', 'entity_f1']
    
    def __init__(self):
        """Initialize the configuration validator"""
        self.errors = []
        self.warnings = []
    
    def validate_config(self, config: Dict[str, Any], country: str = None) -> bool:
        """Validate a complete configuration
        
        Args:
            config: Configuration dictionary to validate
            country: Country code for context (optional)
            
        Returns:
            True if configuration is valid, False otherwise
        """
        self.errors.clear()
        self.warnings.clear()
        
        # Validate each section
        self._validate_country_section(config.get('country', {}), country)
        self._validate_model_section(config.get('model', {}))
        self._validate_training_section(config.get('training', {}))
        self._validate_data_section(config.get('data', {}))
        self._validate_labels_section(config.get('labels', {}))
        self._validate_evaluation_section(config.get('evaluation', {}))
        self._validate_output_section(config.get('output', {}))
        self._validate_hardware_section(config.get('hardware', {}))
        self._validate_logging_section(config.get('logging', {}))
        
        # Cross-section validation
        self._validate_cross_sections(config)
        
        return len(self.errors) == 0
    
    def get_errors(self) -> List[str]:
        """Get validation errors"""
        return self.errors.copy()
    
    def get_warnings(self) -> List[str]:
        """Get validation warnings"""
        return self.warnings.copy()
    
    def _validate_country_section(self, country_config: Dict[str, Any], country: str = None):
        """Validate country configuration section"""
        required_fields = ['code', 'name']
        
        for field in required_fields:
            if field not in country_config:
                self.errors.append(f"Country section missing required field: {field}")
        
        if 'code' in country_config:
            code = country_config['code']
            # Allow lowercase letters, digits, underscore, hyphen; length 2-20
            if not isinstance(code, str) or not re.match(r'^[a-z0-9_-]{2,20}$', code):
                self.errors.append("Country code must be 2-20 characters: lowercase letters, digits, '_' or '-'")
            
            if country and code != country:
                self.warnings.append(f"Country code '{code}' doesn't match expected '{country}'")
        
        if 'name' in country_config:
            name = country_config['name']
            if not isinstance(name, str) or len(name.strip()) == 0:
                self.errors.append("Country name must be a non-empty string")
    
    def _validate_model_section(self, model_config: Dict[str, Any]):
        """Validate model configuration section"""
        required_fields = ['name', 'type', 'pretrained_model']
        
        for field in required_fields:
            if field not in model_config:
                self.errors.append(f"Model section missing required field: {field}")
        
        if 'type' in model_config:
            model_type = model_config['type']
            if model_type not in self.VALID_MODEL_TYPES:
                self.errors.append(
                    f"Invalid model type '{model_type}'. Valid types: {self.VALID_MODEL_TYPES}"
                )
        
        if 'pretrained_model' in model_config:
            pretrained = model_config['pretrained_model']
            if not isinstance(pretrained, str) or len(pretrained.strip()) == 0:
                self.errors.append("Pretrained model must be a non-empty string")
        
        # Validate optional fields
        if 'max_length' in model_config:
            max_length = model_config['max_length']
            if not isinstance(max_length, int) or max_length <= 0 or max_length > 2048:
                self.errors.append("Model max_length must be a positive integer <= 2048")
        
        if 'dropout' in model_config:
            dropout = model_config['dropout']
            if not isinstance(dropout, (int, float)) or dropout < 0 or dropout > 1:
                self.errors.append("Model dropout must be a number between 0 and 1")
    
    def _validate_training_section(self, training_config: Dict[str, Any]):
        """Validate training configuration section"""
        required_fields = ['epochs', 'batch_size', 'learning_rate']
        
        for field in required_fields:
            if field not in training_config:
                self.errors.append(f"Training section missing required field: {field}")
        
        if 'epochs' in training_config:
            epochs = training_config['epochs']
            if not isinstance(epochs, int) or epochs <= 0:
                self.errors.append("Training epochs must be a positive integer")
        
        if 'batch_size' in training_config:
            batch_size = training_config['batch_size']
            if not isinstance(batch_size, int) or batch_size <= 0:
                self.errors.append("Training batch_size must be a positive integer")
        
        if 'learning_rate' in training_config:
            lr = training_config['learning_rate']
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.errors.append("Training learning_rate must be a positive number")
        
        # Validate optional fields
        if 'optimizer' in training_config:
            optimizer = training_config['optimizer']
            if optimizer not in self.VALID_OPTIMIZERS:
                self.errors.append(
                    f"Invalid optimizer '{optimizer}'. Valid optimizers: {self.VALID_OPTIMIZERS}"
                )
        
        if 'scheduler' in training_config:
            scheduler = training_config['scheduler']
            if scheduler not in self.VALID_SCHEDULERS:
                self.errors.append(
                    f"Invalid scheduler '{scheduler}'. Valid schedulers: {self.VALID_SCHEDULERS}"
                )
        
        if 'warmup_steps' in training_config:
            warmup = training_config['warmup_steps']
            if not isinstance(warmup, int) or warmup < 0:
                self.errors.append("Training warmup_steps must be a non-negative integer")
        
        if 'weight_decay' in training_config:
            weight_decay = training_config['weight_decay']
            if not isinstance(weight_decay, (int, float)) or weight_decay < 0:
                self.errors.append("Training weight_decay must be a non-negative number")
    
    def _validate_data_section(self, data_config: Dict[str, Any]):
        """Validate data configuration section"""
        required_fields = ['train_file', 'val_file']
        
        for field in required_fields:
            if field not in data_config:
                self.errors.append(f"Data section missing required field: {field}")
        
        # Validate file paths (basic validation)
        for file_field in ['train_file', 'val_file', 'test_file']:
            if file_field in data_config:
                file_path = data_config[file_field]
                if not isinstance(file_path, str) or len(file_path.strip()) == 0:
                    self.errors.append(f"Data {file_field} must be a non-empty string")
        
        # Validate preprocessing options
        if 'preprocessing' in data_config:
            preprocessing = data_config['preprocessing']
            if not isinstance(preprocessing, dict):
                self.errors.append("Data preprocessing must be a dictionary")
            else:
                if 'lowercase' in preprocessing:
                    if not isinstance(preprocessing['lowercase'], bool):
                        self.errors.append("Preprocessing lowercase must be a boolean")
                
                if 'remove_diacritics' in preprocessing:
                    if not isinstance(preprocessing['remove_diacritics'], bool):
                        self.errors.append("Preprocessing remove_diacritics must be a boolean")
    
    def _validate_labels_section(self, labels_config: Dict[str, Any]):
        """Validate labels configuration section"""
        required_fields = ['num_labels', 'label_names', 'label_mapping']
        
        for field in required_fields:
            if field not in labels_config:
                self.errors.append(f"Labels section missing required field: {field}")
                return
        
        num_labels = labels_config['num_labels']
        label_names = labels_config['label_names']
        label_mapping = labels_config['label_mapping']
        
        # Validate num_labels
        if not isinstance(num_labels, int) or num_labels <= 0:
            self.errors.append("Labels num_labels must be a positive integer")
            return
        
        # Validate label_names
        if not isinstance(label_names, list):
            self.errors.append("Labels label_names must be a list")
            return
        
        if len(label_names) != num_labels:
            self.errors.append(
                f"Number of label names ({len(label_names)}) doesn't match num_labels ({num_labels})"
            )
        
        # Check for duplicate labels
        if len(set(label_names)) != len(label_names):
            self.errors.append("Label names contain duplicates")
        
        # Validate label_mapping
        if not isinstance(label_mapping, dict):
            self.errors.append("Labels label_mapping must be a dictionary")
            return
        
        if len(label_mapping) != num_labels:
            self.errors.append(
                f"Number of label mappings ({len(label_mapping)}) doesn't match num_labels ({num_labels})"
            )
        
        # Validate mapping consistency
        for label_name in label_names:
            if label_name not in label_mapping:
                self.errors.append(f"Label '{label_name}' missing from label_mapping")
        
        # Validate mapping values are unique integers
        mapping_values = list(label_mapping.values())
        if not all(isinstance(v, int) for v in mapping_values):
            self.errors.append("All label mapping values must be integers")
        
        if len(set(mapping_values)) != len(mapping_values):
            self.errors.append("Label mapping values contain duplicates")
        
        # Check if mapping values are in valid range
        expected_values = set(range(num_labels))
        actual_values = set(mapping_values)
        if actual_values != expected_values:
            self.errors.append(
                f"Label mapping values must be exactly {{0, 1, ..., {num_labels-1}}}"
            )
    
    def _validate_evaluation_section(self, eval_config: Dict[str, Any]):
        """Validate evaluation configuration section"""
        if 'metrics' in eval_config:
            metrics = eval_config['metrics']
            if not isinstance(metrics, list):
                self.errors.append("Evaluation metrics must be a list")
            else:
                for metric in metrics:
                    if metric not in self.VALID_METRICS:
                        self.errors.append(
                            f"Invalid metric '{metric}'. Valid metrics: {self.VALID_METRICS}"
                        )
        
        if 'save_predictions' in eval_config:
            if not isinstance(eval_config['save_predictions'], bool):
                self.errors.append("Evaluation save_predictions must be a boolean")
    
    def _validate_output_section(self, output_config: Dict[str, Any]):
        """Validate output configuration section"""
        if 'model_dir' in output_config:
            model_dir = output_config['model_dir']
            if not isinstance(model_dir, str) or len(model_dir.strip()) == 0:
                self.errors.append("Output model_dir must be a non-empty string")
        
        if 'save_steps' in output_config:
            save_steps = output_config['save_steps']
            if not isinstance(save_steps, int) or save_steps <= 0:
                self.errors.append("Output save_steps must be a positive integer")
    
    def _validate_hardware_section(self, hardware_config: Dict[str, Any]):
        """Validate hardware configuration section"""
        if 'device' in hardware_config:
            device = hardware_config['device']
            if device not in ['auto', 'cpu', 'cuda', 'mps']:
                self.errors.append("Hardware device must be 'auto', 'cpu', 'cuda', or 'mps'")
        
        if 'mixed_precision' in hardware_config:
            if not isinstance(hardware_config['mixed_precision'], bool):
                self.errors.append("Hardware mixed_precision must be a boolean")
        
        if 'dataloader_num_workers' in hardware_config:
            workers = hardware_config['dataloader_num_workers']
            if not isinstance(workers, int) or workers < 0:
                self.errors.append("Hardware dataloader_num_workers must be a non-negative integer")
    
    def _validate_logging_section(self, logging_config: Dict[str, Any]):
        """Validate logging configuration section"""
        if 'level' in logging_config:
            level = logging_config['level']
            if level not in self.VALID_LOG_LEVELS:
                self.errors.append(
                    f"Invalid log level '{level}'. Valid levels: {self.VALID_LOG_LEVELS}"
                )
        
        if 'log_file' in logging_config:
            log_file = logging_config['log_file']
            if not isinstance(log_file, str) or len(log_file.strip()) == 0:
                self.errors.append("Logging log_file must be a non-empty string")
        
        # Validate wandb config
        if 'wandb' in logging_config:
            wandb_config = logging_config['wandb']
            if not isinstance(wandb_config, dict):
                self.errors.append("Logging wandb must be a dictionary")
            else:
                if 'enabled' in wandb_config:
                    if not isinstance(wandb_config['enabled'], bool):
                        self.errors.append("Logging wandb enabled must be a boolean")
    
    def _validate_cross_sections(self, config: Dict[str, Any]):
        """Validate consistency across configuration sections"""
        # Check if model max_length is compatible with training batch_size
        if 'model' in config and 'training' in config:
            model_config = config['model']
            training_config = config['training']
            
            if 'max_length' in model_config and 'batch_size' in training_config:
                max_length = model_config['max_length']
                batch_size = training_config['batch_size']
                
                # Warn if memory usage might be high
                estimated_memory = max_length * batch_size
                if estimated_memory > 100000:  # Arbitrary threshold
                    self.warnings.append(
                        f"High memory usage expected: max_length ({max_length}) * "
                        f"batch_size ({batch_size}) = {estimated_memory}"
                    )
        
        # Check if output directories are consistent
        if 'output' in config and 'logging' in config:
            output_config = config['output']
            logging_config = config['logging']
            
            if 'model_dir' in output_config and 'log_file' in logging_config:
                model_dir = Path(output_config['model_dir'])
                log_file = Path(logging_config['log_file'])
                
                # Warn if they're in completely different locations
                if not str(log_file).startswith(str(model_dir.parent)):
                    self.warnings.append(
                        "Model directory and log file are in different locations"
                    )