"""Configuration Validator for DAPT

Provides comprehensive validation for DAPT configurations including:
- Schema validation
- Data consistency checks
- Parameter range validation
- Cross-field validation
"""

import re
from typing import Dict, Any, List, Tuple, Optional

class ConfigValidator:
    """Validates DAPT configuration files"""
    
    def __init__(self):
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate(self, config: Dict[str, Any]) -> Tuple[bool, List[str], List[str]]:
        """Validate complete DAPT configuration
        
        Returns:
            Tuple of (is_valid, errors, warnings)
        """
        self.errors = []
        self.warnings = []
        
        # Validate core DAPT sections only
        self._validate_country_section(config.get("country", {}))
        self._validate_model_section(config.get("model", {}))
        self._validate_training_section(config.get("training", {}))
        self._validate_data_section(config.get("data", {}))
        
        # Optional sections for DAPT
        if "output" in config:
            self._validate_output_section(config["output"])
        if "hardware" in config:
            self._validate_hardware_section(config["hardware"])
        if "logging" in config:
            self._validate_logging_section(config["logging"])
        
        # Cross-section validation for DAPT
        self._validate_cross_sections(config)
        
        return len(self.errors) == 0, self.errors, self.warnings
    
    def _validate_country_section(self, country: Dict[str, Any]) -> None:
        """Validate country configuration section"""
        required_fields = ["code", "name"]
        
        for field in required_fields:
            if not country.get(field):
                self.errors.append(f"Country section missing required field: {field}")
        
        # Validate country code format
        if "code" in country:
            code = country["code"]
            if not isinstance(code, str) or not re.match(r'^[a-z]{2,3}$', code):
                self.errors.append("Country code must be 2-3 lowercase letters")
        
        # Validate language
        if "language" in country:
            valid_languages = ["ar", "en", "fr", "es"]  # Add more as needed
            if country["language"] not in valid_languages:
                self.warnings.append(f"Language '{country['language']}' not in common list: {valid_languages}")
    
    def _validate_model_section(self, model: Dict[str, Any]) -> None:
        """Validate model configuration section for DAPT"""
        required_fields = ["base_model"]
        
        for field in required_fields:
            if field not in model:
                self.errors.append(f"Model section missing required field: {field}")
        
        # Validate max_length (optional for DAPT)
        if "max_length" in model:
            max_length = model["max_length"]
            if not isinstance(max_length, int) or max_length <= 0 or max_length > 512:
                self.errors.append("Model max_length must be a positive integer <= 512")
        
        # Validate dropout_rate (optional)
        if "dropout_rate" in model:
            dropout_rate = model["dropout_rate"]
            if not isinstance(dropout_rate, (int, float)) or dropout_rate < 0 or dropout_rate > 1:
                self.errors.append("Model dropout_rate must be a number between 0 and 1")
    
    def _validate_training_section(self, training: Dict[str, Any]) -> None:
        """Validate training configuration section"""
        required_fields = ["epochs", "batch_size", "learning_rate"]
        
        for field in required_fields:
            if field not in training:
                self.errors.append(f"Training section missing required field: {field}")
        
        # Validate epochs
        if "epochs" in training:
            epochs = training["epochs"]
            if not isinstance(epochs, int) or epochs <= 0:
                self.errors.append("Training epochs must be a positive integer")
            elif epochs > 100:
                self.warnings.append("Training epochs > 100 might be excessive")
        
        # Validate batch_size
        if "batch_size" in training:
            batch_size = training["batch_size"]
            if not isinstance(batch_size, int) or batch_size <= 0:
                self.errors.append("Training batch_size must be a positive integer")
            elif batch_size > 64:
                self.warnings.append("Large batch_size might cause memory issues")
        
        # Validate learning_rate
        if "learning_rate" in training:
            lr = training["learning_rate"]
            if not isinstance(lr, (int, float)) or lr <= 0:
                self.errors.append("Training learning_rate must be a positive number")
            elif lr > 1e-3:
                self.warnings.append("Learning rate > 1e-3 might be too high for fine-tuning")
        
        # Validate weight_decay
        if "weight_decay" in training:
            wd = training["weight_decay"]
            if not isinstance(wd, (int, float)) or wd < 0:
                self.errors.append("Training weight_decay must be a non-negative number")
        
        # Validate steps
        step_fields = ["warmup_steps", "save_steps", "eval_steps", "logging_steps"]
        for field in step_fields:
            if field in training:
                steps = training[field]
                if not isinstance(steps, int) or steps < 0:
                    self.errors.append(f"Training {field} must be a non-negative integer")
        
        # Validate gradient_accumulation_steps
        if "gradient_accumulation_steps" in training:
            gas = training["gradient_accumulation_steps"]
            if not isinstance(gas, int) or gas <= 0:
                self.errors.append("Training gradient_accumulation_steps must be a positive integer")
        
        # Validate max_grad_norm
        if "max_grad_norm" in training:
            mgn = training["max_grad_norm"]
            if not isinstance(mgn, (int, float)) or mgn <= 0:
                self.errors.append("Training max_grad_norm must be a positive number")
    
    def _validate_data_section(self, data: Dict[str, Any]) -> None:
        """Validate data configuration section for DAPT"""
        # DAPT requires plain text files
        required_fields = ["train_file", "validation_file", "test_file"]
        
        for field in required_fields:
            if not data.get(field):
                self.errors.append(f"Data section missing required field: {field}")
            else:
                file_path = data[field]
                if not isinstance(file_path, str):
                    self.errors.append(f"Data {field} must be a string")
                elif not file_path.endswith('.txt'):
                    self.warnings.append(f"Data {field} should be a .txt file for DAPT training")
        
        # Validate preprocessing section (optional for DAPT)
        if "preprocessing" in data:
            preprocessing = data["preprocessing"]
            
            # Validate boolean fields
            bool_fields = ["lowercase", "remove_diacritics", "normalize_arabic"]
            for field in bool_fields:
                if field in preprocessing and not isinstance(preprocessing[field], bool):
                    self.errors.append(f"Data preprocessing {field} must be a boolean")
            
            # Validate max_length
            if "max_length" in preprocessing:
                max_length = preprocessing["max_length"]
                if not isinstance(max_length, int) or max_length <= 0:
                    self.errors.append("Data preprocessing max_length must be a positive integer")
    
    def _validate_labels_section(self, labels: Dict[str, Any]) -> None:
        """Validate labels configuration section"""
        required_fields = ["label_names", "label_mapping"]
        
        for field in required_fields:
            if field not in labels:
                self.errors.append(f"Labels section missing required field: {field}")
                return
        
        label_names = labels["label_names"]
        label_mapping = labels["label_mapping"]
        
        # Validate label_names
        if not isinstance(label_names, list) or len(label_names) == 0:
            self.errors.append("Labels label_names must be a non-empty list")
            return
        
        # Validate label_mapping
        if not isinstance(label_mapping, dict):
            self.errors.append("Labels label_mapping must be a dictionary")
            return
        
        # Check consistency between label_names and label_mapping
        if len(label_names) != len(label_mapping):
            self.errors.append("Labels label_names and label_mapping must have the same length")
        
        # Check all label names are in mapping
        for name in label_names:
            if name not in label_mapping:
                self.errors.append(f"Label '{name}' not found in label_mapping")
        
        # Check all mapping values are valid integers
        expected_ids = set(range(len(label_names)))
        actual_ids = set(label_mapping.values())
        
        if actual_ids != expected_ids:
            self.errors.append(f"Label mapping IDs should be consecutive integers from 0 to {len(label_names)-1}")
        
        # Validate BIO tagging scheme if applicable
        if any(name.startswith(('B-', 'I-')) for name in label_names):
            self._validate_bio_scheme(label_names)
    
    def _validate_bio_scheme(self, label_names: List[str]) -> None:
        """Validate BIO tagging scheme consistency"""
        entities = set()
        
        for label in label_names:
            if label.startswith('B-'):
                entity = label[2:]
                entities.add(entity)
            elif label.startswith('I-'):
                entity = label[2:]
                entities.add(entity)
        
        # Check that every entity has both B- and I- tags
        for entity in entities:
            b_tag = f"B-{entity}"
            i_tag = f"I-{entity}"
            
            if b_tag not in label_names:
                self.warnings.append(f"Missing B-tag for entity: {entity}")
            if i_tag not in label_names:
                self.warnings.append(f"Missing I-tag for entity: {entity}")
    
    def _validate_evaluation_section(self, evaluation: Dict[str, Any]) -> None:
        """Validate evaluation configuration section"""
        if "metrics" in evaluation:
            metrics = evaluation["metrics"]
            if not isinstance(metrics, list):
                self.errors.append("Evaluation metrics must be a list")
            else:
                valid_metrics = ["accuracy", "f1", "precision", "recall", "classification_report"]
                for metric in metrics:
                    if metric not in valid_metrics:
                        self.warnings.append(f"Unknown metric: {metric}. Valid metrics: {valid_metrics}")
        
        if "average" in evaluation:
            average = evaluation["average"]
            valid_averages = ["micro", "macro", "weighted", "binary"]
            if average not in valid_averages:
                self.errors.append(f"Evaluation average must be one of: {valid_averages}")
    
    def _validate_output_section(self, output: Dict[str, Any]) -> None:
        """Validate output configuration section"""
        if "save_total_limit" in output:
            limit = output["save_total_limit"]
            if not isinstance(limit, int) or limit < 1:
                self.errors.append("Output save_total_limit must be a positive integer")
        
        bool_fields = ["save_best_model", "load_best_model_at_end"]
        for field in bool_fields:
            if field in output and not isinstance(output[field], bool):
                self.errors.append(f"Output {field} must be a boolean")
    
    def _validate_hardware_section(self, hardware: Dict[str, Any]) -> None:
        """Validate hardware configuration section"""
        if "device" in hardware:
            device = hardware["device"]
            valid_devices = ["auto", "cpu", "cuda", "mps"]
            if device not in valid_devices and not device.startswith("cuda:"):
                self.warnings.append(f"Device '{device}' might not be valid. Common values: {valid_devices}")
        
        if "fp16" in hardware and not isinstance(hardware["fp16"], bool):
            self.errors.append("Hardware fp16 must be a boolean")
        
        if "dataloader_num_workers" in hardware:
            workers = hardware["dataloader_num_workers"]
            if not isinstance(workers, int) or workers < 0:
                self.errors.append("Hardware dataloader_num_workers must be a non-negative integer")
    
    def _validate_logging_section(self, logging: Dict[str, Any]) -> None:
        """Validate logging configuration section"""
        if "level" in logging:
            level = logging["level"]
            valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if level not in valid_levels:
                self.errors.append(f"Logging level must be one of: {valid_levels}")
    
    def _validate_cross_sections(self, config: Dict[str, Any]) -> None:
        """Validate consistency across different sections for DAPT"""
        # Check data preprocessing max_length matches model max_length
        if "model" in config and "data" in config:
            model_max_length = config["model"].get("max_length")
            data_preprocessing = config["data"].get("preprocessing", {})
            data_max_length = data_preprocessing.get("max_length")
            
            if model_max_length and data_max_length and model_max_length != data_max_length:
                self.warnings.append(
                    f"Model max_length ({model_max_length}) differs from data preprocessing max_length ({data_max_length})"
                )
        
        # Check training steps consistency
        if "training" in config:
            training = config["training"]
            save_steps = training.get("save_steps")
            eval_steps = training.get("eval_steps")
            logging_steps = training.get("logging_steps")
            
            if save_steps and eval_steps and save_steps < eval_steps:
                self.warnings.append("save_steps is less than eval_steps - you might lose the best model")
            
            if logging_steps and eval_steps and logging_steps > eval_steps:
                self.warnings.append("logging_steps is greater than eval_steps - you might miss evaluation logs")
    
    def get_validation_report(self) -> str:
        """Generate a formatted validation report"""
        report = []
        
        if self.errors:
            report.append("ERRORS:")
            for error in self.errors:
                report.append(f"  ❌ {error}")
            report.append("")
        
        if self.warnings:
            report.append("WARNINGS:")
            for warning in self.warnings:
                report.append(f"  ⚠️  {warning}")
            report.append("")
        
        if not self.errors and not self.warnings:
            report.append("✅ Configuration validation passed with no issues")
        elif not self.errors:
            report.append("✅ Configuration is valid (with warnings)")
        else:
            report.append("❌ Configuration validation failed")
        
        return "\n".join(report)