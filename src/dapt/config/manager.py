"""Configuration Manager for DAPT

Manages country-specific configurations for DAPT training including:
- Loading and saving configuration files
- Template management
- Configuration validation
- Multi-country support
"""

import json
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

class ConfigManager:
    """Manages DAPT configurations for multiple countries"""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize templates directory
        self.templates_dir = self.config_dir / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        
        # Initialize country configs directory
        self.countries_dir = self.config_dir / "countries"
        self.countries_dir.mkdir(exist_ok=True)
        
        # Create default templates if they don't exist
        self._create_default_templates()
    
    def _create_default_templates(self) -> None:
        """Create default configuration templates"""
        default_template = {
            "country": {
                "code": "",
                "name": "",
                "language": "ar",
                "script": "arabic"
            },
            "model": {
                "base_model": "aubmindlab/bert-base-arabertv2",
                "max_length": 128,
                "num_labels": 9,
                "dropout_rate": 0.1
            },
            "training": {
                "epochs": 10,
                "batch_size": 16,
                "learning_rate": 2e-5,
                "weight_decay": 0.01,
                "warmup_steps": 500,
                "save_steps": 1000,
                "eval_steps": 500,
                "logging_steps": 100,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0
            },
            "data": {
                "train_file": "",
                "validation_file": "",
                "test_file": "",
                "text_column": "text",
                "label_column": "labels",
                "preprocessing": {
                    "lowercase": False,
                    "remove_diacritics": True,
                    "normalize_arabic": True,
                    "max_length": 128
                }
            },
            "labels": {
                "label_names": [
                    "O", "B-STREET", "I-STREET", "B-BUILDING", "I-BUILDING",
                    "B-AREA", "I-AREA", "B-CITY", "I-CITY"
                ],
                "label_mapping": {
                    "O": 0, "B-STREET": 1, "I-STREET": 2, "B-BUILDING": 3,
                    "I-BUILDING": 4, "B-AREA": 5, "I-AREA": 6, "B-CITY": 7, "I-CITY": 8
                }
            },
            "evaluation": {
                "metrics": ["accuracy", "f1", "precision", "recall"],
                "average": "weighted",
                "save_predictions": True
            },
            "output": {
                "model_name": "",
                "save_total_limit": 3,
                "save_best_model": True,
                "load_best_model_at_end": True
            },
            "hardware": {
                "device": "auto",
                "fp16": True,
                "dataloader_num_workers": 4
            },
            "logging": {
                "level": "INFO",
                "log_file": "",
                "tensorboard_dir": ""
            }
        }
        
        # Save default template
        default_template_path = self.templates_dir / "default.json"
        if not default_template_path.exists():
            with open(default_template_path, 'w', encoding='utf-8') as f:
                json.dump(default_template, f, indent=2, ensure_ascii=False)
    
    def create_country_config(self, country_code: str, template: str = "default", force: bool = False) -> bool:
        """Create a new country configuration from template"""
        country_config_path = self.countries_dir / f"{country_code}.json"
        
        # Check if config already exists
        if country_config_path.exists() and not force:
            print(f"Configuration for {country_code} already exists. Use --force to overwrite.")
            return False
        
        # Load template
        template_path = self.templates_dir / f"{template}.json"
        if not template_path.exists():
            print(f"Template '{template}' not found")
            return False
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Customize for country
            config["country"]["code"] = country_code
            config["country"]["name"] = country_code.upper()
            config["output"]["model_name"] = f"{country_code}_dapt_model"
            config["data"]["train_file"] = f"{country_code}_train.csv"
            config["data"]["validation_file"] = f"{country_code}_val.csv"
            config["data"]["test_file"] = f"{country_code}_test.csv"
            config["logging"]["log_file"] = f"{country_code}_training.log"
            config["logging"]["tensorboard_dir"] = f"tensorboard_{country_code}"
            
            # Add metadata
            config["metadata"] = {
                "created_at": datetime.now().isoformat(),
                "template_used": template,
                "version": "1.0.0"
            }
            
            # Save country config
            with open(country_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            print(f"Configuration created for {country_code} at {country_config_path}")
            return True
            
        except Exception as e:
            print(f"Error creating configuration: {str(e)}")
            return False
    
    def get_country_config(self, country_code: str) -> Dict[str, Any]:
        """Load country-specific configuration"""
        country_config_path = self.countries_dir / f"{country_code}.json"
        
        if not country_config_path.exists():
            raise FileNotFoundError(f"Configuration for {country_code} not found. Create it first using 'config create --country {country_code}'")
        
        try:
            with open(country_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading configuration for {country_code}: {str(e)}")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file path"""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {str(e)}")
    
    def save_config(self, config: Dict[str, Any], country_code: str) -> bool:
        """Save configuration for a country"""
        country_config_path = self.countries_dir / f"{country_code}.json"
        
        try:
            # Update metadata
            if "metadata" not in config:
                config["metadata"] = {}
            config["metadata"]["updated_at"] = datetime.now().isoformat()
            
            with open(country_config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            return True
        except Exception as e:
            print(f"Error saving configuration: {str(e)}")
            return False
    
    def list_configs(self) -> List[str]:
        """List all available country configurations"""
        configs = []
        for config_file in self.countries_dir.glob("*.json"):
            configs.append(config_file.stem)
        return sorted(configs)
    
    def list_templates(self) -> List[str]:
        """List all available templates"""
        templates = []
        for template_file in self.templates_dir.glob("*.json"):
            templates.append(template_file.stem)
        return sorted(templates)
    
    def validate_country_config(self, country_code: str) -> bool:
        """Validate a country configuration"""
        try:
            config = self.get_country_config(country_code)
            return self._validate_config_structure(config)
        except Exception as e:
            print(f"Validation failed: {str(e)}")
            return False
    
    def validate_config_file(self, config_path: str) -> bool:
        """Validate a configuration file"""
        try:
            config = self.load_config(config_path)
            return self._validate_config_structure(config)
        except Exception as e:
            print(f"Validation failed: {str(e)}")
            return False
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure for DAPT training"""
        # DAPT only requires core sections for domain adaptive pre-training
        required_sections = ["country", "model", "training", "data"]
        
        for section in required_sections:
            if section not in config:
                print(f"Missing required section: {section}")
                return False
        
        # Validate country section
        country = config["country"]
        if not country.get("code") or not country.get("name"):
            print("Country section must have 'code' and 'name'")
            return False
        
        # Validate model section
        model = config["model"]
        if not model.get("base_model"):
            print("Model section must have 'base_model'")
            return False
        
        # Validate training section
        training = config["training"]
        required_training_params = ["epochs", "batch_size", "learning_rate"]
        for param in required_training_params:
            if param not in training:
                print(f"Training section missing required parameter: {param}")
                return False
        
        # Validate data section - DAPT uses plain text files
        data = config["data"]
        required_data_files = ["train_file", "validation_file", "test_file"]
        for file_param in required_data_files:
            if not data.get(file_param):
                print(f"Data section missing required parameter: {file_param}")
                return False
        
        print("Configuration validation passed")
        return True
    
    def update_config(self, country_code: str, updates: Dict[str, Any]) -> bool:
        """Update specific fields in a country configuration"""
        try:
            config = self.get_country_config(country_code)
            
            # Deep merge updates
            def deep_merge(base_dict, update_dict):
                for key, value in update_dict.items():
                    if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                        deep_merge(base_dict[key], value)
                    else:
                        base_dict[key] = value
            
            deep_merge(config, updates)
            
            # Validate updated config
            if not self._validate_config_structure(config):
                print("Updated configuration is invalid")
                return False
            
            return self.save_config(config, country_code)
            
        except Exception as e:
            print(f"Error updating configuration: {str(e)}")
            return False
    
    def copy_config(self, source_country: str, target_country: str, force: bool = False) -> bool:
        """Copy configuration from one country to another"""
        try:
            source_config = self.get_country_config(source_country)
            
            # Update country-specific fields
            source_config["country"]["code"] = target_country
            source_config["country"]["name"] = target_country.upper()
            source_config["output"]["model_name"] = f"{target_country}_dapt_model"
            source_config["data"]["train_file"] = f"{target_country}_train.csv"
            source_config["data"]["validation_file"] = f"{target_country}_val.csv"
            source_config["data"]["test_file"] = f"{target_country}_test.csv"
            source_config["logging"]["log_file"] = f"{target_country}_training.log"
            source_config["logging"]["tensorboard_dir"] = f"tensorboard_{target_country}"
            
            # Update metadata
            source_config["metadata"] = {
                "created_at": datetime.now().isoformat(),
                "copied_from": source_country,
                "version": "1.0.0"
            }
            
            return self.save_config(source_config, target_country)
            
        except Exception as e:
            print(f"Error copying configuration: {str(e)}")
            return False
    
    def delete_config(self, country_code: str) -> bool:
        """Delete a country configuration"""
        country_config_path = self.countries_dir / f"{country_code}.json"
        
        if not country_config_path.exists():
            print(f"Configuration for {country_code} not found")
            return False
        
        try:
            country_config_path.unlink()
            print(f"Configuration for {country_code} deleted")
            return True
        except Exception as e:
            print(f"Error deleting configuration: {str(e)}")
            return False