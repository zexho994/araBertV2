"""Configuration Templates for DAPT

Provides pre-defined configuration templates for different countries and use cases.
Includes specialized templates for various Arabic dialects and regions.
"""

import json
from typing import Dict, Any, List
from pathlib import Path

class ConfigTemplates:
    """Manages configuration templates for different countries and scenarios"""
    
    def __init__(self, templates_dir: str):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Create all predefined templates
        self._create_all_templates()
    
    def _create_all_templates(self) -> None:
        """Create all predefined configuration templates"""
        templates = {
            "default": self._get_default_template(),
            "uae": self._get_uae_template(),
            "saudi": self._get_saudi_template(),
            "egypt": self._get_egypt_template(),
            "morocco": self._get_morocco_template(),
            "lebanon": self._get_lebanon_template(),
            "jordan": self._get_jordan_template(),
            "quick_test": self._get_quick_test_template(),
            "production": self._get_production_template()
        }
        
        for template_name, template_config in templates.items():
            template_path = self.templates_dir / f"{template_name}.json"
            if not template_path.exists():
                with open(template_path, 'w', encoding='utf-8') as f:
                    json.dump(template_config, f, indent=2, ensure_ascii=False)
    
    def _get_default_template(self) -> Dict[str, Any]:
        """Get default configuration template"""
        return {
            "country": {
                "code": "",
                "name": "",
                "language": "ar",
                "script": "arabic",
                "dialect": "msa"  # Modern Standard Arabic
            },
            "model": {
                "base_model": "aubmindlab/bert-base-arabertv2",
                "max_length": 128,
                "num_labels": 9,
                "dropout_rate": 0.1,
                "hidden_dropout_prob": 0.1,
                "attention_probs_dropout_prob": 0.1
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
                "max_grad_norm": 1.0,
                "lr_scheduler_type": "linear",
                "metric_for_best_model": "f1",
                "greater_is_better": True
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
                    "max_length": 128,
                    "padding": "max_length",
                    "truncation": True
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
                "save_predictions": True,
                "compute_metrics_each_epoch": True
            },
            "output": {
                "model_name": "",
                "save_total_limit": 3,
                "save_best_model": True,
                "load_best_model_at_end": True,
                "save_strategy": "steps",
                "evaluation_strategy": "steps"
            },
            "hardware": {
                "device": "auto",
                "fp16": True,
                "dataloader_num_workers": 4,
                "dataloader_pin_memory": True
            },
            "logging": {
                "level": "INFO",
                "log_file": "",
                "tensorboard_dir": "",
                "report_to": ["tensorboard"]
            }
        }
    
    def _get_uae_template(self) -> Dict[str, Any]:
        """Get UAE-specific configuration template"""
        template = self._get_default_template()
        
        # UAE-specific customizations
        template["country"].update({
            "code": "uae",
            "name": "United Arab Emirates",
            "dialect": "gulf",
            "currency": "AED",
            "timezone": "Asia/Dubai"
        })
        
        # UAE-specific labels (more detailed for UAE addresses)
        template["labels"] = {
            "label_names": [
                "O", "B-STREET", "I-STREET", "B-BUILDING", "I-BUILDING",
                "B-AREA", "I-AREA", "B-CITY", "I-CITY", "B-EMIRATE", "I-EMIRATE",
                "B-LANDMARK", "I-LANDMARK", "B-POBOX", "I-POBOX"
            ],
            "label_mapping": {
                "O": 0, "B-STREET": 1, "I-STREET": 2, "B-BUILDING": 3, "I-BUILDING": 4,
                "B-AREA": 5, "I-AREA": 6, "B-CITY": 7, "I-CITY": 8,
                "B-EMIRATE": 9, "I-EMIRATE": 10, "B-LANDMARK": 11, "I-LANDMARK": 12,
                "B-POBOX": 13, "I-POBOX": 14
            }
        }
        
        template["model"]["num_labels"] = len(template["labels"]["label_names"])
        
        # UAE-specific data files
        template["data"].update({
            "train_file": "uae_train.csv",
            "validation_file": "uae_val.csv",
            "test_file": "uae_test.csv"
        })
        
        template["output"]["model_name"] = "uae_dapt_model"
        template["logging"]["log_file"] = "uae_training.log"
        template["logging"]["tensorboard_dir"] = "tensorboard_uae"
        
        return template
    
    def _get_saudi_template(self) -> Dict[str, Any]:
        """Get Saudi Arabia-specific configuration template"""
        template = self._get_default_template()
        
        template["country"].update({
            "code": "saudi",
            "name": "Saudi Arabia",
            "dialect": "najdi",
            "currency": "SAR",
            "timezone": "Asia/Riyadh"
        })
        
        # Saudi-specific labels
        template["labels"] = {
            "label_names": [
                "O", "B-STREET", "I-STREET", "B-BUILDING", "I-BUILDING",
                "B-DISTRICT", "I-DISTRICT", "B-CITY", "I-CITY", "B-PROVINCE", "I-PROVINCE",
                "B-POSTAL", "I-POSTAL"
            ],
            "label_mapping": {
                "O": 0, "B-STREET": 1, "I-STREET": 2, "B-BUILDING": 3, "I-BUILDING": 4,
                "B-DISTRICT": 5, "I-DISTRICT": 6, "B-CITY": 7, "I-CITY": 8,
                "B-PROVINCE": 9, "I-PROVINCE": 10, "B-POSTAL": 11, "I-POSTAL": 12
            }
        }
        
        template["model"]["num_labels"] = len(template["labels"]["label_names"])
        
        template["data"].update({
            "train_file": "saudi_train.csv",
            "validation_file": "saudi_val.csv",
            "test_file": "saudi_test.csv"
        })
        
        template["output"]["model_name"] = "saudi_dapt_model"
        template["logging"]["log_file"] = "saudi_training.log"
        template["logging"]["tensorboard_dir"] = "tensorboard_saudi"
        
        return template
    
    def _get_egypt_template(self) -> Dict[str, Any]:
        """Get Egypt-specific configuration template"""
        template = self._get_default_template()
        
        template["country"].update({
            "code": "egypt",
            "name": "Egypt",
            "dialect": "egyptian",
            "currency": "EGP",
            "timezone": "Africa/Cairo"
        })
        
        # Egypt-specific labels
        template["labels"] = {
            "label_names": [
                "O", "B-STREET", "I-STREET", "B-BUILDING", "I-BUILDING",
                "B-AREA", "I-AREA", "B-CITY", "I-CITY", "B-GOVERNORATE", "I-GOVERNORATE",
                "B-POSTAL", "I-POSTAL"
            ],
            "label_mapping": {
                "O": 0, "B-STREET": 1, "I-STREET": 2, "B-BUILDING": 3, "I-BUILDING": 4,
                "B-AREA": 5, "I-AREA": 6, "B-CITY": 7, "I-CITY": 8,
                "B-GOVERNORATE": 9, "I-GOVERNORATE": 10, "B-POSTAL": 11, "I-POSTAL": 12
            }
        }
        
        template["model"]["num_labels"] = len(template["labels"]["label_names"])
        
        template["data"].update({
            "train_file": "egypt_train.csv",
            "validation_file": "egypt_val.csv",
            "test_file": "egypt_test.csv"
        })
        
        template["output"]["model_name"] = "egypt_dapt_model"
        template["logging"]["log_file"] = "egypt_training.log"
        template["logging"]["tensorboard_dir"] = "tensorboard_egypt"
        
        return template
    
    def _get_morocco_template(self) -> Dict[str, Any]:
        """Get Morocco-specific configuration template"""
        template = self._get_default_template()
        
        template["country"].update({
            "code": "morocco",
            "name": "Morocco",
            "dialect": "maghrebi",
            "currency": "MAD",
            "timezone": "Africa/Casablanca"
        })
        
        # Morocco-specific labels
        template["labels"] = {
            "label_names": [
                "O", "B-STREET", "I-STREET", "B-BUILDING", "I-BUILDING",
                "B-QUARTER", "I-QUARTER", "B-CITY", "I-CITY", "B-REGION", "I-REGION",
                "B-POSTAL", "I-POSTAL"
            ],
            "label_mapping": {
                "O": 0, "B-STREET": 1, "I-STREET": 2, "B-BUILDING": 3, "I-BUILDING": 4,
                "B-QUARTER": 5, "I-QUARTER": 6, "B-CITY": 7, "I-CITY": 8,
                "B-REGION": 9, "I-REGION": 10, "B-POSTAL": 11, "I-POSTAL": 12
            }
        }
        
        template["model"]["num_labels"] = len(template["labels"]["label_names"])
        
        template["data"].update({
            "train_file": "morocco_train.csv",
            "validation_file": "morocco_val.csv",
            "test_file": "morocco_test.csv"
        })
        
        template["output"]["model_name"] = "morocco_dapt_model"
        template["logging"]["log_file"] = "morocco_training.log"
        template["logging"]["tensorboard_dir"] = "tensorboard_morocco"
        
        return template
    
    def _get_lebanon_template(self) -> Dict[str, Any]:
        """Get Lebanon-specific configuration template"""
        template = self._get_default_template()
        
        template["country"].update({
            "code": "lebanon",
            "name": "Lebanon",
            "dialect": "levantine",
            "currency": "LBP",
            "timezone": "Asia/Beirut"
        })
        
        template["data"].update({
            "train_file": "lebanon_train.csv",
            "validation_file": "lebanon_val.csv",
            "test_file": "lebanon_test.csv"
        })
        
        template["output"]["model_name"] = "lebanon_dapt_model"
        template["logging"]["log_file"] = "lebanon_training.log"
        template["logging"]["tensorboard_dir"] = "tensorboard_lebanon"
        
        return template
    
    def _get_jordan_template(self) -> Dict[str, Any]:
        """Get Jordan-specific configuration template"""
        template = self._get_default_template()
        
        template["country"].update({
            "code": "jordan",
            "name": "Jordan",
            "dialect": "levantine",
            "currency": "JOD",
            "timezone": "Asia/Amman"
        })
        
        template["data"].update({
            "train_file": "jordan_train.csv",
            "validation_file": "jordan_val.csv",
            "test_file": "jordan_test.csv"
        })
        
        template["output"]["model_name"] = "jordan_dapt_model"
        template["logging"]["log_file"] = "jordan_training.log"
        template["logging"]["tensorboard_dir"] = "tensorboard_jordan"
        
        return template
    
    def _get_quick_test_template(self) -> Dict[str, Any]:
        """Get quick test configuration template for development"""
        template = self._get_default_template()
        
        # Quick test settings
        template["training"].update({
            "epochs": 2,
            "batch_size": 8,
            "save_steps": 50,
            "eval_steps": 50,
            "logging_steps": 10,
            "warmup_steps": 10
        })
        
        template["model"]["max_length"] = 64
        template["data"]["preprocessing"]["max_length"] = 64
        
        template["output"]["model_name"] = "quick_test_model"
        template["logging"]["log_file"] = "quick_test.log"
        template["logging"]["tensorboard_dir"] = "tensorboard_test"
        
        return template
    
    def _get_production_template(self) -> Dict[str, Any]:
        """Get production-ready configuration template"""
        template = self._get_default_template()
        
        # Production settings
        template["training"].update({
            "epochs": 20,
            "batch_size": 32,
            "learning_rate": 1e-5,
            "warmup_steps": 1000,
            "save_steps": 2000,
            "eval_steps": 1000,
            "logging_steps": 200,
            "gradient_accumulation_steps": 2
        })
        
        template["model"]["max_length"] = 256
        template["data"]["preprocessing"]["max_length"] = 256
        
        template["hardware"].update({
            "fp16": True,
            "dataloader_num_workers": 8,
            "dataloader_pin_memory": True
        })
        
        template["output"].update({
            "save_total_limit": 5,
            "save_best_model": True,
            "load_best_model_at_end": True
        })
        
        template["output"]["model_name"] = "production_model"
        template["logging"]["log_file"] = "production_training.log"
        template["logging"]["tensorboard_dir"] = "tensorboard_production"
        
        return template
    
    def get_template(self, template_name: str) -> Dict[str, Any]:
        """Get a specific template by name"""
        template_path = self.templates_dir / f"{template_name}.json"
        
        if not template_path.exists():
            raise FileNotFoundError(f"Template '{template_name}' not found")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_templates(self) -> List[str]:
        """List all available templates"""
        templates = []
        for template_file in self.templates_dir.glob("*.json"):
            templates.append(template_file.stem)
        return sorted(templates)
    
    def create_custom_template(self, template_name: str, base_template: str, customizations: Dict[str, Any]) -> bool:
        """Create a custom template based on an existing template"""
        try:
            # Load base template
            base_config = self.get_template(base_template)
            
            # Apply customizations
            def deep_merge(base_dict, update_dict):
                for key, value in update_dict.items():
                    if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                        deep_merge(base_dict[key], value)
                    else:
                        base_dict[key] = value
            
            deep_merge(base_config, customizations)
            
            # Save custom template
            template_path = self.templates_dir / f"{template_name}.json"
            with open(template_path, 'w', encoding='utf-8') as f:
                json.dump(base_config, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            print(f"Error creating custom template: {str(e)}")
            return False
    
    def get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get information about a template"""
        template = self.get_template(template_name)
        
        info = {
            "name": template_name,
            "country": template.get("country", {}),
            "model_config": {
                "base_model": template.get("model", {}).get("base_model"),
                "max_length": template.get("model", {}).get("max_length"),
                "num_labels": template.get("model", {}).get("num_labels")
            },
            "training_config": {
                "epochs": template.get("training", {}).get("epochs"),
                "batch_size": template.get("training", {}).get("batch_size"),
                "learning_rate": template.get("training", {}).get("learning_rate")
            },
            "labels": template.get("labels", {}).get("label_names", []),
            "metadata": template.get("metadata", {})
        }
        
        return info