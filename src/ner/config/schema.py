"""Configuration Schema for NER System

Defines the structure and default values for NER configuration files.
Provides schema validation and default configuration generation.
"""

from typing import Dict, Any, List

class ConfigSchema:
    """Defines the schema and defaults for NER configurations"""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get the default configuration template
        
        Returns:
            Dictionary containing default configuration
        """
        return {
            "country": {
                "code": "default",
                "name": "Default Country",
                "language": "en",
                "description": "Default configuration template"
            },
            "model": {
                "name": "default_ner_model",
                "type": "bert",
                "pretrained_model": "bert-base-multilingual-cased",
                "max_length": 512,
                "dropout": 0.1,
                "hidden_size": 768,
                "num_attention_heads": 12,
                "num_hidden_layers": 12
            },
            "training": {
                "epochs": 10,
                "batch_size": 16,
                "learning_rate": 2e-5,
                "optimizer": "adamw",
                "scheduler": "linear",
                "warmup_steps": 500,
                "weight_decay": 0.01,
                "gradient_accumulation_steps": 1,
                "max_grad_norm": 1.0,
                "early_stopping": {
                    "enabled": True,
                    "patience": 3,
                    "min_delta": 0.001
                },
                "validation_steps": 500,
                "logging_steps": 100
            },
            "data": {
                "train_file": "train.jsonl",
                "val_file": "val.jsonl",
                "test_file": "test.jsonl",
                "data_format": "jsonl",
                "text_column": "text",
                "labels_column": "labels",
                "preprocessing": {
                    "lowercase": False,
                    "remove_diacritics": False,
                    "normalize_whitespace": True,
                    "remove_special_chars": False
                },
                "augmentation": {
                    "enabled": False,
                    "techniques": []
                }
            },
            "labels": {
                "num_labels": 11,
                "label_names": [
                    "O",
                    "B-STREET", "I-STREET",
                    "B-BUILDING", "I-BUILDING",
                    "B-AREA", "I-AREA",
                    "B-CITY", "I-CITY",
                    "B-COUNTRY", "I-COUNTRY"
                ],
                "label_mapping": {
                    "O": 0,
                    "B-STREET": 1, "I-STREET": 2,
                    "B-BUILDING": 3, "I-BUILDING": 4,
                    "B-AREA": 5, "I-AREA": 6,
                    "B-CITY": 7, "I-CITY": 8,
                    "B-COUNTRY": 9, "I-COUNTRY": 10
                },
                "label_descriptions": {
                    "O": "Outside any entity",
                    "B-STREET": "Beginning of street name",
                    "I-STREET": "Inside street name",
                    "B-BUILDING": "Beginning of building name",
                    "I-BUILDING": "Inside building name",
                    "B-AREA": "Beginning of area/district",
                    "I-AREA": "Inside area/district",
                    "B-CITY": "Beginning of city name",
                    "I-CITY": "Inside city name",
                    "B-COUNTRY": "Beginning of country name",
                    "I-COUNTRY": "Inside country name"
                }
            },
            "evaluation": {
                "metrics": ["precision", "recall", "f1", "accuracy", "entity_f1"],
                "save_predictions": True,
                "save_detailed_report": True,
                "compute_per_label_metrics": True,
                "confusion_matrix": True
            },
            "output": {
                "model_dir": "models",
                "save_steps": 1000,
                "save_total_limit": 3,
                "overwrite_output_dir": True,
                "save_best_model": True,
                "load_best_model_at_end": True
            },
            "hardware": {
                "device": "auto",
                "mixed_precision": True,
                "dataloader_num_workers": 4,
                "pin_memory": True,
                "gradient_checkpointing": False
            },
            "logging": {
                "level": "INFO",
                "log_file": "training.log",
                "error_log_file": "errors.log",
                "wandb": {
                    "enabled": False,
                    "project": "ner-training",
                    "entity": None,
                    "tags": []
                },
                "tensorboard": {
                    "enabled": True,
                    "log_dir": "tensorboard_logs"
                }
            }
        }
    
    @staticmethod
    def get_uae_config() -> Dict[str, Any]:
        """Get UAE-specific configuration
        
        Returns:
            Dictionary containing UAE configuration
        """
        config = ConfigSchema.get_default_config()
        
        # Update UAE-specific settings
        config["country"] = {
            "code": "uae",
            "name": "United Arab Emirates",
            "language": "ar",
            "description": "Configuration for UAE address parsing"
        }
        
        config["model"]["name"] = "uae_address_ner"
        config["model"]["pretrained_model"] = "aubmindlab/bert-base-arabertv2"
        
        # UAE-specific labels (23 labels)
        config["labels"] = {
            "num_labels": 23,
            "label_names": [
                "O",
                "B-COUNTRY", "I-COUNTRY",
                "B-EMIRATE", "I-EMIRATE", 
                "B-CITY", "I-CITY",
                "B-SUB_AREA", "I-SUB_AREA",
                "B-COMPOUND", "I-COMPOUND",
                "B-STREET", "I-STREET",
                "B-BUILDING", "I-BUILDING",
                "B-HOUSE_NUMBER", "I-HOUSE_NUMBER",
                "B-LANDMARK", "I-LANDMARK",
                "B-POSTAL_CODE", "I-POSTAL_CODE",
                "B-MAKANI_NUMBER", "I-MAKANI_NUMBER"
            ],
            "label_mapping": {
                "O": 0,
                "B-COUNTRY": 1, "I-COUNTRY": 2,
                "B-EMIRATE": 3, "I-EMIRATE": 4,
                "B-CITY": 5, "I-CITY": 6,
                "B-SUB_AREA": 7, "I-SUB_AREA": 8,
                "B-COMPOUND": 9, "I-COMPOUND": 10,
                "B-STREET": 11, "I-STREET": 12,
                "B-BUILDING": 13, "I-BUILDING": 14,
                "B-HOUSE_NUMBER": 15, "I-HOUSE_NUMBER": 16,
                "B-LANDMARK": 17, "I-LANDMARK": 18,
                "B-POSTAL_CODE": 19, "I-POSTAL_CODE": 20,
                "B-MAKANI_NUMBER": 21, "I-MAKANI_NUMBER": 22
            },
            "label_descriptions": {
                "O": "Outside any entity",
                "B-COUNTRY": "Beginning of country name",
                "I-COUNTRY": "Inside country name",
                "B-EMIRATE": "Beginning of emirate name",
                "I-EMIRATE": "Inside emirate name",
                "B-CITY": "Beginning of city name",
                "I-CITY": "Inside city name",
                "B-SUB_AREA": "Beginning of sub-area/district",
                "I-SUB_AREA": "Inside sub-area/district",
                "B-COMPOUND": "Beginning of compound/community",
                "I-COMPOUND": "Inside compound/community",
                "B-STREET": "Beginning of street name",
                "I-STREET": "Inside street name",
                "B-BUILDING": "Beginning of building name",
                "I-BUILDING": "Inside building name",
                "B-HOUSE_NUMBER": "Beginning of house/unit number",
                "I-HOUSE_NUMBER": "Inside house/unit number",
                "B-LANDMARK": "Beginning of landmark",
                "I-LANDMARK": "Inside landmark",
                "B-POSTAL_CODE": "Beginning of postal code",
                "I-POSTAL_CODE": "Inside postal code",
                "B-MAKANI_NUMBER": "Beginning of Makani number",
                "I-MAKANI_NUMBER": "Inside Makani number"
            }
        }
        
        # UAE-specific data preprocessing
        config["data"]["preprocessing"]["remove_diacritics"] = True
        config["data"]["preprocessing"]["normalize_whitespace"] = True
        
        # UAE-specific training settings
        config["training"]["epochs"] = 15
        config["training"]["batch_size"] = 32
        config["training"]["learning_rate"] = 3e-5
        
        return config
    
    @staticmethod
    def get_required_fields() -> Dict[str, List[str]]:
        """Get required fields for each configuration section
        
        Returns:
            Dictionary mapping section names to required field lists
        """
        return {
            "country": ["code", "name"],
            "model": ["name", "type", "pretrained_model"],
            "training": ["epochs", "batch_size", "learning_rate"],
            "data": ["train_file", "val_file"],
            "labels": ["num_labels", "label_names", "label_mapping"],
            "evaluation": ["metrics"],
            "output": ["model_dir"],
            "hardware": ["device"],
            "logging": ["level"]
        }
    
    @staticmethod
    def get_optional_fields() -> Dict[str, List[str]]:
        """Get optional fields for each configuration section
        
        Returns:
            Dictionary mapping section names to optional field lists
        """
        return {
            "country": ["language", "description"],
            "model": ["max_length", "dropout", "hidden_size", "num_attention_heads", "num_hidden_layers"],
            "training": ["optimizer", "scheduler", "warmup_steps", "weight_decay", "gradient_accumulation_steps", "max_grad_norm", "early_stopping", "validation_steps", "logging_steps"],
            "data": ["test_file", "data_format", "text_column", "labels_column", "preprocessing", "augmentation"],
            "labels": ["label_descriptions"],
            "evaluation": ["save_predictions", "save_detailed_report", "compute_per_label_metrics", "confusion_matrix"],
            "output": ["save_steps", "save_total_limit", "overwrite_output_dir", "save_best_model", "load_best_model_at_end"],
            "hardware": ["mixed_precision", "dataloader_num_workers", "pin_memory", "gradient_checkpointing"],
            "logging": ["log_file", "error_log_file", "wandb", "tensorboard"]
        }
    
    @staticmethod
    def get_field_types() -> Dict[str, Dict[str, type]]:
        """Get expected types for configuration fields
        
        Returns:
            Dictionary mapping section and field names to expected types
        """
        return {
            "country": {
                "code": str,
                "name": str,
                "language": str,
                "description": str
            },
            "model": {
                "name": str,
                "type": str,
                "pretrained_model": str,
                "max_length": int,
                "dropout": float,
                "hidden_size": int,
                "num_attention_heads": int,
                "num_hidden_layers": int
            },
            "training": {
                "epochs": int,
                "batch_size": int,
                "learning_rate": float,
                "optimizer": str,
                "scheduler": str,
                "warmup_steps": int,
                "weight_decay": float,
                "gradient_accumulation_steps": int,
                "max_grad_norm": float,
                "validation_steps": int,
                "logging_steps": int
            },
            "data": {
                "train_file": str,
                "val_file": str,
                "test_file": str,
                "data_format": str,
                "text_column": str,
                "labels_column": str
            },
            "labels": {
                "num_labels": int,
                "label_names": list,
                "label_mapping": dict,
                "label_descriptions": dict
            },
            "evaluation": {
                "metrics": list,
                "save_predictions": bool,
                "save_detailed_report": bool,
                "compute_per_label_metrics": bool,
                "confusion_matrix": bool
            },
            "output": {
                "model_dir": str,
                "save_steps": int,
                "save_total_limit": int,
                "overwrite_output_dir": bool,
                "save_best_model": bool,
                "load_best_model_at_end": bool
            },
            "hardware": {
                "device": str,
                "mixed_precision": bool,
                "dataloader_num_workers": int,
                "pin_memory": bool,
                "gradient_checkpointing": bool
            },
            "logging": {
                "level": str,
                "log_file": str,
                "error_log_file": str
            }
        }