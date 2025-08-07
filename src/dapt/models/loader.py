"""DAPT Model Loader

Model loading utilities for DAPT training:
- Base model loading from Hugging Face
- Custom model loading
- Model configuration and initialization
- Integration with existing NER models
- Model adaptation for different countries
"""

import os
import json
import torch
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    BertModel, BertTokenizer, BertConfig,
    RobertaModel, RobertaTokenizer, RobertaConfig,
    DistilBertModel, DistilBertTokenizer, DistilBertConfig,
    AutoModelForTokenClassification, AutoModelForSequenceClassification
)
from huggingface_hub import hf_hub_download, list_repo_files
import numpy as np
from abc import ABC, abstractmethod

class BaseModelLoader(ABC):
    """Abstract base class for model loaders"""
    
    @abstractmethod
    def load_model(self, model_path: str, **kwargs) -> Tuple[Optional[AutoModel], Optional[AutoTokenizer], Optional[AutoConfig]]:
        """Load model from path"""
        pass
    
    @abstractmethod
    def validate_model(self, model_path: str) -> bool:
        """Validate model compatibility"""
        pass

class HuggingFaceModelLoader(BaseModelLoader):
    """Hugging Face model loader"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cache_dir = config.get("cache_dir")
        self.use_auth_token = config.get("use_auth_token", False)
        self.trust_remote_code = config.get("trust_remote_code", False)
    
    def load_model(self, model_path: str, **kwargs) -> Tuple[Optional[AutoModel], Optional[AutoTokenizer], Optional[AutoConfig]]:
        """Load model from Hugging Face Hub or local path"""
        try:
            # Load configuration
            config = AutoConfig.from_pretrained(
                model_path,
                cache_dir=self.cache_dir,
                use_auth_token=self.use_auth_token,
                trust_remote_code=self.trust_remote_code,
                **kwargs
            )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                cache_dir=self.cache_dir,
                use_auth_token=self.use_auth_token,
                trust_remote_code=self.trust_remote_code,
                **kwargs
            )
            
            # Load model
            model = AutoModel.from_pretrained(
                model_path,
                config=config,
                cache_dir=self.cache_dir,
                use_auth_token=self.use_auth_token,
                trust_remote_code=self.trust_remote_code,
                **kwargs
            )
            
            return model, tokenizer, config
            
        except Exception as e:
            print(f"Error loading Hugging Face model: {str(e)}")
            return None, None, None
    
    def validate_model(self, model_path: str) -> bool:
        """Validate Hugging Face model"""
        try:
            # Try to load config to validate
            AutoConfig.from_pretrained(
                model_path,
                cache_dir=self.cache_dir,
                use_auth_token=self.use_auth_token
            )
            return True
        except Exception:
            return False

class LocalModelLoader(BaseModelLoader):
    """Local model loader for saved models"""
    
    def load_model(self, model_path: str, **kwargs) -> Tuple[Optional[AutoModel], Optional[AutoTokenizer], Optional[AutoConfig]]:
        """Load model from local directory"""
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                return None, None, None
            
            # Load configuration
            config = AutoConfig.from_pretrained(model_path, **kwargs)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
            
            # Load model
            model = AutoModel.from_pretrained(model_path, config=config, **kwargs)
            
            return model, tokenizer, config
            
        except Exception as e:
            print(f"Error loading local model: {str(e)}")
            return None, None, None
    
    def validate_model(self, model_path: str) -> bool:
        """Validate local model"""
        try:
            model_path = Path(model_path)
            
            # Check if directory exists and contains required files
            required_files = ['config.json', 'pytorch_model.bin']
            
            for file_name in required_files:
                if not (model_path / file_name).exists():
                    return False
            
            return True
            
        except Exception:
            return False

class NERModelLoader(BaseModelLoader):
    """NER model loader for token classification models"""
    
    def load_model(self, model_path: str, **kwargs) -> Tuple[Optional[AutoModelForTokenClassification], Optional[AutoTokenizer], Optional[AutoConfig]]:
        """Load NER model"""
        try:
            # Load configuration
            config = AutoConfig.from_pretrained(model_path, **kwargs)
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, **kwargs)
            
            # Load NER model
            model = AutoModelForTokenClassification.from_pretrained(
                model_path,
                config=config,
                **kwargs
            )
            
            return model, tokenizer, config
            
        except Exception as e:
            print(f"Error loading NER model: {str(e)}")
            return None, None, None
    
    def validate_model(self, model_path: str) -> bool:
        """Validate NER model"""
        try:
            config = AutoConfig.from_pretrained(model_path)
            # Check if it's a token classification model
            return hasattr(config, 'num_labels') and config.num_labels > 1
        except Exception:
            return False

class ModelLoader:
    """Main model loader for DAPT training"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
        self.country_code = config["country"]["code"]
        
        # Model configuration
        self.base_model_name = config["model"]["base_model"]
        self.model_max_length = config["model"].get("max_length", 512)
        self.model_type = config["model"].get("type", "auto")
        
        # Cache configuration
        self.cache_dir = Path(global_config.get("cache_dir", "./cache")) / "models"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize loaders
        loader_config = {
            "cache_dir": str(self.cache_dir),
            "use_auth_token": config["model"].get("use_auth_token", False),
            "trust_remote_code": config["model"].get("trust_remote_code", False)
        }
        
        self.hf_loader = HuggingFaceModelLoader(loader_config)
        self.local_loader = LocalModelLoader()
        self.ner_loader = NERModelLoader()
        
        # Supported model architectures
        self.supported_architectures = {
            'bert': (BertModel, BertTokenizer, BertConfig),
            'roberta': (RobertaModel, RobertaTokenizer, RobertaConfig),
            'distilbert': (DistilBertModel, DistilBertTokenizer, DistilBertConfig),
            'auto': (AutoModel, AutoTokenizer, AutoConfig)
        }
        
        # Country-specific model mappings
        self.country_models = self._load_country_models()
        
        # Logging
        self.logger = self._setup_logging()
        
        self.logger.info(f"Model Loader initialized for country: {self.country_code}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for model loader"""
        logger = logging.getLogger(f"dapt_model_loader_{self.country_code}")
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
        
        return logger
    
    def _load_country_models(self) -> Dict[str, Dict[str, str]]:
        """Load country-specific model recommendations"""
        return {
            'UAE': {
                'primary': 'aubmindlab/bert-base-arabertv2',
                'alternatives': [
                    'asafaya/bert-base-arabic',
                    'CAMeL-Lab/bert-base-arabic-camelbert-mix'
                ],
                'ner_models': [
                    'hatmimoha/arabic-ner',
                    'CAMeL-Lab/bert-base-arabic-camelbert-mix-ner'
                ]
            },
            'SA': {
                'primary': 'aubmindlab/bert-base-arabertv2',
                'alternatives': [
                    'asafaya/bert-base-arabic',
                    'CAMeL-Lab/bert-base-arabic-camelbert-mix'
                ],
                'ner_models': [
                    'hatmimoha/arabic-ner',
                    'CAMeL-Lab/bert-base-arabic-camelbert-mix-ner'
                ]
            },
            'EG': {
                'primary': 'aubmindlab/bert-base-arabertv2',
                'alternatives': [
                    'asafaya/bert-base-arabic',
                    'CAMeL-Lab/bert-base-arabic-camelbert-mix'
                ],
                'ner_models': [
                    'hatmimoha/arabic-ner',
                    'CAMeL-Lab/bert-base-arabic-camelbert-mix-ner'
                ]
            },
            'MA': {
                'primary': 'aubmindlab/bert-base-arabertv2',
                'alternatives': [
                    'asafaya/bert-base-arabic',
                    'CAMeL-Lab/bert-base-arabic-camelbert-mix'
                ],
                'ner_models': [
                    'hatmimoha/arabic-ner'
                ]
            },
            'LB': {
                'primary': 'aubmindlab/bert-base-arabertv2',
                'alternatives': [
                    'asafaya/bert-base-arabic'
                ],
                'ner_models': [
                    'hatmimoha/arabic-ner'
                ]
            },
            'JO': {
                'primary': 'aubmindlab/bert-base-arabertv2',
                'alternatives': [
                    'asafaya/bert-base-arabic'
                ],
                'ner_models': [
                    'hatmimoha/arabic-ner'
                ]
            }
        }
    
    def load_base_model(self, model_name: Optional[str] = None, **kwargs) -> Tuple[Optional[AutoModel], Optional[AutoTokenizer], Optional[AutoConfig]]:
        """Load base model for DAPT training"""
        try:
            if model_name is None:
                model_name = self.base_model_name
            
            self.logger.info(f"Loading base model: {model_name}")
            
            # Determine loader based on model path
            if self._is_local_path(model_name):
                model, tokenizer, config = self.local_loader.load_model(model_name, **kwargs)
            else:
                model, tokenizer, config = self.hf_loader.load_model(model_name, **kwargs)
            
            if model is None:
                self.logger.error(f"Failed to load model: {model_name}")
                return None, None, None
            
            # Configure tokenizer
            if tokenizer is not None:
                tokenizer.model_max_length = self.model_max_length
                
                # Add special tokens if needed
                special_tokens = self._get_country_special_tokens()
                if special_tokens:
                    tokenizer.add_special_tokens(special_tokens)
                    model.resize_token_embeddings(len(tokenizer))
            
            self.logger.info(f"Successfully loaded base model: {model_name}")
            return model, tokenizer, config
            
        except Exception as e:
            self.logger.error(f"Error loading base model: {str(e)}")
            return None, None, None
    
    def load_ner_model(self, model_name: Optional[str] = None, **kwargs) -> Tuple[Optional[AutoModelForTokenClassification], Optional[AutoTokenizer], Optional[AutoConfig]]:
        """Load NER model for evaluation or fine-tuning"""
        try:
            if model_name is None:
                # Use country-specific NER model
                country_models = self.country_models.get(self.country_code, {})
                ner_models = country_models.get('ner_models', [])
                if ner_models:
                    model_name = ner_models[0]
                else:
                    self.logger.error(f"No NER model available for country: {self.country_code}")
                    return None, None, None
            
            self.logger.info(f"Loading NER model: {model_name}")
            
            # Load NER model
            model, tokenizer, config = self.ner_loader.load_model(model_name, **kwargs)
            
            if model is None:
                self.logger.error(f"Failed to load NER model: {model_name}")
                return None, None, None
            
            self.logger.info(f"Successfully loaded NER model: {model_name}")
            return model, tokenizer, config
            
        except Exception as e:
            self.logger.error(f"Error loading NER model: {str(e)}")
            return None, None, None
    
    def load_country_recommended_model(self, model_type: str = "primary") -> Tuple[Optional[AutoModel], Optional[AutoTokenizer], Optional[AutoConfig]]:
        """Load country-recommended model"""
        try:
            country_models = self.country_models.get(self.country_code, {})
            
            if model_type == "primary":
                model_name = country_models.get('primary')
            elif model_type == "alternative":
                alternatives = country_models.get('alternatives', [])
                model_name = alternatives[0] if alternatives else None
            else:
                self.logger.error(f"Unknown model type: {model_type}")
                return None, None, None
            
            if model_name is None:
                self.logger.error(f"No {model_type} model available for country: {self.country_code}")
                return None, None, None
            
            return self.load_base_model(model_name)
            
        except Exception as e:
            self.logger.error(f"Error loading country recommended model: {str(e)}")
            return None, None, None
    
    def validate_model_compatibility(self, model_name: str) -> Dict[str, Any]:
        """Validate model compatibility for DAPT training"""
        try:
            validation_result = {
                "model_name": model_name,
                "is_compatible": False,
                "issues": [],
                "recommendations": [],
                "model_info": {}
            }
            
            # Check if model exists and is accessible
            if self._is_local_path(model_name):
                if not self.local_loader.validate_model(model_name):
                    validation_result["issues"].append("Local model not found or invalid")
                    return validation_result
            else:
                if not self.hf_loader.validate_model(model_name):
                    validation_result["issues"].append("Hugging Face model not found or inaccessible")
                    return validation_result
            
            # Load model config for detailed validation
            try:
                config = AutoConfig.from_pretrained(model_name)
                validation_result["model_info"] = {
                    "model_type": config.model_type,
                    "hidden_size": getattr(config, 'hidden_size', None),
                    "num_attention_heads": getattr(config, 'num_attention_heads', None),
                    "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
                    "vocab_size": getattr(config, 'vocab_size', None),
                    "max_position_embeddings": getattr(config, 'max_position_embeddings', None)
                }
                
                # Check architecture compatibility
                if config.model_type not in ['bert', 'roberta', 'distilbert', 'electra', 'deberta']:
                    validation_result["issues"].append(f"Model architecture '{config.model_type}' may not be fully supported")
                    validation_result["recommendations"].append("Consider using BERT, RoBERTa, or DistilBERT models")
                
                # Check if model supports Arabic
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                arabic_test = "مرحبا بك في العالم العربي"
                tokens = tokenizer.tokenize(arabic_test)
                
                if len(tokens) > len(arabic_test.split()) * 3:  # Too many subword tokens
                    validation_result["issues"].append("Model may not handle Arabic text efficiently")
                    validation_result["recommendations"].append("Consider using Arabic-specific models like AraBERT")
                
                # Check model size
                if hasattr(config, 'hidden_size') and config.hidden_size > 1024:
                    validation_result["recommendations"].append("Large model detected - ensure sufficient computational resources")
                
                validation_result["is_compatible"] = len(validation_result["issues"]) == 0
                
            except Exception as e:
                validation_result["issues"].append(f"Error loading model config: {str(e)}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Error validating model compatibility: {str(e)}")
            return {
                "model_name": model_name,
                "is_compatible": False,
                "issues": [f"Validation error: {str(e)}"],
                "recommendations": [],
                "model_info": {}
            }
    
    def list_available_models(self) -> Dict[str, List[str]]:
        """List available models for the country"""
        try:
            country_models = self.country_models.get(self.country_code, {})
            
            available_models = {
                "primary": [country_models.get('primary')] if country_models.get('primary') else [],
                "alternatives": country_models.get('alternatives', []),
                "ner_models": country_models.get('ner_models', []),
                "all_countries": []
            }
            
            # Collect all unique models from all countries
            all_models = set()
            for country_data in self.country_models.values():
                if country_data.get('primary'):
                    all_models.add(country_data['primary'])
                all_models.update(country_data.get('alternatives', []))
                all_models.update(country_data.get('ner_models', []))
            
            available_models["all_countries"] = sorted(list(all_models))
            
            return available_models
            
        except Exception as e:
            self.logger.error(f"Error listing available models: {str(e)}")
            return {}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a model"""
        try:
            model_info = {
                "model_name": model_name,
                "is_local": self._is_local_path(model_name),
                "config": {},
                "tokenizer_info": {},
                "compatibility": {}
            }
            
            # Load config
            try:
                config = AutoConfig.from_pretrained(model_name)
                model_info["config"] = {
                    "model_type": config.model_type,
                    "architectures": getattr(config, 'architectures', []),
                    "hidden_size": getattr(config, 'hidden_size', None),
                    "num_attention_heads": getattr(config, 'num_attention_heads', None),
                    "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
                    "vocab_size": getattr(config, 'vocab_size', None),
                    "max_position_embeddings": getattr(config, 'max_position_embeddings', None)
                }
            except Exception as e:
                model_info["config"]["error"] = str(e)
            
            # Load tokenizer info
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model_info["tokenizer_info"] = {
                    "vocab_size": len(tokenizer),
                    "model_max_length": getattr(tokenizer, 'model_max_length', None),
                    "special_tokens": {
                        "pad_token": tokenizer.pad_token,
                        "unk_token": tokenizer.unk_token,
                        "cls_token": getattr(tokenizer, 'cls_token', None),
                        "sep_token": getattr(tokenizer, 'sep_token', None),
                        "mask_token": getattr(tokenizer, 'mask_token', None)
                    }
                }
            except Exception as e:
                model_info["tokenizer_info"]["error"] = str(e)
            
            # Compatibility check
            model_info["compatibility"] = self.validate_model_compatibility(model_name)
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {str(e)}")
            return {"model_name": model_name, "error": str(e)}
    
    def _is_local_path(self, model_name: str) -> bool:
        """Check if model name is a local path"""
        return os.path.exists(model_name) or model_name.startswith('./') or model_name.startswith('/')
    
    def _get_country_special_tokens(self) -> Optional[Dict[str, List[str]]]:
        """Get country-specific special tokens"""
        country_tokens = {
            'UAE': {
                'additional_special_tokens': ['[UAE]', '[EMIRATES]', '[DUBAI]', '[ABU_DHABI]']
            },
            'SA': {
                'additional_special_tokens': ['[SA]', '[SAUDI]', '[RIYADH]', '[JEDDAH]']
            },
            'EG': {
                'additional_special_tokens': ['[EG]', '[EGYPT]', '[CAIRO]', '[ALEXANDRIA]']
            },
            'MA': {
                'additional_special_tokens': ['[MA]', '[MOROCCO]', '[RABAT]', '[CASABLANCA]']
            },
            'LB': {
                'additional_special_tokens': ['[LB]', '[LEBANON]', '[BEIRUT]']
            },
            'JO': {
                'additional_special_tokens': ['[JO]', '[JORDAN]', '[AMMAN]']
            }
        }
        
        return country_tokens.get(self.country_code)
    
    def clear_cache(self) -> bool:
        """Clear model cache"""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Model cache cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        try:
            if not self.cache_dir.exists():
                return {
                    "cache_directory": str(self.cache_dir),
                    "exists": False,
                    "size_bytes": 0,
                    "size_mb": 0,
                    "cached_models": 0
                }
            
            # Calculate cache size
            total_size = 0
            model_count = 0
            
            for item in self.cache_dir.rglob('*'):
                if item.is_file():
                    total_size += item.stat().st_size
                elif item.is_dir() and item.parent == self.cache_dir:
                    model_count += 1
            
            return {
                "cache_directory": str(self.cache_dir),
                "exists": True,
                "size_bytes": total_size,
                "size_mb": total_size / (1024 * 1024),
                "cached_models": model_count
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache info: {str(e)}")
            return {}