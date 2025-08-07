"""DAPT Model Manager

Model management utilities for DAPT training:
- Model loading and saving
- Version control and checkpointing
- Integration with existing NER models
- Model metadata management
- Model comparison and selection
"""

import os
import json
import shutil
import pickle
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import logging
import hashlib
import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    TrainingArguments, Trainer
)
from huggingface_hub import HfApi, Repository
import numpy as np
from collections import defaultdict

class ModelMetadata:
    """Model metadata container"""
    
    def __init__(self, model_info: Dict[str, Any]):
        self.model_id = model_info.get("model_id")
        self.model_name = model_info.get("model_name")
        self.version = model_info.get("version", "1.0.0")
        self.country_code = model_info.get("country_code")
        self.base_model = model_info.get("base_model")
        self.training_config = model_info.get("training_config", {})
        self.performance_metrics = model_info.get("performance_metrics", {})
        self.created_at = model_info.get("created_at", datetime.now().isoformat())
        self.updated_at = model_info.get("updated_at", datetime.now().isoformat())
        self.file_path = model_info.get("file_path")
        self.file_size = model_info.get("file_size", 0)
        self.checksum = model_info.get("checksum")
        self.tags = model_info.get("tags", [])
        self.description = model_info.get("description", "")
        self.training_data_info = model_info.get("training_data_info", {})
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "version": self.version,
            "country_code": self.country_code,
            "base_model": self.base_model,
            "training_config": self.training_config,
            "performance_metrics": self.performance_metrics,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "file_path": self.file_path,
            "file_size": self.file_size,
            "checksum": self.checksum,
            "tags": self.tags,
            "description": self.description,
            "training_data_info": self.training_data_info
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create from dictionary"""
        return cls(data)

class ModelVersioning:
    """Model versioning system"""
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.versions_file = models_dir / "versions.json"
        self.versions_data = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version data"""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {"models": {}, "next_version": 1}
    
    def _save_versions(self) -> None:
        """Save version data"""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions_data, f, indent=2)
    
    def get_next_version(self, model_name: str) -> str:
        """Get next version number for model"""
        if model_name not in self.versions_data["models"]:
            return "1.0.0"
        
        versions = self.versions_data["models"][model_name]["versions"]
        if not versions:
            return "1.0.0"
        
        # Get latest version and increment
        latest = max(versions.keys(), key=lambda x: [int(i) for i in x.split('.')])
        major, minor, patch = map(int, latest.split('.'))
        
        return f"{major}.{minor}.{patch + 1}"
    
    def register_version(self, model_name: str, version: str, metadata: ModelMetadata) -> None:
        """Register new model version"""
        if model_name not in self.versions_data["models"]:
            self.versions_data["models"][model_name] = {
                "versions": {},
                "latest": version
            }
        
        self.versions_data["models"][model_name]["versions"][version] = metadata.to_dict()
        self.versions_data["models"][model_name]["latest"] = version
        
        self._save_versions()
    
    def get_version_info(self, model_name: str, version: Optional[str] = None) -> Optional[ModelMetadata]:
        """Get version information"""
        if model_name not in self.versions_data["models"]:
            return None
        
        model_data = self.versions_data["models"][model_name]
        
        if version is None:
            version = model_data["latest"]
        
        if version not in model_data["versions"]:
            return None
        
        return ModelMetadata.from_dict(model_data["versions"][version])
    
    def list_versions(self, model_name: str) -> List[str]:
        """List all versions for a model"""
        if model_name not in self.versions_data["models"]:
            return []
        
        versions = list(self.versions_data["models"][model_name]["versions"].keys())
        return sorted(versions, key=lambda x: [int(i) for i in x.split('.')], reverse=True)
    
    def list_models(self) -> List[str]:
        """List all models"""
        return list(self.versions_data["models"].keys())
    
    def delete_version(self, model_name: str, version: str) -> bool:
        """Delete a model version"""
        if model_name not in self.versions_data["models"]:
            return False
        
        model_data = self.versions_data["models"][model_name]
        
        if version not in model_data["versions"]:
            return False
        
        # Remove version
        del model_data["versions"][version]
        
        # Update latest if necessary
        if model_data["latest"] == version:
            remaining_versions = list(model_data["versions"].keys())
            if remaining_versions:
                model_data["latest"] = max(remaining_versions, 
                                          key=lambda x: [int(i) for i in x.split('.')])
            else:
                # No versions left, remove model
                del self.versions_data["models"][model_name]
        
        self._save_versions()
        return True

class ModelManager:
    """Main model manager for DAPT training"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
        self.country_code = config["country"]["code"]
        
        # Model paths
        self.models_dir = Path(global_config["models_dir"]) / self.country_code
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoints_dir = self.models_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        self.exports_dir = self.models_dir / "exports"
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        
        # Versioning
        self.versioning = ModelVersioning(self.models_dir)
        
        # Configuration
        self.base_model_name = config["model"]["base_model"]
        self.model_max_length = config["model"].get("max_length", 512)
        
        # Logging
        self.logger = self._setup_logging()
        
        self.logger.info(f"Model Manager initialized for country: {self.country_code}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for model manager"""
        logger = logging.getLogger(f"dapt_model_manager_{self.country_code}")
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
    
    def load_base_model(self) -> Tuple[Optional[AutoModel], Optional[AutoTokenizer], Optional[AutoConfig]]:
        """Load base model for DAPT training"""
        try:
            self.logger.info(f"Loading base model: {self.base_model_name}")
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.base_model_name,
                model_max_length=self.model_max_length
            )
            
            # Load config
            config = AutoConfig.from_pretrained(self.base_model_name)
            
            # Load model
            model = AutoModel.from_pretrained(
                self.base_model_name,
                config=config
            )
            
            self.logger.info(f"Successfully loaded base model: {self.base_model_name}")
            return model, tokenizer, config
            
        except Exception as e:
            self.logger.error(f"Error loading base model: {str(e)}")
            return None, None, None
    
    def save_model(self, model: AutoModel, tokenizer: AutoTokenizer, 
                   model_name: str, training_config: Dict[str, Any],
                   performance_metrics: Dict[str, Any],
                   training_data_info: Dict[str, Any],
                   description: str = "",
                   tags: List[str] = None) -> Optional[str]:
        """Save trained model with metadata"""
        try:
            if tags is None:
                tags = []
            
            # Generate model ID and version
            model_id = self._generate_model_id(model_name)
            version = self.versioning.get_next_version(model_name)
            
            # Create model directory
            model_dir = self.models_dir / model_name / version
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            
            # Calculate file size and checksum
            file_size = self._calculate_directory_size(model_dir)
            checksum = self._calculate_directory_checksum(model_dir)
            
            # Create metadata
            metadata = ModelMetadata({
                "model_id": model_id,
                "model_name": model_name,
                "version": version,
                "country_code": self.country_code,
                "base_model": self.base_model_name,
                "training_config": training_config,
                "performance_metrics": performance_metrics,
                "file_path": str(model_dir),
                "file_size": file_size,
                "checksum": checksum,
                "tags": tags,
                "description": description,
                "training_data_info": training_data_info
            })
            
            # Save metadata
            metadata_file = model_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            # Register version
            self.versioning.register_version(model_name, version, metadata)
            
            self.logger.info(f"Model saved: {model_name} v{version} ({model_id})")
            return model_id
            
        except Exception as e:
            self.logger.error(f"Error saving model: {str(e)}")
            return None
    
    def load_model(self, model_name: str, version: Optional[str] = None) -> Tuple[Optional[AutoModel], Optional[AutoTokenizer], Optional[ModelMetadata]]:
        """Load saved model"""
        try:
            # Get version info
            metadata = self.versioning.get_version_info(model_name, version)
            if metadata is None:
                self.logger.error(f"Model not found: {model_name} v{version or 'latest'}")
                return None, None, None
            
            model_dir = Path(metadata.file_path)
            
            if not model_dir.exists():
                self.logger.error(f"Model directory not found: {model_dir}")
                return None, None, None
            
            # Load model and tokenizer
            model = AutoModel.from_pretrained(model_dir)
            tokenizer = AutoTokenizer.from_pretrained(model_dir)
            
            self.logger.info(f"Model loaded: {model_name} v{metadata.version}")
            return model, tokenizer, metadata
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return None, None, None
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> bool:
        """Delete model version"""
        try:
            # Get version info
            metadata = self.versioning.get_version_info(model_name, version)
            if metadata is None:
                self.logger.error(f"Model not found: {model_name} v{version or 'latest'}")
                return False
            
            model_dir = Path(metadata.file_path)
            
            # Remove model directory
            if model_dir.exists():
                shutil.rmtree(model_dir)
            
            # Remove from versioning
            self.versioning.delete_version(model_name, metadata.version)
            
            self.logger.info(f"Model deleted: {model_name} v{metadata.version}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting model: {str(e)}")
            return False
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all models with their metadata"""
        try:
            models_info = []
            
            for model_name in self.versioning.list_models():
                versions = self.versioning.list_versions(model_name)
                latest_metadata = self.versioning.get_version_info(model_name)
                
                model_info = {
                    "model_name": model_name,
                    "latest_version": latest_metadata.version if latest_metadata else None,
                    "total_versions": len(versions),
                    "versions": versions,
                    "country_code": self.country_code,
                    "base_model": latest_metadata.base_model if latest_metadata else None,
                    "created_at": latest_metadata.created_at if latest_metadata else None,
                    "updated_at": latest_metadata.updated_at if latest_metadata else None,
                    "performance_metrics": latest_metadata.performance_metrics if latest_metadata else {},
                    "tags": latest_metadata.tags if latest_metadata else []
                }
                
                models_info.append(model_info)
            
            # Sort by update time (newest first)
            models_info.sort(key=lambda x: x["updated_at"] or "", reverse=True)
            
            return models_info
            
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return []
    
    def get_model_info(self, model_name: str, version: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get detailed model information"""
        try:
            metadata = self.versioning.get_version_info(model_name, version)
            if metadata is None:
                return None
            
            model_dir = Path(metadata.file_path)
            
            info = metadata.to_dict()
            info.update({
                "exists": model_dir.exists(),
                "versions_available": self.versioning.list_versions(model_name),
                "is_latest": version is None or version == self.versioning.get_version_info(model_name).version
            })
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting model info: {str(e)}")
            return None
    
    def compare_models(self, model_specs: List[Tuple[str, Optional[str]]]) -> Dict[str, Any]:
        """Compare multiple models"""
        try:
            comparison = {
                "models": [],
                "metrics_comparison": {},
                "summary": {}
            }
            
            models_data = []
            
            for model_name, version in model_specs:
                metadata = self.versioning.get_version_info(model_name, version)
                if metadata:
                    models_data.append(metadata)
                    comparison["models"].append({
                        "model_name": model_name,
                        "version": metadata.version,
                        "performance_metrics": metadata.performance_metrics,
                        "file_size": metadata.file_size,
                        "created_at": metadata.created_at
                    })
            
            if not models_data:
                return comparison
            
            # Compare metrics
            all_metrics = set()
            for metadata in models_data:
                all_metrics.update(metadata.performance_metrics.keys())
            
            for metric in all_metrics:
                metric_values = []
                for metadata in models_data:
                    value = metadata.performance_metrics.get(metric)
                    if value is not None:
                        metric_values.append(value)
                
                if metric_values:
                    comparison["metrics_comparison"][metric] = {
                        "values": metric_values,
                        "best": max(metric_values),
                        "worst": min(metric_values),
                        "average": sum(metric_values) / len(metric_values)
                    }
            
            # Summary
            comparison["summary"] = {
                "total_models": len(models_data),
                "metrics_compared": len(all_metrics),
                "size_range": {
                    "min": min(m.file_size for m in models_data),
                    "max": max(m.file_size for m in models_data)
                }
            }
            
            return comparison
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            return {}
    
    def export_model(self, model_name: str, version: Optional[str] = None,
                    export_format: str = "pytorch", export_path: Optional[str] = None) -> Optional[str]:
        """Export model in specified format"""
        try:
            # Load model
            model, tokenizer, metadata = self.load_model(model_name, version)
            if model is None:
                return None
            
            # Determine export path
            if export_path is None:
                export_filename = f"{model_name}_v{metadata.version}_{export_format}"
                export_path = self.exports_dir / export_filename
            else:
                export_path = Path(export_path)
            
            export_path.parent.mkdir(parents=True, exist_ok=True)
            
            if export_format.lower() == "pytorch":
                # Save as PyTorch model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'tokenizer': tokenizer,
                    'metadata': metadata.to_dict()
                }, export_path.with_suffix('.pt'))
                final_path = export_path.with_suffix('.pt')
            
            elif export_format.lower() == "onnx":
                # Export to ONNX (requires additional setup)
                try:
                    import torch.onnx
                    
                    # Create dummy input
                    dummy_input = torch.randint(0, 1000, (1, self.model_max_length))
                    
                    torch.onnx.export(
                        model,
                        dummy_input,
                        export_path.with_suffix('.onnx'),
                        export_params=True,
                        opset_version=11,
                        do_constant_folding=True,
                        input_names=['input_ids'],
                        output_names=['output'],
                        dynamic_axes={
                            'input_ids': {0: 'batch_size', 1: 'sequence'},
                            'output': {0: 'batch_size', 1: 'sequence'}
                        }
                    )
                    final_path = export_path.with_suffix('.onnx')
                    
                except ImportError:
                    self.logger.error("ONNX export requires torch.onnx")
                    return None
            
            elif export_format.lower() == "huggingface":
                # Export as Hugging Face model
                hf_export_dir = export_path
                hf_export_dir.mkdir(parents=True, exist_ok=True)
                
                model.save_pretrained(hf_export_dir)
                tokenizer.save_pretrained(hf_export_dir)
                
                # Save additional metadata
                with open(hf_export_dir / "dapt_metadata.json", 'w') as f:
                    json.dump(metadata.to_dict(), f, indent=2)
                
                final_path = hf_export_dir
            
            else:
                self.logger.error(f"Unsupported export format: {export_format}")
                return None
            
            self.logger.info(f"Model exported: {model_name} v{metadata.version} -> {final_path}")
            return str(final_path)
            
        except Exception as e:
            self.logger.error(f"Error exporting model: {str(e)}")
            return None
    
    def _generate_model_id(self, model_name: str) -> str:
        """Generate unique model ID"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        country_prefix = self.country_code.lower()
        return f"{country_prefix}_{model_name}_{timestamp}"
    
    def _calculate_directory_size(self, directory: Path) -> int:
        """Calculate total size of directory"""
        total_size = 0
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def _calculate_directory_checksum(self, directory: Path) -> str:
        """Calculate checksum for directory contents"""
        hasher = hashlib.md5()
        
        for file_path in sorted(directory.rglob('*')):
            if file_path.is_file():
                hasher.update(str(file_path.relative_to(directory)).encode())
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hasher.update(chunk)
        
        return hasher.hexdigest()
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 5) -> int:
        """Clean up old checkpoint files"""
        try:
            checkpoint_files = list(self.checkpoints_dir.glob('checkpoint-*'))
            
            if len(checkpoint_files) <= keep_last_n:
                return 0
            
            # Sort by modification time
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old checkpoints
            removed_count = 0
            for checkpoint_dir in checkpoint_files[keep_last_n:]:
                if checkpoint_dir.is_dir():
                    shutil.rmtree(checkpoint_dir)
                    removed_count += 1
            
            self.logger.info(f"Cleaned up {removed_count} old checkpoints")
            return removed_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up checkpoints: {str(e)}")
            return 0
    
    def get_storage_info(self) -> Dict[str, Any]:
        """Get storage information"""
        try:
            models_size = self._calculate_directory_size(self.models_dir)
            checkpoints_size = self._calculate_directory_size(self.checkpoints_dir)
            exports_size = self._calculate_directory_size(self.exports_dir)
            
            return {
                "models_directory": str(self.models_dir),
                "total_models": len(self.versioning.list_models()),
                "storage_usage": {
                    "models_size_bytes": models_size,
                    "models_size_mb": models_size / (1024 * 1024),
                    "checkpoints_size_bytes": checkpoints_size,
                    "checkpoints_size_mb": checkpoints_size / (1024 * 1024),
                    "exports_size_bytes": exports_size,
                    "exports_size_mb": exports_size / (1024 * 1024),
                    "total_size_bytes": models_size + checkpoints_size + exports_size,
                    "total_size_mb": (models_size + checkpoints_size + exports_size) / (1024 * 1024)
                },
                "directories": {
                    "models": str(self.models_dir),
                    "checkpoints": str(self.checkpoints_dir),
                    "exports": str(self.exports_dir)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error getting storage info: {str(e)}")
            return {}