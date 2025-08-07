"""NER Model Manager

Handles model loading, saving, versioning, and management operations.
"""

import os
import json
import torch
import shutil
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from datetime import datetime
import hashlib

from .model import NERModel, BertNERModel
from ..utils import NERLogger

class NERModelManager:
    """Manager for NER model operations"""
    
    def __init__(self, model_dir: str = "data/ner/models"):
        """
        Initialize Model Manager
        
        Args:
            model_dir: Base directory for model storage
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = NERLogger(name="model_manager")
        
        # Model registry file
        self.registry_file = self.model_dir / "model_registry.json"
        self.registry = self._load_registry()
    
    def _load_registry(self) -> Dict[str, Any]:
        """Load model registry"""
        if self.registry_file.exists():
            with open(self.registry_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {'models': {}}
    
    def _save_registry(self):
        """Save model registry"""
        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.registry, f, indent=2, ensure_ascii=False)
    
    def _generate_model_id(self, country: str, model_type: str, timestamp: str) -> str:
        """Generate unique model ID"""
        base_string = f"{country}_{model_type}_{timestamp}"
        return hashlib.md5(base_string.encode()).hexdigest()[:8]
    
    def register_model(self, model_path: str, country: str, model_type: str = "bert",
                      description: str = "", metadata: Optional[Dict[str, Any]] = None) -> str:
        """Register a new model
        
        Args:
            model_path: Path to model directory
            country: Country code
            model_type: Type of model
            description: Model description
            metadata: Additional metadata
            
        Returns:
            Model ID
        """
        timestamp = datetime.now().isoformat()
        model_id = self._generate_model_id(country, model_type, timestamp)
        
        # Create model entry
        model_entry = {
            'id': model_id,
            'country': country,
            'type': model_type,
            'path': str(model_path),
            'description': description,
            'created_at': timestamp,
            'metadata': metadata or {},
            'status': 'active'
        }
        
        # Add to registry
        self.registry['models'][model_id] = model_entry
        self._save_registry()
        
        self.logger.info(f"Registered model {model_id} for {country}")
        return model_id
    
    def load_model(self, model_identifier: str) -> NERModel:
        """Load model by ID or path
        
        Args:
            model_identifier: Model ID or path
            
        Returns:
            Loaded NER model
        """
        # Check if it's a model ID
        if model_identifier in self.registry['models']:
            model_entry = self.registry['models'][model_identifier]
            model_path = model_entry['path']
            model_type = model_entry['type']
        else:
            # Assume it's a path
            model_path = model_identifier
            model_type = self._detect_model_type(model_path)
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load configuration
        config_path = model_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
                config = config_data.get('config', {})
                label2id = config_data.get('label2id', {})
                id2label = config_data.get('id2label', {})
        else:
            raise FileNotFoundError(f"Model configuration not found: {config_path}")
        
        # Initialize model
        if model_type == 'bert':
            model = BertNERModel.from_pretrained(
                str(model_path),
                num_labels=len(label2id),
                id2label=id2label,
                label2id=label2id
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Set additional attributes
        model.config_data = config
        model.country = config.get('country', 'unknown')
        
        self.logger.info(f"Loaded model from {model_path}")
        return model
    
    def _detect_model_type(self, model_path: str) -> str:
        """Detect model type from path"""
        model_path = Path(model_path)
        
        # Check for BERT model files
        if (model_path / "pytorch_model.bin").exists() or (model_path / "model.safetensors").exists():
            return "bert"
        
        # Default to bert
        return "bert"
    
    def save_model(self, model: NERModel, country: str, description: str = "",
                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save model and register it
        
        Args:
            model: Model to save
            country: Country code
            description: Model description
            metadata: Additional metadata
            
        Returns:
            Model ID
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{country}_model_{timestamp}"
        model_path = self.model_dir / country / model_name
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model.save_pretrained(str(model_path))
        
        # Save tokenizer if available
        if hasattr(model, 'tokenizer') and model.tokenizer:
            model.tokenizer.save_pretrained(str(model_path))
        
        # Save configuration
        config_data = {
            'config': getattr(model, 'config_data', {}),
            'label2id': getattr(model, 'config', {}).label2id or {},
            'id2label': getattr(model, 'config', {}).id2label or {},
            'country': country,
            'saved_at': datetime.now().isoformat()
        }
        
        config_path = model_path / "config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_data, f, indent=2, ensure_ascii=False)
        
        # Register model
        model_id = self.register_model(
            str(model_path), country, model.config.model_type if hasattr(model.config, 'model_type') else 'bert',
            description, metadata
        )
        
        self.logger.info(f"Saved model {model_id} to {model_path}")
        return model_id
    
    def list_models(self, country: Optional[str] = None, status: str = "active") -> List[Dict[str, Any]]:
        """List available models
        
        Args:
            country: Filter by country (optional)
            status: Filter by status
            
        Returns:
            List of model information
        """
        models = []
        
        for model_id, model_entry in self.registry['models'].items():
            if status and model_entry.get('status') != status:
                continue
            
            if country and model_entry.get('country') != country:
                continue
            
            models.append(model_entry.copy())
        
        # Sort by creation date (newest first)
        models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return models
    
    def get_model_info(self, model_identifier: str) -> Dict[str, Any]:
        """Get detailed model information
        
        Args:
            model_identifier: Model ID or path
            
        Returns:
            Model information
        """
        # Check if it's a model ID
        if model_identifier in self.registry['models']:
            model_entry = self.registry['models'][model_identifier].copy()
            model_path = Path(model_entry['path'])
        else:
            # Assume it's a path
            model_path = Path(model_identifier)
            model_entry = {
                'id': 'unknown',
                'path': str(model_path),
                'type': self._detect_model_type(str(model_path))
            }
        
        # Add file system information
        if model_path.exists():
            model_entry['exists'] = True
            model_entry['size_mb'] = self._get_directory_size(model_path) / (1024 * 1024)
            
            # Load configuration if available
            config_path = model_path / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                    model_entry['config'] = config_data
        else:
            model_entry['exists'] = False
        
        return model_entry
    
    def _get_directory_size(self, path: Path) -> int:
        """Get total size of directory in bytes"""
        total_size = 0
        for file_path in path.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size
    
    def delete_model(self, model_identifier: str, remove_files: bool = True) -> bool:
        """Delete model
        
        Args:
            model_identifier: Model ID or path
            remove_files: Whether to remove model files
            
        Returns:
            True if successful
        """
        try:
            # Check if it's a model ID
            if model_identifier in self.registry['models']:
                model_entry = self.registry['models'][model_identifier]
                model_path = Path(model_entry['path'])
                
                # Mark as deleted in registry
                self.registry['models'][model_identifier]['status'] = 'deleted'
                self.registry['models'][model_identifier]['deleted_at'] = datetime.now().isoformat()
                self._save_registry()
                
                model_id = model_identifier
            else:
                # Assume it's a path
                model_path = Path(model_identifier)
                model_id = model_identifier
            
            # Remove files if requested
            if remove_files and model_path.exists():
                shutil.rmtree(model_path)
                self.logger.info(f"Removed model files: {model_path}")
            
            self.logger.info(f"Deleted model {model_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete model {model_identifier}: {e}")
            return False
    
    def model_exists(self, model_identifier: str) -> bool:
        """Check if model exists
        
        Args:
            model_identifier: Model ID or path
            
        Returns:
            True if model exists
        """
        # Check if it's a model ID
        if model_identifier in self.registry['models']:
            model_entry = self.registry['models'][model_identifier]
            if model_entry.get('status') != 'active':
                return False
            model_path = Path(model_entry['path'])
        else:
            # Assume it's a path
            model_path = Path(model_identifier)
        
        return model_path.exists()
    
    def get_latest_model(self, country: str) -> Optional[str]:
        """Get latest model for country
        
        Args:
            country: Country code
            
        Returns:
            Model ID or None
        """
        models = self.list_models(country=country)
        
        if models:
            return models[0]['id']  # Already sorted by creation date
        
        return None
    
    def backup_model(self, model_identifier: str, backup_dir: str) -> str:
        """Backup model to specified directory
        
        Args:
            model_identifier: Model ID or path
            backup_dir: Backup directory
            
        Returns:
            Backup path
        """
        # Get model path
        if model_identifier in self.registry['models']:
            model_entry = self.registry['models'][model_identifier]
            model_path = Path(model_entry['path'])
            model_id = model_identifier
        else:
            model_path = Path(model_identifier)
            model_id = model_path.name
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Create backup
        backup_path = Path(backup_dir) / f"{model_id}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        shutil.copytree(model_path, backup_path / model_path.name)
        
        self.logger.info(f"Backed up model {model_id} to {backup_path}")
        return str(backup_path)
    
    def restore_model(self, backup_path: str, target_path: Optional[str] = None) -> str:
        """Restore model from backup
        
        Args:
            backup_path: Path to backup
            target_path: Target restoration path (optional)
            
        Returns:
            Restored model path
        """
        backup_path = Path(backup_path)
        
        if not backup_path.exists():
            raise FileNotFoundError(f"Backup not found: {backup_path}")
        
        # Find model directory in backup
        model_dirs = [d for d in backup_path.iterdir() if d.is_dir()]
        if not model_dirs:
            raise ValueError(f"No model directory found in backup: {backup_path}")
        
        model_backup_dir = model_dirs[0]
        
        # Determine target path
        if target_path:
            target_path = Path(target_path)
        else:
            target_path = self.model_dir / model_backup_dir.name
        
        # Restore model
        if target_path.exists():
            shutil.rmtree(target_path)
        
        shutil.copytree(model_backup_dir, target_path)
        
        self.logger.info(f"Restored model from {backup_path} to {target_path}")
        return str(target_path)
    
    def cleanup_deleted_models(self) -> int:
        """Clean up models marked as deleted
        
        Returns:
            Number of models cleaned up
        """
        cleaned_count = 0
        
        for model_id, model_entry in list(self.registry['models'].items()):
            if model_entry.get('status') == 'deleted':
                model_path = Path(model_entry['path'])
                
                # Remove files if they still exist
                if model_path.exists():
                    shutil.rmtree(model_path)
                    self.logger.info(f"Cleaned up deleted model files: {model_path}")
                
                # Remove from registry
                del self.registry['models'][model_id]
                cleaned_count += 1
        
        if cleaned_count > 0:
            self._save_registry()
            self.logger.info(f"Cleaned up {cleaned_count} deleted models")
        
        return cleaned_count
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get model statistics
        
        Returns:
            Statistics dictionary
        """
        stats = {
            'total_models': 0,
            'active_models': 0,
            'deleted_models': 0,
            'countries': set(),
            'model_types': {},
            'total_size_mb': 0
        }
        
        for model_entry in self.registry['models'].values():
            stats['total_models'] += 1
            
            status = model_entry.get('status', 'active')
            if status == 'active':
                stats['active_models'] += 1
                stats['countries'].add(model_entry.get('country', 'unknown'))
                
                model_type = model_entry.get('type', 'unknown')
                stats['model_types'][model_type] = stats['model_types'].get(model_type, 0) + 1
                
                # Calculate size
                model_path = Path(model_entry['path'])
                if model_path.exists():
                    stats['total_size_mb'] += self._get_directory_size(model_path) / (1024 * 1024)
            
            elif status == 'deleted':
                stats['deleted_models'] += 1
        
        stats['countries'] = list(stats['countries'])
        
        return stats