"""NER Module - Named Entity Recognition for Address Parsing

This module provides comprehensive tools for training, evaluating, and deploying
Named Entity Recognition models specifically designed for address parsing across
different countries and regions.

Key Features:
- Multi-country address parsing support
- Configurable label schemas for different address formats
- Training pipeline with BERT-based models
- Evaluation and metrics reporting
- CLI interface for easy operation
- Model management and versioning

Main Components:
- cli: Command-line interface
- config: Configuration management
- data: Data processing and loading
- models: Model definitions and management
- training: Training pipeline
- evaluation: Model evaluation and metrics
- utils: Utility functions and helpers

Example Usage:
    from ner.config import ConfigManager
    from ner.training import NERTrainer
    from ner.models import NERModelManager
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_country_config('uae')
    
    # Initialize trainer
    trainer = NERTrainer(config)
    
    # Train model
    trainer.train()
"""

__version__ = "1.0.0"
__author__ = "NER Development Team"
__email__ = "ner-dev@example.com"

# Import main classes for easy access
try:
    from .config import ConfigManager, ConfigValidator
    from .models import NERModelManager, NERModel
    from .training import NERTrainer
    from .evaluation import NEREvaluator
    from .data import NERDataProcessor, NERDataLoader
    from .utils import NERLogger, setup_logging
except ImportError:
    # Handle cases where dependencies might not be available
    pass

# Define what gets imported with "from ner import *"
__all__ = [
    # Core classes
    'ConfigManager',
    'ConfigValidator', 
    'NERModelManager',
    'NERModel',
    'NERTrainer',
    'NEREvaluator',
    'NERDataProcessor',
    'NERDataLoader',
    'NERLogger',
    'setup_logging',
    
    # Module metadata
    '__version__',
    '__author__',
    '__email__'
]

# Module-level configuration
DEFAULT_CONFIG_DIR = "data/ner/configs"
DEFAULT_DATA_DIR = "data/ner/training_data"
DEFAULT_MODEL_DIR = "data/ner/models"
DEFAULT_LOG_DIR = "data/ner/logs"
DEFAULT_EVAL_DIR = "data/ner/evaluation"

# Supported countries (can be extended)
SUPPORTED_COUNTRIES = [
    'uae',  # United Arab Emirates
    # Add more countries as needed
    # 'saudi',  # Saudi Arabia
    # 'qatar',  # Qatar
    # 'kuwait', # Kuwait
    # 'bahrain', # Bahrain
    # 'oman',   # Oman
]

# Default label schema for address parsing
DEFAULT_LABELS = [
    'O',  # Outside any entity
    'B-STREET', 'I-STREET',
    'B-BUILDING', 'I-BUILDING', 
    'B-AREA', 'I-AREA',
    'B-CITY', 'I-CITY',
    'B-COUNTRY', 'I-COUNTRY'
]

def get_version():
    """Get the current version of the NER module."""
    return __version__

def get_supported_countries():
    """Get list of supported countries."""
    return SUPPORTED_COUNTRIES.copy()

def get_default_labels():
    """Get default label schema."""
    return DEFAULT_LABELS.copy()