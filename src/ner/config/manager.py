"""Configuration Manager for NER System

Handles loading, validation, and management of country-specific configurations.
Provides centralized access to all configuration settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from copy import deepcopy

class ConfigManager:
    """Manages NER configuration files and settings"""
    
    def __init__(self, config_dir: str = "data/ner/configs"):
        """Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.countries_dir = self.config_dir / "countries"
        self.templates_dir = self.config_dir / "templates"
        self._config_cache = {}
        
        # Ensure directories exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.countries_dir.mkdir(parents=True, exist_ok=True)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
    
    def load_country_config(self, country: str, use_cache: bool = True) -> Dict[str, Any]:
        """Load configuration for a specific country
        
        Args:
            country: Country code (e.g., 'uae', 'saudi')
            use_cache: Whether to use cached configuration
            
        Returns:
            Dictionary containing country configuration
            
        Raises:
            FileNotFoundError: If country configuration file doesn't exist
            ValueError: If configuration is invalid
        """
        if use_cache and country in self._config_cache:
            return deepcopy(self._config_cache[country])
        
        config_file = self.countries_dir / f"{country}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file for country '{country}' not found at {config_file}"
            )
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate configuration
            self._validate_config(config, country)
            
            # Cache the configuration
            self._config_cache[country] = deepcopy(config)
            
            return config
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file {config_file}: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration for {country}: {e}")
    
    def load_template_config(self, template: str) -> Dict[str, Any]:
        """Load a configuration template
        
        Args:
            template: Template name (e.g., 'default', 'address_ner')
            
        Returns:
            Dictionary containing template configuration
        """
        template_file = self.templates_dir / f"{template}.json"
        
        if not template_file.exists():
            raise FileNotFoundError(
                f"Template '{template}' not found at {template_file}"
            )
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in template file {template_file}: {e}")
    
    def save_country_config(self, country: str, config: Dict[str, Any]) -> None:
        """Save configuration for a country
        
        Args:
            country: Country code
            config: Configuration dictionary to save
        """
        # Validate configuration before saving
        self._validate_config(config, country)
        
        config_file = self.countries_dir / f"{country}.json"
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # Update cache
            self._config_cache[country] = deepcopy(config)
            
        except Exception as e:
            raise ValueError(f"Error saving configuration for {country}: {e}")
    
    def create_country_config(self, country: str, template: str = "default") -> Dict[str, Any]:
        """Create a new country configuration from template
        
        Args:
            country: Country code for new configuration
            template: Template to use as base
            
        Returns:
            New configuration dictionary
        """
        # Load template
        config = self.load_template_config(template)
        
        # Update country-specific information
        config["country"]["code"] = country
        config["country"]["name"] = country.upper()
        
        # Save the new configuration
        self.save_country_config(country, config)
        
        return config
    
    def list_countries(self) -> List[str]:
        """List all available country configurations
        
        Returns:
            List of country codes
        """
        if not self.countries_dir.exists():
            return []
        
        countries = []
        for config_file in self.countries_dir.glob("*.json"):
            countries.append(config_file.stem)
        
        return sorted(countries)
    
    def list_templates(self) -> List[str]:
        """List all available configuration templates
        
        Returns:
            List of template names
        """
        if not self.templates_dir.exists():
            return []
        
        templates = []
        for template_file in self.templates_dir.glob("*.json"):
            templates.append(template_file.stem)
        
        return sorted(templates)
    
    def country_exists(self, country: str) -> bool:
        """Check if a country configuration exists
        
        Args:
            country: Country code to check
            
        Returns:
            True if configuration exists, False otherwise
        """
        config_file = self.countries_dir / f"{country}.json"
        return config_file.exists()
    
    def delete_country_config(self, country: str) -> None:
        """Delete a country configuration
        
        Args:
            country: Country code to delete
        """
        config_file = self.countries_dir / f"{country}.json"
        
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration for country '{country}' not found")
        
        config_file.unlink()
        
        # Remove from cache
        if country in self._config_cache:
            del self._config_cache[country]
    
    def clear_cache(self) -> None:
        """Clear the configuration cache"""
        self._config_cache.clear()
    
    def _validate_config(self, config: Dict[str, Any], country: str) -> None:
        """Validate configuration structure and required fields
        
        Args:
            config: Configuration to validate
            country: Country code for context
            
        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = [
            'country', 'model', 'training', 'data', 'labels', 
            'evaluation', 'output', 'hardware', 'logging'
        ]
        
        for section in required_sections:
            if section not in config:
                raise ValueError(
                    f"Missing required section '{section}' in {country} configuration"
                )
        
        # Validate country section
        country_config = config['country']
        if 'code' not in country_config or 'name' not in country_config:
            raise ValueError("Country section must contain 'code' and 'name' fields")
        
        # Validate labels section
        labels_config = config['labels']
        required_label_fields = ['num_labels', 'label_names', 'label_mapping']
        for field in required_label_fields:
            if field not in labels_config:
                raise ValueError(f"Labels section missing required field '{field}'")
        
        # Validate label consistency
        num_labels = labels_config['num_labels']
        label_names = labels_config['label_names']
        label_mapping = labels_config['label_mapping']
        
        if len(label_names) != num_labels:
            raise ValueError(
                f"Number of label names ({len(label_names)}) doesn't match num_labels ({num_labels})"
            )
        
        if len(label_mapping) != num_labels:
            raise ValueError(
                f"Number of label mappings ({len(label_mapping)}) doesn't match num_labels ({num_labels})"
            )
        
        # Validate model section
        model_config = config['model']
        required_model_fields = ['name', 'type', 'pretrained_model']
        for field in required_model_fields:
            if field not in model_config:
                raise ValueError(f"Model section missing required field '{field}'")
    
    def get_config_summary(self, country: str) -> Dict[str, Any]:
        """Get a summary of configuration for a country
        
        Args:
            country: Country code
            
        Returns:
            Dictionary with configuration summary
        """
        config = self.load_country_config(country)
        
        return {
            'country': config['country'],
            'model_name': config['model']['name'],
            'model_type': config['model']['type'],
            'num_labels': config['labels']['num_labels'],
            'training_epochs': config['training']['epochs'],
            'batch_size': config['training']['batch_size'],
            'learning_rate': config['training']['learning_rate']
        }