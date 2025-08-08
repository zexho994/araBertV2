"""Configuration Manager for NER System

Handles loading, validation, and management of country-specific configurations.
Provides centralized access to all configuration settings.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from copy import deepcopy

# Try to import json5 for enhanced JSON parsing with comments support
try:
    import json5
    HAS_JSON5 = True
except ImportError:
    json5 = None
    HAS_JSON5 = False

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
        # Try to load .json5 file first if json5 is available
        if HAS_JSON5:
            json5_file = self.templates_dir / f"{template}.json5"
            if json5_file.exists():
                try:
                    with open(json5_file, 'r', encoding='utf-8') as f:
                        return json5.load(f)
                except Exception as e:
                    raise ValueError(f"Invalid JSON5 in template file {json5_file}: {e}")
        
        # Fall back to .json file
        template_file = self.templates_dir / f"{template}.json"
        
        if not template_file.exists():
            raise FileNotFoundError(
                f"Template '{template}' not found at {template_file} or {template}.json5"
            )
        
        try:
            with open(template_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in template file {template_file}: {e}")
    
    def load_external_template(self, template_path: str) -> Dict[str, Any]:
        """Load a configuration template from an external file path
        
        Args:
            template_path: Full path to the template file
            
        Returns:
            Dictionary containing template configuration
        """
        template_file = Path(template_path)
        
        if not template_file.exists():
            raise FileNotFoundError(
                f"External template not found at {template_file}"
            )
        
        # Determine file format based on extension
        if template_file.suffix.lower() == '.json5' and HAS_JSON5:
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    return json5.load(f)
            except Exception as e:
                raise ValueError(f"Invalid JSON5 in external template file {template_file}: {e}")
        elif template_file.suffix.lower() in ['.json', '.jsonl']:
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    # Handle JSONL format (load first line only)
                    if template_file.suffix.lower() == '.jsonl':
                        first_line = f.readline().strip()
                        if not first_line:
                            raise ValueError("JSONL file is empty")
                        return json.loads(first_line)
                    else:
                        return json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in external template file {template_file}: {e}")
        else:
            raise ValueError(
                f"Unsupported template file format: {template_file.suffix}. "
                f"Supported formats: .json, .json5{', .jsonl' if HAS_JSON5 else ''}"
            )
    
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
    
    def create_country_config(self, country: str, template: str = "default", 
                                external_template_path: Optional[str] = None) -> Dict[str, Any]:
        """Create a new country configuration from template
        
        Args:
            country: Country code for new configuration
            template: Template to use as base (ignored if external_template_path is provided)
            external_template_path: Optional path to external template file
            
        Returns:
            New configuration dictionary
        """
        # Load template from external path or standard template
        if external_template_path:
            config = self.load_external_template(external_template_path)
            # Fix DAPT template structure for NER compatibility
            self._fix_template_structure(config)
        else:
            config = self.load_template_config(template)
        
        # Apply placeholder replacement
        config = self._replace_placeholders(config, country)
        
        # Update country-specific information (in case placeholders weren't used)
        if "country" in config:
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
        
        templates = set()
        
        # Add .json5 templates if json5 is available
        if HAS_JSON5:
            for template_file in self.templates_dir.glob("*.json5"):
                templates.add(template_file.stem)
        
        # Add .json templates
        for template_file in self.templates_dir.glob("*.json"):
            templates.add(template_file.stem)
        
        return sorted(list(templates))
    
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
    
    def _replace_placeholders(self, config: Dict[str, Any], country: str) -> Dict[str, Any]:
        """Replace placeholders in configuration template
        
        Args:
            config: Configuration dictionary with placeholders
            country: Country code to use for replacement
            
        Returns:
            Configuration dictionary with placeholders replaced
        """
        # Convert config to JSON string for placeholder replacement
        config_str = json.dumps(config, ensure_ascii=False, indent=2)
        
        # Generate country name from country code (capitalize and replace underscores)
        country_name = country.replace('_', ' ').title()
        
        # Replace placeholders
        config_str = config_str.replace('{country_code}', country)
        config_str = config_str.replace('{country_name}', country_name)
        
        # Parse back to dictionary
        try:
            return json.loads(config_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error parsing configuration after placeholder replacement: {e}")
    
    def _fix_template_structure(self, config: Dict[str, Any]) -> None:
        """Fix template structure for NER compatibility
        
        This method handles differences between DAPT and NER template structures,
        particularly moving num_labels from model section to labels section and
        fixing model field names.
        
        Args:
            config: Configuration dictionary to fix (modified in-place)
        """
        # Fix model section fields
        if 'model' in config:
            model_config = config['model']
            
            # Convert base_model to name and pretrained_model
            if 'base_model' in model_config and 'name' not in model_config:
                model_config['name'] = model_config['base_model'].split('/')[-1]  # Extract model name
                model_config['pretrained_model'] = model_config['base_model']
            
            # Add missing type field
            if 'type' not in model_config:
                model_config['type'] = 'bert'  # Default to bert type
        
        # Check if num_labels is in model section but missing from labels section
        if ('model' in config and 'num_labels' in config['model'] and 
            'labels' in config and 'num_labels' not in config['labels']):
            
            # Move num_labels from model to labels section
            config['labels']['num_labels'] = config['model']['num_labels']
            
        # If labels section exists but num_labels is still missing, calculate it
        if ('labels' in config and 'num_labels' not in config['labels'] and 
            'label_names' in config['labels']):
            
            config['labels']['num_labels'] = len(config['labels']['label_names'])
    
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