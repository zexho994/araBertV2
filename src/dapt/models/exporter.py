"""DAPT Model Exporter

Model export utilities for DAPT training:
- Export to different formats (PyTorch, ONNX, TensorFlow, Hugging Face)
- Model optimization and quantization
- Deployment-ready model packaging
- Integration with existing NER pipelines
- Model serving configurations
"""

import os
import json
import shutil
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datetime import datetime
import logging
from abc import ABC, abstractmethod
import pickle
import zipfile
from transformers import (
    AutoModel, AutoTokenizer, AutoConfig,
    AutoModelForTokenClassification,
    pipeline
)
from huggingface_hub import HfApi, Repository
import tempfile

class BaseExporter(ABC):
    """Abstract base class for model exporters"""
    
    @abstractmethod
    def export(self, model: AutoModel, tokenizer: AutoTokenizer, 
              output_path: str, **kwargs) -> bool:
        """Export model to specified format"""
        pass
    
    @abstractmethod
    def validate_export(self, export_path: str) -> bool:
        """Validate exported model"""
        pass

class PyTorchExporter(BaseExporter):
    """PyTorch model exporter"""
    
    def export(self, model: AutoModel, tokenizer: AutoTokenizer, 
              output_path: str, **kwargs) -> bool:
        """Export model as PyTorch format"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare export data
            export_data = {
                'model_state_dict': model.state_dict(),
                'model_config': model.config.to_dict(),
                'tokenizer_config': tokenizer.get_vocab(),
                'tokenizer_special_tokens': {
                    'pad_token': tokenizer.pad_token,
                    'unk_token': tokenizer.unk_token,
                    'cls_token': getattr(tokenizer, 'cls_token', None),
                    'sep_token': getattr(tokenizer, 'sep_token', None),
                    'mask_token': getattr(tokenizer, 'mask_token', None)
                },
                'export_metadata': {
                    'export_time': datetime.now().isoformat(),
                    'pytorch_version': torch.__version__,
                    'model_type': model.config.model_type,
                    **kwargs
                }
            }
            
            # Save as .pt file
            torch.save(export_data, output_path.with_suffix('.pt'))
            
            return True
            
        except Exception as e:
            print(f"Error exporting PyTorch model: {str(e)}")
            return False
    
    def validate_export(self, export_path: str) -> bool:
        """Validate PyTorch export"""
        try:
            export_path = Path(export_path).with_suffix('.pt')
            if not export_path.exists():
                return False
            
            # Try to load the exported model
            data = torch.load(export_path, map_location='cpu')
            required_keys = ['model_state_dict', 'model_config', 'tokenizer_config']
            
            return all(key in data for key in required_keys)
            
        except Exception:
            return False

class ONNXExporter(BaseExporter):
    """ONNX model exporter"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.opset_version = config.get('opset_version', 11)
        self.dynamic_axes = config.get('dynamic_axes', True)
    
    def export(self, model: AutoModel, tokenizer: AutoTokenizer, 
              output_path: str, **kwargs) -> bool:
        """Export model to ONNX format"""
        try:
            import torch.onnx
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Set model to evaluation mode
            model.eval()
            
            # Create dummy input
            max_length = kwargs.get('max_length', 512)
            batch_size = kwargs.get('batch_size', 1)
            
            dummy_input = torch.randint(
                0, tokenizer.vocab_size, 
                (batch_size, max_length),
                dtype=torch.long
            )
            
            # Define dynamic axes if enabled
            dynamic_axes = None
            if self.dynamic_axes:
                dynamic_axes = {
                    'input_ids': {0: 'batch_size', 1: 'sequence'},
                    'output': {0: 'batch_size', 1: 'sequence'}
                }
            
            # Export to ONNX
            torch.onnx.export(
                model,
                dummy_input,
                output_path.with_suffix('.onnx'),
                export_params=True,
                opset_version=self.opset_version,
                do_constant_folding=True,
                input_names=['input_ids'],
                output_names=['output'],
                dynamic_axes=dynamic_axes,
                verbose=False
            )
            
            # Save metadata
            metadata = {
                'model_type': model.config.model_type,
                'max_length': max_length,
                'vocab_size': tokenizer.vocab_size,
                'opset_version': self.opset_version,
                'dynamic_axes': self.dynamic_axes,
                'export_time': datetime.now().isoformat(),
                **kwargs
            }
            
            with open(output_path.with_suffix('.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except ImportError:
            print("ONNX export requires torch.onnx")
            return False
        except Exception as e:
            print(f"Error exporting ONNX model: {str(e)}")
            return False
    
    def validate_export(self, export_path: str) -> bool:
        """Validate ONNX export"""
        try:
            import onnx
            
            onnx_path = Path(export_path).with_suffix('.onnx')
            if not onnx_path.exists():
                return False
            
            # Load and check ONNX model
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            
            return True
            
        except ImportError:
            print("ONNX validation requires onnx package")
            return False
        except Exception:
            return False

class HuggingFaceExporter(BaseExporter):
    """Hugging Face Hub exporter"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.use_auth_token = config.get('use_auth_token', False)
        self.private = config.get('private', True)
    
    def export(self, model: AutoModel, tokenizer: AutoTokenizer, 
              output_path: str, **kwargs) -> bool:
        """Export model to Hugging Face format"""
        try:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save model and tokenizer
            model.save_pretrained(output_path)
            tokenizer.save_pretrained(output_path)
            
            # Create model card
            model_card = self._create_model_card(model, tokenizer, **kwargs)
            with open(output_path / "README.md", 'w', encoding='utf-8') as f:
                f.write(model_card)
            
            # Save additional metadata
            metadata = {
                'model_type': model.config.model_type,
                'country_code': kwargs.get('country_code'),
                'base_model': kwargs.get('base_model'),
                'training_config': kwargs.get('training_config', {}),
                'performance_metrics': kwargs.get('performance_metrics', {}),
                'export_time': datetime.now().isoformat(),
                'dapt_version': kwargs.get('dapt_version', '1.0.0')
            }
            
            with open(output_path / "dapt_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"Error exporting Hugging Face model: {str(e)}")
            return False
    
    def validate_export(self, export_path: str) -> bool:
        """Validate Hugging Face export"""
        try:
            export_path = Path(export_path)
            
            # Check required files
            required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
            
            for file_name in required_files:
                if not (export_path / file_name).exists():
                    return False
            
            # Try to load model and tokenizer
            AutoModel.from_pretrained(export_path)
            AutoTokenizer.from_pretrained(export_path)
            
            return True
            
        except Exception:
            return False
    
    def _create_model_card(self, model: AutoModel, tokenizer: AutoTokenizer, **kwargs) -> str:
        """Create model card for Hugging Face Hub"""
        country_code = kwargs.get('country_code', 'Unknown')
        base_model = kwargs.get('base_model', 'Unknown')
        performance_metrics = kwargs.get('performance_metrics', {})
        
        model_card = f"""---
language: ar
license: apache-2.0
tags:
- arabic
- dapt
- domain-adaptation
- {country_code.lower()}
- bert
datasets:
- custom
metrics:
{self._format_metrics_yaml(performance_metrics)}
---

# DAPT Model for {country_code}

This model was trained using Domain-Adaptive Pre-Training (DAPT) for Arabic text processing, specifically adapted for {country_code} dialect and domain.

## Model Details

- **Base Model**: {base_model}
- **Country/Region**: {country_code}
- **Model Type**: {model.config.model_type}
- **Language**: Arabic
- **Training Method**: Domain-Adaptive Pre-Training (DAPT)

## Performance Metrics

{self._format_metrics_table(performance_metrics)}

## Usage

```python
from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
model = AutoModel.from_pretrained("path/to/this/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/this/model")

# Example usage
text = "مرحبا بك في العالم العربي"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
```

## Training Details

This model was trained using the DAPT (Domain-Adaptive Pre-Training) methodology to adapt a pre-trained Arabic language model for specific country/domain requirements.

### Training Data

- Domain-specific Arabic text data
- Country-specific linguistic patterns
- Regional dialect adaptations

### Training Procedure

- Continued pre-training on domain-specific data
- Vocabulary adaptation for regional terms
- Fine-tuning for downstream tasks

## Limitations and Bias

- This model is specifically adapted for {country_code} Arabic dialect
- Performance may vary on other Arabic dialects
- Potential bias towards the training domain

## Citation

If you use this model, please cite:

```
@misc{{dapt_model_{country_code.lower()},
  title={{DAPT Model for {country_code} Arabic}},
  author={{DAPT Training System}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/path/to/this/model}}}}
}}
```
"""
        
        return model_card
    
    def _format_metrics_yaml(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for YAML frontmatter"""
        if not metrics:
            return ""
        
        yaml_lines = []
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                yaml_lines.append(f"- name: {metric}")
                yaml_lines.append(f"  type: {metric}")
                yaml_lines.append(f"  value: {value}")
        
        return "\n".join(yaml_lines)
    
    def _format_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """Format metrics as markdown table"""
        if not metrics:
            return "No performance metrics available."
        
        table_lines = ["| Metric | Value |", "| --- | --- |"]
        
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                table_lines.append(f"| {metric} | {value:.4f} |")
            else:
                table_lines.append(f"| {metric} | {value} |")
        
        return "\n".join(table_lines)

class TensorFlowExporter(BaseExporter):
    """TensorFlow model exporter"""
    
    def export(self, model: AutoModel, tokenizer: AutoTokenizer, 
              output_path: str, **kwargs) -> bool:
        """Export model to TensorFlow format"""
        try:
            # This would require TensorFlow and transformers TF support
            # For now, return a placeholder implementation
            print("TensorFlow export not yet implemented")
            return False
            
        except Exception as e:
            print(f"Error exporting TensorFlow model: {str(e)}")
            return False
    
    def validate_export(self, export_path: str) -> bool:
        """Validate TensorFlow export"""
        return False

class ModelExporter:
    """Main model exporter for DAPT training"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
        self.country_code = config["country"]["code"]
        
        # Export paths
        self.exports_dir = Path(global_config["exports_dir"]) / self.country_code
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize exporters
        self.exporters = {
            'pytorch': PyTorchExporter(),
            'onnx': ONNXExporter(config.get('onnx', {})),
            'huggingface': HuggingFaceExporter(config.get('huggingface', {})),
            'tensorflow': TensorFlowExporter()
        }
        
        # Logging
        self.logger = self._setup_logging()
        
        self.logger.info(f"Model Exporter initialized for country: {self.country_code}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for model exporter"""
        logger = logging.getLogger(f"dapt_model_exporter_{self.country_code}")
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
    
    def export_model(self, model: AutoModel, tokenizer: AutoTokenizer,
                    model_name: str, export_format: str,
                    export_path: Optional[str] = None,
                    **kwargs) -> Optional[str]:
        """Export model in specified format"""
        try:
            if export_format not in self.exporters:
                self.logger.error(f"Unsupported export format: {export_format}")
                return None
            
            # Determine export path
            if export_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_filename = f"{model_name}_{export_format}_{timestamp}"
                export_path = self.exports_dir / export_filename
            else:
                export_path = Path(export_path)
            
            self.logger.info(f"Exporting model {model_name} to {export_format} format")
            
            # Add metadata to kwargs
            export_kwargs = {
                'country_code': self.country_code,
                'model_name': model_name,
                'export_format': export_format,
                **kwargs
            }
            
            # Export model
            exporter = self.exporters[export_format]
            success = exporter.export(model, tokenizer, str(export_path), **export_kwargs)
            
            if success:
                # Validate export
                if exporter.validate_export(str(export_path)):
                    self.logger.info(f"Model successfully exported to: {export_path}")
                    
                    # Save export metadata
                    self._save_export_metadata(export_path, model_name, export_format, export_kwargs)
                    
                    return str(export_path)
                else:
                    self.logger.error(f"Export validation failed for: {export_path}")
                    return None
            else:
                self.logger.error(f"Export failed for model: {model_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error exporting model: {str(e)}")
            return None
    
    def export_multiple_formats(self, model: AutoModel, tokenizer: AutoTokenizer,
                               model_name: str, formats: List[str],
                               **kwargs) -> Dict[str, Optional[str]]:
        """Export model to multiple formats"""
        results = {}
        
        for export_format in formats:
            try:
                export_path = self.export_model(
                    model, tokenizer, model_name, export_format, **kwargs
                )
                results[export_format] = export_path
                
            except Exception as e:
                self.logger.error(f"Error exporting to {export_format}: {str(e)}")
                results[export_format] = None
        
        return results
    
    def create_deployment_package(self, model: AutoModel, tokenizer: AutoTokenizer,
                                 model_name: str, package_formats: List[str] = None,
                                 include_examples: bool = True,
                                 **kwargs) -> Optional[str]:
        """Create deployment package with multiple formats"""
        try:
            if package_formats is None:
                package_formats = ['pytorch', 'huggingface']
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            package_name = f"{model_name}_deployment_{timestamp}"
            package_dir = self.exports_dir / package_name
            package_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Creating deployment package: {package_name}")
            
            # Export to multiple formats
            export_results = {}
            for export_format in package_formats:
                format_dir = package_dir / export_format
                export_path = self.export_model(
                    model, tokenizer, model_name, export_format,
                    export_path=str(format_dir), **kwargs
                )
                export_results[export_format] = export_path
            
            # Create deployment configuration
            deployment_config = {
                'model_name': model_name,
                'country_code': self.country_code,
                'available_formats': list(export_results.keys()),
                'export_paths': export_results,
                'created_at': datetime.now().isoformat(),
                'model_info': {
                    'model_type': model.config.model_type,
                    'vocab_size': len(tokenizer),
                    'max_length': getattr(tokenizer, 'model_max_length', 512)
                },
                'deployment_instructions': self._generate_deployment_instructions(export_results)
            }
            
            with open(package_dir / "deployment_config.json", 'w') as f:
                json.dump(deployment_config, f, indent=2)
            
            # Create usage examples if requested
            if include_examples:
                self._create_usage_examples(package_dir, model_name, export_results)
            
            # Create ZIP package
            zip_path = package_dir.with_suffix('.zip')
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in package_dir.rglob('*'):
                    if file_path.is_file():
                        arcname = file_path.relative_to(package_dir)
                        zipf.write(file_path, arcname)
            
            # Clean up temporary directory
            shutil.rmtree(package_dir)
            
            self.logger.info(f"Deployment package created: {zip_path}")
            return str(zip_path)
            
        except Exception as e:
            self.logger.error(f"Error creating deployment package: {str(e)}")
            return None
    
    def _save_export_metadata(self, export_path: Path, model_name: str,
                             export_format: str, kwargs: Dict[str, Any]) -> None:
        """Save export metadata"""
        try:
            metadata = {
                'model_name': model_name,
                'export_format': export_format,
                'export_path': str(export_path),
                'country_code': self.country_code,
                'export_time': datetime.now().isoformat(),
                'export_config': kwargs
            }
            
            metadata_file = export_path.parent / f"{export_path.name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Could not save export metadata: {str(e)}")
    
    def _generate_deployment_instructions(self, export_results: Dict[str, Optional[str]]) -> Dict[str, str]:
        """Generate deployment instructions for each format"""
        instructions = {}
        
        if 'pytorch' in export_results and export_results['pytorch']:
            instructions['pytorch'] = """
# PyTorch Deployment

import torch
from transformers import AutoTokenizer

# Load the exported model
model_data = torch.load('pytorch/model.pt', map_location='cpu')
model_state_dict = model_data['model_state_dict']
model_config = model_data['model_config']

# Initialize model and load state
from transformers import AutoModel, AutoConfig
config = AutoConfig.from_dict(model_config)
model = AutoModel.from_config(config)
model.load_state_dict(model_state_dict)
model.eval()

# Use the model
text = "مرحبا بك"
# ... tokenization and inference code ...
"""
        
        if 'huggingface' in export_results and export_results['huggingface']:
            instructions['huggingface'] = """
# Hugging Face Deployment

from transformers import AutoModel, AutoTokenizer

# Load model and tokenizer
model = AutoModel.from_pretrained('huggingface/')
tokenizer = AutoTokenizer.from_pretrained('huggingface/')

# Use the model
text = "مرحبا بك"
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
"""
        
        if 'onnx' in export_results and export_results['onnx']:
            instructions['onnx'] = """
# ONNX Deployment

import onnxruntime as ort
import numpy as np

# Load ONNX model
session = ort.InferenceSession('onnx/model.onnx')

# Prepare input
input_ids = np.array([[101, 102, 103]], dtype=np.int64)  # Example token IDs

# Run inference
outputs = session.run(None, {'input_ids': input_ids})
"""
        
        return instructions
    
    def _create_usage_examples(self, package_dir: Path, model_name: str,
                              export_results: Dict[str, Optional[str]]) -> None:
        """Create usage examples for the deployment package"""
        try:
            examples_dir = package_dir / "examples"
            examples_dir.mkdir(exist_ok=True)
            
            # Create Python example
            python_example = f"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
\"\"\"
Example usage of the exported {model_name} model

This script demonstrates how to use the exported model for Arabic text processing.
\"\"\"

import sys
from pathlib import Path

# Add the package directory to Python path
package_dir = Path(__file__).parent.parent
sys.path.insert(0, str(package_dir))

def load_huggingface_model():
    \"\"\"Load and use the Hugging Face format model\"\"\"
    try:
        from transformers import AutoModel, AutoTokenizer
        
        model_path = package_dir / "huggingface"
        
        print(f"Loading model from: {{model_path}}")
        model = AutoModel.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Example text
        text = "مرحبا بك في العالم العربي"
        print(f"Input text: {{text}}")
        
        # Tokenize and process
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        outputs = model(**inputs)
        
        print(f"Output shape: {{outputs.last_hidden_state.shape}}")
        print("Model loaded and inference completed successfully!")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading Hugging Face model: {{e}}")
        return None, None

def load_pytorch_model():
    \"\"\"Load and use the PyTorch format model\"\"\"
    try:
        import torch
        from transformers import AutoModel, AutoConfig, AutoTokenizer
        
        model_path = package_dir / "pytorch" / "model.pt"
        
        print(f"Loading PyTorch model from: {{model_path}}")
        model_data = torch.load(model_path, map_location='cpu')
        
        # Reconstruct model
        config = AutoConfig.from_dict(model_data['model_config'])
        model = AutoModel.from_config(config)
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        
        print("PyTorch model loaded successfully!")
        return model
        
    except Exception as e:
        print(f"Error loading PyTorch model: {{e}}")
        return None

if __name__ == "__main__":
    print(f"DAPT Model Example - {{model_name}}")
    print("=" * 50)
    
    # Try Hugging Face format first
    print("\\n1. Testing Hugging Face format:")
    hf_model, hf_tokenizer = load_huggingface_model()
    
    # Try PyTorch format
    print("\\n2. Testing PyTorch format:")
    pt_model = load_pytorch_model()
    
    print("\\nExample completed!")
"""
            
            with open(examples_dir / "example_usage.py", 'w', encoding='utf-8') as f:
                f.write(python_example)
            
            # Create README for examples
            readme_content = f"""
# Usage Examples for {model_name}

This directory contains example scripts demonstrating how to use the exported model.

## Files

- `example_usage.py`: Basic usage example showing how to load and use the model
- `README.md`: This file

## Running the Examples

```bash
# Make sure you have the required dependencies installed
pip install torch transformers

# Run the example
python examples/example_usage.py
```

## Available Model Formats

{self._format_available_formats(export_results)}

## Requirements

- Python 3.7+
- PyTorch
- Transformers
- Additional dependencies based on the format used
"""
            
            with open(examples_dir / "README.md", 'w', encoding='utf-8') as f:
                f.write(readme_content)
                
        except Exception as e:
            self.logger.warning(f"Could not create usage examples: {str(e)}")
    
    def _format_available_formats(self, export_results: Dict[str, Optional[str]]) -> str:
        """Format available formats for documentation"""
        format_descriptions = {
            'pytorch': 'PyTorch native format (.pt file)',
            'huggingface': 'Hugging Face Transformers format',
            'onnx': 'ONNX format for cross-platform deployment',
            'tensorflow': 'TensorFlow SavedModel format'
        }
        
        lines = []
        for format_name, path in export_results.items():
            if path:
                description = format_descriptions.get(format_name, f'{format_name} format')
                lines.append(f"- **{format_name}**: {description}")
        
        return "\n".join(lines)
    
    def list_exports(self) -> List[Dict[str, Any]]:
        """List all exported models"""
        try:
            exports = []
            
            for export_path in self.exports_dir.rglob('*_metadata.json'):
                try:
                    with open(export_path, 'r') as f:
                        metadata = json.load(f)
                    
                    # Add file info
                    export_file = export_path.parent / export_path.name.replace('_metadata.json', '')
                    if export_file.exists():
                        metadata['file_size'] = export_file.stat().st_size
                        metadata['exists'] = True
                    else:
                        metadata['file_size'] = 0
                        metadata['exists'] = False
                    
                    exports.append(metadata)
                    
                except Exception as e:
                    self.logger.warning(f"Could not read metadata from {export_path}: {str(e)}")
            
            # Sort by export time (newest first)
            exports.sort(key=lambda x: x.get('export_time', ''), reverse=True)
            
            return exports
            
        except Exception as e:
            self.logger.error(f"Error listing exports: {str(e)}")
            return []
    
    def get_export_info(self) -> Dict[str, Any]:
        """Get export directory information"""
        try:
            exports = self.list_exports()
            total_size = sum(export.get('file_size', 0) for export in exports)
            
            format_counts = {}
            for export in exports:
                export_format = export.get('export_format', 'unknown')
                format_counts[export_format] = format_counts.get(export_format, 0) + 1
            
            return {
                'exports_directory': str(self.exports_dir),
                'total_exports': len(exports),
                'total_size_bytes': total_size,
                'total_size_mb': total_size / (1024 * 1024),
                'format_counts': format_counts,
                'supported_formats': list(self.exporters.keys())
            }
            
        except Exception as e:
            self.logger.error(f"Error getting export info: {str(e)}")
            return {}