"""Model Utilities for NER System

Provides model-related utilities including parameter counting,
model analysis, and optimization helpers.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
from collections import OrderedDict
import json
import os
from pathlib import Path

class ModelUtils:
    """Model utilities and analysis tools"""
    
    @staticmethod
    def count_parameters(model: nn.Module, trainable_only: bool = False) -> Dict[str, int]:
        """Count model parameters
        
        Args:
            model: PyTorch model
            trainable_only: Count only trainable parameters
            
        Returns:
            Parameter counts
        """
        if trainable_only:
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            trainable_params = total_params
            frozen_params = 0
        else:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            frozen_params = total_params - trainable_params
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            'frozen': frozen_params
        }
    
    @staticmethod
    def get_model_size(model: nn.Module) -> Dict[str, float]:
        """Get model size in MB
        
        Args:
            model: PyTorch model
            
        Returns:
            Model size information
        """
        param_size = 0
        buffer_size = 0
        
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        total_size = param_size + buffer_size
        
        return {
            'parameters_mb': param_size / 1024 / 1024,
            'buffers_mb': buffer_size / 1024 / 1024,
            'total_mb': total_size / 1024 / 1024
        }
    
    @staticmethod
    def analyze_model_layers(model: nn.Module) -> Dict[str, Any]:
        """Analyze model layers
        
        Args:
            model: PyTorch model
            
        Returns:
            Layer analysis
        """
        layer_info = []
        total_params = 0
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # Leaf module
                params = sum(p.numel() for p in module.parameters())
                trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
                
                layer_info.append({
                    'name': name,
                    'type': type(module).__name__,
                    'parameters': params,
                    'trainable_parameters': trainable_params,
                    'frozen': params - trainable_params > 0
                })
                
                total_params += params
        
        return {
            'layers': layer_info,
            'total_layers': len(layer_info),
            'total_parameters': total_params
        }
    
    @staticmethod
    def get_layer_gradients(model: nn.Module) -> Dict[str, float]:
        """Get gradient norms for each layer
        
        Args:
            model: PyTorch model
            
        Returns:
            Gradient norms by layer
        """
        grad_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                grad_norms[name] = grad_norm
            else:
                grad_norms[name] = 0.0
        
        return grad_norms
    
    @staticmethod
    def check_gradient_flow(model: nn.Module, threshold: float = 1e-6) -> Dict[str, Any]:
        """Check gradient flow through model
        
        Args:
            model: PyTorch model
            threshold: Minimum gradient threshold
            
        Returns:
            Gradient flow analysis
        """
        grad_norms = ModelUtils.get_layer_gradients(model)
        
        # Find layers with vanishing gradients
        vanishing_layers = [name for name, norm in grad_norms.items() if norm < threshold]
        
        # Find layers with exploding gradients
        exploding_threshold = 10.0
        exploding_layers = [name for name, norm in grad_norms.items() if norm > exploding_threshold]
        
        # Calculate statistics
        grad_values = list(grad_norms.values())
        avg_grad = np.mean(grad_values) if grad_values else 0.0
        max_grad = np.max(grad_values) if grad_values else 0.0
        min_grad = np.min(grad_values) if grad_values else 0.0
        
        return {
            'gradient_norms': grad_norms,
            'vanishing_layers': vanishing_layers,
            'exploding_layers': exploding_layers,
            'avg_gradient': avg_grad,
            'max_gradient': max_grad,
            'min_gradient': min_grad,
            'healthy_flow': len(vanishing_layers) == 0 and len(exploding_layers) == 0
        }
    
    @staticmethod
    def freeze_layers(model: nn.Module, layer_names: List[str]) -> int:
        """Freeze specific layers
        
        Args:
            model: PyTorch model
            layer_names: Names of layers to freeze
            
        Returns:
            Number of parameters frozen
        """
        frozen_params = 0
        
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                frozen_params += param.numel()
        
        return frozen_params
    
    @staticmethod
    def unfreeze_layers(model: nn.Module, layer_names: List[str]) -> int:
        """Unfreeze specific layers
        
        Args:
            model: PyTorch model
            layer_names: Names of layers to unfreeze
            
        Returns:
            Number of parameters unfrozen
        """
        unfrozen_params = 0
        
        for name, param in model.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                unfrozen_params += param.numel()
        
        return unfrozen_params
    
    @staticmethod
    def get_learning_rates(optimizer) -> Dict[str, float]:
        """Get learning rates from optimizer
        
        Args:
            optimizer: PyTorch optimizer
            
        Returns:
            Learning rates by parameter group
        """
        learning_rates = {}
        
        for i, param_group in enumerate(optimizer.param_groups):
            learning_rates[f'group_{i}'] = param_group['lr']
        
        return learning_rates
    
    @staticmethod
    def set_learning_rates(optimizer, learning_rates: Union[float, Dict[str, float]]):
        """Set learning rates in optimizer
        
        Args:
            optimizer: PyTorch optimizer
            learning_rates: Learning rate(s) to set
        """
        if isinstance(learning_rates, float):
            # Set same learning rate for all groups
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rates
        else:
            # Set different learning rates for different groups
            for i, param_group in enumerate(optimizer.param_groups):
                group_name = f'group_{i}'
                if group_name in learning_rates:
                    param_group['lr'] = learning_rates[group_name]
    
    @staticmethod
    def save_model_info(
        model: nn.Module, 
        save_path: Union[str, Path],
        additional_info: Optional[Dict[str, Any]] = None
    ):
        """Save model information to file
        
        Args:
            model: PyTorch model
            save_path: Path to save information
            additional_info: Additional information to save
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Collect model information
        info = {
            'model_class': type(model).__name__,
            'parameters': ModelUtils.count_parameters(model),
            'size': ModelUtils.get_model_size(model),
            'layers': ModelUtils.analyze_model_layers(model)
        }
        
        # Add additional information
        if additional_info:
            info.update(additional_info)
        
        # Save to JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, default=str)
    
    @staticmethod
    def load_model_info(load_path: Union[str, Path]) -> Dict[str, Any]:
        """Load model information from file
        
        Args:
            load_path: Path to load information from
            
        Returns:
            Model information
        """
        with open(load_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def compare_models(
        model1: nn.Module, 
        model2: nn.Module,
        model1_name: str = "Model 1",
        model2_name: str = "Model 2"
    ) -> Dict[str, Any]:
        """Compare two models
        
        Args:
            model1: First model
            model2: Second model
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Comparison results
        """
        # Get model information
        info1 = {
            'parameters': ModelUtils.count_parameters(model1),
            'size': ModelUtils.get_model_size(model1),
            'layers': ModelUtils.analyze_model_layers(model1)
        }
        
        info2 = {
            'parameters': ModelUtils.count_parameters(model2),
            'size': ModelUtils.get_model_size(model2),
            'layers': ModelUtils.analyze_model_layers(model2)
        }
        
        # Calculate differences
        param_diff = info2['parameters']['total'] - info1['parameters']['total']
        size_diff = info2['size']['total_mb'] - info1['size']['total_mb']
        layer_diff = info2['layers']['total_layers'] - info1['layers']['total_layers']
        
        return {
            model1_name: info1,
            model2_name: info2,
            'differences': {
                'parameters': param_diff,
                'size_mb': size_diff,
                'layers': layer_diff
            },
            'ratios': {
                'parameters': info2['parameters']['total'] / info1['parameters']['total'] if info1['parameters']['total'] > 0 else 0,
                'size': info2['size']['total_mb'] / info1['size']['total_mb'] if info1['size']['total_mb'] > 0 else 0
            }
        }
    
    @staticmethod
    def estimate_memory_usage(
        model: nn.Module, 
        batch_size: int,
        sequence_length: int,
        precision: str = 'fp32'
    ) -> Dict[str, float]:
        """Estimate memory usage for model
        
        Args:
            model: PyTorch model
            batch_size: Batch size
            sequence_length: Sequence length
            precision: Model precision (fp32, fp16)
            
        Returns:
            Memory usage estimates in MB
        """
        # Bytes per parameter based on precision
        bytes_per_param = 4 if precision == 'fp32' else 2
        
        # Model parameters
        param_count = ModelUtils.count_parameters(model)['total']
        model_memory = param_count * bytes_per_param / 1024 / 1024
        
        # Estimate activation memory (rough approximation)
        # This is a simplified estimation
        activation_memory = batch_size * sequence_length * 768 * bytes_per_param / 1024 / 1024  # Assuming 768 hidden size
        
        # Gradient memory (same as parameters for training)
        gradient_memory = model_memory
        
        # Optimizer memory (Adam uses 2x parameter memory)
        optimizer_memory = model_memory * 2
        
        total_memory = model_memory + activation_memory + gradient_memory + optimizer_memory
        
        return {
            'model_mb': model_memory,
            'activations_mb': activation_memory,
            'gradients_mb': gradient_memory,
            'optimizer_mb': optimizer_memory,
            'total_mb': total_memory,
            'total_gb': total_memory / 1024
        }
    
    @staticmethod
    def check_model_compatibility(
        model: nn.Module, 
        input_shape: Tuple[int, ...],
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Check model compatibility with input shape
        
        Args:
            model: PyTorch model
            input_shape: Expected input shape
            device: Device to test on
            
        Returns:
            Compatibility check results
        """
        if device is None:
            device = torch.device('cpu')
        
        try:
            # Create dummy input
            dummy_input = torch.randn(input_shape).to(device)
            model = model.to(device)
            
            # Test forward pass
            model.eval()
            with torch.no_grad():
                output = model(dummy_input)
            
            return {
                'compatible': True,
                'input_shape': input_shape,
                'output_shape': output.shape if hasattr(output, 'shape') else 'Unknown',
                'device': str(device),
                'error': None
            }
        
        except Exception as e:
            return {
                'compatible': False,
                'input_shape': input_shape,
                'output_shape': None,
                'device': str(device),
                'error': str(e)
            }
    
    @staticmethod
    def profile_model(
        model: nn.Module,
        input_shape: Tuple[int, ...],
        num_runs: int = 100,
        device: Optional[torch.device] = None
    ) -> Dict[str, float]:
        """Profile model performance
        
        Args:
            model: PyTorch model
            input_shape: Input shape for profiling
            num_runs: Number of runs for averaging
            device: Device to profile on
            
        Returns:
            Profiling results
        """
        if device is None:
            device = torch.device('cpu')
        
        model = model.to(device)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape).to(device)
        
        # Warm up
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Profile forward pass
        import time
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start_time = time.time()
        
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(dummy_input)
        
        torch.cuda.synchronize() if device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs
        throughput = 1.0 / avg_time if avg_time > 0 else 0
        
        return {
            'avg_inference_time_ms': avg_time * 1000,
            'throughput_samples_per_sec': throughput,
            'device': str(device),
            'num_runs': num_runs
        }
    
    @staticmethod
    def get_model_summary(model: nn.Module) -> str:
        """Get a summary string of the model
        
        Args:
            model: PyTorch model
            
        Returns:
            Model summary string
        """
        param_info = ModelUtils.count_parameters(model)
        size_info = ModelUtils.get_model_size(model)
        layer_info = ModelUtils.analyze_model_layers(model)
        
        summary = f"""Model Summary:
{'='*50}
Model Class: {type(model).__name__}
Total Parameters: {param_info['total']:,}
Trainable Parameters: {param_info['trainable']:,}
Frozen Parameters: {param_info['frozen']:,}
Model Size: {size_info['total_mb']:.2f} MB
Total Layers: {layer_info['total_layers']}
{'='*50}"""
        
        return summary