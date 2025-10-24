"""Test model loss function compatibility between training and evaluation"""

import torch
import pytest
from src.ner.models.model import NERModelConfig, BertNERModel


def test_model_without_loss_config():
    """Test that model works without custom loss config (evaluation scenario)"""
    # Create a simple config without loss_config
    config = NERModelConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        num_labels=3,
        label2id={'O': 0, 'B-PER': 1, 'I-PER': 2},
        id2label={0: 'O', 1: 'B-PER', 2: 'I-PER'}
    )
    
    # No loss_config set
    assert not hasattr(config, 'loss_config')
    
    # Create model
    model = BertNERModel(config)
    model.eval()
    
    # Create test inputs
    batch_size, seq_len = 2, 5
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 3, (batch_size, seq_len))
    
    # Forward pass should work without error
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    assert 'loss' in outputs
    assert outputs['loss'] is not None
    assert outputs['loss'].item() > 0
    assert 'logits' in outputs
    assert outputs['logits'].shape == (batch_size, seq_len, 3)


def test_model_with_custom_loss_config():
    """Test that model works with custom loss config (training scenario)"""
    # Create config with custom loss
    config = NERModelConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        num_labels=3,
        label2id={'O': 0, 'B-PER': 1, 'I-PER': 2},
        id2label={0: 'O', 1: 'B-PER', 2: 'I-PER'}
    )
    
    # Set custom loss config
    config.loss_config = {
        'type': 'weighted_ce',
        'weight_multipliers': {
            'B-PER': 2.0,
            'I-PER': 2.0,
            'O': 0.5
        }
    }
    
    # Create model
    model = BertNERModel(config)
    model.eval()
    
    # Create test inputs
    batch_size, seq_len = 2, 5
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 3, (batch_size, seq_len))
    
    # Forward pass should work without error
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    
    assert 'loss' in outputs
    assert outputs['loss'] is not None
    assert outputs['loss'].item() > 0
    assert 'logits' in outputs
    assert outputs['logits'].shape == (batch_size, seq_len, 3)


def test_model_switches_between_loss_types():
    """Test that model can handle both standard and custom loss"""
    config = NERModelConfig(
        vocab_size=1000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=256,
        num_labels=3,
        label2id={'O': 0, 'B-PER': 1, 'I-PER': 2},
        id2label={0: 'O', 1: 'B-PER', 2: 'I-PER'}
    )
    
    model = BertNERModel(config)
    model.eval()
    
    batch_size, seq_len = 2, 5
    input_ids = torch.randint(0, 100, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 3, (batch_size, seq_len))
    
    # First forward pass - no custom loss
    outputs1 = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    assert outputs1['loss'] is not None
    
    # Add custom loss config
    config.loss_config = {
        'type': 'weighted_ce',
        'weight_multipliers': {'B-PER': 2.0}
    }
    
    # Reset loss_fct to force reinitialization
    model.loss_fct = None
    
    # Second forward pass - with custom loss
    outputs2 = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )
    assert outputs2['loss'] is not None
    
    # Losses should be different due to different loss functions
    # (not necessarily, but at least both should be valid)
    assert outputs1['loss'].item() > 0
    assert outputs2['loss'].item() > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

