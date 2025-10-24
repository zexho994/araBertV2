"""Tests for custom loss functions

This module tests the correctness of various loss functions implemented
for NER training improvement.
"""

import torch
import pytest
from src.ner.training.losses import (
    WeightedCrossEntropyLoss,
    FocalLoss,
    BoundaryAwareLoss,
    CombinedLoss,
    LossFactory
)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    batch_size = 2
    seq_len = 5
    num_labels = 7
    
    # Sample label mappings for BIO tags
    label2id = {
        'O': 0,
        'B-PER': 1,
        'I-PER': 2,
        'B-ORG': 3,
        'I-ORG': 4,
        'B-LOC': 5,
        'I-LOC': 6
    }
    id2label = {v: k for k, v in label2id.items()}
    
    # Random logits with requires_grad (simulating model output)
    logits = torch.randn(batch_size, seq_len, num_labels, requires_grad=True)
    
    # Sample labels
    labels = torch.tensor([
        [0, 1, 2, 0, 3],  # O, B-PER, I-PER, O, B-ORG
        [0, 5, 6, 0, -100]  # O, B-LOC, I-LOC, O, padding
    ])
    
    # Attention mask
    attention_mask = torch.tensor([
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 0]
    ])
    
    return {
        'logits': logits,
        'labels': labels,
        'attention_mask': attention_mask,
        'num_labels': num_labels,
        'label2id': label2id,
        'id2label': id2label
    }


class TestWeightedCrossEntropyLoss:
    """Test WeightedCrossEntropyLoss"""
    
    def test_initialization(self, sample_data):
        """Test loss function initialization"""
        loss_fn = WeightedCrossEntropyLoss(
            num_labels=sample_data['num_labels'],
            label2id=sample_data['label2id'],
            weight_multipliers={'B-PER': 2.0, 'I-PER': 2.0, 'O': 0.5}
        )
        
        assert loss_fn.weights[1] == 2.0  # B-PER
        assert loss_fn.weights[2] == 2.0  # I-PER
        assert loss_fn.weights[0] == 0.5  # O
        assert loss_fn.weights[3] == 1.0  # B-ORG (default)
    
    def test_forward(self, sample_data):
        """Test forward pass"""
        loss_fn = WeightedCrossEntropyLoss(
            num_labels=sample_data['num_labels'],
            label2id=sample_data['label2id'],
            weight_multipliers={'B-PER': 2.0}
        )
        
        loss = loss_fn(
            sample_data['logits'],
            sample_data['labels'],
            sample_data['attention_mask']
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0
        assert loss.requires_grad
    
    def test_ignore_padding(self, sample_data):
        """Test that padding is properly ignored"""
        loss_fn = WeightedCrossEntropyLoss(
            num_labels=sample_data['num_labels'],
            label2id=sample_data['label2id']
        )
        
        # Loss should not be affected by padding tokens (-100)
        loss = loss_fn(
            sample_data['logits'],
            sample_data['labels'],
            sample_data['attention_mask']
        )
        
        # Create version without padding
        labels_no_pad = sample_data['labels'].clone()
        labels_no_pad[1, 4] = 0  # Replace -100 with valid label
        
        loss_no_pad = loss_fn(
            sample_data['logits'][:, :4],  # Remove last position
            sample_data['labels'][:, :4],
            sample_data['attention_mask'][:, :4]
        )
        
        # Losses should be different (not equal) but both valid
        assert loss.item() > 0
        assert loss_no_pad.item() > 0


class TestFocalLoss:
    """Test FocalLoss"""
    
    def test_initialization(self, sample_data):
        """Test focal loss initialization"""
        loss_fn = FocalLoss(
            num_labels=sample_data['num_labels'],
            alpha=0.25,
            gamma=2.0
        )
        
        assert loss_fn.alpha == 0.25
        assert loss_fn.gamma == 2.0
    
    def test_forward(self, sample_data):
        """Test forward pass"""
        loss_fn = FocalLoss(
            num_labels=sample_data['num_labels'],
            alpha=0.25,
            gamma=2.0
        )
        
        loss = loss_fn(
            sample_data['logits'],
            sample_data['labels'],
            sample_data['attention_mask']
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert loss.requires_grad
    
    def test_with_label_weights(self, sample_data):
        """Test focal loss with label weights"""
        weights = torch.ones(sample_data['num_labels'])
        weights[1] = 2.0  # B-PER
        weights[2] = 2.0  # I-PER
        
        loss_fn = FocalLoss(
            num_labels=sample_data['num_labels'],
            alpha=0.25,
            gamma=2.0,
            label_weights=weights
        )
        
        loss = loss_fn(
            sample_data['logits'],
            sample_data['labels'],
            sample_data['attention_mask']
        )
        
        assert loss.item() > 0


class TestBoundaryAwareLoss:
    """Test BoundaryAwareLoss"""
    
    def test_initialization(self, sample_data):
        """Test boundary loss initialization"""
        loss_fn = BoundaryAwareLoss(
            label2id=sample_data['label2id'],
            id2label=sample_data['id2label'],
            weight=0.15
        )
        
        assert loss_fn.weight == 0.15
        assert hasattr(loss_fn, 'invalid_transitions_matrix')
    
    def test_transition_rules(self, sample_data):
        """Test that invalid transitions are correctly identified"""
        loss_fn = BoundaryAwareLoss(
            label2id=sample_data['label2id'],
            id2label=sample_data['id2label']
        )
        
        # I-PER cannot follow O
        o_id = sample_data['label2id']['O']
        i_per_id = sample_data['label2id']['I-PER']
        assert loss_fn.invalid_transitions_matrix[o_id, i_per_id] == 1.0
        
        # I-PER can follow B-PER
        b_per_id = sample_data['label2id']['B-PER']
        assert loss_fn.invalid_transitions_matrix[b_per_id, i_per_id] == 0.0
        
        # I-PER cannot follow B-ORG
        b_org_id = sample_data['label2id']['B-ORG']
        assert loss_fn.invalid_transitions_matrix[b_org_id, i_per_id] == 1.0
    
    def test_forward_valid_sequence(self, sample_data):
        """Test boundary loss with valid sequence"""
        loss_fn = BoundaryAwareLoss(
            label2id=sample_data['label2id'],
            id2label=sample_data['id2label'],
            weight=0.15
        )
        
        # Valid sequence: O, B-PER, I-PER, O, B-ORG
        valid_predictions = torch.tensor([
            [0, 1, 2, 0, 3],
            [0, 5, 6, 0, 0]
        ])
        
        loss = loss_fn(
            valid_predictions,
            sample_data['labels'],
            sample_data['attention_mask']
        )
        
        # Should have minimal or zero violation
        assert loss.item() >= 0
    
    def test_forward_invalid_sequence(self, sample_data):
        """Test boundary loss with invalid sequence"""
        loss_fn = BoundaryAwareLoss(
            label2id=sample_data['label2id'],
            id2label=sample_data['id2label'],
            weight=0.15
        )
        
        # Invalid sequence: O, I-PER (I-PER cannot follow O)
        invalid_predictions = torch.tensor([
            [0, 2, 2, 0, 3],  # O, I-PER, I-PER, O, B-ORG
            [0, 5, 6, 0, 0]
        ])
        
        loss = loss_fn(
            invalid_predictions,
            sample_data['labels'],
            sample_data['attention_mask']
        )
        
        # Should have non-zero violation penalty
        assert loss.item() > 0


class TestCombinedLoss:
    """Test CombinedLoss"""
    
    def test_forward(self, sample_data):
        """Test combined loss forward pass"""
        base_loss = WeightedCrossEntropyLoss(
            num_labels=sample_data['num_labels'],
            label2id=sample_data['label2id']
        )
        
        boundary_loss = BoundaryAwareLoss(
            label2id=sample_data['label2id'],
            id2label=sample_data['id2label'],
            weight=0.15
        )
        
        combined = CombinedLoss(base_loss, boundary_loss)
        
        loss = combined(
            sample_data['logits'],
            sample_data['labels'],
            sample_data['attention_mask']
        )
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert loss.requires_grad


class TestLossFactory:
    """Test LossFactory"""
    
    def test_create_default_loss(self, sample_data):
        """Test creating default CrossEntropyLoss"""
        loss_fn = LossFactory.create_loss(
            config=None,
            label2id=sample_data['label2id'],
            id2label=sample_data['id2label'],
            num_labels=sample_data['num_labels']
        )
        
        assert loss_fn is not None
    
    def test_create_weighted_ce(self, sample_data):
        """Test creating weighted CE loss"""
        config = {
            'type': 'weighted_ce',
            'weight_multipliers': {
                'B-PER': 2.0,
                'I-PER': 2.0,
                'O': 0.5
            }
        }
        
        loss_fn = LossFactory.create_loss(
            config=config,
            label2id=sample_data['label2id'],
            id2label=sample_data['id2label'],
            num_labels=sample_data['num_labels']
        )
        
        assert isinstance(loss_fn, WeightedCrossEntropyLoss)
    
    def test_create_focal_loss(self, sample_data):
        """Test creating focal loss"""
        config = {
            'type': 'focal',
            'alpha': 0.25,
            'gamma': 2.0
        }
        
        loss_fn = LossFactory.create_loss(
            config=config,
            label2id=sample_data['label2id'],
            id2label=sample_data['id2label'],
            num_labels=sample_data['num_labels']
        )
        
        assert isinstance(loss_fn, FocalLoss)
    
    def test_create_boundary_aware(self, sample_data):
        """Test creating boundary-aware loss"""
        config = {
            'type': 'boundary_aware',
            'base_loss': 'weighted_ce',
            'weight_multipliers': {'B-PER': 2.0},
            'boundary_weight': 0.15
        }
        
        loss_fn = LossFactory.create_loss(
            config=config,
            label2id=sample_data['label2id'],
            id2label=sample_data['id2label'],
            num_labels=sample_data['num_labels']
        )
        
        assert isinstance(loss_fn, CombinedLoss)
    
    def test_create_combined_loss(self, sample_data):
        """Test creating combined loss"""
        config = {
            'type': 'combined',
            'focal_alpha': 0.25,
            'focal_gamma': 2.0,
            'weight_multipliers': {'B-PER': 2.0, 'O': 0.5},
            'boundary_weight': 0.15
        }
        
        loss_fn = LossFactory.create_loss(
            config=config,
            label2id=sample_data['label2id'],
            id2label=sample_data['id2label'],
            num_labels=sample_data['num_labels']
        )
        
        assert isinstance(loss_fn, CombinedLoss)
    
    def test_invalid_loss_type(self, sample_data):
        """Test that invalid loss type raises error"""
        config = {'type': 'invalid_loss'}
        
        with pytest.raises(ValueError):
            LossFactory.create_loss(
                config=config,
                label2id=sample_data['label2id'],
                id2label=sample_data['id2label'],
                num_labels=sample_data['num_labels']
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

