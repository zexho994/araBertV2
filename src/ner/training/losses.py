"""Custom Loss Functions for NER Training

This module implements various loss functions to improve NER model performance,
particularly for low-frequency entity types like BUILDING and STREET.

Loss Functions:
- WeightedCrossEntropyLoss: Class-weighted CE for handling class imbalance
- FocalLoss: Focus on hard examples while down-weighting easy ones
- BoundaryAwareLoss: Penalize invalid BIO tag transitions
- CombinedLoss: Combine multiple loss functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class WeightedCrossEntropyLoss(nn.Module):
    """Class-weighted Cross Entropy Loss
    
    Args:
        num_labels: Total number of labels
        label2id: Mapping from label names to IDs
        weight_multipliers: Dict mapping label names to weight multipliers
        ignore_index: Index to ignore in loss calculation (default: -100)
    
    Example:
        >>> loss_fn = WeightedCrossEntropyLoss(
        ...     num_labels=15,
        ...     label2id={"O": 0, "B-BUILDING": 13, ...},
        ...     weight_multipliers={"B-BUILDING": 2.5, "I-BUILDING": 2.5, "O": 0.6}
        ... )
    """
    
    def __init__(
        self,
        num_labels: int,
        label2id: Dict[str, int],
        weight_multipliers: Optional[Dict[str, float]] = None,
        ignore_index: int = -100
    ):
        super().__init__()
        self.num_labels = num_labels
        self.label2id = label2id
        self.ignore_index = ignore_index
        
        # Initialize weights tensor (default all 1.0)
        weights = torch.ones(num_labels, dtype=torch.float32)
        
        # Apply weight multipliers if provided
        if weight_multipliers:
            for label_name, multiplier in weight_multipliers.items():
                if label_name in label2id:
                    label_id = label2id[label_name]
                    weights[label_id] = multiplier
        
        # Register as buffer (will be moved to device with model)
        self.register_buffer('weights', weights)
        
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, seq_len, num_labels]
            labels: True labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Weighted cross entropy loss (scalar)
        """
        # Flatten for loss calculation
        # logits: [batch_size * seq_len, num_labels]
        # labels: [batch_size * seq_len]
        active_loss = None
        if attention_mask is not None:
            active_loss = attention_mask.reshape(-1) == 1
            active_logits = logits.reshape(-1, self.num_labels)[active_loss]
            active_labels = labels.reshape(-1)[active_loss]
        else:
            active_logits = logits.reshape(-1, self.num_labels)
            active_labels = labels.reshape(-1)
        
        # Filter out ignore_index
        valid_mask = active_labels != self.ignore_index
        if valid_mask.sum() == 0:
            # Return a zero loss that preserves gradients
            return (logits * 0).sum()
        
        valid_logits = active_logits[valid_mask]
        valid_labels = active_labels[valid_mask]
        
        # Calculate weighted cross entropy
        loss = F.cross_entropy(
            valid_logits,
            valid_labels,
            weight=self.weights,
            reduction='mean'
        )
        
        return loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance
    
    Focal Loss: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    
    Reference:
        Lin et al. "Focal Loss for Dense Object Detection" (2017)
    
    Args:
        num_labels: Total number of labels
        alpha: Balancing factor (default: 0.25)
        gamma: Focusing parameter (default: 2.0)
        ignore_index: Index to ignore in loss calculation (default: -100)
        label_weights: Optional per-class weights
    """
    
    def __init__(
        self,
        num_labels: int,
        alpha: float = 0.25,
        gamma: float = 2.0,
        ignore_index: int = -100,
        label_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_labels = num_labels
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        
        if label_weights is not None:
            self.register_buffer('label_weights', label_weights)
        else:
            self.label_weights = None
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, seq_len, num_labels]
            labels: True labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Focal loss (scalar)
        """
        # Flatten
        if attention_mask is not None:
            active_loss = attention_mask.reshape(-1) == 1
            active_logits = logits.reshape(-1, self.num_labels)[active_loss]
            active_labels = labels.reshape(-1)[active_loss]
        else:
            active_logits = logits.reshape(-1, self.num_labels)
            active_labels = labels.reshape(-1)
        
        # Filter out ignore_index
        valid_mask = active_labels != self.ignore_index
        if valid_mask.sum() == 0:
            # Return a zero loss that preserves gradients
            return (logits * 0).sum()
        
        valid_logits = active_logits[valid_mask]
        valid_labels = active_labels[valid_mask]
        
        # Calculate probabilities
        p = F.softmax(valid_logits, dim=-1)
        
        # Get probabilities of true class
        ce_loss = F.cross_entropy(valid_logits, valid_labels, reduction='none')
        p_t = p.gather(1, valid_labels.unsqueeze(1)).squeeze(1)
        
        # Calculate focal loss
        focal_weight = (1 - p_t) ** self.gamma
        focal_loss = self.alpha * focal_weight * ce_loss
        
        # Apply label weights if provided
        if self.label_weights is not None:
            weight = self.label_weights[valid_labels]
            focal_loss = focal_loss * weight
        
        return focal_loss.mean()


class BoundaryAwareLoss(nn.Module):
    """Boundary-Aware Loss to penalize invalid BIO transitions
    
    Penalizes:
    1. I-X at sequence start
    2. I-X following B-Y or I-Y where X != Y
    3. I-X following O
    
    Args:
        label2id: Mapping from label names to IDs
        id2label: Mapping from IDs to label names
        weight: Weight for boundary loss term (default: 0.15)
    """
    
    def __init__(
        self,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        weight: float = 0.15
    ):
        super().__init__()
        self.label2id = label2id
        self.id2label = id2label
        self.weight = weight
        
        # Build transition rules
        self._build_transition_matrix()
    
    def _build_transition_matrix(self):
        """Build a matrix indicating which transitions are invalid"""
        num_labels = len(self.label2id)
        
        # Initialize: all transitions are valid (0 = valid, 1 = invalid)
        invalid_transitions = torch.zeros(num_labels, num_labels, dtype=torch.float32)
        
        for from_id in range(num_labels):
            from_label = self.id2label.get(from_id, 'O')
            
            for to_id in range(num_labels):
                to_label = self.id2label.get(to_id, 'O')
                
                # Rule: I-X cannot follow O
                if from_label == 'O' and to_label.startswith('I-'):
                    invalid_transitions[from_id, to_id] = 1.0
                
                # Rule: I-X cannot follow B-Y or I-Y where X != Y
                if from_label.startswith(('B-', 'I-')) and to_label.startswith('I-'):
                    from_entity = from_label[2:]  # Remove B- or I-
                    to_entity = to_label[2:]
                    if from_entity != to_entity:
                        invalid_transitions[from_id, to_id] = 1.0
        
        self.register_buffer('invalid_transitions_matrix', invalid_transitions)
    
    def forward(
        self,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            predictions: Predicted label IDs [batch_size, seq_len]
            labels: True labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Boundary violation penalty (scalar)
        """
        seq_len = predictions.shape[1]
        device = predictions.device
        
        if seq_len < 2:
            return torch.tensor(0.0, device=device)
        
        # Move transition matrix to correct device
        if self.invalid_transitions_matrix.device != device:
            self.invalid_transitions_matrix = self.invalid_transitions_matrix.to(device)
        
        # Get pairs of consecutive predictions
        from_preds = predictions[:, :-1]  # [batch_size, seq_len-1]
        to_preds = predictions[:, 1:]     # [batch_size, seq_len-1]
        
        # Check for invalid transitions
        # invalid_transitions_matrix[from_id, to_id] = 1 if invalid
        violations = self.invalid_transitions_matrix[from_preds, to_preds]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Only count violations where both positions are valid
            valid_mask = (attention_mask[:, :-1] == 1) & (attention_mask[:, 1:] == 1)
            violations = violations * valid_mask.float()
        
        # Filter out positions with ignore_index
        if labels is not None:
            label_mask = (labels[:, :-1] != -100) & (labels[:, 1:] != -100)
            violations = violations * label_mask.float()
        
        # Calculate average violation rate
        total_valid_positions = violations.numel()
        if total_valid_positions == 0:
            return torch.tensor(0.0, device=device)
        
        violation_rate = violations.sum() / total_valid_positions
        
        return self.weight * violation_rate


class CombinedLoss(nn.Module):
    """Combination of multiple loss functions
    
    Supports:
    - Base loss (weighted CE or focal)
    - Boundary-aware loss
    
    Args:
        base_loss: Base loss function (WeightedCrossEntropyLoss or FocalLoss)
        boundary_loss: Optional boundary-aware loss
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        boundary_loss: Optional[BoundaryAwareLoss] = None
    ):
        super().__init__()
        self.base_loss = base_loss
        self.boundary_loss = boundary_loss
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            logits: Model predictions [batch_size, seq_len, num_labels]
            labels: True labels [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Combined loss (scalar)
        """
        # Calculate base loss
        total_loss = self.base_loss(logits, labels, attention_mask)
        
        # Add boundary loss if configured
        if self.boundary_loss is not None:
            # Get predictions for boundary loss
            predictions = torch.argmax(logits, dim=-1)
            boundary_penalty = self.boundary_loss(predictions, labels, attention_mask)
            total_loss = total_loss + boundary_penalty
        
        return total_loss


class LossFactory:
    """Factory for creating loss functions based on configuration
    
    Supported loss types:
    - "ce": Standard CrossEntropyLoss
    - "weighted_ce": Class-weighted CrossEntropyLoss
    - "focal": Focal Loss
    - "boundary_aware": Base loss + Boundary-aware loss
    - "combined": Weighted Focal + Boundary loss
    """
    
    @staticmethod
    def create_loss(
        config: Optional[Dict],
        label2id: Dict[str, int],
        id2label: Dict[int, str],
        num_labels: int
    ) -> nn.Module:
        """
        Create a loss function based on configuration
        
        Args:
            config: Loss configuration dict
            label2id: Label name to ID mapping
            id2label: ID to label name mapping
            num_labels: Total number of labels
            
        Returns:
            Configured loss function
        """
        # Default to standard CrossEntropyLoss if no config
        if config is None or not config:
            from torch.nn import CrossEntropyLoss
            return CrossEntropyLoss(ignore_index=-100)
        
        loss_type = config.get('type', 'ce')
        
        # Standard CrossEntropyLoss
        if loss_type == 'ce':
            from torch.nn import CrossEntropyLoss
            return CrossEntropyLoss(ignore_index=-100)
        
        # Weighted CrossEntropyLoss
        elif loss_type == 'weighted_ce':
            weight_multipliers = config.get('weight_multipliers', {})
            return WeightedCrossEntropyLoss(
                num_labels=num_labels,
                label2id=label2id,
                weight_multipliers=weight_multipliers,
                ignore_index=-100
            )
        
        # Focal Loss
        elif loss_type == 'focal':
            alpha = config.get('alpha', 0.25)
            gamma = config.get('gamma', 2.0)
            
            # Support class weights with focal loss
            weight_multipliers = config.get('weight_multipliers', None)
            label_weights = None
            if weight_multipliers:
                weights = torch.ones(num_labels, dtype=torch.float32)
                for label_name, multiplier in weight_multipliers.items():
                    if label_name in label2id:
                        label_id = label2id[label_name]
                        weights[label_id] = multiplier
                label_weights = weights
            
            return FocalLoss(
                num_labels=num_labels,
                alpha=alpha,
                gamma=gamma,
                ignore_index=-100,
                label_weights=label_weights
            )
        
        # Boundary-aware loss (base loss + boundary)
        elif loss_type == 'boundary_aware':
            base_loss_type = config.get('base_loss', 'weighted_ce')
            boundary_weight = config.get('boundary_weight', 0.15)
            
            # Create base loss recursively
            base_config = {
                'type': base_loss_type,
                'weight_multipliers': config.get('weight_multipliers', {}),
                'alpha': config.get('alpha', 0.25),
                'gamma': config.get('gamma', 2.0)
            }
            base_loss = LossFactory.create_loss(base_config, label2id, id2label, num_labels)
            
            # Create boundary loss
            boundary_loss = BoundaryAwareLoss(
                label2id=label2id,
                id2label=id2label,
                weight=boundary_weight
            )
            
            return CombinedLoss(base_loss, boundary_loss)
        
        # Combined loss (weighted focal + boundary)
        elif loss_type == 'combined':
            # Create weighted focal loss
            alpha = config.get('focal_alpha', 0.25)
            gamma = config.get('focal_gamma', 2.0)
            weight_multipliers = config.get('weight_multipliers', {})
            
            weights = torch.ones(num_labels, dtype=torch.float32)
            for label_name, multiplier in weight_multipliers.items():
                if label_name in label2id:
                    label_id = label2id[label_name]
                    weights[label_id] = multiplier
            
            focal_loss = FocalLoss(
                num_labels=num_labels,
                alpha=alpha,
                gamma=gamma,
                ignore_index=-100,
                label_weights=weights
            )
            
            # Create boundary loss
            boundary_weight = config.get('boundary_weight', 0.15)
            boundary_loss = BoundaryAwareLoss(
                label2id=label2id,
                id2label=id2label,
                weight=boundary_weight
            )
            
            return CombinedLoss(focal_loss, boundary_loss)
        
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

