# Custom Loss Functions Guide

## Overview

This guide describes the custom loss functions implemented to improve NER model performance, particularly for BUILDING and STREET entities.

## Available Loss Functions

### 1. Weighted Cross Entropy Loss

**Purpose**: Addresses class imbalance by assigning higher weights to underrepresented entity types.

**Configuration**:
```json
{
  "training": {
    "loss": {
      "type": "weighted_ce",
      "weight_multipliers": {
        "B-BUILDING": 2.5,
        "I-BUILDING": 2.5,
        "B-STREET": 2.5,
        "I-STREET": 2.5,
        "O": 0.6
      }
    }
  }
}
```

**When to use**:
- First approach for addressing class imbalance
- When certain entity types have significantly lower F1 scores
- Conservative approach with predictable behavior

**Expected Impact**: +10-15% F1 improvement for targeted entities

### 2. Focal Loss

**Purpose**: Automatically focuses on hard-to-classify examples while down-weighting easy ones.

**Configuration**:
```json
{
  "training": {
    "loss": {
      "type": "focal",
      "alpha": 0.25,
      "gamma": 2.0,
      "weight_multipliers": {
        "B-BUILDING": 2.0,
        "I-BUILDING": 2.0,
        "B-STREET":2.0,
        "I-STREET":2.0
      }
    }
  }
}
```

**Parameters**:
- `alpha`: Balance factor (default: 0.25)
- `gamma`: Focusing parameter (default: 2.0). Higher values = more focus on hard examples

**When to use**:
- When weighted CE doesn't provide enough improvement
- When you have many easy examples dominating the loss
- For handling noisy or difficult boundaries

**Expected Impact**: +15-20% F1 improvement for targeted entities

### 3. Boundary-Aware Loss

**Purpose**: Explicitly penalizes invalid BIO tag transitions.

**Configuration**:
```json
{
  "training": {
    "loss": {
      "type": "boundary_aware",
      "base_loss": "weighted_ce",
      "boundary_weight": 0.15,
      "weight_multipliers": {
        "B-BUILDING": 2.5,
        "I-BUILDING": 2.5
      }
    }
  }
}
```

**Invalid Transitions Penalized**:
1. I-X at sequence start
2. I-X following B-Y or I-Y where X ≠ Y
3. I-X following O

**When to use**:
- When model produces many invalid BIO sequences
- As an additional regularization term
- When entity boundaries are inconsistent

**Expected Impact**: +5-10% F1 improvement (combined with base loss)

### 4. Combined Loss

**Purpose**: Combines weighted focal loss with boundary-aware regularization.

**Configuration**:
```json
{
  "training": {
    "loss": {
      "type": "combined",
      "focal_alpha": 0.25,
      "focal_gamma": 2.0,
      "weight_multipliers": {
        "B-BUILDING": 3.0,
        "I-BUILDING": 3.0,
        "B-STREET": 3.0,
        "I-STREET": 3.0,
        "O": 0.5
      },
      "boundary_weight": 0.2
    }
  }
}
```

**When to use**:
- When single loss functions don't reach target F1
- For maximum performance (at cost of complexity)
- When both class imbalance and boundary issues exist

**Expected Impact**: +20-30% F1 improvement for targeted entities

## Recommended Training Strategy

### Phase 1: Baseline (Weighted CE)
1. Start with conservative weighted CE configuration
2. Train for 3 epochs
3. Evaluate BUILDING and STREET F1 scores

**Conservative Config**:
```json
{
  "type": "weighted_ce",
  "weight_multipliers": {
    "B-BUILDING": 2.5,
    "I-BUILDING": 2.5,
    "B-STREET": 2.5,
    "I-STREET": 2.5,
    "O": 0.6
  }
}
```

### Phase 2: If F1 < 0.75 (Add Focal Loss)
1. Switch to focal loss with class weights
2. Adjust gamma parameter if needed (higher = more focus on hard examples)

**Moderate Config**:
```json
{
  "type": "focal",
  "alpha": 0.25,
  "gamma": 2.0,
  "weight_multipliers": {
    "B-BUILDING": 2.5,
    "I-BUILDING": 2.5,
    "B-STREET": 2.5,
    "I-STREET": 2.5,
    "O": 0.6
  }
}
```

### Phase 3: If F1 < 0.82 (Add Boundary Loss)
1. Use combined loss with both focal and boundary components
2. Increase weight multipliers if needed

**Aggressive Config**:
```json
{
  "type": "combined",
  "focal_alpha": 0.25,
  "focal_gamma": 2.0,
  "weight_multipliers": {
    "B-BUILDING": 3.0,
    "I-BUILDING": 3.0,
    "B-STREET": 3.0,
    "I-STREET": 3.0,
    "O": 0.5
  },
  "boundary_weight": 0.2
}
```

## Hyperparameter Tuning Guidelines

### Weight Multipliers
- **Conservative**: 1.5-2.5x for target entities
- **Moderate**: 2.5-3.5x for target entities
- **Aggressive**: 3.0-5.0x for target entities
- **O label**: 0.5-0.8x (prevent over-prediction of O)

**Warning**: Too high weights can cause:
- Model instability
- Overfitting to target entities
- Poor performance on other entities

### Focal Loss Parameters

**Alpha (α)**:
- Lower (0.1-0.2): Less class balancing
- Default (0.25): Balanced
- Higher (0.3-0.5): More aggressive class balancing

**Gamma (γ)**:
- Lower (1.0-1.5): Less focus on hard examples
- Default (2.0): Balanced
- Higher (2.5-3.0): More focus on hard examples

### Boundary Loss Weight
- **Conservative**: 0.1
- **Default**: 0.15
- **Aggressive**: 0.2-0.3

**Warning**: Too high boundary weight can:
- Prevent valid edge case patterns
- Slow convergence

## Implementation Details

### Code Structure
- **Loss Functions**: `src/ner/training/losses.py`
- **Model Integration**: `src/ner/models/model.py` (line 365-383)
- **Trainer Integration**: `src/ner/training/trainer.py` (line 301-305)
- **Configuration**: `data/ner/configs/countries/uae.json`

### Testing
Run loss function tests:
```bash
python3 -m pytest tests/ner/test_losses.py -v
```

### Monitoring Training
Loss function configuration is logged at training start:
```
INFO: Configured custom loss function: weighted_ce
```

TensorBoard metrics to monitor:
- `train/batch_loss`: Training loss per batch
- `train/epoch_loss`: Average training loss per epoch
- `val/f1`: Validation F1 score (overall)
- `val/entity_{ENTITY}_f1`: Per-entity F1 scores

## Troubleshooting

### Loss Increases During Training
**Possible causes**:
- Weight multipliers too aggressive
- Learning rate too high
- Boundary weight too high

**Solutions**:
- Reduce weight multipliers by 20-30%
- Lower learning rate
- Reduce boundary_weight

### No Improvement in Target Entities
**Possible causes**:
- Weight multipliers too conservative
- Data quality issues
- Insufficient training data

**Solutions**:
- Increase weight multipliers
- Check data labeling consistency
- Add more training examples

### Model Overfits to Target Entities
**Possible causes**:
- Weight multipliers too high
- Training too long

**Solutions**:
- Reduce weight multipliers
- Enable early stopping
- Reduce training epochs

## References

- Focal Loss: [Lin et al., "Focal Loss for Dense Object Detection" (2017)](https://arxiv.org/abs/1708.02002)
- Class Weighting: Standard practice in imbalanced classification
- BIO Tagging: [Ramshaw & Marcus, "Text Chunking using Transformation-Based Learning" (1995)](https://aclanthology.org/W95-0107/)

