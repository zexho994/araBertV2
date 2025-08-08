"""Model Wrapper for Transformers Compatibility

Provides wrapper classes to make standard transformers models compatible
with the NEREvaluator interface.
"""

import torch
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForTokenClassification


class TransformersNERModelWrapper:
    """Wrapper to make standard transformers models compatible with NEREvaluator"""
    
    def __init__(self, model: AutoModelForTokenClassification, tokenizer: AutoTokenizer, 
                 id2label: Dict[int, str], label2id: Dict[str, int]):
        """
        Initialize wrapper
        
        Args:
            model: Standard transformers model
            tokenizer: Tokenizer
            id2label: ID to label mapping
            label2id: Label to ID mapping
        """
        self.model = model
        self.tokenizer = tokenizer
        self.id2label = id2label
        self.label2id = label2id
        self.device = next(model.parameters()).device
        
        # Ensure model is in eval mode
        self.model.eval()
    
    def predict(
        self, 
        text: str, 
        tokenizer=None, 
        confidence_threshold: float = 0.5,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """Make prediction on text
        
        Args:
            text: Input text
            tokenizer: Tokenizer to use (ignored, uses self.tokenizer)
            confidence_threshold: Confidence threshold for predictions
            device: Device to use for inference (ignored, uses self.device)
            
        Returns:
            Prediction results compatible with NEREvaluator
        """
        # Use wrapper's tokenizer and device
        tokenizer = self.tokenizer
        device = self.device
        
        # Tokenize input
        words = text.split()
        tokenized = tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        
        # Get word IDs before moving to device
        word_ids = tokenized.word_ids(batch_index=0)
        
        # Move to device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**tokenized)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        # Align predictions with words
        word_predictions = []
        word_confidences = []
        previous_word_idx = None
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != previous_word_idx:
                if word_idx < len(words):
                    pred_id = predictions[0][i].item()
                    confidence = probabilities[0][i][pred_id].item()
                    
                    if confidence >= confidence_threshold:
                        label = self.id2label.get(pred_id, 'O')
                    else:
                        label = 'O'
                    
                    word_predictions.append(label)
                    word_confidences.append(confidence)
                    
                previous_word_idx = word_idx
        
        # Extract entities
        entities = self._extract_entities(words, word_predictions, word_confidences)
        
        return {
            'text': text,
            'tokens': words,
            'labels': word_predictions,
            'confidences': word_confidences,
            'entities': entities
        }
    
    def _extract_entities(self, tokens: List[str], labels: List[str], 
                         confidences: List[float]) -> List[Dict[str, Any]]:
        """Extract entities from BIO-tagged sequence"""
        entities = []
        current_entity = None
        
        for i, (token, label, confidence) in enumerate(zip(tokens, labels, confidences)):
            if label.startswith('B-'):
                # Start of new entity
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]
                current_entity = {
                    'type': entity_type,
                    'tokens': [token],
                    'start': i,
                    'end': i + 1,
                    'text': token,
                    'confidence': confidence
                }
            
            elif label.startswith('I-') and current_entity:
                # Continuation of current entity
                entity_type = label[2:]
                if current_entity['type'] == entity_type:
                    current_entity['tokens'].append(token)
                    current_entity['end'] = i + 1
                    current_entity['text'] += ' ' + token
                    # Update confidence (average)
                    current_entity['confidence'] = (
                        current_entity['confidence'] + confidence
                    ) / 2
                else:
                    # Entity type mismatch, start new entity
                    entities.append(current_entity)
                    current_entity = {
                        'type': entity_type,
                        'tokens': [token],
                        'start': i,
                        'end': i + 1,
                        'text': token,
                        'confidence': confidence
                    }
            
            else:
                # Outside entity or entity end
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add final entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def to(self, device):
        """Move model to device"""
        self.model.to(device)
        self.device = device
        return self
    
    def eval(self):
        """Set model to evaluation mode"""
        self.model.eval()
        return self
    
    def train(self, mode=True):
        """Set model to training mode"""
        self.model.train(mode)
        return self
    
    @property
    def config(self):
        """Access model config"""
        return self.model.config
    
    def __call__(self, *args, **kwargs):
        """Forward call to underlying model"""
        return self.model(*args, **kwargs)


def wrap_transformers_model(model_path: str) -> TransformersNERModelWrapper:
    """Load and wrap a standard transformers model
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Wrapped model compatible with NEREvaluator
    """
    # Load model and tokenizer
    model = AutoModelForTokenClassification.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Get label mappings from config
    id2label = model.config.id2label
    label2id = model.config.label2id
    
    # Create wrapper
    wrapper = TransformersNERModelWrapper(model, tokenizer, id2label, label2id)
    
    return wrapper


def create_compatible_evaluator(model_path: str, device=None):
    """Create NEREvaluator with wrapped model
    
    Args:
        model_path: Path to the saved model
        device: Device to use
        
    Returns:
        NEREvaluator instance with wrapped model
    """
    from ..evaluation.evaluator import NEREvaluator
    
    # Wrap the model
    wrapped_model = wrap_transformers_model(model_path)
    
    # Move to device if specified
    if device is not None:
        wrapped_model.to(device)
    
    # Create label list
    label_list = [wrapped_model.id2label[i] for i in range(len(wrapped_model.id2label))]
    
    # Create evaluator
    evaluator = NEREvaluator(
        model=wrapped_model,
        tokenizer=wrapped_model.tokenizer,
        label_list=label_list,
        device=wrapped_model.device
    )
    
    return evaluator