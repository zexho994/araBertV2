"""NER Model Definitions

Implements base NER model class and specific model implementations
including BERT-based NER models.
"""

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import (
    AutoModel, AutoConfig, AutoTokenizer,
    BertModel, BertConfig, BertTokenizer,
    PreTrainedModel, PretrainedConfig
)
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np

class NERModelConfig(PretrainedConfig):
    """Configuration class for NER models"""
    
    model_type = "ner"
    
    def __init__(
        self,
        num_labels: int = 2,
        hidden_dropout_prob: float = 0.1,
        classifier_dropout: Optional[float] = None,
        label2id: Optional[Dict[str, int]] = None,
        id2label: Optional[Dict[int, str]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.num_labels = num_labels
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_dropout = classifier_dropout
        
        if label2id is not None and id2label is not None:
            if len(label2id) != len(id2label):
                raise ValueError("label2id and id2label must have the same length")
            self.label2id = label2id
            self.id2label = id2label
        elif label2id is not None:
            self.label2id = label2id
            self.id2label = {v: k for k, v in label2id.items()}
        elif id2label is not None:
            self.id2label = id2label
            self.label2id = {v: k for k, v in id2label.items()}
        else:
            self.label2id = {}
            self.id2label = {}

class NERModel(PreTrainedModel):
    """Base class for NER models"""
    
    config_class = NERModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward method")
    
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
            tokenizer: Tokenizer to use
            confidence_threshold: Confidence threshold for predictions
            device: Device to use for inference
            
        Returns:
            Prediction results
        """
        if tokenizer is None:
            raise ValueError("Tokenizer is required for prediction")
        
        if device is None:
            device = next(self.parameters()).device
        
        self.eval()
        
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
        
        # Move to device
        tokenized = {k: v.to(device) for k, v in tokenized.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self(**tokenized)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
        
        # Get word IDs for alignment
        word_ids = tokenized.word_ids()
        
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
                        label = self.config.id2label.get(pred_id, 'O')
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
                # Outside or end of entity
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # Add last entity if exists
        if current_entity:
            entities.append(current_entity)
        
        return entities

class BertNERModel(NERModel):
    """BERT-based NER model"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Ensure num_labels is set
        self.num_labels = config.num_labels
        
        # Load BERT model
        if hasattr(config, 'model_name'):
            self.bert = AutoModel.from_pretrained(config.model_name)
        else:
            self.bert = BertModel(config)
        
        # Dropout
        classifier_dropout = (
            config.classifier_dropout 
            if config.classifier_dropout is not None 
            else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        # Classification head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        # Initialize weights
        self.init_weights()
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        num_labels: int,
        dropout: float = 0.1,
        **kwargs
    ):
        """Load pretrained BERT model for NER
        
        Args:
            pretrained_model_name_or_path: Model name or path
            num_labels: Number of labels
            dropout: Dropout rate
            **kwargs: Additional arguments
            
        Returns:
            BertNERModel instance
        """
        # Load base configuration
        base_config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        
        # Create NER configuration
        config = NERModelConfig(
            vocab_size=base_config.vocab_size,
            hidden_size=base_config.hidden_size,
            num_hidden_layers=base_config.num_hidden_layers,
            num_attention_heads=base_config.num_attention_heads,
            intermediate_size=base_config.intermediate_size,
            hidden_act=base_config.hidden_act,
            hidden_dropout_prob=base_config.hidden_dropout_prob,
            attention_probs_dropout_prob=base_config.attention_probs_dropout_prob,
            max_position_embeddings=base_config.max_position_embeddings,
            type_vocab_size=base_config.type_vocab_size,
            initializer_range=base_config.initializer_range,
            layer_norm_eps=base_config.layer_norm_eps,
            pad_token_id=base_config.pad_token_id,
            position_embedding_type=getattr(base_config, 'position_embedding_type', 'absolute'),
            num_labels=num_labels,
            classifier_dropout=dropout,
            model_name=pretrained_model_name_or_path,
            **kwargs
        )
        
        # Ensure num_labels is properly set
        config.num_labels = num_labels
        
        # Create model
        model = cls(config)
        
        # Load pretrained weights for BERT
        model.bert = AutoModel.from_pretrained(pretrained_model_name_or_path)
        
        return model
    
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """Forward pass
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            token_type_ids: Token type IDs
            labels: Labels for training
            
        Returns:
            Model outputs
        """
        # BERT forward pass
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Get sequence output
        sequence_output = outputs.last_hidden_state
        
        # Apply dropout
        sequence_output = self.dropout(sequence_output)
        
        # Classification
        logits = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss,
                    labels.view(-1),
                    torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            'attentions': outputs.attentions if hasattr(outputs, 'attentions') else None
        }
    
    def get_input_embeddings(self):
        """Get input embeddings"""
        return self.bert.embeddings.word_embeddings
    
    def set_input_embeddings(self, value):
        """Set input embeddings"""
        self.bert.embeddings.word_embeddings = value
    
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings"""
        old_embeddings = self.get_input_embeddings()
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.set_input_embeddings(new_embeddings)
        return self.get_input_embeddings()
    
    def _get_resized_embeddings(
        self, old_embeddings: nn.Embedding, new_num_tokens: int
    ) -> nn.Embedding:
        """Resize embeddings"""
        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        
        if old_num_tokens == new_num_tokens:
            return old_embeddings
        
        # Create new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)
        
        # Copy old weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        return new_embeddings
    
    def freeze_bert_layers(self, num_layers: int = 0):
        """Freeze BERT layers
        
        Args:
            num_layers: Number of layers to freeze (0 = freeze all)
        """
        if num_layers == 0:
            # Freeze all BERT parameters
            for param in self.bert.parameters():
                param.requires_grad = False
        else:
            # Freeze specific number of layers
            layers_to_freeze = self.bert.encoder.layer[:num_layers]
            for layer in layers_to_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
    
    def unfreeze_bert_layers(self):
        """Unfreeze all BERT layers"""
        for param in self.bert.parameters():
            param.requires_grad = True
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params
        }
    
    def save_pretrained(self, save_directory: str):
        """Save model"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # Save model state
        model_path = os.path.join(save_directory, 'pytorch_model.bin')
        torch.save(self.state_dict(), model_path)
        
        # Save configuration
        config_path = os.path.join(save_directory, 'config.json')
        self.config.save_pretrained(save_directory)