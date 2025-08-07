"""NER Data Loader

Implements PyTorch Dataset and DataLoader for NER training and evaluation.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from transformers import AutoTokenizer

class NERDataset(Dataset):
    """PyTorch Dataset for NER data"""
    
    def __init__(self, examples: List[Dict[str, Any]], tokenizer, 
                 label2id: Dict[str, int], max_length: int = 512,
                 pad_token_label_id: int = -100):
        """
        Initialize NER Dataset
        
        Args:
            examples: List of examples with tokens and labels
            tokenizer: Tokenizer for encoding text
            label2id: Label to ID mapping
            max_length: Maximum sequence length
            pad_token_label_id: Label ID for padding tokens
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.pad_token_label_id = pad_token_label_id
        
        # Process examples
        self.processed_examples = self._process_examples()
    
    def _process_examples(self) -> List[Dict[str, Any]]:
        """Process examples for model input"""
        processed = []
        
        for example in self.examples:
            tokens = example['tokens']
            labels = example['labels']
            
            # Tokenize and align labels
            tokenized = self._tokenize_and_align_labels(tokens, labels)
            
            if tokenized:
                processed.append(tokenized)
        
        return processed
    
    def _tokenize_and_align_labels(self, tokens: List[str], labels: List[str]) -> Optional[Dict[str, Any]]:
        """Tokenize tokens and align labels with subword tokens"""
        # Tokenize each word and keep track of word boundaries
        tokenized_inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get word IDs to align labels
        word_ids = tokenized_inputs.word_ids()
        
        # Align labels with tokenized inputs
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens (CLS, SEP, PAD)
                aligned_labels.append(self.pad_token_label_id)
            elif word_idx != previous_word_idx:
                # First subword of a word
                if word_idx < len(labels):
                    label = labels[word_idx]
                    aligned_labels.append(self.label2id.get(label, 0))
                else:
                    aligned_labels.append(self.pad_token_label_id)
            else:
                # Subsequent subwords of the same word
                if word_idx < len(labels):
                    label = labels[word_idx]
                    if label.startswith('B-'):
                        # Convert B- to I- for subwords
                        i_label = 'I-' + label[2:]
                        aligned_labels.append(self.label2id.get(i_label, 0))
                    else:
                        aligned_labels.append(self.label2id.get(label, 0))
                else:
                    aligned_labels.append(self.pad_token_label_id)
            
            previous_word_idx = word_idx
        
        return {
            'input_ids': tokenized_inputs['input_ids'].squeeze(),
            'attention_mask': tokenized_inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long),
            'original_tokens': tokens,
            'original_labels': labels
        }
    
    def __len__(self) -> int:
        return len(self.processed_examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_examples[idx]

class NERDataLoader:
    """Data loader manager for NER tasks"""
    
    def __init__(self, tokenizer_name: str, label2id: Dict[str, int], 
                 max_length: int = 512, pad_token_label_id: int = -100):
        """
        Initialize NER Data Loader
        
        Args:
            tokenizer_name: Name or path of tokenizer
            label2id: Label to ID mapping
            max_length: Maximum sequence length
            pad_token_label_id: Label ID for padding tokens
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.label2id = label2id
        self.max_length = max_length
        self.pad_token_label_id = pad_token_label_id
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def create_dataset(self, examples: List[Dict[str, Any]]) -> NERDataset:
        """Create NER dataset from examples
        
        Args:
            examples: List of examples
            
        Returns:
            NERDataset instance
        """
        return NERDataset(
            examples=examples,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            max_length=self.max_length,
            pad_token_label_id=self.pad_token_label_id
        )
    
    def create_dataloader(self, dataset: NERDataset, batch_size: int = 16, 
                         shuffle: bool = True, num_workers: int = 0) -> DataLoader:
        """Create PyTorch DataLoader
        
        Args:
            dataset: NER dataset
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            
        Returns:
            DataLoader instance
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate function for batching"""
        # Stack tensors
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def prepare_data_loaders(self, train_examples: List[Dict[str, Any]], 
                           val_examples: Optional[List[Dict[str, Any]]] = None,
                           test_examples: Optional[List[Dict[str, Any]]] = None,
                           batch_size: int = 16, 
                           num_workers: int = 0) -> Dict[str, DataLoader]:
        """Prepare data loaders for training, validation, and testing
        
        Args:
            train_examples: Training examples
            val_examples: Validation examples (optional)
            test_examples: Test examples (optional)
            batch_size: Batch size
            num_workers: Number of worker processes
            
        Returns:
            Dictionary of data loaders
        """
        data_loaders = {}
        
        # Training data loader
        train_dataset = self.create_dataset(train_examples)
        data_loaders['train'] = self.create_dataloader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        
        # Validation data loader
        if val_examples:
            val_dataset = self.create_dataset(val_examples)
            data_loaders['val'] = self.create_dataloader(
                val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        
        # Test data loader
        if test_examples:
            test_dataset = self.create_dataset(test_examples)
            data_loaders['test'] = self.create_dataloader(
                test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
            )
        
        return data_loaders
    
    def get_tokenizer(self):
        """Get the tokenizer instance"""
        return self.tokenizer

class NERTokenizer:
    """Wrapper for NER tokenization operations"""
    
    def __init__(self, tokenizer_name: str, max_length: int = 512):
        """
        Initialize NER Tokenizer
        
        Args:
            tokenizer_name: Name or path of tokenizer
            max_length: Maximum sequence length
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def tokenize_text(self, text: str) -> Dict[str, Any]:
        """Tokenize text for prediction
        
        Args:
            text: Input text
            
        Returns:
            Tokenized inputs
        """
        # Split text into words
        words = text.split()
        
        # Tokenize
        tokenized = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get word IDs for alignment
        word_ids = tokenized.word_ids()
        
        return {
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'word_ids': word_ids,
            'words': words
        }
    
    def decode_predictions(self, predictions: torch.Tensor, word_ids: List[Optional[int]], 
                         words: List[str], id2label: Dict[int, str]) -> List[Tuple[str, str]]:
        """Decode model predictions to word-label pairs
        
        Args:
            predictions: Model predictions (token-level)
            word_ids: Word IDs for alignment
            words: Original words
            id2label: ID to label mapping
            
        Returns:
            List of (word, label) pairs
        """
        # Convert predictions to labels
        predicted_labels = [id2label.get(pred.item(), 'O') for pred in predictions]
        
        # Align with words
        word_labels = []
        previous_word_idx = None
        
        for word_idx, label in zip(word_ids, predicted_labels):
            if word_idx is not None and word_idx != previous_word_idx:
                if word_idx < len(words):
                    word_labels.append((words[word_idx], label))
                previous_word_idx = word_idx
        
        return word_labels
    
    def batch_tokenize(self, texts: List[str]) -> Dict[str, torch.Tensor]:
        """Tokenize batch of texts
        
        Args:
            texts: List of input texts
            
        Returns:
            Batched tokenized inputs
        """
        return self.tokenizer(
            texts,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    def get_vocab_size(self) -> int:
        """Get tokenizer vocabulary size"""
        return len(self.tokenizer)
    
    def get_special_tokens(self) -> Dict[str, str]:
        """Get special tokens"""
        return {
            'pad_token': self.tokenizer.pad_token,
            'cls_token': self.tokenizer.cls_token,
            'sep_token': self.tokenizer.sep_token,
            'unk_token': self.tokenizer.unk_token,
            'mask_token': getattr(self.tokenizer, 'mask_token', None)
        }