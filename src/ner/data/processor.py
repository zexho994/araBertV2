"""NER Data Processor

Handles data loading, validation, preprocessing, and format conversion
for NER training and evaluation.
"""

import os
import json
import re
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

class NERDataProcessor:
    """Main data processor for NER tasks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.labels = self.config.get('labels', {}).get('entities', [])
        self.label_scheme = self.config.get('labels', {}).get('scheme', 'BIO')
        self.max_length = self.config.get('data', {}).get('max_length', 512)
        self.encoding = self.config.get('data', {}).get('encoding', 'utf-8')
        
        # Initialize label mappings
        self._init_label_mappings()
    
    def _init_label_mappings(self):
        """Initialize label to ID mappings"""
        if self.labels:
            # Create BIO tags for each entity
            bio_labels = ['O']  # Outside
            for entity in self.labels:
                bio_labels.extend([f'B-{entity}', f'I-{entity}'])
            
            self.label2id = {label: idx for idx, label in enumerate(bio_labels)}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
        else:
            self.label2id = {}
            self.id2label = {}
    
    def load_conll_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from CoNLL format file
        
        Args:
            file_path: Path to CoNLL format file
            
        Returns:
            List of examples with tokens and labels
        """
        examples = []
        current_tokens = []
        current_labels = []
        
        with open(file_path, 'r', encoding=self.encoding) as f:
            for line in f:
                line = line.strip()
                
                if not line:  # Empty line indicates end of sentence
                    if current_tokens:
                        examples.append({
                            'tokens': current_tokens.copy(),
                            'labels': current_labels.copy(),
                            'text': ' '.join(current_tokens)
                        })
                        current_tokens.clear()
                        current_labels.clear()
                else:
                    parts = line.split()
                    if len(parts) >= 2:
                        token = parts[0]
                        label = parts[-1]  # Last column is label
                        current_tokens.append(token)
                        current_labels.append(label)
        
        # Handle last sentence if file doesn't end with empty line
        if current_tokens:
            examples.append({
                'tokens': current_tokens,
                'labels': current_labels,
                'text': ' '.join(current_tokens)
            })
        
        return examples
    
    def load_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON format file
        
        Args:
            file_path: Path to JSON format file
            
        Returns:
            List of examples
        """
        with open(file_path, 'r', encoding=self.encoding) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'examples' in data:
            return data['examples']
        else:
            raise ValueError(f"Unsupported JSON format in {file_path}")
    
    def load_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from CSV format file
        
        Args:
            file_path: Path to CSV format file
            
        Returns:
            List of examples
        """
        df = pd.read_csv(file_path, encoding=self.encoding)
        
        examples = []
        for _, row in df.iterrows():
            if 'text' in row and 'labels' in row:
                # Assume labels are space-separated or JSON string
                labels = row['labels']
                if isinstance(labels, str):
                    try:
                        labels = json.loads(labels)
                    except:
                        labels = labels.split()
                
                examples.append({
                    'text': row['text'],
                    'tokens': row['text'].split() if 'tokens' not in row else row['tokens'],
                    'labels': labels
                })
        
        return examples
    
    def load_data_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from file (auto-detect format)
        
        Args:
            file_path: Path to data file
            
        Returns:
            List of examples
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix in ['.conll', '.conllu', '.txt']:
            return self.load_conll_file(str(file_path))
        elif suffix == '.json':
            return self.load_json_file(str(file_path))
        elif suffix == '.csv':
            return self.load_csv_file(str(file_path))
        else:
            # Try to auto-detect format
            with open(file_path, 'r', encoding=self.encoding) as f:
                first_line = f.readline().strip()
                
            if first_line.startswith('{') or first_line.startswith('['):
                return self.load_json_file(str(file_path))
            elif '\t' in first_line or len(first_line.split()) == 2:
                return self.load_conll_file(str(file_path))
            else:
                raise ValueError(f"Cannot determine format for file: {file_path}")
    
    def validate_data_file(self, file_path: str) -> bool:
        """Validate data file format and content
        
        Args:
            file_path: Path to data file
            
        Returns:
            True if valid, False otherwise
        """
        try:
            examples = self.load_data_file(file_path)
            return self.validate_examples(examples)
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    def validate_examples(self, examples: List[Dict[str, Any]]) -> bool:
        """Validate list of examples
        
        Args:
            examples: List of examples to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not examples:
            print("No examples found")
            return False
        
        for i, example in enumerate(examples):
            # Check required fields
            if 'tokens' not in example or 'labels' not in example:
                print(f"Example {i}: Missing required fields (tokens, labels)")
                return False
            
            tokens = example['tokens']
            labels = example['labels']
            
            # Check tokens and labels length match
            if len(tokens) != len(labels):
                print(f"Example {i}: Token count ({len(tokens)}) != Label count ({len(labels)})")
                return False
            
            # Check label validity
            if self.labels:
                valid_labels = set(['O'] + [f'B-{entity}' for entity in self.labels] + [f'I-{entity}' for entity in self.labels])
                for j, label in enumerate(labels):
                    if label not in valid_labels:
                        print(f"Example {i}, Token {j}: Invalid label '{label}'")
                        return False
            
            # Check BIO consistency
            if not self._validate_bio_sequence(labels):
                print(f"Example {i}: Invalid BIO sequence")
                return False
        
        return True
    
    def _validate_bio_sequence(self, labels: List[str]) -> bool:
        """Validate BIO label sequence
        
        Args:
            labels: List of BIO labels
            
        Returns:
            True if valid BIO sequence
        """
        for i, label in enumerate(labels):
            if label.startswith('I-'):
                entity_type = label[2:]
                # I- must be preceded by B- or I- of same entity type
                if i == 0:
                    return False
                prev_label = labels[i-1]
                if not (prev_label == f'B-{entity_type}' or prev_label == f'I-{entity_type}'):
                    return False
        
        return True
    
    def process_file(self, input_file: str, output_file: str, format: str = 'json'):
        """Process and convert data file
        
        Args:
            input_file: Input file path
            output_file: Output file path
            format: Output format ('json', 'conll', 'csv')
        """
        examples = self.load_data_file(input_file)
        
        # Apply preprocessing
        processed_examples = []
        for example in examples:
            processed_example = self.preprocess_example(example)
            if processed_example:
                processed_examples.append(processed_example)
        
        # Save in specified format
        self.save_examples(processed_examples, output_file, format)
    
    def preprocess_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Preprocess a single example
        
        Args:
            example: Input example
            
        Returns:
            Processed example or None if should be filtered
        """
        tokens = example['tokens']
        labels = example['labels']
        
        # Filter by length
        if len(tokens) > self.max_length:
            # Truncate
            tokens = tokens[:self.max_length]
            labels = labels[:self.max_length]
        
        if len(tokens) == 0:
            return None
        
        # Clean tokens
        cleaned_tokens = []
        cleaned_labels = []
        
        for token, label in zip(tokens, labels):
            # Basic cleaning
            token = token.strip()
            if token:
                cleaned_tokens.append(token)
                cleaned_labels.append(label)
        
        if not cleaned_tokens:
            return None
        
        return {
            'tokens': cleaned_tokens,
            'labels': cleaned_labels,
            'text': ' '.join(cleaned_tokens)
        }
    
    def save_examples(self, examples: List[Dict[str, Any]], output_file: str, format: str = 'json'):
        """Save examples to file
        
        Args:
            examples: List of examples
            output_file: Output file path
            format: Output format
        """
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_file, 'w', encoding=self.encoding) as f:
                json.dump(examples, f, indent=2, ensure_ascii=False)
        
        elif format == 'conll':
            with open(output_file, 'w', encoding=self.encoding) as f:
                for example in examples:
                    tokens = example['tokens']
                    labels = example['labels']
                    
                    for token, label in zip(tokens, labels):
                        f.write(f"{token}\t{label}\n")
                    f.write("\n")  # Empty line between sentences
        
        elif format == 'csv':
            df_data = []
            for example in examples:
                df_data.append({
                    'text': example['text'],
                    'tokens': json.dumps(example['tokens'], ensure_ascii=False),
                    'labels': json.dumps(example['labels'], ensure_ascii=False)
                })
            
            df = pd.DataFrame(df_data)
            df.to_csv(output_file, index=False, encoding=self.encoding)
        
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
    def split_data(self, input_file: str, output_dir: str, 
                   train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1,
                   random_state: int = 42):
        """Split data into train/validation/test sets
        
        Args:
            input_file: Input data file
            output_dir: Output directory
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            test_ratio: Test data ratio
            random_state: Random seed
        """
        examples = self.load_data_file(input_file)
        
        # First split: train + val vs test
        train_val, test = train_test_split(
            examples, 
            test_size=test_ratio, 
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        train, val = train_test_split(
            train_val, 
            test_size=val_size, 
            random_state=random_state
        )
        
        # Save splits
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.save_examples(train, output_path / 'train.json')
        self.save_examples(val, output_path / 'val.json')
        self.save_examples(test, output_path / 'test.json')
        
        print(f"Data split completed:")
        print(f"  Train: {len(train)} examples")
        print(f"  Validation: {len(val)} examples")
        print(f"  Test: {len(test)} examples")
    
    def get_label_statistics(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get label statistics from examples
        
        Args:
            examples: List of examples
            
        Returns:
            Dictionary with label statistics
        """
        label_counts = {}
        entity_counts = {}
        total_tokens = 0
        
        for example in examples:
            labels = example['labels']
            total_tokens += len(labels)
            
            for label in labels:
                label_counts[label] = label_counts.get(label, 0) + 1
                
                if label.startswith('B-'):
                    entity = label[2:]
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        return {
            'total_examples': len(examples),
            'total_tokens': total_tokens,
            'label_counts': label_counts,
            'entity_counts': entity_counts,
            'avg_tokens_per_example': total_tokens / len(examples) if examples else 0
        }
    
    def convert_to_model_format(self, examples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert examples to model input format
        
        Args:
            examples: List of examples
            
        Returns:
            List of examples in model format
        """
        model_examples = []
        
        for example in examples:
            tokens = example['tokens']
            labels = example['labels']
            
            # Convert labels to IDs if mapping available
            if self.label2id:
                label_ids = [self.label2id.get(label, 0) for label in labels]
            else:
                label_ids = labels
            
            model_examples.append({
                'tokens': tokens,
                'labels': labels,
                'label_ids': label_ids,
                'text': example.get('text', ' '.join(tokens))
            })
        
        return model_examples
    
    def extract_entities(self, tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
        """Extract entities from BIO-tagged sequence
        
        Args:
            tokens: List of tokens
            labels: List of BIO labels
            
        Returns:
            List of extracted entities
        """
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
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
                    'text': token
                }
            
            elif label.startswith('I-') and current_entity:
                # Continuation of current entity
                entity_type = label[2:]
                if current_entity['type'] == entity_type:
                    current_entity['tokens'].append(token)
                    current_entity['end'] = i + 1
                    current_entity['text'] += ' ' + token
                else:
                    # Entity type mismatch, start new entity
                    entities.append(current_entity)
                    current_entity = {
                        'type': entity_type,
                        'tokens': [token],
                        'start': i,
                        'end': i + 1,
                        'text': token
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