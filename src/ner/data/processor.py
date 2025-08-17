"""NER Data Processor

Handles data loading, validation, preprocessing, and format conversion
for NER training and evaluation.
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

from ..utils import NERLogger

class NERDataProcessor:
    """Main data processor for NER tasks"""
    
    def __init__(self, config: Optional[Dict[str, Any]], logger: NERLogger):
        if logger is None:
            raise ValueError("NERDataProcessor requires a logger instance")
        self.logger = logger
        self.config = config or {}
        self.labels = self.config.get('labels', {}).get('label_mapping', [])
        self.max_length = self.config.get('data', {}).get('max_length', 512)
        self.encoding = self.config.get('data', {}).get('encoding', 'utf-8')
        
        self.logger.info("Initializing NERDataProcessor")
        self.logger.debug(f"Processor config: max_length={self.max_length}, encoding={self.encoding}")
        
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
            self.logger.info(f"Initialized label mappings for {len(self.labels)} entities, total BIO labels={len(bio_labels)}")
        else:
            self.label2id = {}
            self.id2label = {}
            self.logger.warning("No labels provided in config; label mappings are empty")
    
    def _load_conll_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from CoNLL format file
        
        Args:
            file_path: Path to CoNLL format file
            
        Returns:
            List of examples with tokens and labels
        """
        examples = []
        current_tokens = []
        current_labels = []
        
        self.logger.info(f"Loading CoNLL data from {file_path}")
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
        
        self.logger.info(f"Loaded {len(examples)} examples from CoNLL file")
        return examples
    
    def _load_json_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON format file
        
        Args:
            file_path: Path to JSON format file
            
        Returns:
            List of examples
        """
        self.logger.info(f"Loading JSON data from {file_path}")
        with open(file_path, 'r', encoding=self.encoding) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            self.logger.info(f"Loaded {len(data)} examples from JSON file")
            return data
        elif isinstance(data, dict) and 'examples' in data:
            self.logger.info(f"Loaded {len(data['examples'])} examples from JSON file (wrapped)")
            return data['examples']
        else:
            raise ValueError(f"Unsupported JSON format in {file_path}")

    def _load_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from JSON Lines (JSONL) format file

        Args:
            file_path: Path to JSONL format file

        Returns:
            List of examples (each line is a JSON object)
        """
        self.logger.info(f"Loading JSONL data from {file_path}")
        examples: List[Dict[str, Any]] = []
        with open(file_path, 'r', encoding=self.encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    examples.append(obj)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL line in {file_path}: {e}")
        self.logger.info(f"Loaded {len(examples)} examples from JSONL file")
        return examples
    
    def _load_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from CSV format file
        
        Args:
            file_path: Path to CSV format file
            
        Returns:
            List of examples
        """
        self.logger.info(f"Loading CSV data from {file_path}")
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
        
        self.logger.info(f"Loaded {len(examples)} examples from CSV file")
        return examples
    
    def load_data_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Load data from file (auto-detect format)
        
        Args:
            file_path: Path to data file
            
        Returns:
            List of examples
        """
        self.logger.info(f"Loading data file: {file_path}")
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        suffix = file_path.suffix.lower()
        
        if suffix in ['.conll', '.conllu', '.txt']:
            return self._load_conll_file(str(file_path))
        elif suffix == '.json':
            return self._load_json_file(str(file_path))
        elif suffix == '.jsonl':
            return self._load_jsonl_file(str(file_path))
        elif suffix == '.csv':
            return self._load_csv_file(str(file_path))
        else:
            # Try to auto-detect format
            with open(file_path, 'r', encoding=self.encoding) as f:
                first_line = f.readline().strip()
                
            if first_line.startswith('{') or first_line.startswith('['):
                # Try JSON first, then fallback to JSONL
                try:
                    return self._load_json_file(str(file_path))
                except Exception:
                    return self._load_jsonl_file(str(file_path))
            elif '\t' in first_line or len(first_line.split()) == 2:
                return self._load_conll_file(str(file_path))
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
            valid = self._validate_examples(examples)
            if valid:
                self.logger.info(f"Data file '{file_path}' validation passed: {len(examples)} examples")
            else:
                self.logger.warning(f"Data file '{file_path}' validation failed")
            return valid
        except Exception as e:
            self.logger.error(f"Validation error for '{file_path}': {e}")
            return False
    
    def _validate_examples(self, examples: List[Dict[str, Any]]) -> bool:
        """Validate list of examples
        
        Args:
            examples: List of examples to validate
            
        Returns:
            True if valid, False otherwise
        """
        if not examples:
            self.logger.warning("No examples found during validation")
            return False
        
        for i, example in enumerate(examples):
            # Check required fields
            if 'tokens' not in example or 'labels' not in example:
                self.logger.warning(f"Example {i}: Missing required fields (tokens, labels)")
                return False
            
            tokens = example['tokens']
            labels = example['labels']
            
            # Check tokens and labels length match
            if len(tokens) != len(labels):
                self.logger.warning(f"Example {i}: Token count ({len(tokens)}) != Label count ({len(labels)})")
                return False
            
            # Check label validity
            if self.labels:
                valid_labels = set(['O'] + [f'B-{entity}' for entity in self.labels] + [f'I-{entity}' for entity in self.labels])
                for j, label in enumerate(labels):
                    if label not in valid_labels:
                        self.logger.warning(f"Example {i}, Token {j}: Invalid label '{label}'")
                        return False
            
            # Check BIO consistency
            if not self._validate_bio_sequence(labels):
                self.logger.warning(f"Example {i}: Invalid BIO sequence")
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
        self.logger.info(f"Processing file: {input_file} -> {output_file} (format={format})")
        examples = self.load_data_file(input_file)
        
        # Apply preprocessing
        processed_examples = []
        for example in examples:
            processed_example = self._preprocess_example(example)
            if processed_example:
                processed_examples.append(processed_example)
        
        self.logger.info(f"Processed {len(processed_examples)} examples (from {len(examples)})")
        # Save in specified format
        self._save_examples(processed_examples, output_file, format)
    
    def _preprocess_example(self, example: Dict[str, Any]) -> Optional[Dict[str, Any]]:
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
    
    def _save_examples(self, examples: List[Dict[str, Any]], output_file: str, format: str = 'json'):
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
            self.logger.info(f"Saved JSON examples to {output_file}")
        
        elif format == 'conll':
            with open(output_file, 'w', encoding=self.encoding) as f:
                for example in examples:
                    tokens = example['tokens']
                    labels = example['labels']
                    
                    for token, label in zip(tokens, labels):
                        f.write(f"{token}\t{label}\n")
                    f.write("\n")  # Empty line between sentences
            self.logger.info(f"Saved CoNLL examples to {output_file}")
        
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
            self.logger.info(f"Saved CSV examples to {output_file}")
        
        else:
            raise ValueError(f"Unsupported output format: {format}")
    
