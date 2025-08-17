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

        self.config = config or None
        if self.config is None:
            raise ValueError("NERDataProcessor requires a config instance")

        self.labels = self.config.get('labels', {}).get('label_mapping', [])
        if not self.labels:
            raise ValueError("NERDataProcessor requires labels")

        self.max_length = self.config.get('data', {}).get('max_length', 0)
        if self.max_length <= 0:
            raise ValueError("max_length must be greater than 0")

        self.encoding = self.config.get('data', {}).get('encoding', 'utf-8')
        self.logger.info("Initializing NERDataProcessor")
        self.logger.debug(f"Processor config: max_length={self.max_length}, encoding={self.encoding}")
        
        self._init_label_mappings()
    
    def _init_label_mappings(self):
        """初始化标签到ID的映射关系
        
        基于配置中的实体类型列表，构建完整的BIO标注体系的标签映射：
        - 从 config['labels']['label_mapping'] 读取实体类型列表
        - 自动生成BIO格式标签：'O' + 'B-{entity}' + 'I-{entity}'
        - 创建双向映射：label2id（标签名→索引）和 id2label（索引→标签名）
        - 用于模型训练时的标签编码和解码
        
        生成的标签顺序：
        1. 'O' (Outside，非实体标签)
        2. 'B-{entity1}', 'I-{entity1}' (第一个实体的开始和内部标签)
        3. 'B-{entity2}', 'I-{entity2}' (第二个实体的开始和内部标签)
        4. ...以此类推
        
        Raises:
            ValueError: 当配置中未提供标签时抛出异常
        """
        if self.labels:
            # 创建BIO标签
            bio_labels = ['O']  # Outside
            for entity in self.labels:
                bio_labels.extend([f'B-{entity}', f'I-{entity}'])
            
            # 创建标签到ID的映射, ex. {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4}
            self.label2id = {label: idx for idx, label in enumerate(bio_labels)}
            # 创建ID到标签的映射, ex. {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG'}
            self.id2label = {idx: label for label, idx in self.label2id.items()}
            self.logger.info(f"Initialized label mappings for {len(self.labels)} entities, total BIO labels={len(bio_labels)}")
        else:
            self.label2id = {}
            self.id2label = {}
            raise ValueError("No labels provided in config")
    
    def _load_conll_file(self, file_path: str) -> List[Dict[str, Any]]:
        """加载CoNLL格式数据文件
        
        Args:
            file_path: CoNLL格式数据文件路径
            
        Returns:
            数据示例列表
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
        """加载JSON格式数据文件
        
        Args:
            file_path: JSON格式数据文件路径
            
        Returns:
            数据示例列表
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
        """加载JSON Lines (JSONL)格式数据文件

        Args:
            file_path: JSONL格式数据文件路径

        Returns:
            数据示例列表
        """
        self.logger.info(f"Loading JSONL data from {file_path}")
        dataset: List[Dict[str, Any]] = []
        with open(file_path, 'r', encoding=self.encoding) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    dataset.append(obj)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSONL line in {file_path}: {e}")
        self.logger.info(f"Loaded {len(dataset)} dataset from {file_path}")
        return dataset
    
    def _load_csv_file(self, file_path: str) -> List[Dict[str, Any]]:
        """加载CSV格式数据文件
        
        Args:
            file_path: CSV格式数据文件路径
            
        Returns:
            数据示例列表
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
        """加载数据文件, 自动检测文件格式
        
        Args:
            file_path: 数据文件路径
            
        Returns:
            数据示例列表
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
            raise ValueError(f"Unsupported data file format: {suffix}")
    
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
    
