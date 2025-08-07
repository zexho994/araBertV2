"""DAPT Data Processor

Data processing pipeline for DAPT training:
- Data loading and validation
- Text preprocessing and tokenization
- Format conversion and standardization
- Country-specific data handling
"""

import os
import json
import csv
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
from datasets import Dataset, DatasetDict
import logging
from sklearn.model_selection import train_test_split
import re

class DataProcessor:
    """Main data processor for DAPT training"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
        self.country_code = config["country"]["code"]
        
        # Data paths
        self.data_dir = Path(global_config["data_dir"]) / self.country_code
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Configuration
        self.data_config = config["data"]
        # Note: DAPT is unsupervised, no label config needed
        
        # Logging
        self.logger = self._setup_logging()
        
        # Data storage
        self.raw_data = None
        self.processed_data = None
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        self.logger.info(f"Data Processor initialized for country: {self.country_code}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for data processor"""
        logger = logging.getLogger(f"data/dapt_{self.country_code}")
        logger.setLevel(getattr(logging, self.config["logging"]["level"]))
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # File handler
        log_file = self.config["logging"].get("log_file")
        if log_file:
            log_path = Path(self.global_config["log_dir"]) / log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def load_data(self, data_path: Optional[str] = None) -> bool:
        """Load raw data from file"""
        try:
            if data_path is None:
                data_path = self.data_config.get("train_file")
            
            if not data_path:
                raise ValueError("No data path specified")
            
            data_file = Path(data_path)
            if not data_file.exists():
                # Try relative to data directory
                data_file = self.data_dir / data_path
            
            if not data_file.exists():
                raise FileNotFoundError(f"Data file not found: {data_path}")
            
            self.logger.info(f"Loading data from: {data_file}")
            
            # Load based on file extension
            if data_file.suffix.lower() == '.csv':
                self.raw_data = pd.read_csv(data_file, encoding='utf-8')
            elif data_file.suffix.lower() == '.json':
                with open(data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.raw_data = pd.DataFrame(data)
            elif data_file.suffix.lower() == '.jsonl':
                data = []
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line.strip()))
                self.raw_data = pd.DataFrame(data)
            elif data_file.suffix.lower() == '.tsv':
                self.raw_data = pd.read_csv(data_file, sep='\t', encoding='utf-8')
            elif data_file.suffix.lower() == '.txt':
                # For DAPT training: plain text file, one text per line
                texts = []
                with open(data_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:  # Skip empty lines
                            texts.append(line)
                
                # Create DataFrame with text column
                text_column = self.data_config.get("text_column", "text")
                self.raw_data = pd.DataFrame({text_column: texts})
            else:
                raise ValueError(f"Unsupported file format: {data_file.suffix}")
            
            self.logger.info(f"Loaded {len(self.raw_data)} records")
            self.logger.info(f"Data columns: {list(self.raw_data.columns)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False
    
    def validate_data(self) -> Dict[str, Any]:
        """Validate loaded data"""
        if self.raw_data is None:
            return {"valid": False, "error": "No data loaded"}
        
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {}
        }
        
        try:
            # Check required columns
            text_column = self.data_config["text_column"]
            label_column = self.data_config.get("label_column")  # Optional for DAPT
            
            # For DAPT training, only text column is required
            required_columns = [text_column]
            missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
            
            if missing_columns:
                validation_results["errors"].append(f"Missing required columns: {missing_columns}")
                validation_results["valid"] = False
            
            # Check for empty data
            if len(self.raw_data) == 0:
                validation_results["errors"].append("Dataset is empty")
                validation_results["valid"] = False
                return validation_results
            
            # Check for null values in required columns
            null_counts = self.raw_data[required_columns].isnull().sum()
            for col, count in null_counts.items():
                if count > 0:
                    validation_results["warnings"].append(f"Column '{col}' has {count} null values")
            
            # Check label column if it exists
            if label_column and label_column in self.raw_data.columns:
                label_nulls = self.raw_data[label_column].isnull().sum()
                if label_nulls > 0:
                    validation_results["warnings"].append(f"Label column '{label_column}' has {label_nulls} null values")
            
            # Validate text data
            if text_column in self.raw_data.columns:
                text_stats = self._validate_text_column(self.raw_data[text_column])
                validation_results["statistics"]["text"] = text_stats
            
            # Validate labels
            if label_column in self.raw_data.columns:
                label_stats = self._validate_label_column(self.raw_data[label_column])
                validation_results["statistics"]["labels"] = label_stats
            
            # Check data distribution
            validation_results["statistics"]["total_records"] = len(self.raw_data)
            validation_results["statistics"]["columns"] = list(self.raw_data.columns)
            
            self.logger.info(f"Data validation completed. Valid: {validation_results['valid']}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error during data validation: {str(e)}")
            return {"valid": False, "error": str(e)}
    
    def _validate_text_column(self, text_series: pd.Series) -> Dict[str, Any]:
        """Validate text column"""
        stats = {
            "total_texts": len(text_series),
            "empty_texts": text_series.isnull().sum() + (text_series == "").sum(),
            "avg_length": 0,
            "min_length": 0,
            "max_length": 0,
            "language_distribution": {}
        }
        
        # Filter out null/empty texts
        valid_texts = text_series.dropna()
        valid_texts = valid_texts[valid_texts != ""]
        
        if len(valid_texts) > 0:
            lengths = valid_texts.str.len()
            stats["avg_length"] = float(lengths.mean())
            stats["min_length"] = int(lengths.min())
            stats["max_length"] = int(lengths.max())
            
            # Basic language detection (simplified)
            arabic_pattern = re.compile(r'[\u0600-\u06FF]')
            english_pattern = re.compile(r'[a-zA-Z]')
            
            arabic_count = sum(1 for text in valid_texts if arabic_pattern.search(str(text)))
            english_count = sum(1 for text in valid_texts if english_pattern.search(str(text)))
            
            stats["language_distribution"] = {
                "arabic": arabic_count,
                "english": english_count,
                "other": len(valid_texts) - arabic_count - english_count
            }
        
        return stats
    
    def _validate_label_column(self, label_series: pd.Series) -> Dict[str, Any]:
        """Validate label column"""
        stats = {
            "total_labels": len(label_series),
            "null_labels": label_series.isnull().sum(),
            "unique_labels": [],
            "label_distribution": {},
            "format_type": "unknown"
        }
        
        # Filter out null labels
        valid_labels = label_series.dropna()
        
        if len(valid_labels) > 0:
            # Determine label format
            first_label = valid_labels.iloc[0]
            
            if isinstance(first_label, str):
                try:
                    # Try to parse as JSON (for token-level labels)
                    parsed = json.loads(first_label)
                    if isinstance(parsed, list):
                        stats["format_type"] = "token_level_json"
                        # Flatten all labels to get unique label types
                        all_labels = []
                        for label_str in valid_labels:
                            try:
                                labels = json.loads(label_str)
                                all_labels.extend(labels)
                            except:
                                continue
                        
                        unique_labels = list(set(all_labels))
                        stats["unique_labels"] = unique_labels
                        stats["label_distribution"] = {label: all_labels.count(label) for label in unique_labels}
                    else:
                        stats["format_type"] = "single_label_json"
                except:
                    stats["format_type"] = "string"
                    unique_labels = valid_labels.unique().tolist()
                    stats["unique_labels"] = unique_labels
                    stats["label_distribution"] = valid_labels.value_counts().to_dict()
            
            elif isinstance(first_label, list):
                stats["format_type"] = "token_level_list"
                # Flatten all labels
                all_labels = []
                for label_list in valid_labels:
                    if isinstance(label_list, list):
                        all_labels.extend(label_list)
                
                unique_labels = list(set(all_labels))
                stats["unique_labels"] = unique_labels
                stats["label_distribution"] = {label: all_labels.count(label) for label in unique_labels}
            
            else:
                stats["format_type"] = "numeric"
                unique_labels = valid_labels.unique().tolist()
                stats["unique_labels"] = unique_labels
                stats["label_distribution"] = valid_labels.value_counts().to_dict()
        
        return stats
    
    def preprocess_data(self) -> bool:
        """Preprocess the loaded data"""
        if self.raw_data is None:
            self.logger.error("No data loaded for preprocessing")
            return False
        
        try:
            self.logger.info("Starting data preprocessing")
            
            # Create a copy for processing
            self.processed_data = self.raw_data.copy()
            
            # Clean text data
            text_column = self.data_config["text_column"]
            if text_column in self.processed_data.columns:
                self.processed_data = self._clean_text_data(self.processed_data, text_column)
            
            # Process labels (optional for DAPT)
            label_column = self.data_config.get("label_column")
            if label_column and label_column in self.processed_data.columns:
                self.processed_data = self._process_labels(self.processed_data, label_column)
            
            # Remove invalid records
            initial_count = len(self.processed_data)
            self.processed_data = self._remove_invalid_records(self.processed_data)
            final_count = len(self.processed_data)
            
            if initial_count != final_count:
                self.logger.info(f"Removed {initial_count - final_count} invalid records")
            
            # Apply country-specific preprocessing
            self.processed_data = self._apply_country_specific_preprocessing(self.processed_data)
            
            self.logger.info(f"Preprocessing completed. Final dataset size: {len(self.processed_data)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error during preprocessing: {str(e)}")
            return False
    
    def _clean_text_data(self, data: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Clean text data"""
        self.logger.info("Cleaning text data")
        
        # Remove null/empty texts
        data = data.dropna(subset=[text_column])
        data = data[data[text_column] != ""]
        
        # Basic text cleaning
        if self.data_config.get("clean_text", True):
            # Remove extra whitespace
            data[text_column] = data[text_column].str.strip()
            data[text_column] = data[text_column].str.replace(r'\s+', ' ', regex=True)
            
            # Remove control characters
            data[text_column] = data[text_column].str.replace(r'[\x00-\x1f\x7f-\x9f]', '', regex=True)
            
            # Country-specific cleaning
            if self.country_code in ['ae', 'sa', 'eg', 'ma', 'lb', 'jo']:  # Arabic countries
                # Normalize Arabic text
                data[text_column] = data[text_column].apply(self._normalize_arabic_text)
        
        return data
    
    def _normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text"""
        if not isinstance(text, str):
            return text
        
        # Normalize Arabic characters
        text = re.sub(r'[إأآا]', 'ا', text)  # Normalize Alef
        text = re.sub(r'ى', 'ي', text)  # Normalize Yeh
        text = re.sub(r'ة', 'ه', text)  # Normalize Teh Marbuta
        
        # Remove diacritics
        text = re.sub(r'[\u064B-\u0652]', '', text)
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _process_labels(self, data: pd.DataFrame, label_column: str) -> pd.DataFrame:
        """Process labels - for DAPT training, labels are not needed"""
        self.logger.info("DAPT training does not require label processing")
        
        # For DAPT (unsupervised training), we can remove the label column
        # or keep it as-is without processing
        if label_column in data.columns:
            # Keep the column but don't process it
            self.logger.info(f"Label column '{label_column}' present but not processed for DAPT")
        
        return data
    
    def _remove_invalid_records(self, data: pd.DataFrame) -> pd.DataFrame:
        """Remove invalid records"""
        text_column = self.data_config["text_column"]
        # label_column not needed for DAPT training
        
        # Remove records with empty text
        data = data[data[text_column].str.len() > 0]
        
        # Remove records with text that's too short or too long
        min_length = self.data_config.get("min_text_length", 5)
        max_length = self.data_config.get("max_text_length", 512)
        
        data = data[data[text_column].str.len() >= min_length]
        data = data[data[text_column].str.len() <= max_length]
        
        return data
    
    def _apply_country_specific_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply country-specific preprocessing"""
        country_config = self.config["country"]
        
        # Apply country-specific filters or transformations
        if "preprocessing_rules" in country_config:
            rules = country_config["preprocessing_rules"]
            
            for rule in rules:
                if rule["type"] == "filter":
                    # Apply filter
                    column = rule["column"]
                    condition = rule["condition"]
                    value = rule["value"]
                    
                    if condition == "contains":
                        data = data[data[column].str.contains(value, na=False)]
                    elif condition == "not_contains":
                        data = data[~data[column].str.contains(value, na=False)]
                    elif condition == "equals":
                        data = data[data[column] == value]
                    elif condition == "not_equals":
                        data = data[data[column] != value]
        
        return data
    
    def split_data(self, test_size: float = 0.2, val_size: float = 0.1, random_state: int = 42) -> bool:
        """Split data into train/validation/test sets"""
        if self.processed_data is None:
            self.logger.error("No processed data available for splitting")
            return False
        
        try:
            self.logger.info(f"Splitting data: train={1-test_size-val_size:.1f}, val={val_size:.1f}, test={test_size:.1f}")
            
            # First split: separate test set
            train_val_data, test_data = train_test_split(
                self.processed_data,
                test_size=test_size,
                random_state=random_state,
                stratify=None  # Can add stratification based on labels if needed
            )
            
            # Second split: separate validation from training
            if val_size > 0:
                val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
                train_data, val_data = train_test_split(
                    train_val_data,
                    test_size=val_size_adjusted,
                    random_state=random_state
                )
            else:
                train_data = train_val_data
                val_data = None
            
            # Convert to datasets
            self.train_dataset = Dataset.from_pandas(train_data.reset_index(drop=True))
            self.test_dataset = Dataset.from_pandas(test_data.reset_index(drop=True))
            
            if val_data is not None:
                self.val_dataset = Dataset.from_pandas(val_data.reset_index(drop=True))
            
            self.logger.info(f"Data split completed:")
            self.logger.info(f"  Train: {len(self.train_dataset)} samples")
            if self.val_dataset:
                self.logger.info(f"  Validation: {len(self.val_dataset)} samples")
            self.logger.info(f"  Test: {len(self.test_dataset)} samples")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error splitting data: {str(e)}")
            return False
    
    def load_datasets(self) -> Tuple[Optional[Dataset], Optional[Dataset]]:
        """Load and return train and validation datasets"""
        # Load data if not already loaded
        if self.raw_data is None:
            if not self.load_data():
                return None, None
        
        # Validate data
        validation_result = self.validate_data()
        if not validation_result["valid"]:
            self.logger.error(f"Data validation failed: {validation_result.get('errors', [])}")
            return None, None
        
        # Preprocess data
        if not self.preprocess_data():
            return None, None
        
        # Split data
        val_size = self.data_config.get("validation_split", 0.1)
        test_size = self.data_config.get("test_split", 0.2)
        
        if not self.split_data(test_size=test_size, val_size=val_size):
            return None, None
        
        return self.train_dataset, self.val_dataset
    
    def save_processed_data(self, output_dir: str) -> bool:
        """Save processed datasets"""
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save datasets
            if self.train_dataset:
                train_path = output_path / "train_dataset"
                self.train_dataset.save_to_disk(str(train_path))
                self.logger.info(f"Saved training dataset to: {train_path}")
            
            if self.val_dataset:
                val_path = output_path / "val_dataset"
                self.val_dataset.save_to_disk(str(val_path))
                self.logger.info(f"Saved validation dataset to: {val_path}")
            
            if self.test_dataset:
                test_path = output_path / "test_dataset"
                self.test_dataset.save_to_disk(str(test_path))
                self.logger.info(f"Saved test dataset to: {test_path}")
            
            # Save processing metadata
            metadata = {
                "country_code": self.country_code,
                "processing_date": pd.Timestamp.now().isoformat(),
                "original_size": len(self.raw_data) if self.raw_data is not None else 0,
                "processed_size": len(self.processed_data) if self.processed_data is not None else 0,
                "train_size": len(self.train_dataset) if self.train_dataset else 0,
                "val_size": len(self.val_dataset) if self.val_dataset else 0,
                "test_size": len(self.test_dataset) if self.test_dataset else 0,
                "config": self.config
            }
            
            metadata_path = output_path / "processing_metadata.json"
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")
            return False
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get comprehensive data statistics"""
        stats = {
            "country_code": self.country_code,
            "raw_data": None,
            "processed_data": None,
            "datasets": {}
        }
        
        if self.raw_data is not None:
            stats["raw_data"] = {
                "size": len(self.raw_data),
                "columns": list(self.raw_data.columns),
                "memory_usage": self.raw_data.memory_usage(deep=True).sum()
            }
        
        if self.processed_data is not None:
            stats["processed_data"] = {
                "size": len(self.processed_data),
                "columns": list(self.processed_data.columns)
            }
        
        if self.train_dataset:
            stats["datasets"]["train"] = len(self.train_dataset)
        if self.val_dataset:
            stats["datasets"]["validation"] = len(self.val_dataset)
        if self.test_dataset:
            stats["datasets"]["test"] = len(self.test_dataset)
        
        return stats