"""DAPT Data Preprocessor

Data preprocessing utilities for DAPT training:
- Arabic text normalization and cleaning
- Label standardization and validation
- Country-specific preprocessing rules
- Text tokenization and encoding
- Data augmentation techniques
"""

import re
import string
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from pathlib import Path
import logging
from abc import ABC, abstractmethod
import unicodedata
from collections import Counter
import random
from datetime import datetime

# Arabic text processing
try:
    import pyarabic.araby as araby
    import pyarabic.normalize as normalize
    ARABIC_SUPPORT = True
except ImportError:
    ARABIC_SUPPORT = False
    print("Warning: pyarabic not installed. Arabic text processing will be limited.")

class BasePreprocessor(ABC):
    """Abstract base class for preprocessors"""
    
    @abstractmethod
    def process(self, text: str) -> str:
        """Process text"""
        pass

class ArabicTextPreprocessor(BasePreprocessor):
    """Arabic text preprocessing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Arabic characters and patterns
        self.arabic_pattern = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+')
        self.diacritics_pattern = re.compile(r'[\u064B-\u0652\u0670\u0640]')
        self.tatweel_pattern = re.compile(r'\u0640+')
        
        # Normalization mappings
        self.char_mappings = {
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا',  # Alef variations
            'ة': 'ه',  # Teh marbuta to heh
            'ى': 'ي',  # Alef maksura to yeh
        }
        
        # Country-specific rules
        self.country_rules = self._load_country_rules()
    
    def _load_country_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load country-specific preprocessing rules"""
        return {
            'UAE': {
                'dialect_mappings': {
                    'شلون': 'كيف',
                    'وين': 'أين',
                    'شنو': 'ماذا'
                },
                'common_terms': ['الإمارات', 'دبي', 'أبوظبي']
            },
            'SA': {
                'dialect_mappings': {
                    'ايش': 'ماذا',
                    'وش': 'ماذا',
                    'كيفك': 'كيف حالك'
                },
                'common_terms': ['السعودية', 'الرياض', 'جدة']
            },
            'EG': {
                'dialect_mappings': {
                    'ايه': 'ماذا',
                    'فين': 'أين',
                    'ازيك': 'كيف حالك'
                },
                'common_terms': ['مصر', 'القاهرة', 'الإسكندرية']
            },
            'MA': {
                'dialect_mappings': {
                    'أشنو': 'ماذا',
                    'فين': 'أين',
                    'كيفاش': 'كيف'
                },
                'common_terms': ['المغرب', 'الرباط', 'الدار البيضاء']
            },
            'LB': {
                'dialect_mappings': {
                    'شو': 'ماذا',
                    'وين': 'أين',
                    'كيفك': 'كيف حالك'
                },
                'common_terms': ['لبنان', 'بيروت', 'طرابلس']
            },
            'JO': {
                'dialect_mappings': {
                    'شو': 'ماذا',
                    'وين': 'أين',
                    'كيفك': 'كيف حالك'
                },
                'common_terms': ['الأردن', 'عمان', 'إربد']
            }
        }
    
    def process(self, text: str) -> str:
        """Process Arabic text"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        # Basic cleaning
        text = self._basic_clean(text)
        
        # Arabic-specific processing
        if ARABIC_SUPPORT:
            text = self._advanced_arabic_clean(text)
        else:
            text = self._basic_arabic_clean(text)
        
        # Apply country-specific rules
        text = self._apply_country_rules(text)
        
        return text.strip()
    
    def _basic_clean(self, text: str) -> str:
        """Basic text cleaning"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if unicodedata.category(char)[0] != 'C')
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove phone numbers (basic pattern)
        text = re.sub(r'\+?\d[\d\s\-\(\)]{7,}\d', '', text)
        
        return text
    
    def _basic_arabic_clean(self, text: str) -> str:
        """Basic Arabic text cleaning without pyarabic"""
        # Remove diacritics
        text = self.diacritics_pattern.sub('', text)
        
        # Remove tatweel (kashida)
        text = self.tatweel_pattern.sub('', text)
        
        # Normalize common characters
        for old_char, new_char in self.char_mappings.items():
            text = text.replace(old_char, new_char)
        
        return text
    
    def _advanced_arabic_clean(self, text: str) -> str:
        """Advanced Arabic text cleaning with pyarabic"""
        # Remove diacritics
        text = araby.strip_diacritics(text)
        
        # Remove tatweel
        text = araby.strip_tatweel(text)
        
        # Normalize text
        text = normalize.normalize_alef(text)
        text = normalize.normalize_teh(text)
        text = normalize.normalize_yeh(text)
        
        return text
    
    def _apply_country_rules(self, text: str) -> str:
        """Apply country-specific preprocessing rules"""
        country_code = self.config.get('country', {}).get('code', '')
        
        if country_code in self.country_rules:
            rules = self.country_rules[country_code]
            
            # Apply dialect mappings
            for dialect_term, standard_term in rules.get('dialect_mappings', {}).items():
                text = text.replace(dialect_term, standard_term)
        
        return text

class LabelPreprocessor(BasePreprocessor):
    """Label preprocessing and standardization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.label_mappings = self._load_label_mappings()
        self.valid_labels = set(config.get('labels', {}).get('classes', []))
    
    def _load_label_mappings(self) -> Dict[str, str]:
        """Load label mappings for standardization"""
        return {
            # Common variations
            'PERSON': 'PER',
            'ORGANIZATION': 'ORG',
            'LOCATION': 'LOC',
            'MISCELLANEOUS': 'MISC',
            
            # Case variations
            'per': 'PER',
            'org': 'ORG',
            'loc': 'LOC',
            'misc': 'MISC',
            
            # BIO format
            'B-PERSON': 'B-PER',
            'I-PERSON': 'I-PER',
            'B-ORGANIZATION': 'B-ORG',
            'I-ORGANIZATION': 'I-ORG',
            'B-LOCATION': 'B-LOC',
            'I-LOCATION': 'I-LOC',
        }
    
    def process(self, label: str) -> str:
        """Process and standardize label"""
        if not isinstance(label, str):
            return str(label)
        
        label = label.strip()
        
        # Apply mappings
        if label in self.label_mappings:
            label = self.label_mappings[label]
        
        return label
    
    def validate_label(self, label: str) -> bool:
        """Validate if label is in valid set"""
        processed_label = self.process(label)
        return processed_label in self.valid_labels or processed_label == 'O'

class DataPreprocessor:
    """Main data preprocessor for DAPT training"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
        self.country_code = config["country"]["code"]
        
        # Initialize preprocessors
        self.text_preprocessor = ArabicTextPreprocessor(config)
        self.label_preprocessor = LabelPreprocessor(config)
        
        # Configuration
        self.text_column = config["data"]["text_column"]
        self.label_column = config["data"]["label_column"]
        self.min_text_length = config["data"].get("min_text_length", 5)
        self.max_text_length = config["data"].get("max_text_length", 512)
        
        # Logging
        self.logger = self._setup_logging()
        
        self.logger.info(f"Data Preprocessor initialized for country: {self.country_code}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for preprocessor"""
        logger = logging.getLogger(f"dapt_preprocessor_{self.country_code}")
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
    
    def preprocess_dataset(self, data: pd.DataFrame, 
                          apply_augmentation: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess entire dataset"""
        try:
            self.logger.info(f"Starting preprocessing of {len(data)} records")
            
            original_count = len(data)
            stats = {
                "original_count": original_count,
                "processed_count": 0,
                "removed_count": 0,
                "text_changes": 0,
                "label_changes": 0,
                "augmented_count": 0
            }
            
            # Create copy for processing
            processed_data = data.copy()
            
            # Preprocess text
            self.logger.info("Preprocessing text...")
            processed_data, text_stats = self._preprocess_text_column(processed_data)
            stats.update(text_stats)
            
            # Preprocess labels
            if self.label_column in processed_data.columns:
                self.logger.info("Preprocessing labels...")
                processed_data, label_stats = self._preprocess_label_column(processed_data)
                stats.update(label_stats)
            
            # Filter invalid records
            self.logger.info("Filtering invalid records...")
            processed_data = self._filter_invalid_records(processed_data)
            
            # Apply data augmentation if requested
            if apply_augmentation:
                self.logger.info("Applying data augmentation...")
                processed_data, aug_count = self._apply_augmentation(processed_data)
                stats["augmented_count"] = aug_count
            
            stats["processed_count"] = len(processed_data)
            stats["removed_count"] = original_count - len(processed_data)
            
            self.logger.info(f"Preprocessing completed. {stats['processed_count']} records remaining")
            
            return processed_data, stats
            
        except Exception as e:
            self.logger.error(f"Error in dataset preprocessing: {str(e)}")
            raise
    
    def _preprocess_text_column(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess text column"""
        stats = {"text_changes": 0}
        
        if self.text_column not in data.columns:
            self.logger.warning(f"Text column '{self.text_column}' not found")
            return data, stats
        
        original_texts = data[self.text_column].copy()
        
        # Apply text preprocessing
        data[self.text_column] = data[self.text_column].apply(
            lambda x: self.text_preprocessor.process(str(x)) if pd.notna(x) else ""
        )
        
        # Count changes
        changes = (original_texts != data[self.text_column]).sum()
        stats["text_changes"] = changes
        
        self.logger.info(f"Text preprocessing: {changes} texts modified")
        
        return data, stats
    
    def _preprocess_label_column(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Preprocess label column"""
        stats = {"label_changes": 0, "invalid_labels": 0}
        
        if self.label_column not in data.columns:
            self.logger.warning(f"Label column '{self.label_column}' not found")
            return data, stats
        
        original_labels = data[self.label_column].copy()
        
        # Apply label preprocessing
        data[self.label_column] = data[self.label_column].apply(
            lambda x: self.label_preprocessor.process(str(x)) if pd.notna(x) else "O"
        )
        
        # Validate labels
        invalid_mask = ~data[self.label_column].apply(
            self.label_preprocessor.validate_label
        )
        
        if invalid_mask.any():
            invalid_count = invalid_mask.sum()
            self.logger.warning(f"Found {invalid_count} invalid labels")
            
            # Set invalid labels to 'O'
            data.loc[invalid_mask, self.label_column] = 'O'
            stats["invalid_labels"] = invalid_count
        
        # Count changes
        changes = (original_labels != data[self.label_column]).sum()
        stats["label_changes"] = changes
        
        self.logger.info(f"Label preprocessing: {changes} labels modified")
        
        return data, stats
    
    def _filter_invalid_records(self, data: pd.DataFrame) -> pd.DataFrame:
        """Filter out invalid records"""
        original_count = len(data)
        
        # Filter by text length
        if self.text_column in data.columns:
            text_lengths = data[self.text_column].str.len()
            valid_length_mask = (
                (text_lengths >= self.min_text_length) & 
                (text_lengths <= self.max_text_length)
            )
            data = data[valid_length_mask]
        
        # Remove empty texts
        if self.text_column in data.columns:
            data = data[data[self.text_column].str.strip() != ""]
        
        # Remove rows with missing required columns
        required_columns = [self.text_column]
        if self.label_column in data.columns:
            required_columns.append(self.label_column)
        
        data = data.dropna(subset=required_columns)
        
        removed_count = original_count - len(data)
        if removed_count > 0:
            self.logger.info(f"Filtered out {removed_count} invalid records")
        
        return data
    
    def _apply_augmentation(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Apply data augmentation techniques"""
        augmentation_config = self.config.get("augmentation", {})
        
        if not augmentation_config.get("enabled", False):
            return data, 0
        
        augmented_records = []
        
        # Synonym replacement (basic implementation)
        if augmentation_config.get("synonym_replacement", False):
            augmented_records.extend(
                self._synonym_replacement_augmentation(data, 
                    augmentation_config.get("synonym_ratio", 0.1))
            )
        
        # Back translation (placeholder - would need translation API)
        if augmentation_config.get("back_translation", False):
            # This would require translation services
            pass
        
        # Random insertion/deletion
        if augmentation_config.get("random_modification", False):
            augmented_records.extend(
                self._random_modification_augmentation(data,
                    augmentation_config.get("modification_ratio", 0.05))
            )
        
        if augmented_records:
            augmented_df = pd.DataFrame(augmented_records)
            data = pd.concat([data, augmented_df], ignore_index=True)
        
        return data, len(augmented_records)
    
    def _synonym_replacement_augmentation(self, data: pd.DataFrame, ratio: float) -> List[Dict[str, Any]]:
        """Basic synonym replacement augmentation"""
        # This is a simplified implementation
        # In practice, you'd use a proper Arabic synonym dictionary
        
        synonyms = {
            'كبير': ['ضخم', 'عظيم', 'هائل'],
            'صغير': ['ضئيل', 'قليل', 'محدود'],
            'جميل': ['رائع', 'حسن', 'بديع'],
            'سيء': ['رديء', 'قبيح', 'مؤذي']
        }
        
        augmented = []
        sample_size = min(int(len(data) * ratio), 100)  # Limit augmentation
        
        for _, row in data.sample(n=sample_size).iterrows():
            text = row[self.text_column]
            
            # Replace synonyms
            for word, synonym_list in synonyms.items():
                if word in text:
                    synonym = random.choice(synonym_list)
                    augmented_text = text.replace(word, synonym, 1)
                    
                    augmented_row = row.copy()
                    augmented_row[self.text_column] = augmented_text
                    augmented.append(augmented_row.to_dict())
                    break
        
        return augmented
    
    def _random_modification_augmentation(self, data: pd.DataFrame, ratio: float) -> List[Dict[str, Any]]:
        """Random text modification augmentation"""
        augmented = []
        sample_size = min(int(len(data) * ratio), 50)  # Limit augmentation
        
        for _, row in data.sample(n=sample_size).iterrows():
            text = row[self.text_column]
            words = text.split()
            
            if len(words) > 3:  # Only modify if enough words
                # Random word removal
                if random.random() < 0.5 and len(words) > 1:
                    remove_idx = random.randint(0, len(words) - 1)
                    modified_words = words[:remove_idx] + words[remove_idx + 1:]
                    modified_text = ' '.join(modified_words)
                else:
                    # Random word duplication
                    dup_idx = random.randint(0, len(words) - 1)
                    modified_words = words[:dup_idx + 1] + [words[dup_idx]] + words[dup_idx + 1:]
                    modified_text = ' '.join(modified_words)
                
                augmented_row = row.copy()
                augmented_row[self.text_column] = modified_text
                augmented.append(augmented_row.to_dict())
        
        return augmented
    
    def get_preprocessing_stats(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        try:
            stats = {
                "total_records": len(data),
                "text_column": self.text_column,
                "label_column": self.label_column
            }
            
            if self.text_column in data.columns:
                text_lengths = data[self.text_column].str.len()
                stats.update({
                    "text_stats": {
                        "min_length": int(text_lengths.min()),
                        "max_length": int(text_lengths.max()),
                        "mean_length": float(text_lengths.mean()),
                        "median_length": float(text_lengths.median())
                    }
                })
                
                # Arabic text ratio
                arabic_ratio = data[self.text_column].apply(
                    lambda x: bool(self.text_preprocessor.arabic_pattern.search(str(x)))
                ).mean()
                stats["arabic_text_ratio"] = float(arabic_ratio)
            
            if self.label_column in data.columns:
                label_counts = data[self.label_column].value_counts().to_dict()
                stats["label_distribution"] = label_counts
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error getting preprocessing stats: {str(e)}")
            return {}
    
    def preprocess_single_text(self, text: str) -> str:
        """Preprocess a single text"""
        return self.text_preprocessor.process(text)
    
    def preprocess_single_label(self, label: str) -> str:
        """Preprocess a single label"""
        return self.label_preprocessor.process(label)
    
    def validate_single_label(self, label: str) -> bool:
        """Validate a single label"""
        return self.label_preprocessor.validate_label(label)