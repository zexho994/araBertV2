"""Text Processing Utilities for NER System

Provides text processing, cleaning, and manipulation utilities.
"""

import re
import string
import unicodedata
from typing import List, Dict, Tuple, Optional, Set, Any
from collections import Counter
import numpy as np

class TextUtils:
    """Text processing utilities"""
    
    # Common patterns
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    URL_PATTERN = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    PHONE_PATTERN = re.compile(r'\b(?:\+?1[-.]?)?\(?([0-9]{3})\)?[-.]?([0-9]{3})[-.]?([0-9]{4})\b')
    NUMBER_PATTERN = re.compile(r'\b\d+(?:\.\d+)?\b')
    
    # Arabic text patterns
    ARABIC_PATTERN = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]+')
    ARABIC_DIACRITICS = re.compile(r'[\u064B-\u0652\u0670\u0640]')
    
    @staticmethod
    def clean_text(
        text: str, 
        remove_extra_whitespace: bool = True,
        remove_punctuation: bool = False,
        remove_numbers: bool = False,
        remove_urls: bool = True,
        remove_emails: bool = True,
        lowercase: bool = False,
        remove_arabic_diacritics: bool = True
    ) -> str:
        """Clean text with various options
        
        Args:
            text: Input text
            remove_extra_whitespace: Remove extra whitespace
            remove_punctuation: Remove punctuation
            remove_numbers: Remove numbers
            remove_urls: Remove URLs
            remove_emails: Remove email addresses
            lowercase: Convert to lowercase
            remove_arabic_diacritics: Remove Arabic diacritics
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove URLs
        if remove_urls:
            text = TextUtils.URL_PATTERN.sub('', text)
        
        # Remove emails
        if remove_emails:
            text = TextUtils.EMAIL_PATTERN.sub('', text)
        
        # Remove numbers
        if remove_numbers:
            text = TextUtils.NUMBER_PATTERN.sub('', text)
        
        # Remove Arabic diacritics
        if remove_arabic_diacritics:
            text = TextUtils.ARABIC_DIACRITICS.sub('', text)
        
        # Remove punctuation
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        if remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def normalize_unicode(text: str, form: str = 'NFKC') -> str:
        """Normalize Unicode text
        
        Args:
            text: Input text
            form: Normalization form (NFC, NFKC, NFD, NFKD)
            
        Returns:
            Normalized text
        """
        return unicodedata.normalize(form, text)
    
    @staticmethod
    def tokenize_simple(text: str, split_on_whitespace: bool = True) -> List[str]:
        """Simple tokenization
        
        Args:
            text: Input text
            split_on_whitespace: Whether to split on whitespace
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        if split_on_whitespace:
            return text.split()
        else:
            # Split on punctuation and whitespace
            tokens = re.findall(r'\b\w+\b', text)
            return tokens
    
    @staticmethod
    def extract_entities_from_bio(
        tokens: List[str], 
        labels: List[str]
    ) -> List[Dict[str, Any]]:
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
    
    @staticmethod
    def convert_entities_to_bio(
        tokens: List[str], 
        entities: List[Dict[str, Any]]
    ) -> List[str]:
        """Convert entities to BIO labels
        
        Args:
            tokens: List of tokens
            entities: List of entities with start, end, and type
            
        Returns:
            List of BIO labels
        """
        labels = ['O'] * len(tokens)
        
        for entity in entities:
            start = entity['start']
            end = entity['end']
            entity_type = entity['type']
            
            if start < len(labels):
                labels[start] = f'B-{entity_type}'
                
                for i in range(start + 1, min(end, len(labels))):
                    labels[i] = f'I-{entity_type}'
        
        return labels
    
    @staticmethod
    def validate_bio_sequence(labels: List[str]) -> Tuple[bool, List[str]]:
        """Validate BIO label sequence
        
        Args:
            labels: List of BIO labels
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        for i, label in enumerate(labels):
            if label.startswith('I-'):
                entity_type = label[2:]
                
                # Check if previous label is compatible
                if i == 0:
                    errors.append(f"Position {i}: I-{entity_type} at beginning of sequence")
                else:
                    prev_label = labels[i-1]
                    if prev_label == 'O':
                        errors.append(f"Position {i}: I-{entity_type} after O")
                    elif prev_label.startswith('B-') or prev_label.startswith('I-'):
                        prev_entity_type = prev_label[2:]
                        if prev_entity_type != entity_type:
                            errors.append(
                                f"Position {i}: I-{entity_type} after {prev_label} "
                                f"(entity type mismatch)"
                            )
        
        return len(errors) == 0, errors
    
    @staticmethod
    def fix_bio_sequence(labels: List[str]) -> List[str]:
        """Fix BIO label sequence
        
        Args:
            labels: List of BIO labels
            
        Returns:
            Fixed BIO labels
        """
        fixed_labels = labels.copy()
        
        for i, label in enumerate(fixed_labels):
            if label.startswith('I-'):
                entity_type = label[2:]
                
                # Check if previous label is compatible
                if i == 0 or fixed_labels[i-1] == 'O':
                    # Convert I- to B- at beginning or after O
                    fixed_labels[i] = f'B-{entity_type}'
                elif fixed_labels[i-1].startswith(('B-', 'I-')):
                    prev_entity_type = fixed_labels[i-1][2:]
                    if prev_entity_type != entity_type:
                        # Convert I- to B- when entity type changes
                        fixed_labels[i] = f'B-{entity_type}'
        
        return fixed_labels
    
    @staticmethod
    def get_label_statistics(labels: List[List[str]]) -> Dict[str, Any]:
        """Get statistics about labels
        
        Args:
            labels: List of label sequences
            
        Returns:
            Label statistics
        """
        # Flatten labels
        flat_labels = [label for seq in labels for label in seq]
        
        # Count labels
        label_counts = Counter(flat_labels)
        
        # Extract entity types
        entity_types = set()
        for label in flat_labels:
            if label != 'O' and '-' in label:
                entity_type = label.split('-', 1)[1]
                entity_types.add(entity_type)
        
        # Count entities
        entity_counts = Counter()
        for seq in labels:
            entities = TextUtils.extract_entities_from_bio([''] * len(seq), seq)
            for entity in entities:
                entity_counts[entity['type']] += 1
        
        return {
            'total_tokens': len(flat_labels),
            'total_sequences': len(labels),
            'unique_labels': len(label_counts),
            'label_counts': dict(label_counts),
            'entity_types': sorted(list(entity_types)),
            'entity_counts': dict(entity_counts),
            'o_ratio': label_counts.get('O', 0) / len(flat_labels) if flat_labels else 0
        }
    
    @staticmethod
    def align_tokens_and_labels(
        tokens: List[str], 
        labels: List[str],
        tokenizer,
        max_length: int = 512
    ) -> Tuple[List[str], List[str], List[int]]:
        """Align tokens and labels with tokenizer
        
        Args:
            tokens: Original tokens
            labels: Original labels
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            
        Returns:
            Tuple of (aligned_tokens, aligned_labels, word_ids)
        """
        # Tokenize with word IDs
        tokenized = tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None
        )
        
        # Get word IDs
        word_ids = tokenized.word_ids()
        
        # Align labels
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # Special token
                aligned_labels.append('O')
            elif word_idx != previous_word_idx:
                # First subtoken of word
                if word_idx < len(labels):
                    aligned_labels.append(labels[word_idx])
                else:
                    aligned_labels.append('O')
            else:
                # Subsequent subtoken of word
                if word_idx < len(labels):
                    label = labels[word_idx]
                    if label.startswith('B-'):
                        # Convert B- to I- for subsequent subtokens
                        aligned_labels.append('I-' + label[2:])
                    else:
                        aligned_labels.append(label)
                else:
                    aligned_labels.append('O')
            
            previous_word_idx = word_idx
        
        # Get aligned tokens
        aligned_tokens = tokenizer.convert_ids_to_tokens(tokenized['input_ids'])
        
        return aligned_tokens, aligned_labels, word_ids
    
    @staticmethod
    def calculate_text_similarity(text1: str, text2: str, method: str = 'jaccard') -> float:
        """Calculate similarity between two texts
        
        Args:
            text1: First text
            text2: Second text
            method: Similarity method (jaccard, cosine)
            
        Returns:
            Similarity score
        """
        if method == 'jaccard':
            tokens1 = set(TextUtils.tokenize_simple(text1.lower()))
            tokens2 = set(TextUtils.tokenize_simple(text2.lower()))
            
            intersection = len(tokens1 & tokens2)
            union = len(tokens1 | tokens2)
            
            return intersection / union if union > 0 else 0.0
        
        elif method == 'cosine':
            tokens1 = TextUtils.tokenize_simple(text1.lower())
            tokens2 = TextUtils.tokenize_simple(text2.lower())
            
            # Create vocabulary
            vocab = set(tokens1 + tokens2)
            
            # Create vectors
            vec1 = np.array([tokens1.count(token) for token in vocab])
            vec2 = np.array([tokens2.count(token) for token in vocab])
            
            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        else:
            raise ValueError(f"Unknown similarity method: {method}")
    
    @staticmethod
    def extract_ngrams(tokens: List[str], n: int = 2) -> List[str]:
        """Extract n-grams from tokens
        
        Args:
            tokens: List of tokens
            n: N-gram size
            
        Returns:
            List of n-grams
        """
        if len(tokens) < n:
            return []
        
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        
        return ngrams
    
    @staticmethod
    def detect_language(text: str) -> str:
        """Simple language detection
        
        Args:
            text: Input text
            
        Returns:
            Detected language code
        """
        # Count Arabic characters
        arabic_chars = len(TextUtils.ARABIC_PATTERN.findall(text))
        total_chars = len(re.findall(r'\w', text))
        
        if total_chars == 0:
            return 'unknown'
        
        arabic_ratio = arabic_chars / total_chars
        
        if arabic_ratio > 0.5:
            return 'ar'
        else:
            return 'en'
    
    @staticmethod
    def split_sentences(text: str, language: str = 'en') -> List[str]:
        """Split text into sentences
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            List of sentences
        """
        if language == 'ar':
            # Arabic sentence splitting
            sentences = re.split(r'[.!?؟]', text)
        else:
            # English sentence splitting
            sentences = re.split(r'[.!?]', text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    @staticmethod
    def mask_entities(
        text: str, 
        entities: List[Dict[str, Any]], 
        mask_token: str = '[MASK]'
    ) -> str:
        """Mask entities in text
        
        Args:
            text: Input text
            entities: List of entities with start and end positions
            mask_token: Token to use for masking
            
        Returns:
            Text with masked entities
        """
        # Sort entities by start position (descending) to avoid position shifts
        sorted_entities = sorted(entities, key=lambda x: x['start'], reverse=True)
        
        masked_text = text
        for entity in sorted_entities:
            start = entity['start']
            end = entity['end']
            masked_text = masked_text[:start] + mask_token + masked_text[end:]
        
        return masked_text
    
    @staticmethod
    def anonymize_text(
        text: str, 
        remove_emails: bool = True,
        remove_phones: bool = True,
        remove_urls: bool = True
    ) -> str:
        """Anonymize sensitive information in text
        
        Args:
            text: Input text
            remove_emails: Whether to remove email addresses
            remove_phones: Whether to remove phone numbers
            remove_urls: Whether to remove URLs
            
        Returns:
            Anonymized text
        """
        anonymized = text
        
        if remove_emails:
            anonymized = TextUtils.EMAIL_PATTERN.sub('[EMAIL]', anonymized)
        
        if remove_phones:
            anonymized = TextUtils.PHONE_PATTERN.sub('[PHONE]', anonymized)
        
        if remove_urls:
            anonymized = TextUtils.URL_PATTERN.sub('[URL]', anonymized)
        
        return anonymized