"""DAPT Data Loader

Data loading utilities for DAPT training:
- Multi-format data loading (CSV, JSON, JSONL, TSV)
- Streaming data loading for large datasets
- Data caching and optimization
- Format conversion utilities
"""

import os
import json
import csv
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Iterator, Generator
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
import logging
from abc import ABC, abstractmethod
import pickle
import hashlib
from datetime import datetime

class BaseDataLoader(ABC):
    """Abstract base class for data loaders"""
    
    @abstractmethod
    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from file"""
        pass
    
    @abstractmethod
    def validate_format(self, file_path: str) -> bool:
        """Validate file format"""
        pass

class CSVDataLoader(BaseDataLoader):
    """CSV data loader"""
    
    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load CSV data"""
        encoding = kwargs.get('encoding', 'utf-8')
        separator = kwargs.get('separator', ',')
        
        return pd.read_csv(
            file_path,
            encoding=encoding,
            sep=separator,
            **{k: v for k, v in kwargs.items() if k not in ['encoding', 'separator']}
        )
    
    def validate_format(self, file_path: str) -> bool:
        """Validate CSV format"""
        try:
            # Try to read first few lines
            pd.read_csv(file_path, nrows=5)
            return True
        except Exception:
            return False

class JSONDataLoader(BaseDataLoader):
    """JSON data loader"""
    
    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load JSON data"""
        encoding = kwargs.get('encoding', 'utf-8')
        
        with open(file_path, 'r', encoding=encoding) as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return pd.DataFrame(data)
        elif isinstance(data, dict):
            # If it's a single record, wrap in list
            return pd.DataFrame([data])
        else:
            raise ValueError("JSON data must be a list or dictionary")
    
    def validate_format(self, file_path: str) -> bool:
        """Validate JSON format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                json.load(f)
            return True
        except Exception:
            return False

class JSONLDataLoader(BaseDataLoader):
    """JSONL (JSON Lines) data loader"""
    
    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load JSONL data"""
        encoding = kwargs.get('encoding', 'utf-8')
        
        data = []
        with open(file_path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
        
        return pd.DataFrame(data)
    
    def validate_format(self, file_path: str) -> bool:
        """Validate JSONL format"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= 5:  # Check first 5 lines
                        break
                    line = line.strip()
                    if line:
                        json.loads(line)
            return True
        except Exception:
            return False

class TSVDataLoader(BaseDataLoader):
    """TSV data loader"""
    
    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load TSV data"""
        encoding = kwargs.get('encoding', 'utf-8')
        
        return pd.read_csv(
            file_path,
            encoding=encoding,
            sep='\t',
            **{k: v for k, v in kwargs.items() if k != 'encoding'}
        )
    
    def validate_format(self, file_path: str) -> bool:
        """Validate TSV format"""
        try:
            pd.read_csv(file_path, sep='\t', nrows=5)
            return True
        except Exception:
            return False

class DataLoader:
    """Main data loader for DAPT training"""
    
    def __init__(self, config: Dict[str, Any], global_config: Dict[str, Any]):
        self.config = config
        self.global_config = global_config
        self.country_code = config["country"]["code"]
        
        # Data paths
        self.data_dir = Path(global_config["data_dir"]) / self.country_code
        self.cache_dir = Path(global_config["data_dir"]) / "cache" / self.country_code
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Loaders
        self.loaders = {
            '.csv': CSVDataLoader(),
            '.json': JSONDataLoader(),
            '.jsonl': JSONLDataLoader(),
            '.tsv': TSVDataLoader()
        }
        
        # Configuration
        self.use_cache = global_config.get("use_data_cache", True)
        self.cache_ttl_hours = global_config.get("cache_ttl_hours", 24)
        
        # Logging
        self.logger = self._setup_logging()
        
        self.logger.info(f"Data Loader initialized for country: {self.country_code}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for data loader"""
        logger = logging.getLogger(f"dapt_loader_{self.country_code}")
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
    
    def load_data(self, file_path: str, **kwargs) -> Optional[pd.DataFrame]:
        """Load data from file with caching support"""
        try:
            file_path = Path(file_path)
            
            # Try relative to data directory if not absolute
            if not file_path.is_absolute():
                file_path = self.data_dir / file_path
            
            if not file_path.exists():
                self.logger.error(f"Data file not found: {file_path}")
                return None
            
            # Check cache first
            if self.use_cache:
                cached_data = self._load_from_cache(file_path, **kwargs)
                if cached_data is not None:
                    self.logger.info(f"Loaded data from cache: {file_path}")
                    return cached_data
            
            # Load data
            self.logger.info(f"Loading data from: {file_path}")
            data = self._load_file(file_path, **kwargs)
            
            if data is not None:
                # Cache the data
                if self.use_cache:
                    self._save_to_cache(file_path, data, **kwargs)
                
                self.logger.info(f"Successfully loaded {len(data)} records")
                return data
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error loading data from {file_path}: {str(e)}")
            return None
    
    def _load_file(self, file_path: Path, **kwargs) -> Optional[pd.DataFrame]:
        """Load file using appropriate loader"""
        file_extension = file_path.suffix.lower()
        
        if file_extension not in self.loaders:
            self.logger.error(f"Unsupported file format: {file_extension}")
            return None
        
        loader = self.loaders[file_extension]
        
        # Validate format first
        if not loader.validate_format(str(file_path)):
            self.logger.error(f"Invalid file format: {file_path}")
            return None
        
        # Load data
        return loader.load(str(file_path), **kwargs)
    
    def _get_cache_key(self, file_path: Path, **kwargs) -> str:
        """Generate cache key for file"""
        # Include file path, modification time, and kwargs in hash
        file_stat = file_path.stat()
        cache_input = {
            "file_path": str(file_path),
            "mtime": file_stat.st_mtime,
            "size": file_stat.st_size,
            "kwargs": sorted(kwargs.items())
        }
        
        cache_string = json.dumps(cache_input, sort_keys=True)
        return hashlib.md5(cache_string.encode()).hexdigest()
    
    def _load_from_cache(self, file_path: Path, **kwargs) -> Optional[pd.DataFrame]:
        """Load data from cache if available and valid"""
        try:
            cache_key = self._get_cache_key(file_path, **kwargs)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            metadata_file = self.cache_dir / f"{cache_key}.meta"
            
            if not cache_file.exists() or not metadata_file.exists():
                return None
            
            # Check cache validity
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            cache_time = datetime.fromisoformat(metadata["created_at"])
            current_time = datetime.now()
            
            # Check TTL
            if (current_time - cache_time).total_seconds() > (self.cache_ttl_hours * 3600):
                self.logger.debug(f"Cache expired for {file_path}")
                return None
            
            # Load cached data
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            
            return data
            
        except Exception as e:
            self.logger.warning(f"Error loading from cache: {str(e)}")
            return None
    
    def _save_to_cache(self, file_path: Path, data: pd.DataFrame, **kwargs) -> None:
        """Save data to cache"""
        try:
            cache_key = self._get_cache_key(file_path, **kwargs)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            metadata_file = self.cache_dir / f"{cache_key}.meta"
            
            # Save data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            # Save metadata
            metadata = {
                "file_path": str(file_path),
                "created_at": datetime.now().isoformat(),
                "data_shape": data.shape,
                "kwargs": kwargs
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.logger.debug(f"Cached data for {file_path}")
            
        except Exception as e:
            self.logger.warning(f"Error saving to cache: {str(e)}")
    
    def load_streaming(self, file_path: str, chunk_size: int = 1000, **kwargs) -> Generator[pd.DataFrame, None, None]:
        """Load data in streaming mode for large files"""
        try:
            file_path = Path(file_path)
            
            if not file_path.is_absolute():
                file_path = self.data_dir / file_path
            
            if not file_path.exists():
                self.logger.error(f"Data file not found: {file_path}")
                return
            
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.csv':
                yield from self._stream_csv(file_path, chunk_size, **kwargs)
            elif file_extension == '.tsv':
                yield from self._stream_csv(file_path, chunk_size, sep='\t', **kwargs)
            elif file_extension == '.jsonl':
                yield from self._stream_jsonl(file_path, chunk_size, **kwargs)
            else:
                self.logger.error(f"Streaming not supported for format: {file_extension}")
                return
                
        except Exception as e:
            self.logger.error(f"Error in streaming load: {str(e)}")
    
    def _stream_csv(self, file_path: Path, chunk_size: int, **kwargs) -> Generator[pd.DataFrame, None, None]:
        """Stream CSV data in chunks"""
        encoding = kwargs.get('encoding', 'utf-8')
        separator = kwargs.get('sep', ',')
        
        for chunk in pd.read_csv(
            file_path,
            encoding=encoding,
            sep=separator,
            chunksize=chunk_size,
            **{k: v for k, v in kwargs.items() if k not in ['encoding', 'sep']}
        ):
            yield chunk
    
    def _stream_jsonl(self, file_path: Path, chunk_size: int, **kwargs) -> Generator[pd.DataFrame, None, None]:
        """Stream JSONL data in chunks"""
        encoding = kwargs.get('encoding', 'utf-8')
        
        chunk_data = []
        
        with open(file_path, 'r', encoding=encoding) as f:
            for line in f:
                line = line.strip()
                if line:
                    chunk_data.append(json.loads(line))
                    
                    if len(chunk_data) >= chunk_size:
                        yield pd.DataFrame(chunk_data)
                        chunk_data = []
            
            # Yield remaining data
            if chunk_data:
                yield pd.DataFrame(chunk_data)
    
    def convert_format(self, input_path: str, output_path: str, output_format: str, **kwargs) -> bool:
        """Convert data between formats"""
        try:
            # Load data
            data = self.load_data(input_path, **kwargs)
            if data is None:
                return False
            
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save in new format
            if output_format.lower() == 'csv':
                data.to_csv(output_path, index=False, encoding='utf-8')
            elif output_format.lower() == 'json':
                data.to_json(output_path, orient='records', force_ascii=False, indent=2)
            elif output_format.lower() == 'jsonl':
                with open(output_path, 'w', encoding='utf-8') as f:
                    for _, row in data.iterrows():
                        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\n')
            elif output_format.lower() == 'tsv':
                data.to_csv(output_path, sep='\t', index=False, encoding='utf-8')
            else:
                self.logger.error(f"Unsupported output format: {output_format}")
                return False
            
            self.logger.info(f"Converted {input_path} to {output_path} ({output_format})")
            return True
            
        except Exception as e:
            self.logger.error(f"Error converting format: {str(e)}")
            return False
    
    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Get information about a data file"""
        try:
            file_path = Path(file_path)
            
            if not file_path.is_absolute():
                file_path = self.data_dir / file_path
            
            if not file_path.exists():
                return None
            
            file_stat = file_path.stat()
            file_extension = file_path.suffix.lower()
            
            info = {
                "file_path": str(file_path),
                "file_name": file_path.name,
                "file_size_bytes": file_stat.st_size,
                "file_size_mb": file_stat.st_size / (1024 * 1024),
                "modified_time": datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                "file_format": file_extension,
                "supported": file_extension in self.loaders
            }
            
            # Try to get row count for supported formats
            if info["supported"]:
                try:
                    if file_extension in ['.csv', '.tsv']:
                        # Count lines (approximate for CSV/TSV)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            line_count = sum(1 for _ in f) - 1  # Subtract header
                        info["estimated_rows"] = max(0, line_count)
                    
                    elif file_extension == '.jsonl':
                        # Count lines
                        with open(file_path, 'r', encoding='utf-8') as f:
                            line_count = sum(1 for line in f if line.strip())
                        info["estimated_rows"] = line_count
                    
                    elif file_extension == '.json':
                        # Load and count
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        if isinstance(data, list):
                            info["estimated_rows"] = len(data)
                        else:
                            info["estimated_rows"] = 1
                            
                except Exception:
                    info["estimated_rows"] = "unknown"
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting file info: {str(e)}")
            return None
    
    def list_data_files(self, directory: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all data files in directory"""
        try:
            if directory is None:
                search_dir = self.data_dir
            else:
                search_dir = Path(directory)
                if not search_dir.is_absolute():
                    search_dir = self.data_dir / directory
            
            if not search_dir.exists():
                self.logger.warning(f"Directory not found: {search_dir}")
                return []
            
            files_info = []
            
            for file_path in search_dir.rglob('*'):
                if file_path.is_file() and file_path.suffix.lower() in self.loaders:
                    file_info = self.get_file_info(str(file_path))
                    if file_info:
                        files_info.append(file_info)
            
            # Sort by modification time (newest first)
            files_info.sort(key=lambda x: x["modified_time"], reverse=True)
            
            return files_info
            
        except Exception as e:
            self.logger.error(f"Error listing data files: {str(e)}")
            return []
    
    def clear_cache(self, file_path: Optional[str] = None) -> bool:
        """Clear data cache"""
        try:
            if file_path is None:
                # Clear all cache
                for cache_file in self.cache_dir.glob('*'):
                    cache_file.unlink()
                self.logger.info("Cleared all cache")
            else:
                # Clear cache for specific file
                file_path = Path(file_path)
                if not file_path.is_absolute():
                    file_path = self.data_dir / file_path
                
                cache_key = self._get_cache_key(file_path)
                cache_file = self.cache_dir / f"{cache_key}.pkl"
                metadata_file = self.cache_dir / f"{cache_key}.meta"
                
                if cache_file.exists():
                    cache_file.unlink()
                if metadata_file.exists():
                    metadata_file.unlink()
                
                self.logger.info(f"Cleared cache for {file_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {str(e)}")
            return False
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information"""
        try:
            cache_files = list(self.cache_dir.glob('*.pkl'))
            total_size = sum(f.stat().st_size for f in cache_files)
            
            return {
                "cache_directory": str(self.cache_dir),
                "cached_files": len(cache_files),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "cache_enabled": self.use_cache,
                "cache_ttl_hours": self.cache_ttl_hours
            }
            
        except Exception as e:
            self.logger.error(f"Error getting cache info: {str(e)}")
            return {}