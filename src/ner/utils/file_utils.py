"""File Utilities for NER System

Provides file and directory management utilities.
"""

import os
import json
import pickle
import shutil
import zipfile
import tarfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
import hashlib
from datetime import datetime

class FileUtils:
    """File and directory utilities"""
    
    @staticmethod
    def ensure_dir(path: Union[str, Path]) -> Path:
        """Ensure directory exists
        
        Args:
            path: Directory path
            
        Returns:
            Path object
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    @staticmethod
    def safe_remove(path: Union[str, Path]) -> bool:
        """Safely remove file or directory
        
        Args:
            path: Path to remove
            
        Returns:
            True if removed successfully
        """
        try:
            path = Path(path)
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
            return True
        except Exception as e:
            print(f"Error removing {path}: {e}")
            return False
    
    @staticmethod
    def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Copy file
        
        Args:
            src: Source file path
            dst: Destination file path
            
        Returns:
            True if copied successfully
        """
        try:
            src = Path(src)
            dst = Path(dst)
            
            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy2(src, dst)
            return True
        except Exception as e:
            print(f"Error copying {src} to {dst}: {e}")
            return False
    
    @staticmethod
    def move_file(src: Union[str, Path], dst: Union[str, Path]) -> bool:
        """Move file
        
        Args:
            src: Source file path
            dst: Destination file path
            
        Returns:
            True if moved successfully
        """
        try:
            src = Path(src)
            dst = Path(dst)
            
            # Ensure destination directory exists
            dst.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.move(str(src), str(dst))
            return True
        except Exception as e:
            print(f"Error moving {src} to {dst}: {e}")
            return False
    
    @staticmethod
    def get_file_size(path: Union[str, Path]) -> int:
        """Get file size in bytes
        
        Args:
            path: File path
            
        Returns:
            File size in bytes
        """
        try:
            return Path(path).stat().st_size
        except Exception:
            return 0
    
    @staticmethod
    def get_file_hash(path: Union[str, Path], algorithm: str = "md5") -> str:
        """Get file hash
        
        Args:
            path: File path
            algorithm: Hash algorithm (md5, sha1, sha256)
            
        Returns:
            File hash
        """
        try:
            hash_obj = hashlib.new(algorithm)
            with open(path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_obj.update(chunk)
            return hash_obj.hexdigest()
        except Exception as e:
            print(f"Error calculating hash for {path}: {e}")
            return ""
    
    @staticmethod
    def list_files(
        directory: Union[str, Path], 
        pattern: str = "*",
        recursive: bool = False
    ) -> List[Path]:
        """List files in directory
        
        Args:
            directory: Directory path
            pattern: File pattern
            recursive: Whether to search recursively
            
        Returns:
            List of file paths
        """
        try:
            directory = Path(directory)
            if recursive:
                return list(directory.rglob(pattern))
            else:
                return list(directory.glob(pattern))
        except Exception as e:
            print(f"Error listing files in {directory}: {e}")
            return []
    
    @staticmethod
    def get_directory_size(directory: Union[str, Path]) -> int:
        """Get total size of directory
        
        Args:
            directory: Directory path
            
        Returns:
            Total size in bytes
        """
        try:
            total_size = 0
            for file_path in Path(directory).rglob('*'):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
            return total_size
        except Exception as e:
            print(f"Error calculating directory size for {directory}: {e}")
            return 0
    
    @staticmethod
    def backup_file(path: Union[str, Path], backup_dir: Optional[Union[str, Path]] = None) -> Optional[Path]:
        """Create backup of file
        
        Args:
            path: File path to backup
            backup_dir: Backup directory (default: same directory)
            
        Returns:
            Backup file path
        """
        try:
            path = Path(path)
            if not path.exists():
                return None
            
            if backup_dir is None:
                backup_dir = path.parent
            else:
                backup_dir = Path(backup_dir)
                backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{path.stem}_{timestamp}{path.suffix}"
            backup_path = backup_dir / backup_name
            
            shutil.copy2(path, backup_path)
            return backup_path
        except Exception as e:
            print(f"Error creating backup for {path}: {e}")
            return None
    
    @staticmethod
    def compress_directory(
        directory: Union[str, Path], 
        output_path: Union[str, Path],
        format: str = "zip"
    ) -> bool:
        """Compress directory
        
        Args:
            directory: Directory to compress
            output_path: Output archive path
            format: Archive format (zip, tar, tar.gz)
            
        Returns:
            True if compressed successfully
        """
        try:
            directory = Path(directory)
            output_path = Path(output_path)
            
            if format == "zip":
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    for file_path in directory.rglob('*'):
                        if file_path.is_file():
                            arcname = file_path.relative_to(directory)
                            zipf.write(file_path, arcname)
            
            elif format in ["tar", "tar.gz"]:
                mode = "w:gz" if format == "tar.gz" else "w"
                with tarfile.open(output_path, mode) as tarf:
                    tarf.add(directory, arcname=directory.name)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            return True
        except Exception as e:
            print(f"Error compressing {directory}: {e}")
            return False
    
    @staticmethod
    def extract_archive(
        archive_path: Union[str, Path], 
        extract_dir: Union[str, Path]
    ) -> bool:
        """Extract archive
        
        Args:
            archive_path: Archive file path
            extract_dir: Directory to extract to
            
        Returns:
            True if extracted successfully
        """
        try:
            archive_path = Path(archive_path)
            extract_dir = Path(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zipf:
                    zipf.extractall(extract_dir)
            
            elif archive_path.suffix in ['.tar', '.gz']:
                with tarfile.open(archive_path, 'r:*') as tarf:
                    tarf.extractall(extract_dir)
            
            else:
                raise ValueError(f"Unsupported archive format: {archive_path.suffix}")
            
            return True
        except Exception as e:
            print(f"Error extracting {archive_path}: {e}")
            return False
    
    @staticmethod
    def read_json(path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        """Read JSON file
        
        Args:
            path: JSON file path
            
        Returns:
            JSON data or None if error
        """
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error reading JSON file {path}: {e}")
            return None
    
    @staticmethod
    def write_json(
        data: Dict[str, Any], 
        path: Union[str, Path],
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> bool:
        """Write JSON file
        
        Args:
            data: Data to write
            path: JSON file path
            indent: JSON indentation
            ensure_ascii: Whether to ensure ASCII encoding
            
        Returns:
            True if written successfully
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)
            return True
        except Exception as e:
            print(f"Error writing JSON file {path}: {e}")
            return False
    
    @staticmethod
    def read_pickle(path: Union[str, Path]) -> Optional[Any]:
        """Read pickle file
        
        Args:
            path: Pickle file path
            
        Returns:
            Pickled data or None if error
        """
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error reading pickle file {path}: {e}")
            return None
    
    @staticmethod
    def write_pickle(data: Any, path: Union[str, Path]) -> bool:
        """Write pickle file
        
        Args:
            data: Data to pickle
            path: Pickle file path
            
        Returns:
            True if written successfully
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'wb') as f:
                pickle.dump(data, f)
            return True
        except Exception as e:
            print(f"Error writing pickle file {path}: {e}")
            return False
    
    @staticmethod
    def read_text(path: Union[str, Path], encoding: str = 'utf-8') -> Optional[str]:
        """Read text file
        
        Args:
            path: Text file path
            encoding: File encoding
            
        Returns:
            File content or None if error
        """
        try:
            with open(path, 'r', encoding=encoding) as f:
                return f.read()
        except Exception as e:
            print(f"Error reading text file {path}: {e}")
            return None
    
    @staticmethod
    def write_text(
        text: str, 
        path: Union[str, Path], 
        encoding: str = 'utf-8'
    ) -> bool:
        """Write text file
        
        Args:
            text: Text to write
            path: Text file path
            encoding: File encoding
            
        Returns:
            True if written successfully
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding=encoding) as f:
                f.write(text)
            return True
        except Exception as e:
            print(f"Error writing text file {path}: {e}")
            return False
    
    @staticmethod
    def read_lines(
        path: Union[str, Path], 
        encoding: str = 'utf-8',
        strip_whitespace: bool = True
    ) -> List[str]:
        """Read text file lines
        
        Args:
            path: Text file path
            encoding: File encoding
            strip_whitespace: Whether to strip whitespace
            
        Returns:
            List of lines
        """
        try:
            with open(path, 'r', encoding=encoding) as f:
                lines = f.readlines()
                if strip_whitespace:
                    lines = [line.strip() for line in lines]
                return lines
        except Exception as e:
            print(f"Error reading lines from {path}: {e}")
            return []
    
    @staticmethod
    def write_lines(
        lines: List[str], 
        path: Union[str, Path], 
        encoding: str = 'utf-8',
        add_newlines: bool = True
    ) -> bool:
        """Write text file lines
        
        Args:
            lines: Lines to write
            path: Text file path
            encoding: File encoding
            add_newlines: Whether to add newlines
            
        Returns:
            True if written successfully
        """
        try:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(path, 'w', encoding=encoding) as f:
                for line in lines:
                    if add_newlines and not line.endswith('\n'):
                        line += '\n'
                    f.write(line)
            return True
        except Exception as e:
            print(f"Error writing lines to {path}: {e}")
            return False
    
    @staticmethod
    def find_files_by_extension(
        directory: Union[str, Path], 
        extensions: Union[str, List[str]],
        recursive: bool = True
    ) -> List[Path]:
        """Find files by extension
        
        Args:
            directory: Directory to search
            extensions: File extensions (with or without dot)
            recursive: Whether to search recursively
            
        Returns:
            List of matching files
        """
        if isinstance(extensions, str):
            extensions = [extensions]
        
        # Normalize extensions
        extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]
        
        files = []
        try:
            directory = Path(directory)
            pattern = "**/*" if recursive else "*"
            
            for file_path in directory.glob(pattern):
                if file_path.is_file() and file_path.suffix.lower() in extensions:
                    files.append(file_path)
        except Exception as e:
            print(f"Error finding files in {directory}: {e}")
        
        return files
    
    @staticmethod
    def get_file_info(path: Union[str, Path]) -> Dict[str, Any]:
        """Get file information
        
        Args:
            path: File path
            
        Returns:
            File information dictionary
        """
        try:
            path = Path(path)
            stat = path.stat()
            
            return {
                'name': path.name,
                'size': stat.st_size,
                'created': datetime.fromtimestamp(stat.st_ctime),
                'modified': datetime.fromtimestamp(stat.st_mtime),
                'accessed': datetime.fromtimestamp(stat.st_atime),
                'is_file': path.is_file(),
                'is_dir': path.is_dir(),
                'extension': path.suffix,
                'parent': str(path.parent),
                'absolute_path': str(path.absolute())
            }
        except Exception as e:
            print(f"Error getting file info for {path}: {e}")
            return {}
    
    @staticmethod
    def clean_directory(
        directory: Union[str, Path], 
        older_than_days: Optional[int] = None,
        pattern: str = "*"
    ) -> int:
        """Clean directory by removing old files
        
        Args:
            directory: Directory to clean
            older_than_days: Remove files older than this many days
            pattern: File pattern to match
            
        Returns:
            Number of files removed
        """
        try:
            directory = Path(directory)
            removed_count = 0
            
            if older_than_days is not None:
                cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 60 * 60)
            
            for file_path in directory.glob(pattern):
                if file_path.is_file():
                    should_remove = False
                    
                    if older_than_days is not None:
                        file_time = file_path.stat().st_mtime
                        should_remove = file_time < cutoff_time
                    else:
                        should_remove = True
                    
                    if should_remove:
                        file_path.unlink()
                        removed_count += 1
            
            return removed_count
        except Exception as e:
            print(f"Error cleaning directory {directory}: {e}")
            return 0