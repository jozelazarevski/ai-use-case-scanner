# -*- coding: utf-8 -*-
"""
Advanced Universal File Reader
Enhanced with compression support, data validation, type inference, and more
"""

import pandas as pd
import numpy as np
import chardet
import io
import csv
import json
import os
import sys
import re
import warnings
import logging
import traceback
import gzip
import zipfile
import tarfile
import bz2
import lzma
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple, Union, Any
from dataclasses import dataclass
from functools import lru_cache
import concurrent.futures
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class FileInfo:
    """Store file metadata and reading results"""
    path: str
    size: int
    encoding: Optional[str] = None
    delimiter: Optional[str] = None
    has_header: Optional[bool] = None
    row_count: Optional[int] = None
    column_count: Optional[int] = None
    compression: Optional[str] = None
    format_type: Optional[str] = None
    read_time: Optional[float] = None
    memory_usage: Optional[float] = None
    sample_data: Optional[str] = None
    issues: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []


class DataTypeInferencer:
    """Advanced data type inference for columns"""
    
    @staticmethod
    def infer_column_types(df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, str]:
        """Infer better data types for columns"""
        type_suggestions = {}
        
        for col in df.columns:
            # Sample the column
            sample = df[col].dropna().head(sample_size)
            
            if sample.empty:
                type_suggestions[col] = 'empty'
                continue
            
            # Try to infer the best type
            suggestion = DataTypeInferencer._infer_single_column(sample, col)
            type_suggestions[col] = suggestion
        
        return type_suggestions
    
    @staticmethod
    def _infer_single_column(series: pd.Series, col_name: str) -> str:
        """Infer type for a single column"""
        # Check if already numeric
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                return 'int'
            return 'float'
        
        # Convert to string for analysis
        str_series = series.astype(str)
        
        # Check for boolean
        unique_vals = str_series.str.lower().unique()
        if len(unique_vals) <= 2 and all(val in ['true', 'false', 'yes', 'no', '0', '1', 't', 'f', 'y', 'n'] 
                                          for val in unique_vals):
            return 'bool'
        
        # Check for datetime
        if DataTypeInferencer._is_datetime(str_series):
            return 'datetime'
        
        # Check for numeric
        if DataTypeInferencer._is_numeric(str_series):
            if '.' in ''.join(str_series.head(100).values):
                return 'float'
            return 'int'
        
        # Check for categorical
        if len(unique_vals) < len(series) * 0.5:  # Less than 50% unique
            return 'category'
        
        # Check for URLs, emails, etc.
        if col_name.lower() in ['url', 'website', 'link']:
            return 'url'
        if col_name.lower() in ['email', 'e-mail', 'mail']:
            return 'email'
        
        return 'string'
    
    @staticmethod
    def _is_datetime(series: pd.Series) -> bool:
        """Check if series contains datetime values"""
        try:
            # Try common datetime formats
            pd.to_datetime(series.head(100), errors='coerce').notna().sum() > 80
            return True
        except:
            return False
    
    @staticmethod
    def _is_numeric(series: pd.Series) -> bool:
        """Check if series contains numeric values"""
        try:
            pd.to_numeric(series.head(100), errors='coerce').notna().sum() > 80
            return True
        except:
            return False


class CompressionHandler:
    """Handle compressed files"""
    
    COMPRESSION_FORMATS = {
        '.gz': gzip.open,
        '.bz2': bz2.open,
        '.xz': lzma.open,
        '.zip': 'zip',
        '.tar': 'tar',
        '.tar.gz': 'tar',
        '.tgz': 'tar'
    }
    
    @staticmethod
    def is_compressed(filepath: str) -> Tuple[bool, Optional[str]]:
        """Check if file is compressed and return compression type"""
        path = Path(filepath)
        
        # Check double extensions like .tar.gz
        if path.suffix == '.gz' and path.stem.endswith('.tar'):
            return True, '.tar.gz'
        
        for ext, handler in CompressionHandler.COMPRESSION_FORMATS.items():
            if filepath.endswith(ext):
                return True, ext
        
        # Check file magic bytes
        try:
            with open(filepath, 'rb') as f:
                magic = f.read(4)
                
            if magic[:2] == b'\x1f\x8b':  # gzip
                return True, '.gz'
            elif magic[:3] == b'BZh':  # bz2
                return True, '.bz2'
            elif magic[:6] == b'\xfd7zXZ\x00':  # xz
                return True, '.xz'
            elif magic[:4] == b'PK\x03\x04':  # zip
                return True, '.zip'
                
        except:
            pass
        
        return False, None
    
    @staticmethod
    def decompress_file(filepath: str, compression_type: str) -> Union[str, io.IOBase]:
        """Decompress file and return path or file object"""
        if compression_type in ['.gz', '.bz2', '.xz']:
            handler = CompressionHandler.COMPRESSION_FORMATS[compression_type]
            return handler(filepath, 'rt')
        
        elif compression_type == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zf:
                # List files in the zip
                files = zf.namelist()
                data_files = [f for f in files if not f.startswith('__MACOSX') and not f.startswith('.')]
                
                if len(data_files) == 1:
                    # Single file, return its content
                    return io.StringIO(zf.read(data_files[0]).decode('utf-8'))
                else:
                    # Multiple files, let user choose or return first data file
                    logger.warning(f"Multiple files in zip: {data_files}")
                    return io.StringIO(zf.read(data_files[0]).decode('utf-8'))
        
        elif compression_type in ['.tar', '.tar.gz', '.tgz']:
            mode = 'r:gz' if compression_type in ['.tar.gz', '.tgz'] else 'r'
            with tarfile.open(filepath, mode) as tf:
                # Find data files
                data_files = [m for m in tf.getmembers() if m.isfile() and not m.name.startswith('.')]
                
                if data_files:
                    # Extract first data file
                    member = data_files[0]
                    return io.StringIO(tf.extractfile(member).read().decode('utf-8'))
        
        return filepath


class DataValidator:
    """Validate data quality and consistency"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, file_info: FileInfo) -> Dict[str, Any]:
        """Comprehensive dataframe validation"""
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'stats': {}
        }
        
        # Check for empty dataframe
        if df.empty:
            validation_results['errors'].append("DataFrame is empty")
            validation_results['is_valid'] = False
            return validation_results
        
        # Check for suspicious patterns
        if len(df.columns) == 1 and df.shape[0] > 10:
            # Check if single column contains delimited data
            sample_val = str(df.iloc[0, 0])
            if any(delim in sample_val for delim in [',', ';', '\t', '|']):
                validation_results['errors'].append("Data appears to be improperly parsed (delimiters found in single column)")
                validation_results['is_valid'] = False
        
        # Check for too many unnamed columns
        unnamed_cols = sum(1 for col in df.columns if 'Unnamed' in str(col))
        if unnamed_cols > len(df.columns) * 0.5:
            validation_results['warnings'].append(f"{unnamed_cols} unnamed columns detected")
        
        # Check for duplicate columns
        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            validation_results['warnings'].append(f"Duplicate column names: {duplicate_cols}")
        
        # Check for missing values
        missing_stats = df.isnull().sum()
        high_missing = missing_stats[missing_stats > len(df) * 0.9]
        if not high_missing.empty:
            validation_results['warnings'].append(f"Columns with >90% missing: {high_missing.index.tolist()}")
        
        # Data statistics
        validation_results['stats'] = {
            'total_missing': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100),
            'duplicate_rows': df.duplicated().sum(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        return validation_results


class UniversalFileReader:
    """Main class for reading files with all enhancements"""
    
    def __init__(self, 
                 verbose: bool = True,
                 validate: bool = True,
                 infer_types: bool = True,
                 handle_compression: bool = True,
                 parallel: bool = False,
                 cache_size: int = 128):
        """
        Initialize the Universal File Reader
        
        Args:
            verbose: Print detailed progress information
            validate: Perform data validation after reading
            infer_types: Attempt to infer better data types
            handle_compression: Automatically handle compressed files
            parallel: Use parallel processing for large files
            cache_size: Size of LRU cache for encoding detection
        """
        self.verbose = verbose
        self.validate = validate
        self.infer_types = infer_types
        self.handle_compression = handle_compression
        self.parallel = parallel
        self._setup_logging()
        
        # Cache for encoding detection
        self._detect_encoding = lru_cache(maxsize=cache_size)(self._detect_encoding_impl)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if self.verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)
    
    def read_file(self, 
                  filepath: str, 
                  output_format: str = 'dataframe',
                  **kwargs) -> Union[pd.DataFrame, Dict[str, Any], None]:
        """
        Main method to read any file
        
        Args:
            filepath: Path to the file
            output_format: 'dataframe', 'dict', or 'records'
            **kwargs: Additional arguments for pandas readers
            
        Returns:
            Data in requested format or None if failed
        """
        start_time = datetime.now()
        file_info = FileInfo(path=filepath, size=os.path.getsize(filepath))
        
        try:
            # Check if file exists
            if not os.path.exists(filepath):
                logger.error(f"File not found: {filepath}")
                return None
            
            # Handle compression
            if self.handle_compression:
                is_compressed, comp_type = CompressionHandler.is_compressed(filepath)
                if is_compressed:
                    logger.info(f"Detected {comp_type} compression")
                    file_info.compression = comp_type
                    file_obj = CompressionHandler.decompress_file(filepath, comp_type)
                    if isinstance(file_obj, io.IOBase):
                        # Read from decompressed stream
                        df = self._read_from_stream(file_obj, file_info, **kwargs)
                    else:
                        # Got a new filepath
                        df = self._read_file_internal(file_obj, file_info, **kwargs)
                else:
                    df = self._read_file_internal(filepath, file_info, **kwargs)
            else:
                df = self._read_file_internal(filepath, file_info, **kwargs)
            
            if df is None:
                return None
            
            # Record reading time
            file_info.read_time = (datetime.now() - start_time).total_seconds()
            file_info.row_count = len(df)
            file_info.column_count = len(df.columns)
            file_info.memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
            
            # Validate if requested
            if self.validate:
                validation_results = DataValidator.validate_dataframe(df, file_info)
                if not validation_results['is_valid']:
                    logger.error(f"Validation failed: {validation_results['errors']}")
                    file_info.issues.extend(validation_results['errors'])
                for warning in validation_results['warnings']:
                    logger.warning(warning)
                    file_info.issues.append(f"Warning: {warning}")
            
            # Infer types if requested
            if self.infer_types and not df.empty:
                logger.info("Inferring optimal data types...")
                type_suggestions = DataTypeInferencer.infer_column_types(df)
                df = self._apply_type_suggestions(df, type_suggestions)
            
            # Convert to requested format
            if output_format == 'dict':
                return {'data': df.to_dict(), 'info': file_info.__dict__}
            elif output_format == 'records':
                return {'data': df.to_dict('records'), 'info': file_info.__dict__}
            else:
                # Add metadata as attributes
                df.attrs['file_info'] = file_info.__dict__
                return df
                
        except Exception as e:
            logger.error(f"Failed to read file: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
    
    def _read_file_internal(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Internal method to read files based on extension"""
        file_ext = Path(filepath).suffix.lower()
        
        # Route to appropriate reader
        if file_ext in ['.xlsx', '.xls', '.xlsm', '.xlsb']:
            return self._read_excel(filepath, file_info, **kwargs)
        elif file_ext in ['.json', '.jsonl']:
            return self._read_json(filepath, file_info, **kwargs)
        elif file_ext == '.parquet':
            return self._read_parquet(filepath, file_info, **kwargs)
        elif file_ext in ['.pkl', '.pickle']:
            return self._read_pickle(filepath, file_info, **kwargs)
        elif file_ext in ['.feather']:
            return self._read_feather(filepath, file_info, **kwargs)
        elif file_ext in ['.h5', '.hdf5', '.hdf']:
            return self._read_hdf(filepath, file_info, **kwargs)
        elif file_ext in ['.sas7bdat']:
            return self._read_sas(filepath, file_info, **kwargs)
        elif file_ext in ['.dta']:
            return self._read_stata(filepath, file_info, **kwargs)
        elif file_ext in ['.sav', '.zsav']:
            return self._read_spss(filepath, file_info, **kwargs)
        elif file_ext in ['.xml']:
            return self._read_xml(filepath, file_info, **kwargs)
        elif file_ext in ['.html', '.htm']:
            return self._read_html(filepath, file_info, **kwargs)
        else:
            # Default to text-based reader
            return self._read_text(filepath, file_info, **kwargs)
    
    def _read_from_stream(self, stream: io.IOBase, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read from a file stream"""
        # For streams, we'll try CSV reading
        try:
            df = pd.read_csv(stream, **kwargs)
            return df
        except Exception as e:
            logger.error(f"Failed to read from stream: {e}")
            return None
    
    def _detect_encoding_impl(self, filepath: str, sample_size: int = 1024*1024) -> List[str]:
        """Detect file encoding (cached implementation)"""
        try:
            with open(filepath, 'rb') as f:
                sample = f.read(min(sample_size, os.path.getsize(filepath)))
            
            result = chardet.detect(sample)
            encoding = result['encoding']
            confidence = result['confidence']
            
            logger.debug(f"Detected encoding: {encoding} (confidence: {confidence:.2%})")
            
            encodings = []
            if encoding and confidence > 0.7:
                encodings.append(encoding)
            
            # Add common encodings
            common = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'utf-16', 'utf-32', 'ascii']
            encodings.extend(common)
            
            # Remove None and duplicates
            encodings = [e for e in encodings if e]
            return list(dict.fromkeys(encodings))
            
        except Exception as e:
            logger.error(f"Encoding detection failed: {e}")
            return ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    
    def _read_text(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read text-based files with advanced detection"""
        encodings = self._detect_encoding(filepath)
        
        for encoding in encodings:
            try:
                logger.debug(f"Trying encoding: {encoding}")
                
                # Read sample
                with open(filepath, 'r', encoding=encoding, errors='replace') as f:
                    sample = f.read(10000)
                
                # Check for too many errors
                if sample.count('�') > len(sample) * 0.05:
                    continue
                
                # Detect delimiter
                delimiter = self._detect_delimiter(sample)
                file_info.delimiter = delimiter
                
                # Detect if file has header
                has_header = self._detect_header(sample, delimiter)
                file_info.has_header = has_header
                
                # Try reading with detected parameters
                read_kwargs = kwargs.copy()
                if delimiter:
                    read_kwargs['sep'] = delimiter
                if not has_header:
                    read_kwargs['header'] = None
                
                df = pd.read_csv(filepath, encoding=encoding, **read_kwargs)
                
                if self._validate_parse(df):
                    file_info.encoding = encoding
                    logger.info(f"Successfully read with {encoding} encoding and '{repr(delimiter)[1:-1]}' delimiter")
                    return df
                    
            except Exception as e:
                logger.debug(f"Failed with {encoding}: {str(e)[:100]}")
                continue
        
        return None
    
    def _detect_delimiter(self, content: str) -> Optional[str]:
        """Advanced delimiter detection"""
        lines = content.strip().split('\n')[:20]
        
        # Common delimiters to check
        delimiters = [',', ';', '\t', '|', ' ', ':', '~', '^', '¦']
        
        scores = {}
        for delim in delimiters:
            counts = [line.count(delim) for line in lines if line.strip()]
            if not counts or all(c == 0 for c in counts):
                scores[delim] = 0
                continue
            
            # Calculate consistency score
            mean_count = np.mean(counts)
            std_count = np.std(counts)
            consistency = 1 - (std_count / (mean_count + 1))
            
            # Weight by total occurrences and consistency
            scores[delim] = sum(counts) * (consistency ** 2)
        
        # Get best delimiter
        if scores:
            best_delim = max(scores.items(), key=lambda x: x[1])
            if best_delim[1] > 1:  # Minimum threshold
                return best_delim[0]
        
        return None
    
    def _detect_header(self, content: str, delimiter: Optional[str]) -> bool:
        """Detect if file has a header row"""
        if not delimiter:
            return True  # Default assumption
        
        lines = content.strip().split('\n')[:5]
        if len(lines) < 2:
            return True
        
        # Parse first two rows
        first_row = lines[0].split(delimiter)
        second_row = lines[1].split(delimiter)
        
        # Check if first row looks like headers
        # Headers typically have more text and fewer numbers
        first_numeric = sum(1 for val in first_row if val.strip().replace('.', '').replace('-', '').isdigit())
        second_numeric = sum(1 for val in second_row if val.strip().replace('.', '').replace('-', '').isdigit())
        
        return first_numeric < second_numeric
    
    def _validate_parse(self, df: pd.DataFrame) -> bool:
        """Quick validation of parsed dataframe"""
        if df is None or df.empty:
            return False
        
        # Single column with delimiters suggests bad parse
        if len(df.columns) == 1 and df.shape[0] > 1:
            first_val = str(df.iloc[0, 0])
            if any(d in first_val for d in [',', ';', '\t', '|']):
                return False
        
        return True
    
    def _apply_type_suggestions(self, df: pd.DataFrame, suggestions: Dict[str, str]) -> pd.DataFrame:
        """Apply suggested data types to dataframe"""
        for col, suggested_type in suggestions.items():
            if col not in df.columns:
                continue
                
            try:
                if suggested_type == 'int':
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('int64')
                elif suggested_type == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif suggested_type == 'bool':
                    df[col] = df[col].astype(str).str.lower().map({
                        'true': True, 'false': False,
                        'yes': True, 'no': False,
                        '1': True, '0': False,
                        't': True, 'f': False,
                        'y': True, 'n': False
                    })
                elif suggested_type == 'datetime':
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                elif suggested_type == 'category':
                    df[col] = df[col].astype('category')
            except Exception as e:
                logger.debug(f"Could not convert {col} to {suggested_type}: {e}")
        
        return df
    
    # Additional format readers
    def _read_excel(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read Excel files"""
        engines = ['openpyxl', 'xlrd', None]
        for engine in engines:
            try:
                if engine:
                    df = pd.read_excel(filepath, engine=engine, **kwargs)
                else:
                    df = pd.read_excel(filepath, **kwargs)
                file_info.format_type = 'excel'
                return df
            except:
                continue
        return None
    
    def _read_json(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read JSON files"""
        try:
            df = pd.read_json(filepath, **kwargs)
            file_info.format_type = 'json'
            return df
        except:
            # Try JSON lines
            try:
                df = pd.read_json(filepath, lines=True, **kwargs)
                file_info.format_type = 'jsonl'
                return df
            except:
                return None
    
    def _read_parquet(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read Parquet files"""
        try:
            df = pd.read_parquet(filepath, **kwargs)
            file_info.format_type = 'parquet'
            return df
        except:
            return None
    
    def _read_pickle(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read Pickle files"""
        try:
            df = pd.read_pickle(filepath, **kwargs)
            file_info.format_type = 'pickle'
            return df
        except:
            return None
    
    def _read_feather(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read Feather files"""
        try:
            df = pd.read_feather(filepath, **kwargs)
            file_info.format_type = 'feather'
            return df
        except:
            return None
    
    def _read_hdf(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read HDF5 files"""
        try:
            # If no key specified, try to read the first one
            if 'key' not in kwargs:
                with pd.HDFStore(filepath, 'r') as store:
                    if store.keys():
                        kwargs['key'] = store.keys()[0]
            
            df = pd.read_hdf(filepath, **kwargs)
            file_info.format_type = 'hdf5'
            return df
        except:
            return None
    
    def _read_sas(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read SAS files"""
        try:
            df = pd.read_sas(filepath, **kwargs)
            file_info.format_type = 'sas'
            return df
        except:
            return None
    
    def _read_stata(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read Stata files"""
        try:
            df = pd.read_stata(filepath, **kwargs)
            file_info.format_type = 'stata'
            return df
        except:
            return None
    
    def _read_spss(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read SPSS files"""
        try:
            df = pd.read_spss(filepath, **kwargs)
            file_info.format_type = 'spss'
            return df
        except:
            return None
    
    def _read_xml(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read XML files"""
        try:
            df = pd.read_xml(filepath, **kwargs)
            file_info.format_type = 'xml'
            return df
        except:
            return None
    
    def _read_html(self, filepath: str, file_info: FileInfo, **kwargs) -> Optional[pd.DataFrame]:
        """Read HTML files"""
        try:
            dfs = pd.read_html(filepath, **kwargs)
            if dfs:
                file_info.format_type = 'html'
                # Return first table or concatenate if multiple
                if len(dfs) == 1:
                    return dfs[0]
                else:
                    logger.warning(f"Found {len(dfs)} tables in HTML, returning first")
                    return dfs[0]
        except:
            return None


# Convenience functions
def read_file(filepath: str, **kwargs) -> Union[pd.DataFrame, None]:
    """Simple function to read any file"""
    reader = UniversalFileReader()
    return reader.read_file(filepath, **kwargs)


def read_files(filepaths: List[str], parallel: bool = True, **kwargs) -> Dict[str, pd.DataFrame]:
    """Read multiple files, optionally in parallel"""
    reader = UniversalFileReader(parallel=parallel)
    results = {}
    
    if parallel and len(filepaths) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4, len(filepaths))) as executor:
            future_to_file = {executor.submit(reader.read_file, fp, **kwargs): fp 
                              for fp in filepaths}
            
            for future in concurrent.futures.as_completed(future_to_file):
                filepath = future_to_file[future]
                try:
                    df = future.result()
                    if df is not None:
                        results[filepath] = df
                except Exception as e:
                    logger.error(f"Failed to read {filepath}: {e}")
    else:
        for filepath in filepaths:
            df = reader.read_file(filepath, **kwargs)
            if df is not None:
                results[filepath] = df
    
    return results


def analyze_file(filepath: str) -> Dict[str, Any]:
    """Analyze a file without fully reading it"""
    reader = UniversalFileReader(validate=True, infer_types=True)
    
    # Try to read with minimal rows
    df = reader.read_file(filepath, nrows=1000)
    
    if df is not None and hasattr(df, 'attrs') and 'file_info' in df.attrs:
        info = df.attrs['file_info']
        
        # Add analysis results
        analysis = {
            'file_info': info,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'sample_data': df.head(5).to_dict(),
            'statistics': df.describe(include='all').to_dict() if not df.empty else {}
        }
        
        return analysis
    
    return {'error': 'Could not analyze file'}


# Backward compatibility function
def read_data_flexible(filepath: str, **kwargs) -> Optional[pd.DataFrame]:
    """
    Backward compatibility wrapper for the old read_data_flexible function.
    
    Args:
        filepath: Path to the file to read
        **kwargs: Additional arguments passed to the reader
        
    Returns:
        pd.DataFrame or None if reading fails
    """
    reader = UniversalFileReader(
        verbose=True,
        validate=True,
        infer_types=True,
        handle_compression=True
    )
    return reader.read_file(filepath, output_format='dataframe', **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("\n" + "="*80)
    print("Universal File Reader - Testing Module")
    print("="*80)
    
    # Example 1: Simple usage with read_file function
    print("\n1. SIMPLE USAGE EXAMPLE:")
    print("-" * 40)
    print("df = read_file('your_data.csv')")
    print("df = read_file('compressed_data.csv.gz')")
    print("df = read_file('excel_data.xlsx')")
    
    # Example 2: Using the backward compatibility function
    print("\n2. BACKWARD COMPATIBILITY:")
    print("-" * 40)
    print("df = read_data_flexible('your_data.csv')")
    
    # Example 3: Advanced usage with UniversalFileReader class
    print("\n3. ADVANCED USAGE EXAMPLE:")
    print("-" * 40)
    print("""
# Create reader with custom settings
reader = UniversalFileReader(
    verbose=True,           # Show detailed progress
    validate=True,          # Validate data quality
    infer_types=True,       # Optimize data types
    handle_compression=True # Auto-handle compressed files
)

# Read with specific options
result = reader.read_file(
    'data.csv',
    output_format='dict',    # Returns dict with data and metadata
    nrows=10000,            # Limit rows for testing
    usecols=['col1', 'col2'] # Read specific columns only
)

# Access the data and metadata
df = result['data']
metadata = result['info']
    """)
    
    # Example 4: Reading multiple files
    print("\n4. BATCH PROCESSING EXAMPLE:")
    print("-" * 40)
    print("""
# Read multiple files in parallel
files = ['file1.csv', 'file2.xlsx', 'file3.json.gz']
dataframes = read_files(files, parallel=True)

# Access each dataframe
for filepath, df in dataframes.items():
    print(f"{filepath}: {df.shape}")
    """)
    
    # Example 5: File analysis
    print("\n5. FILE ANALYSIS EXAMPLE:")
    print("-" * 40)
    print("""
# Analyze file without loading all data
analysis = analyze_file('large_dataset.csv')

# View analysis results
print(analysis['file_info'])     # File metadata
print(analysis['columns'])       # Column names
print(analysis['dtypes'])        # Data types
print(analysis['statistics'])    # Summary statistics
    """)
    
    # Example 6: Working with different file formats
    print("\n6. SUPPORTED FILE FORMATS:")
    print("-" * 40)
    formats = {
        'CSV/TSV': ['.csv', '.tsv', '.txt', '.data'],
        'Excel': ['.xlsx', '.xls', '.xlsm', '.xlsb'],
        'JSON': ['.json', '.jsonl'],
        'Compressed': ['.gz', '.bz2', '.xz', '.zip', '.tar.gz'],
        'Binary': ['.parquet', '.feather', '.pkl', '.pickle'],
        'Statistical': ['.sas7bdat', '.dta', '.sav'],
        'Other': ['.xml', '.html', '.hdf5']
    }
    
    for category, extensions in formats.items():
        print(f"{category:12} : {', '.join(extensions)}")
    
    print("\n" + "="*80)
    print("For production use, simply import: from utils.read_file import read_file")
    print("Or for backward compatibility: from utils.read_file import read_data_flexible")
    print("="*80)