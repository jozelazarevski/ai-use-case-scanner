"""
Performance-Optimized Column Identifier

This module provides performance optimizations for the column identification system,
focusing on speed, memory efficiency, and scalability for large datasets.
"""

import pandas as pd
import numpy as np
import re
import time
import logging
import multiprocessing
from typing import Dict, List, Tuple, Union, Optional, Any, Set, Callable
from functools import lru_cache
import threading
from collections import defaultdict
import gc
import math
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import warnings
import psutil

# Set up logging
logger = logging.getLogger('column_identifier.performance')

class PerformanceOptimizer:
    """
    Performance optimization manager for column identification systems.
    Enhances processing speed and memory efficiency for large datasets.
    """
    
    def __init__(self, 
                max_sample_size: int = 10000, 
                min_sample_size: int = 100,
                use_multithreading: bool = True,
                use_multiprocessing: bool = False,
                cache_size: int = 128,
                memory_limit_percentage: float = 75.0,
                dynamic_sampling: bool = True,
                enable_fast_exit: bool = True):
        """
        Initialize the performance optimizer
        
        Args:
            max_sample_size: Maximum number of rows to sample for analysis
            min_sample_size: Minimum number of rows to sample for analysis
            use_multithreading: Enable multithreaded processing for column analysis
            use_multiprocessing: Enable multiprocessing for heavy computations
            cache_size: Size of LRU cache for pattern matching results
            memory_limit_percentage: Memory usage threshold (% of system memory)
            dynamic_sampling: Adjust sample size based on complexity
            enable_fast_exit: Enable early stopping for pattern detection when high confidence
        """
        self.max_sample_size = max_sample_size
        self.min_sample_size = min_sample_size
        self.use_multithreading = use_multithreading
        self.use_multiprocessing = use_multiprocessing
        self.cache_size = cache_size
        self.memory_limit_percentage = memory_limit_percentage
        self.dynamic_sampling = dynamic_sampling
        self.enable_fast_exit = enable_fast_exit
        
        # Initialize performance metrics tracking
        self.performance_metrics = {
            'column_processing_times': {},
            'memory_usage': [],
            'total_processing_time': 0,
            'pattern_match_times': {},
            'sampling_ratios': {},
            'skipped_operations': 0
        }
        
        # Initialize caches
        self._setup_caches()
        
        # Initialize regex pattern compiler
        self._compiled_patterns = {}
        
        # Threading/multiprocessing resources
        self.thread_pool = None
        self.process_pool = None
        
        # Calculate system resources
        self._calculate_system_resources()
    
    def _setup_caches(self):
        """Set up various caches for performance optimization"""
        # Cache for compiled regex patterns
        self.pattern_cache = {}
        
        # Cache for column type detection results
        self.column_type_cache = {}
        
        # Cache for column statistics
        self.stats_cache = {}
        
        # Cache for string pattern matching results
        self.string_pattern_cache = {}
        
        # Cache for value set matching results
        self.valueset_match_cache = {}
    
    def _calculate_system_resources(self):
        """Calculate available system resources for optimization decisions"""
        # Get number of CPUs for parallel processing
        self.cpu_count = multiprocessing.cpu_count()
        self.optimal_thread_count = min(self.cpu_count * 2, 16)  # 2x logical cores, max 16
        
        # Get system memory
        self.system_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
        self.memory_limit = (self.memory_limit_percentage / 100) * self.system_memory
        
        logger.info(f"System resources: {self.cpu_count} CPUs, {self.system_memory:.1f}GB RAM")
        logger.info(f"Using up to {self.optimal_thread_count} threads and {self.memory_limit:.1f}GB RAM")
    
    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize a DataFrame for analysis by downcasting dtypes and removing unnecessary memory usage
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Optimized DataFrame
        """
        start_time = time.time()
        start_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        
        logger.info(f"Optimizing DataFrame: {len(df)} rows, {len(df.columns)} columns, {start_memory:.2f}MB")
        
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Process each column for optimization
        for col in result.columns:
            col_type = result[col].dtype
            
            # For integer columns, downcast to smallest possible integer type
            if pd.api.types.is_integer_dtype(col_type):
                result[col] = pd.to_numeric(result[col], downcast='integer')
            
            # For float columns, downcast to smallest possible float type
            elif pd.api.types.is_float_dtype(col_type):
                result[col] = pd.to_numeric(result[col], downcast='float')
            
            # For object columns, convert to category if cardinality is low
            elif pd.api.types.is_object_dtype(col_type) or pd.api.types.is_string_dtype(col_type):
                num_unique = result[col].nunique()
                if num_unique < len(result) * 0.5:  # If less than 50% unique values
                    result[col] = result[col].astype('category')
        
        # Calculate memory savings
        end_memory = result.memory_usage(deep=True).sum() / (1024 * 1024)  # MB
        end_time = time.time()
        
        logger.info(f"DataFrame optimized: {end_memory:.2f}MB ({(start_memory - end_memory):.2f}MB or {(1-(end_memory/start_memory))*100:.1f}% saved)")
        logger.info(f"Optimization completed in {(end_time - start_time):.2f} seconds")
        
        return result
    
    def get_optimal_sample(self, series: pd.Series, col_name: str = None) -> pd.Series:
        """
        Get an optimally sized sample from a series based on data characteristics
        
        Args:
            series: The pandas Series to sample
            col_name: Column name for tracking
            
        Returns:
            Sampled pandas Series
        """
        # Skip sampling for small series
        if len(series) <= self.max_sample_size:
            return series
        
        # Base sample size on series length with limits
        if self.dynamic_sampling:
            # Calculate base sample size with logarithmic scaling
            base_size = int(self.min_sample_size * (1 + math.log10(len(series) / 1000 + 1)))
            
            # Adjust based on uniqueness - more unique values need larger samples
            unique_ratio = min(1.0, series.nunique() / len(series))
            adjusted_size = int(base_size * (1 + unique_ratio))
            
            # Adjust based on dtype - complex types need larger samples
            dtype_factor = 1.0
            if pd.api.types.is_string_dtype(series.dtype) or pd.api.types.is_object_dtype(series.dtype):
                dtype_factor = 2.0
            elif pd.api.types.is_numeric_dtype(series.dtype):
                # For numeric, increase with variance
                try:
                    if series.std() > 3 * series.mean():
                        dtype_factor = 1.5
                except:
                    pass
            
            final_size = min(self.max_sample_size, max(self.min_sample_size, int(adjusted_size * dtype_factor)))
        else:
            # Use static sample size
            final_size = min(self.max_sample_size, max(self.min_sample_size, int(len(series) * 0.1)))
        
        # Record sampling ratio
        if col_name:
            self.performance_metrics['sampling_ratios'][col_name] = final_size / len(series)
        
        # Take stratified sample if possible for categorical data
        if series.nunique() < 100:  # Only for low-cardinality columns
            try:
                # Try to use stratified sampling
                return series.groupby(series).apply(
                    lambda x: x.sample(min(len(x), max(1, int(final_size * len(x) / len(series)))))
                )
            except:
                # Fall back to random sampling
                return series.sample(final_size)
        else:
            # Use random sampling
            return series.sample(final_size)
    
    @lru_cache(maxsize=1024)
    def compile_regex(self, pattern: str) -> re.Pattern:
        """
        Compile regex pattern with caching
        
        Args:
            pattern: Regex pattern string
            
        Returns:
            Compiled regex pattern
        """
        try:
            return re.compile(pattern)
        except Exception as e:
            logger.warning(f"Invalid regex pattern: {pattern}. Error: {str(e)}")
            # Return a pattern that will never match anything
            return re.compile(r'^\b$')
    
    def analyze_columns_parallel(self, 
                               df: pd.DataFrame, 
                               analyzer_func: Callable, 
                               **kwargs) -> Dict[str, Any]:
        """
        Analyze DataFrame columns in parallel using the provided analyzer function
        
        Args:
            df: DataFrame to analyze
            analyzer_func: Function to analyze each column
            **kwargs: Additional arguments to pass to analyzer_func
            
        Returns:
            Dict of column analysis results
        """
        total_start_time = time.time()
        
        # Initialize thread pool if needed
        if self.use_multithreading and self.thread_pool is None:
            self.thread_pool = ThreadPoolExecutor(max_workers=self.optimal_thread_count)
        
        # Initialize process pool if needed
        if self.use_multiprocessing and self.process_pool is None:
            self.process_pool = ProcessPoolExecutor(max_workers=max(1, self.cpu_count - 1))
        
        # Choose appropriate executor
        executor = self.thread_pool if self.use_multithreading else None
        
        # Function to analyze a single column with timing
        def analyze_column_with_timing(col_name):
            start_time = time.time()
            
            # Get optimal sample of the column
            sample = self.get_optimal_sample(df[col_name], col_name)
            
            # Check cache first
            cache_key = f"{col_name}_{len(sample)}_{sample.dtype}"
            if cache_key in self.column_type_cache:
                result = self.column_type_cache[cache_key]
                logger.debug(f"Cache hit for column {col_name}")
            else:
                # Analyze the column
                result = analyzer_func(sample, **kwargs)
                
                # Cache the result
                self.column_type_cache[cache_key] = result
            
            end_time = time.time()
            self.performance_metrics['column_processing_times'][col_name] = end_time - start_time
            
            return col_name, result
        
        results = {}
        
        # Process columns in parallel if multithreading is enabled
        if self.use_multithreading:
            # Submit all tasks
            future_to_col = {
                executor.submit(analyze_column_with_timing, col): col 
                for col in df.columns
            }
            
            # Collect results as they complete
            for future in future_to_col:
                try:
                    col_name, result = future.result()
                    results[col_name] = result
                except Exception as e:
                    logger.error(f"Error analyzing column {future_to_col[future]}: {str(e)}")
        else:
            # Process columns sequentially
            for col in df.columns:
                try:
                    col_name, result = analyze_column_with_timing(col)
                    results[col_name] = result
                except Exception as e:
                    logger.error(f"Error analyzing column {col}: {str(e)}")
        
        total_end_time = time.time()
        self.performance_metrics['total_processing_time'] = total_end_time - total_start_time
        
        # Record current memory usage
        self.performance_metrics['memory_usage'].append(
            psutil.Process().memory_info().rss / (1024 * 1024)  # MB
        )
        
        return results
    
    def batch_process_large_dataframe(self, 
                                     df: pd.DataFrame, 
                                     processor_func: Callable, 
                                     batch_size: int = 100000, 
                                     **kwargs) -> pd.DataFrame:
        """
        Process a large DataFrame in batches to limit memory usage
        
        Args:
            df: Large DataFrame to process
            processor_func: Function to process each batch
            batch_size: Number of rows in each batch
            **kwargs: Additional arguments for processor_func
            
        Returns:
            Processed DataFrame
        """
        total_rows = len(df)
        
        if total_rows <= batch_size:
            # For small DataFrames, process directly
            return processor_func(df, **kwargs)
        
        # Calculate number of batches
        num_batches = (total_rows // batch_size) + (1 if total_rows % batch_size > 0 else 0)
        
        logger.info(f"Processing large DataFrame ({total_rows} rows) in {num_batches} batches")
        
        result_pieces = []
        
        # Process each batch
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_rows)
            
            logger.info(f"Processing batch {i+1}/{num_batches} (rows {start_idx}-{end_idx})")
            
            # Extract batch
            batch = df.iloc[start_idx:end_idx].copy()
            
            # Process batch
            processed_batch = processor_func(batch, **kwargs)
            
            # Store results
            result_pieces.append(processed_batch)
            
            # Explicit garbage collection to free memory
            del batch
            gc.collect()
        
        # Combine results
        return pd.concat(result_pieces, ignore_index=True)
    
    def estimate_memory_usage(self, df: pd.DataFrame, operation: str = 'default') -> float:
        """
        Estimate memory usage for an operation on a DataFrame
        
        Args:
            df: DataFrame to estimate for
            operation: Type of operation to estimate for
            
        Returns:
            Estimated memory usage in GB
        """
        base_size = df.memory_usage(deep=True).sum() / (1024 * 1024 * 1024)  # GB
        
        # Different operations have different memory multipliers
        multipliers = {
            'default': 2.0,      # General operations
            'transform': 3.0,    # Operations that create new dataframes
            'groupby': 2.5,      # GroupBy operations
            'join': 3.0,         # Join operations
            'string_processing': 4.0,  # String processing can be very memory intensive
            'sampling': 1.0      # Sampling generally uses less memory
        }
        
        multiplier = multipliers.get(operation, multipliers['default'])
        
        return base_size * multiplier
    
    def check_memory_safety(self, df: pd.DataFrame, operation: str = 'default') -> bool:
        """
        Check if an operation on a DataFrame is safe given memory constraints
        
        Args:
            df: DataFrame to check
            operation: Type of operation to check
            
        Returns:
            True if the operation is memory-safe, False otherwise
        """
        estimated_usage = self.estimate_memory_usage(df, operation)
        return estimated_usage < self.memory_limit
    
    @lru_cache(maxsize=1024)
    def optimized_pattern_match(self, pattern: str, text_sample: Tuple[str]) -> float:
        """
        Optimized pattern matching with caching
        
        Args:
            pattern: Regex pattern
            text_sample: Tuple of text samples (converted from list/series for hashability)
            
        Returns:
            Match ratio (0.0-1.0)
        """
        start_time = time.time()
        
        compiled_pattern = self.compile_regex(pattern)
        
        # Quick pre-check with a small sample
        quick_sample = text_sample[:min(5, len(text_sample))]
        quick_matches = sum(1 for text in quick_sample if compiled_pattern.search(text))
        
        # If no matches in quick sample, likely no matches overall
        if quick_matches == 0 and len(quick_sample) > 0:
            self.performance_metrics['skipped_operations'] += 1
            return 0.0
        
        # Full check
        match_count = sum(1 for text in text_sample if compiled_pattern.search(text))
        match_ratio = match_count / len(text_sample) if text_sample else 0.0
        
        # Record timing
        end_time = time.time()
        pattern_key = pattern[:20] + ('...' if len(pattern) > 20 else '')
        if pattern_key not in self.performance_metrics['pattern_match_times']:
            self.performance_metrics['pattern_match_times'][pattern_key] = []
        self.performance_metrics['pattern_match_times'][pattern_key].append(end_time - start_time)
        
        return match_ratio
    
    def optimized_value_set_match(self, reference_set: Tuple[str], value_sample: Tuple[str]) -> float:
        """
        Optimized value set matching with caching
        
        Args:
            reference_set: Tuple of reference values
            value_sample: Tuple of sample values to check
            
        Returns:
            Match ratio (0.0-1.0)
        """
        # Create hashable key for cache
        cache_key = (hash(reference_set), hash(value_sample))
        
        if cache_key in self.valueset_match_cache:
            return self.valueset_match_cache[cache_key]
        
        # Convert reference set to set for O(1) lookups
        ref_set = set(reference_set)
        
        # Count matches
        match_count = sum(1 for value in value_sample if value in ref_set)
        match_ratio = match_count / len(value_sample) if value_sample else 0.0
        
        # Cache result
        self.valueset_match_cache[cache_key] = match_ratio
        
        return match_ratio
    
    def adaptive_regex_strategy(self, pattern: str, text_series: pd.Series) -> float:
        """
        Use adaptive strategy for regex matching based on data size
        
        Args:
            pattern: Regex pattern
            text_series: Series of text values
            
        Returns:
            Match ratio (0.0-1.0)
        """
        # For very large series, use a small sample first
        if len(text_series) > 10000:
            small_sample = text_series.sample(1000).astype(str).values
            small_ratio = self.optimized_pattern_match(pattern, tuple(small_sample))
            
            # If clear result from small sample, return it
            if small_ratio > 0.9 or small_ratio < 0.1:
                return small_ratio
        
        # Get medium sample for more accurate result
        sample_size = min(len(text_series), self.max_sample_size)
        text_sample = text_series.sample(sample_size).astype(str).values
        
        return self.optimized_pattern_match(pattern, tuple(text_sample))
    
    def get_column_stats_fast(self, series: pd.Series, cache_key: str = None) -> Dict[str, Any]:
        """
        Calculate column statistics with optimized performance
        
        Args:
            series: Column data
            cache_key: Optional cache key
            
        Returns:
            Dict of statistics
        """
        # Check cache first
        if cache_key and cache_key in self.stats_cache:
            return self.stats_cache[cache_key]
        
        # Get sample for large series
        if len(series) > self.max_sample_size:
            sample = self.get_optimal_sample(series)
        else:
            sample = series
            
        # Basic stats calculation depends on dtype
        stats = {}
        
        # Generic stats for all types
        stats['count'] = len(series)
        stats['null_count'] = series.isna().sum()
        stats['null_percentage'] = stats['null_count'] / stats['count'] if stats['count'] > 0 else 0
        
        try:
            stats['unique_count'] = sample.nunique()
            stats['unique_percentage'] = stats['unique_count'] / len(sample) if len(sample) > 0 else 0
        except:
            # Fallback for columns where nunique() fails
            stats['unique_count'] = 0
            stats['unique_percentage'] = 0
        
        # Type-specific stats
        if pd.api.types.is_numeric_dtype(sample.dtype):
            try:
                non_null = sample.dropna()
                if len(non_null) > 0:
                    # Basic stats
                    stats['min'] = float(non_null.min())
                    stats['max'] = float(non_null.max())
                    stats['mean'] = float(non_null.mean())
                    stats['median'] = float(non_null.median())
                    stats['std'] = float(non_null.std())
                    
                    # Check if mostly integers
                    stats['is_mostly_integer'] = (non_null % 1 == 0).mean() > 0.9
                    
                    # Efficient histogram calculation
                    hist_values, hist_bins = np.histogram(non_null, bins=min(10, stats['unique_count']))
                    stats['histogram_values'] = hist_values.tolist()
                    stats['histogram_bins'] = hist_bins.tolist()
            except:
                # Fallback for numeric stats failures
                stats['min'] = 0
                stats['max'] = 0
                stats['mean'] = 0
                stats['median'] = 0
                stats['std'] = 0
        
        elif pd.api.types.is_string_dtype(sample.dtype) or pd.api.types.is_object_dtype(sample.dtype):
            try:
                # Only calculate string stats on a sample of strings
                string_sample = sample.dropna().astype(str)
                if len(string_sample) > 0:
                    # String length stats
                    lengths = string_sample.str.len()
                    stats['avg_length'] = float(lengths.mean())
                    stats['min_length'] = int(lengths.min())
                    stats['max_length'] = int(lengths.max())
                    stats['std_length'] = float(lengths.std())
                    
                    # Character composition stats
                    stats['contains_space_pct'] = (string_sample.str.contains(' ')).mean()
                    stats['contains_special_pct'] = (string_sample.str.contains('[^a-zA-Z0-9\\s]')).mean()
                    stats['contains_digit_pct'] = (string_sample.str.contains('[0-9]')).mean()
            except:
                # Fallback for string stats failures
                stats['avg_length'] = 0
                stats['min_length'] = 0
                stats['max_length'] = 0
                stats['std_length'] = 0
        
        elif pd.api.types.is_datetime64_dtype(sample.dtype):
            try:
                non_null = sample.dropna()
                if len(non_null) > 0:
                    # Date range stats
                    stats['min_date'] = non_null.min().strftime('%Y-%m-%d')
                    stats['max_date'] = non_null.max().strftime('%Y-%m-%d')
                    stats['date_range_days'] = (non_null.max() - non_null.min()).days
                    
                    # Check for time component
                    has_time = False
                    try:
                        time_values = non_null.dt.time
                        # Check if there are non-zero time values
                        if len(time_values) > 0 and (time_values.astype(str) != '00:00:00').any():
                            has_time = True
                    except:
                        pass
                    
                    stats['has_time_component'] = has_time
            except:
                # Fallback for datetime stats failures
                stats['min_date'] = ""
                stats['max_date'] = ""
                stats['date_range_days'] = 0
                stats['has_time_component'] = False
        
        # Cache results
        if cache_key:
            self.stats_cache[cache_key] = stats
        
        return stats
    
    def incremental_learning(self, df: pd.DataFrame, column_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Improve analysis results by incremental learning from previous results
        
        Args:
            df: Current DataFrame
            column_results: Current analysis results
            
        Returns:
            Improved column results
        """
        # Find patterns across similar columns
        similar_columns = defaultdict(list)
        
        # Group similar columns by type
        for col, result in column_results.items():
            col_type = result.get('type', 'unknown')
            similar_columns[col_type].append(col)
        
        # Look for common patterns in naming
        for col_type, cols in similar_columns.items():
            if len(cols) < 2:
                continue
                
            # Look for common prefixes/suffixes
            prefixes = defaultdict(int)
            suffixes = defaultdict(int)
            
            for col in cols:
                parts = re.split(r'[^a-zA-Z0-9]', col)
                if parts:
                    prefixes[parts[0].lower()] += 1
                    suffixes[parts[-1].lower()] += 1
            
            # Find dominant prefix/suffix
            if prefixes:
                dominant_prefix = max(prefixes.items(), key=lambda x: x[1])
                if dominant_prefix[1] >= len(cols) * 0.5:  # At least 50% have this prefix
                    for col in cols:
                        parts = re.split(r'[^a-zA-Z0-9]', col)
                        if parts and parts[0].lower() == dominant_prefix[0]:
                            # Increase confidence for this column
                            if 'confidence' in column_results[col]:
                                column_results[col]['confidence'] = min(
                                    1.0, column_results[col]['confidence'] + 0.1
                                )
            
            if suffixes:
                dominant_suffix = max(suffixes.items(), key=lambda x: x[1])
                if dominant_suffix[1] >= len(cols) * 0.5:  # At least 50% have this suffix
                    for col in cols:
                        parts = re.split(r'[^a-zA-Z0-9]', col)
                        if parts and parts[-1].lower() == dominant_suffix[0]:
                            # Increase confidence for this column
                            if 'confidence' in column_results[col]:
                                column_results[col]['confidence'] = min(
                                    1.0, column_results[col]['confidence'] + 0.1
                                )
        
        # Look for entity relationships
        self._identify_entity_relationships(df, column_results)
        
        return column_results
    
    def _identify_entity_relationships(self, df: pd.DataFrame, column_results: Dict[str, Any]) -> None:
        """
        Identify entity relationships between columns and improve confidence
        
        Args:
            df: DataFrame
            column_results: Column analysis results to update
        """
        # Find ID columns
        id_columns = [col for col, result in column_results.items() 
                     if 'id' in result.get('type', '').lower() and result.get('confidence', 0) > 0.5]
        
        # Find name columns
        name_columns = [col for col, result in column_results.items() 
                       if 'name' in result.get('type', '').lower() and result.get('confidence', 0) > 0.5]
        
        # Check for potential relationships
        for id_col in id_columns:
            for name_col in name_columns:
                if self._check_id_name_relationship(df, id_col, name_col):
                    # Update confidence for both columns
                    if 'confidence' in column_results[id_col]:
                        column_results[id_col]['confidence'] = min(
                            1.0, column_results[id_col]['confidence'] + 0.15
                        )
                        
                    if 'confidence' in column_results[name_col]:
                        column_results[name_col]['confidence'] = min(
                            1.0, column_results[name_col]['confidence'] + 0.15
                        )
    
    def _check_id_name_relationship(self, df: pd.DataFrame, id_col: str, name_col: str) -> bool:
        """
        Check if ID and name columns have a relationship
        
        Args:
            df: DataFrame
            id_col: ID column name
            name_col: Name column name
            
        Returns:
            True if they appear to be related
        """
        try:
            # Get sample if DataFrame is large
            if len(df) > 10000:
                df_sample = df.sample(10000)
            else:
                df_sample = df
                
            # Count how many unique id values map to a single name value
            mapping = df_sample.groupby(id_col)[name_col].nunique()
            # If most id values map to a single name value, they're likely related
            mostly_one_to_one = (mapping == 1).mean() > 0.8
            return mostly_one_to_one
        except:
            return False
    
    def cleanup_resources(self):
        """Clean up resources used by the performance optimizer"""
        # Close thread pool if it exists
        if self.thread_pool:
            self.thread_pool.shutdown()
            self.thread_pool = None
        
        # Close process pool if it exists
        if self.process_pool:
            self.process_pool.shutdown()
            self.process_pool = None
        
        # Clear caches
        self.pattern_cache.clear()
        self.column_type_cache.clear()
        self.stats_cache.clear()
        self.string_pattern_cache.clear()
        self.valueset_match_cache.clear()
        
        # Force garbage collection
        gc.collect()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance metrics
        
        Returns:
            Dict with performance summary
        """
        summary = {
            'total_time': self.performance_metrics['total_processing_time'],
            'column_count': len(self.performance_metrics['column_processing_times']),
            'avg_column_time': 0,
            'max_column_time': 0,
            'slowest_column': None,
            'peak_memory_mb': max(self.performance_metrics['memory_usage']) if self.performance_metrics['memory_usage'] else 0,
            'skipped_operations': self.performance_metrics['skipped_operations'],
            'cache_hit_percentage': 0,
            'avg_sampling_ratio': 0
        }
        
        # Calculate average and max column processing times
        if self.performance_metrics['column_processing_times']:
            summary['avg_column_time'] = sum(self.performance_metrics['column_processing_times'].values()) / len(self.performance_metrics['column_processing_times'])
            max_time = max(self.performance_metrics['column_processing_times'].items(), key=lambda x: x[1])
            summary['max_column_time'] = max_time[1]
            summary['slowest_column'] = max_time[0]
        
        # Calculate average sampling ratio
        if self.performance_metrics['sampling_ratios']:
            summary['avg_sampling_ratio'] = sum(self.performance_metrics['sampling_ratios'].values()) / len(self.performance_metrics['sampling_ratios'])
        
        return summary

class OptimizedColumnIdentifier:
    """
    A performance-optimized column identifier that applies high-performance techniques
    to identify column types in large datasets.
    """
    
    def __init__(self, 
                 max_sample_size: int = 10000,
                 use_multithreading: bool = True,
                 use_multiprocessing: bool = False,
                 enable_incremental_learning: bool = True,
                 enable_fast_exit: bool = True):
        """
        Initialize the optimized column identifier
        
        Args:
            max_sample_size: Maximum sample size for column analysis
            use_multithreading: Enable multithreaded processing
            use_multiprocessing: Enable multiprocessing for heavy computations
            enable_incremental_learning: Enable incremental learning from previous results
            enable_fast_exit: Enable early stopping when high confidence is reached
        """
        # Create performance optimizer
        self.optimizer = PerformanceOptimizer(
            max_sample_size=max_sample_size,
            use_multithreading=use_multithreading,
            use_multiprocessing=use_multiprocessing,
            enable_fast_exit=enable_fast_exit
        )
        
        self.enable_incremental_learning = enable_incremental_learning
        
        # Pre-compile frequently used regex patterns
        self._precompile_common_patterns()
        
        # Cache recently analyzed columns
        self.column_cache = {}
    
    def _precompile_common_patterns(self):
        """Pre-compile frequently used regex patterns for performance"""
        common_patterns = [
            r'^\d+$',                                    # Integer
            r'^\d+\.\d+$',                               # Float
            r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',  # Email
            r'^https?://(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(?:/[^\s]*)?$',  # URL
            r'^\+?[\d\s-\(\).]{7,}$',                    # Phone
            r'^\d{5}(?:-\d{4})?$',                       # ZIP code
            r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$',          # IP address
            r'^(?:[0-9a-f]{2}[:-]){5}[0-9a-f]{2}$',      # MAC address
            r'^[A-Z]{1,3}-?\d{3,6}$',                    # Product code
            r'^\d{4}-\d{2}-\d{2}$',                      # ISO date
            r'^\d{1,2}/\d{1,2}/\d{4}$',                  # US date
            r'^\d{1,2}/\d{1,2}/\d{2}$',                  # Short US date
            r'^\d{1,2}:\d{2}(?::\d{2})?$',               # Time
            r'^\$?\s?[+-]?[0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]{1,2})?$'  # Currency
        ]
        
        for pattern in common_patterns:
            self.optimizer.compile_regex(pattern)
    
    def identify_column_types(self, df: pd.DataFrame, verbose: bool = False) -> Dict[str, Dict[str, Any]]:
        """
        Identify column types with optimized performance
        
        Args:
            df: DataFrame to analyze
            verbose: Whether to print detailed progress
            
        Returns:
            Dict mapping column names to their type information
        """
        start_time = time.time()
        
        if verbose:
            logger.info(f"Starting column type identification for {len(df.columns)} columns in DataFrame with {len(df)} rows")
        
        # Optimize DataFrame if it's large
        if len(df) > 10000 or df.memory_usage(deep=True).sum() > 100 * 1024 * 1024:  # > 100MB
            if verbose:
                logger.info("Optimizing DataFrame for performance")
            df_optimized = self.optimizer.optimize_dataframe(df)
        else:
            df_optimized = df
        
        # Use parallel processing for column analysis
        if verbose:
            logger.info("Analyzing columns in parallel")
            
        column_results = self.optimizer.analyze_columns_parallel(df_optimized, self._analyze_single_column)
        
        # Apply incremental learning if enabled
        if self.enable_incremental_learning:
            if verbose:
                logger.info("Applying incremental learning to improve results")
            column_results = self.optimizer.incremental_learning(df_optimized, column_results)
        
        end_time = time.time()
        
        if verbose:
            # Get performance summary
            summary = self.optimizer.get_performance_summary()
            logger.info(f"Column identification completed in {end_time - start_time:.2f} seconds")
            logger.info(f"Average column processing time: {summary['avg_column_time']:.4f} seconds")
            logger.info(f"Peak memory usage: {summary['peak_memory_mb']:.2f} MB")
            logger.info(f"Operations skipped by optimization: {summary['skipped_operations']}")
        
        return column_results
    
    def _analyze_single_column(self, series: pd.Series) -> Dict[str, Any]:
        """
        Analyze a single column with optimized performance
        
        Args:
            series: Column data
            
        Returns:
            Dict with column type information
        """
        # Create cache key
        cache_key = f"{series.name}_{len(series)}_{series.dtype}_{hash(str(series.head(5)))}"
        
        # Check cache first
        if cache_key in self.column_cache:
            return self.column_cache[cache_key]
        
        # Get column statistics
        stats = self.optimizer.get_column_stats_fast(series, cache_key)
        
        # Determine base type
        if pd.api.types.is_numeric_dtype(series.dtype):
            base_type = "numeric"
        elif pd.api.types.is_datetime64_dtype(series.dtype):
            base_type = "datetime"
        elif pd.api.types.is_bool_dtype(series.dtype):
            base_type = "boolean"
        else:
            base_type = "string"
        
        # Initialize result
        result = {
            "type": base_type,
            "confidence": 0.7,  # Default confidence
            "subtypes": {},
            "statistics": stats
        }
        
        # Perform type-specific analysis
        if base_type == "numeric":
            result = self._analyze_numeric_column(series, stats, result)
        elif base_type == "string":
            result = self._analyze_string_column(series, stats, result)
        elif base_type == "datetime":
            result = self._analyze_datetime_column(series, stats, result)
        elif base_type == "boolean":
            result = self._analyze_boolean_column(series, stats, result)
        
        # Cache result
        self.column_cache[cache_key] = result
        
        return result
    
    def _analyze_numeric_column(self, series: pd.Series, stats: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a numeric column
        
        Args:
            series: Column data
            stats: Pre-computed statistics
            result: Initial result dict to update
            
        Returns:
            Updated result dict
        """
        # Fast path: If series has all integers between 0 and 1, likely boolean
        if stats.get('unique_count') == 2 and stats.get('min') == 0 and stats.get('max') == 1:
            result['type'] = 'boolean'
            result['subtypes']['flag'] = 0.9
            result['confidence'] = 0.9
            return result
        
        # Check for common numeric subtypes
        subtypes = {}
        
        # ID check - high unique ratio, integers, positive
        if (stats.get('is_mostly_integer', False) and 
            stats.get('unique_percentage', 0) > 0.8 and 
            stats.get('min', 0) >= 0):
            subtypes['id'] = 0.8
        
        # Year check
        if (stats.get('is_mostly_integer', False) and 
            1900 <= stats.get('min', 0) <= stats.get('max', 3000) <= datetime.now().year + 10):
            subtypes['year'] = 0.9
        
        # Month check
        if (stats.get('is_mostly_integer', False) and 
            1 <= stats.get('min', 0) <= stats.get('max', 13) <= 12):
            subtypes['month'] = 0.9
        
        # Day check
        if (stats.get('is_mostly_integer', False) and 
            1 <= stats.get('min', 0) <= stats.get('max', 32) <= 31):
            subtypes['day'] = 0.8
        
        # Percentage check
        if 0 <= stats.get('min', -1) <= stats.get('max', 2) <= 1 and not stats.get('is_mostly_integer', True):
            subtypes['percentage'] = 0.9
        elif 0 <= stats.get('min', -1) <= stats.get('max', 101) <= 100:
            subtypes['percentage'] = 0.7
        
        # Count/quantity check - positive integers with moderate mean
        if (stats.get('is_mostly_integer', False) and 
            stats.get('min', -1) >= 0 and 
            stats.get('mean', 1000) < 100):
            subtypes['count'] = 0.7
            subtypes['quantity'] = 0.7
        
        # Price/monetary check - positive with decimals
        if (stats.get('min', -1) >= 0 and 
            not stats.get('is_mostly_integer', True) and 
            stats.get('mean', 0) > 0):
            subtypes['price'] = 0.7
            subtypes['monetary'] = 0.7
        
        # Fast exit if we have confident subtype match
        if self.optimizer.enable_fast_exit:
            high_conf_subtype = next((k for k, v in subtypes.items() if v > 0.85), None)
            if high_conf_subtype:
                result['type'] = high_conf_subtype
                result['subtypes'] = {high_conf_subtype: subtypes[high_conf_subtype]}
                result['confidence'] = subtypes[high_conf_subtype]
                return result
        
        # Update result with all subtypes
        result['subtypes'] = subtypes
        
        # Set primary type to the highest confidence subtype
        if subtypes:
            best_subtype = max(subtypes.items(), key=lambda x: x[1])
            result['type'] = best_subtype[0]
            result['confidence'] = best_subtype[1]
        
        return result
    
    def _analyze_string_column(self, series: pd.Series, stats: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a string column
        
        Args:
            series: Column data
            stats: Pre-computed statistics
            result: Initial result dict to update
            
        Returns:
            Updated result dict
        """
        # Fast path: If all values are "true"/"false", likely boolean
        if stats.get('unique_count') == 2:
            # Sample values for quick check
            sample = series.dropna().astype(str).sample(min(len(series.dropna()), 100))
            lower_sample = sample.str.lower()
            if set(lower_sample.unique()) == {'true', 'false'} or set(lower_sample.unique()) == {'yes', 'no'}:
                result['type'] = 'boolean'
                result['subtypes']['flag'] = 0.9
                result['confidence'] = 0.9
                return result
        
        # Take a sample for pattern matching
        sample_size = min(1000, len(series.dropna()))
        if sample_size == 0:
            # No non-null values to analyze
            return result
            
        sample = series.dropna().sample(sample_size).astype(str).values
        sample_tuple = tuple(sample)  # Convert to tuple for caching
        
        # Check for common string patterns with fast exit
        pattern_checks = [
            ('email', r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'),
            ('url', r'^https?://(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*(?:/[^\s]*)?$'),
            ('phone', r'^\+?[\d\s-\(\).]{7,}$'),
            ('zip_code', r'^\d{5}(?:-\d{4})?$'),
            ('ip_address', r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$'),
            ('date_iso', r'^\d{4}-\d{2}-\d{2}$'),
            ('date_us', r'^\d{1,2}/\d{1,2}/\d{4}$'),
            ('date_short', r'^\d{1,2}/\d{1,2}/\d{2}$'),
            ('time', r'^\d{1,2}:\d{2}(?::\d{2})?$'),
            ('currency', r'^\$?\s?[+-]?[0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]{1,2})?$'),
            ('product_code', r'^[A-Z]{1,3}-?\d{3,6}$')
        ]
        
        # Check each pattern with fast exit on high confidence match
        subtypes = {}
        for subtype, pattern in pattern_checks:
            match_ratio = self.optimizer.optimized_pattern_match(pattern, sample_tuple)
            if match_ratio > 0.7:
                subtypes[subtype] = match_ratio
                
                # Fast exit if we have a very confident match
                if match_ratio > 0.9 and self.optimizer.enable_fast_exit:
                    result['type'] = subtype
                    result['subtypes'] = {subtype: match_ratio}
                    result['confidence'] = match_ratio
                    return result
        
        # Special case checks based on statistics
        
        # Name detection - titlecase, moderate length
        if (stats.get('contains_space_pct', 0) > 0.8 and 
            10 <= stats.get('avg_length', 0) <= 40):
            subtypes['name'] = 0.7
        
        # Text content - longer with spaces
        if (stats.get('contains_space_pct', 0) > 0.8 and 
            stats.get('avg_length', 0) > 40):
            subtypes['text_content'] = 0.8
        
        # ID detection - alphanumeric with special chars, moderate length
        if (stats.get('contains_special_pct', 0) > 0 and 
            stats.get('contains_digit_pct', 0) > 0 and 
            5 <= stats.get('avg_length', 0) <= 20):
            subtypes['id_code'] = 0.7
        
        # Update result with all subtypes
        result['subtypes'] = subtypes
        
        # Set primary type to the highest confidence subtype
        if subtypes:
            best_subtype = max(subtypes.items(), key=lambda x: x[1])
            result['type'] = best_subtype[0]
            result['confidence'] = best_subtype[1]
        else:
            # Default to general text if no specific subtype found
            result['type'] = 'text'
            result['confidence'] = 0.5
        
        return result
    
    def _analyze_datetime_column(self, series: pd.Series, stats: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a datetime column
        
        Args:
            series: Column data
            stats: Pre-computed statistics
            result: Initial result dict to update
            
        Returns:
            Updated result dict
        """
        subtypes = {}
        
        # Datetime vs date only
        if stats.get('has_time_component', False):
            subtypes['timestamp'] = 0.9
        else:
            subtypes['date'] = 0.9
        
        # Check for recent vs historical dates
        current_year = datetime.now().year
        min_date_str = stats.get('min_date', '')
        max_date_str = stats.get('max_date', '')
        
        try:
            min_year = int(min_date_str.split('-')[0])
            max_year = int(max_date_str.split('-')[0])
            
            if min_year > current_year - 5:
                # Recent dates (last 5 years)
                subtypes['recent_date'] = 0.8
                
                # Check for future dates
                if max_year > current_year:
                    subtypes['future_date'] = 0.8
                    subtypes['scheduled_date'] = 0.7
            elif max_year < current_year - 5:
                # Historical dates (more than 5 years ago)
                subtypes['historical_date'] = 0.8
                
                # Could be birth date if in reasonable range
                if current_year - 100 < min_year < current_year - 10:
                    subtypes['birth_date'] = 0.7
        except:
            pass
        
        # Update result with subtypes
        result['subtypes'] = subtypes
        
        # Set primary type to the highest confidence subtype
        if subtypes:
            best_subtype = max(subtypes.items(), key=lambda x: x[1])
            result['type'] = best_subtype[0]
            result['confidence'] = best_subtype[1]
        
        return result
    
    def _analyze_boolean_column(self, series: pd.Series, stats: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a boolean column
        
        Args:
            series: Column data
            stats: Pre-computed statistics
            result: Initial result dict to update
            
        Returns:
            Updated result dict
        """
        subtypes = {
            'boolean': 0.9,
            'flag': 0.8,
            'indicator': 0.8
        }
        
        # Check true ratio for specific types
        true_ratio = stats.get('true_ratio', 0.5)
        
        if 0.05 <= true_ratio <= 0.15:
            # Low true ratio suggests rare events
            subtypes['rare_event'] = 0.7
        elif 0.85 <= true_ratio <= 0.95:
            # High true ratio suggests common events
            subtypes['common_event'] = 0.7
        
        # Update result with subtypes
        result['subtypes'] = subtypes
        
        # Set primary type to the highest confidence subtype
        best_subtype = max(subtypes.items(), key=lambda x: x[1])
        result['type'] = best_subtype[0]
        result['confidence'] = best_subtype[1]
        
        return result
        
    def cleanup(self):
        """Clean up resources used by the column identifier"""
        self.optimizer.cleanup_resources()
        self.column_cache.clear()

# Example usage
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Generate a test DataFrame
    def generate_test_data(rows=100000, cols=20):
        """Generate test data with various column types"""
        import numpy as np
        from datetime import datetime, timedelta
        
        # Start with a base date
        base_date = datetime(2020, 1, 1)
        
        data = {
            # ID columns
            'customer_id': [f'CUST-{i:06d}' for i in range(rows)],
            'order_id': np.arange(1, rows + 1),
            
            # Date columns
            'order_date': [base_date + timedelta(days=i % 365) for i in range(rows)],
            'delivery_date': [base_date + timedelta(days=(i % 365) + np.random.randint(1, 7)) for i in range(rows)],
            
            # Numeric columns
            'quantity': np.random.randint(1, 10, rows),
            'unit_price': np.random.uniform(10, 200, rows).round(2),
            'total_price': np.zeros(rows),  # Will calculate this
            'discount_pct': np.random.uniform(0, 0.3, rows).round(2),
            
            # Categorical columns
            'product_category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Sports'], rows),
            'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Bank Transfer', 'Cash'], rows),
            'shipping_method': np.random.choice(['Standard', 'Express', 'Next Day'], rows),
            
            # Boolean columns
            'is_new_customer': np.random.choice([True, False], rows, p=[0.3, 0.7]),
            'has_coupon': np.random.choice([True, False], rows, p=[0.15, 0.85]),
            'is_gift': np.random.choice([True, False], rows, p=[0.05, 0.95]),
            
            # Text columns
            'email': [f'customer{i}@example.com' for i in range(rows)],
            'shipping_address': [f'{np.random.randint(100, 999)} Main St, City-{i % 100}' for i in range(rows)],
            'notes': [f'Order note {i}' if i % 10 == 0 else '' for i in range(rows)],
            
            # Additional columns for larger datasets
            'zipcode': [f'{np.random.randint(10000, 99999)}' for i in range(rows)],
            'phone': [f'+1-{np.random.randint(100, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}' for i in range(rows)],
            'rating': np.random.uniform(1, 5, rows).round(1)
        }
        
        # Calculate total_price
        data['total_price'] = data['quantity'] * data['unit_price'] * (1 - data['discount_pct'])
        
        return pd.DataFrame(data)
    
    print("Generating test data...")
    df = generate_test_data(rows=100000, cols=20)
    
    print(f"Test DataFrame created: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Create optimized column identifier
    identifier = OptimizedColumnIdentifier(
        max_sample_size=5000,
        use_multithreading=True,
        enable_incremental_learning=True
    )
    
    print("Starting column identification...")
    start_time = time.time()
    
    results = identifier.identify_column_types(df, verbose=True)
    
    end_time = time.time()
    print(f"Column identification completed in {end_time - start_time:.2f} seconds")
    
    # Show results
    print("\nColumn Type Identification Results:")
    for col, result in results.items():
        print(f"\n{col}:")
        print(f"  Type: {result['type']} (Confidence: {result['confidence']:.2f})")
        if 'subtypes' in result and result['subtypes']:
            print("  Subtypes:")
            for subtype, confidence in sorted(result['subtypes'].items(), key=lambda x: x[1], reverse=True):
                print(f"    - {subtype}: {confidence:.2f}")
    
    # Clean up resources
    identifier.cleanup()
