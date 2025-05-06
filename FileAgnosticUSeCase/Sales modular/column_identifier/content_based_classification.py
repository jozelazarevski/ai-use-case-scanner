"""
Content-Based Classification for Column Identification

This module provides advanced content-based classification methods to identify column types
based on the actual values in the data, using statistical analysis, pattern recognition, 
and machine learning techniques.
"""

import numpy as np
import pandas as pd
import re
from datetime import datetime
from collections import Counter
import string
import math
from typing import Dict, List, Tuple, Union, Optional, Any, Set
import json

class ContentClassifier:
    """
    Advanced content-based classifier for identifying column types based on values
    rather than just column names.
    """
    
    def __init__(self):
        """Initialize the content classifier"""
        # Load reference data
        self._initialize_reference_data()
    
    def _initialize_reference_data(self):
        """Initialize reference datasets for content classification"""
        # Common value dictionaries
        self.reference_data = {
            # Common status values
            'status_values': {
                'active', 'inactive', 'pending', 'completed', 'cancelled', 'on hold', 
                'approved', 'rejected', 'in process', 'failed', 'shipped', 'delivered',
                'returned', 'refunded', 'new', 'open', 'closed', 'suspended', 'expired',
                'draft', 'published', 'archived', 'enabled', 'disabled', 'verified',
                'unverified', 'paid', 'unpaid', 'overdue', 'processing', 'error'
            },
            
            # Common category terms
            'category_terms': {
                'electronics', 'clothing', 'furniture', 'books', 'toys', 'sports',
                'automotive', 'beauty', 'health', 'food', 'grocery', 'office', 'pet',
                'garden', 'home', 'kitchen', 'tools', 'jewelry', 'outdoors', 'music',
                'movies', 'games', 'baby', 'kids', 'travel', 'digital', 'services'
            },
            
            # Common units of measurement
            'units': {
                # Length
                'mm', 'cm', 'dm', 'm', 'km', 'in', 'ft', 'yd', 'mi',
                'millimeter', 'centimeter', 'meter', 'kilometer', 'inch', 'foot', 'feet', 'yard', 'mile',
                
                # Weight/Mass
                'mg', 'g', 'kg', 'oz', 'lb', 'ton',
                'milligram', 'gram', 'kilogram', 'ounce', 'pound',
                
                # Volume
                'ml', 'l', 'cl', 'dl', 'pt', 'qt', 'gal', 'fl oz', 
                'milliliter', 'liter', 'centiliter', 'deciliter', 'pint', 'quart', 'gallon',
                
                # Time
                'ms', 's', 'min', 'h', 'hr', 'day', 'wk', 'mo', 'yr',
                'millisecond', 'second', 'minute', 'hour', 'day', 'week', 'month', 'year',
                
                # Area
                'sq ft', 'sq m', 'ac', 'ha',
                'square foot', 'square meter', 'acre', 'hectare',
                
                # Digital
                'b', 'kb', 'mb', 'gb', 'tb', 'pb',
                'byte', 'kilobyte', 'megabyte', 'gigabyte', 'terabyte', 'petabyte',
                
                # Currency symbols
                '$', '€', '£', '¥', '₹', '₽', '₩', 'CHF', 'A$', 'C$', '¢'
            },
            
            # Common currency codes
            'currency_codes': {
                'USD', 'EUR', 'GBP', 'JPY', 'CNY', 'AUD', 'CAD', 'CHF', 
                'HKD', 'NZD', 'SEK', 'KRW', 'SGD', 'NOK', 'MXN', 'INR', 
                'RUB', 'ZAR', 'BRL', 'TRY'
            },
            
            # ISO country codes (2-letter)
            'country_codes': {
                'US', 'CA', 'GB', 'DE', 'FR', 'JP', 'CN', 'AU', 'IT', 'ES',
                'NL', 'BE', 'SE', 'NO', 'DK', 'FI', 'PT', 'GR', 'IE', 'AT',
                'CH', 'IN', 'BR', 'MX', 'AR', 'ZA', 'RU', 'AE', 'SA', 'IL'
            },
            
            # US state codes
            'us_state_codes': {
                'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
                'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
                'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
                'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
                'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY'
            },
            
            # Common city names
            'city_samples': {
                'New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix',
                'Philadelphia', 'San Antonio', 'San Diego', 'Dallas', 'San Jose',
                'London', 'Paris', 'Tokyo', 'Berlin', 'Rome', 'Madrid', 'Toronto',
                'Sydney', 'Singapore', 'Mumbai', 'Shanghai', 'Beijing', 'Istanbul'
            },
            
            # Common first names
            'first_name_samples': {
                'James', 'John', 'Robert', 'Michael', 'William', 'David', 'Richard',
                'Mary', 'Patricia', 'Jennifer', 'Linda', 'Elizabeth', 'Barbara', 'Susan',
                'Joseph', 'Thomas', 'Charles', 'Christopher', 'Daniel', 'Matthew',
                'Sarah', 'Jessica', 'Karen', 'Nancy', 'Lisa', 'Margaret', 'Sandra',
                'Emma', 'Olivia', 'Noah', 'Liam', 'Sophia', 'Ava', 'Isabella'
            },
            
            # Common last names
            'last_name_samples': {
                'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
                'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
                'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin',
                'Lee', 'Perez', 'Thompson', 'White', 'Harris', 'Sanchez', 'Clark',
                'Lewis', 'Robinson', 'Walker', 'Young', 'Allen', 'King', 'Wright'
            },
            
            # Common product attributes
            'product_attributes': {
                'color', 'size', 'weight', 'height', 'width', 'depth', 'material',
                'brand', 'manufacturer', 'model', 'style', 'type', 'price', 'cost',
                'quantity', 'stock', 'inventory', 'sku', 'upc', 'ean', 'isbn',
                'description', 'features', 'specifications', 'dimensions', 'packaging'
            },
            
            # Common shipping methods
            'shipping_methods': {
                'standard', 'express', 'overnight', 'next day', 'two-day', '2-day',
                'ground', 'air', 'freight', 'pickup', 'local delivery', 'international',
                'economy', 'premium', 'expedited', 'priority', 'usps', 'ups', 'fedex',
                'dhl', 'royal mail', 'auspost', 'canada post', 'courier', 'free'
            },
            
            # Common payment methods
            'payment_methods': {
                'credit card', 'debit card', 'paypal', 'bank transfer', 'check',
                'cash', 'money order', 'wire transfer', 'bitcoin', 'cryptocurrency',
                'apple pay', 'google pay', 'amazon pay', 'venmo', 'zelle', 'affirm',
                'afterpay', 'klarna', 'visa', 'mastercard', 'amex', 'discover'
            }
        }
        
        # Regular expression patterns for common data formats
        self.regex_patterns = {
            'email': r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$',
            'url': r'^(https?://)?([a-zA-Z0-9]([a-zA-Z0-9\-]*[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}(/[^\s]*)?$',
            'ip_address': r'^(?:[0-9]{1,3}\.){3}[0-9]{1,3}$',
            'phone_us': r'^(\+?1[-\s.]?)?(\([0-9]{3}\)|[0-9]{3})[-\s.]?[0-9]{3}[-\s.]?[0-9]{4}$',
            'phone_simple': r'^\+?[0-9\-\(\)\s\.]{7,20}$',
            'zip_us': r'^\d{5}(-\d{4})?$',
            'ssn': r'^\d{3}-\d{2}-\d{4}$',
            'credit_card': r'^(?:\d{4}[ -]?){3}\d{4}|\d{16}$',
            'isbn': r'^(?:ISBN(?:-1[03])?:? )?(?=[0-9X]{10}$|(?=(?:[0-9]+[- ]){3})[- 0-9X]{13}$|97[89][0-9]{10}$|(?=(?:[0-9]+[- ]){4})[- 0-9]{17}$)(?:97[89][- ]?)?[0-9]{1,5}[- ]?[0-9]+[- ]?[0-9]+[- ]?[0-9X]$',
            'uuid': r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$',
            'sha1': r'^[0-9a-f]{40}$',
            'sha256': r'^[0-9a-f]{64}$',
            'md5': r'^[0-9a-f]{32}$',
            'ipv6': r'^([0-9a-f]{1,4}:){7}[0-9a-f]{1,4}$',
            'mac_address': r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$',
            'date_ymd': r'^([12]\d{3}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01]))$',
            'date_mdy': r'^(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/([12]\d{3})$',
            'date_dmy': r'^(0[1-9]|[12]\d|3[01])/(0[1-9]|1[0-2])/([12]\d{3})$',
            'time': r'^([01]?[0-9]|2[0-3]):[0-5][0-9](:[0-5][0-9])?$',
            'currency': r'^\$?\s?[+-]?[0-9]{1,3}(?:,?[0-9]{3})*(?:\.[0-9]{1,2})?$',
            'percentage': r'^[+-]?[0-9]+\.?[0-9]*\s*%$',
            'hex_color': r'^#[0-9A-Fa-f]{6}$',
            'product_code': r'^[A-Z]{1,5}[-_]?[0-9]{1,10}$'
        }
    
    def analyze_column(self, series: pd.Series, sample_size: int = 1000) -> Dict[str, Any]:
        """
        Analyze a column to determine its likely content type and characteristics
        
        Args:
            series (Series): The pandas Series to analyze
            sample_size (int): Number of samples to use for analysis
            
        Returns:
            dict: Column analysis results including type predictions and statistics
        """
        # Drop nulls and sample if necessary
        clean_series = series.dropna()
        if len(clean_series) > sample_size and sample_size > 0:
            sample = clean_series.sample(sample_size)
        else:
            sample = clean_series
        
        # If no data, return minimal analysis
        if len(sample) == 0:
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'subtypes': {},
                'statistics': {'count': 0, 'nulls': len(series)}
            }
        
        # Get common type characteristics
        dtype_analysis = self._analyze_dtype(sample)
        
        # Perform different analyses based on determined basic type
        if dtype_analysis['base_type'] == 'numeric':
            return self._analyze_numeric_column(sample, series, dtype_analysis)
        elif dtype_analysis['base_type'] == 'string':
            return self._analyze_string_column(sample, series, dtype_analysis)
        elif dtype_analysis['base_type'] == 'datetime':
            return self._analyze_datetime_column(sample, series, dtype_analysis)
        elif dtype_analysis['base_type'] == 'boolean':
            return self._analyze_boolean_column(sample, series, dtype_analysis)
        else:
            # Fallback for unknown types
            return {
                'type': 'unknown',
                'confidence': 0.0,
                'subtypes': {},
                'statistics': {
                    'count': len(clean_series),
                    'nulls': len(series) - len(clean_series),
                    'null_percentage': (len(series) - len(clean_series)) / len(series) if len(series) > 0 else 0,
                    'unique_count': clean_series.nunique(),
                    'unique_percentage': clean_series.nunique() / len(clean_series) if len(clean_series) > 0 else 0
                }
            }
    
    def _analyze_dtype(self, sample: pd.Series) -> Dict[str, Any]:
        """
        Analyze the data type of a sample Series
        
        Args:
            sample (Series): Sample data to analyze
            
        Returns:
            dict: Data type analysis results
        """
        dtype = sample.dtype
        
        # Check basic type
        base_type = None
        specific_type = None
        confidence = 0.5  # Default medium confidence
        
        if pd.api.types.is_numeric_dtype(dtype):
            base_type = 'numeric'
            if pd.api.types.is_integer_dtype(dtype):
                specific_type = 'integer'
                confidence = 0.9
            elif pd.api.types.is_float_dtype(dtype):
                specific_type = 'float'
                confidence = 0.9
            else:
                specific_type = 'other_numeric'
                confidence = 0.7
        elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            # Check if it's actually string content
            try:
                # Try to convert a sample to string to check
                str_sample = sample.astype(str)
                if not str_sample.empty:
                    base_type = 'string'
                    specific_type = 'text'
                    confidence = 0.8
            except:
                base_type = 'object'
                specific_type = 'mixed'
                confidence = 0.5
        elif pd.api.types.is_datetime64_dtype(dtype):
            base_type = 'datetime'
            specific_type = 'datetime'
            confidence = 0.9
        elif pd.api.types.is_bool_dtype(dtype):
            base_type = 'boolean'
            specific_type = 'boolean'
            confidence = 0.9
        elif pd.api.types.is_categorical_dtype(dtype):
            base_type = 'categorical'
            specific_type = 'categorical'
            confidence = 0.8
        else:
            base_type = 'unknown'
            specific_type = 'unknown'
            confidence = 0.3
        
        return {
            'base_type': base_type,
            'specific_type': specific_type,
            'confidence': confidence
        }
    
    def _analyze_numeric_column(self, sample: pd.Series, full_series: pd.Series, dtype_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a numeric column to determine its specific characteristics and likely content type
        
        Args:
            sample (Series): Sample data to analyze
            full_series (Series): The full original series
            dtype_analysis (dict): Results from dtype analysis
            
        Returns:
            dict: Numeric column analysis results
        """
        # Get basic statistics
        stats = {
            'count': len(full_series.dropna()),
            'nulls': full_series.isna().sum(),
            'null_percentage': full_series.isna().sum() / len(full_series) if len(full_series) > 0 else 0,
            'min': float(sample.min()),
            'max': float(sample.max()),
            'mean': float(sample.mean()),
            'median': float(sample.median()),
            'std': float(sample.std()),
            'unique_count': sample.nunique(),
            'unique_percentage': sample.nunique() / len(sample) if len(sample) > 0 else 0
        }
        
        # Check if it's mostly integers
        is_mostly_int = (sample % 1 == 0).mean() > 0.9
        
        # Initialize type prediction
        type_prediction = {
            'type': 'numeric',
            'confidence': dtype_analysis['confidence'],
            'subtypes': {}
        }
        
        # Check for various numeric subtypes
        
        # 1. Check for IDs
        if is_mostly_int and stats['unique_percentage'] > 0.9 and stats['min'] >= 0:
            type_prediction['subtypes']['id'] = 0.8
        
        # 2. Check for years
        if is_mostly_int and 1900 <= stats['min'] <= stats['max'] <= datetime.now().year + 10:
            type_prediction['subtypes']['year'] = 0.9
        
        # 3. Check for days/months
        if is_mostly_int:
            if 1 <= stats['min'] <= stats['max'] <= 31:
                type_prediction['subtypes']['day'] = 0.8
            if 1 <= stats['min'] <= stats['max'] <= 12:
                type_prediction['subtypes']['month'] = 0.9
        
        # 4. Check for percentages
        if 0 <= stats['min'] <= stats['max'] <= 1 and not is_mostly_int:
            type_prediction['subtypes']['percentage'] = 0.9
        elif 0 <= stats['min'] <= stats['max'] <= 100:
            type_prediction['subtypes']['percentage'] = 0.7
        
        # 5. Check for ratings
        if 1 <= stats['min'] <= stats['max'] <= 5 and stats['mean'] >= 1:
            type_prediction['subtypes']['rating'] = 0.8
        elif 0 <= stats['min'] <= stats['max'] <= 10 and stats['mean'] >= 1:
            type_prediction['subtypes']['rating'] = 0.7
        
        # 6. Check for counts/quantities
        if is_mostly_int and stats['min'] >= 0 and stats['mean'] < 100:
            type_prediction['subtypes']['count'] = 0.7
            type_prediction['subtypes']['quantity'] = 0.7
        
        # 7. Check for monetary amounts
        if stats['min'] >= 0 and stats['mean'] > 10 and stats['std'] > 5:
            type_prediction['subtypes']['monetary'] = 0.7
            type_prediction['subtypes']['price'] = 0.6
        
        # 8. Check for durations/measures
        if stats['min'] >= 0:
            if stats['mean'] < 60 and stats['max'] < 60:  # Seconds, maybe
                type_prediction['subtypes']['duration_seconds'] = 0.6
            elif stats['mean'] < 24 and stats['max'] < 24:  # Hours, maybe
                type_prediction['subtypes']['duration_hours'] = 0.6
            elif stats['mean'] < 100 and stats['max'] < 1000:  # Generic measure, maybe
                type_prediction['subtypes']['measurement'] = 0.5
        
        # Determine primary type based on highest confidence subtype
        if type_prediction['subtypes']:
            primary_subtype = max(type_prediction['subtypes'].items(), key=lambda x: x[1])
            type_prediction['type'] = primary_subtype[0]
            type_prediction['confidence'] = primary_subtype[1]
        else:
            # Default to generic numeric
            type_prediction['type'] = 'number'
            type_prediction['confidence'] = 0.5
        
        # Add statistics to the result
        type_prediction['statistics'] = stats
        
        return type_prediction
    
    def _analyze_string_column(self, sample: pd.Series, full_series: pd.Series, dtype_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a string column to determine its specific characteristics and likely content type
        
        Args:
            sample (Series): Sample data to analyze
            full_series (Series): The full original series
            dtype_analysis (dict): Results from dtype analysis
            
        Returns:
            dict: String column analysis results
        """
        # Get basic statistics
        str_sample = sample.astype(str)
        
        # Calculate string characteristics
        length_stats = str_sample.str.len().describe()
        
        # Calculate more detailed statistics
        stats = {
            'count': len(full_series.dropna()),
            'nulls': full_series.isna().sum(),
            'null_percentage': full_series.isna().sum() / len(full_series) if len(full_series) > 0 else 0,
            'unique_count': sample.nunique(),
            'unique_percentage': sample.nunique() / len(sample) if len(sample) > 0 else 0,
            'avg_length': float(length_stats['mean']),
            'min_length': float(length_stats['min']),
            'max_length': float(length_stats['max']),
            'std_length': float(length_stats['std'])
        }
        
        # Check if strings contain certain patterns
        contains_space = (str_sample.str.contains(' ')).mean()
        contains_special = (str_sample.str.contains('[^a-zA-Z0-9\\s]')).mean()
        is_uppercase = (str_sample.str.isupper()).mean()
        is_lowercase = (str_sample.str.islower()).mean()
        is_titlecase = (str_sample.str.istitle()).mean() if hasattr(str_sample.str, 'istitle') else 0
        contains_number = (str_sample.str.contains('[0-9]')).mean()
        
        # Initialize type prediction
        type_prediction = {
            'type': 'string',
            'confidence': dtype_analysis['confidence'],
            'subtypes': {}
        }
        
        # Check for specific string patterns using regex
        pattern_matches = self._check_string_patterns(str_sample)
        
        # Add pattern matches to subtypes
        for pattern_name, match_ratio in pattern_matches.items():
            if match_ratio > 0.7:  # Only include high confidence matches
                type_prediction['subtypes'][pattern_name] = match_ratio
        
        # Check for common categories based on value set comparison
        category_matches = self._check_value_sets(str_sample)
        
        # Add category matches to subtypes
        for category_name, match_ratio in category_matches.items():
            if match_ratio > 0.5:  # Include moderate confidence matches
                type_prediction['subtypes'][category_name] = match_ratio
        
        # Text characteristics based analysis
        if contains_space > 0.8 and stats['avg_length'] > 20:
            # Likely full text/paragraphs
            type_prediction['subtypes']['text_content'] = 0.8
        elif contains_space > 0.8 and 10 <= stats['avg_length'] <= 100:
            # Likely names, titles, short descriptions
            type_prediction['subtypes']['name'] = 0.7
            type_prediction['subtypes']['title'] = 0.7
        elif stats['avg_length'] < 10 and contains_special < 0.3:
            # Short codes or abbreviations
            type_prediction['subtypes']['code'] = 0.6
            type_prediction['subtypes']['abbreviation'] = 0.6
        
        # Check for IDs/codes with specific patterns
        if contains_special > 0 and contains_number > 0 and 5 <= stats['avg_length'] <= 20:
            # Pattern like ABC-123, A12B34, etc.
            type_prediction['subtypes']['id_code'] = 0.7
        
        # Check for personal names
        if is_titlecase > 0.7 and 'first_name' in category_matches:
            type_prediction['subtypes']['personal_name'] = category_matches['first_name']
        
        # Determine primary type based on highest confidence subtype
        if type_prediction['subtypes']:
            primary_subtype = max(type_prediction['subtypes'].items(), key=lambda x: x[1])
            type_prediction['type'] = primary_subtype[0]
            type_prediction['confidence'] = primary_subtype[1]
        else:
            # Default to generic string
            type_prediction['type'] = 'text'
            type_prediction['confidence'] = 0.5
        
        # Add statistics to the result
        type_prediction['statistics'] = stats
        
        return type_prediction
    
    def _analyze_datetime_column(self, sample: pd.Series, full_series: pd.Series, dtype_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a datetime column to determine its specific characteristics
        
        Args:
            sample (Series): Sample data to analyze
            full_series (Series): The full original series
            dtype_analysis (dict): Results from dtype analysis
            
        Returns:
            dict: Datetime column analysis results
        """
        # Calculate statistics on dates
        min_date = sample.min()
        max_date = sample.max()
        
        # Convert timestamps to datetime for easier handling
        if not isinstance(min_date, (datetime, pd.Timestamp)):
            min_date = pd.to_datetime(min_date)
        if not isinstance(max_date, (datetime, pd.Timestamp)):
            max_date = pd.to_datetime(max_date)
        
        # Calculate range in days
        try:
            date_range_days = (max_date - min_date).days
        except:
            date_range_days = 0
        
        # Get basic statistics
        stats = {
            'count': len(full_series.dropna()),
            'nulls': full_series.isna().sum(),
            'null_percentage': full_series.isna().sum() / len(full_series) if len(full_series) > 0 else 0,
            'min_date': min_date.strftime('%Y-%m-%d'),
            'max_date': max_date.strftime('%Y-%m-%d'),
            'date_range_days': date_range_days,
            'unique_count': sample.nunique(),
            'unique_percentage': sample.nunique() / len(sample) if len(sample) > 0 else 0
        }
        
        # Check for time component
        has_time = False
        try:
            time_values = sample.dt.time
            # Check if there are non-zero time values
            if time_values.astype(str).str.match(r'00:00:00').mean() < 0.9:
                has_time = True
        except:
            pass
        
        # Initialize type prediction
        type_prediction = {
            'type': 'datetime',
            'confidence': dtype_analysis['confidence'],
            'subtypes': {}
        }
        
        # Check for date-related subtypes
        
        # 1. If has time component, likely timestamp
        if has_time:
            type_prediction['subtypes']['timestamp'] = 0.9
        else:
            type_prediction['subtypes']['date'] = 0.9
        
        # 2. Check recent vs. historical dates
        current_year = datetime.now().year
        recent_threshold = pd.Timestamp(datetime(current_year - 5, 1, 1))
        future_threshold = pd.Timestamp(datetime(current_year + 5, 1, 1))
        
        if min_date > recent_threshold:
            # Recent dates (last 5 years)
            type_prediction['subtypes']['recent_date'] = 0.8
            
            # Check for future dates
            if max_date > pd.Timestamp(datetime.now()):
                type_prediction['subtypes']['future_date'] = 0.8
                
                # Could be due date, scheduled date, etc.
                type_prediction['subtypes']['scheduled_date'] = 0.7
        elif max_date < recent_threshold:
            # Historical dates (more than 5 years ago)
            type_prediction['subtypes']['historical_date'] = 0.8
            
            # Could be birth date if in reasonable range
            if pd.Timestamp(datetime(current_year - 100, 1, 1)) < min_date < pd.Timestamp(datetime(current_year - 10, 1, 1)):
                type_prediction['subtypes']['birth_date'] = 0.7
        
        # 3. Check for event-related date subtypes
        # Determine primary type based on highest confidence subtype
        if type_prediction['subtypes']:
            primary_subtype = max(type_prediction['subtypes'].items(), key=lambda x: x[1])
            type_prediction['type'] = primary_subtype[0]
            type_prediction['confidence'] = primary_subtype[1]
        
        # Add statistics to the result
        type_prediction['statistics'] = stats
        
        return type_prediction
    
    
    def _analyze_boolean_column(self, sample: pd.Series, full_series: pd.Series, dtype_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a boolean column to determine its specific characteristics
        
        Args:
            sample (Series): Sample data to analyze
            full_series (Series): The full original series
            dtype_analysis (dict): Results from dtype analysis
            
        Returns:
            dict: Boolean column analysis results
        """
        # Convert categorical dtype to numeric before calculating statistics
        if pd.api.types.is_categorical_dtype(sample.dtype):
            # Convert to string first, then to numeric boolean
            if all(str(v).lower() in ['true', 'false', '1', '0', 'yes', 'no'] for v in sample.dropna().unique()):
                # Convert to proper boolean values
                sample = sample.astype(str).map({'true': True, 'false': False, '1': True, '0': False, 
                                                  'yes': True, 'no': False, 'True': True, 'False': False}).astype('boolean')
            else:
                # If not clearly boolean values, return minimal analysis
                return {
                    'type': 'categorical',
                    'confidence': 0.6,
                    'subtypes': {'categorical': 0.9},
                    'statistics': {
                        'count': len(full_series.dropna()),
                        'nulls': full_series.isna().sum(),
                        'null_percentage': full_series.isna().sum() / len(full_series) if len(full_series) > 0 else 0,
                        'unique_count': sample.nunique(),
                        'unique_percentage': sample.nunique() / len(sample) if len(sample) > 0 else 0
                    }
                }
                
        # Calculate statistics
        try:
            true_count = sample.sum()
            false_count = len(sample) - true_count
            true_ratio = true_count / len(sample) if len(sample) > 0 else 0
        except TypeError:
            # If sum() fails, try an alternative approach for non-numeric boolean-like values
            value_counts = sample.value_counts()
            if len(value_counts) <= 2:
                # Try to identify which values represent "true"
                likely_true_values = [True, 1, 'True', 'true', 'YES', 'yes', 'Y', 'y', '1']
                true_values = [v for v in value_counts.index if v in likely_true_values]
                
                if true_values:
                    true_count = sum(value_counts[v] for v in true_values)
                else:
                    # If no clear true values, take the first value arbitrarily
                    true_count = value_counts.iloc[0]
                    
                false_count = len(sample) - true_count
                true_ratio = true_count / len(sample) if len(sample) > 0 else 0
            else:
                # Not a binary column
                return {
                    'type': 'categorical',
                    'confidence': 0.7,
                    'subtypes': {'categorical': 0.9},
                    'statistics': {
                        'count': len(full_series.dropna()),
                        'nulls': full_series.isna().sum(),
                        'null_percentage': full_series.isna().sum() / len(full_series) if len(full_series) > 0 else 0,
                        'unique_count': sample.nunique(),
                        'unique_percentage': sample.nunique() / len(sample) if len(sample) > 0 else 0
                    }
                }
        
        stats = {
            'count': len(full_series.dropna()),
            'nulls': full_series.isna().sum(),
            'null_percentage': full_series.isna().sum() / len(full_series) if len(full_series) > 0 else 0,
            'true_count': int(true_count),
            'false_count': int(false_count),
            'true_ratio': float(true_ratio),
            'unique_count': sample.nunique(),
            'unique_percentage': sample.nunique() / len(sample) if len(sample) > 0 else 0
        }
        
        # Initialize type prediction
        type_prediction = {
            'type': 'boolean',
            'confidence': dtype_analysis['confidence'],
            'subtypes': {
                'boolean': 0.9,
                'flag': 0.8,
                'indicator': 0.8
            }
        }
        
        # Check for specific boolean subtypes
        if 0.05 <= true_ratio <= 0.15:
            # Low true ratio suggests rare events
            type_prediction['subtypes']['rare_event'] = 0.7
        elif 0.85 <= true_ratio <= 0.95:
            # High true ratio suggests common events
            type_prediction['subtypes']['common_event'] = 0.7
        
        # Determine primary type based on highest confidence subtype
        if type_prediction['subtypes']:
            primary_subtype = max(type_prediction['subtypes'].items(), key=lambda x: x[1])
            type_prediction['type'] = primary_subtype[0]
            type_prediction['confidence'] = primary_subtype[1]
        
        # Add statistics to the result
        type_prediction['statistics'] = stats
        
        return type_prediction
    
    def _check_string_patterns(self, str_sample: pd.Series) -> Dict[str, float]:
        """
        Check string values against regex patterns to identify common data formats
        
        Args:
            str_sample (Series): String values to check
            
        Returns:
            dict: Match ratios for each pattern
        """
        matches = {}
        
        for pattern_name, regex in self.regex_patterns.items():
            try:
                # Calculate how many values match this pattern
                match_count = str_sample.str.match(regex).sum()
                match_ratio = match_count / len(str_sample) if len(str_sample) > 0 else 0
                
                # Only include significant matches
                if match_ratio > 0.5:
                    matches[pattern_name] = float(match_ratio)
            except:
                # Skip any patterns that cause errors
                continue
        
        return matches
    
    def _check_value_sets(self, str_sample: pd.Series) -> Dict[str, float]:
        """
        Check string values against reference datasets to identify common categories
        
        Args:
            str_sample (Series): String values to check
            
        Returns:
            dict: Match ratios for each reference dataset
        """
        matches = {}
        
        # Normalize sample strings for comparison
        normalized_sample = str_sample.str.lower().str.strip()
        
        # Check each reference dataset
        for dataset_name, reference_values in self.reference_data.items():
            try:
                # Create a set for faster lookups
                reference_set = {value.lower() if isinstance(value, str) else value for value in reference_values}
                
                # Count how many unique values are in the reference set
                unique_values = normalized_sample.unique()
                
                if len(unique_values) == 0:
                    continue
                
                matches_count = sum(1 for value in unique_values if value in reference_set)
                match_ratio = matches_count / len(unique_values)
                
                # Only include significant matches
                if match_ratio > 0.3:
                    matches[dataset_name] = float(match_ratio)
            except:
                # Skip any datasets that cause errors
                continue
        
        return matches
    
    def analyze_dataframe(self, df: pd.DataFrame, sample_size: int = 1000) -> Dict[str, Dict[str, Any]]:
        """
        Analyze all columns in a DataFrame
        
        Args:
            df (DataFrame): The pandas DataFrame to analyze
            sample_size (int): Number of samples to use for analysis
            
        Returns:
            dict: Analysis results for each column
        """
        results = {}
        
        for column in df.columns:
            results[column] = self.analyze_column(df[column], sample_size)
        
        return results
    
    def detect_column_relationships(self, df: pd.DataFrame) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect relationships between columns in a DataFrame
        
        Args:
            df (DataFrame): The pandas DataFrame to analyze
            
        Returns:
            dict: Detected relationships between columns
        """
        # Analyze individual columns first
        column_analyses = self.analyze_dataframe(df)
        
        relationships = {}
        
        # 1. Look for ID-to-name relationships
        id_columns = [col for col, analysis in column_analyses.items() 
                     if 'id' in analysis['type'].lower() or 'code' in analysis['type'].lower()]
        
        name_columns = [col for col, analysis in column_analyses.items() 
                       if 'name' in analysis['type'].lower() or 'title' in analysis['type'].lower() 
                       or 'description' in analysis['type'].lower()]
        
        for id_col in id_columns:
            id_relationships = []
            for name_col in name_columns:
                # Check if they might be related
                if self._check_potential_relationship(df, id_col, name_col):
                    id_relationships.append({
                        'related_column': name_col,
                        'relationship_type': 'id_to_name',
                        'confidence': 0.8
                    })
            
            if id_relationships:
                relationships[id_col] = id_relationships
        
        # 2. Look for parent-child relationships (e.g., category-subcategory)
        categorical_columns = [col for col, analysis in column_analyses.items() 
                              if analysis['type'] == 'categorical' or 'category' in analysis['type'].lower()]
        
        for i, parent_col in enumerate(categorical_columns):
            for child_col in categorical_columns[i+1:]:
                if self._check_hierarchical_relationship(df, parent_col, child_col):
                    if parent_col not in relationships:
                        relationships[parent_col] = []
                    
                    relationships[parent_col].append({
                        'related_column': child_col,
                        'relationship_type': 'parent_child',
                        'confidence': 0.7
                    })
        
        # 3. Look for calculated columns (e.g., price * quantity = total)
        numeric_columns = [col for col, analysis in column_analyses.items() 
                          if analysis['statistics'].get('min') is not None]  # Only numeric columns have min statistic
        
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns[i+1:], i+1):
                for k, col3 in enumerate(numeric_columns):
                    if k != i and k != j:
                        # Check if col3 ≈ col1 * col2
                        if self._check_multiplication_relationship(df, col1, col2, col3):
                            if col3 not in relationships:
                                relationships[col3] = []
                            
                            relationships[col3].append({
                                'related_columns': [col1, col2],
                                'relationship_type': 'product',
                                'formula': f"{col3} = {col1} * {col2}",
                                'confidence': 0.85
                            })
                        
                        # Check if col3 ≈ col1 + col2
                        elif self._check_addition_relationship(df, col1, col2, col3):
                            if col3 not in relationships:
                                relationships[col3] = []
                            
                            relationships[col3].append({
                                'related_columns': [col1, col2],
                                'relationship_type': 'sum',
                                'formula': f"{col3} = {col1} + {col2}",
                                'confidence': 0.85
                            })
        
        return relationships
    
    def _check_potential_relationship(self, df: pd.DataFrame, col1: str, col2: str) -> bool:
        """
        Check if two columns might be related (e.g., ID to name mapping)
        
        Args:
            df (DataFrame): The DataFrame containing both columns
            col1 (str): First column name
            col2 (str): Second column name
            
        Returns:
            bool: True if columns appear to be related
        """
        # Count how many unique col1 values map to a single col2 value
        try:
            mapping = df.groupby(col1)[col2].nunique()
            # If most col1 values map to a single col2 value, they're likely related
            mostly_one_to_one = (mapping == 1).mean() > 0.8
            return mostly_one_to_one
        except:
            return False
    
    def _check_hierarchical_relationship(self, df: pd.DataFrame, parent_col: str, child_col: str) -> bool:
        """
        Check if two categorical columns have a hierarchical relationship
        
        Args:
            df (DataFrame): The DataFrame containing both columns
            parent_col (str): Potential parent column name
            child_col (str): Potential child column name
            
        Returns:
            bool: True if columns appear to have a hierarchical relationship
        """
        try:
            # For each unique parent value, count unique child values
            child_counts = df.groupby(parent_col)[child_col].nunique()
            
            # Parent should have fewer unique values
            if df[parent_col].nunique() >= df[child_col].nunique():
                return False
            
            # Most parent values should map to multiple child values
            multiple_children = (child_counts > 1).mean() > 0.5
            
            # Child values should mostly belong to a single parent
            reverse_counts = df.groupby(child_col)[parent_col].nunique()
            single_parent = (reverse_counts == 1).mean() > 0.7
            
            return multiple_children and single_parent
        except:
            return False
    
    def _check_multiplication_relationship(self, df: pd.DataFrame, col1: str, col2: str, result_col: str) -> bool:
        """
        Check if result_col ≈ col1 * col2
        
        Args:
            df (DataFrame): The DataFrame containing the columns
            col1 (str): First factor column name
            col2 (str): Second factor column name
            result_col (str): Potential product column name
            
        Returns:
            bool: True if relationship appears valid
        """
        try:
            # Drop rows with nulls
            clean_df = df[[col1, col2, result_col]].dropna()
            if len(clean_df) < 10:  # Need sufficient data
                return False
            
            # Calculate the product
            product = clean_df[col1] * clean_df[col2]
            
            # Check if product is close to result_col
            diff_ratio = (abs(product - clean_df[result_col]) / (clean_df[result_col] + 1e-10)).mean()
            
            # If average difference is less than 1%, it's likely a product relationship
            return diff_ratio < 0.01
        except:
            return False
    
    def _check_addition_relationship(self, df: pd.DataFrame, col1: str, col2: str, result_col: str) -> bool:
        """
        Check if result_col ≈ col1 + col2
        
        Args:
            df (DataFrame): The DataFrame containing the columns
            col1 (str): First term column name
            col2 (str): Second term column name
            result_col (str): Potential sum column name
            
        Returns:
            bool: True if relationship appears valid
        """
        try:
            # Drop rows with nulls
            clean_df = df[[col1, col2, result_col]].dropna()
            if len(clean_df) < 10:  # Need sufficient data
                return False
            
            # Calculate the sum
            sum_vals = clean_df[col1] + clean_df[col2]
            
            # Check if sum is close to result_col
            diff_ratio = (abs(sum_vals - clean_df[result_col]) / (clean_df[result_col] + 1e-10)).mean()
            
            # If average difference is less than 1%, it's likely a sum relationship
            return diff_ratio < 0.01
        except:
            return False
    
    def generate_column_fingerprints(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generate fingerprints for each column to enable matching with similar columns
        
        Args:
            df (DataFrame): The DataFrame to analyze
            
        Returns:
            dict: Column fingerprints
        """
        fingerprints = {}
        
        # Analyze the DataFrame
        analyses = self.analyze_dataframe(df)
        
        for column, analysis in analyses.items():
            fingerprint = {
                'type': analysis.get('type', 'unknown'),
                'base_type': None,
                'statistics': {},
                'patterns': {},
                'value_distribution': {}
            }
            
            # Get basic type
            if 'numeric' in analysis.get('type', '').lower() or analysis.get('statistics', {}).get('min') is not None:
                fingerprint['base_type'] = 'numeric'
            elif 'string' in analysis.get('type', '').lower() or 'text' in analysis.get('type', '').lower():
                fingerprint['base_type'] = 'string'
            elif 'date' in analysis.get('type', '').lower() or 'time' in analysis.get('type', '').lower():
                fingerprint['base_type'] = 'datetime'
            elif 'bool' in analysis.get('type', '').lower() or 'flag' in analysis.get('type', '').lower():
                fingerprint['base_type'] = 'boolean'
            else:
                fingerprint['base_type'] = 'unknown'
            
            # Extract relevant statistics based on type
            stats = analysis.get('statistics', {})
            
            if fingerprint['base_type'] == 'numeric':
                for stat_name in ['min', 'max', 'mean', 'median', 'std']:
                    if stat_name in stats:
                        fingerprint['statistics'][stat_name] = stats[stat_name]
            elif fingerprint['base_type'] == 'string':
                for stat_name in ['avg_length', 'min_length', 'max_length']:
                    if stat_name in stats:
                        fingerprint['statistics'][stat_name] = stats[stat_name]
            elif fingerprint['base_type'] == 'datetime':
                for stat_name in ['min_date', 'max_date', 'date_range_days']:
                    if stat_name in stats:
                        fingerprint['statistics'][stat_name] = stats[stat_name]
            
            # Add universal statistics
            for stat_name in ['unique_count', 'unique_percentage', 'null_percentage']:
                if stat_name in stats:
                    fingerprint['statistics'][stat_name] = stats[stat_name]
            
            # Add pattern information
            if 'subtypes' in analysis:
                fingerprint['patterns'] = analysis['subtypes']
            
            # Add value distribution information if feasible
            if fingerprint['base_type'] == 'string' and stats.get('unique_count', 0) < 100:
                try:
                    # For categorical columns, include value frequencies
                    value_counts = df[column].value_counts(normalize=True, dropna=True)
                    fingerprint['value_distribution'] = value_counts.head(20).to_dict()  # Top 20 values
                except:
                    pass
            elif fingerprint['base_type'] == 'numeric':
                try:
                    # For numeric columns, include histogram data
                    hist, bins = np.histogram(df[column].dropna(), bins=10)
                    fingerprint['value_distribution'] = {
                        'histogram': hist.tolist(),
                        'bins': bins.tolist()
                    }
                except:
                    pass
            
            fingerprints[column] = fingerprint
        
        return fingerprints
    
    def match_columns(self, fingerprint1: Dict[str, Any], fingerprint2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Match two column fingerprints to determine similarity
        
        Args:
            fingerprint1 (dict): First column fingerprint
            fingerprint2 (dict): Second column fingerprint
            
        Returns:
            dict: Matching results including similarity score and reasons
        """
        match_result = {
            'similarity': 0.0,
            'match_aspects': [],
            'mismatch_aspects': []
        }
        
        # 1. Check basic type compatibility
        if fingerprint1['base_type'] != fingerprint2['base_type']:
            match_result['similarity'] = 0.0
            match_result['mismatch_aspects'].append('different_base_types')
            return match_result
        
        # 2. Check specific type similarity
        if fingerprint1['type'] == fingerprint2['type']:
            match_result['similarity'] += 0.4
            match_result['match_aspects'].append('same_specific_type')
        else:
            # Check for type similarity
            type1_words = set(fingerprint1['type'].lower().split('_'))
            type2_words = set(fingerprint2['type'].lower().split('_'))
            type_overlap = len(type1_words.intersection(type2_words))
            
            if type_overlap > 0:
                overlap_ratio = type_overlap / max(len(type1_words), len(type2_words))
                match_result['similarity'] += 0.3 * overlap_ratio
                match_result['match_aspects'].append('similar_type')
            else:
                match_result['similarity'] += 0.1  # Same base type but different specific type
                match_result['mismatch_aspects'].append('different_specific_type')
        
        # 3. Check statistics similarity
        stats_similarity = self._compare_statistics(fingerprint1, fingerprint2)
        match_result['similarity'] += 0.4 * stats_similarity
        
        if stats_similarity > 0.7:
            match_result['match_aspects'].append('similar_statistics')
        elif stats_similarity > 0.3:
            match_result['match_aspects'].append('moderately_similar_statistics')
        else:
            match_result['mismatch_aspects'].append('different_statistics')
        
        # 4. Check pattern similarity
        pattern_similarity = self._compare_patterns(fingerprint1, fingerprint2)
        match_result['similarity'] += 0.2 * pattern_similarity
        
        if pattern_similarity > 0.7:
            match_result['match_aspects'].append('similar_patterns')
        elif pattern_similarity > 0.3:
            match_result['match_aspects'].append('some_common_patterns')
        
        # Round and ensure similarity is between 0 and 1
        match_result['similarity'] = round(min(max(match_result['similarity'], 0.0), 1.0), 2)
        
        return match_result
    
    def _compare_statistics(self, fingerprint1: Dict[str, Any], fingerprint2: Dict[str, Any]) -> float:
        """
        Compare statistics between two fingerprints
        
        Args:
            fingerprint1 (dict): First fingerprint
            fingerprint2 (dict): Second fingerprint
            
        Returns:
            float: Similarity score for statistics (0-1)
        """
        stats1 = fingerprint1.get('statistics', {})
        stats2 = fingerprint2.get('statistics', {})
        
        # If either is empty, can't compare
        if not stats1 or not stats2:
            return 0.0
        
        # Find common statistics
        common_stats = set(stats1.keys()) & set(stats2.keys())
        
        if not common_stats:
            return 0.0
        
        similarities = []
        
        for stat in common_stats:
            val1 = stats1[stat]
            val2 = stats2[stat]
            
            # Skip non-numeric values
            if not isinstance(val1, (int, float)) or not isinstance(val2, (int, float)):
                continue
            
            # Calculate normalized difference
            if stat in ['unique_percentage', 'null_percentage']:
                # For percentages, direct comparison
                similarity = 1.0 - abs(val1 - val2)
            elif stat in ['min', 'max', 'mean', 'median']:
                # For range statistics, use relative difference
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    rel_diff = abs(val1 - val2) / max_val
                    similarity = 1.0 - min(rel_diff, 1.0)
                else:
                    similarity = 1.0  # Both are 0
            else:
                # For other statistics, more lenient comparison
                max_val = max(abs(val1), abs(val2))
                if max_val > 0:
                    rel_diff = abs(val1 - val2) / max_val
                    similarity = 1.0 - min(rel_diff * 0.5, 1.0)  # More lenient
                else:
                    similarity = 1.0
            
            similarities.append(similarity)
        
        # Return average similarity if we have values
        if similarities:
            return sum(similarities) / len(similarities)
        else:
            return 0.0
    
    def _compare_patterns(self, fingerprint1: Dict[str, Any], fingerprint2: Dict[str, Any]) -> float:
        """
        Compare patterns between two fingerprints
        
        Args:
            fingerprint1 (dict): First fingerprint
            fingerprint2 (dict): Second fingerprint
            
        Returns:
            float: Similarity score for patterns (0-1)
        """
        patterns1 = fingerprint1.get('patterns', {})
        patterns2 = fingerprint2.get('patterns', {})
        
        # If either is empty, can't compare
        if not patterns1 or not patterns2:
            return 0.0
        
        # Find common and unique patterns
        pattern_keys1 = set(patterns1.keys())
        pattern_keys2 = set(patterns2.keys())
        common_patterns = pattern_keys1 & pattern_keys2
        all_patterns = pattern_keys1 | pattern_keys2
        
        # Jaccard similarity - ratio of common patterns to all patterns
        jaccard_sim = len(common_patterns) / len(all_patterns) if all_patterns else 0
        
        # Calculate confidence-weighted similarity for common patterns
        weighted_similarity = 0.0
        total_weight = 0.0
        
        for pattern in common_patterns:
            conf1 = patterns1[pattern]
            conf2 = patterns2[pattern]
            weight = (conf1 + conf2) / 2  # Average confidence as weight
            similarity = 1.0 - abs(conf1 - conf2)  # Confidence similarity
            weighted_similarity += weight * similarity
            total_weight += weight
        
        if total_weight > 0:
            weighted_pattern_sim = weighted_similarity / total_weight
        else:
            weighted_pattern_sim = 0.0
        
        # Combine Jaccard and weighted similarity
        return 0.4 * jaccard_sim + 0.6 * weighted_pattern_sim

# Example usage
if __name__ == "__main__":
    # Create sample data
    data = {
        'customer_id': ['CUST001', 'CUST002', 'CUST003', 'CUST004', 'CUST005'],
        'customer_name': ['John Smith', 'Jane Doe', 'Bob Johnson', 'Alice Brown', 'Charlie Davis'],
        'email': ['john@example.com', 'jane@example.com', 'bob@example.com', 'alice@example.com', 'charlie@example.com'],
        'signup_date': pd.to_datetime(['2020-01-15', '2020-02-20', '2020-03-10', '2020-04-05', '2020-05-22']),
        'last_purchase': pd.to_datetime(['2020-06-10', '2020-06-15', '2020-06-01', '2020-06-20', '2020-06-25']),
        'total_orders': [5, 8, 3, 12, 7],
        'avg_order_value': [75.50, 120.25, 45.80, 150.75, 95.20],
        'total_spent': [377.50, 962.00, 137.40, 1809.00, 666.40],
        'loyalty_tier': ['Silver', 'Gold', 'Bronze', 'Platinum', 'Gold'],
        'is_active': [True, True, False, True, True],
        'zipcode': ['10001', '94107', '60601', '02110', '30305']
    }
    
    df = pd.DataFrame(data)
    
    # Create classifier
    classifier = ContentClassifier()
    
    # Analyze a specific column
    email_analysis = classifier.analyze_column(df['email'])
    print("\nEmail column analysis:")
    print(json.dumps(email_analysis, indent=2))
    
    # Detect relationships
    relationships = classifier.detect_column_relationships(df)
    print("\nDetected relationships:")
    for col, rels in relationships.items():
        print(f"\n{col}:")
        for rel in rels:
            print(f"  - {rel}")
    
    # Generate fingerprints
    fingerprints = classifier.generate_column_fingerprints(df)
    
    # Match columns from different datasets
    # Create a second sample DataFrame with similar structure
    data2 = {
        'client_id': ['CL001', 'CL002', 'CL003', 'CL004', 'CL005'],
        'client_name': ['Mark Wilson', 'Sarah Miller', 'Tom Anderson', 'Lisa Taylor', 'Greg White'],
        'contact_email': ['mark@example.com', 'sarah@example.com', 'tom@example.com', 'lisa@example.com', 'greg@example.com'],
        'registration_date': pd.to_datetime(['2020-01-10', '2020-02-15', '2020-03-20', '2020-04-25', '2020-05-30']),
        'last_activity': pd.to_datetime(['2020-06-05', '2020-06-10', '2020-06-15', '2020-06-20', '2020-06-25']),
        'orders_count': [6, 9, 4, 11, 8],
        'average_value': [80.25, 115.75, 50.30, 145.60, 90.80],
        'total_value': [481.50, 1041.75, 201.20, 1601.60, 726.40],
        'status_level': ['Silver', 'Gold', 'Bronze', 'Platinum', 'Gold'],
        'account_active': [True, True, False, True, True],
        'postal_code': ['20001', '95108', '60602', '02111', '30306']
    }
    
    df2 = pd.DataFrame(data2)
    fingerprints2 = classifier.generate_column_fingerprints(df2)
    
    # Match columns between the two DataFrames
    print("\nMatching columns between datasets:")
    for col1, fp1 in fingerprints.items():
        best_match = None
        best_score = 0.0
        
        for col2, fp2 in fingerprints2.items():
            match_result = classifier.match_columns(fp1, fp2)
            if match_result['similarity'] > best_score:
                best_score = match_result['similarity']
                best_match = (col2, match_result)
        
        if best_match and best_score >= 0.5:
            col2, match_result = best_match
            print(f"\n{col1} matches with {col2} (similarity: {best_score})")
            print(f"  Match aspects: {match_result['match_aspects']}")
            if match_result['mismatch_aspects']:
                print(f"  Mismatch aspects: {match_result['mismatch_aspects']}")
        