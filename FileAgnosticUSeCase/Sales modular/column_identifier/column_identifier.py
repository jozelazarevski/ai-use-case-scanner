# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:55:28 2025

@author: joze_
"""

"""
Enhanced Column Identifier Module with Advanced Regex Rules

This module contains improved functionality for automatically identifying important columns
in sales data, with enhanced regex pattern matching, contextual analysis, and robust error handling.
"""

import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import traceback
import string
from collections import Counter
import json
import time
import column_name_mappings as cnm

logger = logging.getLogger('sales_forecaster.column_identifier')

# Mock DEFAULT_CONFIG for standalone testing
DEFAULT_CONFIG = {
    # Date-related patterns
    'date_patterns': [
        'date', 'time', 'day', 'timestamp', 'order_date', 'invoice_date', 'transaction_date', 
        'purchase_date', 'sale_date', 'delivery_date', 'period', 'reporting_date', 'month', 'year',
        'quarter', 'week', 'fiscal'
    ],
    
    # Product-related patterns
    'product_patterns': [
        'product', 'item', 'sku', 'merchandise', 'good', 'article', 'material', 'model', 
        'part', 'brand', 'description', 'desc', 'title', 'variant', 'style', 'item_number',
        'product_id', 'item_code', 'catalog', 'inventory', 'prod', 'line'
    ],
    
    # Sales amount patterns
    'sales_patterns': [
        'sales', 'revenue', 'amount', 'value', 'total', 'sum', 'selling', 'income',
        'sales_value', 'gross_sales', 'net_sales', 'total_sales', 'sale_amount',
        'sale_value', 'invoice_amount', 'transaction_value', 'order_value', 'turnover',
        'gross_revenue', 'net_revenue', 'proceeds'
    ],
    
    # Quantity patterns
    'quantity_patterns': [
        'quantity', 'qty', 'count', 'units', 'unit', 'volume', 'number', 'pieces', 'pcs',
        'amount', 'sold', 'total_sold', 'sales_count', 'ordered', 'shipped', 'net_units',
        'gross_units', 'items_sold', 'order_quantity', 'total_quantity', 'sold_count'
    ],
    
    # Price patterns
    'price_patterns': [
        'price', 'unit_price', 'rate', 'cost', 'fee', 'charge', 'amount_per', 'price_per',
        'sale_price', 'list_price', 'retail_price', 'wholesale_price', 'msrp', 'base_price',
        'discounted_price', 'net_price', 'price_point', 'price_level', 'selling_price'
    ],
    
    # Return-related patterns
    'return_patterns': [
        'return', 'returns', 'returned', 'return_units', 'return_amount', 'return_value',
        'total_returns', 'return_quantity', 'returned_items', 'refund', 'refunds', 'credit',
        'return_rate', 'return_pct', 'return_percentage', 'items_returned'
    ],
    
    # Geographic patterns
    'geography_patterns': [
        'territory', 'region', 'area', 'zone', 'country', 'state', 'province', 'city',
        'location', 'market', 'district', 'geo', 'postal_code', 'zip', 'address',
        'locale', 'county', 'division', 'geography', 'sales_territory'
    ],
    
    # Channel patterns
    'channel_patterns': [
        'channel', 'segment', 'outlet', 'store', 'shop', 'dealer', 'distributor',
        'retailer', 'reseller', 'partner', 'marketplace', 'platform', 'sales_channel',
        'distribution_channel', 'e-commerce', 'retail', 'wholesale', 'online', 'offline'
    ],
    
    # Customer patterns
    'customer_patterns': [
        'customer', 'client', 'buyer', 'account', 'consumer', 'purchaser', 'patron',
        'shopper', 'end_user', 'contact', 'customer_id', 'client_name', 'account_no',
        'buyer_code', 'customer_group', 'client_category', 'customer_type'
    ],
    
    # Discount patterns
    'discount_patterns': [
        'discount', 'discount_amount', 'discount_pct', 'discount_rate', 'markdown',
        'promotion', 'promo', 'reduction', 'savings', 'deal', 'offer', 'special',
        'discount_value', 'markdown_amount', 'discount_percentage', 'promo_discount'
    ],
    
    # Royalty/commission patterns
    'royalty_patterns': [
        'royalty', 'commission', 'royalty_pct', 'royalty_rate', 'royalty_amount',
        'royalty_pay', 'royalty_payable', 'royalty_percentage', 'commission_rate',
        'commission_amount', 'commission_pct', 'fee', 'licensing', 'license_fee'
    ],
    
    # Profit patterns
    'profit_patterns': [
        'profit', 'margin', 'gp', 'gross_profit', 'net_profit', 'profit_margin',
        'contribution', 'earnings', 'income', 'surplus', 'gain', 'markup',
        'profit_amount', 'profit_percentage', 'profit_pct', 'profit_rate'
    ],
    
    # Tax patterns
    'tax_patterns': [
        'tax', 'vat', 'gst', 'sales_tax', 'tax_amount', 'tax_rate', 'tax_pct',
        'tax_value', 'duty', 'excise', 'levy', 'tax_total', 'tax_charged',
        'tax_applied', 'tax_collected', 'tax_percentage'
    ],
    
    # Product property/category patterns
    'property_patterns': [
        'property', 'category', 'class', 'type', 'format', 'style', 'group',
        'classification', 'family', 'line', 'series', 'collection', 'department',
        'division', 'segment', 'genre', 'attributes', 'characteristic'
    ],
    
    # Sales type patterns
    'sales_type_patterns': [
        'sales_type', 'order_type', 'transaction_type', 'sale_type', 'type',
        'order_category', 'transaction_category', 'sale_kind', 'order_class',
        'transaction_class', 'sales_classification', 'order_nature'
    ]
}

class ColumnIdentifier:
    """Enhanced class for identifying important columns in sales data with advanced regex patterns"""
        
    def __init__(self, data, date_col=None, product_col=None, sales_col=None):
        """
        Initialize the column identifier
        
        Args:
            data (DataFrame): The pandas DataFrame containing the sales data
            date_col (str, optional): Name of the date column if known
            product_col (str, optional): Name of the product column if known
            sales_col (str, optional): Name of the sales column if known
        """
        # Check if data is None and handle it properly
        if data is None:
            raise ValueError("Input data cannot be None. Please provide a valid pandas DataFrame.")
        
        self.data = data.copy()  # Create a copy to avoid modifying the original data
        self.date_col = date_col
        self.product_col = product_col
        self.sales_col = sales_col
        
    
        # Use mappings from external file
        self.column_name_mappings = cnm.COLUMN_NAME_MAPPINGS
        self.column_name_scores = cnm.COLUMN_NAME_SCORES
        self.column_type_fallbacks = cnm.COLUMN_TYPE_FALLBACKS
        self.composite_columns = cnm.COMPOSITE_COLUMNS
        
        # Column patterns from config
        # Column patterns from config with enhanced regex patterns
        self.date_patterns = DEFAULT_CONFIG['date_patterns']
        self.date_regex = [
            r'\b(?:order|invoice|transaction|sales|purchase)[\s_-]*(?:date|time|day|dt)\b',
            r'\b(?:date|dt)[\s_-]*(?:of[\s_-]*(?:order|invoice|transaction|sale|purchase))?\b',
            r'\btimestamp\b',
            r'\b(?:created|modified|entered|posted|recorded)[\s_-]*(?:date|time|on|at)\b',
            r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',  # YYYY-MM-DD or YYYY/MM/DD format
            r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$',  # MM-DD-YYYY or DD-MM-YYYY format
            r'^\d{2,4}[Qq][1-4]$',  # Quarter format like 2023Q1, 23Q4
            r'(?i)^(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[\s_-]?\d{2,4}$'  # Month Year format
        ]
        
        self.product_patterns = DEFAULT_CONFIG['product_patterns']
        self.product_regex = [
            r'\b(?:product|item|article|good|merchandise)[\s_-]*(?:id|code|number|no|identifier|key)\b',
            r'\b(?:prod|prd|itm|art)[\s_-]*(?:id|code|num|no)\b',
            r'\b(?:sku|upc|ean|isbn|mpn|model)[\s_-]*(?:code|number|no|id)?\b',
            r'\b(?:catalog|inventory)[\s_-]*(?:item|product|code|id)\b',
            r'^[A-Z]{1,3}[-_]?\d{3,}$',  # Format like ABC-123, A123
            r'^\d{3,}[-_]?[A-Z]{1,3}$'   # Format like 123-ABC, 123A
        ]
        
        self.sales_patterns = DEFAULT_CONFIG['sales_patterns']
        self.sales_regex = [
            r'\b(?:sales|revenue|income)[\s_-]*(?:amount|value|total|sum)?\b',
            r'\b(?:total|gross|net)[\s_-]*(?:sales|revenue|income|amount|value)\b',
            r'\b(?:extended|line)[\s_-]*(?:amount|price|total|value)\b',
            r'\b(?:invoice|order|transaction)[\s_-]*(?:amount|value|total)\b',
            r'\$[\s_]*(?:amount|value|total|sum)\b',
            r'\b(?:sales|revenue)[\s_-]*\d{4}\b'  # Sales followed by year like "sales 2023"
        ]
        
        self.quantity_patterns = DEFAULT_CONFIG['quantity_patterns']
        self.quantity_regex = [
            r'\b(?:qty|quantity|count|units?)[\s_-]*(?:ordered|sold|purchased|shipped)?\b',
            r'\b(?:total|net|gross|shipped)[\s_-]*(?:qty|quantity|count|units?)\b',
            r'\b(?:number|count)[\s_-]*(?:of[\s_-]*(?:items?|units?|pieces?))?\b',
            r'\b(?:order|sales|purchase|shipment)[\s_-]*(?:qty|quantity|count|units?)\b',
            r'\b(?:sold|shipped|ordered|purchased)[\s_-]*(?:qty|quantity|count|units?)\b',
            r'\b(?:items?|units?|pcs)[\s_-]*(?:sold|shipped|ordered|purchased)\b',
            r'(?i)^units?$',  # Matches singular "unit" or plural "units" case-insensitive
            r'(?i)^qty$'      # Matches "qty" case-insensitive
        ]
        
        self.price_patterns = DEFAULT_CONFIG['price_patterns']
        self.price_regex = [
            r'\b(?:unit|single|per[\s_-]*item)[\s_-]*(?:price|cost|value|amount|rate)\b',
            r'\b(?:price|cost|fee|charge|rate)[\s_-]*(?:per[\s_-]*(?:unit|item|piece))?\b',
            r'\b(?:item|unit)[\s_-]*(?:price|cost|value|amount|rate)\b',
            r'\b(?:list|retail|wholesale|msrp)[\s_-]*(?:price|cost|amount)\b',
            r'\b(?:price[\s_-]*point|price[\s_-]*level)\b',
            r'\$[\s_]*(?:price|rate|cost)\b',
            r'(?i)^price$'  # Matches "price" case-insensitive
        ]
        
        self.return_patterns = DEFAULT_CONFIG['return_patterns']
        self.return_regex = [
            r'\b(?:return|returned)[\s_-]*(?:units?|quantity|items?|goods?|amount|value)\b',
            r'\b(?:total|gross|net)[\s_-]*(?:returns?|refunds?)\b',
            r'\b(?:return|refund)[\s_-]*(?:rate|pct|percentage|ratio)\b',
            r'\b(?:items?|units?|goods?|products?)[\s_-]*(?:returned|refunded)\b',
            r'(?i)^returns?[\s_-]*(?:units?)?$'  # Matches "return", "returns", "return units", etc.
        ]
        
        self.geography_patterns = DEFAULT_CONFIG['geography_patterns']
        self.geography_regex = [
            r'\b(?:sales[\s_-]*)?(?:territory|region|area|zone|district|market)\b',
            r'\b(?:geo(?:graphy)?|location)[\s_-]*(?:id|code|name)?\b',
            r'\b(?:country|state|province|city|county)[\s_-]*(?:name|code)?\b',
            r'\b(?:postal|zip)[\s_-]*(?:code|area)\b',
            r'(?i)^(?:territory|region|area|zone|market|country|state)$'  # Exact match for common geography terms
        ]
        
        self.channel_patterns = DEFAULT_CONFIG['channel_patterns']
        self.channel_regex = [
            r'\b(?:sales[\s_-]*)?(?:channel|segment|division|class)\b',
            r'\b(?:distribution|marketing)[\s_-]*(?:channel|segment|path)\b',
            r'\b(?:market[\s_-]*segment|outlet[\s_-]*type)\b',
            r'\b(?:retail|wholesale|online|offline|e[\s_-]*commerce|b2b|b2c)[\s_-]*(?:channel|sales)?\b',
            r'(?i)^channel$'  # Exact match for "channel"
        ]
        
        self.royalty_patterns = DEFAULT_CONFIG['royalty_patterns']
        self.royalty_regex = [
            r'\b(?:royalty|commission)[\s_-]*(?:amount|value|total|payment|pay)\b',
            r'\b(?:royalty|commission)[\s_-]*(?:rate|pct|percentage|ratio|points?|share)\b',
            r'\b(?:royalty|commission)[\s_-]*(?:%|percent)\b',
            r'\b(?:total|gross|net)[\s_-]*(?:royalty|commission)\b',
            r'\b(?:royalty|commission)[\s_-]*(?:calculation|formula|basis)\b',
            r'(?i)^royalty[\s_-]*(?:%|percentage|pay|payable)?$'  # Matches common royalty column names
        ]
        
        self.property_patterns = DEFAULT_CONFIG['property_patterns']
        self.property_regex = [
            r'\b(?:product[\s_-]*)?(?:property|category|class|type|format|style)\b',
            r'\b(?:item[\s_-]*)?(?:group|family|line|series|collection)\b',
            r'\b(?:product[\s_-]*)?(?:classification|segment|genre|attribute)\b',
            r'\b(?:department|division)[\s_-]*(?:name|code|id)?\b',
            r'(?i)^(?:property|category|type|format|style|class)$'  # Exact matches
        ]
        
        self.sales_type_patterns = DEFAULT_CONFIG['sales_type_patterns']
        self.sales_type_regex = [
            r'\b(?:sales|order|transaction)[\s_-]*(?:type|category|class|kind|nature)\b',
            r'\b(?:type|category|class)[\s_-]*(?:of[\s_-]*(?:sales|order|transaction))?\b',
            r'\b(?:sales|order|transaction)[\s_-]*(?:classification|segment)\b',
            r'(?i)^(?:sales[\s_-]*type|order[\s_-]*type|type)$'  # Common exact matches
        ]
                
        # Advanced regex patterns for column name matching
        self.regex_patterns = self._initialize_regex_patterns()
        
        # Add specific component patterns for date elements
        self.year_patterns = ['year', 'yr', 'yyyy', 'yy', 'fiscal_year', 'fy']
        self.month_patterns = ['month', 'mon', 'mm', 'mo', 'mnth']
        self.day_patterns = ['day', 'dy', 'dd', 'dom']
        
        # Add enhanced patterns for product identification
        self.product_id_patterns = ['sku', 'item_number', 'product_code', 'part_number', 'material_number']
        self.product_name_patterns = ['product_name', 'item_name', 'product_desc', 'description', 'material_desc']
        self.product_category_patterns = ['category', 'product_cat', 'product_type', 'product_group', 'family']
        
        # Dictionary to store detected columns
        self.detected_columns = {
            'date': None,
            'year': None,
            'month': None,
            'day': None,
            'product': None,
            'product_id': None,
            'product_name': None,
            'product_category': None,
            'sales': None,
            'quantity': None,
            'price': None,
            'revenue': None,
            'cost': None,
            'profit': None,
            'customer': None,
            'location': None,
            'saled_units':None
            
        }
        
        # Dictionary to store column classifications
        self.column_classes = {
            'date_candidates': [],
            'date_component_candidates': {
                'year': [],
                'month': [],
                'day': []
            },
            'numeric_candidates': [],
            'categorical_candidates': [],
            'text_candidates': [],
            'id_candidates': [],
            'boolean_candidates': []
        }
         
     
    def _classify_columns(self):
        """
        Classify all columns by data type and potential role
        This pre-classification helps with subsequent specific column identification
        """
        # Store column names and types for debugging
        column_info = []
        
        for col in self.data.columns:
            try:
                col_lower = str(col).lower()
                col_type = str(self.data[col].dtype)
                
                # Store column info for debugging
                column_info.append(f"{col} ({col_type})")
                
                # Skip columns with too many nulls
                null_percentage = self.data[col].isna().mean()
                if null_percentage > 0.7:  # More lenient null threshold (70%)
                    continue
                
                # Check for date columns
                is_date = False
                if pd.api.types.is_datetime64_dtype(self.data[col].dtype):
                    self.column_classes['date_candidates'].append(col)
                    is_date = True
                
                # Try to convert to datetime if not already a date type
                if not is_date:
                    try:
                        # Only test a sample for performance reasons
                        sample = self.data[col].dropna().head(500)
                        if len(sample) > 0:
                            # Try direct datetime conversion
                            date_conversion = pd.to_datetime(sample, errors='coerce')
                            if date_conversion.notna().mean() > 0.7:
                                self.column_classes['date_candidates'].append(col)
                                is_date = True
                            
                            # If not a datetime but might be a string column, check for date formats
                            elif pd.api.types.is_string_dtype(self.data[col]) and not is_date:
                                # Check for regex date patterns in string columns
                                for pattern_name, pattern in self.regex_patterns['date']['format_patterns'].items():
                                    # Check a sample of values against the pattern
                                    matched = sample.astype(str).str.match(pattern).mean()
                                    if matched > 0.7:  # If > 70% match the pattern
                                        self.column_classes['date_candidates'].append(col)
                                        is_date = True
                                        break
                    except (TypeError, ValueError):
                        pass
                
                # Skip further classification if it's a date
                if is_date:
                    continue
                
                # Check if column name matches any regex patterns
                column_matched = False
                for col_type, patterns in self.regex_patterns.items():
                    if 'name_patterns' in patterns:
                        for pattern in patterns['name_patterns']:
                            if re.search(pattern, col_lower, re.IGNORECASE):
                                if col_type in ['year', 'month', 'day']:
                                    self.column_classes['date_component_candidates'][col_type].append(col)
                                elif col_type == 'date':
                                    self.column_classes['date_candidates'].append(col)
                                column_matched = True
                                break
                        if column_matched:
                            break
                
                # Check for date components (year, month, day) if not already matched
                if not column_matched and self._is_numeric_column(col):
                    valid_values = self.data[col].dropna()
                    
                    # Skip empty columns
                    if len(valid_values) == 0:
                        continue
                    
                    # Year detection
                    if any(pattern in col_lower for pattern in self.year_patterns):
                        try:
                            min_val = valid_values.min()
                            max_val = valid_values.max()
                            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                                if 1900 <= min_val <= max_val <= datetime.now().year + 5:  # Allow slightly future years
                                    self.column_classes['date_component_candidates']['year'].append(col)
                                    continue
                        except (TypeError, ValueError):
                            pass
                    
                    # Month detection
                    if any(pattern in col_lower for pattern in self.month_patterns):
                        try:
                            min_val = valid_values.min()
                            max_val = valid_values.max()
                            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                                if 1 <= min_val <= max_val <= 12:
                                    self.column_classes['date_component_candidates']['month'].append(col)
                                    continue
                        except (TypeError, ValueError):
                            pass
                    
                    # Day detection
                    if any(pattern in col_lower for pattern in self.day_patterns):
                        try:
                            min_val = valid_values.min()
                            max_val = valid_values.max()
                            if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                                if 1 <= min_val <= max_val <= 31:
                                    self.column_classes['date_component_candidates']['day'].append(col)
                                    continue
                        except (TypeError, ValueError):
                            pass
                    
                    # Advanced: Auto-detect date components without keywords in column name
                    # Detect potential year columns
                    if not any(col in candidates for candidates in self.column_classes['date_component_candidates'].values()):
                        try:
                            # Check integer columns in range 1900-current year
                            if (valid_values % 1 == 0).all():  # Is integer
                                min_val = valid_values.min()
                                max_val = valid_values.max()
                                if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                                    current_year = datetime.now().year
                                    if 1900 <= min_val <= max_val <= current_year + 5 and min_val != max_val:
                                        # Check if values are years (values in 1900-current_year range)
                                        self.column_classes['date_component_candidates']['year'].append(col)
                                        continue
                                    elif 1 <= min_val <= max_val <= 12 and min_val != max_val:
                                        # Check if values are months (1-12 range)
                                        self.column_classes['date_component_candidates']['month'].append(col)
                                        continue
                                    elif 1 <= min_val <= max_val <= 31 and min_val != max_val:
                                        # Check if values are days (1-31 range)
                                        self.column_classes['date_component_candidates']['day'].append(col)
                                        continue
                        except (TypeError, ValueError):
                            pass
                
                # Numeric columns classification
                if self._is_numeric_column(col):
                    self.column_classes['numeric_candidates'].append(col)
                    
                    try:
                        unique_count = self.data[col].nunique()
                        total_count = len(self.data)
                        
                        # Boolean-like columns (0/1, True/False)
                        if unique_count <= 2:
                            self.column_classes['boolean_candidates'].append(col)
                        # Categorical numeric columns
                        elif unique_count <= 20 or (unique_count / total_count < 0.01 and unique_count > 1):
                            self.column_classes['categorical_candidates'].append(col)
                        
                        # Check for ID patterns in numeric columns
                        if unique_count / total_count > 0.5 and self._has_id_pattern(col_lower):
                            self.column_classes['id_candidates'].append(col)
                    except (TypeError, ValueError):
                        pass
                
                # Categorical/string columns
                elif pd.api.types.is_string_dtype(self.data[col]) or pd.api.types.is_categorical_dtype(self.data[col]):
                    try:
                        unique_count = self.data[col].nunique()
                        total_count = len(self.data)
                        
                        # Check patterns in string columns
                        if unique_count > 0:
                            # Check for ID patterns in values
                            is_id = self._check_if_id_format(self.data[col])
                            
                            if is_id:
                                self.column_classes['id_candidates'].append(col)
                            
                            # Calculate average length and variation of values
                            sample = self.data[col].dropna().sample(min(100, unique_count)).astype(str)
                            if not sample.empty:
                                lengths = sample.str.len()
                                avg_length = lengths.mean()
                                std_length = lengths.std() if len(lengths) > 1 else 0
                                
                                # Criteria for categorical vs text
                                if unique_count <= 100 or unique_count / total_count < 0.3:
                                    self.column_classes['categorical_candidates'].append(col)
                                else:
                                    # High cardinality columns
                                    if self._has_id_pattern(col_lower) or is_id:
                                        self.column_classes['id_candidates'].append(col)
                                    elif avg_length > 20 or std_length / avg_length > 0.3:
                                        # Long strings or high variation in length = text
                                        self.column_classes['text_candidates'].append(col)
                                    else:
                                        self.column_classes['categorical_candidates'].append(col)
                    except (TypeError, ValueError):
                        # Fall back to assuming it's a text column if we can't analyze it
                        self.column_classes['text_candidates'].append(col)
            
            except Exception as e:
                logger.warning(f"Error classifying column '{col}': {str(e)}")
                continue
        
        # Log classification summary
        logger.debug(f"Columns found: {', '.join(column_info)}")
        logger.debug(f"Date candidates: {', '.join(self.column_classes['date_candidates'])}")
        logger.debug(f"Year candidates: {', '.join(self.column_classes['date_component_candidates']['year'])}")
        logger.debug(f"Month candidates: {', '.join(self.column_classes['date_component_candidates']['month'])}")
        logger.debug(f"Day candidates: {', '.join(self.column_classes['date_component_candidates']['day'])}")
        logger.debug(f"Numeric candidates: {len(self.column_classes['numeric_candidates'])}")
        logger.debug(f"Categorical candidates: {len(self.column_classes['categorical_candidates'])}")
        logger.debug(f"ID candidates: {len(self.column_classes['id_candidates'])}")
        logger.debug(f"Text candidates: {len(self.column_classes['text_candidates'])}")
    
    def _is_numeric_column(self, col):
        """Safely check if a column is numeric"""
        try:
            return pd.api.types.is_numeric_dtype(self.data[col])
        except:
            return False
    
    def _has_id_pattern(self, column_name):
        """Check if a column name suggests it's an ID column"""
        id_keywords = ['id', 'code', 'key', 'no', 'number', 'num', 'identifier', '#']
        
        # Handle non-string column names
        if not isinstance(column_name, str):
            return False
            
        for keyword in id_keywords:
            if keyword == column_name or column_name.endswith('_' + keyword) or column_name.startswith(keyword + '_') or \
               keyword + 's' == column_name or column_name.endswith('_' + keyword + 's') or column_name.startswith(keyword + 's_'):
                return True
        
        # Check for patterns like productid, customerid, etc.
        if re.search(r'[a-z]+id$', column_name) or re.search(r'^id[a-z]+', column_name):
            return True
            
        return False
    
    def _check_if_id_format(self, series):
        """Check if a series contains values that match typical ID formats using advanced patterns"""
        # Safety check
        if series.empty:
            return False
            
        try:
            # Sample some values to check
            sample_size = min(50, len(series))
            if sample_size == 0:
                return False
                
            sample = series.dropna().sample(sample_size).astype(str)
            
            # Count how many match ID patterns
            id_pattern_count = 0
            
            # Use more sophisticated ID patterns
            id_patterns = [
                # Single letter followed by numbers: A123, B456
                r'^[A-Za-z]\d{2,}$',
                
                # Multiple letters followed by numbers: ABC123, XYZ456
                r'^[A-Za-z]{2,}\d{2,}$',
                
                # Numbers followed by letters: 123A, 456XYZ
                r'^\d{2,}[A-Za-z]+$',
                
                # Combinations with separators: AB-123, 123-XY, A-1-B
                r'^[A-Za-z0-9]+-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?$',
                
                # UUIDs and similar formats
                r'^[A-Fa-f0-9]{8}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{12}$',
                
                # Alphanumeric with consistent pattern and length
                r'^[A-Za-z0-9]{4,12}$'
            ]
            
            # Additional pattern check for very specific ID formats
            # Check for consistent patterns across the sample
            if not sample.empty:
                patterns = []
                for val in sample:
                    if not isinstance(val, str):
                        continue
                        
                    # Create a pattern representation of the value
                    pattern = ''
                    for char in val:
                        if char in string.ascii_uppercase:
                            pattern += 'U'
                        elif char in string.ascii_lowercase:
                            pattern += 'L'
                        elif char in string.digits:
                            pattern += 'D'
                        elif char in '-_/':
                            pattern += 'S'  # Separator
                        else:
                            pattern += 'O'  # Other
                    patterns.append(pattern)
                
                # Count pattern frequencies
                pattern_counter = Counter(patterns)
                most_common_pattern, count = pattern_counter.most_common(1)[0] if pattern_counter else (None, 0)
                
                # If most common pattern appears in at least 70% of samples
                # and contains a mix of letters/digits, it's likely an ID
                if most_common_pattern and count / len(sample) >= 0.7:
                    if ('U' in most_common_pattern or 'L' in most_common_pattern) and 'D' in most_common_pattern:
                        return True
            
            # Check standard ID patterns
            for pattern in id_patterns:
                for val in sample:
                    if val and isinstance(val, str) and re.match(pattern, val):
                        id_pattern_count += 1
                        break
            
            # Additional check for numeric IDs (high cardinality of integers)
            if all(val.isdigit() for val in sample if val):
                unique_ratio = series.nunique() / len(series)
                if unique_ratio > 0.7:  # High uniqueness suggests IDs
                    return True
            
            # If more than 30% match any ID pattern, this is likely an ID column
            return id_pattern_count / len(sample) > 0.3
        except Exception as e:
            logger.warning(f"Error checking ID format: {str(e)}")
            return False
            
    
        """Identify the date column in the data with enhanced pattern matching"""
        # If date column is provided, use it
        if self.date_col is not None:
            self.detected_columns['date'] = self.date_col
            return True
        
        # Method 1: Check for explicit date columns using regex patterns
        date_scores = {}
        
        # Score date candidates
        for col in self.data.columns:
            try:
                col_lower = str(col).lower()
                score = 0
                
                # Skip columns with too many nulls
                null_percentage = self.data[col].isna().mean()
                if null_percentage > 0.7:
                    continue
                
                # Check column name against advanced regex patterns
                for pattern in self.regex_patterns['date']['name_patterns']:
                    if re.search(pattern, col_lower, re.IGNORECASE):
                        score += 10
                        break
                
                # If the column is already a datetime type, it's very likely a date
                if pd.api.types.is_datetime64_dtype(self.data[col].dtype):
                    score += 15
                    date_scores[col] = score
                    continue
                
                # Only check further if the column might be a date based on name or is in date candidates
                if score > 0 or col in self.column_classes['date_candidates']:
                    # Try to convert to datetime and check validity
                    try:
                        # Use a sample for performance
                        sample = self.data[col].dropna().head(500)
                        if len(sample) > 0:
                            # Try generic conversion
                            date_conversion = pd.to_datetime(sample, errors='coerce')
                            valid_pct = date_conversion.notna().mean()
                            
                            # If high conversion rate, this is likely a date
                            if valid_pct > 0.7:
                                score += 15 * valid_pct  # Up to 15 points for 100% valid
                                date_scores[col] = score
                                continue
                            
                            # If string column, try specific date formats
                            if pd.api.types.is_string_dtype(self.data[col]):
                                # Check against common date formats
                                date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', 
                                               '%m-%d-%Y', '%d-%m-%Y', '%Y%m%d', '%m%d%Y', '%d%m%Y']
                                
                                for date_format in date_formats:
                                    try:
                                        converted = pd.to_datetime(sample, format=date_format, errors='coerce')
                                        valid_format_pct = converted.notna().mean()
                                        
                                        if valid_format_pct > 0.7:
                                            score += 12 * valid_format_pct
                                            date_scores[col] = score
                                            break
                                    except:
                                        continue
                    except:
                        pass
                
                # Score based on column name keywords
                if score == 0:  # If we haven't already scored this column
                    # Check exact matches with date patterns
                    if col_lower in self.date_patterns:
                        score += 10
                    else:
                        # Check partial matches
                        for keyword in self.date_patterns:
                            if keyword in col_lower:
                                score += 5
                                break
                    
                    # Prefer columns with more obvious date names
                    priority_keywords = ['order_date', 'orderdate', 'invoice_date', 'transaction_date', 'date']
                    for keyword in priority_keywords:
                        if keyword in col_lower:
                            score += 3
                            break
                
                # Only add columns with non-zero scores
                if score > 0:
                    date_scores[col] = score
                
            except Exception as e:
                logger.debug(f"Error evaluating column '{col}' as date: {str(e)}")
                continue
        
        # Find best date candidate
        if date_scores:
            best_date_col = max(date_scores.items(), key=lambda x: x[1])[0]
            self.detected_columns['date'] = best_date_col
            logger.info(f"Date column identified: {best_date_col} (score: {date_scores[best_date_col]})")
            return True
        
        # Method 2: PRIORITIZE creating a composite date from year and month
        # Get year and month candidates
        year_candidates = self.column_classes['date_component_candidates']['year']
        month_candidates = self.column_classes['date_component_candidates']['month']
        
        # Check if we have both year and month columns
        if year_candidates and month_candidates:
            year_col = self._select_best_component('year', year_candidates)
            month_col = self._select_best_component('month', month_candidates)
            
            # Get day component if available
            day_candidates = self.column_classes['date_component_candidates']['day']
            day_col = self._select_best_component('day', day_candidates) if day_candidates else None
            
            # Store detected components
            if year_col:
                self.detected_columns['year'] = year_col
            if month_col:
                self.detected_columns['month'] = month_col
            if day_col:
                self.detected_columns['day'] = day_col
            
            # Only proceed if we have both year and month
            if year_col and month_col:
                logger.info(f"Found date components: Year({year_col}), Month({month_col}), Day({day_col if day_col else 'Not found'})")
                
                try:
                    # Create a new date column
                    comp_date_col = 'composite_date'
                    
                    if day_col:
                        # Format: YYYY-MM-DD
                        try:
                            self.data[comp_date_col] = pd.to_datetime(
                                self.data[year_col].astype(str) + '-' + 
                                self.data[month_col].astype(str).str.zfill(2) + '-' + 
                                self.data[day_col].astype(str).str.zfill(2),
                                errors='coerce'
                            )
                        except:
                            # Try alternative approach if the above fails
                            try:
                                self.data[comp_date_col] = pd.to_datetime({
                                    'year': self.data[year_col],
                                    'month': self.data[month_col],
                                    'day': self.data[day_col]
                                }, errors='coerce')
                            except Exception as e:
                                logger.warning(f"Error in second attempt to create date with day: {str(e)}")
                    else:
                        # Format: YYYY-MM-01 (default to first day of month)
                        try:
                            self.data[comp_date_col] = pd.to_datetime(
                                self.data[year_col].astype(str) + '-' + 
                                self.data[month_col].astype(str).str.zfill(2) + '-01',
                                errors='coerce'
                            )
                        except Exception as e:
                            logger.warning(f"Error creating date from string concat: {str(e)}")
                            # Try alternative approach if the above fails
                            try:
                                self.data[comp_date_col] = pd.to_datetime({
                                    'year': self.data[year_col],
                                    'month': self.data[month_col],
                                    'day': 1
                                }, errors='coerce')
                            except Exception as e:
                                logger.warning(f"Error in second attempt to create date: {str(e)}")
                    
                    # Verify the composite date is valid
                    valid_dates = self.data[comp_date_col].notna().mean()
                    if valid_dates > 0.7:  # At least 70% valid dates
                        self.detected_columns['date'] = comp_date_col
                        logger.info(f"Created composite date column from {year_col}, {month_col}{', ' + day_col if day_col else ''}")
                        logger.info(f"Composite date is {valid_dates:.1%} valid")
                        return True
                    else:
                        logger.warning(f"Created composite date column has too many invalid dates ({valid_dates:.1%} valid)")
                except Exception as e:
                    logger.warning(f"Error creating composite date: {str(e)}")
                    logger.debug(traceback.format_exc())
        
        # Method 3 (fallback): Create a date from just year if that's all we have and Method 2 failed
        if (not self.detected_columns.get('date') and 
            year_candidates and self.column_classes['date_component_candidates']['year']):
            year_col = self._select_best_component('year', year_candidates)
            if year_col:
                self.detected_columns['year'] = year_col
                logger.warning("Only found year column, but no month column to create a proper date")
                logger.info(f"Using year column {year_col} as a fallback")
                
                try:
                    # Create a date using year with January 1st
                    year_date_col = 'year_date'
                    self.data[year_date_col] = pd.to_datetime(
                        self.data[year_col].astype(str) + '-01-01',
                        errors='coerce'
                    )
                    
                    valid_dates = self.data[year_date_col].notna().mean()
                    if valid_dates > 0.7:
                        self.detected_columns['date'] = year_date_col
                        logger.info(f"Created date column from year column {year_col} (fallback)")
                        return True
                except Exception as e:
                    logger.warning(f"Error creating year-based date: {str(e)}")
        
        # Failed to identify a date column
        logger.warning("Could not identify date column. No suitable date column or components found.")
        return False
        """Identify the date column in the data with enhanced pattern matching"""
        # If date column is provided, use it
        if self.date_col is not None:
            self.detected_columns['date'] = self.date_col
            return True
        
        # Method 1: Check for explicit date columns using regex patterns
        date_scores = {}
        
        # Score date candidates
        for col in self.data.columns:
            try:
                col_lower = str(col).lower()
                score = 0
                
                # Skip columns with too many nulls
                null_percentage = self.data[col].isna().mean()
                if null_percentage > 0.7:
                    continue
                
                # Check column name against advanced regex patterns
                for pattern in self.regex_patterns['date']['name_patterns']:
                    if re.search(pattern, col_lower, re.IGNORECASE):
                        score += 10
                        break
                
                # If the column is already a datetime type, it's very likely a date
                if pd.api.types.is_datetime64_dtype(self.data[col].dtype):
                    score += 15
                    date_scores[col] = score
                    continue
                
                # Only check further if the column might be a date based on name or is in date candidates
                if score > 0 or col in self.column_classes['date_candidates']:
                    # Try to convert to datetime and check validity
                    try:
                        # Use a sample for performance
                        sample = self.data[col].dropna().head(500)
                        if len(sample) > 0:
                            # Try generic conversion
                            date_conversion = pd.to_datetime(sample, errors='coerce')
                            valid_pct = date_conversion.notna().mean()
                            
                            # If high conversion rate, this is likely a date
                            if valid_pct > 0.7:
                                score += 15 * valid_pct  # Up to 15 points for 100% valid
                                date_scores[col] = score
                                continue
                            
                            # If string column, try specific date formats
                            if pd.api.types.is_string_dtype(self.data[col]):
                                # Check against common date formats
                                date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', 
                                               '%m-%d-%Y', '%d-%m-%Y', '%Y%m%d', '%m%d%Y', '%d%m%Y']
                                
                                for date_format in date_formats:
                                    try:
                                        converted = pd.to_datetime(sample, format=date_format, errors='coerce')
                                        valid_format_pct = converted.notna().mean()
                                        
                                        if valid_format_pct > 0.7:
                                            score += 12 * valid_format_pct
                                            date_scores[col] = score
                                            break
                                    except:
                                        continue
                    except:
                        pass
                
                # Score based on column name keywords
                if score == 0:  # If we haven't already scored this column
                    # Check exact matches with date patterns
                    if col_lower in self.date_patterns:
                        score += 10
                    else:
                        # Check partial matches
                        for keyword in self.date_patterns:
                            if keyword in col_lower:
                                score += 5
                                break
                    
                    # Prefer columns with more obvious date names
                    priority_keywords = ['order_date', 'orderdate', 'invoice_date', 'transaction_date', 'date']
                    for keyword in priority_keywords:
                        if keyword in col_lower:
                            score += 3
                            break
                
                # Only add columns with non-zero scores
                if score > 0:
                    date_scores[col] = score
                
            except Exception as e:
                logger.debug(f"Error evaluating column '{col}' as date: {str(e)}")
                continue
        
        # Find best date candidate
        if date_scores:
            best_date_col = max(date_scores.items(), key=lambda x: x[1])[0]
            self.detected_columns['date'] = best_date_col
            logger.info(f"Date column identified: {best_date_col} (score: {date_scores[best_date_col]})")
            return True
        
        # Method 2: Check for separate year, month, day columns
        # Get year, month, day candidates
        year_candidates = self.column_classes['date_component_candidates']['year']
        month_candidates = self.column_classes['date_component_candidates']['month']
        day_candidates = self.column_classes['date_component_candidates']['day']
        
        # Take the first of each if available, or score them if multiple
        year_col = self._select_best_component('year', year_candidates) if year_candidates else None
        month_col = self._select_best_component('month', month_candidates) if month_candidates else None
        day_col = self._select_best_component('day', day_candidates) if day_candidates else None
        
        # Store detected components
        if year_col:
            self.detected_columns['year'] = year_col
        if month_col:
            self.detected_columns['month'] = month_col
        if day_col:
            self.detected_columns['day'] = day_col
        
        # Try to create a composite date if we have sufficient components
        if year_col and month_col:
            logger.info(f"Found date components: Year({year_col}), Month({month_col}), Day({day_col if day_col else 'Not found'})")
            
            try:
                # Create a new date column
                comp_date_col = 'composite_date'
                
                if day_col:
                    # Format: YYYY-MM-DD
                    try:
                        self.data[comp_date_col] = pd.to_datetime(
                            self.data[year_col].astype(str) + '-' + 
                            self.data[month_col].astype(str).str.zfill(2) + '-' + 
                            self.data[day_col].astype(str).str.zfill(2),
                            errors='coerce'
                        )
                    except:
                        # Try alternative approach if the above fails
                        self.data[comp_date_col] = pd.to_datetime({
                            'year': self.data[year_col],
                            'month': self.data[month_col],
                            'day': self.data[day_col]
                        }, errors='coerce')
                else:
                    # Format: YYYY-MM-01 (default to first day of month)
                    try:
                        self.data[comp_date_col] = pd.to_datetime(
                            self.data[year_col].astype(str) + '-' + 
                            self.data[month_col].astype(str).str.zfill(2) + '-01',
                            errors='coerce'
                        )
                    except:
                        # Try alternative approach if the above fails
                        self.data[comp_date_col] = pd.to_datetime({
                            'year': self.data[year_col],
                            'month': self.data[month_col],
                            'day': 1
                        }, errors='coerce')
                
                # Verify the composite date is valid
                valid_dates = self.data[comp_date_col].notna().mean()
                if valid_dates > 0.7:  # At least 70% valid dates
                    self.detected_columns['date'] = comp_date_col
                    logger.info(f"Created composite date column from {year_col}, {month_col}{', ' + day_col if day_col else ''}")
                    return True
                else:
                    logger.warning(f"Created composite date column has too many invalid dates ({valid_dates:.1%} valid)")
            except Exception as e:
                logger.warning(f"Error creating composite date: {str(e)}")
        
        # Method 3: Create a date from a single year column if that's all we have
        if year_col and not month_col and not self.detected_columns['date']:
            try:
                # Create a date using year with January 1st
                year_date_col = 'year_date'
                self.data[year_date_col] = pd.to_datetime(
                    self.data[year_col].astype(str) + '-01-01',
                    errors='coerce'
                )
                
                valid_dates = self.data[year_date_col].notna().mean()
                if valid_dates > 0.7:
                    self.detected_columns['date'] = year_date_col
                    logger.info(f"Created date column from year column {year_col}")
                    return True
            except Exception as e:
                logger.warning(f"Error creating year-based date: {str(e)}")
        
        # Failed to identify a date column
        logger.warning("Could not identify date column. Forecasting may be limited.")
        return False
    
    def _select_best_component(self, component_type, candidates):
        """Select the best column from multiple candidates for date components"""
        if not candidates:
            return None
            
        if len(candidates) == 1:
            return candidates[0]
            
        # Score candidates
        scores = {}
        for col in candidates:
            score = 0
            col_lower = str(col).lower()
            
            # Check for exact match with component name
            if col_lower == component_type:
                score += 10
            
            # Check for name patterns in regex_patterns
            for pattern in self.regex_patterns[component_type]['name_patterns']:
                if re.search(pattern, col_lower, re.IGNORECASE):
                    score += 8
                    break
            
            # Check content validity
            valid_values = self.data[col].dropna()
            if len(valid_values) > 0:
                # Check if values match expected ranges
                if component_type == 'year':
                    if valid_values.min() >= 1900 and valid_values.max() <= datetime.now().year + 5:
                        score += 5
                elif component_type == 'month':
                    if valid_values.min() >= 1 and valid_values.max() <= 12:
                        score += 5
                elif component_type == 'day':
                    if valid_values.min() >= 1 and valid_values.max() <= 31:
                        score += 5
            
            scores[col] = score
        
        # Return the column with the highest score
        return max(scores.items(), key=lambda x: x[1])[0]
            
    def identify_product_column(self):
        """Enhanced method to identify product column with regex pattern matching"""
        # If product column is provided, use it
        if self.product_col is not None:
            self.detected_columns['product'] = self.product_col
            return True
        
        # Score candidates for different product-related roles
        product_id_candidates = {}
        product_name_candidates = {}
        product_category_candidates = {}
        
        # Collect all possible candidates for product
        all_candidates = set()
        all_candidates.update(self.column_classes['categorical_candidates'])
        all_candidates.update(self.column_classes['text_candidates'])
        all_candidates.update(self.column_classes['id_candidates'])
        
        # Remove date column from candidates if detected
        if self.detected_columns['date']:
            all_candidates.discard(self.detected_columns['date'])
        
        # Convert to list for iteration
        relevant_candidates = list(all_candidates)
        
        # First pass - check column names against advanced regex patterns
        for col in relevant_candidates:
            try:
                col_lower = str(col).lower()
                
                # Check product ID patterns
                for pattern in self.regex_patterns['product']['name_patterns']:
                    if re.search(pattern, col_lower, re.IGNORECASE):
                        product_id_candidates[col] = 10
                        break
                
                # Check product name patterns
                for pattern in self.regex_patterns['product_name']['name_patterns']:
                    if re.search(pattern, col_lower, re.IGNORECASE):
                        product_name_candidates[col] = 10
                        break
                
                # Check product category patterns
                for pattern in self.regex_patterns['product_category']['name_patterns']:
                    if re.search(pattern, col_lower, re.IGNORECASE):
                        product_category_candidates[col] = 10
                        break
                
                # Check generic product-related terms if not already categorized
                if col not in product_id_candidates and col not in product_name_candidates and col not in product_category_candidates:
                    # Direct keyword matching
                    for keyword in self.product_patterns:
                        if keyword in col_lower:
                            # Determine which type it might be
                            if self._has_id_pattern(col_lower) or col in self.column_classes['id_candidates']:
                                product_id_candidates[col] = 8
                            elif pd.api.types.is_string_dtype(self.data[col]) and col in self.column_classes['text_candidates']:
                                product_name_candidates[col] = 8
                            else:
                                product_category_candidates[col] = 8
                            break
            except Exception as e:
                logger.warning(f"Error in product column name matching for '{col}': {str(e)}")
                continue
        
        # Second pass - analyze column content characteristics
        for col in relevant_candidates:
            try:
                # Skip columns already with high confidence
                if (col in product_id_candidates and product_id_candidates[col] >= 9) or \
                   (col in product_name_candidates and product_name_candidates[col] >= 9) or \
                   (col in product_category_candidates and product_category_candidates[col] >= 9):
                    continue
                
                # Analyze column contents
                unique_count = self.data[col].nunique()
                total_count = len(self.data)
                
                # Calculate uniqueness ratio
                if total_count > 0:
                    unique_ratio = unique_count / total_count
                else:
                    continue
                
                # Sample values for content analysis
                sample_size = min(50, unique_count)
                if sample_size <= 0:
                    continue
                    
                sample = self.data[col].dropna().sample(sample_size).astype(str)
                
                # Skip empty samples
                if sample.empty:
                    continue
                
                # Check product ID patterns in values
                id_pattern_match = False
                for pattern in self.regex_patterns['product']['id_patterns']:
                    pattern_match_ratio = sample.str.match(pattern).mean()
                    if pattern_match_ratio > 0.7:  # If > 70% match the pattern
                        id_pattern_match = True
                        break
                
                # Calculate text characteristics for products
                avg_length = sample.str.len().mean()
                word_counts = sample.str.split().str.len()
                avg_words = word_counts.mean() if not word_counts.empty else 0
                
                # Product ID characteristics
                if id_pattern_match or self._check_if_id_format(self.data[col]):
                    score = 7 + min(3, (1 - abs(0.1 - unique_ratio)) * 5)  # Ideal ratio around 0.1
                    product_id_candidates[col] = max(product_id_candidates.get(col, 0), score)
                
                # Product name characteristics
                elif pd.api.types.is_string_dtype(self.data[col]) and \
                     (0.01 <= unique_ratio <= 0.9) and \
                     (avg_length > 10 or avg_words > 2):
                    
                    # Check for title case patterns (typical for product names)
                    title_case_ratio = 0
                    for val in sample:
                        words = val.split()
                        if words and any(word and word[0].isupper() for word in words):
                            title_case_ratio += 1
                    title_case_ratio /= len(sample)
                    
                    score = 6 + (title_case_ratio * 2) + min(2, avg_words / 2)
                    product_name_candidates[col] = max(product_name_candidates.get(col, 0), score)
                
                # Product category characteristics
                elif (0.001 <= unique_ratio <= 0.1) and unique_count >= 2:
                    score = 6 + min(4, (1 - unique_ratio * 10) * 4)  # Fewer categories get higher scores
                    product_category_candidates[col] = max(product_category_candidates.get(col, 0), score)
            except Exception as e:
                logger.warning(f"Error analyzing column '{col}' for product detection: {str(e)}")
                continue
        
        # Store the detected columns
        if product_id_candidates:
            best_id_col = max(product_id_candidates.items(), key=lambda x: x[1])[0]
            self.detected_columns['product_id'] = best_id_col
            logger.info(f"Product ID column identified: {best_id_col} (score: {product_id_candidates[best_id_col]:.1f}/10)")
        
        if product_name_candidates:
            best_name_col = max(product_name_candidates.items(), key=lambda x: x[1])[0]
            self.detected_columns['product_name'] = best_name_col
            logger.info(f"Product name column identified: {best_name_col} (score: {product_name_candidates[best_name_col]:.1f}/10)")
        
        if product_category_candidates:
            best_category_col = max(product_category_candidates.items(), key=lambda x: x[1])[0]
            self.detected_columns['product_category'] = best_category_col
            logger.info(f"Product category column identified: {best_category_col} (score: {product_category_candidates[best_category_col]:.1f}/10)")
        
        # Decide which column to use as the primary product column
        # Preference order: ID > Name > Category
        if self.detected_columns['product_id']:
            self.detected_columns['product'] = self.detected_columns['product_id']
        elif self.detected_columns['product_name']:
            self.detected_columns['product'] = self.detected_columns['product_name']
        elif self.detected_columns['product_category']:
            self.detected_columns['product'] = self.detected_columns['product_category']
        else:
            # Last resort: look for any column with product-related keywords
            for col in self.data.columns:
                col_lower = str(col).lower()
                for keyword in self.product_patterns:
                    if keyword in col_lower and col != self.detected_columns['date']:
                        self.detected_columns['product'] = col
                        logger.info(f"Product column identified as fallback: {col}")
                        return True
                    
            logger.warning("Could not identify product column. Please specify it manually.")
            return False
        
        logger.info(f"Primary product column selected: {self.detected_columns['product']}")
        return True
                
    def _identify_additional_columns(self):
        """Identify additional important columns beyond date, product, and sales"""
        # Identify price column
        if not self.detected_columns.get('price'):
            price_candidates = {}
            for col in self.column_classes['numeric_candidates']:
                col_lower = str(col).lower()
                
                # Skip already identified columns
                if col in [self.detected_columns.get(key) for key in self.detected_columns if self.detected_columns.get(key)]:
                    continue
                
                # Check for exact price column name matches
                if col_lower in ['price', 'unit_price', 'item_price']:
                    price_candidates[col] = 15
                elif 'price' in col_lower:
                    price_candidates[col] = 10
                    
            if price_candidates:
                best_price = max(price_candidates.items(), key=lambda x: x[1])[0]
                self.detected_columns['price'] = best_price
                logger.info(f"Price column identified: {best_price}")
        
        # Identify quantity column
        if not self.detected_columns.get('quantity'):
            quantity_candidates = {}
            for col in self.column_classes['numeric_candidates']:
                col_lower = str(col).lower()
                
                # Skip already identified columns
                if col in [self.detected_columns.get(key) for key in self.detected_columns if self.detected_columns.get(key)]:
                    continue
                
                # Check for exact quantity column name matches
                if col_lower in ['quantity', 'qty', 'units', 'count']:
                    quantity_candidates[col] = 15
                elif any(keyword in col_lower for keyword in ['quantity', 'qty', 'units', 'count']):
                    quantity_candidates[col] = 10
                # Check if the column contains mostly small integers (typical of quantity)
                else:
                    try:
                        non_null = self.data[col].dropna()
                        if len(non_null) > 0:
                            # Check if values are mostly integers
                            is_mostly_int = (non_null % 1 == 0).mean() > 0.9
                            # Check if values are small positive integers
                            is_small_range = (non_null >= 0).all() and non_null.mean() < 100
                            if is_mostly_int and is_small_range:
                                quantity_candidates[col] = 8
                    except:
                        pass
                    
            if quantity_candidates:
                best_quantity = max(quantity_candidates.items(), key=lambda x: x[1])[0]
                self.detected_columns['quantity'] = best_quantity
                logger.info(f"Quantity column identified: {best_quantity}")
        
        # Identify year, month, day columns
        for component in ['year', 'month', 'day']:
            if not self.detected_columns.get(component) and self.column_classes['date_component_candidates'].get(component):
                candidates = self.column_classes['date_component_candidates'][component]
                if candidates:
                    self.detected_columns[component] = candidates[0]
                    logger.info(f"{component.capitalize()} column identified: {candidates[0]}")
                    
        # Identify cost column
        if not self.detected_columns.get('cost'):
            cost_candidates = {}
            for col in self.column_classes['numeric_candidates']:
                col_lower = str(col).lower()
                
                # Skip already identified columns
                if col in [self.detected_columns.get(key) for key in self.detected_columns if self.detected_columns.get(key)]:
                    continue
                
                # Check for cost column name matches
                if col_lower in ['cost', 'unit_cost', 'total_cost', 'cogs']:
                    cost_candidates[col] = 15
                elif 'cost' in col_lower or 'expense' in col_lower:
                    cost_candidates[col] = 10
                    
            if cost_candidates:
                best_cost = max(cost_candidates.items(), key=lambda x: x[1])[0]
                self.detected_columns['cost'] = best_cost
                logger.info(f"Cost column identified: {best_cost}")
        
        # Identify profit column
        if not self.detected_columns.get('profit'):
            profit_candidates = {}
            for col in self.column_classes['numeric_candidates']:
                col_lower = str(col).lower()
                
                # Skip already identified columns
                if col in [self.detected_columns.get(key) for key in self.detected_columns if self.detected_columns.get(key)]:
                    continue
                
                # Check for profit column name matches
                if col_lower in ['profit', 'margin', 'gross_profit', 'net_profit']:
                    profit_candidates[col] = 15
                elif 'profit' in col_lower or 'margin' in col_lower or 'earnings' in col_lower:
                    profit_candidates[col] = 10
                    
                # Check if column could be profit based on values
                if col_lower not in profit_candidates and self.detected_columns.get('sales') and self.detected_columns.get('cost'):
                    try:
                        # Check if this column might be sales - cost (profit)
                        sales_col = self.detected_columns.get('sales')
                        cost_col = self.detected_columns.get('cost')
                        
                        # Calculate correlation with sales-cost
                        if sales_col and cost_col:
                            sample_size = min(1000, len(self.data))
                            indices = np.random.choice(len(self.data), sample_size, replace=False)
                            
                            # Calculate expected profit
                            expected_profit = self.data[sales_col].iloc[indices] - self.data[cost_col].iloc[indices]
                            
                            # Calculate correlation
                            corr_df = pd.DataFrame({
                                'expected': expected_profit,
                                'actual': self.data[col].iloc[indices]
                            }).dropna()
                            
                            if len(corr_df) > 10:
                                correlation = corr_df['expected'].corr(corr_df['actual'])
                                if correlation > 0.8:
                                    profit_candidates[col] = 12
                    except:
                        pass
                    
            if profit_candidates:
                best_profit = max(profit_candidates.items(), key=lambda x: x[1])[0]
                self.detected_columns['profit'] = best_profit
                logger.info(f"Profit column identified: {best_profit}")
        
        # Identify revenue column
        if not self.detected_columns.get('revenue'):
            revenue_candidates = {}
            for col in self.column_classes['numeric_candidates']:
                col_lower = str(col).lower()
                
                # Skip already identified columns
                if col in [self.detected_columns.get(key) for key in self.detected_columns if self.detected_columns.get(key)]:
                    continue
                
                # Check for revenue column name matches
                if col_lower in ['revenue', 'total_revenue', 'sales', 'total_sales']:
                    revenue_candidates[col] = 15
                elif 'revenue' in col_lower or 'sales' in col_lower or 'income' in col_lower:
                    revenue_candidates[col] = 10
                    
            if revenue_candidates:
                best_revenue = max(revenue_candidates.items(), key=lambda x: x[1])[0]
                self.detected_columns['revenue'] = best_revenue
                logger.info(f"Revenue column identified: {best_revenue}")
        
        # Identify customer column
        if not self.detected_columns.get('customer'):
            customer_candidates = {}
            for col in self.data.columns:
                col_lower = str(col).lower()
                
                # Skip already identified columns
                if col in [self.detected_columns.get(key) for key in self.detected_columns if self.detected_columns.get(key)]:
                    continue
                
                # Check for customer column name matches
                if col_lower in ['customer', 'customer_id', 'customer_name', 'client', 'client_id']:
                    customer_candidates[col] = 15
                elif 'customer' in col_lower or 'client' in col_lower or 'buyer' in col_lower:
                    customer_candidates[col] = 10
                    
                # Check if this might be a customer ID column based on format
                if col in self.column_classes['id_candidates'] and ('cust' in col_lower or 'client' in col_lower):
                    customer_candidates[col] = 12
                    
            if customer_candidates:
                best_customer = max(customer_candidates.items(), key=lambda x: x[1])[0]
                self.detected_columns['customer'] = best_customer
                logger.info(f"Customer column identified: {best_customer}")
        
        # Identify location column
        if not self.detected_columns.get('location'):
            location_candidates = {}
            location_keywords = ['location', 'region', 'country', 'state', 'province', 'city', 
                                 'address', 'area', 'territory', 'zone', 'postal_code', 'zip']
            
            for col in self.data.columns:
                col_lower = str(col).lower()
                
                # Skip already identified columns
                if col in [self.detected_columns.get(key) for key in self.detected_columns if self.detected_columns.get(key)]:
                    continue
                
                # Check for location column name matches
                if col_lower in location_keywords:
                    location_candidates[col] = 15
                elif any(keyword in col_lower for keyword in location_keywords):
                    location_candidates[col] = 10
                    
                # Try to detect location columns by looking at values (if string column)
                if col_lower not in location_candidates and pd.api.types.is_string_dtype(self.data[col]):
                    # Sample values to check for location patterns
                    try:
                        sample = self.data[col].dropna().sample(min(100, len(self.data[col].dropna()))).astype(str)
                        
                        # Common state abbreviations pattern
                        state_pattern = r'^[A-Z]{2}$'
                        state_matches = sample.str.match(state_pattern).mean()
                        
                        # City pattern (capitalized words)
                        city_pattern = r'^[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*$'
                        city_matches = sample.str.match(city_pattern).mean()
                        
                        # Zip code pattern
                        zip_pattern = r'^\d{5}(?:-\d{4})?$'
                        zip_matches = sample.str.match(zip_pattern).mean()
                        
                        # Address pattern (contains numbers and street names)
                        address_pattern = r'^\d+\s+[A-Za-z\s]+(?:St|Ave|Rd|Blvd|Drive|Lane|Way|Court|Plaza|Square)'
                        address_matches = sample.str.contains(address_pattern, regex=True).mean()
                        
                        # Country pattern
                        country_pattern = r'^(?:USA|United States|Canada|Mexico|UK|United Kingdom|France|Germany|China|Japan|Australia)$'
                        country_matches = sample.str.match(country_pattern, case=False).mean()
                        
                        # If any pattern has high match rate, likely a location column
                        max_match = max(state_matches, city_matches, zip_matches, address_matches, country_matches)
                        if max_match > 0.5:
                            location_candidates[col] = 8 + (max_match * 2)
                    except:
                        pass
                    
            if location_candidates:
                best_location = max(location_candidates.items(), key=lambda x: x[1])[0]
                self.detected_columns['location'] = best_location
                logger.info(f"Location column identified: {best_location}")
        
        # Handle special case for product_name if it wasn't identified correctly earlier
        if self.detected_columns.get('product_name') and not any(col_lower == 'product_name' for col_lower in [str(col).lower() for col in self.data.columns]):
            # Try to find a better product_name column
            product_name_candidates = {}
            for col in self.data.columns:
                col_lower = str(col).lower()
                
                # Skip already identified columns except the current product_name
                if col in [self.detected_columns.get(key) for key in self.detected_columns if self.detected_columns.get(key) and key != 'product_name']:
                    continue
                
                # Check for exact product_name column matches
                if col_lower == 'product_name':
                    product_name_candidates[col] = 15
                elif 'product' in col_lower and ('name' in col_lower or 'desc' in col_lower or 'title' in col_lower):
                    product_name_candidates[col] = 12
            
            if product_name_candidates:
                best_product_name = max(product_name_candidates.items(), key=lambda x: x[1])[0]
                self.detected_columns['product_name'] = best_product_name
                logger.info(f"Product name column re-identified: {best_product_name}")
        """Identify all important columns in the data"""
        try:
            # First, scan all columns to classify their data types and potential roles
            logger.info("Pre-classifying columns by data type and characteristics...")
            self._classify_columns()
            logger.info("Column classification complete.")
            
            # Track identification success
            identification_success = True
            
            # Identify columns in a specific order with proper error handling
            try:
                logger.info("Attempting to identify date column...")
                success = self.identify_date_column()
                if not success:
                    logger.warning("Failed to identify date column, but continuing...")
                    identification_success = False
            except Exception as e:
                logger.error(f"Error in date column identification: {str(e)}")
                logger.debug(traceback.format_exc())
                identification_success = False
            
            try:
                logger.info("Attempting to identify product column...")
                success = self.identify_product_column()
                if not success:
                    logger.warning("Failed to identify product column, but continuing...")
                    identification_success = False
            except Exception as e:
                logger.error(f"Error in product column identification: {str(e)}")
                logger.debug(traceback.format_exc())
                identification_success = False
            
            try:
                logger.info("Attempting to identify sales column...")
                success = self.identify_sales_column()
                if not success:
                    logger.warning("Failed to identify sales column, but continuing...")
                    identification_success = False
            except Exception as e:
                logger.error(f"Error in sales column identification: {str(e)}")
                logger.debug(traceback.format_exc())
                identification_success = False
            
            # Additional identifications for columns we're missing
            try:
                self._identify_additional_columns()
            except Exception as e:
                logger.warning(f"Error in identifying additional columns: {str(e)}")
                # Don't fail if additional columns can't be identified
            
            # Update instance variables with detected columns
            self.date_col = self.detected_columns['date']
            self.product_col = self.detected_columns['product']
            self.sales_col = self.detected_columns['sales']
            
            # Log the identified columns
            logger.info("\nColumn Detection Summary:")
            for col_type, col_name in self.detected_columns.items():
                if col_name:
                    logger.info(f"  {col_type.capitalize()}: {col_name}")
            
            return identification_success
        except Exception as e:
            logger.error(f"Unhandled error in column identification: {str(e)}")
            logger.debug(traceback.format_exc())
            return False

    
    def _initialize_regex_patterns(self):
        """Initialize advanced regex patterns for column detection"""
        patterns = {
            # Date patterns
            'date': {
                # Full date column patterns
                'name_patterns': [
                    r'\b(?:order|invoice|transaction|sales|purchase|ship(?:ping|ment)?|delivery)[\s_-]*(?:date|time|day|dt)\b',
                    r'\b(?:date|dt)[\s_-]*(?:of[\s_-]*(?:order|invoice|transaction|sale|purchase|ship(?:ping|ment)?|delivery))?\b',
                    r'\btimestamp\b',
                    r'\b(?:created|modified|entered|posted|recorded)[\s_-]*(?:date|time|on|at)\b',
                ],
                # Format detection patterns
                'format_patterns': {
                    'iso_date': r'^\d{4}-\d{2}-\d{2}$',                        # YYYY-MM-DD
                    'us_date': r'^(?:\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4})$',       # MM/DD/YYYY or DD/MM/YYYY
                    'year_month_day': r'^(\d{4})(\d{2})(\d{2})$',              # YYYYMMDD
                    'timestamp': r'^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}',         # YYYY-MM-DD HH:MM
                }
            },
            
            # Date component patterns
            'year': {
                'name_patterns': [
                    r'\b(?:fiscal[\s_-]*)?year\b',
                    r'\b(?:yr|fy|yyyy)\b',
                    r'\byear[\s_-]*(?:of[\s_-]*(?:order|invoice|transaction|sale|purchase))?\b',
                ],
                'value_pattern': r'^\d{4}$'  # Four-digit year
            },
            
            'month': {
                'name_patterns': [
                    r'\bmonth\b',
                    r'\b(?:mon|mm|mo|mnth)\b',
                    r'\bmonth[\s_-]*(?:of[\s_-]*(?:order|invoice|transaction|sale|purchase))?\b',
                ],
                'value_pattern': r'^(?:[1-9]|1[0-2])$'  # 1-12
            },
            
            'day': {
                'name_patterns': [
                    r'\bday\b',
                    r'\b(?:dy|dd|dom)\b',
                    r'\bday[\s_-]*(?:of[\s_-]*(?:order|invoice|transaction|sale|purchase|month))?\b',
                ],
                'value_pattern': r'^(?:[1-9]|[12][0-9]|3[01])$'  # 1-31
            },
            
            # Product patterns
            'product': {
                'name_patterns': [
                    r'\b(?:product|item|article|good|merchandise|sku)[\s_-]*(?:id|code|number|no|identifier|key)?\b',
                    r'\b(?:prod|prd|itm|art)[\s_-]*(?:id|code|num|no)?\b',
                ],
                'id_patterns': [
                    r'^[A-Z]{1,3}[-_]?\d{3,}$',            # ABC123, A-123, etc.
                    r'^\d{3,}[-_]?[A-Z]{1,3}$',            # 123ABC, 123-A, etc.
                    r'^[A-Z0-9]{2,3}[-_]?\d{4,}$',         # AB1234, XYZ9876, etc.
                    r'^(?:[A-Z0-9]{2,5}[-_]){1,2}[A-Z0-9]{2,5}$',  # AB-12-XY, 123-ABC-45, etc.
                ]
            },
            
            'product_name': {
                'name_patterns': [
                    r'\b(?:product|item|article|good|merchandise)[\s_-]*(?:name|title|label|description|desc)\b',
                    r'\b(?:prod|prd|itm|art)[\s_-]*(?:name|desc)\b',
                    r'\bdescription\b',
                    r'\b(?:name|title|label)[\s_-]*(?:of[\s_-]*(?:product|item|article|good|merchandise))?\b',
                ],
                'value_patterns': [
                    r'[A-Z][a-z]+\s+[A-Z][a-z]+',          # Title Case Multi Word
                    r'[A-Z][a-z]+\s+[a-z]+',               # Title case followed by lowercase
                ]
            },
            
            'product_category': {
                'name_patterns': [
                    r'\b(?:product|item)[\s_-]*(?:category|cat|type|group|class|classification|family)\b',
                    r'\b(?:category|cat|type|group|class|classification|family)[\s_-]*(?:of[\s_-]*(?:product|item|article|good|merchandise))?\b',
                ],
                'value_patterns': [  # Added value_patterns for product_category
                    r'^(?:electronics|clothing|food|beverage|furniture|home|garden|tools|automotive|beauty|health|toys|sports|books|music)$'
                ]
            },
            
            # Sales/financial patterns
            'sales': {
                'name_patterns': [
                    r'\b(?:sales|revenue|income)[\s_-]*(?:amount|value|total|sum)?\b',
                    r'\b(?:total|gross)[\s_-]*(?:sales|revenue|income|amount|value|price)\b',
                    r'\b(?:extended|line)[\s_-]*(?:amount|price|total|value)\b',
                    r'\b(?:sales|revenue)[\s_-]*(?:before[\s_-]*(?:tax|discount))?\b',
                ],
                'value_patterns': [  # Added value_patterns for sales
                    r'^\d+(?:\.\d+)?$'  # Numeric values with optional decimals
                ]
            },
            
            'quantity': {
                'name_patterns': [
                    r'\b(?:quantity|qty|count|units|pieces|volume|number)[\s_-]*(?:ordered|sold|purchased|shipped)?\b',
                    r'\b(?:order|sales|purchase|shipment)[\s_-]*(?:quantity|qty|count|units|volume)\b',
                    r'\b(?:number|count)[\s_-]*(?:of[\s_-]*(?:items|units|pieces))?\b',
                ],
                'value_pattern': r'^\d+$'  # Integer values
            },
            
            'price': {
                'name_patterns': [
                    r'\b(?:unit|single|per[\s_-]*item)[\s_-]*(?:price|cost|value|amount|rate)\b',
                    r'\b(?:price|cost|fee|charge|rate)[\s_-]*(?:per[\s_-]*(?:unit|item|piece))?\b',
                    r'\b(?:item|unit)[\s_-]*(?:price|cost|value|amount|rate)\b',
                ],
                'value_patterns': [  # Added value_patterns for price
                    r'^\d+(?:\.\d+)?$'  # Numeric values with decimals
                ]
            },
            
            'cost': {
                'name_patterns': [
                    r'\b(?:cost|expense|expenditure)[\s_-]*(?:amount|value|total|sum)?\b',
                    r'\b(?:total|gross)[\s_-]*(?:cost|expense|expenditure)\b',
                    r'\b(?:unit|item)[\s_-]*cost\b',
                    r'\bcogs\b',  # Cost of goods sold
                    r'\b(?:purchase|wholesale|supplier)[\s_-]*(?:price|cost|amount)\b',
                ],
                'value_patterns': [  # Added value_patterns for cost
                    r'^\d+(?:\.\d+)?$'  # Numeric values with decimals
                ]
            },
            
            'profit': {
                'name_patterns': [
                    r'\b(?:profit|margin|gain|earnings|income)[\s_-]*(?:amount|value|total|sum)?\b',
                    r'\b(?:gross|net)[\s_-]*(?:profit|margin|gain|earnings|income)\b',
                    r'\b(?:profit|margin|gain)[\s_-]*(?:per[\s_-]*(?:unit|item|piece))?\b',
                ],
                'value_patterns': [  # Added value_patterns for profit
                    r'^-?\d+(?:\.\d+)?$'  # Numeric values with optional negative sign and decimals
                ]
            },
            
            # Customer patterns
            'customer': {
                'name_patterns': [
                    r'\b(?:customer|client|buyer|account|purchaser)[\s_-]*(?:id|code|number|no|identifier|key|name)?\b',
                    r'\b(?:cust|clnt|acct)[\s_-]*(?:id|code|num|no|name)?\b',
                ],
                'value_patterns': [  # Added value_patterns for customer
                    r'^[A-Z]{1,3}-\d{4,10}$',
                    r'^CUST\d{4,}$',
                    r'^\d{5,10}$'  # Simple numeric customer IDs
                ]
            },
            
            # Location patterns
            'location': {
                'name_patterns': [
                    r'\b(?:location|region|country|state|province|city|area|territory|zone|district)[\s_-]*(?:id|code|name)?\b',
                    r'\b(?:ship(?:ping)?|delivery)[\s_-]*(?:location|address|destination)\b',
                    r'\b(?:postal|zip)[\s_-]*code\b',
                ],
                'value_patterns': [  # Added value_patterns for location
                    r'^[A-Z]{2}$',  # State codes
                    r'^\d{5}(?:-\d{4})?$',  # Zip codes
                    r'^[A-Z][a-z]+(?:,\s*[A-Z]{2})?$'  # City, State format
                ],
                'revenue': {
                   'name_patterns': [
                       r'\b(?:revenue|income|turnover|earnings|proceeds)\b',
                       r'\b(?:total|gross|net)[\s_-]*(?:revenue|income|turnover|earnings)\b',
                       r'\b(?:sales|revenue)[\s_-]*(?:volume|figure|number|total)\b',
                       r'(?i)^(?:revenue|income|turnover|sales)$',
                   ],
                   'value_patterns': [r'^\d+(?:\.\d+)?$']
               },
               'cost': {
                   'name_patterns': [
                       r'\b(?:cost|expense|expenditure|outlay|outgo)\b',
                       r'\b(?:total|unit|production|operational)[\s_-]*(?:cost|expense|expenditure)\b',
                       r'\b(?:cost[\s_-]*of[\s_-]*(?:goods|sales|revenue|production))\b',
                       r'\bcogs\b',
                       r'(?i)^(?:cost|expense|expenditure|cogs)$',
                   ],
                   'value_patterns': [r'^\d+(?:\.\d+)?$']
               },
               'profit': {
                   'name_patterns': [
                       r'\b(?:profit|margin|gain|earnings|benefit)\b',
                       r'\b(?:gross|net|operating)[\s_-]*(?:profit|margin|income|earnings)\b',
                       r'\b(?:profit[\s_-]*(?:margin|rate|ratio|percentage|pct|%))\b',
                       r'(?i)^(?:profit|margin|gain|earnings)$',
                   ],
                   'value_patterns': [r'^-?\d+(?:\.\d+)?$']
               },
               'customer': {
                   'name_patterns': [
                       r'\b(?:customer|client|buyer|purchaser|account)[\s_-]*(?:id|name|number|code|reference)?\b',
                       r'\b(?:cust|client)[\s_-]*(?:id|no|name|ref)\b',
                       r'\b(?:account[\s_-]*(?:id|name|number|holder))\b',
                       r'(?i)^(?:customer|client|buyer|account)$',
                   ],
                   'value_patterns': [r'^[\w\d\s-]+$']
               },
               'saled_units': {
                   'name_patterns': [
                       r'\b(?:saled|sold|sales|sale)[\s_-]*(?:units?|quantity|volume|count|pieces|items)\b',
                       r'\b(?:units?|quantity|volume|count)[\s_-]*(?:saled|sold|sales|sale)\b',
                       r'\b(?:total|gross|net)[\s_-]*(?:units?|quantity)[\s_-]*(?:saled|sold)\b',
                       r'(?i)^(?:units|unit|quantity|qty)$',
                   ],
                   'value_patterns': [r'^\d+$']
               }
                
            },
        }
        
        return patterns
    
    def _check_if_id_format(self, series):
        """Check if a series contains values that match typical ID formats using advanced patterns"""
        # Safety check
        if series.empty:
            return False
            
        try:
            # Sample some values to check
            sample_size = min(50, len(series))
            if sample_size == 0:
                return False
                
            sample = series.dropna().sample(sample_size).astype(str)
            
            # Count how many match ID patterns
            id_pattern_count = 0
            
            # Use more sophisticated ID patterns
            id_patterns = [
                # Single letter followed by numbers: A123, B456
                r'^[A-Za-z]\d{2,}$',
                
                # Multiple letters followed by numbers: ABC123, XYZ456
                r'^[A-Za-z]{2,}\d{2,}$',
                
                # Numbers followed by letters: 123A, 456XYZ
                r'^\d{2,}[A-Za-z]+$',
                
                # Combinations with separators: AB-123, 123-XY, A-1-B
                r'^[A-Za-z0-9]+-[A-Za-z0-9]+(?:-[A-Za-z0-9]+)?$',
                
                # UUIDs and similar formats
                r'^[A-Fa-f0-9]{8}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{4}-?[A-Fa-f0-9]{12}$',
                
                # Alphanumeric with consistent pattern and length
                r'^[A-Za-z0-9]{4,12}$'
            ]
            
            # Additional pattern check for very specific ID formats
            # Check for consistent patterns across the sample
            if not sample.empty:
                patterns = []
                for val in sample:
                    if not isinstance(val, str):
                        continue
                        
                    # Create a pattern representation of the value
                    pattern = ''
                    for char in val:
                        if char in string.ascii_uppercase:
                            pattern += 'U'
                        elif char in string.ascii_lowercase:
                            pattern += 'L'
                        elif char in string.digits:
                            pattern += 'D'
                        elif char in '-_/':
                            pattern += 'S'  # Separator
                        else:
                            pattern += 'O'  # Other
                    patterns.append(pattern)
                
                # Count pattern frequencies
                if patterns:  # Check if patterns list is not empty
                    pattern_counter = Counter(patterns)
                    most_common_pattern, count = pattern_counter.most_common(1)[0] if pattern_counter else (None, 0)
                    
                    # If most common pattern appears in at least 70% of samples
                    # and contains a mix of letters/digits, it's likely an ID
                    if most_common_pattern and count / len(patterns) >= 0.7:  # Fixed to use len(patterns) instead of len(sample)
                        if ('U' in most_common_pattern or 'L' in most_common_pattern) and 'D' in most_common_pattern:
                            return True
            
            # Check standard ID patterns
            for pattern in id_patterns:
                pattern_match_count = 0
                for val in sample:
                    if val and isinstance(val, str) and re.match(pattern, val):
                        pattern_match_count += 1
                
                # If more than 30% match this pattern, it's likely an ID pattern
                if pattern_match_count / len(sample) > 0.3:
                    id_pattern_count += 1
                    break
            
            # Additional check for numeric IDs (high cardinality of integers)
            is_all_digits = True
            for val in sample:
                if not (val and isinstance(val, str) and val.isdigit()):
                    is_all_digits = False
                    break
            
            if is_all_digits:
                unique_ratio = series.nunique() / len(series)
                if unique_ratio > 0.7:  # High uniqueness suggests IDs
                    return True
            
            # If id_pattern_count > 0, this is likely an ID column
            return id_pattern_count > 0
        except Exception as e:
            logger.warning(f"Error checking ID format: {str(e)}")
            return False
            
    
    
    def identify_sales_column(self):
        """Identify the sales column in the data with advanced regex patterns"""
        # If sales column is provided, use it
        if self.sales_col is not None:
            self.detected_columns['sales'] = self.sales_col
            return True
        
        # Score candidates based on name and content
        sales_candidates = {}
        
        for col in self.column_classes['numeric_candidates']:
            try:
                col_lower = str(col).lower()
                score = 0
                
                # Skip columns with too many nulls
                null_percentage = self.data[col].isna().mean()
                if null_percentage > 0.7:
                    continue
                
                # Skip columns already identified for date or product
                if col == self.detected_columns.get('date') or col == self.detected_columns.get('product'):
                    continue
                
                # Check column name against sales patterns
                for pattern in self.regex_patterns['sales']['name_patterns']:
                    if re.search(pattern, col_lower, re.IGNORECASE):
                        score += 10
                        break
                
                # Check direct matches with sales keywords
                if col_lower in self.sales_patterns:
                    score += 8
                elif any(pattern in col_lower for pattern in self.sales_patterns):
                    score += 5
                
                # Check for numeric properties that suggest sales values
                if self._is_numeric_column(col):
                    try:
                        # Sales columns typically have higher values and are positive
                        non_zero = self.data[col][self.data[col] != 0]
                        if len(non_zero) > 0:
                            avg_value = non_zero.mean()
                            if avg_value > 0:  # Positive values
                                # Scale score based on average value (typical sales are higher)
                                score += 3 + min(4, avg_value / 100)
                    except:
                        pass
                
                # Store score if high enough
                if score > 5:
                    sales_candidates[col] = score
                    
            except Exception as e:
                logger.warning(f"Error evaluating column '{col}' as sales: {str(e)}")
                continue
        
        # Find best sales candidate
        if sales_candidates:
            best_sales_col = max(sales_candidates.items(), key=lambda x: x[1])[0]
            self.detected_columns['sales'] = best_sales_col
            logger.info(f"Sales column identified: {best_sales_col} (score: {sales_candidates[best_sales_col]})")
            return True
        
        # Try to find a column with sales-related keywords as a fallback
        for col in self.data.columns:
            col_lower = str(col).lower()
            if any(keyword in col_lower for keyword in ['sales', 'revenue', 'amount', 'total']):
                if col != self.detected_columns.get('date') and col != self.detected_columns.get('product'):
                    self.detected_columns['sales'] = col
                    logger.info(f"Sales column identified as fallback: {col}")
                    return True
        
        logger.warning("Could not identify sales column. Forecasting may be limited.")
        return False

                
    def _identify_column_by_type(self, column_type, aliases, score_threshold=5):
        """Generic helper method to identify columns by type with consistent scoring"""
        candidates = {}
        
        # Convert aliases to list if it's a string
        if isinstance(aliases, str):
            aliases = [aliases]
        
        # Get appropriate column class candidates based on column type
        if column_type in ['sales', 'price', 'quantity', 'cost', 'profit', 'discount']:
            relevant_candidates = self.column_classes['numeric_candidates']
        elif column_type in ['product', 'product_name', 'customer', 'location']:
            relevant_candidates = list(set(self.column_classes['categorical_candidates'] + 
                                     self.column_classes['text_candidates'] +
                                     self.column_classes['id_candidates']))
        elif column_type in ['year', 'month', 'day']:
            relevant_candidates = self.column_classes['date_component_candidates'].get(column_type, [])
        elif column_type == 'date':
            relevant_candidates = self.column_classes['date_candidates']
        else:
            # Default to all columns
            relevant_candidates = self.data.columns
        
        # First pass: Check for exact column name matches (highest priority)
        for col in self.data.columns:
            col_lower = str(col).lower()
            
            # Skip columns already identified for other purposes
            if col in [self.detected_columns.get(key) for key in self.detected_columns if self.detected_columns.get(key)]:
                continue
            
            # Check for exact matches with aliases
            if col_lower in [alias.lower() for alias in aliases]:
                candidates[col] = 15
                logger.debug(f"Found exact match for {column_type}: {col}")
        
        # If we found exact matches, use the best one
        if candidates:
            best_col = max(candidates.items(), key=lambda x: x[1])[0]
            return best_col
        
        # Second pass: Score candidates based on several criteria
        for col in relevant_candidates:
            try:
                col_lower = str(col).lower()
                score = 0
                
                # Skip already identified columns
                if col in [self.detected_columns.get(key) for key in self.detected_columns 
                          if self.detected_columns.get(key) and key != column_type]:
                    continue
                
                # Check for partial matches with aliases
                for alias in aliases:
                    alias_lower = alias.lower()
                    if alias_lower in col_lower or col_lower in alias_lower:
                        score = max(score, 10)
                        logger.debug(f"Found partial match for {column_type}: {col} (score: {score})")
                        break
                
                # Check for regex pattern matches if available
                if column_type in self.regex_patterns and 'name_patterns' in self.regex_patterns[column_type]:
                    for pattern in self.regex_patterns[column_type]['name_patterns']:
                        if re.search(pattern, col_lower, re.IGNORECASE):
                            score = max(score, 12)
                            logger.debug(f"Found regex match for {column_type}: {col} (score: {score})")
                            break
                
                # Additional column-specific checks
                if column_type == 'date' and col in self.column_classes['date_candidates']:
                    score = max(score, 10)
                elif column_type == 'product' and col in self.column_classes['id_candidates']:
                    score = max(score, 8)
                elif column_type == 'quantity' and self._is_numeric_column(col):
                    # Check if values are mostly integers
                    try:
                        non_null = self.data[col].dropna()
                        if len(non_null) > 0 and (non_null % 1 == 0).mean() > 0.9:
                            score = max(score, 7)
                    except:
                        pass
                elif column_type == 'sales' and self._is_numeric_column(col):
                    # Sales columns typically have higher values
                    try:
                        mean_val = self.data[col].mean()
                        if mean_val > 0:
                            score = max(score, 5 + min(3, mean_val / 1000))
                    except:
                        pass
                
                # Store score if high enough
                if score >= score_threshold:
                    candidates[col] = score
            
            except Exception as e:
                logger.warning(f"Error evaluating {col} as {column_type}: {str(e)}")
                continue
        
        # Return the column with the highest score
        if candidates:
            best_col = max(candidates.items(), key=lambda x: x[1])[0]
            return best_col
        
        return None
    
    def _create_composite_date(self):
        """Create a composite date column from year and month columns if available"""
        # Check if we have both year and month columns identified
        year_candidates = self.column_classes['date_component_candidates'].get('year', [])
        month_candidates = self.column_classes['date_component_candidates'].get('month', [])
        
        if not year_candidates or not month_candidates:
            logger.debug("Cannot create composite date: missing year or month candidates")
            return None
        
        # Get the best year and month columns
        year_col = self._select_best_component('year', year_candidates)
        month_col = self._select_best_component('month', month_candidates)
        
        if not year_col or not month_col:
            logger.debug("Cannot create composite date: failed to select best year or month column")
            return None
        
        logger.info(f"Creating composite date from {year_col} (year) and {month_col} (month)")
        
        try:
            # Create a new date column
            comp_date_col = 'composite_date'
            
            # Try to get day column if available
            day_candidates = self.column_classes['date_component_candidates'].get('day', [])
            day_col = self._select_best_component('day', day_candidates) if day_candidates else None
            
            if day_col:
                # Format: YYYY-MM-DD
                logger.info(f"Including day column {day_col} in composite date")
                try:
                    # First try string concatenation method
                    self.data[comp_date_col] = pd.to_datetime(
                        self.data[year_col].astype(str) + '-' + 
                        self.data[month_col].astype(str).str.zfill(2) + '-' + 
                        self.data[day_col].astype(str).str.zfill(2),
                        errors='coerce'
                    )
                except Exception as e:
                    logger.warning(f"First method failed: {str(e)}")
                    # Try alternative approach if the above fails
                    try:
                        self.data[comp_date_col] = pd.to_datetime({
                            'year': self.data[year_col],
                            'month': self.data[month_col],
                            'day': self.data[day_col]
                        }, errors='coerce')
                    except Exception as e:
                        logger.warning(f"Second method failed: {str(e)}")
                        return None
            else:
                # Format: YYYY-MM-01 (default to first day of month)
                try:
                    # First try string concatenation approach
                    self.data[comp_date_col] = pd.to_datetime(
                        self.data[year_col].astype(str) + '-' + 
                        self.data[month_col].astype(str).str.zfill(2) + '-01',
                        errors='coerce'
                    )
                except Exception as e:
                    logger.warning(f"First method failed: {str(e)}")
                    # Try alternative dictionary approach
                    try:
                        self.data[comp_date_col] = pd.to_datetime({
                            'year': self.data[year_col],
                            'month': self.data[month_col],
                            'day': 1
                        }, errors='coerce')
                    except Exception as e:
                        logger.warning(f"Second method failed: {str(e)}")
                        return None
            
            # Verify the composite date is valid
            valid_dates = self.data[comp_date_col].notna().mean()
            if valid_dates > 0.7:  # At least 70% valid dates
                # Store the detected components
                self.detected_columns['year'] = year_col
                self.detected_columns['month'] = month_col
                if day_col:
                    self.detected_columns['day'] = day_col
                    
                logger.info(f"Successfully created composite date column with {valid_dates:.1%} valid dates")
                return comp_date_col
            else:
                logger.warning(f"Created composite date column has too many invalid dates ({valid_dates:.1%} valid)")
                return None
        except Exception as e:
            logger.warning(f"Error creating composite date: {str(e)}")
            logger.debug(traceback.format_exc())
            return None
            
    def force_composite_date_creation(self):
        """Explicitly search for year and month columns and force composite date creation"""
        # Step 1: Look for columns that might contain years
        year_candidates = []
        month_candidates = []
        
        for col in self.data.columns:
            col_lower = str(col).lower()
            
            # Look for year columns
            if 'year' in col_lower or 'yr' in col_lower.split() or 'fy' in col_lower.split():
                # Verify the content
                try:
                    values = self.data[col].dropna()
                    if len(values) > 0:
                        min_val = values.min()
                        max_val = values.max()
                        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                            if 1900 <= min_val <= max_val <= datetime.now().year + 5:
                                year_candidates.append(col)
                                logger.info(f"Found year column: {col}")
                except:
                    pass
            
            # Look for month columns
            if 'month' in col_lower or 'mo' in col_lower.split() or 'mon' in col_lower.split():
                # Verify the content
                try:
                    values = self.data[col].dropna()
                    if len(values) > 0:
                        min_val = values.min()
                        max_val = values.max()
                        if isinstance(min_val, (int, float)) and isinstance(max_val, (int, float)):
                            if 1 <= min_val <= max_val <= 12:
                                month_candidates.append(col)
                                logger.info(f"Found month column: {col}")
                except:
                    pass
        
        # Step 2: If we have both year and month columns, force create a composite date
        if year_candidates and month_candidates:
            year_col = year_candidates[0]  # Use the first year column
            month_col = month_candidates[0]  # Use the first month column
            
            logger.info(f"Creating composite date from {year_col} and {month_col}")
            
            try:
                # Create a new date column
                comp_date_col = 'composite_date'
                
                # First attempt: string concatenation
                try:
                    self.data[comp_date_col] = pd.to_datetime(
                        self.data[year_col].astype(str) + '-' + 
                        self.data[month_col].astype(str).str.zfill(2) + '-01',
                        errors='coerce'
                    )
                except Exception as e:
                    logger.warning(f"First attempt failed: {str(e)}")
                    
                    # Second attempt: dictionary approach
                    self.data[comp_date_col] = pd.to_datetime({
                        'year': self.data[year_col],
                        'month': self.data[month_col],
                        'day': 1
                    }, errors='coerce')
                
                # Check if date creation was successful
                valid_pct = self.data[comp_date_col].notna().mean()
                if valid_pct > 0.5:  # Over 50% valid dates
                    self.detected_columns['date'] = comp_date_col
                    self.detected_columns['year'] = year_col
                    self.detected_columns['month'] = month_col
                    logger.info(f"Composite date created successfully with {valid_pct:.1%} valid dates")
                    return comp_date_col
                else:
                    logger.warning(f"Composite date has too many invalid values: {valid_pct:.1%} valid")
                    return None
            except Exception as e:
                logger.error(f"Failed to create composite date: {str(e)}")
                return None
        else:
            if not year_candidates:
                logger.warning("No year columns found for composite date")
            if not month_candidates:
                logger.warning("No month columns found for composite date")
            return None
                    

    
    def _analyze_column_role(self, column_name):
        """
        Analyze a column to determine its potential roles
        
        Args:
            column_name (str): The name of the column to analyze
            
        Returns:
            dict: Dictionary of potential roles with scores
        """
        potential_roles = {}
        col_lower = str(column_name).lower()
        
        try:
            # Skip columns with too many nulls
            null_percentage = self.data[column_name].isna().mean()
            if null_percentage > 0.7:
                return potential_roles
            
            # Check if the column might be a date
            if pd.api.types.is_datetime64_dtype(self.data[column_name]):
                potential_roles['date'] = 10
            elif column_name in self.column_classes['date_candidates']:
                potential_roles['date'] = 8
            elif 'date' in col_lower or 'time' in col_lower:
                potential_roles['date'] = 7
            
            # Check if the column might be a year
            if column_name in self.column_classes['date_component_candidates'].get('year', []):
                potential_roles['year'] = 10
            elif any(pattern in col_lower for pattern in ['year', 'yr', 'yyyy']):
                # Verify content
                values = self.data[column_name].dropna()
                if len(values) > 0:
                    try:
                        min_val = values.min()
                        max_val = values.max()
                        if 1900 <= min_val <= max_val <= datetime.now().year + 5:
                            potential_roles['year'] = 9
                    except:
                        pass
            
            # Check if the column might be a month
            if column_name in self.column_classes['date_component_candidates'].get('month', []):
                potential_roles['month'] = 10
            elif any(pattern in col_lower for pattern in ['month', 'mon', 'mm']):
                # Verify content
                values = self.data[column_name].dropna()
                if len(values) > 0:
                    try:
                        min_val = values.min()
                        max_val = values.max()
                        if 1 <= min_val <= max_val <= 12:
                            potential_roles['month'] = 9
                    except:
                        pass
            
            # Check if the column might be a product
            if any(pattern in col_lower for pattern in ['product', 'item', 'sku']):
                potential_roles['product'] = 8
            
            # Check if the column might be sales/revenue
            if any(pattern in col_lower for pattern in ['sales', 'revenue', 'amount']):
                if self._is_numeric_column(column_name):
                    potential_roles['sales'] = 8
            
            # Check if the column might be quantity
            if any(pattern in col_lower for pattern in ['quantity', 'qty', 'units']):
                if self._is_numeric_column(column_name):
                    potential_roles['quantity'] = 8
            
            # Check if the column might be price
            if any(pattern in col_lower for pattern in ['price', 'unit_price', 'cost']):
                if self._is_numeric_column(column_name):
                    potential_roles['price'] = 8
            
            # Add more role checks as needed...
            
            # If no roles identified yet, try to infer from content
            if not potential_roles:
                # Example: numeric columns with low cardinality might be categorical
                if self._is_numeric_column(column_name):
                    unique_ratio = self.data[column_name].nunique() / len(self.data)
                    if unique_ratio < 0.01:
                        potential_roles['category'] = 5
                
                # Example: high cardinality string columns might be products or descriptions
                elif pd.api.types.is_string_dtype(self.data[column_name]):
                    unique_ratio = self.data[column_name].nunique() / len(self.data)
                    if 0.01 < unique_ratio < 0.5:
                        potential_roles['product'] = 4
                        
            return potential_roles
            
        except Exception as e:
            logger.warning(f"Error analyzing column {column_name}: {str(e)}")
            return potential_roles
    
    def _analyze_column_role(self, column_name):
        """
        Analyze a column to determine its potential roles
        
        Args:
            column_name (str): The name of the column to analyze
            
        Returns:
            dict: Dictionary of potential roles with scores
        """
        potential_roles = {}
        col_lower = str(column_name).lower()
        
        try:
            # Skip columns with too many nulls
            null_percentage = self.data[column_name].isna().mean()
            if null_percentage > 0.7:
                return potential_roles
            
            # Check if the column might be a date
            if pd.api.types.is_datetime64_dtype(self.data[column_name]):
                potential_roles['date'] = 10
            elif column_name in self.column_classes['date_candidates']:
                potential_roles['date'] = 8
            elif 'date' in col_lower or 'time' in col_lower:
                potential_roles['date'] = 7
            
            # Check if the column might be a year
            if column_name in self.column_classes['date_component_candidates'].get('year', []):
                potential_roles['year'] = 10
            elif any(pattern in col_lower for pattern in ['year', 'yr', 'yyyy']):
                # Verify content
                values = self.data[column_name].dropna()
                if len(values) > 0:
                    try:
                        min_val = values.min()
                        max_val = values.max()
                        if 1900 <= min_val <= max_val <= datetime.now().year + 5:
                            potential_roles['year'] = 9
                    except:
                        pass
            
            # Check if the column might be a month
            if column_name in self.column_classes['date_component_candidates'].get('month', []):
                potential_roles['month'] = 10
            elif any(pattern in col_lower for pattern in ['month', 'mon', 'mm']):
                # Verify content
                values = self.data[column_name].dropna()
                if len(values) > 0:
                    try:
                        min_val = values.min()
                        max_val = values.max()
                        if 1 <= min_val <= max_val <= 12:
                            potential_roles['month'] = 9
                    except:
                        pass
            
            # Check if the column might be a product
            if any(pattern in col_lower for pattern in ['product', 'item', 'sku']):
                potential_roles['product'] = 8
            
            # Check if the column might be sales/revenue
            if any(pattern in col_lower for pattern in ['sales', 'revenue', 'amount']):
                if self._is_numeric_column(column_name):
                    potential_roles['sales'] = 8
            
            # Check if the column might be quantity
            if any(pattern in col_lower for pattern in ['quantity', 'qty', 'units']):
                if self._is_numeric_column(column_name):
                    potential_roles['quantity'] = 8
            
            # Check if the column might be price
            if any(pattern in col_lower for pattern in ['price', 'unit_price', 'cost']):
                if self._is_numeric_column(column_name):
                    potential_roles['price'] = 8
            
            # If no roles identified yet, try to infer from content
            if not potential_roles:
                # Example: numeric columns with low cardinality might be categorical
                if self._is_numeric_column(column_name):
                    unique_ratio = self.data[column_name].nunique() / len(self.data)
                    if unique_ratio < 0.01:
                        potential_roles['category'] = 5
                
                # Example: high cardinality string columns might be products or descriptions
                elif pd.api.types.is_string_dtype(self.data[column_name]):
                    unique_ratio = self.data[column_name].nunique() / len(self.data)
                    if 0.01 < unique_ratio < 0.5:
                        potential_roles['product'] = 4
                        
            return potential_roles
            
        except Exception as e:
            logger.warning(f"Error analyzing column {column_name}: {str(e)}")
            return potential_roles
    
    def _suggest_alternatives(self, column_type):
        """
        Suggest alternative columns when a specific column type is missing
        
        Args:
            column_type (str): The type of column to suggest alternatives for
            
        Returns:
            list: List of suggested column names
        """
        suggestions = []
        
        try:
            # Different suggestion strategies based on the column type
            if column_type == 'date':
                # Suggest date-like columns
                for col in self.data.columns:
                    col_lower = str(col).lower()
                    if ('date' in col_lower or 'time' in col_lower or 
                        col in self.column_classes.get('date_candidates', [])):
                        suggestions.append(col)
                
                # Suggest creating from year/month
                year_cols = self.column_classes.get('date_component_candidates', {}).get('year', [])
                month_cols = self.column_classes.get('date_component_candidates', {}).get('month', [])
                if year_cols and month_cols:
                    suggestions.append(f"create a date from {year_cols[0]} and {month_cols[0]}")
            
            
            
            # Limit to top 3 suggestions
            return suggestions[:3]
        except Exception as e:
            logger.warning(f"Error suggesting alternatives for {column_type}: {str(e)}")
            return []
                
    def identify_all_columns(self, expected_columns=None, manual_mapping=None):
        """
        Identify and map columns in the data based on their content,
        and return both a JSON dictionary mapping original column names to expected column types
        and a DataFrame with only the requested columns
        
        Args:
            expected_columns (list, optional): List of column types to identify and include
                in the returned mapping. Default is ['date', 'product', 'sales']
            manual_mapping (dict, optional): Dictionary to override automatic mapping 
                Format: {'expected_col_type': 'original_col_name', ...}
        
        Returns:
            tuple: (dict, pandas.DataFrame) where:
                - dict: A dictionary mapping original column names to their expected column types
                  Format: {'original_col_name': 'expected_col_type', ...}
                - pandas.DataFrame: A DataFrame containing only the expected columns
        """
        start_time = time.time()
        
        # Default expected columns if none provided
        if expected_columns is None:
            expected_columns = ['date', 'product', 'sales']
        
        # Define a more comprehensive set of column types to identify
        all_column_types = [
            'date', 'product', 'sales', 'quantity', 'price', 'returns', 
            'territory', 'channel', 'customer', 'royalty', 'property', 
            'sales_type', 'net_units', 'gross_sales', 'net_sales', 'year', 
            'month', 'day', 'cost', 'profit', 'discount'
        ]
        
        # Extend expected columns with additional common columns to check for
        extended_columns = list(expected_columns)
        for col_type in all_column_types:
            if col_type not in extended_columns:
                extended_columns.append(col_type)
        
        logger.info(f"Starting column identification for: {', '.join(expected_columns)}")
        logger.info(f"Also checking for: {', '.join([c for c in extended_columns if c not in expected_columns])}")
        
        # Initialize mapping dictionary for the result
        column_mapping = {}
        
        try:
            # Skip automatic detection if manual mapping is provided for all expected columns
            if manual_mapping is not None and all(col in manual_mapping for col in expected_columns):
                logger.info("Using provided manual mapping for all columns")
                self.detected_columns = {col_type: orig_col for col_type, orig_col in manual_mapping.items()}
                # Create the reverse mapping for return value
                for col_type, orig_col in manual_mapping.items():
                    column_mapping[orig_col] = col_type
                    
                    # Update instance variables for key columns
                    if col_type == 'date':
                        self.date_col = orig_col
                    elif col_type == 'product':
                        self.product_col = orig_col
                    elif col_type == 'sales':
                        self.sales_col = orig_col
            else:
                # Perform automatic detection
                # Initialize regex patterns
                regex_patterns = self._initialize_regex_patterns()
                
                # First, scan all columns to classify their data types and potential roles
                logger.info("Pre-classifying columns by data type and characteristics...")
                self._classify_columns()
                logger.info("Column classification complete.")
                
                # Track identification success
                identification_success = True
                
                # Initialize a dictionary to store potential mappings for each column
                column_candidates = {}
                
                # Scan all columns in the DataFrame to determine their likely roles
                for col in self.data.columns:
                    col_name = str(col)
                    col_lower = col_name.lower()
                    
                    # Analyze column content and name to determine possible roles
                    column_roles = {}
                    
                    # Check each column type against regex patterns and keywords
                    for col_type, patterns in regex_patterns.items():
                        score = 0
                        
                        # Check name patterns (regex)
                        if 'name_patterns' in patterns:
                            for pattern in patterns['name_patterns']:
                                if re.search(pattern, col_lower, re.IGNORECASE):
                                    score += 10
                                    break
                        
                        # Exact matches get highest score
                        if col_lower == col_type.lower() or col_lower == col_type.lower().replace('_', ''):
                            score += 15
                        
                        # Check for direct keyword matches with our comprehensive patterns
                        attr_name = f"{col_type}_patterns"
                        if hasattr(self, attr_name) and getattr(self, attr_name):
                            pattern_list = getattr(self, attr_name)
                            if col_lower in pattern_list:
                                score += 12
                            elif any(pattern in col_lower for pattern in pattern_list):
                                score += 8
                        
                        # Special case for certain column types
                        if col_type == 'quantity' and col_lower in ['units', 'unit', 'qty']:
                            score += 15  # Boost for exact matches like "Units"
                        elif col_type == 'returns' and col_lower in ['returns', 'return_units', 'return units']:
                            score += 15  # Boost for exact matches like "Return Units"
                        elif col_type == 'territory' and col_lower == 'territory':
                            score += 15  # Boost for exact match "Territory"
                        elif col_type == 'channel' and col_lower == 'channel':
                            score += 15  # Boost for exact match "Channel"
                        elif col_type == 'property' and col_lower == 'property':
                            score += 15  # Boost for exact match "Property"
                        elif col_type == 'sales_type' and col_lower in ['sales type', 'salestype', 'type']:
                            score += 15  # Boost for exact match "Sales Type"
                        elif col_type == 'net_units' and col_lower in ['net units', 'netunits']:
                            score += 15  # Boost for exact match "Net Units"
                        elif col_type == 'gross_sales' and col_lower in ['gross sales', 'grosssales']:
                            score += 15  # Boost for exact match "Gross Sales"
                        elif col_type == 'net_sales' and col_lower in ['net sales', 'netsales']:
                            score += 15  # Boost for exact match "Net Sales"
                        elif col_type == 'royalty' and col_lower in ['royalty %', 'royalty pay', 'royalty payable']:
                            score += 15  # Boost for royalty columns
                        elif col_type == 'year' and col_lower == 'year':
                            score += 15  # Boost for exact match "YEAR"
                        elif col_type == 'month' and col_lower == 'month':
                            score += 15  # Boost for exact match "MONTH"
                        
                        # Analyze column content if regex patterns for values exist
                        if score > 0 and 'value_patterns' in patterns:
                            try:
                                # Sample some values
                                sample = self.data[col].dropna().head(50).astype(str)
                                if not sample.empty:
                                    pattern_match_ratio = 0
                                    for pattern in patterns['value_patterns']:
                                        matches = sample.str.match(pattern).mean()
                                        pattern_match_ratio = max(pattern_match_ratio, matches)
                                    
                                    # Boost score based on content match
                                    score += pattern_match_ratio * 5
                            except:
                                pass
                        
                        # Include column in candidates if score is significant
                        if score > 3:
                            if col_type not in column_candidates:
                                column_candidates[col_type] = []
                            column_candidates[col_type].append((col_name, score))
                            column_roles[col_type] = score
                    
                    # Also use the existing analysis method to catch anything missed
                    potential_roles = self._analyze_column_role(col)
                    for role, score in potential_roles.items():
                        if role not in column_roles or score > column_roles[role]:
                            if role not in column_candidates:
                                column_candidates[role] = []
                            column_candidates[role].append((col_name, score))
                
                # Sort candidates by score for each role
                for role in column_candidates:
                    column_candidates[role].sort(key=lambda x: x[1], reverse=True)
                    logger.info(f"Candidates for {role}: {column_candidates[role]}")
                
                # Special case: Check if we need to create a date from year and month
                if 'date' in extended_columns and 'year' in column_candidates and 'month' in column_candidates:
                    if column_candidates['year'] and column_candidates['month']:
                        # Get best year and month columns
                        year_col, year_score = column_candidates['year'][0]
                        month_col, month_score = column_candidates['month'][0]
                        
                        # Only create date if scores are high enough
                        if year_score > 5 and month_score > 5:
                            logger.info(f"Creating date from {year_col} and {month_col}")
                            self.detected_columns['year'] = year_col
                            self.detected_columns['month'] = month_col
                            
                            # Create the date column in the DataFrame
                            try:
                                date_col_name = 'date'
                                year_str = self.data[year_col].astype(str)
                                month_str = self.data[month_col].astype(str).str.zfill(2)
                                
                                self.data[date_col_name] = pd.to_datetime(
                                    year_str + '-' + month_str + '-01',
                                    errors='coerce'
                                )
                                
                                # Add this as a date candidate with high score
                                if 'date' not in column_candidates:
                                    column_candidates['date'] = []
                                column_candidates['date'].insert(0, (date_col_name, 15))  # High score
                                
                                # Add the source columns to the mapping with special indicators
                                column_mapping[year_col] = 'date_year_component'
                                column_mapping[month_col] = 'date_month_component'
                                
                                logger.info(f"Date column created from {year_col} and {month_col}")
                            except Exception as e:
                                logger.error(f"Failed to create date column: {str(e)}")
                                logger.debug(traceback.format_exc())
                
                # Determine the best column for each extended role
                for role in extended_columns:
                    # Check if we have a manual mapping for this role
                    if manual_mapping and role in manual_mapping:
                        # Use the manually specified column
                        manual_col = manual_mapping[role]
                        if manual_col in self.data.columns:
                            self.detected_columns[role] = manual_col
                            column_mapping[manual_col] = role
                            logger.info(f"Using manual mapping: {manual_col} as {role}")
                            
                            # Update instance variables for key columns
                            if role == 'date':
                                self.date_col = manual_col
                            elif role == 'product':
                                self.product_col = manual_col
                            elif role == 'sales':
                                self.sales_col = manual_col
                            elif role == 'quantity':
                                self.quantity_col = manual_col
                        else:
                            logger.error(f"Manual mapping error: Column '{manual_col}' not found in data")
                            if role in expected_columns:  # Only mark as failure if it's a required column
                                identification_success = False
                    elif role in column_candidates and column_candidates[role]:  # Automatic detection
                        best_col, score = column_candidates[role][0]
                        if score > 3:  # Only use if score is reasonable
                            self.detected_columns[role] = best_col
                            logger.info(f"Selected {best_col} as {role} (score: {score})")
                            
                            # Add to our mapping dictionary
                            column_mapping[best_col] = role
                            
                            # Update instance variables for key columns
                            if role == 'date':
                                self.date_col = best_col
                            elif role == 'product':
                                self.product_col = best_col
                            elif role == 'sales':
                                self.sales_col = best_col
                            elif role == 'quantity':
                                self.quantity_col = best_col
                    elif role in expected_columns:  # Only log warning if it's a required column
                        logger.warning(f"No candidates found for required column type: {role}")
                        identification_success = False
                
                # Log the identified columns
                logger.info("\nColumn Detection Summary:")
                for col_type, col_name in self.detected_columns.items():
                    if col_name:
                        logger.info(f"  {col_type.capitalize()}: {col_name}")
            
            end_time = time.time()
            logger.info(f"Column identification completed in {end_time - start_time:.2f} seconds")
            
            # Create a new DataFrame with only the expected columns
            mapped_df = pd.DataFrame(index=self.data.index)
            missing_columns = []
            
            for col_type in expected_columns:
                col_name = self.detected_columns.get(col_type)
                if col_name and col_name in self.data.columns:
                    mapped_df[col_type] = self.data[col_name]
                else:
                    missing_columns.append(col_type)
            
            # Only report on columns that were actually requested but couldn't be mapped
            if missing_columns:
                logger.warning(f"Warning: Could not map requested columns: {', '.join(missing_columns)}")
                print(f"Warning: Could not map requested columns: {', '.join(missing_columns)}")
                print("Suggestions for missing columns:")
                for col_type in missing_columns:
                    suggestions = self._suggest_alternatives(col_type)
                    if suggestions:
                        print(f"  For '{col_type}': Consider {', '.join(suggestions)}")
                    else:
                        print(f"  For '{col_type}': No suitable columns found")
            
            # Return both the mapping dictionary and the mapped DataFrame
            return column_mapping, mapped_df
        
        except Exception as e:
            logger.error(f"Unhandled error in column identification: {str(e)}")
            logger.debug(traceback.format_exc())
            # Return an empty mapping and empty DataFrame
            return {}, pd.DataFrame(columns=expected_columns)     
            
 
    def correct_column_mapping(self, corrections):
        """
        Apply corrections to the column mapping and regenerate the mapped DataFrame
        
        Args:
            corrections (dict): Dictionary with corrections to the mapping
                Format: {'expected_col_type': 'original_col_name', ...}
        
        Returns:
            tuple: (dict, pandas.DataFrame) where:
                - dict: Updated mapping dictionary from original column names to expected column types
                - pandas.DataFrame: Updated DataFrame containing only the expected columns
        """
        logger.info(f"Applying manual corrections to column mapping: {corrections}")
        
        # Get the current expected columns
        expected_columns = list(self.detected_columns.keys())
        
        # Re-run the identification with manual mapping
        return self.identify_all_columns(expected_columns=expected_columns, manual_mapping=corrections)
    
        
    def identify_all_columns(self, expected_columns=None, manual_mapping=None):
        """
        Identify and map columns in the data based on their content,
        and return both a JSON dictionary mapping original column names to expected column types
        and a DataFrame with only the requested columns
        
        Args:
            expected_columns (list, optional): List of column types to identify and include
                in the returned mapping. Default is ['date', 'product', 'sales']
            manual_mapping (dict, optional): Dictionary to override automatic mapping 
                Format: {'expected_col_type': 'original_col_name', ...}
        
        Returns:
            tuple: (dict, pandas.DataFrame) where:
                - dict: A dictionary mapping original column names to their expected column types
                  Format: {'original_col_name': 'expected_col_type', ...}
                - pandas.DataFrame: A DataFrame containing only the expected columns
        """
        start_time = time.time()
        
        # Default expected columns if none provided
        if expected_columns is None:
            expected_columns = ['date', 'product', 'sales']
        
        # Define a more comprehensive set of column types to identify
        all_column_types = [
            'date', 'product', 'sales', 'quantity', 'price', 'returns', 
            'territory', 'channel', 'customer', 'royalty', 'property', 
            'sales_type', 'net_units', 'gross_sales', 'net_sales', 'year', 
            'month', 'day', 'cost', 'profit', 'discount', 'revenue', 'saled_units'
        ]
        
        # Extend expected columns with additional common columns to check for
        extended_columns = list(expected_columns)
        for col_type in all_column_types:
            if col_type not in extended_columns:
                extended_columns.append(col_type)
        
        logger.info(f"Starting column identification for: {', '.join(expected_columns)}")
        logger.info(f"Also checking for: {', '.join([c for c in extended_columns if c not in expected_columns])}")
        
        # Initialize mapping dictionary for the result
        column_mapping = {}
        
        try:
            # Skip automatic detection if manual mapping is provided for all expected columns
            if manual_mapping is not None and all(col in manual_mapping for col in expected_columns):
                logger.info("Using provided manual mapping for all columns")
                self.detected_columns = {col_type: orig_col for col_type, orig_col in manual_mapping.items()}
                # Create the reverse mapping for return value
                for col_type, orig_col in manual_mapping.items():
                    column_mapping[orig_col] = col_type
                    
                    # Update instance variables for key columns
                    if col_type == 'date':
                        self.date_col = orig_col
                    elif col_type == 'product':
                        self.product_col = orig_col
                    elif col_type == 'sales':
                        self.sales_col = orig_col
            else:
                # Perform automatic detection
                # Initialize regex patterns
                regex_patterns = self._initialize_regex_patterns()
                
                # Additional patterns for specific column types that were missing
                # additional_patterns = {
                
                
                # Merge additional patterns with existing patterns
                for col_type, patterns in additional_patterns.items():
                    regex_patterns[col_type] = patterns
                
                # Define mappings between similar column types (fallbacks)
                column_type_fallbacks = {
                    'revenue': ['sales', 'gross_sales', 'net_sales'],
                    'saled_units': ['quantity', 'units', 'net_units'],
                    'cost': ['price'],
                    'profit': ['royalty', 'margin']
                }
                
                # First, scan all columns to classify their data types and potential roles
                logger.info("Pre-classifying columns by data type and characteristics...")
                self._classify_columns()
                logger.info("Column classification complete.")
                
                # Track identification success
                identification_success = True
                
                # Initialize a dictionary to store potential mappings for each column
                column_candidates = {}
                
                # Scan all columns in the DataFrame to determine their likely roles
                for col in self.data.columns:
                    col_name = str(col)
                    col_lower = col_name.lower()
                    
                    # Analyze column content and name to determine possible roles
                    column_roles = {}
                    
                    # Check each column type against regex patterns and keywords
                    for col_type, patterns in regex_patterns.items():
                        score = 0
                        
                        # Check name patterns (regex)
                        if 'name_patterns' in patterns:
                            for pattern in patterns['name_patterns']:
                                if re.search(pattern, col_lower, re.IGNORECASE):
                                    score += 10
                                    break
                        
                        # Exact matches get highest score
                        if col_lower == col_type.lower() or col_lower == col_type.lower().replace('_', ''):
                            score += 15
                        
                        # Check for specific column name matches
                        if col_type == 'quantity' and col_lower in ['units', 'unit', 'qty']:
                            score += 15
                        elif col_type == 'saled_units' and col_lower in ['units', 'unit', 'qty', 'quantity']:
                            score += 12  # Boost for potential saled_units columns
                        elif col_type == 'revenue' and col_lower in ['gross sales', 'net sales', 'sales']:
                            score += 12  # Revenue often appears as sales
                        elif col_type == 'cost' and col_lower in ['price', 'unit price', 'cost']:
                            score += 12
                        elif col_type == 'profit' and col_lower in ['royalty', 'royalty pay', 'margin']:
                            score += 10
                        elif col_type == 'customer' and 'client' in col_lower:
                            score += 12
                        elif col_type == 'returns' and col_lower in ['returns', 'return_units', 'return units']:
                            score += 15
                        elif col_type == 'territory' and col_lower == 'territory':
                            score += 15
                        elif col_type == 'channel' and col_lower == 'channel':
                            score += 15
                        elif col_type == 'property' and col_lower == 'property':
                            score += 15
                        elif col_type == 'sales_type' and col_lower in ['sales type', 'salestype', 'type']:
                            score += 15
                        elif col_type == 'net_units' and col_lower in ['net units', 'netunits']:
                            score += 15
                        elif col_type == 'gross_sales' and col_lower in ['gross sales', 'grosssales']:
                            score += 15
                        elif col_type == 'net_sales' and col_lower in ['net sales', 'netsales']:
                            score += 15
                        elif col_type == 'royalty' and col_lower in ['royalty %', 'royalty pay', 'royalty payable']:
                            score += 15
                        elif col_type == 'year' and col_lower == 'year':
                            score += 15
                        elif col_type == 'month' and col_lower == 'month':
                            score += 15
                        
                        # Analyze column content if regex patterns for values exist
                        if score > 0 and 'value_patterns' in patterns:
                            try:
                                # Sample some values
                                sample = self.data[col].dropna().head(50).astype(str)
                                if not sample.empty:
                                    pattern_match_ratio = 0
                                    for pattern in patterns['value_patterns']:
                                        matches = sample.str.match(pattern).mean()
                                        pattern_match_ratio = max(pattern_match_ratio, matches)
                                    
                                    # Boost score based on content match
                                    score += pattern_match_ratio * 5
                            except:
                                pass
                        
                        # Include column in candidates if score is significant
                        if score > 3:
                            if col_type not in column_candidates:
                                column_candidates[col_type] = []
                            column_candidates[col_type].append((col_name, score))
                            column_roles[col_type] = score
                    
                    # Also use the existing analysis method to catch anything missed
                    potential_roles = self._analyze_column_role(col)
                    for role, score in potential_roles.items():
                        if role not in column_roles or score > column_roles[role]:
                            if role not in column_candidates:
                                column_candidates[role] = []
                            column_candidates[role].append((col_name, score))
                
                # Sort candidates by score for each role
                for role in column_candidates:
                    column_candidates[role].sort(key=lambda x: x[1], reverse=True)
                    logger.info(f"Candidates for {role}: {column_candidates[role]}")
                
                # Apply fallbacks for missing columns - look for alternate column types
                for target_type, fallback_types in column_type_fallbacks.items():
                    if target_type not in column_candidates or not column_candidates[target_type]:
                        for fallback_type in fallback_types:
                            if fallback_type in column_candidates and column_candidates[fallback_type]:
                                # Use the best candidate from the fallback type with slightly reduced score
                                best_col, best_score = column_candidates[fallback_type][0]
                                adjusted_score = best_score * 0.9  # Slightly reduce score
                                
                                if target_type not in column_candidates:
                                    column_candidates[target_type] = []
                                
                                column_candidates[target_type].append((best_col, adjusted_score))
                                logger.info(f"Using {fallback_type} column '{best_col}' as fallback for {target_type}")
                                break
                
                # Special case: Check if we need to create a date from year and month
                if 'date' in extended_columns and 'year' in column_candidates and 'month' in column_candidates:
                    if column_candidates['year'] and column_candidates['month']:
                        # Get best year and month columns
                        year_col, year_score = column_candidates['year'][0]
                        month_col, month_score = column_candidates['month'][0]
                        
                        # Only create date if scores are high enough
                        if year_score > 5 and month_score > 5:
                            logger.info(f"Creating date from {year_col} and {month_col}")
                            self.detected_columns['year'] = year_col
                            self.detected_columns['month'] = month_col
                            
                            # Create the date column in the DataFrame
                            try:
                                date_col_name = 'date'
                                year_str = self.data[year_col].astype(str)
                                month_str = self.data[month_col].astype(str).str.zfill(2)
                                
                                self.data[date_col_name] = pd.to_datetime(
                                    year_str + '-' + month_str + '-01',
                                    errors='coerce'
                                )
                                
                                # Add this as a date candidate with high score
                                if 'date' not in column_candidates:
                                    column_candidates['date'] = []
                                column_candidates['date'].insert(0, (date_col_name, 15))  # High score
                                
                                # Add the source columns to the mapping with special indicators
                                column_mapping[year_col] = 'date_year_component'
                                column_mapping[month_col] = 'date_month_component'
                                
                                logger.info(f"Date column created from {year_col} and {month_col}")
                            except Exception as e:
                                logger.error(f"Failed to create date column: {str(e)}")
                                logger.debug(traceback.format_exc())
                
                # Special case for saled_units: If Units column exists, consider it a strong candidate
                if 'saled_units' in extended_columns and 'Units' in self.data.columns:
                    if 'saled_units' not in column_candidates:
                        column_candidates['saled_units'] = []
                    column_candidates['saled_units'].insert(0, ('Units', 14))
                    logger.info(f"Added 'Units' as candidate for saled_units with high score")
                
                # Determine the best column for each extended role
                for role in extended_columns:
                    # Check if we have a manual mapping for this role
                    if manual_mapping and role in manual_mapping:
                        # Use the manually specified column
                        manual_col = manual_mapping[role]
                        if manual_col in self.data.columns:
                            self.detected_columns[role] = manual_col
                            column_mapping[manual_col] = role
                            logger.info(f"Using manual mapping: {manual_col} as {role}")
                            
                            # Update instance variables for key columns
                            if role == 'date':
                                self.date_col = manual_col
                            elif role == 'product':
                                self.product_col = manual_col
                            elif role == 'sales':
                                self.sales_col = manual_col
                            elif role == 'quantity' or role == 'saled_units':
                                self.quantity_col = manual_col
                        else:
                            logger.error(f"Manual mapping error: Column '{manual_col}' not found in data")
                            if role in expected_columns:  # Only mark as failure if it's a required column
                                identification_success = False
                    elif role in column_candidates and column_candidates[role]:  # Automatic detection
                        best_col, score = column_candidates[role][0]
                        if score > 3:  # Only use if score is reasonable
                            self.detected_columns[role] = best_col
                            logger.info(f"Selected {best_col} as {role} (score: {score})")
                            
                            # Add to our mapping dictionary
                            column_mapping[best_col] = role
                            
                            # Update instance variables for key columns
                            if role == 'date':
                                self.date_col = best_col
                            elif role == 'product':
                                self.product_col = best_col
                            elif role == 'sales':
                                self.sales_col = best_col
                            elif role == 'quantity' or role == 'saled_units':
                                self.quantity_col = best_col
                    elif role in expected_columns:  # Only log warning if it's a required column
                        logger.warning(f"No candidates found for required column type: {role}")
                        identification_success = False
                
                # Special case for common synonyms for required columns
                for req_role in expected_columns:
                    if req_role not in self.detected_columns:
                        # Try to use equivalent columns
                        equivalents = {
                            'revenue': ['sales', 'gross_sales', 'net_sales'],
                            'sales': ['revenue', 'gross_sales', 'net_sales'],
                            'saled_units': ['quantity', 'units', 'net_units'],
                            'quantity': ['saled_units', 'units', 'net_units'],
                            'cost': ['price'],
                            'price': ['cost'],
                            'profit': ['royalty', 'margin'],
                            'customer': ['client']
                        }
                        
                        if req_role in equivalents:
                            for alt_role in equivalents[req_role]:
                                if alt_role in self.detected_columns:
                                    equiv_col = self.detected_columns[alt_role]
                                    self.detected_columns[req_role] = equiv_col
                                    logger.info(f"Using equivalent column {equiv_col} ({alt_role}) for {req_role}")
                                    
                                    # Only make a new mapping if not already mapped
                                    if equiv_col not in column_mapping:
                                        column_mapping[equiv_col] = req_role
                                    
                                    # Update instance variables
                                    if req_role == 'date':
                                        self.date_col = equiv_col
                                    elif req_role == 'product':
                                        self.product_col = equiv_col
                                    elif req_role == 'sales' or req_role == 'revenue':
                                        self.sales_col = equiv_col
                                    elif req_role == 'quantity' or req_role == 'saled_units':
                                        self.quantity_col = equiv_col
                                    break
                
                # Log the identified columns
                logger.info("\nColumn Detection Summary:")
                for col_type, col_name in self.detected_columns.items():
                    if col_name:
                        logger.info(f"  {col_type.capitalize()}: {col_name}")
            
            end_time = time.time()
            logger.info(f"Column identification completed in {end_time - start_time:.2f} seconds")
            
            # Create a new DataFrame with only the expected columns
            mapped_df = pd.DataFrame(index=self.data.index)
            missing_columns = []
            
            for col_type in expected_columns:
                col_name = self.detected_columns.get(col_type)
                if col_name and col_name in self.data.columns:
                    mapped_df[col_type] = self.data[col_name]
                else:
                    missing_columns.append(col_type)
            
            # Only report on columns that were actually requested but couldn't be mapped
            if missing_columns:
                logger.warning(f"Warning: Could not map requested columns: {', '.join(missing_columns)}")
                print(f"Warning: Could not map requested columns: {', '.join(missing_columns)}")
                print("Suggestions for missing columns:")
                for col_type in missing_columns:
                    suggestions = self._suggest_alternatives(col_type)
                    if suggestions:
                        print(f"  For '{col_type}': Consider {', '.join(suggestions)}")
                    else:
                        # Enhanced suggestions for common missing columns
                        if col_type == 'revenue':
                            print(f"  For 'revenue': Try using 'Sales', 'Gross Sales', or 'Net Sales' columns if available")
                        elif col_type == 'cost':
                            print(f"  For 'cost': Try using 'Price' column or create a derived cost column")
                        elif col_type == 'profit':
                            print(f"  For 'profit': Try creating a derived profit column from sales minus cost")
                        elif col_type == 'customer':
                            print(f"  For 'customer': Look for client-related columns or potentially 'Territory'")
                        elif col_type == 'saled_units':
                            print(f"  For 'saled_units': Try using 'Units', 'Quantity', or 'Net Units' columns")
                        else:
                            print(f"  For '{col_type}': No suitable columns found")
            
            # Return both the mapping dictionary and the mapped DataFrame
            return column_mapping, mapped_df
        
        except Exception as e:
            logger.error(f"Unhandled error in column identification: {str(e)}")
            logger.debug(traceback.format_exc())
            # Return an empty mapping and empty DataFrame
            return {}, pd.DataFrame(columns=expected_columns)