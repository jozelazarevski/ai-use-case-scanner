"""
Enhanced Column Identifier Module

This module contains improved functionality for automatically identifying important columns
in sales data, with enhanced date detection, product clustering, and sales identification.
"""

import re
import logging
import pandas as pd
import numpy as np
from datetime import datetime
from config import DEFAULT_CONFIG

logger = logging.getLogger('sales_forecaster.column_identifier')

class ColumnIdentifier:
    """Enhanced class for identifying important columns in sales data"""
    
    def __init__(self, data, date_col=None, product_col=None, sales_col=None):
        """
        Initialize the column identifier
        
        Args:
            data (DataFrame): The pandas DataFrame containing the sales data
            date_col (str, optional): Name of the date column if known
            product_col (str, optional): Name of the product column if known
            sales_col (str, optional): Name of the sales column if known
        """
        self.data = data
        self.date_col = date_col
        self.product_col = product_col
        self.sales_col = sales_col
        
        # Column patterns from config
        self.date_patterns = DEFAULT_CONFIG['date_patterns']
        self.product_patterns = DEFAULT_CONFIG['product_patterns']
        self.sales_patterns = DEFAULT_CONFIG['sales_patterns']
        self.quantity_patterns = DEFAULT_CONFIG['quantity_patterns']
        self.price_patterns = DEFAULT_CONFIG['price_patterns']
        
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
            'location': None
        }
    
    def identify_all_columns(self):
        """Identify all important columns in the data"""
        # First, scan all columns to classify their data types and potential roles
        self._classify_columns()
        
        # Identify columns in a specific order
        success = self.identify_date_column()
        if not success:
            return False
            
        success = self.identify_product_column()
        if not success:
            return False
            
        success = self.identify_sales_column()
        if not success:
            return False
        
        # Additional identifications (optional)
        self.identify_customer_column()
        self.identify_location_column()
        
        # Update instance variables with detected columns
        self.date_col = self.detected_columns['date']
        self.product_col = self.detected_columns['product']
        self.sales_col = self.detected_columns['sales']
        
        # Log the identified columns
        logger.info("\nColumn Detection Summary:")
        for col_type, col_name in self.detected_columns.items():
            if col_name:
                logger.info(f"  {col_type.capitalize()}: {col_name}")
        
        return True
    
    def _classify_columns(self):
        """
        Classify all columns by data type and potential role
        This pre-classification helps with subsequent specific column identification
        """
        logger.info("Pre-classifying columns by data type and characteristics...")
        
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
        
        for col in self.data.columns:
            col_lower = str(col).lower()
            dtype = self.data[col].dtype
            unique_count = self.data[col].nunique()
            total_count = len(self.data)
            
            # Skip columns with too many nulls
            null_percentage = self.data[col].isna().mean()
            if null_percentage > 0.5:
                continue
            
            # Check for date columns
            if pd.api.types.is_datetime64_dtype(dtype):
                self.column_classes['date_candidates'].append(col)
                continue
            
            # Try to convert to datetime
            try:
                if pd.to_datetime(self.data[col], errors='coerce').notna().mean() > 0.7:
                    self.column_classes['date_candidates'].append(col)
                    continue
            except:
                pass
            
            # Check for date components (year, month, day)
            if pd.api.types.is_integer_dtype(dtype) or pd.api.types.is_float_dtype(dtype):
                # Year detection
                if any(pattern in col_lower for pattern in self.year_patterns):
                    year_values = self.data[col].dropna()
                    if year_values.min() >= 1900 and year_values.max() <= datetime.now().year + 1:
                        self.column_classes['date_component_candidates']['year'].append(col)
                        continue
                
                # Month detection
                if any(pattern in col_lower for pattern in self.month_patterns):
                    month_values = self.data[col].dropna()
                    if month_values.min() >= 1 and month_values.max() <= 12:
                        self.column_classes['date_component_candidates']['month'].append(col)
                        continue
                
                # Day detection
                if any(pattern in col_lower for pattern in self.day_patterns):
                    day_values = self.data[col].dropna()
                    if day_values.min() >= 1 and day_values.max() <= 31:
                        self.column_classes['date_component_candidates']['day'].append(col)
                        continue
            
            # Numeric columns
            if pd.api.types.is_numeric_dtype(dtype):
                self.column_classes['numeric_candidates'].append(col)
                
                # Further classify numeric columns
                if unique_count <= 2:
                    self.column_classes['boolean_candidates'].append(col)
                elif unique_count / total_count < 0.01 and unique_count > 1:
                    # Small number of unique values relative to dataset size
                    self.column_classes['categorical_candidates'].append(col)
                
                # Check for ID patterns in numeric columns
                if unique_count / total_count > 0.5 and self._has_id_pattern(col_lower):
                    self.column_classes['id_candidates'].append(col)
            
            # Categorical/string columns
            elif pd.api.types.is_string_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
                if unique_count / total_count < 0.3:
                    self.column_classes['categorical_candidates'].append(col)
                else:
                    # High cardinality text columns
                    if self._has_id_pattern(col_lower) or self._check_if_id_format(self.data[col]):
                        self.column_classes['id_candidates'].append(col)
                    else:
                        self.column_classes['text_candidates'].append(col)
        
        logger.info("Column classification complete.")
        logger.debug(f"Date candidates: {len(self.column_classes['date_candidates'])}")
        logger.debug(f"Date component candidates: Year({len(self.column_classes['date_component_candidates']['year'])}) "
                    f"Month({len(self.column_classes['date_component_candidates']['month'])}) "
                    f"Day({len(self.column_classes['date_component_candidates']['day'])})")
        logger.debug(f"Numeric candidates: {len(self.column_classes['numeric_candidates'])}")
        logger.debug(f"Categorical candidates: {len(self.column_classes['categorical_candidates'])}")
    
    def _has_id_pattern(self, column_name):
        """Check if a column name suggests it's an ID column"""
        id_keywords = ['id', 'code', 'key', 'no', 'number', 'num', 'identifier']
        for keyword in id_keywords:
            if keyword in column_name or column_name.endswith('_' + keyword) or column_name.startswith(keyword + '_'):
                return True
        return False
    
    def _check_if_id_format(self, series):
        """Check if a series contains values that match typical ID formats"""
        # Sample some values to check
        sample = series.dropna().sample(min(20, len(series))).astype(str)
        
        # Count how many match ID patterns
        id_pattern_count = 0
        for val in sample:
            # Check for patterns like: ABC123, 123-456, A1B2C3, etc.
            if re.match(r'^[A-Z0-9]{3,}$', val) or \
               re.match(r'^\w+[-_]\w+$', val) or \
               re.match(r'^[A-Z][0-9]+$', val):
                id_pattern_count += 1
        
        # If more than 30% match, this is likely an ID column
        return id_pattern_count / len(sample) > 0.3
        
    def identify_date_column(self):
        """Identify the date column in the data, handling composite date fields"""
        if self.date_col is not None:
            self.detected_columns['date'] = self.date_col
            return True
            
        logger.info("Attempting to identify date column...")
        
        # Method 1: Use pre-classified date candidates
        if self.column_classes['date_candidates']:
            # Prioritize by name and quality
            best_candidate = None
            best_score = 0
            
            for col in self.column_classes['date_candidates']:
                col_lower = str(col).lower()
                score = 0
                
                # Check exact matches
                if col_lower in self.date_patterns:
                    score += 10
                else:
                    # Check partial matches
                    for keyword in self.date_patterns:
                        if keyword in col_lower:
                            score += 5
                            break
                
                # Add points for percentage of valid dates
                valid_percentage = pd.to_datetime(self.data[col], errors='coerce').notna().mean()
                score += valid_percentage * 5
                
                # Prefer columns with more obvious date names
                priority_keywords = ['order_date', 'orderdate', 'invoice_date', 'transaction_date', 'date']
                for keyword in priority_keywords:
                    if keyword in col_lower:
                        score += 3
                        break
                
                if score > best_score:
                    best_score = score
                    best_candidate = col
            
            if best_candidate:
                self.detected_columns['date'] = best_candidate
                logger.info(f"Date column identified: {best_candidate}")
                return True
        
        # Method 2: Check for separate year, month, day columns that can be combined
        year_col = None
        month_col = None
        day_col = None
        
        # Look for year, month, day columns
        if self.column_classes['date_component_candidates']['year']:
            year_col = self.column_classes['date_component_candidates']['year'][0]
            self.detected_columns['year'] = year_col
            
        if self.column_classes['date_component_candidates']['month']:
            month_col = self.column_classes['date_component_candidates']['month'][0]
            self.detected_columns['month'] = month_col
            
        if self.column_classes['date_component_candidates']['day']:
            day_col = self.column_classes['date_component_candidates']['day'][0]
            self.detected_columns['day'] = day_col
        
        # If we have at least year and month, we can create a date column
        if year_col and month_col:
            logger.info(f"Found date components: Year({year_col}), Month({month_col}), Day({day_col if day_col else 'Not found'})")
            
            try:
                # Create a new date column
                if day_col:
                    # Format: YYYY-MM-DD
                    self.data['composite_date'] = pd.to_datetime(
                        self.data[year_col].astype(str) + '-' + 
                        self.data[month_col].astype(str).str.zfill(2) + '-' + 
                        self.data[day_col].astype(str).str.zfill(2)
                    )
                else:
                    # Format: YYYY-MM-01 (default to first day of month)
                    self.data['composite_date'] = pd.to_datetime(
                        self.data[year_col].astype(str) + '-' + 
                        self.data[month_col].astype(str).str.zfill(2) + '-01'
                    )
                
                self.detected_columns['date'] = 'composite_date'
                logger.info(f"Created composite date column from {year_col}, {month_col}{', ' + day_col if day_col else ''}")
                return True
            except Exception as e:
                logger.warning(f"Error creating composite date: {e}")
                
        # Method 3: Check for columns that might be date but not detected earlier
        for col in self.data.columns:
            # Skip columns we've already checked
            if col in self.column_classes['date_candidates']:
                continue
            
            # Try different date formats for string columns
            if pd.api.types.is_string_dtype(self.data[col]):
                try:
                    # Try multiple common date formats
                    for date_format in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d', '%m-%d-%Y', '%d-%m-%Y']:
                        converted = pd.to_datetime(self.data[col], format=date_format, errors='coerce')
                        if converted.notna().mean() > 0.7:
                            self.data[f'parsed_date_{col}'] = converted
                            self.detected_columns['date'] = f'parsed_date_{col}'
                            logger.info(f"Parsed date column from {col} using format {date_format}")
                            return True
                except:
                    pass
        
        # We couldn't find or construct a proper date column
        logger.error("Could not identify date column. Please specify it manually.")
        return False
            
    def identify_product_column(self):
        """Enhanced method to identify product column with better heuristics"""
        if self.product_col is not None:
            self.detected_columns['product'] = self.product_col
            return True
            
        logger.info("Attempting to identify product column...")
        
        # Score candidates for different product-related roles
        product_id_candidates = {}
        product_name_candidates = {}
        product_category_candidates = {}
        
        # Check categorical and text candidates
        relevant_candidates = self.column_classes['categorical_candidates'] + self.column_classes['text_candidates'] + self.column_classes['id_candidates']
        
        # First pass - direct name matching
        for col in relevant_candidates:
            col_lower = str(col).lower()
            
            # Skip the identified date column
            if col == self.detected_columns['date']:
                continue
            
            # Check for product ID column
            for keyword in self.product_id_patterns:
                if keyword in col_lower:
                    product_id_candidates[col] = 8
                    break
            
            # Check for product name column
            for keyword in self.product_name_patterns:
                if keyword in col_lower:
                    product_name_candidates[col] = 8
                    break
            
            # Check for product category column
            for keyword in self.product_category_patterns:
                if keyword in col_lower:
                    product_category_candidates[col] = 8
                    break
            
            # Check generic product column matches
            if col not in product_id_candidates and col not in product_name_candidates and col not in product_category_candidates:
                for keyword in self.product_patterns:
                    if keyword in col_lower:
                        # Determine which type it might be
                        if self._has_id_pattern(col_lower) or col in self.column_classes['id_candidates']:
                            product_id_candidates[col] = 7
                        elif len(self.data[col].astype(str).str.len().mean()) > 15:
                            product_name_candidates[col] = 7
                        else:
                            product_category_candidates[col] = 7
                        break
        
        # Second pass - analyze column characteristics
        for col in relevant_candidates:
            if col == self.detected_columns['date']:
                continue
                
            # Skip columns already with high confidence
            if (col in product_id_candidates and product_id_candidates[col] >= 7) or \
               (col in product_name_candidates and product_name_candidates[col] >= 7) or \
               (col in product_category_candidates and product_category_candidates[col] >= 7):
                continue
            
            # Analyze unique count and value patterns
            unique_count = self.data[col].nunique()
            total_count = len(self.data)
            unique_ratio = unique_count / total_count
            
            # Sample values
            sample_values = self.data[col].dropna().sample(min(10, unique_count)).astype(str)
            avg_length = sample_values.str.len().mean()
            
            # Product IDs typically have moderate cardinality and standardized formats
            if 0.01 <= unique_ratio <= 0.7:
                if self._check_if_id_format(self.data[col]) or avg_length <= 10:
                    score = 5 + (1 - abs(0.1 - unique_ratio)) * 3  # Ideal ratio around 0.1
                    product_id_candidates[col] = score
            
            # Product names typically have higher cardinality and longer text
            if 0.1 <= unique_ratio <= 0.9 and avg_length > 10:
                score = 4 + min(avg_length / 20, 3)  # Longer names get higher scores
                product_name_candidates[col] = score
            
            # Categories typically have low cardinality
            if 0.001 <= unique_ratio <= 0.1:
                score = 4 + (1 - unique_ratio * 10) * 3  # Fewer categories get higher scores
                product_category_candidates[col] = score
        
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
            logger.error("Could not identify any product column. Please specify it manually.")
            return False
        
        logger.info(f"Primary product column selected: {self.detected_columns['product']}")
        return True
            
    def identify_sales_column(self):
        """Enhanced method to identify sales and related financial columns with better heuristics"""
        if self.sales_col is not None:
            self.detected_columns['sales'] = self.sales_col
            return True
            
        logger.info("Attempting to identify sales and financial columns...")
        
        # Score candidates for different roles
        sales_candidates = {}
        quantity_candidates = {}
        price_candidates = {}
        revenue_candidates = {}
        cost_candidates = {}
        profit_candidates = {}
        
        # Revenue/sales patterns
        revenue_patterns = ['revenue', 'sales', 'amount', 'total', 'line_total', 'extended_price']
        
        # Cost patterns
        cost_patterns = ['cost', 'expense', 'cogs', 'wholesale', 'purchase_price']
        
        # Profit patterns
        profit_patterns = ['profit', 'margin', 'contribution', 'earnings', 'net_amount']
        
        # First pass - direct name matching on numeric columns
        for col in self.column_classes['numeric_candidates']:
            col_lower = str(col).lower()
            
            # Skip the identified date and product columns
            if col == self.detected_columns['date'] or col == self.detected_columns['product']:
                continue
            
            # Skip columns with high null percentage
            if self.data[col].isna().mean() > 0.5:
                continue
            
            # Check for sales/revenue column
            for keyword in revenue_patterns:
                if keyword in col_lower:
                    score = 8 + (2 if col_lower == keyword else 0)  # Exact match bonus
                    sales_candidates[col] = score
                    revenue_candidates[col] = score
                    break
            
            # Check for quantity column
            for keyword in self.quantity_patterns:
                if keyword in col_lower:
                    # Quantity columns typically contain integers
                    is_mostly_int = (self.data[col] % 1 == 0).mean() > 0.9
                    score = 7 + (2 if is_mostly_int else 0)
                    quantity_candidates[col] = score
                    break
            
            # Check for price column
            for keyword in self.price_patterns:
                if keyword in col_lower:
                    # Price columns typically have decimal points
                    is_mostly_int = (self.data[col] % 1 == 0).mean() > 0.9
                    score = 7 + (2 if not is_mostly_int else 0)
                    price_candidates[col] = score
                    break
            
            # Check for cost column
            for keyword in cost_patterns:
                if keyword in col_lower:
                    score = 8 + (2 if col_lower == keyword else 0)
                    cost_candidates[col] = score
                    break
            
            # Check for profit column
            for keyword in profit_patterns:
                if keyword in col_lower:
                    score = 8 + (2 if col_lower == keyword else 0)
                    profit_candidates[col] = score
                    break
        
        # Second pass - statistical analysis for numeric columns
        for col in self.column_classes['numeric_candidates']:
            # Skip already identified columns
            if col == self.detected_columns['date'] or col == self.detected_columns['product']:
                continue
            
            # Skip columns with high confidence already
            if (col in sales_candidates and sales_candidates[col] >= 8) or \
               (col in quantity_candidates and quantity_candidates[col] >= 8) or \
               (col in price_candidates and price_candidates[col] >= 8):
                continue
                
            # Get column statistics
            try:
                mean_val = self.data[col].mean()
                median_val = self.data[col].median()
                std_val = self.data[col].std()
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                is_mostly_int = (self.data[col] % 1 == 0).mean() > 0.9
                
                # Skip columns with negative values for price and quantity
                if min_val < 0 and (col not in revenue_candidates and col not in profit_candidates):
                    continue
                
                # Quantity characteristics
                if is_mostly_int and mean_val >= 1 and mean_val <= 100 and max_val < 10000:
                    score = 5 + (mean_val / 20)  # Higher mean values more likely to be quantity
                    quantity_candidates[col] = max(quantity_candidates.get(col, 0), score)
                
                # Price characteristics
                if not is_mostly_int and mean_val > 1 and mean_val < 1000:
                    # Prices often have decimal places and are within a reasonable range
                    score = 5 + min(mean_val / 100, 3)
                    price_candidates[col] = max(price_candidates.get(col, 0), score)
                
                # Revenue/sales characteristics
                if mean_val > 10 and std_val > 10:
                    # Revenue often has large values with high variability
                    score = 4 + min(mean_val / 1000, 3) + min(std_val / mean_val, 3)
                    sales_candidates[col] = max(sales_candidates.get(col, 0), score)
                    revenue_candidates[col] = max(revenue_candidates.get(col, 0), score)
                
                # Additional checks for identifying sales vs. price vs. quantity
                if col in sales_candidates and col in price_candidates and col in quantity_candidates:
                    # If a column is a candidate for multiple roles, resolve ambiguity
                    if is_mostly_int and mean_val < 100:
                        # More likely to be quantity
                        quantity_candidates[col] += 2
                        del sales_candidates[col]
                        del price_candidates[col]
                    elif not is_mostly_int and 1 < mean_val < 100:
                        # More likely to be price
                        price_candidates[col] += 2
                        del sales_candidates[col]
                        del quantity_candidates[col]
                    elif mean_val > 100:
                        # More likely to be sales/revenue
                        sales_candidates[col] += 2
                        del price_candidates[col]
                        del quantity_candidates[col]
            except:
                pass
        
        # Third pass - check for relationships between columns (e.g., quantity * price ≈ sales)
        quantity_col = None
        price_col = None
        
        if quantity_candidates and price_candidates:
            # Get best candidates
            quantity_col = max(quantity_candidates.items(), key=lambda x: x[1])[0]
            price_col = max(price_candidates.items(), key=lambda x: x[1])[0]
            
            # Check if a column approximately equals quantity * price
            calculated_sales = self.data[quantity_col] * self.data[price_col]
            
            for col in self.column_classes['numeric_candidates']:
                if col == quantity_col or col == price_col:
                    continue
                
                # Calculate correlation with the product of quantity and price
                try:
                    correlation = calculated_sales.corr(self.data[col])
                    if correlation > 0.9:  # High correlation indicates this is likely sales
                        sales_candidates[col] = max(sales_candidates.get(col, 0), 9)
                        logger.info(f"Column {col} identified as sales based on correlation with quantity*price")
                except:
                    pass
        
        # If we found quantity and price but no sales, calculate it
        if not sales_candidates and quantity_col and price_col:
            logger.info(f"Found quantity column: {quantity_col} and price column: {price_col}")
            logger.info("Calculating sales column from quantity and price")
            
            self.data['calculated_sales'] = self.data[quantity_col] * self.data[price_col]
            sales_candidates['calculated_sales'] = 9
        
        # Store detected financial columns
        if sales_candidates:
            best_sales_col = max(sales_candidates.items(), key=lambda x: x[1])[0]
            self.detected_columns['sales'] = best_sales_col
            logger.info(f"Sales column identified: {best_sales_col} (score: {sales_candidates[best_sales_col]:.1f}/10)")
        
        if quantity_candidates:
            best_quantity_col = max(quantity_candidates.items(), key=lambda x: x[1])[0]
            self.detected_columns['quantity'] = best_quantity_col
            logger.info(f"Quantity column identified: {best_quantity_col} (score: {quantity_candidates[best_quantity_col]:.1f}/10)")
        
        if price_candidates:
            best_price_col = max(price_candidates.items(), key=lambda x: x[1])[0]
            self.detected_columns['price'] = best_price_col
            logger.info(f"Price column identified: {best_price_col} (score: {price_candidates[best_price_col]:.1f}/10)")
        
        if revenue_candidates:
            best_revenue_col = max(revenue_candidates.items(), key=lambda x: x[1])[0]
            self.detected_columns['revenue'] = best_revenue_col
            logger.info(f"Revenue column identified: {best_revenue_col} (score: {revenue_candidates[best_revenue_col]:.1f}/10)")
        
        if cost_candidates:
            best_cost_col = max(cost_candidates.items(), key=lambda x: x[1])[0]
            self.detected_columns['cost'] = best_cost_col
            logger.info(f"Cost column identified: {best_cost_col} (score: {cost_candidates[best_cost_col]:.1f}/10)")
        
        if profit_candidates:
            best_profit_col = max(profit_candidates.items(), key=lambda x: x[1])[0]
            self.detected_columns['profit'] = best_profit_col
            logger.info(f"Profit column identified: {best_profit_col} (score: {profit_candidates[best_profit_col]:.1f}/10)")
        
        # Check if we identified a sales column
        if self.detected_columns['sales']:
            return True
        else:
            logger.error("Could not identify sales column. Please specify it manually.")
            return False
    
    def identify_customer_column(self):
        """Identify customer-related columns"""
        # Customer column patterns
        customer_patterns = ['customer', 'client', 'buyer', 'account', 'cust_']
        
        # Score candidates
        customer_candidates = {}
        
        # Check categorical and text candidates
        relevant_candidates = self.column_classes['categorical_candidates'] + self.column_classes['text_candidates'] + self.column_classes['id_candidates']
        
        for col in relevant_candidates:
            col_lower = str(col).lower()
            
            # Skip already identified columns
            if col in [self.detected_columns[key] for key in self.detected_columns if self.detected_columns[key]]:
                continue
            
            # Check for customer column
            for keyword in customer_patterns:
                if keyword in col_lower:
                    customer_candidates[col] = 8
                    break
        
        if customer_candidates:
            best_customer_col = max(customer_candidates.items(), key=lambda x: x[1])[0]
            self.detected_columns['customer'] = best_customer_col
            logger.info(f"Customer column identified: {best_customer_col}")
        
        return True
    
    def identify_location_column(self):
        """Identify location-related columns"""
        # Location column patterns
        location_patterns = ['location', 'region', 'country', 'state', 'city', 'zip', 'postal', 'territory']
        
        # Score candidates
        location_candidates = {}
        
        # Check categorical and text candidates
        relevant_candidates = self.column_classes['categorical_candidates'] + self.column_classes['text_candidates']
        
        for col in relevant_candidates:
            col_lower = str(col).lower()
            
            # Skip already identified columns
            if col in [self.detected_columns[key] for key in self.detected_columns if self.detected_columns[key]]:
                continue
            
            # Check for location column
            for keyword in location_patterns:
                if keyword in col_lower:
                    location_candidates[col] = 8
                    break
        
        if location_candidates:
            best_location_col = max(location_candidates.items(), key=lambda x: x[1])[0]
            self.detected_columns['location'] = best_location_col
            logger.info(f"Location column identified: {best_location_col}")
        
        return True