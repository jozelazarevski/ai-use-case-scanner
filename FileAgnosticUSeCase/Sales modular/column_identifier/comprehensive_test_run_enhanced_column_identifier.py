"""
Comprehensive Column Identifier Test Script - Simplified Version

This script provides a thorough test of the column identifier system
without using special Unicode characters that might cause encoding issues.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import uuid
import random
import string
import traceback

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('comprehensive_column_identifier')

# Import the column identifier
try:
    from column_identifier import ColumnIdentifier
except ImportError as e:
    logger.error(f"Could not import ColumnIdentifier: {str(e)}")
    logger.error("Please make sure the column_identifier module is properly installed")
    exit(1)

# Define the matrix of must-have columns for analysis
MUST_HAVE_COLUMNS = {
    'date': ['date', 'order_date', 'created_at', 'composite_date', 'year_date'],
    'year': ['year', 'fiscal_year'],
    'month': ['month'],
    'day': ['day'],
    'product': ['product_id', 'product_name'],
    'product_id': ['product_id'],
    'product_name': ['product_name'],
    'product_category': ['product_category', 'category'],
    'sales': ['sales', 'revenue', 'calculated_sales'],
    'quantity': ['quantity'],
    'price': ['price', 'unit_price'],
    'revenue': ['revenue', 'sales'],
    'cost': ['cost'],
    'profit': ['profit'],
    'customer': ['customer_id', 'customer_name'],
    'location': ['city', 'state', 'country', 'region', 'address']
}

def generate_random_string(length=10):
    """Generate a random string of given length"""
    return ''.join(random.choice(string.ascii_letters) for _ in range(length))

def generate_comprehensive_test_data(rows=1000):
    """Generate sample data covering all comprehensive column types"""
    # Base date for date columns
    base_date = datetime(2023, 1, 1)
    dates = [base_date + timedelta(days=i % 365) for i in range(rows)]
    
    # Dictionary to store all our data
    data = {}
    
    # ----- Core Types -----
    
    # ID columns
    data['id'] = list(range(1, rows + 1))
    data['uuid'] = [str(uuid.uuid4()) for _ in range(rows)]
    data['customer_id'] = [f'CUST-{np.random.randint(1, 5000):04d}' for _ in range(rows)]
    data['product_id'] = [f'PROD-{np.random.randint(1, 1000):04d}' for _ in range(rows)]
    data['order_id'] = [f'ORD-{i:06d}' for i in range(rows)]
    
    # Name columns
    data['name'] = [f"Person {i}" for i in range(rows)]
    data['product_name'] = [f"Product {i}" for i in range(rows)]
    data['company_name'] = [f"Company {random.choice(['Inc', 'LLC', 'Corp', 'Ltd'])} {i}" for i in range(rows)]
    
    # Description columns
    data['description'] = [f"This is a description for item {i}" for i in range(rows)]
    data['notes'] = [f"Note for record {i}" if i % 5 == 0 else "" for i in range(rows)]
    
    # Boolean columns
    data['is_active'] = np.random.choice([True, False], rows, p=[0.7, 0.3])
    data['has_subscription'] = np.random.choice([1, 0], rows, p=[0.5, 0.5])
    data['approved'] = np.random.choice(['Yes', 'No'], rows, p=[0.8, 0.2])
    
    # Category columns
    data['category'] = np.random.choice(['Electronics', 'Clothing', 'Home', 'Food', 'Sports'], rows)
    data['status'] = np.random.choice(['Pending', 'Approved', 'Rejected', 'On Hold'], rows)
    data['priority'] = np.random.choice(['Low', 'Medium', 'High', 'Critical'], rows)
    
    # ----- Date and Time Types -----
    
    # Date columns
    data['date'] = dates
    data['order_date'] = [date.strftime('%Y-%m-%d') for date in dates]
    data['created_at'] = [date.strftime('%Y-%m-%d %H:%M:%S') for date in dates]
    
    # Date components
    data['year'] = [date.year for date in dates]
    data['month'] = [date.month for date in dates]
    data['day'] = [date.day for date in dates]
    data['quarter'] = [((date.month - 1) // 3) + 1 for date in dates]
    data['fiscal_year'] = [date.year if date.month >= 7 else date.year - 1 for date in dates]
    data['week'] = [(date - datetime(date.year, 1, 1)).days // 7 + 1 for date in dates]
    
    # Time columns
    data['time'] = [f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}:{random.randint(0, 59):02d}" for _ in range(rows)]
    data['hour'] = [random.randint(0, 23) for _ in range(rows)]
    data['minute'] = [random.randint(0, 59) for _ in range(rows)]
    data['second'] = [random.randint(0, 59) for _ in range(rows)]
    
    # Timezone and Duration
    data['timezone'] = np.random.choice(['UTC', 'EST', 'CST', 'MST', 'PST'], rows)
    data['duration_minutes'] = [random.randint(1, 180) for _ in range(rows)]
    data['duration_hours'] = [round(random.uniform(0.5, 10), 1) for _ in range(rows)]
    
    # ----- Customer Related Types -----
    
    data['customer_name'] = [f"Customer {i}" for i in range(rows)]
    data['customer_email'] = [f"customer{i}@example.com" for i in range(rows)]
    data['customer_phone'] = [f"+1-{random.randint(200, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}" for _ in range(rows)]
    data['customer_status'] = np.random.choice(['Active', 'Inactive', 'Pending'], rows)
    data['customer_segment'] = np.random.choice(['Gold', 'Silver', 'Bronze', 'Standard'], rows)
    data['customer_age'] = np.random.randint(18, 90, rows)
    data['customer_gender'] = np.random.choice(['M', 'F', 'Other'], rows)
    data['customer_acquisition_source'] = np.random.choice(['Web', 'Email', 'Social', 'Referral', 'Direct'], rows)
    data['customer_lifetime_value'] = [round(random.uniform(100, 10000), 2) for _ in range(rows)]
    
    # ----- Location Related Types -----
    
    data['address'] = [f"{random.randint(1, 9999)} Main St." for _ in range(rows)]
    data['city'] = np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia'], rows)
    data['state'] = np.random.choice(['NY', 'CA', 'TX', 'FL', 'IL', 'PA'], rows)
    data['postal_code'] = [f"{random.randint(10000, 99999)}" for _ in range(rows)]
    data['country'] = np.random.choice(['US', 'CA', 'UK', 'DE', 'FR', 'JP'], rows)
    data['region'] = np.random.choice(['North', 'South', 'East', 'West', 'Central'], rows)
    data['latitude'] = [round(random.uniform(25, 50), 6) for _ in range(rows)]
    data['longitude'] = [round(random.uniform(-130, -70), 6) for _ in range(rows)]
    data['store_location'] = [f"Store-{random.randint(1, 100)}" for _ in range(rows)]
    
    # ----- Product Related Types -----
    
    data['product_category'] = np.random.choice(['Electronics', 'Clothing', 'Home', 'Food', 'Sports'], rows)
    data['product_brand'] = np.random.choice(['Brand A', 'Brand B', 'Brand C', 'Brand D', 'Brand E'], rows)
    data['product_color'] = np.random.choice(['Red', 'Blue', 'Green', 'Black', 'White'], rows)
    data['product_size'] = np.random.choice(['S', 'M', 'L', 'XL', 'XXL'], rows)
    data['product_weight'] = [round(random.uniform(0.1, 50), 2) for _ in range(rows)]
    data['product_material'] = np.random.choice(['Cotton', 'Wool', 'Polyester', 'Leather', 'Metal', 'Plastic'], rows)
    data['product_rating'] = [round(random.uniform(1, 5), 1) for _ in range(rows)]
    data['product_availability'] = np.random.choice(['In Stock', 'Out of Stock', 'Backordered', 'Discontinued'], rows)
    data['product_url'] = [f"https://example.com/products/{random.randint(1000, 9999)}" for _ in range(rows)]
    
    # ----- Financial Types -----
    
    data['price'] = [round(random.uniform(10, 1000), 2) for _ in range(rows)]
    data['unit_price'] = [round(random.uniform(5, 500), 2) for _ in range(rows)]
    data['quantity'] = np.random.randint(1, 20, rows)
    data['sales'] = [round(random.uniform(100, 10000), 2) for _ in range(rows)]
    data['revenue'] = [round(random.uniform(1000, 100000), 2) for _ in range(rows)]
    data['cost'] = [round(random.uniform(50, 5000), 2) for _ in range(rows)]
    data['profit'] = [round(random.uniform(-500, 5000), 2) for _ in range(rows)]
    data['discount'] = [round(random.uniform(0, 0.5), 2) for _ in range(rows)]
    data['tax'] = [round(random.uniform(0.05, 0.25), 2) for _ in range(rows)]
    data['shipping_cost'] = [round(random.uniform(0, 100), 2) for _ in range(rows)]
    data['currency'] = np.random.choice(['USD', 'EUR', 'GBP', 'CAD', 'JPY', 'AUD'], rows)
    data['total'] = [round(random.uniform(50, 5000), 2) for _ in range(rows)]
    data['margin'] = [round(random.uniform(0.1, 0.5), 2) for _ in range(rows)]
    
    # ----- Marketing Related Types -----
    
    data['campaign_id'] = [f"CAM-{random.randint(100, 999)}" for _ in range(rows)]
    data['channel'] = np.random.choice(['Email', 'Social', 'Search', 'Display', 'Direct', 'Partner'], rows)
    data['promotion_code'] = [f"PROMO{random.randint(100, 999)}" for _ in range(rows)]
    data['segment'] = np.random.choice(['New', 'Active', 'Lapsed', 'Loyal', 'High Value'], rows)
    data['conversion_rate'] = [round(random.uniform(0.01, 0.25), 4) for _ in range(rows)]
    data['marketing_cost'] = [round(random.uniform(100, 10000), 2) for _ in range(rows)]
    data['marketing_roi'] = [round(random.uniform(-0.5, 5), 2) for _ in range(rows)]
    data['attribution'] = np.random.choice(['First Click', 'Last Click', 'Linear', 'Position', 'Time Decay'], rows)
    
    # ----- Operational Types -----
    
    data['order_status'] = np.random.choice(['New', 'Processing', 'Shipped', 'Delivered', 'Cancelled'], rows)
    data['shipping_method'] = np.random.choice(['Standard', 'Express', 'Next Day', 'Ground', 'Pickup'], rows)
    data['tracking_id'] = [f"{random.choice(['UPS', 'FDX', 'DHL'])}{random.randint(1000000, 9999999)}" for _ in range(rows)]
    data['inventory_status'] = np.random.choice(['In Stock', 'Low Stock', 'Out of Stock', 'On Order'], rows)
    data['batch_number'] = [f"BN-{random.randint(1000, 9999)}" for _ in range(rows)]
    data['order_line'] = np.random.randint(1, 10, rows)
    
    # Create DataFrame from all of this data
    df = pd.DataFrame(data)
    
    # Create composite date columns
    try:
        # Create a year+month column
        df['year_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
        
        # Create a year+month+day column
        df['year_month_day'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-' + df['day'].astype(str).str.zfill(2)
        
        # Create a calculated sales column (quantity * price)
        df['calculated_sales'] = df['quantity'] * df['unit_price']
    except Exception as e:
        logger.warning(f"Error creating composite columns: {e}")
    
    return df

def check_column_identification(identifier, expected_columns_matrix):
    """
    Check if the identifier correctly identified the expected columns
    
    Args:
        identifier: Column identifier with detected columns
        expected_columns_matrix: Dictionary with column types and lists of potential column names
        
    Returns:
        Dictionary with identification results
    """
    results = {}
    
    for col_type, expected_names in expected_columns_matrix.items():
        detected_col = identifier.detected_columns.get(col_type)
        
        if detected_col:
            # Check if the detected column is one of the expected ones
            if detected_col in expected_names:
                results[col_type] = {
                    'status': 'CORRECT',
                    'detected': detected_col,
                    'candidates': expected_names
                }
            else:
                # Special handling for composite dates
                if col_type == 'date' and (
                    'year_month' in detected_col or 
                    'year_month_day' in detected_col or 
                    'composite_date' in detected_col
                ):
                    results[col_type] = {
                        'status': 'CORRECT (composite)',
                        'detected': detected_col,
                        'candidates': expected_names
                    }
                # Special handling for calculated sales
                elif col_type == 'sales' and 'calculated' in detected_col:
                    results[col_type] = {
                        'status': 'CORRECT (calculated)',
                        'detected': detected_col,
                        'candidates': expected_names
                    }
                else:
                    results[col_type] = {
                        'status': 'INCORRECT',
                        'detected': detected_col,
                        'candidates': expected_names
                    }
        else:
            results[col_type] = {
                'status': 'MISSING',
                'detected': None,
                'candidates': expected_names
            }
    
    return results



def identify_sales_columns(self, column_matrix=None):
    """
    Identify sales-specific columns with enhanced handling of related columns
    
    Args:
        column_matrix (list, optional): Custom column matrix for sales data columns.
            If None, uses a default sales column matrix.
    
    Returns:
        bool: True if critical columns were successfully identified, False otherwise
    """
    try:
        # Use provided column matrix or default to sales-specific columns
        if column_matrix is None:
            # Define default matrix of columns needed for sales analysis
            column_matrix = [
                # Critical columns - analysis can't work without these
                ('date', 1, ['date', 'order_date', 'transaction_date', 'invoice_date'], 
                    ['date', 'time', 'day', 'timestamp'], 'date'),
                ('product', 1, ['product_id', 'product', 'item_id', 'sku'], 
                    ['product', 'item', 'sku', 'merchandise'], 'id'),
                ('sales', 1, ['sales', 'revenue', 'amount', 'total_sales'], 
                    ['sales', 'revenue', 'amount', 'value', 'total'], 'numeric'),
                
                # Important columns - improve analysis quality
                ('quantity', 2, ['quantity', 'qty', 'units', 'count'], 
                    ['quantity', 'qty', 'count', 'units', 'volume'], 'numeric'),
                ('price', 2, ['price', 'unit_price', 'item_price', 'rate'], 
                    ['price', 'unit_price', 'rate', 'cost'], 'numeric'),
                ('customer', 2, ['customer_id', 'customer', 'client_id', 'client'], 
                    ['customer', 'client', 'buyer', 'account'], 'id'),
                ('product_id', 2, ['product_id', 'sku', 'item_id'], 
                    ['id', 'code', 'number', 'sku'], 'id'),
                ('product_name', 2, ['product_name', 'item_name', 'product_desc'], 
                    ['name', 'description', 'desc', 'title'], 'text'),
                
                # Useful but optional columns
                ('year', 3, ['year', 'fiscal_year', 'yr'], 
                    ['year', 'yr', 'yyyy'], 'numeric'),
                ('month', 3, ['month', 'mon', 'mm'], 
                    ['month', 'mon', 'mm'], 'numeric'),
                ('day', 3, ['day', 'dy', 'dd'], 
                    ['day', 'dy', 'dd'], 'numeric'),
                ('product_category', 3, ['category', 'product_category', 'product_type'], 
                    ['category', 'type', 'group', 'class'], 'categorical'),
                ('discount', 3, ['discount', 'discount_amount', 'discount_pct'], 
                    ['discount', 'reduction', 'markdown'], 'numeric'),
                ('cost', 3, ['cost', 'unit_cost', 'total_cost'], 
                    ['cost', 'expense', 'expenditure'], 'numeric'),
                ('profit', 3, ['profit', 'margin', 'gross_profit'], 
                    ['profit', 'margin', 'earnings'], 'numeric'),
                ('revenue', 3, ['revenue', 'total_revenue', 'sales_revenue'], 
                    ['revenue', 'income', 'turnover'], 'numeric'),
                ('location', 3, ['location', 'region', 'country', 'city', 'state'], 
                    ['location', 'region', 'country', 'city', 'address'], 'text')
            ]
        
        # First, perform the base column classification
        self._classify_columns()
        
        # Track successful identifications
        identified_columns = {}
        critical_columns_found = 0
        total_critical_columns = sum(1 for col in column_matrix if col[1] == 1)
        
        # First pass: Process direct matches by exact name
        for col_key, importance, exact_names, patterns, data_type in column_matrix:
            for col in self.data.columns:
                col_lower = str(col).lower()
                
                # Check for exact matches (case insensitive)
                if col_lower in [name.lower() for name in exact_names]:
                    identified_columns[col_key] = col
                    self.detected_columns[col_key] = col
                    
                    # Count critical columns
                    if importance == 1:
                        critical_columns_found += 1
                    
                    # Skip to next column type
                    break
        
        # Second pass: Pattern matching for columns not found
        for col_key, importance, exact_names, patterns, data_type in column_matrix:
            # Skip if already identified
            if col_key in identified_columns:
                continue
                
            candidates = {}
            
            for col in self.data.columns:
                col_lower = str(col).lower()
                score = 0
                
                # Skip columns already assigned to another type
                if col in identified_columns.values():
                    continue
                
                # Check for pattern matches in column name
                for pattern in patterns:
                    if pattern in col_lower:
                        score += 3
                        break
                
                # Prioritize by data type
                if data_type == 'date':
                    if col in self.column_classes['date_candidates']:
                        score += 5
                        
                elif data_type == 'id':
                    if col in self.column_classes['id_candidates']:
                        score += 3
                    elif self._has_id_pattern(col_lower):
                        score += 2
                        
                elif data_type == 'numeric':
                    if self._is_numeric_column(col):
                        score += 3
                        
                        # Special case for sales/amount columns
                        if col_key == 'sales':
                            # Sales columns typically have larger values
                            try:
                                mean_val = self.data[col].mean()
                                if mean_val > 10:  # Typically sales values are larger
                                    score += 2
                            except:
                                pass
                        
                        # Special case for quantity columns
                        elif col_key == 'quantity':
                            # Quantity columns are usually integers
                            try:
                                if (self.data[col] % 1 == 0).mean() > 0.9:  # Mostly integers
                                    score += 2
                            except:
                                pass
                
                elif data_type == 'text':
                    if col in self.column_classes['text_candidates']:
                        score += 3
                        
                elif data_type == 'categorical':
                    if col in self.column_classes['categorical_candidates']:
                        score += 3
                
                # Store score if positive
                if score > 0:
                    candidates[col] = score
            
            # Select best candidate if any
            if candidates:
                best_col = max(candidates.items(), key=lambda x: x[1])[0]
                identified_columns[col_key] = best_col
                self.detected_columns[col_key] = best_col
                
                # Count critical columns
                if importance == 1:
                    critical_columns_found += 1
        
        # Third pass: Handle relationships between columns
        # Map of column types that can reuse the same original column
        related_columns = {
            'product_id': ['product'],       # product_id can be the same as product
            'product': ['product_id'],       # product can be the same as product_id
            'revenue': ['sales'],            # revenue can be the same as sales
            'sales': ['revenue']             # sales can be the same as revenue
        }
        
        # Process related columns
        for col_key, related_keys in related_columns.items():
            # Skip if already identified
            if col_key in identified_columns:
                continue
                
            # Check if any related column is identified
            for related_key in related_keys:
                if related_key in identified_columns:
                    related_col = identified_columns[related_key]
                    # Check if the related column is valid for this column type
                    col_candidates = [name.lower() for name in dict(column_matrix)[col_key][2]]
                    if str(related_col).lower() in col_candidates:
                        # Use the same original column for this type
                        identified_columns[col_key] = related_col
                        self.detected_columns[col_key] = related_col
                        
                        # Count critical column if needed
                        if dict(column_matrix)[col_key][1] == 1:
                            critical_columns_found += 1
                        
                        break
        
        # Fourth pass: Create composite date from year, month, day if needed
        if 'date' not in identified_columns and 'year' in identified_columns and 'month' in identified_columns:
            year_col = identified_columns['year']
            month_col = identified_columns['month']
            
            try:
                # Create a new date column by combining year and month
                comp_date_col = 'composite_date'
                
                if 'day' in identified_columns:
                    day_col = identified_columns['day']
                    self.data[comp_date_col] = pd.to_datetime(
                        self.data[year_col].astype(str) + '-' + 
                        self.data[month_col].astype(str).str.zfill(2) + '-' + 
                        self.data[day_col].astype(str).str.zfill(2),
                        errors='coerce'
                    )
                else:
                    # Use first day of month if no day column
                    self.data[comp_date_col] = pd.to_datetime(
                        self.data[year_col].astype(str) + '-' + 
                        self.data[month_col].astype(str).str.zfill(2) + '-01',
                        errors='coerce'
                    )
                
                # Check if the date conversion was successful
                if self.data[comp_date_col].notna().mean() > 0.7:  # At least 70% valid dates
                    identified_columns['date'] = comp_date_col
                    self.detected_columns['date'] = comp_date_col
                    
                    # Count as critical if date is critical
                    if any(col[0] == 'date' and col[1] == 1 for col in column_matrix):
                        critical_columns_found += 1
            except Exception as e:
                logger.warning(f"Error creating composite date: {str(e)}")
        
        # Update instance variables for standard columns
        if 'date' in self.detected_columns:
            self.date_col = self.detected_columns['date']
        if 'product' in self.detected_columns:
            self.product_col = self.detected_columns['product']
        if 'sales' in self.detected_columns:
            self.sales_col = self.detected_columns['sales']
        
        # Log results
        logger.info("\nSales Columns Detection Summary:")
        for col_type, col_name in sorted(self.detected_columns.items()):
            logger.info(f"  {col_type.capitalize()}: {col_name}")
        
        # Indicate success if all critical columns were found
        if critical_columns_found == total_critical_columns:
            logger.info("All critical sales columns successfully identified")
            return True
        else:
            missing_critical = [col[0] for col in column_matrix 
                              if col[1] == 1 and col[0] not in identified_columns]
            logger.warning(f"Missing critical sales columns: {', '.join(missing_critical)}")
            return critical_columns_found > 0  # Partial success if at least one critical column was found
    
    except Exception as e:
        logger.error(f"Error identifying sales columns: {str(e)}")
        logger.debug(traceback.format_exc())
        return False

import json 
import os 


def run_column_identifier(data=None, column_matrix=None):
    """
    Run the column identifier and return a standardized DataFrame with mapped columns.
    
    Args:
        data (DataFrame, optional): Input DataFrame to process. If None, generates test data.
        column_matrix (list, optional): A list of tuples defining columns to identify.
            Each tuple should be: (column_key, importance, exact_names, pattern_keywords, data_type)
            If None, uses the default matrix from MUST_HAVE_COLUMNS.
        
    Returns:
        tuple: (identifier, results, mapped_df, column_mapping_json)
    """
    try:
        # Generate sample data if none provided
        if data is None:
            logger.info("Generating comprehensive test data...")
            data = generate_comprehensive_test_data()
            logger.info(f"Test data generated: {len(data)} rows, {len(data.columns)} columns")
        print(data.columns)
        # Initialize the column identifier
        logger.info("Initializing column identifier...")
        identifier = ColumnIdentifier(data)
        
        # Run the identification with the provided column matrix if available
        logger.info("Identifying columns...")
        if column_matrix:
            success = identifier.identify_all_columns(column_matrix=column_matrix)
        else:
            success = identifier.identify_all_columns()
        
        if not success:
            logger.error("Column identification failed")
            return None, None, None, None
            
        # Print all the detected columns
        logger.info("\nAll Detected Columns:")
        for col_type, col_name in identifier.detected_columns.items():
            if col_name:
                logger.info(f"  {col_type}: {col_name}")
        
        # Compare identified columns with expected columns
        results = {}
        
        # If column_matrix is provided, use it to define expected columns
        if column_matrix:
            expected_columns = {}
            for col_info in column_matrix:
                col_type, _, exact_names, _, _ = col_info
                expected_columns[col_type] = exact_names
        else:
            # Otherwise use MUST_HAVE_COLUMNS
            expected_columns = MUST_HAVE_COLUMNS
        
        # Check each expected column type
        for col_type, expected_names in expected_columns.items():
            detected_col = identifier.detected_columns.get(col_type)
            
            if detected_col:
                # Check if the detected column is one of the expected ones
                if detected_col in expected_names:
                    results[col_type] = {
                        'status': 'CORRECT',
                        'detected': detected_col,
                        'candidates': expected_names
                    }
                else:
                    # Special handling for composite dates
                    if col_type == 'date' and (
                        'year_month' in detected_col or 
                        'year_month_day' in detected_col or 
                        'composite_date' in detected_col
                    ):
                        results[col_type] = {
                            'status': 'CORRECT (composite)',
                            'detected': detected_col,
                            'candidates': expected_names
                        }
                    # Special handling for calculated sales
                    elif col_type == 'sales' and 'calculated' in detected_col:
                        results[col_type] = {
                            'status': 'CORRECT (calculated)',
                            'detected': detected_col,
                            'candidates': expected_names
                        }
                    else:
                        results[col_type] = {
                            'status': 'INCORRECT',
                            'detected': detected_col,
                            'candidates': expected_names
                        }
            else:
                results[col_type] = {
                    'status': 'MISSING',
                    'detected': None,
                    'candidates': expected_names
                }
        
        # Print the identification results matrix
        logger.info("\n" + "="*80)
        logger.info("COLUMN IDENTIFICATION MATRIX")
        logger.info("="*80)
        
        # Calculate the maximum length for better formatting
        max_col_type_len = max(len(col_type) for col_type in expected_columns.keys())
        max_detected_len = max(len(str(identifier.detected_columns.get(col_type, ''))) 
                              for col_type in expected_columns.keys())
        max_expected_len = max(len(str(candidates)) 
                              for candidates in expected_columns.values())
        
        # Print header
        header_format = f"| {{:<{max_col_type_len}}} | {{:<{max_detected_len}}} | {{:<20}} | {{:<{max_expected_len}}} |"
        logger.info(header_format.format("COLUMN TYPE", "DETECTED", "STATUS", "EXPECTED CANDIDATES"))
        logger.info("-" * (max_col_type_len + max_detected_len + max_expected_len + 40))
        
        # Print results for each column type
        for col_type, result in sorted(results.items()):
            status = result['status']
            detected = result['detected'] or 'None'
            candidates = ', '.join(result['candidates'])
            
            # Format status with ASCII indicators
            status_display = status
            if status.startswith('CORRECT'):
                status_display = "(+) " + status
            elif status == 'INCORRECT':
                status_display = "(-) " + status
            elif status == 'MISSING':
                status_display = "(!) " + status
            
            logger.info(header_format.format(col_type, detected, status_display, candidates))
        
        # Calculate summary statistics
        correct_count = sum(1 for result in results.values() if result['status'].startswith('CORRECT'))
        incorrect_count = sum(1 for result in results.values() if result['status'] == 'INCORRECT')
        missing_count = sum(1 for result in results.values() if result['status'] == 'MISSING')
        total_count = len(results)
        
        # Print summary
        logger.info("="*80)
        logger.info(f"SUMMARY: {correct_count}/{total_count} correct identifications")
        logger.info(f"  - Correct: {correct_count}")
        logger.info(f"  - Incorrect: {incorrect_count}")
        logger.info(f"  - Missing: {missing_count}")
        logger.info("="*80)
        
        # Create a new DataFrame with standardized column names
        logger.info("Creating standardized DataFrame with mapped columns...")
        
        # Start with a copy of the original data
        mapped_df = data.copy()
        
        # Dictionary to store column mappings (standard_name -> original_name)
        column_mapping = {}
        
        # Add all correctly identified columns to the mapping
        for col_type, result in results.items():
            if result['status'].startswith('CORRECT') and result['detected']:
                column_mapping[col_type] = result['detected']
        
        # Add additional detected columns not in results
        for col_type, col_name in identifier.detected_columns.items():
            if col_name and col_type not in column_mapping:
                column_mapping[col_type] = col_name
        
        # Create the final DataFrame with standardized column names
        final_columns = {}
        for standard_name, original_name in column_mapping.items():
            if original_name in mapped_df.columns:
                final_columns[standard_name] = mapped_df[original_name]
        
        final_df = pd.DataFrame(final_columns)
        
        # Log information about the final DataFrame
        logger.info(f"Created standardized DataFrame with {len(final_df.columns)} columns:")
        for col in final_df.columns:
            logger.info(f"  - {col} (from original: {column_mapping[col]})")
        
        # Create JSON representation of the column mapping
        column_mapping_json = json.dumps(column_mapping, indent=2)
        
        # Store the mapping as an attribute on the identifier
        identifier.column_mapping = column_mapping
        identifier.column_mapping_json = column_mapping_json
        identifier.mapped_df = final_df
        
        # Return the requested values
        return identifier, results, final_df, column_mapping_json
            
    except Exception as e:
        logger.error(f"Error running column identifier: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None, None
    
    
def main():
    """Main function to run the script"""
    logger.info("Starting Comprehensive Column Identifier Test")
    
    # Run with generated comprehensive test data
    identifier, results, mapped_df, column_mapping_json = run_column_identifier()
    
    if mapped_df is not None:
        # Save the standardized DataFrame
        mapped_df.to_csv('standardized_data.csv', index=False)
        logger.info(f"Standardized DataFrame saved to standardized_data.csv")
        
        # Save the column mapping as JSON
        with open('column_mapping.json', 'w') as f:
            f.write(column_mapping_json)
        logger.info(f"Column mapping saved to column_mapping.json")
    
    logger.info("Comprehensive Column Identifier Test complete")
    return identifier, results

if __name__ == "__main__":
    main()