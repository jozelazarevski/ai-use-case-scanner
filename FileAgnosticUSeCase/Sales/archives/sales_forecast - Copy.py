#!/usr/bin/env python3
"""
Sales Forecasting Script

This script is designed to:
1. Load various file formats (CSV, Excel, JSON, etc.)
2. Automatically identify time series and sales columns
3. Perform data cleaning and transformation
4. Build and train a forecasting model
5. Generate future predictions and visualizations

Usage:
    python sales_forecast.py --file path/to/data_file --date_column "Date" --sales_column "Sales" --forecast_periods 12

If date_column and sales_column are not provided, the script will attempt to identify them automatically.
"""

import argparse
import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import xgboost as xgb
from pmdarima import auto_arima
import warnings
import re
from patterns_keyowrds import patterns,exact_keywords
# Optional but recommended for fuzzy matching
try:
    from fuzzywuzzy import fuzz1
except ImportError:
    pass  # The code will still work without fuzzy matching
    
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('sales_forecast')

# Suppress warnings
warnings.filterwarnings("ignore")

class SalesForecaster:
    """Class for sales forecasting operations"""
    
    def __init__(self, file_path, date_col=None, sales_col=None, forecast_periods=12):
         
        # Add this line to store detected columns
        self.detected_columns = {}
        """
        Initialize the forecaster
        
        Args:
            file_path (str): Path to the data file
            date_col (str, optional): Name of the date column
            sales_col (str, optional): Name of the sales column
            forecast_periods (int): Number of periods to forecast
        """
        self.file_path = file_path
        self.date_col = date_col
        self.sales_col = sales_col
        self.forecast_periods = forecast_periods
        self.data = None
        self.cleaned_data = None
        self.train_data = None
        self.test_data = None
        self.model = None
        self.predictions = None
        self.future_predictions = None
        self.seasonality = None
        self.time_frequency = None
        
    def load_data(self):
        """Load data from various file formats"""
        logger.info(f"Loading data from {self.file_path}...")
        
        file_extension = os.path.splitext(self.file_path)[1].lower()
        
        try:
            if file_extension == '.csv':
                self.data = pd.read_csv(self.file_path)
            elif file_extension in ['.xls', '.xlsx']:
                self.data = pd.read_excel(self.file_path)
            elif file_extension == '.json':
                self.data = pd.read_json(self.file_path)
            elif file_extension == '.parquet':
                self.data = pd.read_parquet(self.file_path)
            elif file_extension == '.txt':
                # Try different delimiters for text files
                for delimiter in [',', '\t', '|', ';']:
                    try:
                        self.data = pd.read_csv(self.file_path, delimiter=delimiter)
                        if len(self.data.columns) > 1:
                            break
                    except:
                        continue
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
            logger.info(f"Data loaded successfully with {self.data.shape[0]} rows and {self.data.shape[1]} columns.")
            logger.info(f"Columns: {', '.join(self.data.columns)}")
            
            # Display data summary
            logger.info("\nData Sample:")
            logger.info(self.data.head(5).to_string())
            
            logger.info("\nData Types:")
            logger.info(self.data.dtypes)
            
            logger.info("\nBasic Statistics:")
            logger.info(self.data.describe().to_string())
            
            return True
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return False
    
 
        """Automatically identify date and sales columns if not provided"""
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return False
        
        # Identify date column if not provided
        if self.date_col is None:
            logger.info("Attempting to identify date column...")
            date_columns = []
            
            # Enhanced date column keywords
            date_keywords = [
                'date', 'dt', 'time', 'day', 'month', 'year', 'period', 
                'order_date', 'orderdate', 'ord_date', 'sale_date', 'saledate', 
                'transaction_date', 'transdate', 'invoice_date', 'invoicedate', 
                'ship_date', 'shipdate', 'delivery_date', 'purchase_date'
            ]
            
            # Check column names that might indicate date
            for col in self.data.columns:
                col_lower = str(col).lower()
                # Check exact matches
                if col_lower in date_keywords:
                    date_columns.append(col)
                    continue
                # Check partial matches
                for keyword in date_keywords:
                    if keyword in col_lower:
                        date_columns.append(col)
                        break
            
            # Check column data types for datetime
            for col in self.data.columns:
                if col not in date_columns:
                    try:
                        if pd.to_datetime(self.data[col], errors='coerce').notna().sum() > self.data.shape[0] * 0.7:
                            date_columns.append(col)
                    except:
                        pass
            
            if date_columns:
                # Prioritize columns with more obvious date names
                priority_keywords = ['order_date', 'orderdate', 'invoice_date', 'transaction_date', 'date']
                for keyword in priority_keywords:
                    for col in date_columns:
                        col_lower = str(col).lower()
                        if keyword in col_lower or col_lower == keyword:
                            self.date_col = col
                            break
                    if self.date_col:
                        break
                
                # If no priority match, use the first date column found
                if not self.date_col:
                    self.date_col = date_columns[0]
                
                logger.info(f"Date column identified: {self.date_col}")
            else:
                logger.error("Could not identify date column. Please specify it manually.")
                return False
        
        # Identify sales column if not provided
        if self.sales_col is None:
            logger.info("Attempting to identify sales column...")
            
            # Enhanced column name keywords
            sales_keywords = [
                'sales', 'sale', 'revenue', 'rev', 'income', 'turnover', 
                'total_sales', 'totalsales', 'sales_amount', 'salesamt', 
                'gross_sales', 'net_sales', 'sales_value', 'salesval'
            ]
            
            quantity_keywords = [
                'quantity', 'qty', 'order_qty', 'orderqty', 'ord_qty', 
                'units', 'unit_count', 'count', 'volume', 'vol', 
                'num_items', 'numitems', 'item_count', 'pieces', 'pcs'
            ]
            
            price_keywords = [
                'price', 'unit_price', 'unitprice', 'price_per_unit', 'ppu',
                'rate', 'cost', 'unit_cost', 'unitcost', 'amount', 'amt',
                'item_price', 'item_cost', 'price_per_item'
            ]
            
            discount_value_keywords = [
                'discount_value', 'discountvalue', 'discount_amount', 'discountamt',
                'discount_amt', 'disc_value', 'disc_val', 'discval', 'discount_dollars',
                'rebate_amount', 'promo_value', 'promo_amount'
            ]
            
            discount_pct_keywords = [
                'discount', 'disc', 'discount_rate', 'discountrate', 'discount_percent',
                'discountpct', 'discount_pct', 'disc_rate', 'disc_pct', 'discount_percentage',
                'rebate_rate', 'promo_rate', 'promo_discount'
            ]
            
            # Check for an existing sales/revenue column first
            sales_col = None
            for col in self.data.columns:
                col_lower = str(col).lower()
                # Check exact matches first
                if col_lower in sales_keywords:
                    if pd.api.types.is_numeric_dtype(self.data[col]):
                        sales_col = col
                        logger.info(f"Direct sales column found: {sales_col}")
                        break
                # Then check partial matches
                for keyword in sales_keywords:
                    if keyword in col_lower:
                        if pd.api.types.is_numeric_dtype(self.data[col]):
                            sales_col = col
                            logger.info(f"Direct sales column found: {sales_col}")
                            break
                if sales_col:
                    break
            
            # If a direct sales column exists, check for discount columns to adjust it
            if sales_col:
                # Look for discount columns
                discount_value_col = None
                discount_percentage_col = None
                
                for col in self.data.columns:
                    col_lower = str(col).lower()
                    
                    # Skip non-numeric columns
                    if not pd.api.types.is_numeric_dtype(self.data[col]):
                        continue
                    
                    # Look for discount value column (amount)
                    # Check exact matches first
                    if col_lower in discount_value_keywords:
                        discount_value_col = col
                        break
                    # Then check partial matches
                    for keyword in discount_value_keywords:
                        if keyword in col_lower:
                            discount_value_col = col
                            break
                    if discount_value_col:
                        break
                
                # If no discount value found, look for percentage
                if not discount_value_col:
                    for col in self.data.columns:
                        col_lower = str(col).lower()
                        
                        # Skip non-numeric columns
                        if not pd.api.types.is_numeric_dtype(self.data[col]):
                            continue
                        
                        # Check exact matches first
                        if col_lower in discount_pct_keywords:
                            # Make sure it's not the value column again
                            if 'value' not in col_lower and 'amount' not in col_lower and 'amt' not in col_lower:
                                discount_percentage_col = col
                                break
                        # Then check partial matches
                        for keyword in discount_pct_keywords:
                            if keyword in col_lower:
                                # Make sure it's not the value column again
                                if 'value' not in col_lower and 'amount' not in col_lower and 'amt' not in col_lower:
                                    discount_percentage_col = col
                                    break
                        if discount_percentage_col:
                            break
                
                # Check if we need to adjust the sales value based on discounts
                if discount_value_col:
                    logger.info(f"Discount value column found: {discount_value_col}")
                    logger.info("Adjusting sales figures to add back discount values for accurate forecasting")
                    
                    # Create a new sales column that adds back the discount value
                    original_sales_col = sales_col
                    self.data['adjusted_sales'] = self.data[original_sales_col] + self.data[discount_value_col]
                    sales_col = 'adjusted_sales'
                    logger.info(f"Created adjusted sales column: {sales_col}")
                
                elif discount_percentage_col:
                    logger.info(f"Discount percentage column found: {discount_percentage_col}")
                    logger.info("Adjusting sales figures based on discount percentages")
                    
                    # Check if discount is stored as percentage (e.g., 5 for 5%) or decimal (e.g., 0.05 for 5%)
                    max_discount = self.data[discount_percentage_col].max()
                    
                    # Adjust based on how discount is stored
                    if max_discount > 1:
                        # Stored as percentage (e.g., 5 for 5%)
                        self.data['adjusted_sales'] = self.data[sales_col] / (1 - self.data[discount_percentage_col]/100)
                    else:
                        # Stored as decimal (e.g., 0.05 for 5%)
                        self.data['adjusted_sales'] = self.data[sales_col] / (1 - self.data[discount_percentage_col])
                    
                    sales_col = 'adjusted_sales'
                    logger.info(f"Created adjusted sales column: {sales_col}")
                
                # Set the identified sales column
                self.sales_col = sales_col
                return True
            
            # If no direct sales column, try to calculate from components
            # Check for quantity, price, and discount columns
            quantity_col = None
            price_col = None
            discount_value_col = None
            discount_percentage_col = None
            
            # Find quantity column
            for col in self.data.columns:
                col_lower = str(col).lower()
                
                # Skip the identified date column
                if col == self.date_col:
                    continue
                
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    continue
                
                # Check exact matches first
                if col_lower in quantity_keywords:
                    quantity_col = col
                    break
                # Then check partial matches
                for keyword in quantity_keywords:
                    if keyword in col_lower:
                        quantity_col = col
                        break
                if quantity_col:
                    break
            
            # Find price column
            for col in self.data.columns:
                col_lower = str(col).lower()
                
                # Skip the identified date column and quantity column
                if col == self.date_col or col == quantity_col:
                    continue
                
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    continue
                
                # Check exact matches first
                if col_lower in price_keywords:
                    price_col = col
                    break
                # Then check partial matches
                for keyword in price_keywords:
                    if keyword in col_lower:
                        price_col = col
                        break
                if price_col:
                    break
            
            # Find discount value column
            for col in self.data.columns:
                col_lower = str(col).lower()
                
                # Skip columns already identified
                if col == self.date_col or col == quantity_col or col == price_col:
                    continue
                
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    continue
                
                # Check exact matches first
                if col_lower in discount_value_keywords:
                    discount_value_col = col
                    break
                # Then check partial matches
                for keyword in discount_value_keywords:
                    if keyword in col_lower:
                        discount_value_col = col
                        break
                if discount_value_col:
                    break
            
            # Find discount percentage column if no discount value found
            if not discount_value_col:
                for col in self.data.columns:
                    col_lower = str(col).lower()
                    
                    # Skip columns already identified
                    if col == self.date_col or col == quantity_col or col == price_col:
                        continue
                    
                    # Check if column is numeric
                    if not pd.api.types.is_numeric_dtype(self.data[col]):
                        continue
                    
                    # Check exact matches first
                    if col_lower in discount_pct_keywords:
                        if 'value' not in col_lower and 'amount' not in col_lower and 'amt' not in col_lower:
                            discount_percentage_col = col
                            break
                    # Then check partial matches
                    for keyword in discount_pct_keywords:
                        if keyword in col_lower:
                            if 'value' not in col_lower and 'amount' not in col_lower and 'amt' not in col_lower:
                                discount_percentage_col = col
                                break
                    if discount_percentage_col:
                        break
            
            # Calculate total sales based on available columns
            if quantity_col and price_col:
                logger.info(f"Found quantity column: {quantity_col} and price column: {price_col}")
                
                # Create base sales calculation
                self.data['calculated_sales'] = self.data[quantity_col] * self.data[price_col]
                
                # Apply discount if available
                if discount_value_col:
                    logger.info(f"Found discount value column: {discount_value_col}")
                    logger.info("Calculating total sales with discount value adjustment")
                    # We add back discount value to get pre-discount sales for accurate forecasting
                    self.data['calculated_total_sales'] = self.data['calculated_sales'] + self.data[discount_value_col]
                elif discount_percentage_col:
                    logger.info(f"Found discount percentage column: {discount_percentage_col}")
                    logger.info("Calculating total sales with discount percentage adjustment")
                    
                    # Check if discount is stored as percentage (e.g., 5 for 5%) or decimal (e.g., 0.05 for 5%)
                    max_discount = self.data[discount_percentage_col].max()
                    
                    # Adjust based on how discount is stored
                    if max_discount > 1:
                        # Stored as percentage (e.g., 5 for 5%)
                        self.data['calculated_total_sales'] = self.data['calculated_sales'] / (1 - self.data[discount_percentage_col]/100)
                    else:
                        # Stored as decimal (e.g., 0.05 for 5%)
                        self.data['calculated_total_sales'] = self.data['calculated_sales'] / (1 - self.data[discount_percentage_col])
                else:
                    logger.info("No discount columns found. Using base sales calculation.")
                    self.data['calculated_total_sales'] = self.data['calculated_sales']
                
                self.sales_col = 'calculated_total_sales'
                logger.info(f"Created sales column: {self.sales_col}")
                return True
            
            # If we couldn't calculate sales, look for any numeric column that might be sales
            sales_candidates = {}
            
            # Use enhanced sales keywords
            extended_sales_keywords = sales_keywords + [
                'gp', 'gross_profit', 'grossprofit', 'net_profit', 'netprofit',
                'transaction', 'trans', 'transaction_value', 'transvalue',
                'order_value', 'ordervalue', 'order_amount', 'orderamt',
                'value', 'val', 'total', 'tot'
            ]
            
            for col in self.data.columns:
                col_lower = str(col).lower()
                
                # Skip the identified date column
                if col == self.date_col:
                    continue
                
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(self.data[col]):
                    continue
                
                # Give points based on column name relevance
                points = 0
                
                # Check exact matches (higher points)
                if col_lower in extended_sales_keywords:
                    points += 5
                
                # Check partial matches
                for keyword in extended_sales_keywords:
                    if keyword in col_lower:
                        points += 3
                        
                # Give points based on statistical properties
                # Higher mean and variance generally indicate monetary values
                if self.data[col].mean() > 10:
                    points += 2
                if self.data[col].std() > 5:
                    points += 1
                    
                # Store points for this column
                if points > 0:
                    sales_candidates[col] = points
            
            if sales_candidates:
                # Select the column with the highest points
                self.sales_col = max(sales_candidates.items(), key=lambda x: x[1])[0]
                logger.info(f"Sales column identified: {self.sales_col} (confidence score: {sales_candidates[self.sales_col]})")
            else:
                # If no specific sales column found, check if quantity column exists
                if quantity_col:
                    logger.info(f"No specific sales column found, using quantity column: {quantity_col}")
                    self.sales_col = quantity_col
                else:
                    # If no column found by name, look for numeric columns
                    numeric_cols = [col for col in self.data.columns 
                                   if pd.api.types.is_numeric_dtype(self.data[col]) and col != self.date_col]
                    if numeric_cols:
                        # Choose the column with the highest variation
                        variations = {col: self.data[col].std() / self.data[col].mean() if self.data[col].mean() != 0 else 0 
                                    for col in numeric_cols}
                        self.sales_col = max(variations.items(), key=lambda x: x[1])[0]
                        logger.info(f"Sales column identified based on numeric variation: {self.sales_col}")
                    else:
                        logger.error("Could not identify sales column. Please specify it manually.")
                        return False
        
        # Verify columns exist
        if self.date_col not in self.data.columns:
            logger.error(f"Date column '{self.date_col}' not found in data.")
            return False
            
        if self.sales_col not in self.data.columns:
            logger.error(f"Sales column '{self.sales_col}' not found in data.")
            return False
        
        return True
    
    def calculate_sales_value(self, base_col=None, quantity_col=None, price_col=None, 
                       discount_value_col=None, discount_pct_col=None):
        """
        Calculate the true sales value based on available columns and economic formulas.
        
        This function implements various economic formulas to calculate the correct sales value
        based on the columns available in the dataset.
        
        Args:
            base_col (str): Name of existing sales/revenue column (if available)
            quantity_col (str): Name of quantity column
            price_col (str): Name of unit price column
            discount_value_col (str): Name of discount value (amount) column
            discount_pct_col (str): Name of discount percentage column
            
        Returns:
            str: Name of the calculated sales column
        """
        # Log the calculation approach
        logger.info("Calculating true sales value using economic formulas")
        
        # Case 1: We have a direct sales column and possibly discount information
        if base_col is not None:
            logger.info(f"Starting with direct sales column: {base_col}")
            
            # First check if this is pre-discount or post-discount sales
            # Typically, most sales columns contain the post-discount value (what was actually paid)
            
            if discount_value_col is not None:
                # Formula: Original Sales = Discounted Sales + Discount Amount
                logger.info(f"Discount value column found: {discount_value_col}")
                logger.info("FORMULA USED: Original Sales = Discounted Sales + Discount Amount")
                logger.info(f"Original Price = {base_col} + {discount_value_col}")
                
                # Create a new sales column with the original (pre-discount) value
                self.data['original_sales'] = self.data[base_col] + self.data[discount_value_col]
                return 'original_sales'
                
            elif discount_pct_col is not None:
                # Check if discount is stored as percentage (e.g., 5 for 5%) or decimal (e.g., 0.05 for 5%)
                max_discount = self.data[discount_pct_col].max()
                
                logger.info(f"Discount percentage column found: {discount_pct_col}")
                
                # Formula: Original Sales = Discounted Sales / (1 - Discount Rate)
                if max_discount > 1:
                    # Stored as percentage (e.g., 5 for 5%)
                    logger.info("FORMULA USED: Original Sales = Discounted Sales / (1 - Discount Percentage/100)")
                    logger.info(f"Original Price = {base_col} / (1 - {discount_pct_col}/100)")
                    self.data['original_sales'] = self.data[base_col] / (1 - self.data[discount_pct_col]/100)
                else:
                    # Stored as decimal (e.g., 0.05 for 5%)
                    logger.info("FORMULA USED: Original Sales = Discounted Sales / (1 - Discount Rate)")
                    logger.info(f"Original Price = {base_col} / (1 - {discount_pct_col})")
                    self.data['original_sales'] = self.data[base_col] / (1 - self.data[discount_pct_col])
                
                return 'original_sales'
            
            # If no discount information, use the sales column as is
            logger.info("No discount information found. Using direct sales column as is.")
            return base_col
            
        # Case 2: We need to calculate sales from quantity and price
        elif quantity_col is not None and price_col is not None:
            logger.info(f"Calculating sales from quantity and price: {quantity_col} × {price_col}")
            
            # Formula: Base Sales = Quantity × Unit Price
            logger.info("FORMULA USED: Base Sales = Quantity × Unit Price")
            logger.info(f"Base Sales = {quantity_col} × {price_col}")
            self.data['calculated_sales'] = self.data[quantity_col] * self.data[price_col]
            
            if discount_value_col is not None:
                # If we have the discount value, this is likely a pre-discount calculation
                # Formula: Final Sales = Base Sales - Discount Amount
                logger.info(f"Discount value column found: {discount_value_col}")
                logger.info("FORMULA USED: Final Sales = Base Sales - Discount Amount")
                logger.info(f"Final Sales = {quantity_col} × {price_col} - {discount_value_col}")
                
                self.data['final_sales'] = self.data['calculated_sales'] - self.data[discount_value_col]
                
                # For forecasting, we typically want the pre-discount value for consistency
                logger.info("Using pre-discount value (calculated_sales) for forecasting consistency")
                return 'calculated_sales'
                
            elif discount_pct_col is not None:
                # Check if discount is stored as percentage or decimal
                max_discount = self.data[discount_pct_col].max()
                
                logger.info(f"Discount percentage column found: {discount_pct_col}")
                
                # Formula: Final Sales = Base Sales × (1 - Discount Rate)
                if max_discount > 1:
                    # Stored as percentage
                    logger.info("FORMULA USED: Final Sales = Base Sales × (1 - Discount Percentage/100)")
                    logger.info(f"Final Sales = {quantity_col} × {price_col} × (1 - {discount_pct_col}/100)")
                    self.data['final_sales'] = self.data['calculated_sales'] * (1 - self.data[discount_pct_col]/100)
                else:
                    # Stored as decimal
                    logger.info("FORMULA USED: Final Sales = Base Sales × (1 - Discount Rate)")
                    logger.info(f"Final Sales = {quantity_col} × {price_col} × (1 - {discount_pct_col})")
                    self.data['final_sales'] = self.data['calculated_sales'] * (1 - self.data[discount_pct_col])
                
                # For forecasting, we typically want the pre-discount value for consistency
                logger.info("Using pre-discount value (calculated_sales) for forecasting consistency")
                return 'calculated_sales'
            
            # If no discount information, use the calculated sales as is
            logger.info("No discount information found. Using base sales calculation.")
            return 'calculated_sales'
            
        # Case 3: We couldn't calculate a reliable sales figure
        else:
            return None
        
    
    def identify_columns(self):
        """
        Automatically identify date, sales, and other relevant columns in the dataset.
        
        Uses advanced pattern matching, fuzzy text matching, and column characteristics
        to identify various types of columns in the dataset.
        """
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return False
            
        # Import fuzzy matching library
        try:
            from fuzzywuzzy import fuzz
            has_fuzzy = True
        except ImportError:
            logger.warning("fuzzywuzzy library not found. Using regex matching only.")
            has_fuzzy = False
        
        # Initialize column detection results
        detected_columns = {
            'date': None,
            'sales': None,
            'quantity': None,
            'price': None,
            'discount_value': None,
            'discount_pct': None,
            'product': None,
            'customer': None,
            'location': None,
            'order_id': None,
        }
        
       
        # Identify date column if not provided
        if self.date_col is None:
            logger.info("Attempting to identify date column...")
            
            # Direct match using regex patterns
            date_scores = {}
            
            # First pass - check regex patterns
            for col in self.data.columns:
                col_lower = str(col).lower()
                for pattern in patterns['date']:
                    if re.search(pattern, col_lower):
                        # Higher score for matches at the beginning
                        if re.match(pattern, col_lower):
                            date_scores[col] = 100
                        else:
                            date_scores[col] = 80
                        break
            
            # Second pass - check data types for datetime format if no strong matches
            if not date_scores or max(date_scores.values() or [0]) < 90:
                for col in self.data.columns:
                    if col not in date_scores:
                        try:
                            if pd.to_datetime(self.data[col], errors='coerce').notna().sum() > self.data.shape[0] * 0.7:
                                date_scores[col] = 90  # High score for valid datetime columns
                        except:
                            pass
            
            # Third pass - fuzzy matching if available
            if has_fuzzy and (not date_scores or max(date_scores.values() or [0]) < 80):
                for col in self.data.columns:
                    col_lower = str(col).lower()
                    # Skip columns already scored highly
                    if col in date_scores and date_scores[col] >= 80:
                        continue
                        
                    # Calculate fuzzy match score with each keyword
                    max_score = 0
                    for keyword in exact_keywords['date']:
                        # Score exact matches higher
                        if col_lower == keyword:
                            score = 100
                        # Score contains matches
                        elif keyword in col_lower:
                            score = 85
                        # Score fuzzy matches
                        else:
                            try:
                                score = fuzz.partial_ratio(col_lower, keyword)
                            except Exception as e:
                                logger.warning(f"Error in fuzzy matching: {str(e)}")
                                score = 0
                        max_score = max(max_score, score)
                    
                    if max_score >= 75:  # Threshold for fuzzy matches
                        date_scores[col] = max_score
            
            # Select best candidate
            if date_scores:
                try:
                    best_col = max(date_scores.items(), key=lambda x: x[1])[0]
                    detected_columns['date'] = best_col
                    self.date_col = best_col
                    logger.info(f"Date column identified: {self.date_col} (confidence: {date_scores[best_col]}%)")
                except ValueError:
                    logger.error("Error finding best date column from scores")
                    return False
            else:
                logger.error("Could not identify date column. Please specify it manually.")
                return False
        else:
            detected_columns['date'] = self.date_col
            
        # Identify sales-related columns if not provided
        if self.sales_col is None:
            logger.info("Attempting to identify sales and related columns...")
            
            # Column scores for each type
            column_scores = {
                   'date': {},  # Add this line to include 'date' in column_scores
                   'sales': {},
                   'quantity': {},
                   'price': {},
                   'discount_value': {},
                   'discount_pct': {},
                   'product': {},
                   'customer': {},
                   'location': {},
                   'order_id': {}
               }
            
            # First pass - direct regex matching
            for col_type, col_patterns in patterns.items():
                for col in self.data.columns:
                    # Skip the identified date column
                    if col == self.date_col:
                        continue
                        
                    # Sales-related columns should be numeric
                    if col_type in ['sales', 'quantity', 'price', 'discount_value', 'discount_pct'] and \
                       not pd.api.types.is_numeric_dtype(self.data[col]):
                        continue
                    
                    col_lower = str(col).lower()
                    for pattern in col_patterns:
                        if re.search(pattern, col_lower):
                            # Higher score for exact matches
                            if re.match(pattern, col_lower):
                                column_scores[col_type][col] = 100
                            else:
                                column_scores[col_type][col] = 85
                            break
            
            # Second pass - fuzzy matching if available
            if has_fuzzy:
                for col_type, keywords in exact_keywords.items():
                    # Skip 'date' since we already processed it separately
                    if col_type == 'date':
                        continue
                        
                    # Only analyze numeric columns for numeric fields
                    cols_to_check = []
                    for col in self.data.columns:
                        if col == self.date_col:
                            continue
                        if col_type in ['sales', 'quantity', 'price', 'discount_value', 'discount_pct'] and \
                           not pd.api.types.is_numeric_dtype(self.data[col]):
                            continue
                        cols_to_check.append(col)
                        
                    for col in cols_to_check:
                        # Skip columns already scored highly
                        if col in column_scores[col_type] and column_scores[col_type][col] >= 85:
                            continue
                            
                        col_lower = str(col).lower()
                        max_score = 0
                        for keyword in keywords:
                            # Exact match
                            if col_lower == keyword:
                                score = 100
                            # Contains match
                            elif keyword in col_lower:
                                score = 90
                            # Token set ratio handles word order and partial matches
                            else:
                                try:
                                    score = fuzz.token_set_ratio(col_lower, keyword)
                                except Exception as e:
                                    logger.warning(f"Error in fuzzy matching for {col_type}: {str(e)}")
                                    score = 0
                            max_score = max(max_score, score)
                        
                        if max_score >= 75:  # Threshold for fuzzy matches
                            if col not in column_scores[col_type]:
                                column_scores[col_type][col] = max_score
                            else:
                                column_scores[col_type][col] = max(column_scores[col_type][col], max_score)
            
            # Third pass - statistical analysis for sales columns
            # Look for columns with monetary characteristics
            numeric_cols = [col for col in self.data.columns 
                             if pd.api.types.is_numeric_dtype(self.data[col]) and col != self.date_col]
            
            for col in numeric_cols:
                # Skip columns already with high confidence
                if col in column_scores['sales'] and column_scores['sales'][col] >= 85:
                    continue
                    
                stats_score = 0
                
                # Higher mean values suggest monetary columns
                mean_val = self.data[col].mean()
                if mean_val > 100:
                    stats_score += 20
                elif mean_val > 50:
                    stats_score += 15
                elif mean_val > 10:
                    stats_score += 10
                    
                # Higher variability suggests monetary values
                if self.data[col].std() > 100:
                    stats_score += 15
                elif self.data[col].std() > 50:
                    stats_score += 10
                elif self.data[col].std() > 10:
                    stats_score += 5
                    
                # Non-negative values are typical for sales/quantity
                if self.data[col].min() >= 0:
                    stats_score += 10
                    
                # Non-integer values suggest monetary amounts vs counts
                is_mostly_int = (self.data[col] % 1 == 0).mean() > 0.9
                if not is_mostly_int:
                    stats_score += 10  # More likely to be price or sales than quantity
                    
                # Add to scores if significant
                if stats_score >= 30:
                    if col not in column_scores['sales']:
                        column_scores['sales'][col] = stats_score
                    else:
                        column_scores['sales'][col] += stats_score / 4  # Add some weight but don't double count
            
            # Resolve the best candidates for each column type
            for col_type in column_scores:
                if column_scores[col_type]:
                    try:
                        best_col = max(column_scores[col_type].items(), key=lambda x: x[1])[0]
                        score = column_scores[col_type][best_col]
                        if score >= 70:  # Threshold for accepting a column
                            detected_columns[col_type] = best_col
                            logger.info(f"{col_type.capitalize()} column identified: {best_col} (confidence: {score:.0f}%)")
                    except ValueError:
                        logger.warning(f"No valid scores for {col_type} column")
            
            # Special case for quantity columns that are integers
            if detected_columns['quantity'] and pd.api.types.is_numeric_dtype(self.data[detected_columns['quantity']]):
                is_mostly_int = (self.data[detected_columns['quantity']] % 1 == 0).mean() > 0.9
                if is_mostly_int:
                    logger.info(f"Quantity column {detected_columns['quantity']} contains integer values, confirming it as quantity")
                    column_scores['quantity'][detected_columns['quantity']] += 10
            
            # Calculate sales value using the appropriate method
            if not detected_columns['sales']:
                # Try to calculate from components            
                calculated_sales_col = self.calculate_sales_value(
                    base_col=None,
                    quantity_col=detected_columns['quantity'],
                    price_col=detected_columns['price'],
                    discount_value_col=detected_columns['discount_value'],
                    discount_pct_col=detected_columns['discount_pct']
                )
                
                if calculated_sales_col:
                    detected_columns['sales'] = calculated_sales_col
                    logger.info(f"Sales column calculated from components: {calculated_sales_col}")
                else:
                    # Last resort: find most likely numeric column
                    fallback_col = None
                    highest_score = 0
                    
                    for col in numeric_cols:
                        if col == self.date_col:
                            continue
                            
                        score = 0
                        # Prefer columns with higher variation
                        if self.data[col].std() > 0:
                            variation = self.data[col].std() / self.data[col].mean() if self.data[col].mean() != 0 else 0
                            score = min(variation * 30, 70)  # Cap at 70
                        
                        if score > highest_score:
                            highest_score = score
                            fallback_col = col
                    
                    if fallback_col:
                        detected_columns['sales'] = fallback_col
                        logger.info(f"Sales column identified by statistical properties: {fallback_col} (confidence: {highest_score:.0f}%)")
                    else:
                        logger.error("Could not identify or calculate sales column. Please specify it manually.")
                        return False
            else:
                # Check if we need to adjust for discounts
                calculated_sales_col = self.calculate_sales_value(
                    base_col=detected_columns['sales'],
                    quantity_col=None,
                    price_col=None,
                    discount_value_col=detected_columns['discount_value'],
                    discount_pct_col=detected_columns['discount_pct']
                )
                
                if calculated_sales_col and calculated_sales_col != detected_columns['sales']:
                    detected_columns['sales'] = calculated_sales_col
                    logger.info(f"Sales column adjusted for discounts: {calculated_sales_col}")
            
            # Set final sales column
            self.sales_col = detected_columns['sales']
            logger.info(f"Final sales column for forecasting: {self.sales_col}")
        else:
            detected_columns['sales'] = self.sales_col
            
        # Verify that required columns exist
        if self.date_col not in self.data.columns:
            logger.error(f"Date column '{self.date_col}' not found in data.")
            return False
            
        if self.sales_col not in self.data.columns:
            logger.error(f"Sales column '{self.sales_col}' not found in data.")
            return False
        
        # Store detected columns for potential later use
        self.detected_columns = detected_columns
        
        # Report all detected columns as a summary
        logger.info("\nColumn Detection Summary:")
        for col_type, col_name in detected_columns.items():
            if col_name:
                logger.info(f"  {col_type.capitalize()}: {col_name}")
        
        return True    
    
    def preprocess_data(self):
        """Clean and transform data for forecasting"""
        if self.data is None or self.date_col is None or self.sales_col is None:
            logger.error("Data or columns not set. Call load_data() and identify_columns() first.")
            return False
            
        logger.info("Preprocessing data...")
        
        try:
            # Make a copy to avoid modifying the original data
            data = self.data.copy()
            
            # Convert date column to datetime
            data[self.date_col] = pd.to_datetime(data[self.date_col], errors='coerce')
            
            # Drop rows with invalid dates
            data = data.dropna(subset=[self.date_col])
            
            # Sort by date
            data = data.sort_values(by=self.date_col)
            
            # Ensure sales column is numeric
            data[self.sales_col] = pd.to_numeric(data[self.sales_col], errors='coerce')
            
            # Fill missing sales values with interpolation
            data[self.sales_col] = data[self.sales_col].interpolate(method='linear')
            
            # Create a clean dataframe with just the needed columns
            clean_df = data[[self.date_col, self.sales_col]].copy()
            clean_df = clean_df.dropna()
            
            # Detect time frequency (daily, weekly, monthly, etc.)
            date_diffs = clean_df[self.date_col].diff().dropna()
            if len(date_diffs) > 0:
                most_common_diff = date_diffs.value_counts().idxmax()
                
                if most_common_diff <= timedelta(days=1):
                    self.time_frequency = 'D'  # Daily
                elif most_common_diff <= timedelta(days=7):
                    self.time_frequency = 'W'  # Weekly
                elif most_common_diff <= timedelta(days=31):
                    self.time_frequency = 'M'  # Monthly
                elif most_common_diff <= timedelta(days=92):
                    self.time_frequency = 'Q'  # Quarterly
                else:
                    self.time_frequency = 'Y'  # Yearly
                    
                logger.info(f"Detected time frequency: {self.time_frequency}")
            else:
                self.time_frequency = 'D'  # Default to daily
            
            # If data is not regular, resample it
            clean_df = clean_df.set_index(self.date_col)
            
            # For aggregation, use sum by default for sales data
            clean_df = clean_df.resample(self.time_frequency).sum()
            
            # Reset index to keep date as a column
            clean_df = clean_df.reset_index()
            
            # Get the name of the date column after resampling
            self.date_col = clean_df.columns[0]
            
            # Create time-based features
            clean_df['year'] = clean_df[self.date_col].dt.year
            clean_df['month'] = clean_df[self.date_col].dt.month
            clean_df['day_of_week'] = clean_df[self.date_col].dt.dayofweek
            clean_df['quarter'] = clean_df[self.date_col].dt.quarter
            clean_df['day_of_year'] = clean_df[self.date_col].dt.dayofyear
            
            # Add lag features
            for lag in [1, 2, 3, 4, 12]:
                if len(clean_df) > lag:
                    clean_df[f'lag_{lag}'] = clean_df[self.sales_col].shift(lag)
            
            # Add rolling statistics
            for window in [3, 6, 12]:
                if len(clean_df) > window:
                    clean_df[f'rolling_mean_{window}'] = clean_df[self.sales_col].rolling(window=window).mean()
                    clean_df[f'rolling_std_{window}'] = clean_df[self.sales_col].rolling(window=window).std()
            
            # Drop NaN values created by lag and rolling features
            clean_df = clean_df.dropna()
            
            # Check for stationarity
            self.check_stationarity(clean_df[self.sales_col])
            
            # Check for seasonality
            self.check_seasonality(clean_df)
            
            self.cleaned_data = clean_df
            logger.info(f"Data preprocessing complete. Cleaned data shape: {clean_df.shape}")
            
            # Show preview of cleaned data
            logger.info("\nCleaned Data Sample:")
            logger.info(clean_df.head().to_string())
            
            return True
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return False
    
    def check_stationarity(self, series):
        """Check if time series is stationary using Augmented Dickey-Fuller test"""
        try:
            result = adfuller(series.dropna())
            logger.info(f"ADF Statistic: {result[0]}")
            logger.info(f"p-value: {result[1]}")
            
            if result[1] <= 0.05:
                logger.info("Series is stationary (p <= 0.05)")
            else:
                logger.info("Series is not stationary (p > 0.05)")
        except Exception as e:
            logger.warning(f"Could not check stationarity: {e}")
    
    def check_seasonality(self, data):
        """Check for seasonality in the time series"""
        try:
            if len(data) >= 2 * 12:  # Need at least two cycles for seasonal decomposition
                # Set the period based on the frequency
                if self.time_frequency == 'D':
                    period = 7  # Weekly seasonality
                elif self.time_frequency == 'W':
                    period = 52  # Yearly seasonality in weeks
                elif self.time_frequency == 'M':
                    period = 12  # Yearly seasonality in months
                elif self.time_frequency == 'Q':
                    period = 4  # Yearly seasonality in quarters
                else:
                    period = 1  # No seasonality for yearly data
                
                if period > 1:
                    # Perform seasonal decomposition
                    decomposition = seasonal_decompose(
                        data[self.sales_col], 
                        model='additive', 
                        period=period
                    )
                    
                    # Calculate strength of seasonality
                    seasonal_strength = 1 - (decomposition.resid.var() / decomposition.seasonal.var() if decomposition.seasonal.var() != 0 else 0)
                    
                    logger.info(f"Seasonal strength: {seasonal_strength:.4f}")
                    self.seasonality = seasonal_strength
                    
                    if seasonal_strength > 0.6:
                        logger.info("Strong seasonality detected")
                    elif seasonal_strength > 0.3:
                        logger.info("Moderate seasonality detected")
                    else:
                        logger.info("Weak or no seasonality detected")
        except Exception as e:
            logger.warning(f"Could not check seasonality: {e}")
            self.seasonality = None
    
    def split_data(self, test_size=0.2):
        """Split data into training and testing sets"""
        if self.cleaned_data is None:
            logger.error("No cleaned data available. Call preprocess_data() first.")
            return False
            
        try:
            # Determine the split point
            split_idx = int(len(self.cleaned_data) * (1 - test_size))
            
            # Split the data
            self.train_data = self.cleaned_data.iloc[:split_idx].copy()
            self.test_data = self.cleaned_data.iloc[split_idx:].copy()
            
            logger.info(f"Data split: Training set size: {len(self.train_data)}, Testing set size: {len(self.test_data)}")
            return True
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return False
    
    def build_model(self):
        """Build and train a forecasting model"""
        if self.train_data is None or self.test_data is None:
            logger.error("Training and testing data not available. Call split_data() first.")
            return False
            
        try:
            # Determine which model to use based on data characteristics
            if self.seasonality is not None and self.seasonality > 0.3:
                logger.info("Using SARIMA model due to detected seasonality")
                self._build_sarima_model()
            else:
                logger.info("Using XGBoost model")
                self._build_xgboost_model()
                
            return True
        except Exception as e:
            logger.error(f"Error building model: {e}")
            return False
    
    def _build_sarima_model(self):
        """Build and train a SARIMA model"""
        try:
            # Set up the training data
            y_train = self.train_data[self.sales_col]
            
            # Determine seasonality period
            if self.time_frequency == 'D':
                m = 7  # Weekly
            elif self.time_frequency == 'W':
                m = 52  # Annual
            elif self.time_frequency == 'M':
                m = 12  # Annual
            elif self.time_frequency == 'Q':
                m = 4  # Annual
            else:
                m = 1  # No seasonality
            
            # Use auto_arima to find the best parameters
            logger.info("Finding optimal SARIMA parameters...")
            auto_model = auto_arima(
                y_train,
                seasonal=True,
                m=m,
                stepwise=True,
                trace=False,
                error_action='ignore',
                suppress_warnings=True,
                max_p=5, max_d=2, max_q=5,
                max_P=2, max_D=1, max_Q=2
            )
            
            # Get parameters from auto_arima
            p, d, q = auto_model.order
            P, D, Q, m = auto_model.seasonal_order
            
            logger.info(f"Best SARIMA model: ({p},{d},{q})x({P},{D},{Q},{m})")
            
            # Fit the SARIMA model with the best parameters
            model = SARIMAX(
                y_train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            results = model.fit(disp=False)
            self.model = results
            
            # Generate predictions for the test set
            y_test = self.test_data[self.sales_col]
            predictions = results.forecast(steps=len(y_test))
            
            self.predictions = pd.Series(predictions, index=self.test_data.index)
            
            # Evaluate the model
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            logger.info(f"Model evaluation: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")
            
            # Generate future predictions
            future_steps = self.forecast_periods
            future_predictions = results.forecast(steps=future_steps)
            
            # Create a date range for future predictions
            last_date = self.cleaned_data[self.date_col].iloc[-1]
            
            if self.time_frequency == 'D':
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_steps, freq='D')
            elif self.time_frequency == 'W':
                future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=future_steps, freq='W')
            elif self.time_frequency == 'M':
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=31), periods=future_steps, freq='M')
            elif self.time_frequency == 'Q':
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=92), periods=future_steps, freq='Q')
            else:
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=365), periods=future_steps, freq='Y')
            
            # Create a DataFrame for future predictions
            self.future_predictions = pd.DataFrame({
                self.date_col: future_dates,
                self.sales_col: future_predictions
            })
            
            logger.info(f"Future predictions generated for {future_steps} periods ahead")
            
        except Exception as e:
            logger.error(f"Error building SARIMA model: {e}")
            return False
    
    def _build_xgboost_model(self):
        """Build and train an XGBoost model"""
        try:
            # Prepare features and target
            feature_cols = [col for col in self.train_data.columns if col not in [self.date_col, self.sales_col]]
            
            X_train = self.train_data[feature_cols]
            y_train = self.train_data[self.sales_col]
            
            X_test = self.test_data[feature_cols]
            y_test = self.test_data[self.sales_col]
            
            # Standardize features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Define XGBoost parameters
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.1,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 100,
                'silent': 1
            }
            
            # Train the XGBoost model
            model = xgb.XGBRegressor(**params)
            model.fit(X_train_scaled, y_train)
            
            # Save the model
            self.model = model
            
            # Get predictions for test set
            predictions = model.predict(X_test_scaled)
            
            self.predictions = pd.Series(predictions, index=self.test_data.index)
            
            # Evaluate the model
            mae = mean_absolute_error(y_test, predictions)
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            r2 = r2_score(y_test, predictions)
            
            logger.info(f"Model evaluation: MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")
            
            # Get feature importance
            importance = model.feature_importances_
            importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
            importance_df = importance_df.sort_values('Importance', ascending=False)
            logger.info("Top important features:")
            for i, (feature, imp) in enumerate(zip(importance_df['Feature'][:5], importance_df['Importance'][:5])):
                logger.info(f"{i+1}. {feature}: {imp:.4f}")
            
            # Generate future predictions
            # We need to create future features first
            future_features = self._generate_future_features(self.forecast_periods)
            
            # Scale the future features
            future_features_scaled = scaler.transform(future_features[feature_cols])
            
            # Predict
            future_predictions = model.predict(future_features_scaled)
            
            # Create a DataFrame for future predictions
            self.future_predictions = pd.DataFrame({
                self.date_col: future_features[self.date_col],
                self.sales_col: future_predictions
            })
            
            logger.info(f"Future predictions generated for {self.forecast_periods} periods ahead")
            
        except Exception as e:
            logger.error(f"Error building XGBoost model: {e}")
            return False
    
    def _generate_future_features(self, steps):
        """Generate features for future time periods"""
        last_date = self.cleaned_data[self.date_col].iloc[-1]
        last_data = self.cleaned_data.copy()
        
        # Create a date range for future dates
        if self.time_frequency == 'D':
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=steps, freq='D')
        elif self.time_frequency == 'W':
            future_dates = pd.date_range(start=last_date + pd.Timedelta(weeks=1), periods=steps, freq='W')
        elif self.time_frequency == 'M':
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=31), periods=steps, freq='M')
        elif self.time_frequency == 'Q':
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=92), periods=steps, freq='Q')
        else:
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=365), periods=steps, freq='Y')
        
        # Create a DataFrame with future dates
        future_df = pd.DataFrame({self.date_col: future_dates})
        
        # Generate time-based features
        future_df['year'] = future_df[self.date_col].dt.year
        future_df['month'] = future_df[self.date_col].dt.month
        future_df['day_of_week'] = future_df[self.date_col].dt.dayofweek
        future_df['quarter'] = future_df[self.date_col].dt.quarter
        future_df['day_of_year'] = future_df[self.date_col].dt.dayofyear
        
        # Initialize sales column with dummy values (will be replaced by predictions)
        future_df[self.sales_col] = 0
        
        # Combine last data with future data to generate lag features
        combined_df = pd.concat([last_data, future_df], ignore_index=True)
        
        # Generate lag features for future data
        for lag in [1, 2, 3, 4, 12]:
            if len(combined_df) > lag:
                combined_df[f'lag_{lag}'] = combined_df[self.sales_col].shift(lag)
        
        # Generate rolling features for future data
        for window in [3, 6, 12]:
            if len(combined_df) > window:
                combined_df[f'rolling_mean_{window}'] = combined_df[self.sales_col].rolling(window=window).mean()
                combined_df[f'rolling_std_{window}'] = combined_df[self.sales_col].rolling(window=window).std()
        
        # Get only the future part with generated features
        future_with_features = combined_df.tail(steps).fillna(0)
        
        return future_with_features
    
    def visualize_results(self):
        """Visualize the forecasting results"""
        if self.cleaned_data is None or self.predictions is None or self.future_predictions is None:
            logger.error("Predictions not available. Call build_model() first.")
            return False
            
        try:
            # Set up the plot style
            sns.set(style="whitegrid")
            plt.figure(figsize=(12, 8))
            
            # Plot historical data
            plt.plot(
                self.cleaned_data[self.date_col], 
                self.cleaned_data[self.sales_col],
                label='Historical Data',
                color='blue'
            )
            
            # Plot test predictions
            test_dates = self.test_data[self.date_col]
            plt.plot(
                test_dates,
                self.predictions,
                label='Test Predictions',
                color='green',
                linestyle='--'
            )
            
            # Plot future predictions
            plt.plot(
                self.future_predictions[self.date_col],
                self.future_predictions[self.sales_col],
                label='Future Forecast',
                color='red',
                linestyle='-.'
            )
            
            # Add confidence intervals for future predictions (simplified)
            std_dev = self.cleaned_data[self.sales_col].std()
            plt.fill_between(
                self.future_predictions[self.date_col],
                self.future_predictions[self.sales_col] - 1.96 * std_dev,
                self.future_predictions[self.sales_col] + 1.96 * std_dev,
                color='red',
                alpha=0.2,
                label='95% Confidence Interval'
            )
            
            # Add labels and title
            plt.xlabel('Date')
            plt.ylabel(f'{self.sales_col}')
            plt.title(f'Sales Forecast for {self.forecast_periods} {self.time_frequency} Periods')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save the plot
            output_file = 'sales_forecast_plot.png'
            plt.savefig(output_file)
            logger.info(f"Forecast visualization saved as {output_file}")
            
            plt.close()
            
            # Create a results table
            forecast_df = self.future_predictions.copy()
            forecast_df = forecast_df.rename(columns={self.sales_col: 'Forecasted Sales'})
            
            # Format date
            forecast_df[self.date_col] = forecast_df[self.date_col].dt.strftime('%Y-%m-%d')
            
            logger.info("\nSales Forecast Results:")
            logger.info(forecast_df.to_string(index=False))
            
            return True
        except Exception as e:
            logger.error(f"Error visualizing results: {e}")
            return False
    
    def export_results(self, output_format='csv'):
        """Export forecasting results to a file"""
        if self.future_predictions is None:
            logger.error("Predictions not available. Call build_model() first.")
            return False
            
        try:
            # Create a results DataFrame
            results_df = self.future_predictions.copy()
            
            # Determine output file name
            input_file_name = os.path.splitext(os.path.basename(self.file_path))[0]
            output_file = f"{input_file_name}_forecast.{output_format}"
            
            # Export based on format
            if output_format == 'csv':
                results_df.to_csv(output_file, index=False)
            elif output_format == 'excel':
                results_df.to_excel(output_file, index=False)
            elif output_format == 'json':
                results_df.to_json(output_file, orient='records', date_format='iso')
            else:
                logger.error(f"Unsupported export format: {output_format}")
                return False
                
            logger.info(f"Forecast results exported to {output_file}")
            
            # Also export the model evaluation metrics
            if hasattr(self, 'model'):
                if isinstance(self.model, xgb.XGBRegressor):
                    # Export feature importance for XGBoost model
                    feature_cols = [col for col in self.train_data.columns if col not in [self.date_col, self.sales_col]]
                    importance = self.model.feature_importances_
                    importance_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importance})
                    importance_df = importance_df.sort_values('Importance', ascending=False)
                    
                    importance_file = f"{input_file_name}_feature_importance.{output_format}"
                    if output_format == 'csv':
                        importance_df.to_csv(importance_file, index=False)
                    elif output_format == 'excel':
                        importance_df.to_excel(importance_file, index=False)
                    elif output_format == 'json':
                        importance_df.to_json(importance_file, orient='records')
                    
                    logger.info(f"Feature importance exported to {importance_file}")
                
                elif hasattr(self.model, 'summary'):
                    # Export model summary for SARIMA model
                    summary_file = f"{input_file_name}_model_summary.txt"
                    with open(summary_file, 'w') as f:
                        f.write(str(self.model.summary()))
                    
                    logger.info(f"Model summary exported to {summary_file}")
            
            return True
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False
    
    def run_pipeline(self):
        """Run the complete forecasting pipeline"""
        steps = [
            self.load_data,
            self.identify_columns,
            self.preprocess_data,
            self.split_data,
            self.build_model,
            self.visualize_results,
            self.export_results
        ]
        
        for step in steps:
            success = step()
            if not success:
                logger.error(f"Pipeline failed at step: {step.__name__}")
                return False
        
        logger.info("Forecasting pipeline completed successfully")
        return True


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Sales Forecasting Script")
    
    parser.add_argument('--file', required=True, help='Path to the data file')
    parser.add_argument('--date_column', help='Name of the date column')
    parser.add_argument('--sales_column', help='Name of the sales column')
    parser.add_argument('--forecast_periods', type=int, default=12, help='Number of periods to forecast')
    parser.add_argument('--export_format', default='csv', choices=['csv', 'excel', 'json'], help='Export format for results')
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_arguments()
    
    # Create and run the forecaster
    forecaster = SalesForecaster(
        file_path=args.file,
        date_col=args.date_column,
        sales_col=args.sales_column,
        forecast_periods=args.forecast_periods
    )
    
    # Run the forecasting pipeline
    success = forecaster.run_pipeline()
    
    if success:
        logger.info("Sales forecasting completed successfully.")
    else:
        logger.error("Sales forecasting failed.")