
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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('sales_forecast')

# Suppress warnings
warnings.filterwarnings("ignore")
file_path="D:/AI Use Case App_GOOD/data_sets/RetailDS/Online Retail.csv"

data=pd.read_csv(file_path)

date_col=None
def identify_columns():
        """Automatically identify date and sales columns if not provided"""
        if data is None:
            logger.error("No data loaded. Call load_data() first.")
            return False
        
        # Identify date column if not provided
        if date_col is None:
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
            for col in data.columns:
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
            for col in data.columns:
                if col not in date_columns:
                    try:
                        if pd.to_datetime(data[col], errors='coerce').notna().sum() > data.shape[0] * 0.7:
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
                            date_col = col
                            break
                    if date_col:
                        break
                
                # If no priority match, use the first date column found
                if not date_col:
                    date_col = date_columns[0]
                
                logger.info(f"Date column identified: {date_col}")
            else:
                logger.error("Could not identify date column. Please specify it manually.")
                return False
        
        # Identify sales column if not provided
        if sales_col is None:
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
            for col in data.columns:
                col_lower = str(col).lower()
                # Check exact matches first
                if col_lower in sales_keywords:
                    if pd.api.types.is_numeric_dtype(data[col]):
                        sales_col = col
                        logger.info(f"Direct sales column found: {sales_col}")
                        break
                # Then check partial matches
                for keyword in sales_keywords:
                    if keyword in col_lower:
                        if pd.api.types.is_numeric_dtype(data[col]):
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
                
                for col in data.columns:
                    col_lower = str(col).lower()
                    
                    # Skip non-numeric columns
                    if not pd.api.types.is_numeric_dtype(data[col]):
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
                    for col in data.columns:
                        col_lower = str(col).lower()
                        
                        # Skip non-numeric columns
                        if not pd.api.types.is_numeric_dtype(data[col]):
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
                    data['adjusted_sales'] = data[original_sales_col] + data[discount_value_col]
                    sales_col = 'adjusted_sales'
                    logger.info(f"Created adjusted sales column: {sales_col}")
                
                elif discount_percentage_col:
                    logger.info(f"Discount percentage column found: {discount_percentage_col}")
                    logger.info("Adjusting sales figures based on discount percentages")
                    
                    # Check if discount is stored as percentage (e.g., 5 for 5%) or decimal (e.g., 0.05 for 5%)
                    max_discount = data[discount_percentage_col].max()
                    
                    # Adjust based on how discount is stored
                    if max_discount > 1:
                        # Stored as percentage (e.g., 5 for 5%)
                        data['adjusted_sales'] = data[sales_col] / (1 - data[discount_percentage_col]/100)
                    else:
                        # Stored as decimal (e.g., 0.05 for 5%)
                        data['adjusted_sales'] = data[sales_col] / (1 - data[discount_percentage_col])
                    
                    sales_col = 'adjusted_sales'
                    logger.info(f"Created adjusted sales column: {sales_col}")
                
                # Set the identified sales column
                sales_col = sales_col
                return True
            
            # If no direct sales column, try to calculate from components
            # Check for quantity, price, and discount columns
            quantity_col = None
            price_col = None
            discount_value_col = None
            discount_percentage_col = None
            
            # Find quantity column
            for col in data.columns:
                col_lower = str(col).lower()
                
                # Skip the identified date column
                if col == date_col:
                    continue
                
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(data[col]):
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
            for col in data.columns:
                col_lower = str(col).lower()
                
                # Skip the identified date column and quantity column
                if col == date_col or col == quantity_col:
                    continue
                
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(data[col]):
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
            for col in data.columns:
                col_lower = str(col).lower()
                
                # Skip columns already identified
                if col == date_col or col == quantity_col or col == price_col:
                    continue
                
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(data[col]):
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
                for col in data.columns:
                    col_lower = str(col).lower()
                    
                    # Skip columns already identified
                    if col == date_col or col == quantity_col or col == price_col:
                        continue
                    
                    # Check if column is numeric
                    if not pd.api.types.is_numeric_dtype(data[col]):
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
                data['calculated_sales'] = data[quantity_col] * data[price_col]
                
                # Apply discount if available
                if discount_value_col:
                    logger.info(f"Found discount value column: {discount_value_col}")
                    logger.info("Calculating total sales with discount value adjustment")
                    # We add back discount value to get pre-discount sales for accurate forecasting
                    data['calculated_total_sales'] = data['calculated_sales'] + data[discount_value_col]
                elif discount_percentage_col:
                    logger.info(f"Found discount percentage column: {discount_percentage_col}")
                    logger.info("Calculating total sales with discount percentage adjustment")
                    
                    # Check if discount is stored as percentage (e.g., 5 for 5%) or decimal (e.g., 0.05 for 5%)
                    max_discount = data[discount_percentage_col].max()
                    
                    # Adjust based on how discount is stored
                    if max_discount > 1:
                        # Stored as percentage (e.g., 5 for 5%)
                        data['calculated_total_sales'] = data['calculated_sales'] / (1 - data[discount_percentage_col]/100)
                    else:
                        # Stored as decimal (e.g., 0.05 for 5%)
                        data['calculated_total_sales'] = data['calculated_sales'] / (1 - data[discount_percentage_col])
                else:
                    logger.info("No discount columns found. Using base sales calculation.")
                    data['calculated_total_sales'] = data['calculated_sales']
                
                sales_col = 'calculated_total_sales'
                logger.info(f"Created sales column: {sales_col}")
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
            
            for col in data.columns:
                col_lower = str(col).lower()
                
                # Skip the identified date column
                if col == date_col:
                    continue
                
                # Check if column is numeric
                if not pd.api.types.is_numeric_dtype(data[col]):
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
    
   