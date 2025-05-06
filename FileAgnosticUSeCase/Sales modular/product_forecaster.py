"""
Product Sales Forecaster Module

This module contains the main forecasting functionality for predicting
product sales over time, including data preprocessing, model building, 
and forecast generation.
"""

import os
import re
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

from column_identifier import ColumnIdentifier
from eda_analyzer import EDAAnalyzer
from config import DEFAULT_CONFIG

logger = logging.getLogger('sales_forecaster.product_forecaster')

class ProductSalesForecaster:
    """Class for product-level sales forecasting operations"""
    
    def __init__(self, file_path, date_col=None, sales_col=None, product_col=None, 
                 forecast_months=DEFAULT_CONFIG['forecast_months'], 
                 min_data_points=DEFAULT_CONFIG['min_data_points'], 
                 group_similar_products=DEFAULT_CONFIG['group_similar_products'], 
                 use_pooled_data=DEFAULT_CONFIG['use_pooled_data'],
                 output_dir=DEFAULT_CONFIG['output_dir'],
                 export_format=DEFAULT_CONFIG['export_format']):
        """
        Initialize the forecaster
        
        Args:
            file_path (str): Path to the data file
            date_col (str, optional): Name of the date column
            sales_col (str, optional): Name of the sales column
            product_col (str, optional): Name of the product column
            forecast_months (int): Number of months to forecast
            min_data_points (int): Minimum number of data points required for forecasting
            group_similar_products (bool): Whether to group similar products for better forecasting
            use_pooled_data (bool): Whether to use pooled data for products with insufficient data
            output_dir (str): Directory to store output files
            export_format (str): Format for exporting results (csv, excel, json)
        """
        self.file_path = file_path
        self.date_col = date_col
        self.sales_col = sales_col
        self.product_col = product_col
        self.forecast_months = forecast_months
        self.min_data_points = min_data_points
        self.group_similar_products = group_similar_products
        self.use_pooled_data = use_pooled_data
        self.output_dir = output_dir
        self.export_format = export_format
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
        
        # Initialize data structures
        self.data = None
        self.cleaned_data = None
        self.train_data = {}
        self.test_data = {}
        self.product_models = {}
        self.product_predictions = {}
        self.product_future_predictions = {}
        self.product_groups = {}
        self.fallback_model = None
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
    
    def identify_columns(self):
        """Identify important columns in the data"""
        if self.data is None:
            logger.error("No data loaded. Call load_data() first.")
            return False
            
        try:
            # Create column identifier
            identifier = ColumnIdentifier(
                self.data, 
                date_col=self.date_col, 
                product_col=self.product_col, 
                sales_col=self.sales_col
            )
            
            # Identify all columns
            success = identifier.identify_all_columns()
            
            if success:
                # Update column names
                self.date_col = identifier.detected_columns['date']
                self.product_col = identifier.detected_columns['product']
                self.sales_col = identifier.detected_columns['sales']
                
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error identifying columns: {e}")
            return False
            
    def perform_eda(self):
        """Perform exploratory data analysis on the data"""
        if self.data is None or self.date_col is None or self.product_col is None or self.sales_col is None:
            logger.error("Data and columns must be available. Call load_data() and identify_columns() first.")
            return False
            
        try:
            # Create EDA analyzer
            analyzer = EDAAnalyzer(
                self.data,
                self.date_col,
                self.product_col,
                self.sales_col,
                output_dir=self.output_dir
            )
            
            # Run full EDA
            success = analyzer.run_full_eda()
            
            if success:
                # Store product groups if available for use in forecasting
                if 'product_groups' in analyzer.eda_results:
                    self.product_groups = analyzer.eda_results['product_groups']
                
                return True
            else:
                return False
        except Exception as e:
            logger.error(f"Error performing EDA: {e}")
            return False
    
    def preprocess_data(self):
        """Clean and transform data for forecasting with improved handling of sparse data"""
        if self.data is None or self.date_col is None or self.product_col is None or self.sales_col is None:
            logger.error("Data or required columns not set. Call load_data() and identify_columns() first.")
            return False
            
        logger.info("Preprocessing data...")
        
        try:
            # Make a copy to avoid modifying the original data
            data = self.data.copy()
            
            # Convert date column to datetime
            data[self.date_col] = pd.to_datetime(data[self.date_col], errors='coerce')
            
            # Drop rows with invalid dates
            data = data.dropna(subset=[self.date_col])
            
            # Ensure product column is treated as categorical
            data[self.product_col] = data[self.product_col].astype(str)
            
            # Ensure sales column is numeric
            data[self.sales_col] = pd.to_numeric(data[self.sales_col], errors='coerce')
            
            # Drop rows with missing sales values
            data = data.dropna(subset=[self.sales_col])
            
            # Sort by date
            data = data.sort_values(by=self.date_col)
            
            # Detect time frequency (daily, weekly, monthly, etc.)
            date_diffs = data[self.date_col].diff().dropna()
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
                # Default to monthly since we need to forecast 3 months
                self.time_frequency = 'M'
                logger.info("Defaulting to monthly frequency for 3-month forecast")
            
            # For product-level forecasting with 3-month horizon, convert to monthly if not already
            if self.time_frequency != 'M':
                logger.info(f"Converting data from {self.time_frequency} to monthly for 3-month forecast")
                self.time_frequency = 'M'
            
            # Get unique products
            products = data[self.product_col].unique()
            logger.info(f"Found {len(products)} unique products to preprocess")
            
            # Create a clean dataset for each product
            clean_data_by_product = {}
            skipped_products = []
            
            for product in products:
                # Filter data for this product
                product_data = data[data[self.product_col] == product].copy()
                
                # If very few data points for this product, flag it for potential grouping later
                if len(product_data) < self.min_data_points:
                    skipped_products.append(product)
                    continue
                
                # Create a dataframe for this product with just the needed columns
                product_df = product_data[[self.date_col, self.sales_col]].copy()
                
                # Aggregate by month (resampling)
                product_df = product_df.set_index(self.date_col)
                product_df = product_df.resample('M').sum()  # Monthly sum of sales
                
                # Reset index to keep date as a column
                product_df = product_df.reset_index()
                
                # Get the name of the date column after resampling
                date_col = product_df.columns[0]
                
                # Create time-based features
                product_df['year'] = product_df[date_col].dt.year
                product_df['month'] = product_df[date_col].dt.month
                product_df['quarter'] = product_df[date_col].dt.quarter
                
                # Add lag features (previous months)
                for lag in [1, 2, 3, 6, 12]:
                    if len(product_df) > lag:
                        product_df[f'lag_{lag}'] = product_df[self.sales_col].shift(lag)
                
                # Add rolling statistics
                for window in [3, 6, 12]:
                    if len(product_df) > window:
                        product_df[f'rolling_mean_{window}'] = product_df[self.sales_col].rolling(window=window).mean()
                        product_df[f'rolling_std_{window}'] = product_df[self.sales_col].rolling(window=window).std()
                
                # Fill missing values from lag and rolling features with appropriate values
                # For lag features, use the mean of available lags
                lag_cols = [col for col in product_df.columns if 'lag_' in col]
                if lag_cols:
                    for col in lag_cols:
                        product_df[col] = product_df[col].fillna(product_df[lag_cols].mean(axis=1))
                    
                # For rolling features, use the available sales data mean
                rolling_cols = [col for col in product_df.columns if 'rolling_' in col]
                if rolling_cols:
                    sales_mean = product_df[self.sales_col].mean()
                    sales_std = max(product_df[self.sales_col].std(), 1.0)  # Avoid zero std
                    
                    for col in rolling_cols:
                        if 'mean' in col:
                            product_df[col] = product_df[col].fillna(sales_mean)
                        elif 'std' in col:
                            product_df[col] = product_df[col].fillna(sales_std)
                
                # If we have enough data after feature creation, store it
                if len(product_df) >= self.min_data_points:
                    clean_data_by_product[product] = product_df
                else:
                    skipped_products.append(product)
                    logger.debug(f"Skipping product '{product}' - insufficient data after feature creation")
            
            # Check if we have any products with sufficient data
            if not clean_data_by_product:
                logger.error("No products with sufficient data for individual forecasting")
                return False
            
            # Handle products with insufficient data
            if skipped_products and self.use_pooled_data:
                logger.info(f"Handling {len(skipped_products)} products with insufficient individual data...")
                
                # Try using product groups if available
                if hasattr(self, 'product_groups') and self.product_groups:
                    # For each skipped product, try to find a similar product group
                    for product in skipped_products[:]:  # Use a copy to modify the original list
                        found_group = False
                        
                        # Check if the product is in any group
                        for group_id, products in self.product_groups.items():
                            if product in products:
                                # Find the product with most data in this group
                                group_products = [p for p in products if p in clean_data_by_product]
                                
                                if group_products:
                                    # Get the "template" product with most data points
                                    template_product = max(group_products, key=lambda p: len(clean_data_by_product[p]))
                                    template_data = clean_data_by_product[template_product].copy()
                                    
                                    # Get the sales data for the skipped product
                                    product_data = data[data[self.product_col] == product].copy()
                                    
                                    if len(product_data) > 0:
                                        # Calculate the average ratio of this product's sales to the template
                                        product_monthly = product_data.groupby(pd.Grouper(key=self.date_col, freq='M'))[self.sales_col].sum()
                                        
                                        # Get overlapping months
                                        template_monthly = template_data.set_index(date_col)[self.sales_col]
                                        template_monthly.index = template_monthly.index.to_period('M')
                                        product_monthly.index = product_monthly.index.to_period('M')
                                        
                                        common_months = set(template_monthly.index) & set(product_monthly.index)
                                        
                                        if common_months:
                                            # Calculate sales ratio for common months
                                            ratios = []
                                            for month in common_months:
                                                template_sales = template_monthly[month]
                                                product_sales = product_monthly[month]
                                                
                                                if template_sales > 0:
                                                    ratios.append(product_sales / template_sales)
                                            
                                            if ratios:
                                                # Use median ratio as it's more robust to outliers
                                                sales_ratio = np.median(ratios)
                                                
                                                # Create synthetic data for the skipped product
                                                synthetic_data = template_data.copy()
                                                synthetic_data[self.sales_col] = synthetic_data[self.sales_col] * sales_ratio
                                                
                                                # Store the synthetic data
                                                clean_data_by_product[product] = synthetic_data
                                                logger.info(f"Created synthetic data for product '{product}' based on similar product '{template_product}'")
                                                
                                                # Remove from skipped products
                                                skipped_products.remove(product)
                                                found_group = True
                                                break
                        
                        if not found_group:
                            logger.debug(f"No suitable group found for product '{product}'")
                
                # For remaining skipped products, create a simple average-based model
                if skipped_products:
                    logger.info(f"Creating pooled data model for {len(skipped_products)} products without sufficient data")
                    
                    # Create a pooled dataset
                    pooled_data = pd.DataFrame()
                    
                    for product in skipped_products:
                        product_data = data[data[self.product_col] == product].copy()
                        
                        if len(product_data) > 0:
                            # Aggregate monthly
                            monthly_data = product_data.groupby(pd.Grouper(key=self.date_col, freq='M'))[[self.sales_col]].sum()
                            monthly_data['product'] = product
                            
                            # Add to pooled data
                            if pooled_data.empty:
                                pooled_data = monthly_data.reset_index()
                            else:
                                pooled_data = pd.concat([pooled_data, monthly_data.reset_index()])
                    
                    if not pooled_data.empty:
                        # Calculate average monthly sales for each product
                        product_avg_sales = pooled_data.groupby('product')[self.sales_col].mean().to_dict()
                        
                        # Calculate monthly sales pattern
                        pooled_data['month'] = pooled_data[self.date_col].dt.month
                        monthly_pattern = pooled_data.groupby('month')[self.sales_col].mean()
                        monthly_pattern = monthly_pattern / monthly_pattern.mean()  # Normalize
                        
                        # Create synthetic data for each skipped product
                        for product in skipped_products:
                            if product in product_avg_sales:
                                avg_sales = product_avg_sales[product]
                                
                                # Create a complete monthly dataset
                                start_date = pooled_data[self.date_col].min().replace(day=1)
                                end_date = pooled_data[self.date_col].max().replace(day=1)
                                
                                # Create date range with at least self.min_data_points
                                if (end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1 < self.min_data_points:
                                    # Extend to ensure minimum data points
                                    needed_months = self.min_data_points - ((end_date.year - start_date.year) * 12 + end_date.month - start_date.month + 1)
                                    end_date = end_date + pd.DateOffset(months=needed_months)
                                
                                date_range = pd.date_range(start=start_date, end=end_date, freq='M')
                                
                                # Create synthetic data
                                synthetic_df = pd.DataFrame({
                                    'date': date_range,
                                    self.sales_col: [avg_sales * monthly_pattern.get(date.month, 1.0) for date in date_range]
                                })
                                
                                # Add time features
                                synthetic_df['year'] = synthetic_df['date'].dt.year
                                synthetic_df['month'] = synthetic_df['date'].dt.month
                                synthetic_df['quarter'] = synthetic_df['date'].dt.quarter
                                
                                # Add lag features
                                for lag in [1, 2, 3]:
                                    synthetic_df[f'lag_{lag}'] = synthetic_df[self.sales_col].shift(lag)
                                
                                # Add rolling features
                                for window in [3]:
                                    synthetic_df[f'rolling_mean_{window}'] = synthetic_df[self.sales_col].rolling(window=window).mean()
                                    synthetic_df[f'rolling_std_{window}'] = synthetic_df[self.sales_col].rolling(window=window).std()
                                
                                # Fill missing values
                                synthetic_df = synthetic_df.fillna(method='bfill').fillna(method='ffill')
                                
                                # Store the synthetic data
                                clean_data_by_product[product] = synthetic_df
                                logger.info(f"Created synthetic data for product '{product}' based on pooled average")
            
            # Final check on processed products
            processed_products = len(clean_data_by_product)
            skipped_products_count = len(products) - processed_products
            
            logger.info(f"Data preprocessing complete. {processed_products} products ready for forecasting.")
            logger.info(f"{skipped_products_count} products were skipped due to insufficient data.")
            
            # Store the cleaned data dictionary
            self.cleaned_data = clean_data_by_product
            
            # Show preview of cleaned data for the first product
            if processed_products > 0:
                first_product = next(iter(clean_data_by_product))
                logger.info(f"\nCleaned Data Sample for Product '{first_product}':")
                logger.info(clean_data_by_product[first_product].head().to_string())
            
            return True
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return False
    
    def check_seasonality(self, data):
        """Check for seasonality in the time series"""
        try:
            if len(data) >= 2 * 12:  # Need at least two cycles for seasonal decomposition
                # For monthly data, period is 12 (yearly seasonality)
                period = 12
                
                # Perform seasonal decomposition
                decomposition = seasonal_decompose(
                    data[self.sales_col], 
                    model='additive', 
                    period=period
                )
                
                # Calculate strength of seasonality
                seasonal_strength = 1 - (decomposition.resid.var() / decomposition.seasonal.var() if decomposition.seasonal.var() != 0 else 0)
                
                return seasonal_strength
            else:
                # For shorter time series, use a simpler method
                # Check month-to-month variation vs overall variation
                if 'month' in data.columns and len(data) >= 6:
                    monthly_means = data.groupby('month')[self.sales_col].mean()
                    overall_mean = data[self.sales_col].mean()
                    
                    # Calculate variance of monthly means
                    monthly_variance = monthly_means.var()
                    overall_variance = data[self.sales_col].var()
                    
                    if overall_variance > 0:
                        seasonal_strength = monthly_variance / overall_variance
                        return min(seasonal_strength, 0.8)  # Cap at 0.8 for this simple method
                
                return 0.1  # Default low seasonality for short time series
                
        except Exception as e:
            logger.debug(f"Could not check seasonality: {e}")
            return 0  # Default to no seasonality
    
    def split_data(self, test_size=0.2):
        """Split data into training and testing sets for each product"""
        if not self.cleaned_data:
            logger.error("No cleaned data available. Call preprocess_data() first.")
            return False
            
        try:
            # Dictionary to store training and testing data for each product
            train_data_by_product = {}
            test_data_by_product = {}
            
            for product, product_df in self.cleaned_data.items():
                # For products with limited data, use a smaller test set
                if len(product_df) <= 2 * self.min_data_points:
                    # For very limited data, use just 1 data point for testing
                    min_test_size = 1
                    # Ensure we have at least min_data_points for training
                    split_idx = max(self.min_data_points, len(product_df) - min_test_size)
                else:
                    # For products with more data, use standard split
                    min_test_size = min(3, max(1, int(len(product_df) * test_size)))
                    split_idx = max(self.min_data_points, len(product_df) - min_test_size)
                
                # Split the data
                train_data_by_product[product] = product_df.iloc[:split_idx].copy()
                test_data_by_product[product] = product_df.iloc[split_idx:].copy()
                
                # Only log if we have a decent amount of data to avoid log spam
                if len(product_df) >= 2 * self.min_data_points:
                    logger.info(f"Data split for product '{product}': Training set size: {len(train_data_by_product[product])}, Testing set size: {len(test_data_by_product[product])}")
            
            # Store the splits
            self.train_data = train_data_by_product
            self.test_data = test_data_by_product
            
            logger.info(f"Data split complete for {len(train_data_by_product)} products")
            
            return True
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return False
    
    def _build_simple_model(self, product, y_train, y_test, forecast_periods):
        """Build a simple moving average model for products with limited data"""
        try:
            # Use a simple moving average or weighted average for predictions
            if len(y_train) >= 3:
                # Use a weighted average with more weight on recent values
                weights = np.arange(1, len(y_train) + 1)
                ma_forecast = np.average(y_train, weights=weights)
            else:
                # Simple mean for very short series
                ma_forecast = y_train.mean()
            
            # Create predictions for test set
            predictions = pd.Series([ma_forecast] * len(y_test) if len(y_test) > 0 else [])
            
            # Get the date for forecast
            date_col = self.train_data[product].columns[0]
            last_date = self.test_data[product].iloc[-1 if len(self.test_data[product]) > 0 else -1][date_col]
            
            # Generate future dates for the forecast
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='M')
            
            # Create future predictions dataframe
            future_df = pd.DataFrame({
                'date': future_dates,
                'product': product,
                self.sales_col: [ma_forecast] * forecast_periods
            })
            
            # Return the "model" (just the forecast value), predictions, and future predictions
            return ma_forecast, predictions, future_df
            
        except Exception as e:
            logger.error(f"Error building simple model for product '{product}': {e}")
            
            # Fall back to using the mean if there are any issues
            mean_val = y_train.mean() if not y_train.empty else 0
            predictions = pd.Series([mean_val] * len(y_test) if len(y_test) > 0 else [])
            
            # Generate future dates
            date_col = self.train_data[product].columns[0]
            last_date = self.test_data[product].iloc[-1 if len(self.test_data[product]) > 0 else -1][date_col]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_periods, freq='M')
            
            # Create future predictions dataframe
            future_df = pd.DataFrame({
                'date': future_dates,
                'product': product,
                self.sales_col: [mean_val] * forecast_periods
            })
            
            return mean_val, predictions, future_df
    
    def _build_fallback_model(self, product, y_train, y_test, fallback_model):
        """Build a model based on the overall pattern across all products"""
        try:
            # Extract the monthly pattern and overall mean from the fallback model
            monthly_pattern = fallback_model['monthly_pattern']
            overall_mean = fallback_model['overall_mean']
            
            # Calculate the average sales for this product
            product_mean = y_train.mean()
            
            # Calculate the scaling factor (ratio of this product's mean to overall mean)
            scaling_factor = product_mean / overall_mean if overall_mean > 0 else 1.0
            
            # Generate test predictions
            predictions = []
            if len(y_test) > 0:
                date_col = self.test_data[product].columns[0]
                for _, row in self.test_data[product].iterrows():
                    month = row[date_col].month
                    month_factor = monthly_pattern.get(month, 1.0)
                    predictions.append(product_mean * month_factor)
                
                predictions = pd.Series(predictions)
            
            # Generate future predictions
            date_col = self.train_data[product].columns[0]
            last_date = self.test_data[product].iloc[-1 if len(self.test_data[product]) > 0 else -1][date_col]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=self.forecast_months, freq='M')
            
            future_values = []
            for date in future_dates:
                month = date.month
                month_factor = monthly_pattern.get(month, 1.0)
                future_values.append(product_mean * month_factor)
            
            # Create future predictions dataframe
            future_df = pd.DataFrame({
                'date': future_dates,
                'product': product,
                self.sales_col: future_values
            })
            
            # Return the model, predictions, and future predictions
            return {'type': 'fallback', 'product_mean': product_mean, 'monthly_pattern': monthly_pattern}, predictions, future_df
            
        except Exception as e:
            logger.error(f"Error building fallback model for product '{product}': {e}")
            
            # If all else fails, use a simple mean
            mean_val = y_train.mean() if not y_train.empty else 0
            predictions = pd.Series([mean_val] * len(y_test) if len(y_test) > 0 else [])
            
            # Generate future dates
            date_col = self.train_data[product].columns[0]
            last_date = self.test_data[product].iloc[-1 if len(self.test_data[product]) > 0 else -1][date_col]
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=self.forecast_months, freq='M')
            
            # Create future predictions dataframe
            future_df = pd.DataFrame({
                'date': future_dates,
                'product': product,
                self.sales_col: [mean_val] * self.forecast_months
            })
            
            return {'type': 'mean', 'value': mean_val}, predictions, future_df
    
    def _build_sarima_model(self, product, y_train, y_test, seasonality_strength):
        """Build and train a SARIMA model for a product"""
        try:
            # Determine seasonality period (for monthly data, it's 12)
            m = 12  # Annual seasonality
            
            # Use auto_arima to find the best parameters
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
            
            logger.debug(f"Best SARIMA model for product '{product}': ({p},{d},{q})x({P},{D},{Q},{m})")
            
            # Fit the SARIMA model with the best parameters
            model = SARIMAX(
                y_train,
                order=(p, d, q),
                seasonal_order=(P, D, Q, m),
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            
            results = model.fit(disp=False)
            
            # Generate predictions for the test set
            predictions = results.forecast(steps=len(y_test))
            
            # Generate future predictions (3 months ahead)
            future_predictions = results.forecast(steps=self.forecast_months)
            
            # Create a DataFrame for future predictions
            last_date = self.test_data[product].iloc[-1 if len(self.test_data[product]) > 0 else -1][self.test_data[product].columns[0]]
            
            # Generate future dates (3 months)
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=self.forecast_months, freq='M')
            
            future_df = pd.DataFrame({
                'date': future_dates,
                'product': product,
                self.sales_col: future_predictions
            })
            
            return results, predictions, future_df
            
        except Exception as e:
            logger.error(f"Error building SARIMA model for product '{product}': {e}")
            
            # Fall back to a simpler model
            logger.info(f"Falling back to simple moving average model for product '{product}'")
            
            # Use a simple moving average for predictions
            ma_window = min(3, len(y_train))
            ma_forecast = y_train.rolling(window=ma_window).mean().iloc[-1]
            
            predictions = pd.Series([ma_forecast] * len(y_test))
            
            # Future predictions
            future_dates = pd.date_range(
                start=self.test_data[product].iloc[-1 if len(self.test_data[product]) > 0 else -1][self.test_data[product].columns[0]] + pd.Timedelta(days=1),
                periods=self.forecast_months,
                freq='M'
            )
            
            future_df = pd.DataFrame({
                'date': future_dates,
                'product': product,
                self.sales_col: [ma_forecast] * self.forecast_months
            })
            
            # Return a dummy model (use None as a marker for fallback model)
            return None, predictions, future_df
    
    def _build_xgboost_model(self, product, train_data, test_data):
        """Build and train an XGBoost model for a product"""
        try:
            # Prepare features and target
            feature_cols = [col for col in train_data.columns if col not in [train_data.columns[0], self.sales_col]]
            
            X_train = train_data[feature_cols]
            y_train = train_data[self.sales_col]
            
            # Define XGBoost parameters
            params = {
                'objective': 'reg:squarederror',
                'learning_rate': 0.1,
                'max_depth': 4,  # Reduced from 5 to avoid overfitting
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'n_estimators': 100,
                'silent': 1
            }
            
            # Train the XGBoost model
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
            
            # Predictions for test set
            if len(test_data) > 0:
                X_test = test_data[feature_cols]
                predictions = model.predict(X_test)
            else:
                predictions = np.array([])
            
            # Generate future predictions (3 months ahead)
            last_date = test_data.iloc[-1 if len(test_data) > 0 else -1][test_data.columns[0]]
            
            # Generate future dates
            future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=self.forecast_months, freq='M')
            
            # Create future features
            future_features = pd.DataFrame()
            future_features[train_data.columns[0]] = future_dates
            
            # Generate time-based features
            future_features['year'] = future_features[train_data.columns[0]].dt.year
            future_features['month'] = future_features[train_data.columns[0]].dt.month
            future_features['quarter'] = future_features[train_data.columns[0]].dt.quarter
            
            # For lag and rolling features, use the last known values and predictions
            combined_data = pd.concat([train_data, test_data, pd.DataFrame({
                train_data.columns[0]: future_dates,
                self.sales_col: [None] * self.forecast_months
            })]).reset_index(drop=True)
            
            # Fill in lag features
            for lag in [1, 2, 3, 6, 12]:
                if f'lag_{lag}' in feature_cols:
                    combined_data[f'lag_{lag}'] = combined_data[self.sales_col].shift(lag)
            
            # Fill in rolling features
            for window in [3, 6, 12]:
                if f'rolling_mean_{window}' in feature_cols:
                    combined_data[f'rolling_mean_{window}'] = combined_data[self.sales_col].rolling(window=window).mean()
                if f'rolling_std_{window}' in feature_cols:
                    combined_data[f'rolling_std_{window}'] = combined_data[self.sales_col].rolling(window=window).std()
            
            # Get the future part with features
            future_rows = combined_data.tail(self.forecast_months)
            
            # Fill NaN values with the mean of the training data
            for col in feature_cols:
                if col in future_rows.columns:
                    future_rows[col] = future_rows[col].fillna(train_data[col].mean())
            
            # Make predictions
            future_X = future_rows[feature_cols]
            future_predictions = model.predict(future_X)
            
            # Create a DataFrame for future predictions
            future_df = pd.DataFrame({
                'date': future_dates,
                'product': product,
                self.sales_col: future_predictions
            })
            
            return model, predictions, future_df
            
        except Exception as e:
            logger.error(f"Error building XGBoost model for product '{product}': {e}")
            
            # Fall back to a simple model
            logger.info(f"Falling back to simple moving average model for product '{product}'")
            
            # Use a simple moving average for predictions
            ma_window = min(3, len(train_data))
            ma_forecast = train_data[self.sales_col].rolling(window=ma_window).mean().iloc[-1]
            
            predictions = pd.Series([ma_forecast] * len(test_data) if len(test_data) > 0 else [])
            
            # Future predictions
            future_dates = pd.date_range(
                start=test_data.iloc[-1 if len(test_data) > 0 else -1][test_data.columns[0]] + pd.Timedelta(days=1),
                periods=self.forecast_months,
                freq='M'
            )
            
            future_df = pd.DataFrame({
                'date': future_dates,
                'product': product,
                self.sales_col: [ma_forecast] * self.forecast_months
            })
            
            # Return a dummy model (use None as a marker for fallback model)
            return None, predictions, future_df
    
    def build_models(self):
        """Build and train forecasting models for each product"""
        if not self.train_data or not self.test_data:
            logger.error("Training and testing data not available. Call split_data() first.")
            return False
            
        try:
            # Dictionary to store models and predictions for each product
            self.product_models = {}
            self.product_predictions = {}
            self.product_future_predictions = {}
            
            # Track overall performance metrics
            overall_mae = []
            overall_rmse = []
            overall_r2 = []
            model_types_used = {"SARIMA": 0, "XGBoost": 0, "SimpleMA": 0, "PooledAvg": 0}
            
            # First pass - build a baseline/fallback model using pooled data
            logger.info("Building fallback model using pooled data...")
            all_train_data = pd.concat([df[[df.columns[0], self.sales_col]] for df in self.train_data.values()])
            
            if not all_train_data.empty:
                # Group by date and calculate total sales
                all_train_data = all_train_data.groupby(all_train_data.columns[0])[self.sales_col].sum().reset_index()
                all_train_data.columns = ['date', self.sales_col]
                
                # Extract monthly pattern
                all_train_data['month'] = all_train_data['date'].dt.month
                monthly_pattern = all_train_data.groupby('month')[self.sales_col].mean()
                overall_mean = all_train_data[self.sales_col].mean()
                monthly_pattern = monthly_pattern / overall_mean  # Normalize
                
                # Store in a simple model
                self.fallback_model = {
                    'monthly_pattern': monthly_pattern.to_dict(),
                    'overall_mean': overall_mean
                }
                
                logger.info("Fallback model built successfully")
            
            # Second pass - build product-specific models
            products_total = len(self.train_data)
            products_processed = 0
            
            for product in list(self.train_data.keys()):  # Use list to allow modification during iteration
                products_processed += 1
                # Log progress every 10 products or for the last product
                if products_processed % 10 == 0 or products_processed == products_total:
                    logger.info(f"Building models: {products_processed}/{products_total} products processed")
                
                # Get training and testing data for this product
                y_train = self.train_data[product][self.sales_col]
                y_test = self.test_data[product][self.sales_col]
                
                # Skip detailed logging for products with limited data to reduce log spam
                verbose_logging = len(y_train) >= 2 * self.min_data_points
                
                # Choose model type based on data characteristics and length
                if len(y_train) < 2 * self.min_data_points:
                    # For products with limited data, use a simple moving average model
                    if verbose_logging:
                        logger.info(f"Using Simple Moving Average model for product '{product}' (limited data)")
                    
                    # Use a simple moving average for predictions
                    model, predictions, future_predictions = self._build_simple_model(
                        product, y_train, y_test, self.forecast_months
                    )
                    model_types_used["SimpleMA"] += 1
                else:
                    # For products with more data, use more sophisticated models
                    # Check for seasonality in the training data
                    seasonality_strength = self.check_seasonality(self.train_data[product])
                    
                    # If strong seasonality, use SARIMA, otherwise use XGBoost
                    if seasonality_strength > 0.3 and len(y_train) >= 24:  # Need enough data for SARIMA with seasonality
                        if verbose_logging:
                            logger.info(f"Using SARIMA model for product '{product}' (seasonality strength: {seasonality_strength:.2f})")
                        
                        # Build SARIMA model
                        model, predictions, future_predictions = self._build_sarima_model(
                            product, y_train, y_test, seasonality_strength
                        )
                        model_types_used["SARIMA"] += 1
                    else:
                        if verbose_logging:
                            logger.info(f"Using XGBoost model for product '{product}'")
                        
                        # Build XGBoost model
                        model, predictions, future_predictions = self._build_xgboost_model(
                            product, self.train_data[product], self.test_data[product]
                        )
                        model_types_used["XGBoost"] += 1
                
                # Store the model and predictions
                self.product_models[product] = model
                self.product_predictions[product] = predictions
                self.product_future_predictions[product] = future_predictions
                
                # Evaluate the model if test data is available
                if len(y_test) > 0:
                    mae = mean_absolute_error(y_test, predictions)
                    rmse = np.sqrt(mean_squared_error(y_test, predictions))
                    r2 = r2_score(y_test, predictions) if len(y_test) > 1 else 0
                    
                    if verbose_logging:
                        logger.info(f"Model evaluation for product '{product}': MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")
                    
                    # Add to overall metrics
                    overall_mae.append(mae)
                    overall_rmse.append(rmse)
                    overall_r2.append(r2)
                    
                    # If the model performs badly, try a fallback approach
                    if (r2 < 0 or mae > 2 * y_train.mean()) and self.fallback_model:
                        if verbose_logging:
                            logger.info(f"Poor model performance for product '{product}', switching to fallback model")
                        
                        # Use the fallback model instead
                        model, predictions, future_predictions = self._build_fallback_model(
                            product, y_train, y_test, self.fallback_model
                        )
                        
                        self.product_models[product] = model
                        self.product_predictions[product] = predictions
                        self.product_future_predictions[product] = future_predictions
                        model_types_used["PooledAvg"] += 1
                        
                        # Recalculate metrics
                        if len(y_test) > 0:
                            mae = mean_absolute_error(y_test, predictions)
                            rmse = np.sqrt(mean_squared_error(y_test, predictions))
                            r2 = r2_score(y_test, predictions) if len(y_test) > 1 else 0
                            
                            if verbose_logging:
                                logger.info(f"Fallback model evaluation for product '{product}': MAE={mae:.2f}, RMSE={rmse:.2f}, R²={r2:.2f}")
                            
                            # Update metrics
                            overall_mae[-1] = mae
                            overall_rmse[-1] = rmse
                            overall_r2[-1] = r2
            
            # Report overall performance
            if overall_mae:
                logger.info("\nOverall model performance metrics:")
                logger.info(f"Average MAE: {np.mean(overall_mae):.2f}")
                logger.info(f"Average RMSE: {np.mean(overall_rmse):.2f}")
                logger.info(f"Average R²: {np.mean(overall_r2):.2f}")
                
                logger.info("\nModel types used:")
                for model_type, count in model_types_used.items():
                    if count > 0:
                        logger.info(f"  {model_type}: {count} products ({count/products_total:.1%})")
            
            return True
        except Exception as e:
            logger.error(f"Error building models: {e}")
            return False
    
    def visualize_results(self):
        """Visualize the forecasting results for each product"""
        if not self.product_future_predictions:
            logger.error("Predictions not available. Call build_models() first.")
            return False
            
        try:
            # Create output directory if it doesn't exist
            figures_dir = os.path.join(self.output_dir, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            
            # Set up the plot style
            sns.set(style="whitegrid")
            
            # Create aggregated future predictions dataframe
            all_future_predictions = pd.concat(list(self.product_future_predictions.values()))
            
            # 1. Generate individual product forecasts
            products_to_visualize = min(20, len(self.product_models))  # Limit to 20 to avoid too many files
            
            # Select products to visualize (prioritize top selling products)
            all_products = list(self.product_models.keys())
            if products_to_visualize < len(all_products):
                # Calculate total sales for each product
                product_totals = {}
                for product in all_products:
                    if product in self.cleaned_data:
                        product_totals[product] = self.cleaned_data[product][self.sales_col].sum()
                
                # Select top products by sales
                selected_products = sorted(product_totals.items(), key=lambda x: x[1], reverse=True)
                selected_products = [p[0] for p in selected_products[:products_to_visualize]]
            else:
                selected_products = all_products
            
            logger.info(f"Generating visualizations for {len(selected_products)} products")
            
            for product in selected_products:
                plt.figure(figsize=(12, 6))
                
                # Get historical data
                historical_data = pd.concat([self.train_data[product], self.test_data[product]])
                historical_dates = historical_data[historical_data.columns[0]]
                historical_sales = historical_data[self.sales_col]
                
                # Get future predictions
                future_dates = self.product_future_predictions[product]['date']
                future_sales = self.product_future_predictions[product][self.sales_col]
                
                # Plot historical data
                plt.plot(
                    historical_dates, 
                    historical_sales,
                    label='Historical Sales',
                    color='blue'
                )
                
                # Plot test predictions if available
                if product in self.product_predictions and len(self.product_predictions[product]) > 0:
                    test_dates = self.test_data[product][self.test_data[product].columns[0]]
                    plt.plot(
                        test_dates,
                        self.product_predictions[product],
                        label='Test Predictions',
                        color='green',
                        linestyle='--'
                    )
                
                # Plot future predictions
                plt.plot(
                    future_dates,
                    future_sales,
                    label=f'{self.forecast_months}-Month Forecast',
                    color='red',
                    linestyle='-.'
                )
                
                # Add labels and title
                plt.xlabel('Date')
                plt.ylabel(f'{self.sales_col}')
                plt.title(f'Sales Forecast for Product: {product}')
                plt.legend()
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save the plot
                safe_product_name = re.sub(r'[^\w\-_\. ]', '_', str(product))
                output_file = os.path.join(figures_dir, f'forecast_{safe_product_name}.png')
                plt.savefig(output_file)
                
                plt.close()
            
            # 2. Generate overall forecast summary (top products)
            # Identify top products by total historical sales
            product_totals = {}
            for product in self.cleaned_data.keys():
                product_totals[product] = self.cleaned_data[product][self.sales_col].sum()
            
            # Get top 5 products (or all if less than 5)
            top_products = sorted(product_totals.items(), key=lambda x: x[1], reverse=True)
            top_products = [p[0] for p in top_products[:min(5, len(product_totals))]]
            
            plt.figure(figsize=(14, 8))
            
            # Plot future predictions for top products
            for product in top_products:
                future_dates = self.product_future_predictions[product]['date']
                future_sales = self.product_future_predictions[product][self.sales_col]
                
                plt.plot(
                    future_dates,
                    future_sales,
                    label=f'Product: {product}',
                    marker='o'
                )
            
            # Add labels and title
            plt.xlabel('Month')
            plt.ylabel(f'{self.sales_col}')
            plt.title(f'{self.forecast_months}-Month Sales Forecast for Top Products')
            plt.legend()
            plt.xticks(rotation=45)
            plt.grid(True)
            plt.tight_layout()
            
            # Save the plot
            output_file = os.path.join(figures_dir, f'top_products_forecast.png')
            plt.savefig(output_file)
            plt.close()
            
            logger.info(f"Top products forecast visualization saved as {output_file}")
            
            # 3. Generate a combined forecast table
            # Create a pivot table with products as rows and months as columns
            pivot_data = all_future_predictions.pivot(index='product', columns='date', values=self.sales_col)
            
            # Format the column headers as month-year
            pivot_data.columns = pivot_data.columns.strftime('%b %Y')
            
            # Reset index to make 'product' a regular column
            pivot_data = pivot_data.reset_index()
            
            # Save the pivot table
            pivot_file = os.path.join(self.output_dir, f"product_forecasts.{self.export_format}")
            
            if self.export_format == 'csv':
                pivot_data.to_csv(pivot_file, index=False)
            elif self.export_format == 'excel':
                pivot_data.to_excel(pivot_file, index=False)
            elif self.export_format == 'json':
                pivot_data.to_json(pivot_file, orient='records')
                
            logger.info(f"Product forecast summary saved as {pivot_file}")
            
            return True
        except Exception as e:
            logger.error(f"Error visualizing results: {e}")
            return False
    
    def export_results(self):
        """Export forecasting results to files"""
        if not self.product_future_predictions:
            logger.error("Predictions not available. Call build_models() first.")
            return False
            
        try:
            # 1. Export individual product forecasts
            for product, future_df in self.product_future_predictions.items():
                # Format the date column
                export_df = future_df.copy()
                export_df['date'] = export_df['date'].dt.strftime('%Y-%m-%d')
                
                # Determine output file name
                safe_product_name = re.sub(r'[^\w\-_\. ]', '_', str(product))
                output_file = os.path.join(self.output_dir, f'{safe_product_name}_forecast.{self.export_format}')
                
                # Export based on format
                if self.export_format == 'csv':
                    export_df.to_csv(output_file, index=False)
                elif self.export_format == 'excel':
                    export_df.to_excel(output_file, index=False)
                elif self.export_format == 'json':
                    export_df.to_json(output_file, orient='records')
                else:
                    logger.error(f"Unsupported export format: {self.export_format}")
                    return False
                    
                logger.debug(f"Forecast for product '{product}' exported to {output_file}")
            
            # 2. Export combined forecast summary
            # Combine all product forecasts
            all_forecasts = pd.concat(list(self.product_future_predictions.values()))
            
            # Create a pivot table with products as rows and months as columns
            pivot_forecasts = all_forecasts.pivot(index='product', columns='date', values=self.sales_col)
            
            # Format the column headers as month-year
            pivot_forecasts.columns = pivot_forecasts.columns.strftime('%b %Y')
            
            # Add a total column
            pivot_forecasts['Total Forecast'] = pivot_forecasts.sum(axis=1)
            
            # Sort by total forecast (descending)
            pivot_forecasts = pivot_forecasts.sort_values('Total Forecast', ascending=False)
            
            # Reset index to make 'product' a regular column
            pivot_forecasts = pivot_forecasts.reset_index()
            
            # Export the pivot table
            summary_file = os.path.join(self.output_dir, f'product_forecast_summary.{self.export_format}')
            
            if self.export_format == 'csv':
                pivot_forecasts.to_csv(summary_file, index=False)
            elif self.export_format == 'excel':
                pivot_forecasts.to_excel(summary_file, index=False)
            elif self.export_format == 'json':
                pivot_forecasts.to_json(summary_file, orient='records')
                
            logger.info(f"Combined forecast summary exported to {summary_file}")
            
            # 3. Export model performance metrics
            if hasattr(self, 'product_models') and self.product_models:
                # Prepare performance metrics
                metrics = []
                
                for product in self.product_models.keys():
                    if product in self.product_predictions and len(self.product_predictions[product]) > 0:
                        y_true = self.test_data[product][self.sales_col]
                        y_pred = self.product_predictions[product]
                        
                        if len(y_true) > 0:
                            mae = mean_absolute_error(y_true, y_pred)
                            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                            r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else 0
                            
                            metrics.append({
                                'product': product,
                                'MAE': mae,
                                'RMSE': rmse,
                                'R2': r2,
                                'data_points': len(y_true)
                            })
                
                if metrics:
                    metrics_df = pd.DataFrame(metrics)
                    
                    # Export the metrics
                    metrics_file = os.path.join(self.output_dir, f'model_performance_metrics.{self.export_format}')
                    
                    if self.export_format == 'csv':
                        metrics_df.to_csv(metrics_file, index=False)
                    elif self.export_format == 'excel':
                        metrics_df.to_excel(metrics_file, index=False)
                    elif self.export_format == 'json':
                        metrics_df.to_json(metrics_file, orient='records')
                        
                    logger.info(f"Model performance metrics exported to {metrics_file}")
            
            return True
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return False
    
    def run_pipeline(self):
        """Run the complete forecasting pipeline"""
        steps = [
            self.load_data,
            self.identify_columns,
            self.perform_eda,      # Added EDA step
            self.preprocess_data,
            self.split_data,
            self.build_models,
            self.visualize_results,
            self.export_results
        ]
        
        for step in steps:
            logger.info(f"Running pipeline step: {step.__name__}")
            success = step()
            if not success:
                logger.error(f"Pipeline failed at step: {step.__name__}")
                return False
        
        logger.info("Forecasting pipeline completed successfully")
        return True