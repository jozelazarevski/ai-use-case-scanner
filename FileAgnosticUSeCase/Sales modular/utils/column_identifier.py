# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 10:51:50 2025

@author: joze_
"""

"""
Product Sales Forecaster Example

This example demonstrates how to use the ProductSalesForecaster class
with the enhanced ColumnIdentifier functionality to automatically 
identify columns and generate sales forecasts from your data.
"""

import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
from .product_forecaster import ProductSalesForecaster

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler('forecaster.log')  # Log file
    ]
)

logger = logging.getLogger('forecaster_example')

def generate_sample_data(output_file="sample_sales_data.csv", rows=1000):
    """Generate a sample sales dataset for testing purposes"""
    logger.info(f"Generating sample sales data with {rows} rows...")
    
    import numpy as np
    from datetime import datetime, timedelta
    
    # Generate dates (past 3 years of monthly data)
    start_date = datetime.now() - timedelta(days=3*365)
    dates = []
    
    for i in range(rows):
        # Random date within past 3 years
        days_to_add = np.random.randint(0, 3*365)
        dates.append(start_date + timedelta(days=days_to_add))
    
    # Sort dates for realistic time series
    dates.sort()
    
    # Generate product IDs and names
    products = {
        "P001": "Premium Office Chair",
        "P002": "Standard Desk",
        "P003": "Ergonomic Keyboard",
        "P004": "Wireless Mouse",
        "P005": "Monitor Stand",
        "P006": "Desk Lamp",
        "P007": "Filing Cabinet",
        "P008": "Whiteboard",
        "P009": "Conference Table",
        "P010": "Projector Screen"
    }
    
    product_ids = list(products.keys())
    product_categories = ["Furniture", "Electronics", "Accessories", "Storage", "Office Equipment"]
    
    # Generate sample data
    data = {
        "OrderID": [f"ORD-{i+1000}" for i in range(rows)],
        "OrderYear": [d.year for d in dates],
        "OrderMonth": [d.month for d in dates],
        "OrderDay": [d.day for d in dates],
        "OrderDate": dates,
        "ProductID": np.random.choice(product_ids, size=rows),
        "CustomerID": [f"CUST-{np.random.randint(1, 50)}" for _ in range(rows)],
        "Region": np.random.choice(["North", "South", "East", "West"], size=rows),
        "Quantity": np.random.randint(1, 10, size=rows),
        "UnitPrice": np.random.uniform(10, 200, size=rows).round(2),
    }
    
    # Add product names and categories
    data["ProductName"] = [products[pid] for pid in data["ProductID"]]
    data["Category"] = [np.random.choice(product_categories) for _ in range(rows)]
    
    # Calculate total sales
    data["TotalSales"] = [data["Quantity"][i] * data["UnitPrice"][i] for i in range(rows)]
    
    # Add some seasonal patterns to make forecasting more interesting
    month_factors = {
        1: 0.8,   # January (post-holiday slump)
        2: 0.7,   # February (low)
        3: 0.9,   # March
        4: 1.0,   # April
        5: 1.1,   # May
        6: 1.2,   # June (summer bump)
        7: 1.3,   # July (peak summer)
        8: 1.2,   # August
        9: 1.1,   # September (back to school/office)
        10: 1.0,  # October
        11: 1.2,  # November (pre-holiday)
        12: 1.5   # December (holiday peak)
    }
    
    # Apply seasonal adjustments
    for i in range(rows):
        month = data["OrderMonth"][i]
        data["TotalSales"][i] *= month_factors[month]
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Sample data saved to {output_file}")
    return output_file

def run_basic_example(data_file):
    """Run basic example with auto-detection of columns"""
    logger.info("Running basic example with auto-detection...")
    
    # Create output directory
    output_dir = "forecast_results_basic"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create forecaster with default settings
    forecaster = ProductSalesForecaster(
        file_path=data_file,
        # Let the system auto-detect columns
        forecast_months=3,
        output_dir=output_dir
    )
    
    # Run the complete pipeline
    success = forecaster.run_pipeline()
    
    if success:
        logger.info(f"Basic forecasting completed successfully! Results saved to '{output_dir}'")
    else:
        logger.error("Basic forecasting failed. Check the logs for details.")

def run_advanced_example(data_file):
    """Run advanced example with specific settings"""
    logger.info("Running advanced example with specific settings...")
    
    # Create output directory
    output_dir = "forecast_results_advanced"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create forecaster with specific settings
    forecaster = ProductSalesForecaster(
        file_path=data_file,
        # Specify column names (optional)
        date_col="OrderDate",
        product_col="ProductID",
        sales_col="TotalSales",
        # Additional settings
        forecast_months=6,  # Forecast 6 months ahead
        min_data_points=12,  # Require at least 12 data points per product
        group_similar_products=True,  # Use product grouping for sparse data
        use_pooled_data=True,  # Allow pooled data for products with insufficient history
        output_dir=output_dir,
        export_format="excel"  # Export to Excel format
    )
    
    # Run selected steps of the pipeline
    logger.info("Running selective steps of the pipeline...")
    
    # Load the data
    success = forecaster.load_data()
    if not success:
        logger.error("Failed to load data")
        return
    
    # Identify columns (will use specified columns or auto-detect)
    success = forecaster.identify_columns()
    if not success:
        logger.error("Failed to identify columns")
        return
    
    # Perform exploratory data analysis
    success = forecaster.perform_eda()
    if not success:
        logger.error("Failed to perform EDA")
        return
    
    # Preprocess the data
    success = forecaster.preprocess_data()
    if not success:
        logger.error("Failed to preprocess data")
        return
    
    # Split the data into training and test sets
    success = forecaster.split_data(test_size=0.25)  # Use 25% for testing
    if not success:
        logger.error("Failed to split data")
        return
    
    # Build the forecasting models
    success = forecaster.build_models()
    if not success:
        logger.error("Failed to build models")
        return
    
    # Visualize the forecasting results
    success = forecaster.visualize_results()
    if not success:
        logger.error("Failed to visualize results")
        return
    
    # Export the forecasting results
    success = forecaster.export_results()
    if not success:
        logger.error("Failed to export results")
        return
    
    logger.info(f"Advanced forecasting completed successfully! Results saved to '{output_dir}'")

def run_component_date_example(data_file):
    """Run example with separate year, month, day columns"""
    logger.info("Running example with separate date components...")
    
    # Create output directory
    output_dir = "forecast_results_component_date"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create forecaster with default settings
    forecaster = ProductSalesForecaster(
        file_path=data_file,
        # Don't specify date_col - let it detect and combine the components
        product_col="ProductID",
        sales_col="TotalSales",
        forecast_months=3,
        output_dir=output_dir
    )
    
    # Run the complete pipeline
    success = forecaster.run_pipeline()
    
    if success:
        logger.info(f"Component date forecasting completed successfully! Results saved to '{output_dir}'")
    else:
        logger.error("Component date forecasting failed. Check the logs for details.")

def run_customized_analysis(data_file, product_subset=None):
    """Run customized analysis for specific products or scenarios"""
    logger.info("Running customized analysis...")
    
    # Create output directory
    output_dir = "forecast_results_customized"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data first
    df = pd.read_csv(data_file, parse_dates=["OrderDate"])
    
    # Option to filter for specific products
    if product_subset:
        logger.info(f"Filtering data for specific products: {product_subset}")
        df = df[df["ProductID"].isin(product_subset)]
    
    # Save the filtered data for processing
    filtered_file = os.path.join(output_dir, "filtered_data.csv")
    df.to_csv(filtered_file, index=False)
    
    # Create forecaster with the filtered data
    forecaster = ProductSalesForecaster(
        file_path=filtered_file,
        date_col="OrderDate",
        product_col="ProductID",
        sales_col="TotalSales",
        forecast_months=12,  # Longer forecast horizon
        output_dir=output_dir
    )
    
    # Run the pipeline
    success = forecaster.run_pipeline()
    
    if success:
        logger.info(f"Customized analysis completed successfully! Results saved to '{output_dir}'")
        
        # Additional custom visualization
        logger.info("Creating custom visualizations...")
        
        # Combine all future predictions
        all_predictions = pd.concat(list(forecaster.product_future_predictions.values()))
        
        # Create a monthly total forecast
        monthly_totals = all_predictions.groupby('date')[forecaster.sales_col].sum().reset_index()
        
        # Plot monthly totals
        plt.figure(figsize=(14, 7))
        plt.plot(monthly_totals['date'], monthly_totals[forecaster.sales_col], marker='o')
        plt.title('Total Monthly Sales Forecast')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'total_monthly_forecast.png'))
        plt.close()
        
        logger.info("Custom visualizations created.")
    else:
        logger.error("Customized analysis failed. Check the logs for details.")

if __name__ == "__main__":
    # Generate sample data if needed
    if not os.path.exists("sample_sales_data.csv"):
        data_file = generate_sample_data(rows=2000)
    else:
        data_file = "sample_sales_data.csv"
        logger.info(f"Using existing sample data file: {data_file}")
    
    # Run basic example with auto-detection
    run_basic_example(data_file)
    
    # Run advanced example with specific settings
    run_advanced_example(data_file)
    
    # Run example that demonstrates component date handling
    # This will use the OrderYear, OrderMonth, OrderDay columns
    run_component_date_example(data_file)
    
    # Run customized analysis for specific products
    run_customized_analysis(data_file, product_subset=["P001", "P003", "P008"])
    
    logger.info("All examples completed!")