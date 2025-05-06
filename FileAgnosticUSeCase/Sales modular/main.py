#!/usr/bin/env python3
"""
Product Sales Forecaster - Main Entry Point

This script serves as the main entry point for the product sales forecasting system.
It processes command line arguments and orchestrates the forecasting process.
"""

import argparse
import os
import sys
import logging
from datetime import datetime

from config import DEFAULT_CONFIG
from product_forecaster import ProductSalesForecaster
from logger import setup_logger

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Product-Level Sales Forecasting System")
    
    parser.add_argument('--file', required=True, help='Path to the data file')
    parser.add_argument('--date_column', help='Name of the date column')
    parser.add_argument('--sales_column', help='Name of the sales column')
    parser.add_argument('--product_column', help='Name of the product column')
    parser.add_argument('--forecast_months', type=int, default=DEFAULT_CONFIG['forecast_months'], 
                        help=f'Number of months to forecast (default: {DEFAULT_CONFIG["forecast_months"]})')
    parser.add_argument('--min_data_points', type=int, default=DEFAULT_CONFIG['min_data_points'], 
                        help=f'Minimum number of data points required (default: {DEFAULT_CONFIG["min_data_points"]})')
    parser.add_argument('--group_similar', action='store_true', 
                        help='Group similar products for better forecasting')
    parser.add_argument('--use_pooled_data', action='store_true', 
                        help='Use pooled data for products with insufficient data')
    parser.add_argument('--export_format', default=DEFAULT_CONFIG['export_format'], 
                        choices=['csv', 'excel', 'json'], 
                        help=f'Export format for results (default: {DEFAULT_CONFIG["export_format"]})')
    parser.add_argument('--skip_eda', action='store_true', 
                        help='Skip the EDA step')
    parser.add_argument('--output_dir', default=DEFAULT_CONFIG['output_dir'], 
                        help=f'Directory to store output files (default: {DEFAULT_CONFIG["output_dir"]})')
    parser.add_argument('--log_level', default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                        help='Set the logging level (default: INFO)')
    
    return parser.parse_args()

def create_output_directories(output_dir):
    """Create necessary output directories"""
    try:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "eda"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "figures"), exist_ok=True)
        return True
    except Exception as e:
        print(f"Error creating output directories: {e}")
        return False

def main():
    """Main entry point for the product sales forecasting system"""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_file = f"logs/forecast_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logger('sales_forecaster', args.log_level, log_file)
    
    logger.info("Starting product sales forecasting process")
    
    # Create output directories
    if not create_output_directories(args.output_dir):
        logger.error(f"Failed to create output directories in {args.output_dir}")
        return 1
    
    try:
        # Create the forecaster with specified parameters
        forecaster = ProductSalesForecaster(
            file_path=args.file,
            date_col=args.date_column,
            sales_col=args.sales_column,
            product_col=args.product_column,
            forecast_months=args.forecast_months,
            min_data_points=args.min_data_points,
            group_similar_products=args.group_similar,
            use_pooled_data=args.use_pooled_data,
            output_dir=args.output_dir,
            export_format=args.export_format
        )
        
        # Run custom pipeline if skipping EDA
        if args.skip_eda:
            logger.info("Running pipeline with EDA step skipped")
            steps = [
                forecaster.load_data,
                forecaster.identify_columns,
                forecaster.preprocess_data,
                forecaster.split_data,
                forecaster.build_models,
                forecaster.visualize_results,
                forecaster.export_results
            ]
            
            success = True
            for step in steps:
                logger.info(f"Running pipeline step: {step.__name__}")
                step_success = step()
                if not step_success:
                    logger.error(f"Pipeline failed at step: {step.__name__}")
                    success = False
                    break
        else:
            # Run the full pipeline
            logger.info("Running full pipeline including EDA")
            success = forecaster.run_pipeline()
        
        if success:
            logger.info("Product-level sales forecasting completed successfully")
            
            # Print a summary of the forecast
            products_count = len(forecaster.product_future_predictions)
            logger.info(f"\nForecast Summary:")
            logger.info(f"Total products forecasted: {products_count}")
            
            # Get top 5 products by forecast sales
            if products_count > 0:
                total_forecast_by_product = {}
                for product, future_df in forecaster.product_future_predictions.items():
                    total_forecast_by_product[product] = future_df[forecaster.sales_col].sum()
                
                top_products = sorted(total_forecast_by_product.items(), key=lambda x: x[1], reverse=True)[:min(5, products_count)]
                
                logger.info("\nTop 5 products by forecasted sales for next 3 months:")
                for i, (product, forecast) in enumerate(top_products):
                    logger.info(f"{i+1}. Product: {product}, Forecast: {forecast:.2f}")
                    
                logger.info("\nOutput files location:")
                logger.info(f"- EDA Visualizations: {args.output_dir}/eda/")
                logger.info(f"- EDA Summary Report: {args.output_dir}/eda/eda_summary.txt")
                logger.info(f"- Forecast Visualizations: {args.output_dir}/figures/")
                logger.info(f"- Forecast Data: {args.output_dir}/product_forecast_summary.{args.export_format}")
                logger.info(f"- Log file: {log_file}")
            
            return 0  # Success
        else:
            logger.error("Product-level sales forecasting failed")
            return 1  # Failure
            
    except Exception as e:
        logger.exception(f"An error occurred during forecasting: {str(e)}")
        return 1  # Failure

if __name__ == "__main__":
    sys.exit(main())
