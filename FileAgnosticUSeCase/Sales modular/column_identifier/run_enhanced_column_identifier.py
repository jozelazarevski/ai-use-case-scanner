"""
Simple Enhanced Column Identifier Execution Script

This script provides a simplified way to run the column identifier system
with minimal setup and no command-line arguments.
"""

import pandas as pd
import logging
from datetime import datetime
import traceback

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('simple_column_identifier')

# Import the column identifier
try:
    from column_identifier import ColumnIdentifier
except ImportError as e:
    logger.error(f"Could not import ColumnIdentifier: {str(e)}")
    logger.error("Please make sure the column_identifier module is properly installed")
    exit(1)

def generate_sample_data(rows=1000):
    """Generate a simple sample DataFrame for testing with date components"""
    import numpy as np
    from datetime import datetime, timedelta
    
    # Base date for date columns
    base_date = datetime(2023, 1, 1)
    
    # Create sample retail sales data
    data = {
        'order_id': [f'ORD-{i:04d}' for i in range(rows)],
        'order_date': [base_date + timedelta(days=i % 365) for i in range(rows)],
        'customer_id': [f'CUST-{np.random.randint(1, 500):03d}' for _ in range(rows)],
        'product_id': [f'PROD-{np.random.randint(1, 100):03d}' for _ in range(rows)],
        'product_name': [f'Product {np.random.randint(1, 100)}' for _ in range(rows)],
        'quantity': np.random.randint(1, 10, rows),
        'unit_price': np.random.uniform(10, 100, rows).round(2),
        'total_amount': np.zeros(rows),  # We'll calculate this below
        'category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books', 'Sports'], rows),
        'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Cash', 'Bank Transfer'], rows),
        'is_shipped': np.random.choice([True, False], rows, p=[0.8, 0.2])
    }
    
    # Calculate total amount
    data['total_amount'] = data['quantity'] * data['unit_price']
    
    # Add separate year, month, and day columns
    dates = [base_date + timedelta(days=i % 365) for i in range(rows)]
    data['year'] = [date.year for date in dates]
    data['month'] = [date.month for date in dates]
    data['day'] = [date.day for date in dates]
    
    # Add fiscal year (different from calendar year)
    data['fiscal_year'] = [date.year if date.month >= 7 else date.year - 1 for date in dates]
    
    # Add quarter
    data['quarter'] = [((date.month - 1) // 3) + 1 for date in dates]
    
    # Add week
    data['week'] = [(date - datetime(date.year, 1, 1)).days // 7 + 1 for date in dates]
    
    # Add day of week (0=Monday, 6=Sunday)
    data['day_of_week'] = [date.weekday() for date in dates]
    
    return pd.DataFrame(data)

def run_column_identifier(data=None):
    """Run the column identifier on the given or sample data"""
    try:
        # Generate sample data if none provided
        if data is None:
            logger.info("Generating sample data...")
            data = generate_sample_data()
            logger.info(f"Sample data generated: {len(data)} rows, {len(data.columns)} columns")
        
        # Initialize the column identifier
        logger.info("Initializing column identifier...")
        identifier = ColumnIdentifier(data)
        
        # Run the identification
        logger.info("Identifying columns...")
        success = identifier.identify_all_columns()
        
        if success:
            # Print the results
            logger.info("\nColumn Identification Results:")
            for col_type, col_name in identifier.detected_columns.items():
                if col_name:
                    logger.info(f"  {col_type}: {col_name}")
            
            # Return the identifier for further inspection if needed
            return identifier
        else:
            logger.error("Column identification failed")
            return None
            
    except Exception as e:
        logger.error(f"Error running column identifier: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def main():
    """Main function to run the script"""
    logger.info("Starting Simple Column Identifier")
    
    # Option 1: Run with generated sample data
    identifier = run_column_identifier()
    
    # Option 2: Uncomment to run with data from a CSV file
    # try:
    #     data = pd.read_csv('your_data.csv')
    #     identifier = run_column_identifier(data)
    # except Exception as e:
    #     logger.error(f"Error loading CSV file: {str(e)}")
    
    logger.info("Simple Column Identifier execution complete")
    return identifier

if __name__ == "__main__":
    main()