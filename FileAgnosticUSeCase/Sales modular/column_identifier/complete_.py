import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import random

# Import all the modules
from column_identifier import ColumnIdentifier
from performance_optimization import PerformanceOptimizer
from comprehensive_column_types import ComprehensiveColumnTypeDetector
from content_based_classification import ContentClassifier
from domain_specific_knowledge import DomainKnowledgeBase

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('sales_analyzer')

# Create synthetic sales data
def generate_sales_data(rows=10000):
    """Generate synthetic sales data"""
    # Base date for transaction generation
    base_date = datetime(2023, 1, 1)
    
    # Product information
    products = [
        {"id": "P001", "name": "Laptop", "category": "Electronics", "price": 1200},
        {"id": "P002", "name": "Smartphone", "category": "Electronics", "price": 800},
        {"id": "P003", "name": "Headphones", "category": "Electronics", "price": 150},
        {"id": "P004", "name": "T-shirt", "category": "Clothing", "price": 25},
        {"id": "P005", "name": "Jeans", "category": "Clothing", "price": 60},
        {"id": "P006", "name": "Sneakers", "category": "Footwear", "price": 90},
        {"id": "P007", "name": "Coffee Maker", "category": "Home", "price": 120},
        {"id": "P008", "name": "Blender", "category": "Home", "price": 80},
        {"id": "P009", "name": "Desk Chair", "category": "Furniture", "price": 200},
        {"id": "P010", "name": "Backpack", "category": "Accessories", "price": 45}
    ]
    
    # Store information
    stores = [
        {"id": "S001", "name": "Downtown Store", "city": "New York", "region": "East"},
        {"id": "S002", "name": "Mall Outlet", "city": "Los Angeles", "region": "West"},
        {"id": "S003", "name": "Corner Shop", "city": "Chicago", "region": "Midwest"},
        {"id": "S004", "name": "Shopping Center", "city": "Houston", "region": "South"},
        {"id": "S005", "name": "Main Street", "city": "Phoenix", "region": "West"}
    ]
    
    # Payment methods
    payment_methods = ["Credit Card", "Cash", "Debit Card", "PayPal", "Store Credit"]
    
    # Generate the data
    data = {
        # Transaction identifiers
        "transaction_id": [f"TRX-{i:06d}" for i in range(1, rows + 1)],
        "order_number": np.arange(1001, 1001 + rows),
        
        # Date and time
        "order_date": [base_date + timedelta(days=random.randint(0, 365)) for _ in range(rows)],
        "order_time": [f"{random.randint(9, 21)}:{random.randint(0, 59):02d}" for _ in range(rows)],
        
        # Product information
        "product_id": [random.choice(products)["id"] for _ in range(rows)],
        "product_name": [None] * rows,  # Will fill in
        "product_category": [None] * rows,  # Will fill in
        "unit_price": [None] * rows,  # Will fill in
        
        # Quantity and sales
        "quantity": np.random.randint(1, 6, size=rows),
        "subtotal": [None] * rows,  # Will calculate
        "discount_pct": np.random.choice([0, 0.05, 0.1, 0.15, 0.2], size=rows),
        "tax_amount": [None] * rows,  # Will calculate
        "total_amount": [None] * rows,  # Will calculate
        
        # Store information
        "store_id": [random.choice(stores)["id"] for _ in range(rows)],
        "store_name": [None] * rows,  # Will fill in
        "store_city": [None] * rows,  # Will fill in
        "store_region": [None] * rows,  # Will fill in
        
        # Customer information
        "customer_id": [f"CUST-{random.randint(1, 1000):04d}" for _ in range(rows)],
        "is_member": np.random.choice([True, False], size=rows),
        "payment_method": [random.choice(payment_methods) for _ in range(rows)],
        
        # Additional metrics
        "items_per_order": np.random.randint(1, 10, size=rows),
        "days_to_delivery": np.random.randint(1, 14, size=rows),
        "customer_rating": np.random.uniform(3, 5, size=rows).round(1)
    }
    
    # Fill in the related data fields
    for i in range(rows):
        # Get the product info
        product = next(p for p in products if p["id"] == data["product_id"][i])
        data["product_name"][i] = product["name"]
        data["product_category"][i] = product["category"]
        data["unit_price"][i] = product["price"]
        
        # Get the store info
        store = next(s for s in stores if s["id"] == data["store_id"][i])
        data["store_name"][i] = store["name"]
        data["store_city"][i] = store["city"]
        data["store_region"][i] = store["region"]
        
        # Calculate financial amounts
        quantity = data["quantity"][i]
        unit_price = data["unit_price"][i]
        discount = data["discount_pct"][i]
        
        data["subtotal"][i] = round(quantity * unit_price, 2)
        data["tax_amount"][i] = round(data["subtotal"][i] * (1 - discount) * 0.08, 2)  # 8% tax
        data["total_amount"][i] = round(data["subtotal"][i] * (1 - discount) + data["tax_amount"][i], 2)
    
    df = pd.DataFrame(data)
    
    # Add some missing values to make it more realistic
    for col in df.columns:
        mask = np.random.rand(len(df)) < 0.02  # 2% missing rate
        df.loc[mask, col] = np.nan
    
    return df

# Generate the dataset
print("Generating synthetic sales data...")
sales_df = generate_sales_data(rows=10000)
print(f"Dataset created with shape: {sales_df.shape}")

sales_df = pd.read_csv('D:/AI Use Case App_GOOD/uploads/Warehouse_and_Retail_Sales.csv')

filename="D:/AI Use Case App_GOOD/uploads/Rolex_sales_data.csv"
sales_df = pd.read_csv(filename)


# First, let's optimize the dataframe for better performance
print("\nOptimizing DataFrame for analysis...")
optimizer = PerformanceOptimizer(
    max_sample_size=5000,
    use_multithreading=True,
    memory_limit_percentage=80.0
)
optimized_df = optimizer.optimize_dataframe(sales_df)
 
# Now, let's use the ColumnIdentifier to identify important columns
print("\nIdentifying columns...")
identifier = ColumnIdentifier(optimized_df)
identifier.identify_all_columns()

# Print the detected columns
print("\nBasic Column Detection Results:")
print(f"Date column: {identifier.date_col}")
print(f"Product column: {identifier.product_col}")
print(f"Sales column: {identifier.sales_col}")

print("\nDetailed Column Detection Results:")
for col_type, col_name in identifier.detected_columns.items():
    if col_name:
        print(f"  {col_type.capitalize()}: {col_name}")



# Finally, let's use the ContentClassifier to detect relationships between columns
print("\nAnalyzing column relationships...")
content_classifier = ContentClassifier()
relationships = content_classifier.detect_column_relationships(optimized_df)

print("\nColumn Relationships:")
for col, rels in relationships.items():
    print(f"\n{col} is related to:")
    for rel in rels:
        if 'related_column' in rel:
            print(f"  - {rel['related_column']} ({rel['relationship_type']}, confidence: {rel['confidence']:.2f})")
        elif 'related_columns' in rel:
            print(f"  - {', '.join(rel['related_columns'])} ({rel['relationship_type']}, confidence: {rel['confidence']:.2f})")
            if 'formula' in rel:
                print(f"    Formula: {rel['formula']}")

# Performance summary
summary = optimizer.get_performance_summary()
print("\nPerformance Summary:")
print(f"Total processing time: {summary['total_time']:.2f} seconds")
print(f"Average column processing time: {summary['avg_column_time']:.4f} seconds")
print(f"Peak memory usage: {summary['peak_memory_mb']:.2f} MB")

print("\nAnalysis complete!")

"""
Improved column mapping code with better detection and debug information

This code provides a more robust approach to mapping detected columns to standard column names.
It includes better debug information to understand why mappings are failing.
"""
"""
Column mapping with corrected priorities to handle exact matches first

This code creates a mapping between detected columns and a set of standard column names,
prioritizing exact matches over pattern-based matches.
"""
"""
Column mapping with corrected priorities to handle exact matches first

This code creates a mapping between detected columns and a set of standard column names,
prioritizing exact matches over pattern-based matches.
"""

# Define the columns we're looking for as a set
requested_columns = {
    'date',
    'product',
    'quantity',
    'price',
    'sales',
    'customer',
    'location',
    'category'
}

print("\nCreating column mapping...")

# Debug: Print all detected columns for reference
print("Detected columns by type:")
for col_type, col_name in identifier.detected_columns.items():
    if col_name:
        print(f"  {col_type}: {col_name}")

# Debug: Print all available columns in the DataFrame
print("\nAvailable columns in DataFrame:")
for col in optimized_df.columns:
    print(f"  {col}")

# Create a mapping of detected columns to requested columns
column_mapping = {}

# Check for exact column name matches first (highest priority)
for req_col in requested_columns:
    if req_col in optimized_df.columns:
        column_mapping[req_col] = req_col
        print(f"  Mapped '{req_col}' to '{req_col}' (exact column name)")
    # Special case for quantity - it should match 'quantity' exactly if present
    elif req_col == 'quantity' and 'quantity' in optimized_df.columns:
        column_mapping['quantity'] = 'quantity'
        print(f"  Mapped 'quantity' to 'quantity' (exact name)")

# Then try to map based on detected column types
for req_col in requested_columns:
    if req_col in column_mapping:
        continue  # Skip if already mapped
        
    # Check if we have a detected column of this exact type
    if req_col in identifier.detected_columns and identifier.detected_columns[req_col]:
        column_mapping[req_col] = identifier.detected_columns[req_col]
        print(f"  Mapped '{req_col}' to '{identifier.detected_columns[req_col]}' (direct match)")
        continue
    
    # Check for similar column types
    for detected_type, detected_col in identifier.detected_columns.items():
        if not detected_col or req_col in column_mapping:
            continue
            
        # Check if the detected type is related to the requested column
        if req_col in detected_type.lower() or detected_type.lower() in req_col:
            column_mapping[req_col] = detected_col
            print(f"  Mapped '{req_col}' to '{detected_col}' (type similarity)")
            break

# If exact matches and type similarity didn't work, try matching with common patterns
# from the ColumnIdentifier's internal patterns
pattern_based_mappings = {}

if hasattr(identifier, 'date_patterns') and 'date' not in column_mapping:
    for col in optimized_df.columns:
        if any(pattern.lower() in col.lower() for pattern in identifier.date_patterns):
            pattern_based_mappings['date'] = col
            break
            
if hasattr(identifier, 'product_patterns') and 'product' not in column_mapping:
    for col in optimized_df.columns:
        if any(pattern.lower() in col.lower() for pattern in identifier.product_patterns):
            pattern_based_mappings['product'] = col
            break
            
if hasattr(identifier, 'quantity_patterns') and 'quantity' not in column_mapping:
    # First check for exact 'quantity' column
    if 'quantity' in optimized_df.columns:
        pattern_based_mappings['quantity'] = 'quantity'
    else:
        for col in optimized_df.columns:
            if any(pattern.lower() in col.lower() for pattern in identifier.quantity_patterns):
                pattern_based_mappings['quantity'] = col
                break
            
if hasattr(identifier, 'price_patterns') and 'price' not in column_mapping:
    for col in optimized_df.columns:
        if any(pattern.lower() in col.lower() for pattern in identifier.price_patterns):
            pattern_based_mappings['price'] = col
            break
            
if hasattr(identifier, 'sales_patterns') and 'sales' not in column_mapping:
    for col in optimized_df.columns:
        if any(pattern.lower() in col.lower() for pattern in identifier.sales_patterns):
            pattern_based_mappings['sales'] = col
            break

# Add pattern-based mappings to our column mapping
for req_col, mapped_col in pattern_based_mappings.items():
    if req_col not in column_mapping:  # Only add if not already mapped
        column_mapping[req_col] = mapped_col
        print(f"  Mapped '{req_col}' to '{mapped_col}' (using {req_col}_patterns)")

# Special cases for columns frequently found in the synthetic dataset
if 'date' not in column_mapping and 'order_date' in optimized_df.columns:
    column_mapping['date'] = 'order_date'
    print(f"  Mapped 'date' to 'order_date' (special case)")

if 'sales' not in column_mapping and 'total_amount' in optimized_df.columns:
    column_mapping['sales'] = 'total_amount'
    print(f"  Mapped 'sales' to 'total_amount' (special case)")

if 'product' not in column_mapping and 'product_id' in optimized_df.columns:
    column_mapping['product'] = 'product_id'
    print(f"  Mapped 'product' to 'product_id' (special case)")

if 'quantity' not in column_mapping and 'quantity' in optimized_df.columns:
    column_mapping['quantity'] = 'quantity'
    print(f"  Mapped 'quantity' to 'quantity' (special case)")

if 'price' not in column_mapping and 'unit_price' in optimized_df.columns:
    column_mapping['price'] = 'unit_price'
    print(f"  Mapped 'price' to 'unit_price' (special case)")

if 'customer' not in column_mapping and 'customer_id' in optimized_df.columns:
    column_mapping['customer'] = 'customer_id'
    print(f"  Mapped 'customer' to 'customer_id' (special case)")

if 'location' not in column_mapping and 'store_id' in optimized_df.columns:
    column_mapping['location'] = 'store_id'
    print(f"  Mapped 'location' to 'store_id' (special case)")

if 'category' not in column_mapping and 'product_category' in optimized_df.columns:
    column_mapping['category'] = 'product_category'
    print(f"  Mapped 'category' to 'product_category' (special case)")

# Double-check that we've fixed the quantity mapping
if column_mapping.get('quantity') != 'quantity' and 'quantity' in optimized_df.columns:
    column_mapping['quantity'] = 'quantity'
    print(f"  Corrected 'quantity' mapping to 'quantity' (exact match)")

# Check for any missing requested columns
missing_cols = requested_columns - set(column_mapping.keys())
if missing_cols:
    print(f"\nWarning: Could not map columns: {', '.join(missing_cols)}")
    print("Suggestions for missing columns:")
    
    for missing in missing_cols:
        possible_suggestions = []
        for col in optimized_df.columns:
            # Try to identify by analyzing column values
            try:
                if missing == 'date' and pd.api.types.is_datetime64_dtype(optimized_df[col].dtype):
                    possible_suggestions.append((col, "datetime type"))
                elif missing == 'quantity' and pd.api.types.is_integer_dtype(optimized_df[col].dtype) and optimized_df[col].min() >= 0:
                    possible_suggestions.append((col, "positive integers"))
                elif missing == 'price' and pd.api.types.is_float_dtype(optimized_df[col].dtype) and optimized_df[col].min() >= 0:
                    possible_suggestions.append((col, "positive decimal values"))
            except:
                # Skip any errors in type checking
                pass
            
        if possible_suggestions:
            print(f"  For '{missing}', consider: {', '.join([f'{col} ({reason})' for col, reason in possible_suggestions])}")
        else:
            print(f"  For '{missing}': No suitable columns found")

# Create a new DataFrame with the mapped columns
mapped_df = pd.DataFrame()

for std_col, orig_col in column_mapping.items():
    if orig_col in optimized_df.columns:
        mapped_df[std_col] = optimized_df[orig_col]
    else:
        print(f"Warning: Mapped column '{orig_col}' for '{std_col}' not found in DataFrame")

print(f"\nCreated standardized DataFrame with {len(mapped_df.columns)} columns:")
if not mapped_df.empty:
    print(f"  Columns: {', '.join(mapped_df.columns)}")
    print(f"  Rows: {len(mapped_df)}")
    print("\nStandardized DataFrame preview:")
    print(mapped_df.head(5))
else:
    print("  No columns mapped. DataFrame is empty.")

# Save the column mapping and mapped DataFrame to variables for future use
column_mapping_dict = column_mapping
standardized_df = mapped_df

print("\nColumn mapping dictionary and standardized DataFrame are now available for use.")