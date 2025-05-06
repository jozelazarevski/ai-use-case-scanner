 

import pandas as pd
import re
from collections import defaultdict

class SimpleColumnTypeDetector:
    """A simplified column type detector that can identify common column types."""
    
    def __init__(self):
        self.column_types = self._initialize_column_types()
        
    def _initialize_column_types(self):
        """Initialize the dictionary of column types and their patterns."""
        return {
            # Customer-related types
            "customer_id": [r'customer.*id', r'client.*id', r'cust.*no'],
            "customer_name": [r'customer.*name', r'client.*name', r'full.*name'],
            "email": [r'email', r'e-?mail'],
            "phone": [r'phone', r'telephone', r'mobile', r'cell'],
            
            # Product-related types
            "product_id": [r'product.*id', r'item.*id', r'sku'],
            "product_name": [r'product.*name', r'item.*name', r'description'],
            "product_category": [r'category', r'product.*type', r'product.*group'],
            "price": [r'price', r'cost', r'amount'],
            
            # Order-related types
            "order_id": [r'order.*id', r'transaction.*id', r'invoice.*no'],
            "order_date": [r'order.*date', r'purchase.*date', r'transaction.*date'],
            "quantity": [r'quantity', r'qty', r'count', r'units'],
            
            # Date/time types
            "date": [r'date', r'day', r'created.*on'],
            "time": [r'time', r'hour', r'minute'],
            
            # Location types
            "address": [r'address', r'street', r'location'],
            "city": [r'city', r'town', r'municipality'],
            "state": [r'state', r'province', r'region'],
            "country": [r'country', r'nation'],
            "zip_code": [r'zip', r'postal.*code', r'post.*code'],
            
            # Financial types
            "revenue": [r'revenue', r'sales', r'income'],
            "discount": [r'discount', r'reduction', r'markdown'],
            "tax": [r'tax', r'vat', r'gst'],
            
            # General types
            "id": [r'id$', r'identifier', r'key', r'code'],
            "name": [r'name$', r'title', r'label'],
            "description": [r'description', r'desc', r'details'],
            "status": [r'status', r'state', r'condition'],
            "url": [r'url', r'link', r'website'],
            "boolean": [r'is_', r'has_', r'flag']
        }
        
    def detect_column_types(self, df):
        """
        Detect column types in a DataFrame
        
        Args:
            df: pandas DataFrame
            
        Returns:
            dict: Mapping of column names to their detected types
        """
        column_mapping = {}
        
        for column in df.columns:
            column_lower = column.lower()
            best_match = None
            best_score = 0
            
            # Try to match column name against patterns
            for type_name, patterns in self.column_types.items():
                for pattern in patterns:
                    if re.search(pattern, column_lower):
                        score = 1.0
                        if score > best_score:
                            best_score = score
                            best_match = type_name
            
            # If no match by name, try to infer from values
            if best_match is None:
                best_match = self._infer_from_values(df[column])
                best_score = 0.6  # Lower confidence for value-based inference
            
            column_mapping[column] = {
                'detected_type': best_match,
                'confidence': best_score
            }
            
        return column_mapping
        
    def _infer_from_values(self, series):
        """Infer column type from values."""
        # Drop NA values
        clean_series = series.dropna()
        if len(clean_series) == 0:
            return 'unknown'
            
        # Check if it's a datetime
        if pd.api.types.is_datetime64_dtype(series):
            return 'date'
            
        # Check if it's a boolean
        if pd.api.types.is_bool_dtype(series):
            return 'boolean'
        
        # Check if it's numeric
        if pd.api.types.is_numeric_dtype(series):
            if series.nunique() < 5 and series.isin([0, 1]).all():
                return 'boolean'
            return 'numeric'
            
        # For string/object types, sample values
        if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
            sample = clean_series.sample(min(5, len(clean_series)))
            
            # Check for email pattern
            if all(isinstance(x, str) and re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$', x) for x in sample):
                return 'email'
                
            # Check for URL pattern
            if all(isinstance(x, str) and re.match(r'^https?://[^\s/$.?#].[^\s]*$', x) for x in sample):
                return 'url'
                
            # Check for phone pattern - FIXED REGEX
            if all(isinstance(x, str) and re.match(r'^\+?[\d\s\(\).\-]{7,}$', x) for x in sample):
                return 'phone'
                
            return 'string'
            
        return 'unknown'

# Example usage
def analyze_csv(csv_file_path):
    """
    Analyze a CSV file using the SimpleColumnTypeDetector
    
    Args:
        csv_file_path: Path to the CSV file
        
    Returns:
        tuple: DataFrame and column type mapping
    """
    # Load CSV
    df = pd.read_csv(csv_file_path)
    
    # Initialize detector
    detector = SimpleColumnTypeDetector()
    
    # Detect column types
    column_mapping = detector.detect_column_types(df)
    
    # Create a standardized matrix
    standardized_matrix = pd.DataFrame([
        {'column_type': col_type, 'original_column': col_name, 'confidence': info['confidence']}
        for col_name, info in column_mapping.items()
        for col_type in [info['detected_type']]
    ])
    
    # Save the standardized matrix
    standardized_matrix.to_csv('standardized_column_types.csv', index=False)
    
    print(f"Column type mapping saved to standardized_column_types.csv")
    return df, column_mapping

# Call with your CSV file
filename="D:/AI Use Case App_GOOD/uploads/Rolex_sales_data.csv"
df, mapping = analyze_csv(filename)