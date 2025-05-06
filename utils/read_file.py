# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:12:48 2025

@author: joze_
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import os
import chardet
import numpy as np



import pandas as pd
import chardet
import io
import csv
import json
"""
filename = 'pricerunner_aggregate.csv'
filename = "UCI CBM Dataset.txt"
filename = "crx.data"
target_variable = "y"  # Default target variable

"""
import pandas as pd
import chardet
import json


def read_data_flexible(filepath):
    """
    Reads data from various file formats with flexible handling of encodings and delimiters.
    
    Args:
        filepath (str): Path to the file to read
        
    Returns:
        pd.DataFrame or None: The read data, or None if reading fails
    """
    import os
    import pandas as pd
    import chardet
    import io
    import csv
    import traceback
    
    # Check if file exists
    if not os.path.exists(filepath):
        print(f"Error: File '{filepath}' does not exist.")
        return None
    
    # Try to detect file type based on extension
    file_ext = os.path.splitext(filepath)[1].lower()
    
    # For Excel files
    if file_ext in ['.xlsx', '.xls']:
        try:
            print(f"Reading Excel file: {filepath}")
            df = pd.read_excel(filepath, engine='openpyxl')
            return df
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            traceback.print_exc()
            
            # Try with the default engine as fallback
            try:
                df = pd.read_excel(filepath)
                return df
            except Exception as e2:
                print(f"Error with default Excel engine: {e2}")
                traceback.print_exc()
            
    # For JSON files
    elif file_ext == '.json':
        try:
            print(f"Reading JSON file: {filepath}")
            df = pd.read_json(filepath)
            return df
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            traceback.print_exc()
    
    # For all other files (mainly CSV and text files), try to detect encoding
    try:
        # Read a sample to detect encoding
        with open(filepath, 'rb') as f:
            sample = f.read(min(1024 * 1024, os.path.getsize(filepath)))
        
        result = chardet.detect(sample)
        encoding = result['encoding']
        confidence = result['confidence']
        
        print(f"Detected encoding: {encoding} with confidence: {confidence}")
        
        # If confidence is low, try common encodings
        if confidence < 0.8:
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
        else:
            encodings = [encoding, 'utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            
        # Remove duplicates while preserving order
        encodings = list(dict.fromkeys(encodings))
        
        # Try each encoding
        for enc in encodings:
            try:
                # Try to read with pandas directly (auto-detection of delimiter)
                print(f"Trying pandas read_csv with encoding {enc} and auto-detection...")
                df = pd.read_csv(filepath, encoding=enc, sep=None, engine='python')
                
                # Check if we got a single column dataframe when we shouldn't
                if len(df.columns) == 1 and ',' in df.iloc[0, 0]:
                    print("Got a single column with commas, trying different approach...")
                    raise ValueError("Need to try different delimiter")
                    
                return df
            except Exception as e1:
                print(f"Error with auto-detection for encoding {enc}: {e1}")
                
                try:
                    # Try reading with different separators
                    for sep in [',', ';', '\t', '|', ' ']:
                        try:
                            print(f"Trying with encoding {enc} and separator '{sep}'...")
                            df = pd.read_csv(filepath, encoding=enc, sep=sep)
                            
                            # Check if parsing seems correct
                            if len(df.columns) > 1:
                                return df
                        except Exception as e2:
                            continue  # Try next separator
                except Exception:
                    # Continue to next encoding
                    continue
                            
        # If all else fails, try to use the csv module with different dialects
        print("Trying with csv module...")
        for enc in encodings:
            try:
                with open(filepath, 'r', encoding=enc) as f:
                    sample_content = f.read(1024)
                
                dialect = csv.Sniffer().sniff(sample_content)
                
                with open(filepath, 'r', encoding=enc) as f:
                    csv_reader = csv.reader(f, dialect)
                    rows = [row for row in csv_reader]
                
                if rows:
                    # Convert to DataFrame
                    headers = rows[0]
                    data = rows[1:]
                    df = pd.DataFrame(data, columns=headers)
                    return df
            except Exception as e:
                print(f"CSV sniffer failed with encoding {enc}: {e}")
                continue
        
        # Last resort: try to read line by line and make a best guess
        print("Trying line-by-line parsing as last resort...")
        with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        if not lines:
            print("File is empty.")
            return None
        
        # Try to detect delimiter by counting occurrences in first few lines
        delimiters = {',': 0, ';': 0, '\t': 0, '|': 0}
        for line in lines[:min(10, len(lines))]:
            for delimiter in delimiters:
                delimiters[delimiter] += line.count(delimiter)
        
        # Use the most common delimiter
        most_common_delimiter = max(delimiters.items(), key=lambda x: x[1])[0]
        
        # Parse CSV with the detected delimiter
        if delimiters[most_common_delimiter] > 0:
            try:
                # Convert the lines back to a single string
                content = '\n'.join(lines)
                df = pd.read_csv(io.StringIO(content), sep=most_common_delimiter)
                return df
            except Exception as e:
                print(f"Error with last resort parsing: {e}")
        
        print("All file reading methods failed.")
        return None
        
    except Exception as e:
        print(f"Unexpected error reading file: {e}")
        traceback.print_exc()
        return None