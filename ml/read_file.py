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
    Reads data from various file types (Excel, JSON, delimited text) and
    returns a Pandas DataFrame.

    Args:
        filename (str): The name of the file to read.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the data, or None if
                      reading fails.
    """

    df = None
    filename = os.path.basename(filepath)  # Extract filename from filepath

    try:
        if filename.endswith(('.xlsx', '.xls')):
            print(f"Attempting to read as Excel file: '{filename}'")
            df = pd.read_excel(filepath)
            print(f"Successfully read Excel file: '{filename}'")
            return df
        elif filename.endswith('.json'):
            print(f"Attempting to read as JSON file: '{filename}'")
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                df = pd.json_normalize(data)  # Handle nested JSON
                print(f"Successfully read JSON file: '{filename}'")
                return df
            except json.JSONDecodeError:
                print(f"Error decoding JSON file: '{filename}'")
                return None
        elif filename.endswith('.txt') or not any(
                filename.endswith(ext) for ext in ['.xlsx', '.xls', '.json']):
            # Handle .txt and other potential delimited files
            print(f"Attempting to read as delimited text file: '{filename}'")
            try:
                with open(filepath, 'rb') as f:
                    raw_data = f.read()
                    encoding_result = chardet.detect(raw_data)
                    encoding = encoding_result['encoding']
                    print(f"Detected encoding: {encoding}")

                # Try common separators
                for sep in [',', ';', '\t', '|', ' ']:
                    try:
                        temp_df = pd.read_csv(filepath, sep=sep, encoding=encoding,
                                            skipinitialspace=True)
                        if len(temp_df.columns) > 1:
                            print(
                                f"Successfully read text file '{filename}' with separator: '{sep}'")
                            return temp_df
                    except pd.errors.ParserError:
                        print(f"ParserError with separator: '{sep}'. Trying next.")
                        pass  # Try the next separator
                print(
                    f"Warning: Could not automatically determine the text file separator for '{filename}'.")
                # Fallback: Try reading with tab for .txt, comma for others
                default_sep = '\t' if filename.endswith('.txt') else ','
                df = pd.read_csv(filepath, sep=default_sep, encoding=encoding,
                                skipinitialspace=True, error_bad_lines=False,
                                warn_bad_lines=False)
                print(
                    f"Attempted to read text file '{filename}' with default separator: '{default_sep}'.")
                return df

            except FileNotFoundError:
                print(f"Error: File not found at '{filepath}'")
                return None
            except UnicodeDecodeError:
                print(f"Error decoding text file '{filename}' with detected encoding.")
                return None
            except Exception as e:
                print(
                    f"An unexpected error occurred reading text file '{filepath}': {e}")
                return None

    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
    except Exception as e:
        print(f"An error occurred reading '{filepath}': {e}")

    return None