# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 23:08:38 2025

@author: joze_
"""

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_classification
import os

def generate_synthetic_ecommerce_data(dataset_type, num_samples=1000):
    """
    Generates synthetic e-commerce data for classification tasks.  The data
    includes features like customer demographics, browsing behavior,
    purchase history, and product information.  The target variable
    depends on the dataset_type.

    Args:
        dataset_type (str):  The type of dataset to generate.
            Must be one of: 'binary', 'multiclass', 'bidding'.
        num_samples (int, optional): The number of data points to generate.
            Defaults to 1000.

    Returns:
        pd.DataFrame:  A Pandas DataFrame containing the generated data.

    Raises:
        ValueError:  If an invalid dataset_type is provided.
    """

    if dataset_type not in ['binary', 'multiclass', 'bidding']:
        raise ValueError("Invalid dataset_type. Must be 'binary', 'multiclass', or 'bidding'.")

    # --- Feature Definitions ---
    num_products = 500  # Total number of unique products
    num_categories = 20  # Total number of product categories
    num_users = 2000    # Total number of unique users

    # Define features with more descriptive names and realistic ranges
    features = {
        'age': {'type': 'numeric', 'min': 18, 'max': 70},
        'gender': {'type': 'categorical', 'categories': ['Male', 'Female', 'Other']},
        'location': {'type': 'categorical', 'categories': ['Urban', 'Suburban', 'Rural']},
        'time_on_site': {'type': 'numeric', 'min': 10, 'max': 600},  # Time spent on site in seconds
        'pages_visited': {'type': 'numeric', 'min': 1, 'max': 50},    # Number of pages visited
        'num_purchases': {'type': 'numeric', 'min': 0, 'max': 20},   # Number of past purchases
        'customer_rating': {'type': 'numeric', 'min': 1, 'max': 5},  # Customer satisfaction rating
        'product_id': {'type': 'categorical', 'categories': range(num_products)},
        'category_id': {'type': 'categorical', 'categories': range(num_categories)},
        'price': {'type': 'numeric', 'min': 10, 'max': 500},        # Product price
        'discount': {'type': 'numeric', 'min': 0, 'max': 0.5},      # Discount applied (0 to 0.5)
        'in_stock': {'type': 'categorical', 'categories': [True, False]},
        'delivery_time': {'type': 'numeric', 'min': 1, 'max': 10},    # Delivery time in days
        'user_id': {'type': 'categorical', 'categories': range(num_users)}, # Unique User ID
        'device_type': {'type': 'categorical', 'categories': ['Mobile', 'Desktop', 'Tablet']}, # Device used for browsing
        'session_duration': {'type': 'numeric', 'min': 60, 'max': 3600}, # Length of user session in seconds
        'orders_completed': {'type': 'numeric', 'min': 0, 'max': 15},  # Number of orders completed
        'support_interactions': {'type': 'numeric', 'min': 0, 'max': 5}, # Number of customer support interactions
        'marketing_channel': {'type': 'categorical', 'categories': ['Email', 'Social Media', 'Organic Search', 'Paid Ads']},
    }

    # --- Data Generation ---
    data = {}
    for feature_name, feature_params in features.items():
        if feature_params['type'] == 'numeric':
            data[feature_name] = np.random.randint(feature_params['min'], feature_params['max'] + 1, num_samples)
        elif feature_params['type'] == 'categorical':
            data[feature_name] = np.random.choice(feature_params['categories'], num_samples)

    df = pd.DataFrame(data)

    # --- Target Variable Generation ---
    if dataset_type == 'binary':
        #  High-value customer (1) or not (0)
        df['target'] = ((df['num_purchases'] > 5) & (df['customer_rating'] > 3) & (df['time_on_site'] > 300)).astype(int)
    elif dataset_type == 'multiclass':
        # Customer segment:  0 - Low, 1 - Medium, 2 - High
        df['target'] = np.select(
            [
                (df['num_purchases'] < 3) | (df['customer_rating'] < 2),
                (df['num_purchases'] < 8) & (df['customer_rating'] < 4),
                (df['num_purchases'] >= 8) & (df['customer_rating'] >= 4)
            ],
            [0, 1, 2],
            default=0  #  If none of the conditions are met, default to 0
        )
    elif dataset_type == 'bidding':
        # Simulate bidding success/failure (1 for success, 0 for failure)
        df['target'] = np.random.choice([0, 1], num_samples, p=[0.7, 0.3]) # 70% fail, 30% success
        df['bid_amount'] = np.random.uniform(10, 100, num_samples) # Add a bid amount feature
        df['product_value'] = df['price'] * (1 + df['discount'])  # Estimated value of the product
    return df

def preprocess_data(df, dataset_type):
    """
    Preprocesses the DataFrame.  Handles categorical features and
    splits the data into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame.
        dataset_type (str): The type of dataset ('binary', 'multiclass', or 'bidding').

    Returns:
        tuple:  A tuple containing (X_train, X_test, y_train, y_test).
    """
    # Drop product_id and user_id as they are too granular.  We keep category_id.
    df = df.drop(['product_id', 'user_id'], axis=1, errors='ignore')

    # Convert categorical features to numerical using one-hot encoding
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_features, dummy_na=False) # No need to handle NaN explicitly here

    # Separate features and target variable
    X = df.drop('target', axis=1, errors='ignore') # errors='ignore' prevents error if 'target' is already removed
    y = df['target'] if 'target' in df else None # Handle the case where there is no target

    # Split data into training and testing sets
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test
    else:
        return X, None, None, None # Return X, and None for the others.

def save_dataset(df, dataset_name, directory="synthetic_data_set"):
    """
    Saves the generated dataset to a CSV file in a specified directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        dataset_name (str): The name of the dataset (used for the filename).
        directory (str, optional): The directory to save the dataset in.
            Defaults to "synthetic_data_set".
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{dataset_name}.csv")
    df.to_csv(filepath, index=False)
    print(f"Dataset '{dataset_name}.csv' saved successfully to '{directory}' directory.")

if __name__ == "__main__":
    # --- Generate and Save Datasets ---
    for dataset_type in ['binary', 'multiclass', 'bidding']:
        print(f"Generating {dataset_type} dataset...")
        df = generate_synthetic_ecommerce_data(dataset_type, num_samples=15000) # Increased num_samples
        save_dataset(df, f"ecommerce_{dataset_type}_data")

        # Example of preprocessing and splitting the data (optional, for demonstration)
        X_train, X_test, y_train, y_test = preprocess_data(df.copy(), dataset_type) # Pass a copy to avoid modifying original df
        if y_train is not None: #  Check if y_train exists (it won't for the 'predict' dataset)
            print(f"  {dataset_type.title()} dataset: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"  {dataset_type.title()} dataset: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        else:
            print(f"  {dataset_type.title()} dataset: Preprocessed data shape: {X_train.shape}")
