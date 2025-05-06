# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 23:15:19 2025

@author: joze_
"""

import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os

def generate_synthetic_salesforce_data(dataset_type, num_samples=1000):
    """
    Generates synthetic data resembling Salesforce data for classification tasks.
    The data includes features like account information, opportunity details,
    and sales rep performance.  The target variable depends on the dataset_type.

    Args:
        dataset_type (str): The type of dataset to generate.
            Must be one of: 'opportunity', 'account', 'lead'.
        num_samples (int, optional): The number of data points to generate.
            Defaults to 1000.

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the generated data.

    Raises:
        ValueError: If an invalid dataset_type is provided.
    """
    if dataset_type not in ['opportunity', 'account', 'lead']:
        raise ValueError("Invalid dataset_type. Must be 'opportunity', 'account', or 'lead'.")

    # --- Feature Definitions ---
    num_accounts = 500  # Total number of unique accounts
    num_sales_reps = 50  # Total number of sales representatives
    num_products = 100 # Total number of products
    num_leads = 1000
    # Define features
    features = {
        'account_id': {'type': 'categorical', 'categories': range(num_accounts)},
        'sales_rep_id': {'type': 'categorical', 'categories': range(num_sales_reps)},
        'product_id': {'type': 'categorical', 'categories': range(num_products)},
        'opportunity_amount': {'type': 'numeric', 'min': 100, 'max': 100000},
        'stage': {'type': 'categorical', 'categories': ['Prospecting', 'Qualification', 'Needs Analysis', 'Proposal', 'Negotiation', 'Closed Won', 'Closed Lost']},
        'close_date': {'type': 'datetime', 'start': '2023-01-01', 'end': '2024-12-31'},
        'account_type': {'type': 'categorical', 'categories': ['Customer', 'Prospect', 'Partner']},
        'account_size': {'type': 'numeric', 'min': 1, 'max': 1000},  # Number of employees
        'industry': {'type': 'categorical', 'categories': ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing']},
        'lead_source': {'type': 'categorical', 'categories': ['Web', 'Referral', 'Phone', 'Trade Show', 'Advertisement']},
        'lead_rating': {'type': 'categorical', 'categories': ['Hot', 'Warm', 'Cold']},
        'annual_revenue': {'type': 'numeric', 'min': 100000, 'max': 1000000000},
        'customer_satisfaction': {'type': 'numeric', 'min': 1, 'max': 5},
        'number_of_employees': {'type': 'numeric', 'min': 1, 'max': 50000},
        'days_since_last_activity': {'type': 'numeric', 'min': 0, 'max': 365},
        'contact_method': {'type': 'categorical', 'categories': ['Email', 'Phone', 'Meeting']},
        'campaign_source': {'type': 'categorical', 'categories': ['Campaign A', 'Campaign B', 'Campaign C', 'Organic']},
        'service_level': {'type': 'categorical', 'categories': ['Platinum', 'Gold', 'Silver', 'Bronze']},
        'contract_value': {'type': 'numeric', 'min': 1000, 'max': 1000000},
        'renewal_date': {'type': 'datetime', 'start': '2024-01-01', 'end': '2026-12-31'},
    }

    # --- Data Generation ---
    data = {}
    for feature_name, feature_params in features.items():
        if feature_params['type'] == 'numeric':
            data[feature_name] = np.random.randint(feature_params['min'], feature_params['max'] + 1, num_samples)
        elif feature_params['type'] == 'categorical':
            data[feature_name] = np.random.choice(feature_params['categories'], num_samples)
        elif feature_params['type'] == 'datetime':
            start_date = pd.to_datetime(feature_params['start'])
            end_date = pd.to_datetime(feature_params['end'])
            time_between_dates = end_date - start_date
            days_between_dates = time_between_dates.days
            random_days = np.random.randint(0, days_between_dates, num_samples)
            data[feature_name] = start_date + pd.to_timedelta(random_days, unit='d')

    df = pd.DataFrame(data)

     # --- Target Variable Generation ---
    if dataset_type == 'opportunity':
        # Target:  Probability of winning the opportunity (binary classification)
        df['target'] = np.where(
            (df['stage'].isin(['Proposal', 'Negotiation'])) & (df['opportunity_amount'] > 5000),
            1,
            np.random.choice([0, 1], num_samples, p=[0.7, 0.3])  # 70% chance of losing
        )
    elif dataset_type == 'account':
        # Target: Account health (multiclass classification)
        df['target'] = np.select(
            [
                (df['customer_satisfaction'] < 2) | (df['days_since_last_activity'] > 180),  # At Risk
                (df['customer_satisfaction'] < 4) & (df['days_since_last_activity'] > 90),    # Needs Attention
                (df['customer_satisfaction'] >= 4) & (df['days_since_last_activity'] <= 90)   # Healthy
            ],
            ['At Risk', 'Needs Attention', 'Healthy'],
            default='Healthy'
        )
        df['target'] = df['target'].astype('category') #convert to category
    elif dataset_type == 'lead':
        # Target: Lead conversion status (binary classification)
        df['target'] = np.where(
            (df['lead_rating'].isin(['Hot', 'Warm'])) & (df['contact_method'] == 'Meeting'),
            1,
            np.random.choice([0, 1], num_samples, p=[0.8, 0.2])  # 80% chance of not converting
        )
    return df

def preprocess_data(df, dataset_type):
    """
    Preprocesses the DataFrame, including handling datetime and categorical features.

    Args:
        df (pd.DataFrame): The input DataFrame.
        dataset_type (str): The type of dataset ('opportunity', 'account', or 'lead').

    Returns:
        tuple: A tuple containing (X_train, X_test, y_train, y_test).
    """
    # Drop less useful columns
    df = df.drop(['account_id', 'sales_rep_id', 'product_id'], axis=1, errors='ignore')

    # Convert datetime features to numerical (e.g., days since today)
    datetime_cols = df.select_dtypes(include=['datetime64[ns]']).columns
    for col in datetime_cols:
        df[col + '_days_since'] = (pd.to_datetime('today') - df[col]).dt.days
        df = df.drop(col, axis=1)

    # Convert categorical features to numerical using one-hot encoding
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    df = pd.get_dummies(df, columns=categorical_features, dummy_na=False)

    # Separate features and target variable
    X = df.drop('target', axis=1, errors='ignore')
    y = df['target'] if 'target' in df else None

    # Split data into training and testing sets
    if y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test
    else:
        return X, None, None, None

def save_dataset(df, dataset_name, directory="synthetic_data_set"):
    """
    Saves the generated dataset to a CSV file in a specified directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        dataset_name (str): The name of the dataset.
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
    for dataset_type in ['opportunity', 'account', 'lead']:
        print(f"Generating {dataset_type} dataset...")
        df = generate_synthetic_salesforce_data(dataset_type, num_samples=12000) # slightly increased num_samples
        save_dataset(df, f"salesforce_{dataset_type}_syntetic_data")

        # Example of preprocessing and splitting the data (optional)
        X_train, X_test, y_train, y_test = preprocess_data(df.copy(), dataset_type)
        if y_train is not None:
            print(f"  {dataset_type.title()} dataset: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
            print(f"  {dataset_type.title()} dataset: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
        else:
            print(f"  {dataset_type.title()} dataset: Preprocessed data shape: {X_train.shape}")
