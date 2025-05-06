import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.exceptions import ConvergenceWarning
import warnings
from ml.read_file import read_data_flexible

#Generate additional imports according the specifications

# Define the data file path
DATA_FILE_PATH =[] #update this according the specifications
MODEL_DIR = "trained_models"
METRICS_FILE = os.path.join(MODEL_DIR, "classification_metrics.json")
MODEL_FILE = os.path.join(MODEL_DIR, "best_default_classifier.joblib")
REPORT_FILE = os.path.join(MODEL_DIR, "classification_report.txt")


FEATURE_COLUMNS = []#update this according the specifications
TARGET_COLUMN = []#update this according the specifications

NUMERICAL_FEATURES = []#update this according the specifications
CATEGORICAL_FEATURES =[] #update this according the specifications


def _get_feature_names(column_transformer):
    """Gets feature names from a ColumnTransformer."""
    output_features = []
    for name, transformer, features in column_transformer.transformers_:
        if name == 'remainder': # Skip dropped columns if any
             continue
        if hasattr(transformer, 'get_feature_names_out'):
            # For transformers like OneHotEncoder
            if isinstance(features, str): # Handle single column case
                 features = [features]
            names = transformer.get_feature_names_out(features)
            output_features.extend(names)
        elif name == 'num': # For StandardScaler or other numeric transformers
            # Assumes numeric features are passed directly
             output_features.extend(features)
        else:
             # Fallback for transformers without get_feature_names_out
             output_features.extend(features)
    return output_features


def train_model(data_path=DATA_FILE_PATH):
    """
    Trains and evaluates  models according the specifications

    Args:
        data_path (str): The path to the input CSV file.

    Returns:
        dict: A dictionary containing evaluation metrics and model information.
              Example: {'best_model': 'RandomForestClassifier',
                        'accuracy': 0.98,
                        'f1_score': 0.15,
                        'roc_auc': 0.75,
                        'feature_importance': {'feature1': 0.2, ...},
                        'model_path': 'trained_models/best_default_classifier.joblib',
                        'metrics_path': 'trained_models/classification_metrics.json',
                        'report_path': 'trained_models/classification_report.txt'}
    """
   

    # Return key metrics and feature importance as requested
    return_metrics = {
        "Accuracy": final_output['accuracy'],
        "feature_importance": final_output['feature_importance']
        # Add other key metrics if needed
    }
    return return_metrics # Return only Accuracy and feature importance as per prompt

# Main execution block
if __name__ == "__main__":
    # print(f"Starting model training using data: {DATA_FILE_PATH}")
    training_results = train_model(DATA_FILE_PATH)
    if "error" in training_results:
        print(json.dumps({"error": training_results["error"]}))  # Output JSON error
    else:
        print(json.dumps(training_results))  # Output JSON results