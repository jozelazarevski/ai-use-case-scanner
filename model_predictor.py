"""
Model Predictor Module

This module handles loading trained models and making predictions on new data.
It serves as the backend logic for the model testing interface.
"""

import os
import json
import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def load_trained_model(model_path):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the saved model file
        
    Returns:
        tuple: (model, metadata, features) containing the loaded model,
               its metadata, and required features
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Get model directory and target variable
        model_dir = os.path.dirname(model_path)
        model_filename = os.path.basename(model_path)
        target_variable = model_filename.replace("best_model_", "").replace(".joblib", "")
        
        # Load metadata if available
        metadata = {}
        metadata_file = os.path.join(model_dir, f"metadata_{target_variable}.json")
        if os.path.exists(metadata_file):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
            except Exception as e:
                warnings.warn(f"Error loading metadata: {str(e)}")
        
        # Try to determine model type if not in metadata
        if 'model_type' not in metadata:
            # Infer from model class
            if hasattr(model, 'predict_proba'):
                metadata['model_type'] = "classification"
            else:
                metadata['model_type'] = "regression"
        
        # Load feature names
        features = []
        features_file = os.path.join(model_dir, f"model_features_{target_variable}.joblib")
        if os.path.exists(features_file):
            try:
                features = joblib.load(features_file)
            except Exception as e:
                warnings.warn(f"Error loading feature list: {str(e)}")
        
        # Add target variable to metadata
        metadata['target_variable'] = target_variable
        
        return model, metadata, features
    
    except Exception as e:
        raise RuntimeError(f"Error loading model: {str(e)}")

def preprocess_input_data(data, features, model_type):
    """
    Preprocess input data to match the format expected by the model.
    
    Args:
        data (pd.DataFrame): Input data to preprocess
        features (list): List of features used by the model
        model_type (str): Type of model (classification, regression, clustering)
        
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
    """
    # Ensure all required features are present
    missing_features = set(features) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing required features: {missing_features}")
    
    # Select only the required features and in the correct order
    processed_data = data[features].copy()
    
    # Handle missing values
    for col in processed_data.columns:
        if processed_data[col].isnull().any():
            if pd.api.types.is_numeric_dtype(processed_data[col]):
                # Fill numeric features with median
                processed_data[col].fillna(processed_data[col].median(), inplace=True)
            else:
                # Fill categorical features with mode
                processed_data[col].fillna(processed_data[col].mode()[0], inplace=True)
    
    # Convert data types if needed
    for col in processed_data.columns:
        # Try to convert string numeric values to float
        if processed_data[col].dtype == 'object':
            try:
                # Check if the column contains numeric values as strings
                processed_data[col] = pd.to_numeric(processed_data[col], errors='ignore')
            except:
                pass
    
    return processed_data

def make_predictions(model, data, metadata):
    """
    Make predictions using the loaded model.
    
    Args:
        model: The trained model
        data (pd.DataFrame): Preprocessed input data
        metadata (dict): Model metadata
        
    Returns:
        dict: Prediction results with appropriate information based on model type
    """
    model_type = metadata.get('model_type', 'unknown')
    results = {
        "type": model_type,
        "target_variable": metadata.get('target_variable', 'unknown'),
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        if model_type == "classification":
            # Classification prediction
            predictions = model.predict(data)
            results["prediction"] = predictions[0] if isinstance(predictions, np.ndarray) else predictions
            
            # Add probabilities if available
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(data)
                if isinstance(probabilities, np.ndarray) and probabilities.ndim > 1:
                    max_prob_index = np.argmax(probabilities[0])
                    results["probability"] = f"{probabilities[0][max_prob_index] * 100:.2f}"
                    
                    # If model has classes_ attribute
                    if hasattr(model, 'classes_'):
                        results["all_classes"] = model.classes_.tolist()
                        results["all_probabilities"] = [float(p * 100) for p in probabilities[0]]
        
        elif model_type == "regression":
            # Regression prediction
            predictions = model.predict(data)
            pred_value = predictions[0] if isinstance(predictions, np.ndarray) else predictions
            results["prediction"] = float(pred_value)
            results["formatted_prediction"] = f"{float(pred_value):.4f}"
            
            # Try to add confidence intervals if possible
            try:
                if hasattr(model, 'get_prediction'):
                    # For statsmodels-like models
                    prediction = model.get_prediction(data)
                    ci = prediction.conf_int(alpha=0.05)
                    results["confidence_interval"] = {
                        "lower": float(ci[0][0]),
                        "upper": float(ci[0][1])
                    }
            except:
                # Many models don't support this
                pass
        
        elif model_type == "clustering":
            # Clustering prediction
            predictions = model.predict(data)
            cluster_id = predictions[0] if isinstance(predictions, np.ndarray) else predictions
            results["prediction"] = int(cluster_id)
            results["formatted_prediction"] = f"Cluster {int(cluster_id)}"
            
            # Try to get additional clustering info
            if hasattr(model, 'transform'):
                try:
                    # Get distance to cluster centers for K-means like models
                    distances = model.transform(data)
                    results["distances"] = distances[0].tolist()
                except:
                    pass
        
        else:
            # Unknown model type
            predictions = model.predict(data)
            results["prediction"] = predictions[0] if isinstance(predictions, np.ndarray) else predictions
            results["formatted_prediction"] = str(results["prediction"])
        
        return results
    
    except Exception as e:
        raise RuntimeError(f"Error making prediction: {str(e)}")

def get_available_models(models_dir="trained_models"):
    """
    Get a list of available trained models.
    
    Args:
        models_dir (str): Directory where trained models are stored
        
    Returns:
        list: List of dictionaries containing model information
    """
    available_models = []
    
    if not os.path.exists(models_dir):
        return available_models
    
    # Look for model files
    for filename in os.listdir(models_dir):
        if filename.startswith("best_model_") and filename.endswith(".joblib"):
            model_path = os.path.join(models_dir, filename)
            target_variable = filename.replace("best_model_", "").replace(".joblib", "")
            
            # Get model metadata
            model_type = "unknown"
            accuracy = "N/A"
            created_date = datetime.fromtimestamp(os.path.getctime(model_path)).strftime('%Y-%m-%d %H:%M')
            
            # Try to load metadata
            metadata_file = os.path.join(models_dir, f"metadata_{target_variable}.json")
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    model_type = metadata.get('model_type', 'unknown')
                    accuracy = metadata.get('accuracy', 'N/A')
                    if isinstance(accuracy, (int, float)):
                        accuracy = f"{accuracy:.4f}"
                except:
                    pass
            
            # Add model to list
            available_models.append({
                "path": model_path,
                "name": target_variable.replace('_', ' ').title(),
                "raw_name": target_variable,
                "type": model_type,
                "accuracy": accuracy,
                "created": created_date
            })
    
    # Sort by creation date (newest first)
    available_models.sort(key=lambda x: x["created"], reverse=True)
    
    return available_models

def read_test_data_from_file(file_path):
    """
    Read test data from an uploaded file.
    
    Args:
        file_path (str): Path to the uploaded file
        
    Returns:
        pd.DataFrame: DataFrame containing the test data
    """
    from read_file import read_data_flexible
    
    # Use the flexible reader to handle various file formats
    data = read_data_flexible(file_path)
    
    if data is None:
        raise ValueError(f"Failed to read data from file: {file_path}")
    
    return data

def process_manual_input(input_data, feature_names):
    """
    Process manually entered data to create a DataFrame.
    
    Args:
        input_data (dict): Dictionary of feature name -> value pairs
        feature_names (list): List of expected feature names
        
    Returns:
        pd.DataFrame: DataFrame with a single row containing the input data
    """
    processed_data = {}
    
    # Process each feature
    for feature in feature_names:
        if feature in input_data:
            value = input_data[feature]
            
            # Try to convert numeric strings to numbers
            try:
                # First attempt to convert to int if possible
                value = int(value)
            except:
                try:
                    # Then try float
                    value = float(value)
                except:
                    # Keep as string if both fail
                    pass
            
            processed_data[feature] = [value]
        else:
            # If feature is missing, add as NaN
            processed_data[feature] = [None]
    
    # Create DataFrame
    df = pd.DataFrame(processed_data)
    
    return df

def run_prediction_pipeline(model_path, input_file=None, manual_input=None):
    """
    Complete prediction pipeline from input to result.
    
    Args:
        model_path (str): Path to the trained model
        input_file (str, optional): Path to input file
        manual_input (dict, optional): Dictionary of manual input values
        
    Returns:
        dict: Prediction results
    """
    # Load the model
    model, metadata, features = load_trained_model(model_path)
    
    # Process input data
    if input_file:
        # Read from file
        data = read_test_data_from_file(input_file)
    elif manual_input:
        # Process manual input
        data = process_manual_input(manual_input, features)
    else:
        raise ValueError("No input data provided (either file or manual input required)")
    
    # Preprocess the data
    processed_data = preprocess_input_data(data, features, metadata.get('model_type', 'unknown'))
    
    # Make predictions
    prediction_results = make_predictions(model, processed_data, metadata)
    
    # Add additional information
    prediction_results["features_used"] = features
    prediction_results["input_data"] = data.to_dict(orient='records')[0] if not data.empty else {}
    prediction_results["model_info"] = {
        "name": os.path.basename(model_path),
        "accuracy": metadata.get('accuracy', 'N/A'),
        "model_type": metadata.get('model_type', 'unknown')
    }
    
    return prediction_results