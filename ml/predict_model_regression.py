import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from .read_file import read_data_flexible

def predict_regression(model_path, input_data):
    """
    Predict using a trained regression model
    
    Args:
        model_path (str): Path to the saved model joblib file
        input_data (pd.DataFrame or list or dict): Input data for prediction
    
    Returns:
        pd.DataFrame: Input dataframe with added prediction column
    """
    # If input is a dictionary, convert to DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_data = pd.DataFrame(input_data)
    
    # Ensure input is a DataFrame
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("Input data must be a DataFrame, dictionary, or list")
    
    # Create a copy to avoid modifying the original dataframe
    result_df = input_data.copy()
    
    # Load model and feature names
    model = joblib.load(model_path)
    
    # Extract target variable from model filename
    target_variable = os.path.basename(model_path).replace("best_model_", "").replace(".joblib", "")
    
    # Load feature names and preprocessor
    features_path = model_path.replace("best_model_", "model_features_")
    feature_names = joblib.load(features_path)
    preprocessor_path = model_path.replace("best_model_", "preprocessor_")
    
    # Load preprocessor if it exists, otherwise recreate it
    try:
        preprocessor = joblib.load(preprocessor_path)
    except FileNotFoundError:
        # Recreate preprocessor if saved file not found
        # Identify numerical and categorical features
        numerical_features = input_data.select_dtypes(include=['number']).columns
        categorical_features = input_data.select_dtypes(include=['object', 'category']).columns
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ],
            remainder='passthrough'
        )
        preprocessor.fit(input_data)
    
    # Preprocess the input data
    X = preprocessor.transform(input_data)
    
    # Rename processed features
    feature_names_after_preprocessing = preprocessor.get_feature_names_out()
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names_after_preprocessing)
    else:
        X = pd.DataFrame(X.toarray(), columns=feature_names_after_preprocessing)
    
    # Make prediction
    prediction = model.predict(X)
    
    # Add prediction column to result dataframe
    result_df[f'predicted_{target_variable}'] = prediction
    
    # Add formatted prediction column (rounded to 4 decimal places)
    result_df['formatted_prediction'] = result_df[f'predicted_{target_variable}'].apply(lambda x: f"{float(x):.4f}")
    
    # Add prediction intervals if model supports it
    try:
        if hasattr(model, 'predict_interval'):
            intervals = model.predict_interval(X, alpha=0.05)  # 95% confidence interval
            
            # Add lower and upper bound columns
            result_df['lower_bound'] = [float(interval[0]) for interval in intervals]
            result_df['upper_bound'] = [float(interval[1]) for interval in intervals]
    except Exception:
        # If interval prediction is not supported, we skip it
        pass
    
    # Add metadata as dataframe attributes for reference
    result_df.target_variable = target_variable
    
    return result_df

def load_preprocessing_steps(filepath):
    """
    Load or recreate preprocessing steps for a given dataset
    
    Args:
        filepath (str): Path to the input data file
    
    Returns:
        tuple: Preprocessor and features
    """
    # Read the data
    df = read_data_flexible(filepath)
    
    if df is None:
        raise ValueError("Could not read the input file")
    
    # Identify numerical and categorical features
    numerical_features = df.select_dtypes(include=['number']).columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Fit the preprocessor
    preprocessor.fit(df)
    
    return preprocessor, list(df.columns)