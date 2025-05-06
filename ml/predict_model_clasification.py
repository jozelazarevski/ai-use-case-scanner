import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from .read_file import read_data_flexible

def predict_classification(model_path, input_data):
    """
    Predict using a trained classification model
    
    Args:
        model_path (str): Path to the saved model joblib file
        input_data (pd.DataFrame or list or dict): Input data for prediction
    
    Returns:
        dict: Prediction results with probabilities and other metadata
    """
    # If input is a dictionary, convert to DataFrame
    if isinstance(input_data, dict):
        input_data = pd.DataFrame([input_data])
    elif isinstance(input_data, list):
        input_data = pd.DataFrame(input_data)
    
    # Ensure input is a DataFrame
    if not isinstance(input_data, pd.DataFrame):
        raise ValueError("Input data must be a DataFrame, dictionary, or list")
    
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
    
    # Prepare results dictionary
    results = {
        "prediction": prediction.tolist() if isinstance(prediction, np.ndarray) else prediction,
        "input_data": input_data.to_dict()
    }
    
    # Add probabilities if the model supports it
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X)
        
        # If model has classes_ attribute
        if hasattr(model, 'classes_'):
            results["all_classes"] = list(map(str, model.classes_))
            
            # Handle probabilities correctly based on dimensions
            if len(input_data) == 1:
                # Single sample case - return probabilities for each class
                results["all_probabilities"] = [float(p * 100) for p in probabilities[0]]
                
                # Find max probability
                max_prob_index = np.argmax(probabilities[0])
                results["probability"] = f"{probabilities[0][max_prob_index] * 100:.2f}%"
            else:
                # Multiple samples case - return list of probabilities for each sample
                results["all_probabilities"] = [[float(p * 100) for p in sample_probs] for sample_probs in probabilities]
                
                # Find max probability for each sample
                max_prob_indices = np.argmax(probabilities, axis=1)
                results["probability"] = [f"{probabilities[i][max_prob_indices[i]] * 100:.2f}%" for i in range(len(input_data))]
    
    return results

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