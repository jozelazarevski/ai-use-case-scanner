import pandas as pd
import numpy as np
import joblib
import os
import sys

# Try different import approaches to handle various execution environments
try:
    # First try as absolute import from ml package
    from ml.preprocess_data import preprocess_data, load_preprocessor, get_output_dir, get_model_filename
    from ml.read_file import read_data_flexible
except ImportError:
    try:
        # Then try direct import (when in same directory)
        from preprocess_data import preprocess_data, load_preprocessor, get_output_dir, get_model_filename
        from read_file import read_data_flexible
    except ImportError:
        try:
            # Last resort: try relative import if we're in a package
            from .preprocess_data import preprocess_data, load_preprocessor, get_output_dir, get_model_filename
            from .read_file import read_data_flexible
        except ImportError:
            print("Error importing required modules. Please check your Python path setup.")
            raise

def handle_categorical_for_xgboost(df):
    """
    Convert object dtype columns to category dtype for XGBoost
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: DataFrame with categorical columns converted
    """
    result = df.copy()
    for col in result.columns:
        if result[col].dtype == 'object':
            result[col] = result[col].astype('category')
    return result

def align_features(X, expected_features):
    """
    Align features in X with expected features by adding missing columns
    and removing extra columns.
    
    Args:
        X (pd.DataFrame): Input features
        expected_features (list): Expected feature names
    
    Returns:
        pd.DataFrame: DataFrame with aligned features
    """
    # Create a new DataFrame with the expected features
    X_aligned = pd.DataFrame(index=X.index)
    
    # Add expected features that exist in X
    common_features = [f for f in expected_features if f in X.columns]
    X_aligned[common_features] = X[common_features]
    
    # Add missing features with 0 values
    missing_features = [f for f in expected_features if f not in X.columns]
    for feature in missing_features:
        X_aligned[feature] = 0
    
    # Ensure columns are in the expected order
    X_aligned = X_aligned[expected_features]
    
    return X_aligned

def predict_classification(model_path, input_data, user_id, use_case, target_variable=None):
    """
    Predict using a trained classification model
    
    Args:
        model_path (str): Path to the saved model joblib file
        input_data (pd.DataFrame or list or dict): Input data for prediction
        user_id (str): User ID for organized predictions
        use_case (str): Use case identifier
        target_variable (str, optional): Explicitly specify target variable name
    
    Returns:
        pd.DataFrame: Input dataframe with added prediction and probability columns
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
    
    # Check if input_data is empty
    if input_data.empty:
        raise ValueError("Input data cannot be empty")
    
    try:
        # Extract model info from path
        model_dir = os.path.dirname(model_path)
        model_basename = os.path.basename(model_path)
        
        # Verify model directory exists and is in the expected format
        databases_dir = "databases"
        trained_models_dir = "trained_models"
        if databases_dir not in model_dir or trained_models_dir not in model_dir:
            print(f"Warning: Model path {model_dir} does not include expected 'databases/trained_models' directories")
            
        # Parse filename to extract target variable
        # Filename format should be: {use_case}_{target_variable}_{filename}.joblib
        parts = model_basename.replace(".joblib", "").split('_')
        
        # Determine target variable 
        if target_variable:
            print(f"Using explicitly provided target variable: {target_variable}")
        elif len(parts) >= 2:
            # The second component should be the target variable
            target_variable = parts[1]
            print(f"Extracted target variable from filename: {target_variable}")
        else:
            # Fallback to use_case as default
            target_variable = use_case
            print(f"Falling back to use case as target variable: {target_variable}")
        
        # Load model
        model = joblib.load(model_path)
        
        # Determine the model type (XGBoost, etc.)
        is_xgboost = False
        model_type_name = type(model).__name__
        if 'XGB' in model_type_name:
            is_xgboost = True
            print(f"Detected XGBoost model: {model_type_name}")
        
        # Get expected feature names from model if available
        expected_features = None
        if hasattr(model, 'feature_names_in_'):
            expected_features = list(model.feature_names_in_)
            print(f"Model has feature_names_in_ with {len(expected_features)} features")
        
        # If model doesn't have feature names, try to load from file
        if expected_features is None:
            # Construct features path based on model path
            features_path = model_path.replace(".joblib", ".joblib")
            features_path = features_path.replace(model_basename, f"model_features_{model_basename}")
            
            try:
                feature_names = joblib.load(features_path)
                if isinstance(feature_names, (list, np.ndarray, pd.Index)):
                    expected_features = list(feature_names)
                    print(f"Loaded {len(expected_features)} feature names from file")
            except (FileNotFoundError, ValueError) as e:
                print(f"Warning: Could not load feature names: {e}")
        
        # Try to use the preprocessor if available
        try:
            # Construct preprocessor path based on model path
            preprocessor_path = model_path.replace(".joblib", ".joblib")
            preprocessor_path = preprocessor_path.replace(model_basename, f"preprocessor_{model_basename}")
            
            if os.path.exists(preprocessor_path):
                try:
                    preprocessor = joblib.load(preprocessor_path)
                    if hasattr(preprocessor, 'transform'):
                        print("Successfully loaded preprocessor")
                        
                        # Use the preprocessor to transform the data
                        X_processed = preprocessor.transform(input_data)
                        
                        # Convert to DataFrame with feature names
                        if hasattr(preprocessor, 'get_feature_names_out'):
                            feature_names_out = preprocessor.get_feature_names_out()
                            if isinstance(X_processed, np.ndarray):
                                X = pd.DataFrame(X_processed, columns=feature_names_out, index=input_data.index)
                            else:
                                X = pd.DataFrame(X_processed.toarray(), columns=feature_names_out, index=input_data.index)
                        else:
                            # Use expected_features if available
                            if expected_features is not None:
                                X = pd.DataFrame(X_processed if isinstance(X_processed, np.ndarray) 
                                              else X_processed.toarray(), 
                                              columns=expected_features, 
                                              index=input_data.index)
                            else:
                                # Last resort: use generic column names
                                X = pd.DataFrame(X_processed if isinstance(X_processed, np.ndarray) 
                                              else X_processed.toarray(), 
                                              index=input_data.index)
                        
                        print(f"Preprocessed data shape: {X.shape}")
                    else:
                        preprocessor = None
                        X = None
                        print("Loaded object is not a valid preprocessor")
                except Exception as e:
                    print(f"Error loading preprocessor: {e}")
                    preprocessor = None
                    X = None
            else:
                print(f"Preprocessor not found at {preprocessor_path}")
                preprocessor = None
                X = None
            
            # If preprocessing failed or no preprocessor available
            if X is None:
                print("Preprocessing failed or no preprocessor available")
                
                if expected_features is not None:
                    # Try to manually align features
                    print(f"Attempting to manually align {len(expected_features)} features")
                    X = align_features(input_data, expected_features)
                else:
                    print("No expected features available, falling back to input data")
                    X = input_data
        
        except Exception as e:
            print(f"Error in feature processing: {e}")
            X = input_data
        
        # Handle categorical variables for XGBoost
        if is_xgboost:
            try:
                if X is not None:
                    X = handle_categorical_for_xgboost(X)
            except Exception as e:
                print(f"Error converting categorical variables for XGBoost: {e}")
        
        # Make prediction
        try:
            if is_xgboost:
                print("Making prediction with XGBoost")
                # For XGBoost, we need to handle categorical features
                prediction = model.predict(X, validate_features=False)
            else:
                print("Making prediction with model")
                prediction = model.predict(X)
            
            # Create prediction column names
            # Try to use the most precise naming possible
            predicted_col_name = f'predicted_{target_variable}'
            result_df[predicted_col_name] = prediction
            
            # If specific column names are missing, add alternative column names 
            # for compatibility with different naming conventions
            alternative_names = [
                f'predicted_{use_case}', 
                'predicted_y'  # Generic fallback 
            ]
            
            for alt_name in alternative_names:
                if alt_name not in result_df.columns and alt_name != predicted_col_name:
                    result_df[alt_name] = prediction
            
            # Add probabilities if the model supports it
            if hasattr(model, 'predict_proba'):
                try:
                    if is_xgboost:
                        probabilities = model.predict_proba(X, validate_features=False)
                    else:
                        probabilities = model.predict_proba(X)
                    
                    # If model has classes_ attribute
                    if hasattr(model, 'classes_'):
                        # For each row, get the probability of the predicted class
                        result_df['probability'] = [
                            probabilities[i, np.where(model.classes_ == prediction[i])[0][0]] 
                            for i in range(len(prediction))
                        ]
                    else:
                        # Just get the max probability for each row
                        result_df['probability'] = [max(p) for p in probabilities]
                except Exception as e:
                    print(f"Warning: Could not calculate probabilities: {e}")
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            result_df['error'] = str(e)
    
    except Exception as e:
        print(f"Error during classification: {e}")
        result_df['error'] = str(e)
    
    # Add metadata as dataframe attributes for reference
    if 'target_variable' not in dir(result_df):
        result_df.target_variable = target_variable
    
    if hasattr(model, 'classes_') and 'classes' not in dir(result_df):
        result_df.classes = list(map(str, model.classes_))
    
    return result_df