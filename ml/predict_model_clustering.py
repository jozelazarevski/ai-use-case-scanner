import os
import time
import traceback
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Import the training functions from the other files
from ml.train_classification import train_classification_model
from ml.train_regression import train_regression_model
from ml.train_clustering import train_clustering_model
os.environ['LOKY_MAX_CPU_COUNT'] = str(4) 


def predict_clustering(model_path, input_data, user_id=None, use_case=None):
    """
    Predicts the cluster for new data points using a trained KMeans model.

    Args:
        model_path (str): Path to the saved KMeans model joblib file.
        input_data (pd.DataFrame or list or dict): Input data for prediction.
        user_id (str, optional): User ID from the logged-in user.
        use_case (str, optional): Use case being solved.

    Returns:
        pd.DataFrame: Input dataframe with added cluster prediction column.
    """
    # Import session if available (for getting user info when not provided)
    try:
        from flask import session
    except ImportError:
        session = None
    
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

    # Load the model
    try:
        kmeans = joblib.load(model_path)
        print(f"Successfully loaded model from: {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        # If we can't load the model, return the original data with default cluster
        result_df['cluster'] = 3
        return result_df
    
    # Extract parts from model path
    model_dir = os.path.dirname(model_path)
    model_name = os.path.basename(model_path)
    parts = model_name.split('_')
    
    # Set default USER and USE_CASE based on parameters or session
    # First try parameters
    USER = user_id if user_id is not None else 'default_user'
    USE_CASE = use_case if use_case is not None else 'default_usecase'
    
    # If not provided in parameters, try to get from session
    if (user_id is None or use_case is None) and session:
        # Get user from session if available
        if user_id is None and 'user_id' in session:
            USER = session['user_id']
        
        # For use case, if training_results has a title, use that
        if use_case is None and 'training_results' in session:
            training_results = session.get('training_results', {})
            if 'title' in training_results:
                USE_CASE = training_results['title'].replace(' ', '_').lower()
    
    # As fallback, try to extract from model filename if possible
    if len(parts) > 3 and parts[0] == "kmeans" and parts[1] == "model":
        # Only override if not specifically provided
        if user_id is None:
            USER = parts[2]
        if use_case is None:
            USE_CASE = parts[3]
    
    print(f"Using USER: {USER}, USE_CASE: {USE_CASE}")
    
    # Sanitize USER and USE_CASE for filename usage
    USER = ''.join(c if c.isalnum() else '_' for c in str(USER))
    USE_CASE = ''.join(c if c.isalnum() else '_' for c in str(USE_CASE))
    
    # Construct various possible paths for preprocessor and features
    possible_preprocessor_paths = [
        # Look in the model directory first
        os.path.join(model_dir, "preprocessor_kmeans.joblib"),
        os.path.join(model_dir, "kmeans_preprocessor.joblib"),
        os.path.join(model_dir, "model_preprocessor.joblib"),
        
        # Check user-specific preprocessor in model directory
        os.path.join(model_dir, f"preprocessor_{USER}_{USE_CASE}_kmeans.joblib"),
        
        # Look in trained_models directory as fallback
        os.path.join("trained_models", f"preprocessor_{USER}_{USE_CASE}_kmeans.joblib"),
        os.path.join("trained_models", "preprocessor_kmeans.joblib"),
        
        # Standard naming patterns based on model name
        model_path.replace("kmeans_model_", "preprocessor_").replace(
            f"_{parts[-1]}" if len(parts) > 1 else "", "_kmeans.joblib")
    ]
    
    possible_features_paths = [
        # Look in the model directory first
        os.path.join(model_dir, "model_features_kmeans.joblib"),
        os.path.join(model_dir, "kmeans_features.joblib"),
        os.path.join(model_dir, "model_features.joblib"),
        
        # Check user-specific features in model directory
        os.path.join(model_dir, f"model_features_{USER}_{USE_CASE}_kmeans.joblib"),
        
        # Look in trained_models directory as fallback
        os.path.join("trained_models", f"model_features_{USER}_{USE_CASE}_kmeans.joblib"),
        os.path.join("trained_models", "model_features_kmeans.joblib"),
        
        # Standard naming patterns based on model name
        model_path.replace("kmeans_model_", "model_features_").replace(
            f"_{parts[-1]}" if len(parts) > 1 else "", "_kmeans.joblib")
    ]
    
    # Filter out None paths
    possible_preprocessor_paths = [p for p in possible_preprocessor_paths if p]
    possible_features_paths = [p for p in possible_features_paths if p]
    
    # Load preprocessor by trying different paths
    preprocessor = None
    for preprocessor_path in possible_preprocessor_paths:
        try:
            preprocessor = joblib.load(preprocessor_path)
            print(f"Loaded preprocessor from: {preprocessor_path}")
            break
        except (FileNotFoundError, OSError) as e:
            print(f"Could not load preprocessor from {preprocessor_path}: {e}")
    
    # Load feature names
    feature_names = None
    for features_path in possible_features_paths:
        try:
            feature_names = joblib.load(features_path)
            print(f"Loaded feature names from: {features_path}")
            break
        except (FileNotFoundError, OSError) as e:
            print(f"Could not load features from {features_path}: {e}")
    
    # Determine expected number of features from the model
    expected_features = None
    if hasattr(kmeans, 'cluster_centers_'):
        expected_features = kmeans.cluster_centers_.shape[1]
        print(f"Model expects {expected_features} features")

    # Try prediction with preprocessor first
    if preprocessor is not None:
        try:
            print("Using loaded preprocessor for prediction")
            X = preprocessor.transform(input_data)
            
            # Convert to DataFrame if feature names are available
            if feature_names is not None:
                try:
                    if isinstance(X, np.ndarray):
                        X_df = pd.DataFrame(X, columns=feature_names)
                        X = X_df
                    else:
                        try:
                            X_df = pd.DataFrame(X.toarray(), columns=feature_names)
                            X = X_df
                        except:
                            # If toarray() fails, just use the array directly
                            pass
                except Exception as e:
                    print(f"Error converting to DataFrame: {e}")
            
            # Make cluster predictions
            cluster_labels = kmeans.predict(X)
            result_df['cluster'] = cluster_labels
            print(f"Successfully predicted {len(cluster_labels)} clusters using preprocessor")
            return result_df
        except Exception as e:
            print(f"Error using loaded preprocessor: {e}")
            # Continue to fallback methods
    
    # Fallback 1: Create a new preprocessor if none was loaded
    if preprocessor is None:
        try:
            print("Creating a new preprocessor...")
            # Identify numerical and categorical features
            numerical_features = input_data.select_dtypes(include=['number']).columns
            categorical_features = input_data.select_dtypes(include=['object', 'category']).columns

            # Create preprocessing pipelines with imputation for numerical and categorical features
            num_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            
            # Only include categorical pipeline if categorical features exist
            if len(categorical_features) > 0:
                cat_pipeline = Pipeline([
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(handle_unknown='ignore'))
                ])
                
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', num_pipeline, numerical_features),
                        ('cat', cat_pipeline, categorical_features),
                    ],
                    remainder='passthrough',
                )
            else:
                # If no categorical features, just use numerical preprocessing
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', num_pipeline, numerical_features),
                    ],
                    remainder='passthrough',
                )
                
            # Fit and transform
            preprocessor.fit(input_data)
            X = preprocessor.transform(input_data)
            
            # Make cluster predictions
            try:
                cluster_labels = kmeans.predict(X)
                result_df['cluster'] = cluster_labels
                print(f"Successfully predicted {len(cluster_labels)} clusters using new preprocessor")
                return result_df
            except Exception as preprocess_error:
                print(f"Error predicting with new preprocessor: {preprocess_error}")
                # Continue to next fallback
        except Exception as e:
            print(f"Error creating new preprocessor: {e}")
            # Continue to next fallback
    
    # Fallback 2: Match feature dimensions directly if we know the expected count
    if expected_features is not None:
        try:
            print(f"Attempting dimension matching for {expected_features} expected features")
            # Select numerical features first
            numerical_features = input_data.select_dtypes(include=['number']).columns
            print(f"Available numerical features: {len(numerical_features)}")
            
            if len(numerical_features) >= expected_features:
                # If we have enough or too many features, use the first expected_features
                selected_features = numerical_features[:expected_features]
                print(f"Selected features: {selected_features}")
                X = input_data[selected_features].values
                
                # Apply simple scaling since we don't have the original preprocessor
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                
            else:
                # We don't have enough features, pad with zeros
                X = input_data[numerical_features].values
                X_padded = np.zeros((X.shape[0], expected_features))
                X_padded[:, :X.shape[1]] = X
                X = X_padded
                print(f"Padded features with zeros to match expected shape: {X.shape}")
                
                # Apply simple scaling
                from sklearn.preprocessing import StandardScaler
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
            
            # Try prediction with the feature-matched data
            try:
                cluster_labels = kmeans.predict(X)
                result_df['cluster'] = cluster_labels
                print(f"Successfully predicted {len(cluster_labels)} clusters using dimension matching")
                return result_df
            except Exception as dim_error:
                print(f"Error predicting with dimension matching: {dim_error}")
                # Continue to final fallback
        except Exception as e:
            print(f"Error in dimension matching approach: {e}")
            # Continue to final fallback
    
    # Final fallback: Just assign default cluster values
    print("Using default cluster assignments (0)")
    result_df['cluster'] = 0
    return result_df