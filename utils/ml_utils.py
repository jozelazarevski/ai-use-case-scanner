# -*- coding: utf-8 -*-
"""
ML Utilities Module
Common utility functions for machine learning models
"""

import io
import pickle
import joblib
import uuid
from datetime import datetime
import json




def save_model(model, preprocessor, user_id, model_name, target_variable, model_type, 
               metrics, feature_importances=None, feature_names=None, description=None, additional_metadata=None):
    """
    Save model to disk and store metadata and paths in the database
    """
    import os
    import joblib
    import uuid
    import json
    import traceback
    from datetime import datetime
    import sklearn
    import logging
    
    # Setup logging
    logger = logging.getLogger(__name__)
    
    try:
        # Import save_user_model from user_auth
        from utils.user_auth import save_user_model
        from flask import current_app
        
        # Create unique model ID
        model_id = str(uuid.uuid4())
        
        # Create directory for model files - use RELATIVE paths only
        models_dir = "models"  # Use relative path
        user_model_dir = os.path.join(models_dir, user_id)
        os.makedirs(user_model_dir, exist_ok=True)
        
        # Save model to disk
        model_filename = f"{model_id}_model.joblib"
        model_path = os.path.join(user_model_dir, model_filename)
        
        # Convert backslashes to forward slashes for consistency across platforms
        model_path = model_path.replace('\\', '/')
        
        joblib.dump(model, model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Save preprocessor to disk if available
        preprocessor_path = None
        if preprocessor is not None:
            preprocessor_filename = f"{model_id}_preprocessor.joblib"
            preprocessor_path = os.path.join(user_model_dir, preprocessor_filename)
            
            # Convert backslashes to forward slashes
            preprocessor_path = preprocessor_path.replace('\\', '/')
            
            joblib.dump(preprocessor, preprocessor_path)
            logger.info(f"Preprocessor saved to {preprocessor_path}")
        
        # Save feature names if available
        features_path = None
        if feature_names is not None:
            features_filename = f"{model_id}_features.joblib"
            features_path = os.path.join(user_model_dir, features_filename)
            
            # Convert backslashes to forward slashes
            features_path = features_path.replace('\\', '/')
            
            joblib.dump(feature_names, features_path)
            logger.info(f"Feature names saved to {features_path}")
        
        # ... rest of the function is the same ...
        
        # Create metadata with the normalized paths
        metadata = {
            'target_variable': target_variable,
            'model_type': model_type,
            'metrics': metrics,
            'accuracy': metrics.get('accuracy', metrics.get('r2_score', 0.0)),
            'title': model_name,
            'description': description,
            'created_at': datetime.now().isoformat(),
            'model_info': {
                'type': type(model).__name__,
                'params': getattr(model, 'get_params', lambda: {})()
            },
            'features': {
                'trained_features': [str(f) for f in feature_names] if feature_names is not None else [],
                'importance': {str(k): float(v) for k, v in feature_importances.items()} if feature_importances else {},
                'features_path': features_path
            },
            'model_path': model_path,
            'preprocessor_path': preprocessor_path
        }
        
        # Add any additional metadata
        if additional_metadata:
            metadata.update(additional_metadata)
        
        # Save to database - note we're NOT serializing the model
        try:
            db_model_id = save_user_model(user_id, model_name, None, metadata)
            if db_model_id:
                logger.info(f"Model metadata saved to database with ID: {db_model_id}")
                return db_model_id
            else:
                logger.error("save_user_model returned None")
                return None
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            logger.error(traceback.format_exc())
            return None
    
    except Exception as e:
        logger.error(f"Unexpected error in save_model: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def predict_with_preprocessor(model_id, input_filepath, user_id):
    """
    Make predictions using a model identified by model_id on data from input_filepath.
    Handles both file-based and database-stored models with robust path resolution.
    """
    import os
    import numpy as np
    import joblib
    import pickle
    import io
    import traceback
    import logging
    from ml.read_file import read_data_flexible
    
    logger = logging.getLogger(__name__)
    
    try:
        # Get model data from database
        from utils.user_auth import get_db_connection
        
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get model by ID and user
                cursor.execute(
                    """
                    SELECT metadata, model_data
                    FROM models 
                    WHERE id = %s AND user_id = %s
                    """,
                    (model_id, user_id)
                )
                model_data = cursor.fetchone()
                
                # If not found, try without user_id filter (admin access)
                if not model_data:
                    cursor.execute(
                        """
                        SELECT metadata, model_data
                        FROM models 
                        WHERE id = %s
                        """,
                        (model_id,)
                    )
                    model_data = cursor.fetchone()
                
        if not model_data:
            logger.error(f"Model {model_id} not found")
            return None
            
        # Extract metadata
        metadata = model_data.get('metadata', {})
        model_type = metadata.get('model_type', 'unknown')
        target_variable = metadata.get('target_variable', 'Unknown')
        
        # Get file paths
        model_path = metadata.get('model_path')
        preprocessor_path = metadata.get('preprocessor_path')
        
        # Normalize paths - replace backslashes and handle both absolute and relative paths
        if model_path:
            # Replace backslashes with forward slashes
            model_path = model_path.replace('\\', '/')
            
            # If path starts with a drive letter (like D:), use just the relative part
            if ':' in model_path:
                parts = model_path.split('/')
                # Find 'models' in the path and take everything from there
                try:
                    models_index = parts.index('models')
                    model_path = '/'.join(parts[models_index:])
                except ValueError:
                    # If 'models' not found, just use the filename
                    model_path = parts[-1]
        
        # Do the same for preprocessor path
        if preprocessor_path:
            preprocessor_path = preprocessor_path.replace('\\', '/')
            if ':' in preprocessor_path:
                parts = preprocessor_path.split('/')
                try:
                    models_index = parts.index('models') 
                    preprocessor_path = '/'.join(parts[models_index:])
                except ValueError:
                    preprocessor_path = parts[-1]
        
        # Initialize model and preprocessor
        model = None
        preprocessor = None
        
        # Try to load model from path if available
        if model_path:
            # Check if path exists directly
            if os.path.exists(model_path):
                try:
                    logger.info(f"Loading model from path: {model_path}")
                    model = joblib.load(model_path)
                except Exception as e:
                    logger.error(f"Error loading model from direct path: {str(e)}")
            else:
                # Try to resolve as a relative path from current directory
                app_root = os.getcwd()
                full_model_path = os.path.join(app_root, model_path)
                
                if os.path.exists(full_model_path):
                    try:
                        logger.info(f"Loading model from resolved path: {full_model_path}")
                        model = joblib.load(full_model_path)
                    except Exception as e:
                        logger.error(f"Error loading model from resolved path: {str(e)}")
                else:
                    logger.error(f"Model file not found at either {model_path} or {full_model_path}")
        
        # If loading from path failed, try from binary data in DB
        if model is None:
            logger.info("Trying to load model from database binary data")
            
            # Try 'model' column first
            if model_data.get('model') is not None:
                try:
                    logger.info("Loading model from 'model' column")
                    model = pickle.loads(model_data['model'])
                except Exception as e:
                    logger.error(f"Error loading from 'model' column: {str(e)}")
            
            # Try 'model_data' column next
            if model is None and model_data.get('model_data') is not None:
                try:
                    logger.info("Loading model from 'model_data' column")
                    binary_data = model_data['model_data']
                    
                    # Try direct pickle load
                    try:
                        model = pickle.loads(binary_data)
                    except:
                        # Try loading as buffer
                        try:
                            buffer = io.BytesIO(binary_data)
                            model = joblib.load(buffer)
                        except:
                            # Try loading as nested structure
                            try:
                                data_dict = pickle.loads(binary_data)
                                if isinstance(data_dict, dict) and 'model' in data_dict:
                                    model_buffer = io.BytesIO(data_dict['model'])
                                    model = joblib.load(model_buffer)
                                    
                                    # Also try to load preprocessor
                                    if 'preprocessor' in data_dict and data_dict['preprocessor']:
                                        preprocessor_buffer = io.BytesIO(data_dict['preprocessor'])
                                        preprocessor = joblib.load(preprocessor_buffer)
                            except Exception as e:
                                logger.error(f"Error with nested structure: {str(e)}")
                except Exception as e:
                    logger.error(f"Error loading from 'model_data' column: {str(e)}")
        
        # If we still don't have a model, we can't continue
        if model is None:
            logger.error("Could not load model from any source")
            return None
        
        # Try to load preprocessor from path if available and not already loaded
        if preprocessor is None and preprocessor_path:
            # Check if preprocessor path exists directly
            if os.path.exists(preprocessor_path):
                try:
                    logger.info(f"Loading preprocessor from path: {preprocessor_path}")
                    preprocessor = joblib.load(preprocessor_path)
                except Exception as e:
                    logger.warning(f"Error loading preprocessor from direct path: {str(e)}")
            else:
                # Try to resolve as a relative path
                app_root = os.getcwd()
                full_preprocessor_path = os.path.join(app_root, preprocessor_path)
                
                if os.path.exists(full_preprocessor_path):
                    try:
                        logger.info(f"Loading preprocessor from resolved path: {full_preprocessor_path}")
                        preprocessor = joblib.load(full_preprocessor_path)
                    except Exception as e:
                        logger.warning(f"Error loading preprocessor from resolved path: {str(e)}")
                else:
                    logger.warning(f"Preprocessor file not found at either {preprocessor_path} or {full_preprocessor_path}")
        
        # Get feature names from metadata
        feature_names = []
        if 'features' in metadata and 'trained_features' in metadata['features']:
            feature_names = metadata['features']['trained_features']
        
        # Read input data
        input_df = read_data_flexible(input_filepath)
        if input_df is None or input_df.empty:
            logger.error(f"Failed to read input data from {input_filepath}")
            return None
        
        # Process input data with preprocessor if available
        if preprocessor is not None:
            try:
                logger.info("Applying preprocessor to data")
                X_processed = preprocessor.transform(input_df)
            except Exception as e:
                logger.error(f"Error applying preprocessor: {str(e)}")
                preprocessor = None
        
        # If preprocessor failed or isn't available, use feature selection
        if preprocessor is None:
            logger.info("Using direct feature selection (no preprocessor)")
            if feature_names:
                # Filter to only include available features
                available_features = [f for f in feature_names if f in input_df.columns]
                if available_features:
                    logger.info(f"Using {len(available_features)} available features from feature list")
                    X_processed = input_df[available_features].values
                else:
                    # Use numeric columns if no matching features
                    numeric_cols = input_df.select_dtypes(include=['number']).columns.tolist()
                    logger.warning(f"No matching features found. Using {len(numeric_cols)} numeric columns")
                    X_processed = input_df[numeric_cols].values
            else:
                # No feature names available
                numeric_cols = input_df.select_dtypes(include=['number']).columns.tolist()
                logger.warning(f"No feature names available. Using {len(numeric_cols)} numeric columns")
                X_processed = input_df[numeric_cols].values
        
        # Make predictions based on model type
        logger.info(f"Making predictions with model type: {model_type}")
        
        if model_type == 'classification':
            predictions = model.predict(X_processed)
            probabilities = None
            
            # Get probabilities if available
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X_processed)
                    if probabilities.shape[1] == 2:  # Binary classification
                        probabilities = probabilities[:, 1]
                    else:  # Multi-class classification
                        probabilities = np.max(probabilities, axis=1)
                except Exception as prob_err:
                    logger.warning(f"Could not get probabilities: {str(prob_err)}")
            
            # Create results DataFrame
            result_df = input_df.copy()
            result_df[f'predicted_{target_variable}'] = predictions
            if probabilities is not None:
                result_df['probability'] = probabilities
                
        elif model_type == 'regression':
            predictions = model.predict(X_processed)
            result_df = input_df.copy()
            result_df[f'predicted_{target_variable}'] = predictions
            
        elif model_type == 'clustering':
            cluster_labels = model.predict(X_processed)
            result_df = input_df.copy()
            result_df['cluster'] = cluster_labels
            
            # Add distances to cluster centers if available
            if hasattr(model, 'transform'):
                try:
                    distances = model.transform(X_processed)
                    # Add distance to assigned cluster
                    for i, cluster_idx in enumerate(cluster_labels):
                        result_df.loc[result_df.index[i], f'distance_to_cluster'] = distances[i, cluster_idx]
                except Exception as e:
                    logger.warning(f"Could not compute distances to clusters: {str(e)}")
            
        else:
            logger.error(f"Unsupported model type: {model_type}")
            return None
            
        logger.info(f"Successfully generated predictions for {len(result_df)} rows")
        return result_df
        
    except Exception as e:
        logger.error(f"Direct prediction failed: {str(e)}")
        traceback_str = traceback.format_exc()
        logger.error(traceback_str)
        return None

def load_model_with_preprocessor(model_id, user_id=None):
    """
    Load a model and its associated preprocessor from the database
    
    Args:
        model_id (str): Database ID of the model
        user_id (str, optional): User ID for permission check
        
    Returns:
        tuple: (model_object, preprocessor_object, metadata)
    """
    import joblib
    import io
    import pickle
    import logging
    import traceback
    
    logger = logging.getLogger(__name__)
    
    try:
        # Import the database connection function
        from utils.user_auth import get_db_connection
        
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Query with optional user_id check
                if user_id:
                    cursor.execute(
                        """
                        SELECT id, name, model_data, metadata
                        FROM models 
                        WHERE id = %s AND user_id = %s
                        """,
                        (model_id, user_id)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id, name, model_data, metadata
                        FROM models 
                        WHERE id = %s
                        """,
                        (model_id,)
                    )
                
                model_data = cursor.fetchone()
                
                if not model_data or not model_data.get('model_data'):
                    logger.warning(f"Model with ID {model_id} not found or has no data")
                    return None, None, None
                
                logger.info(f"Found model data with size: {len(model_data['model_data'])} bytes")
                
                # Try to deserialize as combined data package
                try:
                    combined_data = pickle.loads(model_data['model_data'])
                    
                    # Check if it's our expected format
                    if isinstance(combined_data, dict) and 'model' in combined_data and 'preprocessor' in combined_data:
                        # Load model
                        model_buffer = io.BytesIO(combined_data['model'])
                        model_object = joblib.load(model_buffer)
                        
                        # Load preprocessor if available
                        preprocessor_object = None
                        if combined_data['preprocessor']:
                            preprocessor_buffer = io.BytesIO(combined_data['preprocessor'])
                            preprocessor_object = joblib.load(preprocessor_buffer)
                            logger.info("Successfully loaded preprocessor")
                        else:
                            logger.warning("No preprocessor data found in the combined package")
                        
                        return model_object, preprocessor_object, model_data['metadata']
                    else:
                        # Not in combined format, try loading as direct model
                        logger.warning("Model data not in expected combined format, trying direct load")
                        buffer = io.BytesIO(model_data['model_data'])
                        model_object = joblib.load(buffer)
                        return model_object, None, model_data['metadata']
                
                except Exception as e:
                    logger.error(f"Error deserializing model: {str(e)}")
                    logger.error(traceback.format_exc())
                    return None, None, None
    
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None
 
