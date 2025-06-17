# -*- coding: utf-8 -*-
"""
Machine Learning Training Module
This module handles all ML model training logic separately from the web application
"""

import os
import time
import traceback
import uuid
from sklearn.model_selection import train_test_split

# Import the training functions from the other files
from ml.train_classification import train_classification_model
from ml.train_regression import train_regression_model
from ml.train_clustering import train_clustering_model
# Import the save_model function
from utils.ml_utils import save_model
import joblib
  
# In ml_trainer.py
def train_model_with_robust_error_handling(filepath, model_type, proposal_index=0, target_variable=None, user_id=None):
    """
    Wrapper function for train_model_from_file with robust error handling.
    
    Args:
        file_path (str): Path to the input data file
        model_type (str): Type of model to train ('classification', 'regression', 'auto')
        proposal_index (int): Index of the current proposal being processed
        target_variable (str, optional): Specific target variable to use
        user_id (str, optional): User ID for model ownership
    
    Returns:
        dict: Training statistics or error information
    """
    try:
        # If model_type is 'auto', default to 'classification'
        if model_type == 'auto':
            model_type = 'classification'
        
        if model_type not in ['classification', 'regression', 'clustering']:
            return {
                'success': False,
                'error_message': 'Invalid model type',
                'error_trace': f'Model type must be "classification" or "regression" or "clustering", got {model_type}'
            }
        
        # Setup session data
        use_case = None
        try:
            from flask import session
            if user_id is None and 'user_id' in session:
                user_id = session['user_id']
            
            if 'proposals' in session and len(session['proposals']) > proposal_index:
                proposal = session['proposals'][proposal_index]
                if 'title' in proposal:
                    use_case = proposal['title']
        except ImportError:
            # Flask not available
            pass
        
        # Generate defaults if needed
        if not user_id:
            user_id = f"auto_user_{str(uuid.uuid4())[:8]}"
        if not use_case:
            use_case = f"{model_type}_project_{str(uuid.uuid4())[:8]}"
        
        # Import save_model function
        
        
        training_stats = {
            'success': True,
            'user_id': user_id,
            'use_case': use_case,
            'model_type': model_type,
            'target_variable': target_variable
        }
        
        # Train model based on type
        if model_type == 'classification':
            # Assuming train_classification_model returns model and preprocessor
            # If not, these functions would need to be modified
            result = train_classification_model(
                target_variable, 
                filepath, 
                user_id, 
                use_case,
                threshold=0.80,
                use_f1_for_threshold=False,
                return_model=True  # Added parameter to return model objects
            )
            
            # Unpack the return values (will depend on your actual implementation)
            model_filename, features_filename, accuracy, feature_importances, model, preprocessor = result
            
            # Save metadata to database
            metrics = {'accuracy': accuracy}
            model_id = save_model(
                model=model,  # Use model directly without loading
                preprocessor=preprocessor,
                user_id=user_id,
                model_name=use_case,
                target_variable=target_variable,
                model_type=model_type,
                metrics=metrics,
                feature_importances=feature_importances
            )
            
            # Update training stats
            training_stats.update({
                'model_id': model_id,
                'model_filename': model_filename,
                'features_filename': features_filename,
                'accuracy': accuracy,
                'feature_importance': feature_importances
            })
            
        elif model_type == 'regression':
            # Similar approach for regression
            result = train_regression_model(
                target_variable, 
                filepath,
                user_id,
                use_case,
                return_model=True
            )
            
            model_filename, features_filename, r2, feature_importances, model, preprocessor = result
            
            metrics = {'r2_score': r2}
            model_id = save_model(
                model=model,
                preprocessor=preprocessor,
                user_id=user_id,
                model_name=use_case,
                target_variable=target_variable,
                model_type=model_type,
                metrics=metrics,
                feature_importances=feature_importances
            )
            
            training_stats.update({
                'model_id': model_id,
                'model_filename': model_filename,
                'features_filename': features_filename,
                'accuracy': r2,
                'feature_importance': feature_importances
            })
            
        elif model_type == 'clustering':
            # Similar approach for clustering
            result = train_clustering_model(
                filepath, 
                n_clusters=None,
                user_id=user_id,
                use_case=use_case,
                return_model=True
            )
            
            model_filename, features_filename, kmeans, model, preprocessor = result
            
            n_clusters = kmeans.n_clusters if hasattr(kmeans, 'n_clusters') else None
            metrics = {'n_clusters': n_clusters}
            model_id = save_model(
                model=model,
                preprocessor=preprocessor,
                user_id=user_id,
                model_name=use_case,
                target_variable="",
                model_type=model_type,
                metrics=metrics
            )
            
            training_stats.update({
                'model_id': model_id,
                'model_filename': model_filename,
                'features_filename': features_filename,
                'n_clusters': n_clusters,
                'feature_importance': ""
            })
            
        return training_stats
    
    except Exception as e:
        # Comprehensive error handling
        error_trace = traceback.format_exc()
        print(f"Error in train_model_with_robust_error_handling: {str(e)}")
        print(error_trace)
        
        return {
            'success': False,
            'error_message': str(e),
            'error_trace': error_trace,
            'model_type': model_type
        }