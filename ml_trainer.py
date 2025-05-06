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


def train_model_with_robust_error_handling(filepath, model_type, proposal_index=0, target_variable=None):
    """
    Wrapper function for train_model_from_file with robust error handling.
    
    Args:
        file_path (str): Path to the input data file
        model_type (str): Type of model to train ('classification', 'regression', 'auto')
        proposal_index (int): Index of the current proposal being processed
        target_variable (str, optional): Specific target variable to use
    
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
        
        # Generate a default user ID if not provided
        user_id = None
        use_case = None
        
        # Try to get user_id and use_case from Flask session for clustering
        try:
            from flask import session
            if 'user_id' in session:
                user_id = session['user_id']
            
            if 'proposals' in session and len(session['proposals']) > proposal_index:
                proposal = session['proposals'][proposal_index]
                if 'title' in proposal:
                    use_case = proposal['title']
        except ImportError:
            # Flask not available
            pass
        
        # If no user_id is found, generate a unique one
        if not user_id:
            user_id = f"auto_user_{str(uuid.uuid4())[:8]}"
        
        # If no use_case is found, create a default one
        if not use_case:
            use_case = f"{model_type}_project_{str(uuid.uuid4())[:8]}"
        
        # Train the model using the appropriate function
        if model_type == 'classification':
            model_filename, features_filename, accuracy, feature_importances = train_classification_model(
                target_variable, 
                filepath, 
                user_id, 
                use_case,
                threshold=0.80,
                use_f1_for_threshold=False 
            )
            training_stats = {
                'success': True,
                'model_filename': model_filename,
                'features_filename': features_filename,
                'accuracy': accuracy,
                'target_variable': target_variable,
                'model_type': model_type,
                'feature_importance': feature_importances,
                'user_id': user_id,
                'use_case': use_case
            }
        elif model_type == 'regression':
             model_filename, features_filename, r2, feature_importances = train_regression_model(
                 target_variable, 
                 filepath
             )
             training_stats = {
                'success': True,
                'model_filename': model_filename,
                'features_filename': features_filename,
                'accuracy': r2,
                'target_variable': target_variable,
                'model_type': model_type,
                'feature_importance': feature_importances,
                'user_id': user_id,
                'use_case': use_case
            }
        elif model_type == 'clustering':
              # Pass user_id and use_case to clustering training
              model_filename, features_filename, kmeans = train_clustering_model(
                  filepath, 
                  n_clusters=None,
                  user_id=user_id,
                  use_case=use_case
              )
              training_stats = {
                 'success': True,
                 'model_filename': model_filename,
                 'features_filename': features_filename,
                 'n_clusters': kmeans.n_clusters if hasattr(kmeans, 'n_clusters') else None,
                 'target_variable': "",
                 'model_type': model_type,
                 'feature_importance': "",
                 'user_id': user_id,
                 'use_case': use_case
             }
        else:
            return {
                'success': False,
                'error_message': 'Invalid model type',
                'error_trace': f'Model type must be "classification" or "regression", got {model_type}'
            }
        
        
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