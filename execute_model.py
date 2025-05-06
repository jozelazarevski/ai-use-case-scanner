def execute_saved_model(model_path, data_path=None):
    """
    Execute a saved ML model and return the results
    
    Args:
        model_path (str): Path to the saved model Python file
        data_path (str, optional): Path to data for prediction. If None, uses training data.
        
    Returns:
        dict: Results including accuracy, feature importance, and other metrics
    """
    import os
    import importlib.util
    import sys
    import traceback
    import pandas as pd
    import numpy as np
    
    try:
        # Get the model directory and filename
        model_dir = os.path.dirname(model_path)
        model_filename = os.path.basename(model_path)
        model_name = os.path.splitext(model_filename)[0]
        
        # Create a module spec and import the model module
        spec = importlib.util.spec_from_file_location(model_name, model_path)
        model_module = importlib.util.module_from_spec(spec)
        sys.modules[model_name] = model_module
        spec.loader.exec_module(model_module)
        
        # Check if the model has an evaluate function
        if hasattr(model_module, 'evaluate_model'):
            results = model_module.evaluate_model(data_path)
            return results
        
        # If no evaluate function, try to extract variables from the module
        results = {
            'accuracy': getattr(model_module, 'accuracy', None),
            'model_type': getattr(model_module, 'model_type', 'Unknown'),
            'feature_importance': getattr(model_module, 'feature_importance', None),
            'confusion_matrix': getattr(model_module, 'confusion_matrix', None),
            'classification_report': getattr(model_module, 'classification_report', None),
            'model_summary': getattr(model_module, 'model_summary', 'No summary available')
        }
        
        # If we have a trained model object, try to extract more info
        if hasattr(model_module, 'model'):
            model = model_module.model
            results['model_object'] = type(model).__name__
            
            # Try to get feature importance if possible
            if hasattr(model, 'feature_importances_') and results['feature_importance'] is None:
                features = getattr(model_module, 'features', ['Feature_' + str(i) for i in range(len(model.feature_importances_))])
                importance = dict(zip(features, model.feature_importances_))
                results['feature_importance'] = {k: float(v) for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
            
        return results
        
    except Exception as e:
        error_trace = traceback.format_exc()
        return {
            'error': str(e),
            'traceback': error_trace,
            'status': 'failed'
        }
