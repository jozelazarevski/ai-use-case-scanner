"""
Module for handling ML model training with error handling
"""
import os
import logging
import traceback
import subprocess
import sys
import importlib.util

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_model_with_robust_error_handling(model_file_path, data_path=None):
    """
    Trains a machine learning model with robust error handling.
    
    Args:
        model_file_path (str): Path to the Python file containing the model code
        data_path (str, optional): Path to the data file. Defaults to None.
        
    Returns:
        dict: Results of the training, including accuracy, errors, etc.
    """
    logger.info(f"Training model from file: {model_file_path}")
    
    # Ensure model file exists
    if not os.path.exists(model_file_path):
        error_message = f"Model file not found: {model_file_path}"
        logger.error(error_message)
        return {"status": "error", "message": error_message}
    
    try:
        # Check if trained_models directory exists, create if not
        if not os.path.exists("trained_models"):
            os.makedirs("trained_models")
            
        # Option 1: Import the module and run it
        try:
            # Get module name from file path
            module_name = os.path.basename(model_file_path).replace(".py", "")
            
            # Import the module dynamically
            spec = importlib.util.spec_from_file_location(module_name, model_file_path)
            model_module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = model_module
            spec.loader.exec_module(model_module)
            
            # Check if training was successful
            if hasattr(model_module, 'accuracy'):
                logger.info(f"Model trained successfully. Accuracy: {model_module.accuracy}")
                return {
                    "status": "success",
                    "accuracy": getattr(model_module, 'accuracy', None),
                    "model_type": getattr(model_module, 'model_type', None),
                    "feature_importance": getattr(model_module, 'feature_importance', None),
                    "message": "Model trained successfully"
                }
            else:
                # Try Option 2 if Option 1 failed to find accuracy
                logger.warning("No accuracy attribute found after importing module, trying subprocess method")
                
        except Exception as import_error:
            logger.warning(f"Error importing module: {str(import_error)}")
            logger.warning(traceback.format_exc())
            
        # Option 2: Run as a subprocess
        result = subprocess.run(
            [sys.executable, model_file_path],
            capture_output=True,
            text=True,
            check=False
        )
        
        if result.returncode != 0:
            error_message = f"Error training model: {result.stderr}"
            logger.error(error_message)
            return {"status": "error", "message": error_message, "stderr": result.stderr}
        
        logger.info(f"Model training output: {result.stdout}")
        
        # Try to parse accuracy from output
        accuracy = None
        for line in result.stdout.split('\n'):
            if 'accuracy' in line.lower():
                try:
                    accuracy = float(line.split(':')[1].strip())
                    break
                except (IndexError, ValueError):
                    pass
        
        return {
            "status": "success",
            "accuracy": accuracy,
            "message": "Model trained successfully",
            "stdout": result.stdout
        }
        
    except Exception as e:
        error_message = f"Error in training process: {str(e)}"
        error_trace = traceback.format_exc()
        logger.error(f"{error_message}\n{error_trace}")
        return {
            "status": "error",
            "message": error_message,
            "traceback": error_trace
        }

def execute_saved_model(model_path='models/price_model.py', data_path=None):
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
@app.route('/execute_model/<int:proposal_index>', methods=['GET', 'POST'])
def execute_model(proposal_index):
    """
    Execute a trained model from a specific proposal and display the results
    """
    # Check if proposal exists in session
    if 'proposals' not in session or proposal_index >= len(session['proposals']):
        flash('Proposal not found', 'error')
        return redirect(url_for('upload_file'))
    
    # Get proposal and file path
    selected_proposal = session['proposals'][proposal_index]
    file_path = session.get('file_path')
    
    if not file_path or not os.path.exists(file_path):
        flash('Data file not found', 'error')
        return redirect(url_for('upload_file'))
    
    try:
        # Get file content for code generation
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            file_content = f.read()
        
        # Generate model code if needed
        safe_target = selected_proposal.get('target_variable', '').replace(" ", "_").lower()
        model_path = f"models/{safe_target}_model.py"
        
        # Generate the model if it doesn't exist
        if not os.path.exists(model_path):
            model_path = generate_ml_with_llm(selected_proposal, file_content)
        
        # Import the execute_saved_model function (ensure it's defined in your app)
        from execute_model_function import execute_saved_model
        
        # Execute the model and get results
        results = execute_saved_model(model_path, file_path)
        
        # Handle execution errors
        if 'error' in results:
            flash(f"Error executing model: {results['error']}", 'error')
            return render_template('error.html', error=results['error'], trace=results.get('traceback', ''))
        
        # Prepare visualization data if available
        visualizations = []
        if results.get('feature_importance'):
            # Create a sorted list for bar chart
            feature_imp = results.get('feature_importance')
            if isinstance(feature_imp, dict):
                # Sort by importance value
                feature_imp = {k: v for k, v in sorted(feature_imp.items(), key=lambda item: item[1], reverse=True)[:10]}
                visualizations.append({
                    'type': 'feature_importance',
                    'data': feature_imp
                })
        
        # Add confusion matrix if available
        if results.get('confusion_matrix') is not None:
            visualizations.append({
                'type': 'confusion_matrix',
                'data': results.get('confusion_matrix')
            })
        
        return render_template('model_results.html', 
                              proposal=selected_proposal, 
                              results=results,
                              visualizations=visualizations,
                              model_path=model_path)
                              
    except Exception as e:
        error_trace = traceback.format_exc()
        flash(f"Error: {str(e)}", 'error')
        return render_template('error.html', error=str(e), trace=error_trace)