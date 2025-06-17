import os
import json
import pandas as pd
import numpy as np
import joblib
import io
import logging
from datetime import datetime

# Import database connection from user_auth
from utils.user_auth import get_db_connection, get_model_by_id

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("predict_script")

def predict_with_model(model_id, input_data=None, input_file=None):
    """
    Load model with given ID and make predictions.
    
    Args:
        model_id (str): ID of the model to use
        input_data (pd.DataFrame, optional): Data to predict on
        input_file (str, optional): Path to CSV file with data to predict on
        
    Returns:
        pd.DataFrame: Predictions
    """
    logger.info(f"Loading model with ID: {model_id}")
    
    try:
        # Get model data from database
        model_data = get_model_by_id(model_id)
        
        if not model_data:
            logger.error(f"Model with ID {model_id} not found")
            return None
            
        logger.info(f"Successfully retrieved model: {model_data.get('name', 'Unknown')}")
        
        # Extract metadata
        if isinstance(model_data['metadata'], str):
            metadata = json.loads(model_data['metadata'])
        else:
            metadata = model_data['metadata']
        
        # Get important information from metadata - correctly navigating nested structure
        metrics = metadata.get('metrics', {})
        model_type = metrics.get('model_type', 'unknown')
        target_variable = metrics.get('target_variable', 'Unknown')
        model_path = metrics.get('model_path')
        preprocessor_path = metrics.get('preprocessor_path')
        
        logger.info(f"Model type: {model_type}")
        logger.info(f"Target variable: {target_variable}")
        logger.info(f"Model path: {model_path}")
        logger.info(f"Preprocessor path: {preprocessor_path}")
        
        # Load model
      
        if model_path and os.path.exists(model_path):
            # Load from file path in metadata
            logger.info(f"Loading model from path: {model_path}")
            model = joblib.load(model_path)
        else:
            logger.error("No valid model source found")
            return None
            
        logger.info("Model loaded successfully")
        
        # Get data for prediction
        if input_file:
            logger.info(f"Loading data from file: {input_file}")
            if input_file.endswith('.csv'):
                data = pd.read_csv(input_file)
            elif input_file.endswith(('.xlsx', '.xls')):
                data = pd.read_excel(input_file)
            else:
                logger.error("Unsupported file format. Use CSV or Excel files.")
                return None
        elif input_data is not None:
            data = input_data

            
        logger.info(f"Data shape: {data.shape}")
        logger.info(f"Data columns: {data.columns.tolist()}")
        
        # Load preprocessor if available
        preprocessor = None
        if preprocessor_path and os.path.exists(preprocessor_path):
            logger.info(f"Loading preprocessor from: {preprocessor_path}")
            try:
                preprocessor = joblib.load(preprocessor_path)
                logger.info("Preprocessor loaded successfully")
            except Exception as e:
                logger.error(f"Error loading preprocessor: {str(e)}")
                preprocessor = None
        
        # Make predictions
        try:
            # Apply preprocessing if available
            if preprocessor:
                logger.info("Applying preprocessor")
                processed_data = preprocessor.transform(data)
                
                # Convert to DataFrame if needed
                if not isinstance(processed_data, pd.DataFrame):
                    if hasattr(preprocessor, 'get_feature_names_out'):
                        feature_names = preprocessor.get_feature_names_out()
                        processed_df = pd.DataFrame(
                            processed_data if isinstance(processed_data, np.ndarray) 
                            else processed_data.toarray(),
                            columns=feature_names
                        )
                    else:
                        processed_df = pd.DataFrame(
                            processed_data if isinstance(processed_data, np.ndarray) 
                            else processed_data.toarray()
                        )
                else:
                    processed_df = processed_data
                
                logger.info("Making predictions with processed data")
                predictions = model.predict(processed_df)
            else:
                logger.info("Making predictions with original data")
                predictions = model.predict(data)
            
            # Create result DataFrame
            result_df = data.copy()
            result_df[f'predicted_{target_variable}'] = predictions
            
            # Add probabilities if available (for classification)
            if model_type == 'classification' and hasattr(model, 'predict_proba'):
                try:
                    if preprocessor:
                        probabilities = model.predict_proba(processed_df)
                    else:
                        probabilities = model.predict_proba(data)
                    
                    if hasattr(model, 'classes_'):
                        # Add probability for each class
                        for i, class_name in enumerate(model.classes_):
                            result_df[f'probability_{class_name}'] = probabilities[:, i]
                    else:
                        # Just add max probability
                        result_df['probability'] = [max(p) for p in probabilities]
                except Exception as e:
                    logger.warning(f"Couldn't calculate probabilities: {str(e)}")
            
            logger.info("Predictions completed successfully")
            return result_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None
            
    except Exception as e:
        logger.error(f"Error in prediction process: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def main():
    # Model ID to use
    model_id = "581cb452-c869-45be-9cb8-b905d489e914"
    
    # You can specify a file path here or leave as None to use sample data
    input_file = 'D:/data_sets/bank-full.csv'  # Example: "new_data.csv"
    
    # Make predictions
    predictions = predict_with_model(model_id, input_file=input_file)
    
    if predictions is not None:
        print("\nPREDICTION RESULTS:")
        print(predictions)
        
        # Save results to CSV
        output_file = f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        predictions.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
    else:
        print("\nPrediction failed. Check the logs for details.")

if __name__ == "__main__":
    main()