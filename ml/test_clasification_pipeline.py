"""
Integration Test for Updated Classification Pipeline

This script tests the updated classification pipeline with custom path structure:
{user_id}/trained_models/{use_case}_{target_variable}_{filename}.joblib
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add current directory to path
sys.path.append(os.path.abspath('.'))

# Try different import mechanisms
try:
    # Try importing from ml package
    from ml.train_classification import train_classification_model
    from ml.predict_model_classification import predict_classification
    print("Imported from ml package")
except ImportError:
    try:
        # Try importing directly 
        from train_classification import train_classification_model
        from predict_model_classification import predict_classification
        print("Imported directly")
    except ImportError:
        print("Failed to import required modules. Please check your Python path.")
        sys.exit(1)

def test_updated_pipeline():
    """Test the updated classification pipeline with new path structure"""
    print("\n" + "="*80)
    print("TESTING UPDATED CLASSIFICATION PIPELINE")
    print("="*80)
    
    # Setup test parameters
    filepath = "../uploads/bank-full.csv"
    target_variable = "y"
    user_id = "test_user_456"
    use_case = "customer_churn"
    
    # Make sure the file exists
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return False
    
    # Print test parameters
    print(f"Testing with file: {filepath}")
    print(f"Target variable: {target_variable}")
    print(f"User ID: {user_id}")
    print(f"Use case: {use_case}")
    
    # Train the model
    print("\n" + "-"*40)
    print("STEP 1: TRAINING MODEL")
    print("-"*40)
    
    try:
        model_path, features_path, accuracy, feature_importance = train_classification_model(
            target_variable=target_variable,
            filepath=filepath,
            user_id=user_id,
            use_case=use_case
        )
        
        print("\nTraining completed successfully!")
        print(f"Model saved to: {model_path}")
        print(f"Features saved to: {features_path}")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Verify the file was saved in the correct location
        expected_dir = os.path.join("databases", user_id, "trained_models")
        expected_file_pattern = f"{use_case}_{target_variable}"
        
        if not os.path.exists(model_path):
            print(f"Error: Model file does not exist at {model_path}")
            return False
            
        if expected_dir not in model_path:
            print(f"Error: Model not saved in expected directory. Expected: {expected_dir}, Actual: {model_path}")
            return False
            
        if expected_file_pattern not in os.path.basename(model_path):
            print(f"Error: Model filename does not follow expected pattern. Expected pattern: {expected_file_pattern}, Actual: {os.path.basename(model_path)}")
            return False
            
        print(f"Model path verification successful!")
        
        # Print feature importances
        if feature_importance:
            print("\nTop 5 feature importances:")
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            for feature, importance in sorted_features[:5]:
                print(f"  {feature}: {importance:.4f}")
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False
    
    # Test prediction
    print("\n" + "-"*40)
    print("STEP 2: TESTING PREDICTION")
    print("-"*40)
    
    try:
        # Read the bank dataset
        if filepath.endswith('.csv'):
            # Try with different separators
            try:
                df = pd.read_csv(filepath, sep=';')
            except:
                df = pd.read_csv(filepath)
        else:
            print(f"Unsupported file format: {filepath}")
            return False
        
        # Create a test sample
        test_sample = df.sample(5, random_state=42)
        
        # Save original target values
        y_true = test_sample[target_variable].copy()
        
        # Create input data without target
        X_test = test_sample.drop(columns=[target_variable])
        
        print(f"Making predictions on {len(X_test)} samples...")
        
        # Run prediction
        predictions = predict_classification(
            model_path=model_path,
            input_data=X_test,
            user_id=user_id,
            use_case=use_case
        )
        
        # Check prediction results
        pred_col = f"predicted_{target_variable}"
        if pred_col in predictions.columns:
            print("\nPrediction successful!")
            
            # Compare predictions with actual values
            print("\nPredictions vs Actual:")
            comparison = pd.DataFrame({
                'Actual': y_true.values,
                'Predicted': predictions[pred_col].values
            })
            
            if 'probability' in predictions.columns:
                comparison['Probability'] = predictions['probability'].values
                
            print(comparison)
            
            # Calculate accuracy of this sample
            correct = (comparison['Actual'] == comparison['Predicted']).sum()
            sample_accuracy = correct / len(comparison)
            print(f"\nSample accuracy: {sample_accuracy:.2f} ({correct} of {len(comparison)} correct)")
            
        else:
            print(f"Error: Prediction column '{pred_col}' not found in results.")
            print(f"Available columns: {predictions.columns.tolist()}")
            if 'error' in predictions.columns:
                print(f"Error message: {predictions['error'].iloc[0]}")
            return False
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False
    
    # Verify all required files exist
    print("\n" + "-"*40)
    print("STEP 3: VERIFYING FILE STRUCTURE")
    print("-"*40)
    
    try:
        # Get directory of model
        model_dir = os.path.dirname(model_path)
        model_basename = os.path.basename(model_path)
        
        # Check for features file
        features_basename = f"model_features_{model_basename}"
        features_path = os.path.join(model_dir, features_basename)
        
        if not os.path.exists(features_path):
            print(f"Warning: Features file not found at {features_path}")
        else:
            print(f"Features file exists at {features_path}")
            
        # Check for preprocessor file
        preprocessor_basename = f"preprocessor_{model_basename}"
        preprocessor_path = os.path.join(model_dir, preprocessor_basename)
        
        if not os.path.exists(preprocessor_path):
            print(f"Warning: Preprocessor file not found at {preprocessor_path}")
        else:
            print(f"Preprocessor file exists at {preprocessor_path}")
            
        # Verify directory structure
        user_dir = os.path.join(user_id, "trained_models")
        if not os.path.exists(user_dir):
            print(f"Error: User directory not found at {user_dir}")
        else:
            print(f"User directory exists at {user_dir}")
            
            # List files in user directory
            print("\nFiles in user directory:")
            for file in os.listdir(user_dir):
                print(f"  {file}")
        
    except Exception as e:
        print(f"Error during file verification: {str(e)}")
        import traceback
        print(traceback.format_exc())
    
    print("\n" + "="*80)
    print("ALL TESTS COMPLETED SUCCESSFULLY!")
    print("="*80)
    return True

if __name__ == "__main__":
    # Run the test
    start_time = datetime.now()
    success = test_updated_pipeline()
    end_time = datetime.now()
    
    # Print summary
    print("\nTest Summary:")
    print(f"Started: {start_time}")
    print(f"Ended: {end_time}")
    print(f"Duration: {end_time - start_time}")
    print(f"Success: {success}")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)