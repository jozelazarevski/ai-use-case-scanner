"""
Comprehensive Integration Test for Classification Pipeline

This script provides a robust test for the machine learning classification pipeline, 
covering model training, preprocessing, and prediction across different scenarios.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import traceback

# Extend Python path to ensure module imports work correctly
sys.path.extend([
    os.path.abspath('.'),
    os.path.abspath('..'),
    os.path.dirname(os.path.abspath(__file__))
])

# Direct imports with fallback mechanism
def safe_import(module_paths, function_name):
    """
    Safely import a function from multiple possible module paths
    
    Args:
        module_paths (list): List of possible module paths to import from
        function_name (str): Name of the function to import
    
    Returns:
        function: Imported function
    """
    for module_path in module_paths:
        try:
            # Try importing the module and get the function
            if module_path:
                module = __import__(module_path, fromlist=[function_name])
            else:
                module = __import__(function_name)
            
            # Get the function from the module
            func = getattr(module, function_name)
            print(f"Successfully imported {function_name} from {module_path}")
            return func
        except (ImportError, AttributeError) as e:
            print(f"Failed to import {function_name} from {module_path}: {e}")
            continue
    
    # If all import attempts fail
    raise ImportError(f"Could not import function: {function_name}")

# Import required functions
try:
    # Possible import paths for each function
    train_paths = [
        'ml.train_classification',
        'train_classification',
        '.train_classification'
    ]
    predict_paths = [
        'ml.predict_model_classification',
        'predict_model_classification',
        '.predict_model_classification'
    ]
    
    # Import functions with fallback
    train_classification_model = safe_import(train_paths, 'train_classification_model')
    predict_classification = safe_import(predict_paths, 'predict_classification')

except ImportError as e:
    print(f"Critical Import Error: {e}")
    traceback.print_exc()
    sys.exit(1)

class ClassificationPipelineTest:
    """
    Comprehensive test suite for classification pipeline
    """
    def __init__(self, 
                 filepath, 
                 target_variable, 
                 user_id='test_user', 
                 use_case='default_classification'):
        """
        Initialize test parameters
        
        Args:
            filepath (str): Path to the input data file
            target_variable (str): Name of the target variable column
            user_id (str, optional): User identifier for model organization
            use_case (str, optional): Specific use case identifier
        """
        self.filepath = filepath
        self.target_variable = target_variable
        self.user_id = user_id
        self.use_case = use_case
        
        # Test tracking
        self.test_results = {
            'training_successful': False,
            'prediction_successful': False,
            'file_verification_successful': False
        }
        
        # Paths to be populated during testing
        self.model_path = None
        self.features_path = None
        self.preprocessor_path = None
    
    def load_data(self):
        """
        Load data from the specified filepath
        
        Returns:
            pd.DataFrame: Loaded dataset
        """
        try:
            # Try different separators
            try:
                df = pd.read_csv(self.filepath, sep=';')
            except:
                df = pd.read_csv(self.filepath)
            
            # Basic data validation
            if df.empty:
                raise ValueError("Loaded dataset is empty")
            
            if self.target_variable not in df.columns:
                raise ValueError(f"Target variable '{self.target_variable}' not found in dataset")
            
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            traceback.print_exc()
            return None
    
    def train_model(self):
        """
        Train classification model and verify training results
        
        Returns:
            bool: Training success status
        """
        print("\n" + "="*40)
        print("STEP 1: MODEL TRAINING")
        print("="*40)
        
        try:
            # Train the model
            self.model_path, self.features_path, accuracy, feature_importance = train_classification_model(
                target_variable=self.target_variable,
                filepath=self.filepath,
                user_id=self.user_id,
                use_case=self.use_case
            )
            
            # Validation checks
            if not self.model_path or not os.path.exists(self.model_path):
                print("Error: Model file was not created successfully")
                return False
            
            print(f"\nModel Training Successful!")
            print(f"Model saved to: {self.model_path}")
            print(f"Features saved to: {self.features_path}")
            print(f"Model Accuracy: {accuracy:.4f}")
            
            # Display top feature importances
            if feature_importance:
                print("\nTop 5 Feature Importances:")
                sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                for feature, importance in sorted_features[:5]:
                    print(f"  {feature}: {importance:.4f}")
            
            self.test_results['training_successful'] = True
            return True
        
        except Exception as e:
            print(f"Model Training Error: {e}")
            traceback.print_exc()
            return False
    
    def test_prediction(self):
        """
        Test model prediction capabilities
        
        Returns:
            bool: Prediction success status
        """
        print("\n" + "="*40)
        print("STEP 2: MODEL PREDICTION")
        print("="*40)
        
        try:
            # Load dataset
            df = self.load_data()
            if df is None:
                print("Failed to load data for prediction testing")
                return False
            
            # Create test sample
            test_sample = df.sample(5, random_state=42)
            
            # Preserve true target values
            y_true = test_sample[self.target_variable].copy()
            
            # Prepare input data without target
            X_test = test_sample.drop(columns=[self.target_variable])
            
            print(f"Making predictions on {len(X_test)} samples...")
            
            # Run prediction with explicit target variable
            predictions = predict_classification(
                model_path=self.model_path,
                input_data=X_test,
                user_id=self.user_id,
                use_case=self.use_case,
                target_variable=self.target_variable
            )
            
            # Possible prediction column names to check
            pred_columns = [
                f'predicted_{self.target_variable}',
                f'predicted_{self.use_case}',
                'predicted_y'
            ]
            
            # Find the first existing prediction column
            pred_col = next((col for col in pred_columns if col in predictions.columns), None)
            
            if not pred_col:
                print("No prediction column found!")
                print("Available columns:", predictions.columns.tolist())
                return False
            
            # Create comparison DataFrame
            comparison = pd.DataFrame({
                'Actual': y_true.values,
                'Predicted': predictions[pred_col].values
            })
            
            # Add probability if available
            if 'probability' in predictions.columns:
                comparison['Probability'] = predictions['probability'].values
            
            print("\nPrediction Comparison:")
            print(comparison)
            
            # Calculate sample accuracy
            correct = (comparison['Actual'] == comparison['Predicted']).sum()
            sample_accuracy = correct / len(comparison)
            
            print(f"\nSample Prediction Accuracy: {sample_accuracy:.2%} "
                  f"({correct} of {len(comparison)} correct)")
            
            self.test_results['prediction_successful'] = True
            return True
        
        except Exception as e:
            print(f"Prediction Testing Error: {e}")
            traceback.print_exc()
            return False
    
    def verify_file_structure(self):
        """
        Verify the file structure and generated artifacts
        
        Returns:
            bool: File structure verification status
        """
        print("\n" + "="*40)
        print("STEP 3: FILE STRUCTURE VERIFICATION")
        print("="*40)
        
        try:
            # Verify model file exists
            if not os.path.exists(self.model_path):
                print(f"Error: Model file missing - {self.model_path}")
                return False
            
            # Verify features file
            features_filename = os.path.basename(self.model_path).replace(".joblib", "_features.joblib")
            features_dir = os.path.dirname(self.model_path)
            self.features_path = os.path.join(features_dir, features_filename)
            
            if not os.path.exists(self.features_path):
                print(f"Warning: Features file not found - {self.features_path}")
            
            # Verify preprocessor file
            preprocessor_filename = os.path.basename(self.model_path).replace(".joblib", "_preprocessor.joblib")
            preprocessor_dir = os.path.dirname(self.model_path)
            self.preprocessor_path = os.path.join(preprocessor_dir, preprocessor_filename)
            
            if not os.path.exists(self.preprocessor_path):
                print(f"Warning: Preprocessor file not found - {self.preprocessor_path}")
            
            # Verify user directory structure
            user_dir = os.path.join("databases", "trained_models", self.user_id)
            if not os.path.exists(user_dir):
                print(f"Error: User directory not found - {user_dir}")
                return False
            
            print("\nFile Structure Details:")
            print(f"Model Path:         {self.model_path}")
            print(f"Features Path:      {self.features_path}")
            print(f"Preprocessor Path:  {self.preprocessor_path}")
            print(f"User Directory:     {user_dir}")
            
            self.test_results['file_verification_successful'] = True
            return True
        
        except Exception as e:
            print(f"File Structure Verification Error: {e}")
            traceback.print_exc()
            return False
    
    def run_full_test(self):
        """
        Execute full test suite
        
        Returns:
            bool: Overall test success status
        """
        start_time = datetime.now()
        
        print("\n" + "="*60)
        print("CLASSIFICATION PIPELINE INTEGRATION TEST")
        print("="*60)
        
        # Run test steps
        training_result = self.train_model()
        prediction_result = self.test_prediction() if training_result else False
        file_verification_result = self.verify_file_structure() if prediction_result else False
        
        # Compute overall test status
        overall_success = (
            training_result and 
            prediction_result and 
            file_verification_result
        )
        
        end_time = datetime.now()
        
        # Print test summary
        print("\n" + "="*40)
        print("TEST SUMMARY")
        print("="*40)
        print(f"Started:     {start_time}")
        print(f"Completed:   {end_time}")
        print(f"Duration:    {end_time - start_time}")
        print(f"Overall Test Result: {'PASSED' if overall_success else 'FAILED'}")
        
        # Detailed test results
        print("\nTest Results:")
        for key, value in self.test_results.items():
            print(f"  {key.replace('_', ' ').title()}: "
                  f"{'✓ Passed' if value else '✗ Failed'}")
        
        return overall_success

def main():
    """
    Main function to run the classification pipeline test
    """
    # Configuration
    filepath = "uploads/bank-full.csv"
    target_variable = "y"
    user_id = "test_user_456"
    use_case = "customer_churn"
    
    # Verify input file exists
    if not os.path.exists(filepath):
        print(f"Error: Input file not found at {filepath}")
        sys.exit(1)
    
    # Initialize and run test
    test_runner = ClassificationPipelineTest(
        filepath=filepath,
        target_variable=target_variable,
        user_id=user_id,
        use_case=use_case
    )
    
    # Run test and exit with appropriate status
    test_success = test_runner.run_full_test()
    sys.exit(0 if test_success else 1)

if __name__ == "__main__":
    main()