import os
import pandas as pd
import numpy as np
import json
import joblib
import time
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import warnings

# Ignore convergence warnings for cleaner output
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning) # Ignore potential LightGBM warnings

# --- Configuration ---
# Update this path according to your file location
DATA_FILE_PATH = "uploads/bank-full.csv"
MODEL_DIR = "trained_models" # Changed from trained_LLM_models as LLM is not used here
TIMESTAMP = int(time.time() * 1000)
METRICS_FILE = os.path.join(MODEL_DIR, f"metrics/bank_marketing_classification_{TIMESTAMP}_metrics.json")
MODEL_FILE = os.path.join(MODEL_DIR, f"bank_marketing_classifier_{TIMESTAMP}.joblib")
REPORT_FILE = os.path.join(MODEL_DIR, f"report/bank_marketing_classification_{TIMESTAMP}_report.txt")
PREDICTION_SCRIPT_FILE = os.path.join(MODEL_DIR, f"predict_bank_marketing_{TIMESTAMP}.py")

TARGET_COLUMN = 'y'
MODEL_TYPE = 'classification'

# --- Main Function ---

def train_bank_marketing_model(data_path=DATA_FILE_PATH):
    """
    Trains multiple classification models on the bank marketing dataset,
    selects the best one based on accuracy, saves the model, metrics,
    and generates a prediction script.

    Args:
        data_path (str): Path to the CSV data file.

    Returns:
        dict: A dictionary containing:
              'accuracy': Accuracy score of the best model on the test set.
              'model_path': Path to the saved best model file.
              'metrics_path': Path to the saved metrics JSON file.
              'report_path': Path to the saved report file.
              'prediction_script_path': Path to the generated prediction script.
              'best_model_name': Name of the best performing algorithm.
    """
    print(f"Starting model training process for {data_path}...")
    print(f"Model type: {MODEL_TYPE}")

    # Create directories if they don't exist
    os.makedirs(os.path.join(MODEL_DIR, "metrics"), exist_ok=True)
    os.makedirs(os.path.join(MODEL_DIR, "report"), exist_ok=True)

    # --- 1. Load Data ---
    try:
        # Try reading with UTF-8 first, fallback to latin1 if encoding issues arise
        try:
            df = pd.read_csv(data_path, sep=';')
        except UnicodeDecodeError:
            print("UTF-8 decoding failed, trying latin1 encoding...")
            df = pd.read_csv(data_path, sep=';', encoding='latin1')
        print(f"Data loaded successfully from {data_path}. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

    if TARGET_COLUMN not in df.columns:
        print(f"Error: Target column '{TARGET_COLUMN}' not found in the dataset.")
        return None

    # --- 2. Initial Data Preparation ---
    # Map target variable 'yes'/'no' to 1/0
    try:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].map({'yes': 1, 'no': 0}).astype(int)
    except Exception as e:
        print(f"Error mapping target variable '{TARGET_COLUMN}': {e}")
        print(f"Unique values in target column: {df[TARGET_COLUMN].unique()}")
        return None


    # Identify Features
    FEATURE_COLUMNS = [col for col in df.columns if col != TARGET_COLUMN]
    NUMERICAL_FEATURES = df[FEATURE_COLUMNS].select_dtypes(include=np.number).columns.tolist()
    CATEGORICAL_FEATURES = df[FEATURE_COLUMNS].select_dtypes(include='object').columns.tolist()

    print(f"Identified Features: {len(FEATURE_COLUMNS)}")
    print(f"Numerical Features: {NUMERICAL_FEATURES}")
    print(f"Categorical Features: {CATEGORICAL_FEATURES}")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # --- 3. Train/Test Split ---
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Data split into Train ({X_train.shape[0]} samples) and Test ({X_test.shape[0]} samples)")
    except Exception as e:
        print(f"Error during train/test split: {e}")
        return None

    # --- 4. Preprocessing Pipeline ---
    # Create transformers for numerical and categorical features
    numerical_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Use sparse_output=False for compatibility with more estimators if needed, though True is often more memory efficient
    ])

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='passthrough' # Keep any columns not specified (though all should be covered)
    )

    # --- 5. Model Selection and Training ---
    # Define models to evaluate
    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000, solver='liblinear'), # liblinear often good for smaller datasets
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced'), # Balanced class weight helpful for imbalanced data
        "LightGBM": LGBMClassifier(random_state=42, class_weight='balanced') # LightGBM often performs well
        # Add other models like SVC if desired, but SVC can be slow on larger datasets
        # "SVM": SVC(probability=True, random_state=42, class_weight='balanced')
    }

    best_model_name = None
    best_model_pipeline = None
    best_accuracy = -1.0
    results = {}

    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        # Create the full pipeline: Preprocessor -> Model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])

        try:
            # Train the model
            start_time = time.time()
            pipeline.fit(X_train, y_train)
            end_time = time.time()
            training_time = end_time - start_time
            print(f"Training {name} completed in {training_time:.2f} seconds.")

            # Predict on the test set
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else None

            # Evaluate the model
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=['no', 'yes'], output_dict=True)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            cm = confusion_matrix(y_test, y_pred)

            results[name] = {
                'accuracy': accuracy,
                'classification_report': report,
                'auc': auc,
                'confusion_matrix': cm.tolist(), # Convert numpy array to list for JSON serialization
                'training_time_seconds': training_time
            }

            print(f"{name} Test Accuracy: {accuracy:.4f}")
            print(f"{name} Test AUC: {auc:.4f}" if auc is not None else f"{name}: AUC not available")
            # print(classification_report(y_test, y_pred, target_names=['no', 'yes'])) # Print report summary

            # Check if this model is the best so far based on accuracy
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                best_model_pipeline = pipeline
                print(f"*** New Best Model Found: {best_model_name} (Accuracy: {best_accuracy:.4f}) ***")

        except Exception as e:
            print(f"Error training/evaluating {name}: {e}")
            results[name] = {'error': str(e)}

    if best_model_pipeline is None:
        print("Error: No model could be trained successfully.")
        return None

    print(f"\n--- Best Model Selection ---")
    print(f"Best performing model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

    # --- 6. Save Best Model, Metrics, and Report ---
    try:
        # Save the entire pipeline (preprocessor + best model)
        joblib.dump(best_model_pipeline, MODEL_FILE)
        print(f"Best model pipeline saved successfully to: {MODEL_FILE}")
    except Exception as e:
        print(f"Error saving the model: {e}")
        # Continue to save metrics/report if possible

    # Save detailed metrics
    all_metrics = {
        'best_model': best_model_name,
        'best_model_accuracy_on_test': best_accuracy,
        'model_comparison_results': results,
        'data_file': data_path,
        'target_column': TARGET_COLUMN,
        'feature_columns': FEATURE_COLUMNS,
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'model_type': MODEL_TYPE,
        'timestamp': TIMESTAMP
    }
    try:
        with open(METRICS_FILE, 'w') as f:
            json.dump(all_metrics, f, indent=4)
        print(f"Detailed metrics saved successfully to: {METRICS_FILE}")
    except Exception as e:
        print(f"Error saving metrics JSON: {e}")

    # Save text report
    try:
        best_model_results = results.get(best_model_name, {})
        report_str = f"""
        Model Training Report
        ---------------------
        Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(TIMESTAMP/1000))}
        Data Source: {data_path}
        Target Variable: {TARGET_COLUMN}
        Model Type: {MODEL_TYPE}

        Preprocessing:
        - Numerical Features Scaled: StandardScaler
        - Categorical Features Encoded: OneHotEncoder (handle_unknown='ignore')
        - Features Used: {len(FEATURE_COLUMNS)} ({len(NUMERICAL_FEATURES)} numerical, {len(CATEGORICAL_FEATURES)} categorical)

        Models Evaluated: {list(models.keys())}

        Best Model Selected: {best_model_name}
        ---------------------
        Performance on Test Set (Size: {X_test.shape[0]}):
        Accuracy: {best_accuracy:.4f}
        AUC: {best_model_results.get('auc', 'N/A'):.4f}

        Classification Report:
        {classification_report(y_test, best_model_pipeline.predict(X_test), target_names=['no', 'yes'])}

        Confusion Matrix:
        {best_model_results.get('confusion_matrix', 'N/A')}
        (Rows: Actual, Columns: Predicted. Classes: [0: no, 1: yes])

        Training Time for Best Model: {best_model_results.get('training_time_seconds', 'N/A'):.2f} seconds

        Saved Artifacts:
        - Model Pipeline: {MODEL_FILE}
        - Metrics JSON: {METRICS_FILE}
        - Report File: {REPORT_FILE}
        - Prediction Script: {PREDICTION_SCRIPT_FILE}
        """
        with open(REPORT_FILE, 'w') as f:
            f.write(report_str)
        print(f"Summary report saved successfully to: {REPORT_FILE}")
    except Exception as e:
        print(f"Error saving report file: {e}")


    # --- 7. Generate Prediction Script ---
    prediction_script_content = f"""
import pandas as pd
import joblib
import os
import argparse

# --- Configuration ---
# This path should point to the saved model pipeline file from the training script
MODEL_FILE = r"{os.path.abspath(MODEL_FILE)}" # Use absolute path for robustness
TARGET_COLUMN = '{TARGET_COLUMN}' # Needed potentially for output formatting

# --- Prediction Function ---
def predict_term_deposit(input_data_path, output_path=None):
    \"\"\"
    Loads the trained bank marketing model pipeline and makes predictions
    on new data.

    Args:
        input_data_path (str): Path to the CSV file containing new data
                               (must have the same columns as the training data,
                               except the target variable).
        output_path (str, optional): Path to save the predictions CSV.
                                     If None, returns predictions as a DataFrame.

    Returns:
        pandas.DataFrame or None: DataFrame with predictions added if output_path is None,
                                 otherwise None.
    \"\"\"
    print(f"Loading model from: {{MODEL_FILE}}")
    if not os.path.exists(MODEL_FILE):
        print(f"Error: Model file not found at {{MODEL_FILE}}")
        return None

    try:
        # Load the entire pipeline (preprocessor + model)
        pipeline = joblib.load(MODEL_FILE)
        print("Model pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading model pipeline: {{e}}")
        return None

    print(f"Loading new data from: {{input_data_path}}")
    try:
        # Try reading with UTF-8 first, fallback to latin1
        try:
            new_df = pd.read_csv(input_data_path, sep=';')
        except UnicodeDecodeError:
            print("UTF-8 decoding failed for input data, trying latin1...")
            new_df = pd.read_csv(input_data_path, sep=';', encoding='latin1')

        # Ensure the target column isn't present or remove it if it is
        # The pipeline expects only feature columns
        if TARGET_COLUMN in new_df.columns:
            print(f"Warning: Target column '{{TARGET_COLUMN}}' found in input data, removing for prediction.")
            X_new = new_df.drop(columns=[TARGET_COLUMN])
        else:
             X_new = new_df.copy() # Make a copy to avoid modifying original df if passed directly

        # Keep original index if needed later
        original_index = new_df.index

    except FileNotFoundError:
        print(f"Error: Input data file not found at {{input_data_path}}")
        return None
    except Exception as e:
        print(f"Error loading or processing input data: {{e}}")
        return None

    print(f"Making predictions on {{X_new.shape[0]}} samples...")
    try:
        # Use the loaded pipeline to preprocess and predict
        predictions = pipeline.predict(X_new)
        probabilities = None
        if hasattr(pipeline, "predict_proba"):
             probabilities = pipeline.predict_proba(X_new)[:, 1] # Probability of class '1' (yes)

        print("Predictions generated successfully.")

        # Create output DataFrame
        results_df = pd.DataFrame(index=original_index)
        results_df['prediction'] = predictions
        # Map prediction back to yes/no for clarity
        results_df['prediction_label'] = results_df['prediction'].map({{1: 'yes', 0: 'no'}})
        if probabilities is not None:
             results_df['prediction_probability_yes'] = probabilities

        # Add original data if desired (optional)
        # results_df = pd.concat([new_df.set_index(original_index), results_df], axis=1)


        if output_path:
            try:
                # Create output directory if it doesn't exist
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                results_df.to_csv(output_path, index=True)
                print(f"Predictions saved to: {{output_path}}")
                return None # Indicate success when saving to file
            except Exception as e:
                print(f"Error saving predictions to {{output_path}}: {{e}}")
                return results_df # Return df if saving fails
        else:
            return results_df # Return df if no output path specified

    except Exception as e:
        print(f"Error during prediction: {{e}}")
        # This might happen if input data structure is drastically different
        # or contains values the preprocessor cannot handle (despite handle_unknown='ignore')
        return None

# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict bank term deposit subscription using a trained model.")
    parser.add_argument("input_file", help="Path to the input CSV data file for prediction.")
    parser.add_argument("-o", "--output", help="Path to save the output CSV file with predictions.", default=None)

    args = parser.parse_args()

    predict_term_deposit(args.input_file, args.output)
"""
    try:
        with open(PREDICTION_SCRIPT_FILE, 'w') as f:
            f.write(prediction_script_content)
        print(f"Prediction script generated successfully: {PREDICTION_SCRIPT_FILE}")
    except Exception as e:
        print(f"Error generating prediction script: {e}")

    # --- 8. Return Results ---
    results_dict = {
        'accuracy': best_accuracy,
        'model_path': MODEL_FILE,
        'metrics_path': METRICS_FILE,
        'report_path': REPORT_FILE,
        'prediction_script_path': PREDICTION_SCRIPT_FILE,
        'best_model_name': best_model_name
    }
    print("\nTraining process completed.")
    return results_dict


# --- Example of how to call the function ---
if __name__ == "__main__":
    print("Running training script directly...")
    training_results = train_bank_marketing_model()

    if training_results:
        print("\n--- Training Function Output ---")
        print(f"Best Model Accuracy: {training_results['accuracy']:.4f}")
        print(f"Best Model Name: {training_results['best_model_name']}")
        print(f"Saved Model Path: {training_results['model_path']}")
        print(f"Saved Metrics Path: {training_results['metrics_path']}")
        print(f"Saved Report Path: {training_results['report_path']}")
        print(f"Generated Prediction Script: {training_results['prediction_script_path']}")

        # Example of how to use the generated prediction script from command line:
        # Assuming you have a new data file 'new_bank_data.csv' in the same format (without 'y')
        # python generated_prediction_script.py new_bank_data.csv -o predictions.csv

        # Example of using the prediction function directly (if needed within another script)
        # from generated_prediction_script import predict_term_deposit # Assuming script is in python path
        # predictions_df = predict_term_deposit('new_bank_data.csv')
        # if predictions_df is not None:
        #     print("\n--- Direct Prediction Example ---")
        #     print(predictions_df.head())

    else:
        print("\nModel training failed.")