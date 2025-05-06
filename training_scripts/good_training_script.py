import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
import json  # Import the json module

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# --- Configuration ---
MODEL_DIR = "trained_models"
MODEL_TYPE = "classification"
TARGET_VARIABLE = "default"
FEATURES = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan']
NUMERICAL_FEATURES = ['age', 'balance']
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'housing', 'loan']
# Use DATA_FILE_PATH variable for data input as requested
# Example: DATA_FILE_PATH = 'path/to/your/bank.csv'
# Ensure this variable is set before calling the function from another script
# For demonstration purposes within this script, we'll set a placeholder.
# In a real scenario, this path should point to the actual data file.
DATA_FILE_PATH = 'uploads/bank-full.csv'  # Replace with the actual path to your data file

# Create model directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)


def train_classification_model(file_path: str):
    """
    Trains multiple classification models, selects the best one based on F1-score,
    saves the model and metrics, and returns the results as a dictionary.

    Args:
        file_path (str): The path to the input CSV data file.

    Returns:
        dict: A dictionary containing the training results.
              It includes keys like 'accuracy', 'model_path', 'metrics_path',
              and 'error' (if any error occurred).
    """
    try:
        # --- 1. Load Data ---
        try:
            # Try reading with utf-8 first
            df = pd.read_csv(file_path, sep=';', quotechar='"', encoding='utf-8')
        except UnicodeDecodeError:
            # Fallback to latin1 if utf-8 fails
            df = pd.read_csv(file_path, sep=';', quotechar='"', encoding='latin1')
        except FileNotFoundError:
            return {"error": f"Error: Data file not found at {file_path}"}
        except Exception as e:
            return {"error": f"Error reading CSV file: {e}"}

        # --- 2. Data Preparation ---
        # Select relevant columns
        if not all(col in df.columns for col in FEATURES + [TARGET_VARIABLE]):
            missing_cols = [col for col in FEATURES + [TARGET_VARIABLE] if col not in df.columns]
            return {"error": f"Error: Missing required columns: {missing_cols}"}

        df_model = df[FEATURES + [TARGET_VARIABLE]].copy()

        # Handle potential missing values (if any represented as NaN)
        # For this dataset, 'unknown' is treated as a category by OneHotEncoder
        # If there were actual NaNs, we would add imputation steps here.
        # df_model = df_model.dropna() # Example: drop rows with NaNs

        # Encode target variable ('yes'/'no' to 1/0)
        le = LabelEncoder()
        df_model[TARGET_VARIABLE] = le.fit_transform(df_model[TARGET_VARIABLE])
        target_mapping = dict(zip(le.classes_, le.transform(le.classes_)))

        # Check for severe imbalance (not used for returning, but can be useful)
        target_counts = df_model[TARGET_VARIABLE].value_counts(normalize=True)
        is_imbalanced = target_counts.min() < 0.1  # Heuristic

        X = df_model[FEATURES]
        y = df_model[TARGET_VARIABLE]

        # Split data (stratify due to potential imbalance)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- 3. Preprocessing ---
        # Create preprocessing pipelines for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        # Create a column transformer to apply different transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, NUMERICAL_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES)
            ],
            remainder='passthrough'  # Keep other columns if any
        )

        # --- 4. Define Models ---
        models = {
            "LogisticRegression": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
            "DecisionTree": DecisionTreeClassifier(random_state=42, class_weight='balanced'),
            "RandomForest": RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100),
            "GradientBoosting": GradientBoostingClassifier(random_state=42, n_estimators=100)
        }

        # --- 5. Train and Evaluate Models ---
        best_model_name = None
        best_model_pipeline = None
        best_f1_score = -1
        results = {}

        for name, model in models.items():
            # Create the full pipeline: preprocessing + model
            pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                       ('classifier', model)])

            # Train the model
            pipeline.fit(X_train, y_train)

            # Make predictions
            y_pred = pipeline.predict(X_test)
            y_pred_proba = pipeline.predict_proba(X_test)[:, 1]

            # Evaluate
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)
            f1 = report['weighted avg']['f1-score']
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            cm = confusion_matrix(y_test, y_pred)

            results[name] = {
                'pipeline': pipeline,
                'accuracy': accuracy,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'classification_report': report,
                'confusion_matrix': cm.tolist()
            }

            # Update best model based on F1 score
            if f1 > best_f1_score:
                best_f1_score = f1
                best_model_name = name
                best_model_pipeline = pipeline

        # --- 6. Save Best Model and Metrics ---
        best_model_results = results[best_model_name]
        final_accuracy = best_model_results['accuracy']

        # Save the entire pipeline (preprocessor + model)
        model_filename = f"{MODEL_TYPE}_{TARGET_VARIABLE}_best_model.joblib"
        best_model_path = os.path.join(MODEL_DIR, model_filename)
        joblib.dump(best_model_pipeline, best_model_path)

        # Save detailed metrics
        metrics_filename = f"{MODEL_TYPE}_{TARGET_VARIABLE}_best_model_metrics.joblib"
        metrics_path = os.path.join(MODEL_DIR, metrics_filename)
        metrics_to_save = {
            'best_model_name': best_model_name,
            'accuracy': best_model_results['accuracy'],
            'f1_score': best_model_results['f1_score'],
            'roc_auc': best_model_results['roc_auc'],
            'classification_report': best_model_results['classification_report'],
            'confusion_matrix': best_model_results['confusion_matrix'],
            'target_variable': TARGET_VARIABLE,
            'features': FEATURES,
            'target_mapping': target_mapping
        }
        joblib.dump(metrics_to_save, metrics_path)

        return {
            "accuracy": final_accuracy,
            "model_path": best_model_path,
            "metrics_path": metrics_path,
            "target_counts": {k: v for k, v in target_counts.items()},  # Convert Series to dict
        }

    except MemoryError:
        return {"error": "Error: Insufficient memory to load or process the data."}
    except KeyError as e:
        return {"error": f"Error: Column not found in DataFrame - {e}. Check FEATURES and TARGET_VARIABLE."}
    except ValueError as e:
        return {"error": f"Error during data processing or model training: {e}"}
    except Exception as e:
        import traceback
        return {"error": f"An unexpected error occurred: {e}", "traceback": traceback.format_exc()}


def predict_with_model(model_path: str, new_data: pd.DataFrame):
    """
    Loads a saved model pipeline and makes predictions on new data.

    Args:
        model_path (str): Path to the saved .joblib model pipeline.
        new_data (pd.DataFrame): DataFrame containing new data with the same
                                 features used for training.

    Returns:
        dict: A dictionary containing the prediction results,
              including 'predictions', 'probabilities', and 'error' (if any).
    """
    try:
        # Load the pipeline
        pipeline = joblib.load(model_path)

        # Ensure new_data has the required feature columns
        try:
            # Get feature names from the preprocessor step if it's ColumnTransformer
            if isinstance(pipeline.named_steps['preprocessor'], ColumnTransformer):
                # This gets trickier with nested pipelines; let's rely on the initial FEATURES list
                required_features = FEATURES
            else:  # Fallback if preprocessor isn't as expected
                required_features = FEATURES  # Use the global FEATURES list
        except Exception:
            required_features = FEATURES  # Fallback

        if not all(col in new_data.columns for col in required_features):
            missing_cols = [col for col in required_features if col not in new_data.columns]
            return {"error": f"Error: New data is missing required columns: {missing_cols}"}

        # Select only the required feature columns in the correct order
        new_data_features = new_data[required_features]

        # Make predictions
        predictions = pipeline.predict(new_data_features)
        probabilities = pipeline.predict_proba(new_data_features)[:, 1]  # Probability of class '1'

        return {
            "predictions": predictions.tolist(),  # Convert numpy array to list for JSON
            "probabilities": probabilities.tolist(),
            "prediction_error": None,
        }

    except FileNotFoundError:
        return {"error": f"Error: Model file not found at {model_path}"}
    except Exception as e:
        import traceback
        return {"error": f"An error occurred during prediction: {e}", "traceback": traceback.format_exc()}


if __name__ == "__main__":

    training_results = train_classification_model(DATA_FILE_PATH)

    if "error" in training_results:
        print(json.dumps({"error": training_results["error"]}))  # Output JSON error
    else:
        print(json.dumps(training_results))  # Output JSON results
