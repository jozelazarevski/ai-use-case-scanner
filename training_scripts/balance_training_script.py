import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.impute import SimpleImputer

# Define the path to the data file
DATA_FILE_PATH = "uploads/bank-full.csv"
MODEL_DIR = "trained_models"
METRICS_DIR = "evaluation_metrics" # Changed from saving metrics in model dir

def train_balance_predictor(data_file_path: str):
    """
    Trains and evaluates regression models to predict customer account balance.

    Args:
        data_file_path (str): The path to the input CSV file.

    Returns:
        dict: A dictionary containing performance metrics of the best model,
              path to the saved model, and path to the metrics file.
              Example: {
                  'best_model_name': 'RandomForestRegressor',
                  'metrics': {'R2_Score': 0.85, 'MAE': 1500, 'MSE': 5000000},
                  'feature_importances': {'age': 0.1, ...},
                  'model_path': 'trained_models/best_balance_model.joblib',
                  'metrics_file_path': 'evaluation_metrics/best_model_evaluation_metrics.txt'
              }
    """
    # --- 1. Data Loading ---
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    df = None
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(data_file_path, sep=';') # Common separator for bank data
            print(f"Successfully loaded data with encoding: {encoding}")
            break
        except UnicodeDecodeError:
            print(f"Failed to load with encoding: {encoding}")
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            return {"error": "Failed to load data file."}

    if df is None:
        print("Could not load the data file with any attempted encodings.")
        return {"error": "Could not load data file with any attempted encodings."}

    # --- 2. Feature Selection and Preprocessing ---
    # Define features (demographic as requested) and target
    # Include 'balance' in features initially for easier splitting/handling
    demographic_features = ['age', 'job', 'marital', 'education']
    target = 'balance'

    if target not in df.columns:
         return {"error": f"Target variable '{target}' not found in the dataset."}

    required_features = demographic_features
    missing_features = [f for f in required_features if f not in df.columns]
    if missing_features:
        return {"error": f"Missing required features: {missing_features}"}

    X = df[required_features]
    y = df[target]

    # Identify numerical and categorical features
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    # Create preprocessing pipelines for numerical and categorical features
    numerical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handle missing numerical values
        ('scaler', StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), # Handle missing categorical values
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # Use sparse_output=False for easier feature name handling later
    ])

    # Create a column transformer to apply different pipelines to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_pipeline, numerical_features),
            ('cat', categorical_pipeline, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any (though we selected specific ones)
    )

    # --- 3. Train/Test Split ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- 4. Model Definition ---
    # Define models to try
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=42),
        "Lasso": Lasso(random_state=42),
        "DecisionTreeRegressor": DecisionTreeRegressor(random_state=42),
        "RandomForestRegressor": RandomForestRegressor(random_state=42, n_estimators=100), # Specify n_estimators
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        # "SVR": SVR() # SVR can be very slow on larger datasets, uncomment if needed
    }

    # --- 5. Model Training and Evaluation Loop ---
    best_model_name = None
    best_model_pipeline = None
    best_r2 = -np.inf # Initialize with a very low R2 score
    results = {}

    print("\n--- Model Training and Evaluation ---")
    for name, model in models.items():
        print(f"Training {name}...")
        # Create the full pipeline: preprocessor + model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('regressor', model)])

        try:
            # Train the model
            pipeline.fit(X_train, y_train)

            # Predict on the test set
            y_pred = pipeline.predict(X_test)

            # Evaluate the model
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)

            results[name] = {'R2_Score': r2, 'MAE': mae, 'MSE': mse}
            print(f"{name} - R2: {r2:.4f}, MAE: {mae:.2f}, MSE: {mse:.2f}")

            # Check if this model is the best so far based on R2 score
            if r2 > best_r2:
                best_r2 = r2
                best_model_name = name
                best_model_pipeline = pipeline

        except Exception as e:
            print(f"Failed to train or evaluate {name}. Error: {e}")
            results[name] = {'error': str(e)}

    print(f"\nBest Model Found: {best_model_name} with R2 Score: {best_r2:.4f}")

    if best_model_pipeline is None:
        print("No model could be trained successfully.")
        return {"error": "Failed to train any model successfully."}

    # --- 6. Feature Importance / Coefficients ---
    feature_importances_dict = {}
    try:
        # Get feature names after one-hot encoding
        ohe_feature_names = best_model_pipeline.named_steps['preprocessor'] \
                                             .named_transformers_['cat'] \
                                             .named_steps['onehot'] \
                                             .get_feature_names_out(categorical_features)
        all_feature_names = numerical_features + list(ohe_feature_names)

        if hasattr(best_model_pipeline.named_steps['regressor'], 'feature_importances_'):
            importances = best_model_pipeline.named_steps['regressor'].feature_importances_
            feature_importances_dict = dict(zip(all_feature_names, importances))
        elif hasattr(best_model_pipeline.named_steps['regressor'], 'coef_'):
            coefficients = best_model_pipeline.named_steps['regressor'].coef_
            # Handle potential multi-output case (though unlikely for standard regressors)
            if coefficients.ndim > 1:
                 coefficients = coefficients.flatten() # Or handle appropriately if needed
            # Ensure coefficient length matches feature names length
            if len(coefficients) == len(all_feature_names):
                feature_importances_dict = dict(zip(all_feature_names, coefficients))
            else:
                 print(f"Warning: Mismatch between number of coefficients ({len(coefficients)}) and feature names ({len(all_feature_names)}). Cannot extract coefficients reliably.")
                 feature_importances_dict = {"error": "Coefficient/feature name mismatch"}

        # Sort feature importances/coefficients by absolute value for better readability (optional)
        # feature_importances_dict = dict(sorted(feature_importances_dict.items(), key=lambda item: abs(item[1]), reverse=True))

    except Exception as e:
        print(f"Could not extract feature importances/coefficients: {e}")
        feature_importances_dict = {"error": f"Could not extract feature importance/coefficients: {e}"}


    # --- 7. Model Persistence ---
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_file_path = os.path.join(MODEL_DIR, "best_balance_model.joblib")
    try:
        joblib.dump(best_model_pipeline, model_file_path)
        print(f"Best model ({best_model_name}) saved to: {model_file_path}")
    except Exception as e:
        print(f"Error saving model: {e}")
        return {"error": f"Failed to save model: {e}"}


    # --- 8. Save Detailed Evaluation Metrics ---
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_file_path = os.path.join(METRICS_DIR, "best_model_evaluation_metrics.txt")
    final_metrics = results[best_model_name]

    try:
        with open(metrics_file_path, 'w') as f:
            f.write(f"Best Model: {best_model_name}\n\n")
            f.write("Performance Metrics:\n")
            for metric, value in final_metrics.items():
                f.write(f"- {metric}: {value:.4f}\n")
            f.write("\nFeature Importances/Coefficients:\n")
            if isinstance(feature_importances_dict, dict) and "error" not in feature_importances_dict :
                 for feature, importance in feature_importances_dict.items():
                    f.write(f"- {feature}: {importance:.4f}\n")
            else:
                 f.write(f"Could not retrieve feature importances/coefficients: {feature_importances_dict.get('error', 'Unknown error')}\n")
        print(f"Detailed evaluation metrics saved to: {metrics_file_path}")
    except Exception as e:
        print(f"Error saving metrics file: {e}")
        # Continue even if metrics file saving fails

    # --- 9. Prepare Return Dictionary ---
    output_data = {
        'best_model_name': best_model_name,
        'metrics': final_metrics,
        'feature_importances': feature_importances_dict if isinstance(feature_importances_dict, dict) and "error" not in feature_importances_dict else {}, # Return empty if error
        'model_path': model_file_path,
        'metrics_file_path': metrics_file_path
    }

    return output_data


def predict_balance(new_data_df: pd.DataFrame, model_path: str = os.path.join(MODEL_DIR, "best_balance_model.joblib")):
    """
    Loads the saved model and makes predictions on new data.

    Args:
        new_data_df (pd.DataFrame): DataFrame containing new customer data
                                     with the same features used for training
                                     ('age', 'job', 'marital', 'education').
        model_path (str): Path to the saved model file.

    Returns:
        np.ndarray: Array of predicted account balances.
                    Returns None if an error occurs.
    """
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None

    try:
        # Load the entire pipeline (preprocessor + model)
        loaded_pipeline = joblib.load(model_path)

        # Ensure the input DataFrame has the correct columns
        # (The pipeline's preprocessor expects specific columns)
        # No need to manually select columns if the input df has *at least* the required ones.
        # The ColumnTransformer inside the pipeline will select the correct ones.

        predictions = loaded_pipeline.predict(new_data_df)
        return predictions

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except KeyError as e:
         print(f"Error: Input data is missing expected column: {e}. Ensure columns match training data: ['age', 'job', 'marital', 'education']")
         return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None


# --- Main Execution Block ---
if __name__ == "__main__":
    print(f"Starting model training using data from: {DATA_FILE_PATH}")

    # Create dummy data file if it doesn't exist for testing purposes
    # In a real scenario, this file would be uploaded or already present.
   

    # --- Train the model ---
    training_results = train_balance_predictor(DATA_FILE_PATH)

    # --- Prepare and Print JSON Output ---
    final_json_output = {}
    if training_results and "error" not in training_results:
        # Prepare JSON with specific keys requested: R2_Score and feature_importance
        # Use R2 score as the primary metric, approximating 'Accuracy' for regression context
        final_json_output = {
            "R2_Score": training_results.get('metrics', {}).get('R2_Score'),
            "feature_importance": training_results.get('feature_importances', {})
        }
        print("\n--- Training Summary ---")
        print(f"Best Model: {training_results.get('best_model_name')}")
        print(f"Metrics: {training_results.get('metrics')}")
        print(f"Model saved to: {training_results.get('model_path')}")
        print(f"Metrics saved to: {training_results.get('metrics_file_path')}")

        # --- Example Prediction Usage ---
        print("\n--- Prediction Example ---")
        # Create some sample new data (ensure columns match original training features)
        sample_new_data = pd.DataFrame({
            'age': [42, 31, 55],
            'job': ['technician', 'admin.', 'management'],
            'marital': ['single', 'married', 'divorced'],
            'education': ['secondary', 'tertiary', 'tertiary']
            # Add other columns if they were implicitly included via 'remainder=passthrough'
            # and the model depends on them, but based on the prompt, stick to these.
        })
        print("Sample New Data:")
        print(sample_new_data)

        predictions = predict_balance(sample_new_data, model_path=training_results['model_path'])

        if predictions is not None:
            print("\nPredicted Balances:")
            print(predictions)
        else:
            print("\nPrediction failed.")

    else:
        # Training failed, output error
        print("\n--- Training Failed ---")
        error_message = training_results.get('error', 'Unknown error during training.')
        final_json_output = {"error": error_message}
        print(f"Error: {error_message}")


    # --- Output ONLY the JSON as the final step ---
    print("\n--- Final JSON Output ---")
    # Convert numpy types if necessary for JSON serialization
    def convert(o):
        if isinstance(o, np.generic): return o.item()
        raise TypeError
    print(json.dumps(final_json_output, indent=4, default=convert))