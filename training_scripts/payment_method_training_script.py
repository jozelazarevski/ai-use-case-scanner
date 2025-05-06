import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import ConvergenceWarning
import warnings

# --- Configuration ---
INPUT_CSV_PATH = '../uploads/e-commerce.csv'
MODEL_SAVE_DIR = 'trained_models'
TARGET_VARIABLE = 'Payment Method'
MODEL_TYPE = 'classification' # Explicitly stating the model type

# Features to use for training
# Consider Customer ID as categorical for now, capturing repeat customer behavior
FEATURE_COLUMNS = ['Customer ID', 'Order Total', 'Shipping City', 'Product ID']

# --- Helper Functions ---

def load_data(file_path):
    """Loads data from a CSV file, trying different encodings."""
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded data using encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            print(f"Failed to load with encoding: {encoding}")
        except FileNotFoundError:
            print(f"Error: Input file not found at {file_path}")
            return None
    print("Error: Could not load data with any tried encodings.")
    return None

def create_preprocessor(categorical_features, numerical_features):
    """Creates a ColumnTransformer for preprocessing."""
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # sparse=False easier for some models

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough' # Keep other columns if any, though we selected specific features
    )
    return preprocessor

# --- Main Training Script ---

def train_payment_method_model():
    """Loads data, preprocesses, trains multiple models, selects the best, and saves it."""

    print("--- Starting Model Training ---")

    # 1. Load Data
    df = load_data(INPUT_CSV_PATH)
    if df is None:
        return None, None, None # Indicate failure

    # Basic Data Cleaning (optional but recommended)
    df = df.dropna(subset=[TARGET_VARIABLE] + FEATURE_COLUMNS) # Drop rows where target or key features are missing
    if df.empty:
        print("Error: No data remaining after dropping missing values.")
        return None, None, None

    print(f"Data shape after initial load and cleaning: {df.shape}")
    print(f"Target variable distribution:\n{df[TARGET_VARIABLE].value_counts()}")

    # 2. Define Features (X) and Target (y)
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_VARIABLE]

    # Identify feature types automatically (more robust)
    numerical_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    print(f"Identified Numerical Features: {numerical_features}")
    print(f"Identified Categorical Features: {categorical_features}")

    # Ensure 'Order Total' is numeric if it wasn't automatically detected
    if 'Order Total' not in numerical_features and 'Order Total' in X.columns:
         try:
            X['Order Total'] = pd.to_numeric(X['Order Total'])
            numerical_features = X.select_dtypes(include=np.number).columns.tolist()
            categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
            print("Corrected 'Order Total' to numeric.")
         except ValueError:
             print("Warning: Could not convert 'Order Total' to numeric. Check data.")
             # Decide how to handle - drop column? Or error out? Dropping for now.
             if 'Order Total' in categorical_features: categorical_features.remove('Order Total')
             X = X.drop('Order Total', axis=1)


    # Handle cases with no numeric or no categorical features
    if not numerical_features and not categorical_features:
        print("Error: No valid features identified.")
        return None, None, None
    if not numerical_features:
        print("Warning: No numerical features identified for scaling.")
    if not categorical_features:
         print("Warning: No categorical features identified for encoding.")


    # 3. Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y # Stratify helps with imbalanced classes
    )
    print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # 4. Create Preprocessor
    # Re-create based on actual identified types after potential correction
    preprocessor = create_preprocessor(categorical_features, numerical_features)

    # 5. Define Models to Try
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'), # liblinear often good for smaller datasets
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42, n_estimators=100)
    }

    best_model_name = None
    best_model_pipeline = None
    best_accuracy = 0.0
    best_metrics_report = ""
    model_results = {}

    # Suppress ConvergenceWarning for Logistic Regression if iterations aren't enough
    warnings.filterwarnings("ignore", category=ConvergenceWarning)


    # 6. Train and Evaluate Models
    print("\n--- Training and Evaluating Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        # Create a full pipeline: Preprocessing -> Model
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('classifier', model)])

        # Train
        try:
            pipeline.fit(X_train, y_train)
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue # Skip to the next model

        # Predict
        y_pred = pipeline.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0) # Handle cases with no predicted samples for a class

        print(f"Results for {name}:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Classification Report:\n{report}\n")

        model_results[name] = {'accuracy': accuracy, 'report': report, 'pipeline': pipeline}

        # Check if this is the best model so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            best_model_pipeline = pipeline
            best_metrics_report = report

    warnings.filterwarnings("default", category=ConvergenceWarning) # Reset warning filter


    # 7. Save the Best Model
    if best_model_pipeline:
        print(f"\n--- Best Model Selection ---")
        print(f"Best performing model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

        # Create directory if it doesn't exist
        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        model_filename = f"{MODEL_SAVE_DIR}/best_{MODEL_TYPE}_model_payment_method.joblib"

        try:
            joblib.dump(best_model_pipeline, model_filename)
            print(f"Best model (pipeline) saved successfully to: {model_filename}")
        except Exception as e:
            print(f"Error saving model: {e}")
            return None, None, None # Indicate failure during saving

        # Store final metrics
        Accuracy = best_accuracy # Global variable for accuracy
        detailed_metrics = best_metrics_report # Global variable for detailed report

        print("\n--- Detailed Metrics for Best Model ---")
        print(detailed_metrics)

        return best_model_pipeline, Accuracy, detailed_metrics
    else:
        print("\nError: No model was successfully trained.")
        return None, None, None


# --- Prediction Script Preparation ---

def load_prediction_model(model_path=f"{MODEL_SAVE_DIR}/best_{MODEL_TYPE}_model_payment_method.joblib"):
    """Loads the saved model pipeline."""
    try:
        model_pipeline = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model_pipeline
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Train the model first.")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def predict_payment_method(new_data, model_pipeline):
    """
    Predicts the payment method for new data using the loaded model pipeline.

    Args:
        new_data (pd.DataFrame): DataFrame containing the new transaction data.
                                 Must have the same columns used for training
                                 (e.g., 'Customer ID', 'Order Total', 'Shipping City', 'Product ID').
        model_pipeline (Pipeline): The loaded scikit-learn pipeline object.

    Returns:
        np.ndarray: An array of predicted payment method labels.
                    Returns None if prediction fails.
    """
    if model_pipeline is None:
        print("Error: Model pipeline is not loaded. Cannot make predictions.")
        return None
    if not isinstance(new_data, pd.DataFrame):
         print("Error: Input data must be a pandas DataFrame.")
         return None

    # Ensure required columns exist (use FEATURE_COLUMNS defined earlier)
    missing_cols = [col for col in FEATURE_COLUMNS if col not in new_data.columns]
    if missing_cols:
        print(f"Error: Input data is missing required columns: {missing_cols}")
        return None

    # Ensure data types are consistent if possible (basic check)
    # More robust checking might be needed in production
    try:
        if 'Order Total' in new_data.columns:
            new_data['Order Total'] = pd.to_numeric(new_data['Order Total'], errors='coerce')
            if new_data['Order Total'].isnull().any():
                 print("Warning: 'Order Total' contains non-numeric values after conversion. Predictions might be affected.")
                 # Decide on handling: fill with mean/median, or let the pipeline handle if possible
                 # For now, we let the pipeline's scaler potentially error or handle it based on training.
    except Exception as e:
        print(f"Warning: Could not ensure 'Order Total' is numeric. {e}")


    print(f"\n--- Making Predictions on New Data ({len(new_data)} samples) ---")
    try:
        # Use the pipeline's predict method - it handles preprocessing internally
        predictions = model_pipeline.predict(new_data[FEATURE_COLUMNS])
        print("Predictions generated successfully.")
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        # This could happen if new data has categories not seen during training
        # and OneHotEncoder wasn't set to handle_unknown='ignore', or if data types are wrong.
        return None


# --- Main Execution ---
if __name__ == "__main__":
    # Train the model
    trained_pipeline, Accuracy, detailed_metrics = train_payment_method_model()

    if trained_pipeline:
        print("\n--- Training Complete ---")
        print(f"Best Model Accuracy: {Accuracy:.4f}")

        # --- Example of Running Prediction on New Data ---
        print("\n--- Prediction Example ---")
        # Create some sample new data (should match the structure of X)
        # Make sure column names are exactly the same as FEATURE_COLUMNS
        new_transaction_data = pd.DataFrame({
            'Customer ID': ['CUST101', 'CUST999', 'CUST103'],
            'Order Total': [55.0, 1500.75, 22.10],
            'Shipping City': ['Zürich', 'Geneva', 'Lausanne'],
            'Product ID': ['PROD505', 'PROD800', 'PROD710'] # Example includes a potentially new Product ID if PROD800 wasn't in training
        })

        # Load the *just trained* model (or load from file if running separately)
        # In this script flow, we can just use the 'trained_pipeline' variable directly
        # prediction_model = load_prediction_model() # Use this if running prediction separately

        if trained_pipeline: # Or use 'prediction_model' if loaded from file
            predicted_methods = predict_payment_method(new_transaction_data, trained_pipeline) # Or prediction_model

            if predicted_methods is not None:
                new_transaction_data['Predicted Payment Method'] = predicted_methods
                print("\n--- Predictions for New Data ---")
                print(new_transaction_data)
        else:
             print("Skipping prediction example because model loading/training failed.")

    else:
        print("\n--- Model training failed. ---")