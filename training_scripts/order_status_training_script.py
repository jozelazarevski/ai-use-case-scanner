import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.exceptions import NotFittedError
import warnings

warnings.filterwarnings('ignore')

# --- Configuration ---
FILE_PATH = 'uploads/ecommerce_data_flat.csv'
TARGET_VARIABLE = 'order_status'
MODEL_DIR = 'trained_models'
MODEL_FILENAME = 'order_status_prediction_model.joblib'
PREPROCESSOR_FILENAME = 'order_status_preprocessor.joblib'
MODEL_TYPE = 'classification' # Variable indicating the type of ML problem

# Create directory for trained models if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Data Loading ---
def load_data(file_path):
    """Loads data from CSV, attempting different encodings."""
    encodings_to_try = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path,low_memory=False)
            print(f"Successfully loaded data with encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            print(f"Failed to load with encoding: {encoding}")
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during file loading: {e}")
            return None
    print("Error: Could not load the file with any attempted encoding.")
    return None

# --- Feature Engineering & Preprocessing ---
def preprocess_data(df, target_variable, preprocessor=None, fit_preprocessor=False):
    """Prepares data for modeling: feature selection, engineering, imputation, encoding, scaling."""

    # Drop irrelevant or problematic columns
    # IDs, names, emails, free text are generally not useful directly or require complex NLP
    # Date columns require careful handling; we'll extract features or calculate durations
    # Review/Visit columns have many NaNs in the snippet; dropping for simplicity now
    cols_to_drop = [
        'order_id', 'customer_id', 'customer_name', 'customer_email',
        'product_id', 'product_name', # Keep category/sub-category
        'last_visit_visit_id', 'last_visit_customer_id', 'last_visit_visit_timestamp',
        'last_visit_page_visited', 'last_visit_duration_seconds', 'last_visit_device',
        'last_visit_browser', 'last_visit_source', 'review_review_id', 'review_customer_id',
        'review_product_id', 'review_review_text', 'review_review_date', 'customer_city' # High cardinality
    ]
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Handle Date columns
    date_cols = ['order_date', 'customer_registration_date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Feature Engineering: Calculate time differences or extract components
    if 'order_date' in df.columns:
        df['order_month'] = df['order_date'].dt.month
        df['order_dayofweek'] = df['order_date'].dt.dayofweek
        df['order_hour'] = df['order_date'].dt.hour
        if 'customer_registration_date' in df.columns:
             # Calculate customer tenure in days at the time of order
             df['customer_tenure_days'] = (df['order_date'] - df['customer_registration_date']).dt.days
        df = df.drop(columns=['order_date', 'customer_registration_date']) # Drop original date cols


    # Define feature types
    numeric_features = df.select_dtypes(include=np.number).columns.tolist()
    categorical_features = df.select_dtypes(exclude=np.number).columns.tolist()

    # Remove target variable from feature lists
    if target_variable in numeric_features:
        numeric_features.remove(target_variable)
    if target_variable in categorical_features:
        categorical_features.remove(target_variable)

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_variable])
    y = df[target_variable]

    # --- Preprocessing Steps ---
    if fit_preprocessor or preprocessor is None:
        # Create preprocessing pipelines for numeric and categorical features
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), # Use median for robustness to outliers
            ('scaler', StandardScaler())])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # Fill missing categories
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]) # Use one-hot encoding

        # Create a column transformer to apply different transformations to different columns
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, [col for col in numeric_features if col in X.columns]),
                ('cat', categorical_transformer, [col for col in categorical_features if col in X.columns])
            ],
            remainder='passthrough' # Keep other columns (if any) - should be none after selection
        )
        # Fit the preprocessor
        X_processed = preprocessor.fit_transform(X)
        print("Preprocessor fitted.")
    else:
         # Apply the existing preprocessor
        try:
            X_processed = preprocessor.transform(X)
            print("Data transformed using existing preprocessor.")
        except NotFittedError:
             print("Error: Preprocessor provided but not fitted. Fitting it now.")
             X_processed = preprocessor.fit_transform(X)
        except Exception as e:
            print(f"Error applying preprocessor: {e}")
            # Fallback: Try fitting again if transform fails unexpectedly
            try:
                print("Attempting to refit preprocessor as fallback.")
                X_processed = preprocessor.fit_transform(X)
            except Exception as fit_e:
                 print(f"Fallback preprocessor fitting failed: {fit_e}")
                 return None, None, None


    # --- Encode Target Variable ---
    # Check if target variable needs encoding (if it's not already numerical)
    if y.dtype == 'object':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        print(f"Target variable '{target_variable}' encoded.")
        # Save the mapping for potential future use (e.g., decoding predictions)
        target_classes = le.classes_
        joblib.dump(le, os.path.join(MODEL_DIR, 'target_encoder.joblib'))
        joblib.dump(target_classes, os.path.join(MODEL_DIR, 'target_classes.joblib'))

    else:
        y_encoded = y
        target_classes = np.unique(y) # Store unique classes if already numeric
        print(f"Target variable '{target_variable}' is already numeric.")
        joblib.dump(target_classes, os.path.join(MODEL_DIR, 'target_classes.joblib'))


    # Get feature names after one-hot encoding (important for model interpretation later)
    try:
        feature_names_out = preprocessor.get_feature_names_out()
    except Exception:
        # Fallback for older scikit-learn versions or issues
        feature_names_out = None
        print("Could not get feature names from preprocessor automatically.")

    # Convert processed features back to DataFrame (optional, but good for inspection)
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names_out, index=X.index)

    return X_processed_df, y_encoded, preprocessor


# --- Model Training and Evaluation ---
def train_and_evaluate_models(X_train, y_train, X_test, y_test):
    """Trains multiple ML models and selects the best one based on accuracy."""

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }

    best_model_name = None
    best_model = None
    best_accuracy = 0.0
    all_metrics = {}

    print("\n--- Model Training and Evaluation ---")
    for name, model in models.items():
        print(f"Training {name}...")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

            print(f"{name} Accuracy: {accuracy:.4f}")
            # print(f"{name} Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}") # Print full report

            all_metrics[name] = {
                'accuracy': accuracy,
                'classification_report': report
            }

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model
                best_model_name = name
                print(f"*** New Best Model: {name} (Accuracy: {best_accuracy:.4f}) ***")

        except Exception as e:
            print(f"Error training/evaluating {name}: {e}")

    if best_model:
        print(f"\n--- Best Performing Model: {best_model_name} ---")
        detailed_metrics_best = all_metrics[best_model_name]['classification_report']
    else:
        print("\n--- No model trained successfully ---")
        detailed_metrics_best = {}

    return best_model, best_model_name, best_accuracy, detailed_metrics_best

# --- Prediction Function ---
def predict_new_data(data_input, model_path, preprocessor_path, target_encoder_path=None):
    """Loads the trained model and preprocessor to make predictions on new data."""
    try:
        # Load the trained model and preprocessor
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        print("Loaded trained model and preprocessor.")

        # Ensure input is a DataFrame
        if isinstance(data_input, str): # If filepath is given
             new_data_df = load_data(data_input)
             if new_data_df is None:
                 return None
        elif isinstance(data_input, pd.DataFrame):
             new_data_df = data_input.copy()
        else:
            print("Error: Input data must be a pandas DataFrame or a file path string.")
            return None

        # Apply the *exact same* preprocessing
        # Note: We don't need the target variable here for prediction
        # We pass `fit_preprocessor=False` to use the loaded (already fitted) preprocessor
        X_new_processed, _, _ = preprocess_data(new_data_df, target_variable=TARGET_VARIABLE, preprocessor=preprocessor, fit_preprocessor=False)

        if X_new_processed is None:
            print("Error during preprocessing of new data.")
            return None

        # Make predictions
        predictions_encoded = model.predict(X_new_processed)
        predictions_proba = None
        if hasattr(model, "predict_proba"):
             predictions_proba = model.predict_proba(X_new_processed)


        # Decode predictions if a target encoder was saved
        if target_encoder_path and os.path.exists(target_encoder_path):
            try:
                le = joblib.load(target_encoder_path)
                predictions_decoded = le.inverse_transform(predictions_encoded)
                print("Predictions decoded to original labels.")
                return predictions_decoded, predictions_proba, predictions_encoded # Return decoded, probabilities and encoded
            except Exception as e:
                print(f"Warning: Could not load or use target encoder ({e}). Returning encoded predictions.")
                return predictions_encoded, predictions_proba, None # Return encoded and probabilities
        else:
             print("Target encoder not found or not specified. Returning encoded predictions.")
             return predictions_encoded, predictions_proba, None # Return encoded and probabilities


    except FileNotFoundError:
        print(f"Error: Model or preprocessor file not found at specified paths.")
        print(f"Model path: {model_path}")
        print(f"Preprocessor path: {preprocessor_path}")
        return None
    except NotFittedError as nfe:
         print(f"Error: The loaded preprocessor was not fitted: {nfe}. Cannot transform new data.")
         return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting the ML model training process...")

    # 1. Load Data
    df = load_data(FILE_PATH)

    if df is not None and TARGET_VARIABLE in df.columns:
        # Check if target variable has more than one class
        if df[TARGET_VARIABLE].nunique() < 2:
             print(f"Error: Target variable '{TARGET_VARIABLE}' has only one unique value. Cannot train a classifier.")
        else:
            # Handle potential NaN values in the target column before splitting
            df = df.dropna(subset=[TARGET_VARIABLE])
            if df.empty:
                print(f"Error: No valid data remaining after removing rows with missing target variable '{TARGET_VARIABLE}'.")
            else:
                print(f"Target variable '{TARGET_VARIABLE}' value counts:\n{df[TARGET_VARIABLE].value_counts()}")

                # 2. Preprocess Data (Fit preprocessor here)
                X_processed, y_encoded, preprocessor = preprocess_data(df, TARGET_VARIABLE, fit_preprocessor=True)

                if X_processed is not None and y_encoded is not None and preprocessor is not None:

                    # 3. Train/Test Split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_processed, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded # Stratify for classification
                    )
                    print(f"\nData split: Train set size = {X_train.shape[0]}, Test set size = {X_test.shape[0]}")

                    # 4. Train and Evaluate Models
                    best_model, best_model_name, Accuracy, detailed_metrics = train_and_evaluate_models(
                        X_train, y_train, X_test, y_test
                    )

                    # 5. Save the results and the best model
                    if best_model:
                        print("\n--- Saving Best Model and Preprocessor ---")
                        model_save_path = os.path.join(MODEL_DIR, MODEL_FILENAME)
                        preprocessor_save_path = os.path.join(MODEL_DIR, PREPROCESSOR_FILENAME)

                        joblib.dump(best_model, model_save_path)
                        joblib.dump(preprocessor, preprocessor_save_path)

                        print(f"Best Model ({best_model_name}) saved to: {model_save_path}")
                        print(f"Preprocessor saved to: {preprocessor_save_path}")

                        # Display final results
                        print(f"\nFinal Best Model: {best_model_name}")
                        print(f"Final Accuracy: {Accuracy:.4f}") # Saved in Accuracy variable
                        print("Final Detailed Metrics (Classification Report):")
                        # Print formatted report from the dictionary
                        if isinstance(detailed_metrics, dict):
                            report_df = pd.DataFrame(detailed_metrics).transpose()
                            print(report_df)
                        else:
                            print(detailed_metrics) # Print as is if not a dict

                        # --- Example: How to use the prediction function ---
                        print("\n--- Example Prediction ---")
                        # Create a sample of the original data to test prediction function
                        # IMPORTANT: The prediction function needs data in the *original* format (before preprocessing)
                        sample_new_data = df.head(5).drop(columns=[TARGET_VARIABLE], errors='ignore')
                        if not sample_new_data.empty:
                             predictions, probabilities, _ = predict_new_data(
                                 data_input=sample_new_data,
                                 model_path=model_save_path,
                                 preprocessor_path=preprocessor_save_path,
                                 target_encoder_path=os.path.join(MODEL_DIR, 'target_encoder.joblib') # Optional: for decoding
                             )

                             if predictions is not None:
                                 print("\nSample Predictions on new data (first 5 rows):")
                                 print(predictions)
                                 # You can also access probabilities if needed and available:
                                 # if probabilities is not None:
                                 #     print("\nSample Prediction Probabilities:")
                                 #     print(probabilities)
                        else:
                             print("Could not create sample data for prediction example.")

                    else:
                        print("\nModel training failed. No model was saved.")
                else:
                     print("\nData preprocessing failed. Halting execution.")
    elif df is None:
        print("Data loading failed. Halting execution.")
    else:
         print(f"Error: Target variable '{TARGET_VARIABLE}' not found in the loaded data. Available columns: {df.columns.tolist()}")

    print("\nScript execution finished.")