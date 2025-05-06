import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define constants
FILE_PATH = '../uploads/california_housing_train.csv'
MODEL_DIR = 'trained_models'
DENSITY_THRESHOLD = 3000  # Population threshold for high density
TARGET_VARIABLE = 'is_high_density'
MODEL_TYPE = 'classification' # Variable name for model type

# --- 1. Data Loading and Preparation ---
def load_and_prepare_data(file_path, density_threshold):
    """Loads data, creates target variable, and handles basic cleaning."""
    try:
        # Read with specified encoding or let pandas infer, handling potential errors
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='latin1')
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None

    # Create the binary target variable
    df[TARGET_VARIABLE] = (df['population'] > density_threshold).astype(int)

    # Drop the original population column and median_house_value (target for another problem)
    # Keep other columns relevant for density prediction
    df = df.drop(['population', 'median_house_value'], axis=1, errors='ignore')

    return df

# --- 2. Feature Engineering and Selection ---
def engineer_features(df):
    """Creates new features and handles potential issues."""
    # Impute missing 'total_bedrooms' using the median
    imputer = SimpleImputer(strategy='median')
    df['total_bedrooms'] = imputer.fit_transform(df[['total_bedrooms']])

    # Feature combinations - handle potential division by zero
    df['rooms_per_household'] = df['total_rooms'] / df['households'].replace(0, np.nan)
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms'].replace(0, np.nan)
    df['pop_per_household'] = df['total_bedrooms'] / df['households'].replace(0, np.nan) # Using total_bedrooms as a proxy if original population is dropped early

    # Fill any NaNs created by division by zero (or if original NaNs existed in denominators)
    df = df.fillna(df.median()) # Fill with median of the column

    # Define features (X) and target (y)
    features = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'households', 'median_income',
        'rooms_per_household', 'bedrooms_per_room', 'pop_per_household'
    ]
    # Ensure all engineered features are in the dataframe before selection
    final_features = [f for f in features if f in df.columns]

    X = df[final_features]
    y = df[TARGET_VARIABLE]

    return X, y, final_features, imputer # Return imputer for prediction pipeline

# --- 3. Model Training and Evaluation ---
def train_evaluate_models(X_train, y_train, X_test, y_test):
    """Trains multiple classifiers and returns the best one."""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    best_model_name = None
    best_accuracy = 0.0
    best_model_pipeline = None
    all_metrics = {}

    # Create a standard scaler
    scaler = StandardScaler()

    for name, model in models.items():
        print(f"Training {name}...")
        # Create a pipeline with scaling and the model
        pipeline = Pipeline([
            ('scaler', scaler),
            ('classifier', model)
        ])

        # Train the pipeline
        pipeline.fit(X_train, y_train)

        # Predict on the test set
        y_pred = pipeline.predict(X_test)

        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        print(f"{name} Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("-" * 30)

        all_metrics[name] = {
            'accuracy': accuracy,
            'classification_report': report
        }

        # Check if this model is the best so far
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_name = name
            # Store the pipeline for the best model based on test set performance
            # Note: We will retrain this best model type on the full data later
            best_model_pipeline = pipeline 

    print(f"\nBest performing model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

    # Get detailed metrics for the best model
    best_detailed_metrics = all_metrics[best_model_name]['classification_report']

    return best_model_name, best_accuracy, best_detailed_metrics, models[best_model_name]

# --- 4. Save Model ---
def save_model(model_pipeline, model_dir, filename="best_density_classifier.joblib"):
    """Saves the trained model pipeline."""
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_path = os.path.join(model_dir, filename)
    joblib.dump(model_pipeline, model_path)
    print(f"Model saved to {model_path}")
    return model_path

# --- 5. Prepare Prediction Script Structure ---
def predict_density(input_data, model_path, feature_names, imputer):
    """
    Loads the saved model and predicts density for new data.

    Args:
        input_data (pd.DataFrame): DataFrame with the same features used for training.
        model_path (str): Path to the saved .joblib model file.
        feature_names (list): List of feature names the model expects.
        imputer (SimpleImputer): The fitted imputer used during training.

    Returns:
        np.array: Predictions (0 or 1 for is_high_density).
    """
    print(f"\n--- Prediction Function ---")
    # Load the trained model pipeline (includes scaler and classifier)
    try:
        model_pipeline = joblib.load(model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Ensure input_data is a DataFrame
    if not isinstance(input_data, pd.DataFrame):
        try:
            input_data = pd.DataFrame(input_data) # Basic attempt to convert
            print("Input data converted to DataFrame.")
        except Exception as e:
            print(f"Error: Could not convert input_data to DataFrame: {e}")
            return None


    print("Preparing input data for prediction...")
    # Preprocessing steps similar to training:
    # 1. Impute missing 'total_bedrooms' (using the imputer fitted on training data)
    if 'total_bedrooms' in input_data.columns:
         input_data['total_bedrooms'] = imputer.transform(input_data[['total_bedrooms']])
         print("'total_bedrooms' imputed.")
    else:
        print("Warning: 'total_bedrooms' not found in input data.")


    # 2. Feature Engineering (must match training)
    try:
        input_data['rooms_per_household'] = input_data['total_rooms'] / input_data['households'].replace(0, np.nan)
        input_data['bedrooms_per_room'] = input_data['total_bedrooms'] / input_data['total_rooms'].replace(0, np.nan)
        input_data['pop_per_household'] = input_data['total_bedrooms'] / input_data['households'].replace(0, np.nan) # Consistent with training
        print("Engineered features created.")
    except KeyError as e:
        print(f"Error creating engineered features: Missing column {e}")
        return None

    # Fill any NaNs resulting from division by zero or missing base features
    # Use median imputation based on the *training* data if possible, or 0 as fallback.
    # For simplicity here, fill with 0, but using training medians is better.
    input_data = input_data.fillna(0)
    print("NaNs filled (using 0 as fallback - consider using training medians).")


    # 3. Ensure column order and selection matches training features
    try:
        input_data_processed = input_data[feature_names]
        print(f"Input data filtered to required features: {feature_names}")
    except KeyError as e:
        print(f"Error: Input data missing required feature column: {e}")
        print(f"Expected columns: {feature_names}")
        print(f"Available columns: {list(input_data.columns)}")
        return None

    # 4. Predict using the loaded pipeline (handles scaling)
    print("Making predictions...")
    try:
        predictions = model_pipeline.predict(input_data_processed)
        print("Predictions generated successfully.")
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

# --- Main Execution ---
if __name__ == "__main__":
    print("Starting the High Density Housing Classification pipeline...")

    # 1. Load and Prepare Data
    print(f"Loading data from: {FILE_PATH}")
    df = load_and_prepare_data(FILE_PATH, DENSITY_THRESHOLD)

    if df is not None:
        print(f"Target variable '{TARGET_VARIABLE}' created with threshold > {DENSITY_THRESHOLD}")
        print(f"Value counts for '{TARGET_VARIABLE}':\n{df[TARGET_VARIABLE].value_counts(normalize=True)}")

        # 2. Feature Engineering
        print("Engineering features...")
        X, y, feature_names, imputer = engineer_features(df.copy()) # Use copy to avoid modifying df
        print(f"Features used for training: {feature_names}")

        # 3. Split Data
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")

        # 4. Train and Evaluate Models
        print("Training and evaluating different classification models...")
        best_model_name, Accuracy, detailed_metrics, best_model_instance = train_evaluate_models(X_train, y_train, X_test, y_test)

        # Display final best model metrics
        print("\n--- Best Model Summary ---")
        print(f"Algorithm: {best_model_name}")
        print(f"Accuracy on Test Set: {Accuracy:.4f}") # Saved Accuracy variable
        print("Detailed Metrics (Test Set):")
        # Print detailed_metrics dict nicely
        for label, metrics in detailed_metrics.items():
            if isinstance(metrics, dict):
                 print(f"  Class {label}:")
                 for metric_name, value in metrics.items():
                     print(f"    {metric_name}: {value:.4f}")
            else:
                 print(f"  {label}: {metrics:.4f}") # For overall accuracy, macro avg, weighted avg

        # 5. Retrain Best Model on Full Data and Save
        print("\nRetraining the best model type on the full dataset...")
        # Create the final pipeline with scaler and the best model type
        final_pipeline = Pipeline([
            ('scaler', StandardScaler()), # Use a new scaler instance
            ('classifier', best_model_instance) # Use the best model type identified
        ])

        # Fit the final pipeline on the entire dataset (X, y)
        final_pipeline.fit(X, y)
        print("Final model trained on full dataset.")

        # Save the final pipeline
        saved_model_path = save_model(final_pipeline, MODEL_DIR)

        # --- 6. Example Prediction Usage ---
        # Create some sample data matching the structure (excluding target)
        # Important: Sample data needs all columns used in feature_names before prediction function
        print("\n--- Running Prediction Example ---")
        if not X_test.empty:
            sample_data_for_prediction = X_test.head().copy() # Use first 5 rows of test set as example
             # Need to re-run imputer and feature engineering on this raw sample
             # Simpler: use the already processed X_test, but the predict_density function expects raw data
             # Let's recreate the raw structure for demonstration
            raw_sample_df = df.loc[X_test.head().index].drop(TARGET_VARIABLE, axis=1)

            print("Sample input data (first 5 rows of test set - raw format):")
            print(raw_sample_df)

            predictions = predict_density(raw_sample_df, saved_model_path, feature_names, imputer)

            if predictions is not None:
                print("\nPredictions for sample data:")
                print(predictions)
                actual = y_test.head().values
                print("Actual values for sample data:")
                print(actual)
        else:
            print("\nSkipping prediction example as test data is empty.")

    else:
        print("Data loading failed. Exiting.")

    print("\nPipeline finished.")