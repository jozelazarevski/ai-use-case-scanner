import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# --- Configuration ---
FILE_PATH = 'D:\AI Use Case App_GOOD\uploads/Maternal_Health_Risk_Data_Set.csv'
TARGET_VARIABLE = 'BS'
# Define the threshold for high blood sugar based on clinical relevance/data distribution
# Example: BS >= 10 might be a reasonable threshold for early screening
BS_THRESHOLD = 10
MODEL_OUTPUT_FOLDER = 'trained_models'
MODEL_TYPE_CLASSIFICATION = 'classification' # Clear variable name for model type

# --- Ensure output directory exists ---
os.makedirs(MODEL_OUTPUT_FOLDER, exist_ok=True)

# --- 1. Load Data ---
try:
    # Try reading with default UTF-8 encoding
    df = pd.read_csv(FILE_PATH)
except UnicodeDecodeError:
    try:
        # Try reading with latin1 encoding if UTF-8 fails
        df = pd.read_csv(FILE_PATH, encoding='latin1')
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
        exit()
except FileNotFoundError:
    print(f"Error: File not found at {FILE_PATH}")
    exit()

# --- 2. Data Preprocessing and Feature Engineering ---

# Define the binary target variable: 1 if BS >= threshold, 0 otherwise
target_column_name = f'HighBS_{BS_THRESHOLD}'
df[target_column_name] = (df[TARGET_VARIABLE] >= BS_THRESHOLD).astype(int)

# Select features and the new target
features = ['Age', 'SystolicBP', 'DiastolicBP', 'BodyTemp', 'HeartRate']
X = df[features]
y = df[target_column_name]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 3. Define and Train Multiple ML Models ---
models = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Support Vector Machine": SVC(probability=True, random_state=42), # probability=True for consistency if needed later
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

results = {}
best_model_name = None
best_accuracy = 0.0
best_model = None
best_model_metrics = None

print("--- Training and Evaluating Models ---")
for name, model in models.items():
    # Train the model
    model.fit(X_train_scaled, y_train)

    # Predict on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y_test, y_pred)

    results[name] = {
        "model": model,
        "accuracy": accuracy,
        "classification_report": report,
        "confusion_matrix": conf_matrix
    }

    print(f"\nModel: {name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(conf_matrix)

    # Check if this is the best model so far based on accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model
        best_model_metrics = results[name] # Store all metrics for the best model

print(f"\n--- Best Performing Model: {best_model_name} ---")
print(f"Best Accuracy: {best_accuracy:.4f}")

# --- 4. Save Statistics and Detailed Metrics of the Best Model ---
Accuracy = best_accuracy # Save the best accuracy score
Detailed_Metrics = best_model_metrics['classification_report']
Confusion_Matrix = best_model_metrics['confusion_matrix']

print("\n--- Best Model Statistics ---")
print(f"Accuracy Variable: {Accuracy:.4f}")
print("Detailed Metrics (Classification Report):")
print(Detailed_Metrics)
print("Confusion Matrix:")
print(Confusion_Matrix)


# --- 5. Save the Trained Best Model and Scaler ---
best_model_filename = os.path.join(MODEL_OUTPUT_FOLDER, f'best_bs_prediction_{MODEL_TYPE_CLASSIFICATION}_model.joblib')
scaler_filename = os.path.join(MODEL_OUTPUT_FOLDER, 'bs_prediction_scaler.joblib')

joblib.dump(best_model, best_model_filename)
joblib.dump(scaler, scaler_filename)

print(f"\n--- Model Saved ---")
print(f"Best model ({best_model_name}) saved to: {best_model_filename}")
print(f"Scaler saved to: {scaler_filename}")

# --- 6. Prepare Script for Running the Best Model on Other Data ---

def predict_high_bs(new_data, model_path, scaler_path):
    """
    Loads the trained model and scaler to predict high blood sugar likelihood on new data.

    Args:
        new_data (pd.DataFrame): DataFrame containing the new data with columns
                                 matching the training features ('Age', 'SystolicBP',
                                 'DiastolicBP', 'BodyTemp', 'HeartRate').
        model_path (str): Path to the saved trained model file (.joblib).
        scaler_path (str): Path to the saved scaler file (.joblib).

    Returns:
        np.ndarray: An array of predictions (0 or 1), where 1 indicates a high
                    likelihood of BS >= threshold.
        np.ndarray: An array of probabilities for the positive class (High BS),
                    if the model supports predict_proba. Otherwise, None.
    """
    try:
        # Load the scaler and model
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
    except FileNotFoundError as e:
        print(f"Error loading model or scaler: {e}")
        return None, None
    except Exception as e:
        print(f"An error occurred during loading: {e}")
        return None, None

    # Ensure input data has the correct columns in the correct order
    required_features = ['Age', 'SystolicBP', 'DiastolicBP', 'BodyTemp', 'HeartRate']
    if not all(feature in new_data.columns for feature in required_features):
        print(f"Error: Input data must contain columns: {required_features}")
        return None, None
    
    new_data_features = new_data[required_features]

    # Scale the new data
    new_data_scaled = scaler.transform(new_data_features)

    # Make predictions
    predictions = model.predict(new_data_scaled)

    # Get probabilities if the model supports it
    probabilities = None
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(new_data_scaled)[:, 1] # Probability of class 1

    return predictions, probabilities

# --- Example Usage of the Prediction Function (Commented out by default) ---
# """
# print("\n--- Example Prediction on New Data ---")
# # Create sample new data (replace with actual new data)
# sample_data = pd.DataFrame({
#     'Age': [30, 45, 22],
#     'SystolicBP': [135, 110, 120],
#     'DiastolicBP': [85, 70, 80],
#     'BodyTemp': [98.6, 99.0, 98.0],
#     'HeartRate': [75, 80, 68]
# })

# # Make predictions using the saved model and scaler
# predictions, probabilities = predict_high_bs(sample_data, best_model_filename, scaler_filename)

# if predictions is not None:
#     print("Sample Data:")
#     print(sample_data)
#     print("\nPredictions (1 = High BS Risk, 0 = Lower BS Risk):")
#     print(predictions)
#     if probabilities is not None:
#         print("\nProbabilities of High BS Risk:")
#         print(probabilities)
# """

print("\n--- Script Execution Completed ---")