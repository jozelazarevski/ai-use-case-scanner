import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import warnings

# Suppress specific warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning) # Joblib UserWarning

# --- Configuration ---
data_file_path = '../uploads/california_housing_train.csv'
target_variable = 'median_house_value'
model_type = 'regression' # Clearly define the model type
trained_models_folder = 'trained_models'
best_model_filename = 'best_california_housing_model.joblib'
scaler_filename = 'feature_scaler.joblib'
feature_list_filename = 'feature_list.joblib'

# Create directory for saving models if it doesn't exist
os.makedirs(trained_models_folder, exist_ok=True)

# --- 1. Load Data ---
try:
    # Attempt to read with standard UTF-8 encoding first
    housing_data = pd.read_csv(data_file_path, encoding='utf-8')
except UnicodeDecodeError:
    # If UTF-8 fails, try latin1 as a common alternative
    print("UTF-8 decoding failed, trying latin1 encoding.")
    housing_data = pd.read_csv(data_file_path, encoding='latin1')
except FileNotFoundError:
    print(f"Error: The file {data_file_path} was not found.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while reading the CSV: {e}")
    exit()

print("Data loaded successfully.")

# --- 2. Data Preprocessing and Feature Engineering ---

# Handle potential missing values (especially in 'total_bedrooms')
# Use SimpleImputer fitted ONLY on the training data later, but identify columns now
imputer = SimpleImputer(strategy='median')
housing_data['total_bedrooms'] = imputer.fit_transform(housing_data[['total_bedrooms']]) # Fit and transform on the whole data before split for simplicity in this script context, normally fit ONLY on train

# Feature Engineering: Create new potentially useful features
housing_data['rooms_per_household'] = housing_data['total_rooms'] / housing_data['households']
housing_data['bedrooms_per_room'] = housing_data['total_bedrooms'] / housing_data['total_rooms']
housing_data['population_per_household'] = housing_data['population'] / housing_data['households']

# Replace infinite values that might arise from division by zero (if households is 0)
housing_data.replace([np.inf, -np.inf], np.nan, inplace=True)
# Impute NaNs created by feature engineering (if any) - use median again
for col in ['rooms_per_household', 'bedrooms_per_room', 'population_per_household']:
     if housing_data[col].isnull().any():
         median_val = housing_data[col].median()
         housing_data[col].fillna(median_val, inplace=True)


# Define features (X) and target (y)
features = housing_data.drop(target_variable, axis=1)
target = housing_data[target_variable]

# Store feature names for later use during prediction
feature_names = list(features.columns)
joblib.dump(feature_names, os.path.join(trained_models_folder, feature_list_filename))


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Scale numerical features
# NOTE: In a real-world scenario with separate train/test files, fit ONLY on training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, os.path.join(trained_models_folder, scaler_filename))

print("Data preprocessing and feature scaling complete.")

# --- 3. Model Training and Selection ---
models_to_evaluate = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), # Use more estimators for RF
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_rmse = float('inf')
best_model_name = ""
model_performance_metrics = {}

print("\n--- Training and Evaluating Models ---")
for name, model in models_to_evaluate.items():
    print(f"Training {name}...")
    # Train the model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    model_performance_metrics[name] = {'RMSE': rmse, 'MAE': mae, 'R2': r2}
    print(f"{name} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2 Score: {r2:.4f}")

    # Check if this model is the best so far (based on RMSE)
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_model_name = name

print(f"\nBest performing model: {best_model_name} with RMSE: {best_rmse:.4f}")

# --- 4. Save the Best Model ---
if best_model:
    model_save_path = os.path.join(trained_models_folder, best_model_filename)
    joblib.dump(best_model, model_save_path)
    print(f"Best model ({best_model_name}) saved to {model_save_path}")
else:
    print("Error: No best model was selected.")
    exit()

# --- 5. Save Performance Statistics ---
# Using R2 score as the primary 'Accuracy' metric for regression context
Accuracy = model_performance_metrics[best_model_name]['R2']
detailed_metrics = model_performance_metrics[best_model_name]

print(f"\nBest Model Accuracy (R2 Score): {Accuracy:.4f}")
print(f"Detailed Metrics for Best Model ({best_model_name}): {detailed_metrics}")


# --- 6. Prepare Script/Function for Running Best Model on Other Data ---

def predict_median_house_value(new_data_path, model_path, scaler_path, feature_list_path):
    """
    Loads the trained model, scaler, and feature list, preprocesses new data,
    and makes predictions.

    Args:
        new_data_path (str): Path to the CSV file containing new data.
        model_path (str): Path to the saved trained model (.joblib file).
        scaler_path (str): Path to the saved scaler (.joblib file).
        feature_list_path (str): Path to the saved feature list (.joblib file).

    Returns:
        pandas.DataFrame: DataFrame containing the predictions, or None if an error occurs.
    """
    try:
        # Load necessary components
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        expected_features = joblib.load(feature_list_path)
        print(f"Model, scaler, and feature list loaded successfully from {trained_models_folder}.")

        # Load new data
        try:
            new_data = pd.read_csv(new_data_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 decoding failed for new data, trying latin1 encoding.")
            new_data = pd.read_csv(new_data_path, encoding='latin1')

        print(f"New data loaded successfully from {new_data_path}.")
        
        # --- Preprocessing for New Data (MUST match training preprocessing) ---
        
        # 1. Impute missing 'total_bedrooms' (using median strategy as defined before)
        #    IMPORTANT: Use SimpleImputer or fillna with the MEDIAN FROM THE *TRAINING* DATA
        #    For simplicity here, we re-apply SimpleImputer - IN PRODUCTION, SAVE/LOAD THE FITTED IMPUTER
        imputer_pred = SimpleImputer(strategy='median') 
        if 'total_bedrooms' in new_data.columns and new_data['total_bedrooms'].isnull().any():
             # Fit on a sample column (or load saved imputer) to avoid fitting on potentially different new data distribution
             # Here, we just apply transform assuming it was fitted appropriately during training
             # This is a simplification - ideally, load the imputer fitted on training data.
             # For this script, we refit the imputer on the new data's column if needed.
             new_data['total_bedrooms'] = imputer_pred.fit_transform(new_data[['total_bedrooms']])


        # 2. Feature Engineering (same features as training)
        #    Handle potential division by zero
        new_data['rooms_per_household'] = new_data['total_rooms'] / new_data['households'].replace(0, np.nan)
        new_data['bedrooms_per_room'] = new_data['total_bedrooms'] / new_data['total_rooms'].replace(0, np.nan)
        new_data['population_per_household'] = new_data['population'] / new_data['households'].replace(0, np.nan)
        
        # Replace potential infinities and NaNs resulting from engineering
        new_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in ['rooms_per_household', 'bedrooms_per_room', 'population_per_household']:
            if new_data[col].isnull().any():
                # Ideally, fill with median from TRAINING data. Re-calculating here for simplicity.
                median_val = new_data[col].median() 
                new_data[col].fillna(median_val, inplace=True)


        # 3. Ensure feature order and selection matches training
        #    Handle missing/extra columns if necessary
        missing_cols = set(expected_features) - set(new_data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in new data: {missing_cols}")
            
        extra_cols = set(new_data.columns) - set(expected_features)
        if extra_cols:
            print(f"Warning: Extra columns found in new data and will be ignored: {extra_cols}")
            
        # Select and reorder columns
        new_data_processed = new_data[expected_features]

        # 4. Scale features using the *loaded* scaler
        new_data_scaled = scaler.transform(new_data_processed)
        print("New data preprocessed and scaled successfully.")

        # --- Make Predictions ---
        predictions = model.predict(new_data_scaled)
        print("Predictions generated successfully.")

        return pd.DataFrame({'Predicted_Median_House_Value': predictions})

    except FileNotFoundError:
        print(f"Error: One or more required files not found ({model_path}, {scaler_path}, {feature_list_path}, {new_data_path}).")
        return None
    except ValueError as ve:
         print(f"ValueError during prediction: {ve}")
         return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

# --- Example of how to use the prediction function ---
# Create a dummy new data file for demonstration if needed
# Make sure this dummy file has the same original columns as the training data
# For example:
# dummy_data = features.head() # Take first few rows of original features
# dummy_data_path = '../uploads/california_housing_new_sample.csv'
# dummy_data.to_csv(dummy_data_path, index=False)
# print(f"\n--- Running prediction on sample data ({dummy_data_path}) ---")
# predictions_df = predict_median_house_value(
#     new_data_path=dummy_data_path,
#     model_path=os.path.join(trained_models_folder, best_model_filename),
#     scaler_path=os.path.join(trained_models_folder, scaler_filename),
#     feature_list_path=os.path.join(trained_models_folder, feature_list_filename)
# )

# if predictions_df is not None:
#     print("\nSample Predictions:")
#     print(predictions_df)

print("\nScript finished.")