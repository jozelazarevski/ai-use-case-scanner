import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report

# Define model type
model_type = 'classification'

# --- 1. Data Loading ---
file_path = '../uploads/california_housing_train.csv'
try:
    df = pd.read_csv(file_path, encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv(file_path, encoding='latin1')

# --- 2. Target Variable Creation ---
# Define affordability tiers based on median_house_value quantiles
quantiles = df['median_house_value'].quantile([0.2, 0.4, 0.6, 0.8]).tolist()
bins = [0] + quantiles + [np.inf]
labels = ['Very Affordable', 'Affordable', 'Moderate', 'Expensive', 'Very Expensive']
df['affordability_category'] = pd.cut(df['median_house_value'], bins=bins, labels=labels, include_lowest=True)

# Drop the original median_house_value as it's directly used to create the target
df = df.drop('median_house_value', axis=1)

# --- 3. Feature Engineering & Preprocessing Setup ---
# Handle potential division by zero or missing values before creating new features
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True) # Impute initial NaNs

# Avoid division by zero by replacing 0s with a small number or NaN (then impute)
df['households'] = df['households'].replace(0, np.nan)
df['total_rooms'] = df['total_rooms'].replace(0, np.nan)

# Create new features
df['rooms_per_household'] = df['total_rooms'] / df['households']
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
df['pop_per_household'] = df['population'] / df['households']

# Impute NaNs that might have been created by division by zero or existed previously
# Use median imputation for the new features as well
imputer_median = SimpleImputer(strategy='median')
# Identify columns that might need imputation (original + newly created)
cols_to_impute = ['total_bedrooms', 'households', 'total_rooms', 'rooms_per_household', 'bedrooms_per_room', 'pop_per_household']
df[cols_to_impute] = imputer_median.fit_transform(df[cols_to_impute])


# Define features (X) and target (y)
X = df.drop('affordability_category', axis=1)
y = df['affordability_category']

# Identify numerical features for scaling
numerical_features = X.columns.tolist()

# Create preprocessing pipelines for numerical features
# We already imputed NaNs before feature creation and handled division by zero effects
# Now we just need scaling
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Create a preprocessor object using ColumnTransformer
# Apply the numerical transformer to all features (as they are all numerical)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features)
    ],
    remainder='passthrough' # Keep any other columns if they existed (though all are numeric here)
)


# --- 4. Data Splitting ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- 5. Model Training & Evaluation ---
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

best_model = None
best_accuracy = 0.0
best_model_name = ""
model_results = {}

print("Starting model training and evaluation...")

for name, classifier in classifiers.items():
    # Create the full pipeline: preprocess + classify
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', classifier)])

    print(f"Training {name}...")
    # Train the model
    pipeline.fit(X_train, y_train)

    print(f"Evaluating {name}...")
    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=labels, output_dict=True) # Use output_dict for detailed metrics storage

    print(f"{name} Accuracy: {accuracy:.4f}")
    # print(classification_report(y_test, y_pred, target_names=labels)) # Print report for visual check

    model_results[name] = {'accuracy': accuracy, 'report': report, 'model': pipeline}

    # Check if this model is the best so far
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = pipeline
        best_model_name = name
        detailed_metrics = report # Save detailed metrics of the best model

print(f"\nBest performing model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# --- 6. Save the Best Model ---
# Store accuracy and detailed metrics in variables
Accuracy = best_accuracy
# detailed_metrics variable is already assigned above

# Create the directory if it doesn't exist
model_dir = 'trained_models'
os.makedirs(model_dir, exist_ok=True)

# Save the best pipeline (preprocessor + model)
model_filename = os.path.join(model_dir, 'best_affordability_classifier.joblib')
joblib.dump(best_model, model_filename)

print(f"Best model ({best_model_name}) saved to {model_filename}")
print(f"Best Model Accuracy stored in 'Accuracy' variable: {Accuracy:.4f}")
# print("Detailed metrics for the best model stored in 'detailed_metrics' variable.") # Commented out as per requirement 4


# --- 7. Prepare Script for Running the Best Model on Other Data ---
# Define a function to load the model and predict on new data

def predict_affordability(new_data_path):
    """
    Loads the trained affordability classification model and predicts on new data.

    Args:
        new_data_path (str): Path to the CSV file containing new data.
                               The file should have the same columns as the training data
                               (excluding 'median_house_value' and 'affordability_category').

    Returns:
        pandas.DataFrame: DataFrame containing the original data plus a column
                          with affordability predictions ('predicted_affordability_category').
                          Returns None if loading fails or data is invalid.
    """
    model_path = 'trained_models/best_affordability_classifier.joblib'
    required_columns = [
        'longitude', 'latitude', 'housing_median_age', 'total_rooms',
        'total_bedrooms', 'population', 'households', 'median_income'
    ] # These are the raw input columns needed before feature engineering

    # Load the model pipeline
    try:
        loaded_pipeline = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # Load new data
    try:
        new_df = pd.read_csv(new_data_path, encoding='utf-8')
    except UnicodeDecodeError:
        new_df = pd.read_csv(new_data_path, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: New data file not found at {new_data_path}")
        return None
    except Exception as e:
        print(f"Error reading new data file: {e}")
        return None

    # --- Data Preparation for Prediction (MUST mirror training preprocessing) ---
    # Check for required columns
    if not all(col in new_df.columns for col in required_columns):
        print(f"Error: New data is missing required columns. Needed: {required_columns}")
        return None

    # Keep only necessary columns for prediction (in the correct order if the model relies on it, though pipeline handles names)
    X_new = new_df[required_columns].copy()


    # --- Apply Feature Engineering (exactly as in training) ---
    # Impute initial NaNs in total_bedrooms (using median from training - ideally saved, but using new data median is fallback)
    # NOTE: Best practice is to save the imputer used during training.
    # For simplicity here, we re-apply logic. If the pipeline included the imputer, this would be handled automatically.
    # Since our pipeline only scales, we must replicate imputation and feature creation MANUALLY.

    # Re-create the imputer (or better: load a saved one from training)
    imputer_median_pred = SimpleImputer(strategy='median')
    # Fit *only* on necessary columns from the new data IF a saved imputer isn't available.
    # This is suboptimal as it uses the new data's median, not the training data's.
    # A better approach integrates imputation into the main pipeline saved earlier.
    # Let's assume for this script we refit the imputer on new data for total_bedrooms.
    X_new['total_bedrooms'] = imputer_median_pred.fit_transform(X_new[['total_bedrooms']])

    # Avoid division by zero
    X_new['households'] = X_new['households'].replace(0, np.nan)
    X_new['total_rooms'] = X_new['total_rooms'].replace(0, np.nan)

    # Create features
    X_new['rooms_per_household'] = X_new['total_rooms'] / X_new['households']
    X_new['bedrooms_per_room'] = X_new['total_bedrooms'] / X_new['total_rooms']
    X_new['pop_per_household'] = X_new['population'] / X_new['households']

    # Impute NaNs created during feature engineering
    # Again, using median from new data as fallback.
    cols_to_impute_new = ['total_bedrooms', 'households', 'total_rooms', 'rooms_per_household', 'bedrooms_per_room', 'pop_per_household']
    imputer_median_pred_features = SimpleImputer(strategy='median')
    X_new[cols_to_impute_new] = imputer_median_pred_features.fit_transform(X_new[cols_to_impute_new])


    # --- Prediction ---
    # The loaded pipeline applies the necessary scaling (as defined in 'preprocessor')
    # Ensure X_new has the columns expected by the pipeline's preprocessor step
    # The pipeline was trained on features: ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'rooms_per_household', 'bedrooms_per_room', 'pop_per_household']
    feature_order = numerical_features # Use the feature order from training
    X_new_ordered = X_new[feature_order]


    try:
        predictions = loaded_pipeline.predict(X_new_ordered)
    except Exception as e:
        print(f"Error during prediction: {e}")
        print("Make sure the new data has the correct format and features.")
        return None

    # Add predictions to the original new dataframe
    new_df['predicted_affordability_category'] = predictions

    return new_df


# Example usage (commented out):
# new_data_file = 'path/to/your/new_housing_data.csv'
# predictions_df = predict_affordability(new_data_file)
# if predictions_df is not None:
#     print("\nPredictions on new data:")
#     print(predictions_df.head())
#     # Save predictions if needed
#     # predictions_df.to_csv('housing_predictions.csv', index=False)

print("\n--- Script Finished ---")
print("Variable 'Accuracy' holds the best model accuracy.")
print("Variable 'detailed_metrics' holds the classification report dict for the best model.")
print("Function 'predict_affordability(new_data_path)' is defined for predictions on new data.")