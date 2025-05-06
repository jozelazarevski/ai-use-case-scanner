import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
import chardet
import numpy as np
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from ml.read_file import read_data_flexible
from sklearn.model_selection import train_test_split, KFold  # Import KFold
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge

"""
filename = 'pricerunner_aggregate.csv'
filename = "UCI CBM Dataset.txt"
filename = "crx.data"
filename = "bank-full.csv"
target_variable = "y"
filename="bike_sharing_hour_regression.csv"
target_variable=""
filename = "online_retail_II_small.xls"
target_variable = "Quantity"
filepath="../uploads/"+filename

"""


 
    
def train_regression_model(target_variable, filepath):
    """
    Trains and evaluates multiple regression models using a single train-test split.
    Handles mixed data types, including datetime, with extensive debugging, and calculates feature importance.
    """

    output_dir = "trained_models"
    os.makedirs(output_dir, exist_ok=True)
    best_feature_importances = {}  # Initialize for best model feature importances

    print(f"Starting regression training for target: '{target_variable}' on file: '{filepath}'")

    df = read_data_flexible(filepath)
    if df is None:
        print("Error: Failed to read the data. Training aborted.")
        return None, None, 0, best_feature_importances

    X = df.drop(columns=[target_variable])
    y = df[target_variable]
    
    # Store original column names
    original_columns = list(df.columns)

    print(f"Initial X shape: {X.shape}, y shape: {y.shape}")

    # --- Data Type Handling ---
    numerical_cols = X.select_dtypes(include=['number']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    datetime_cols = X.select_dtypes(include=['datetime64']).columns

    for col in numerical_cols:
        try:
            X[col] = pd.to_numeric(X[col], errors='coerce')
        except ValueError:
            print(f"Column '{col}' cannot be converted to numeric.")

    print(f"X shape before dropna: {X.shape}")
    X = X.dropna(subset=numerical_cols)
    print(f"X shape after dropna: {X.shape}")

    y = y.loc[X.index]
    print(f"y shape after aligning with X: {y.shape}")

    for col in categorical_cols:
        X[col] = X[col].astype(str)

    for col in datetime_cols:
        X[col] = pd.to_numeric(X[col])
        if col not in numerical_cols:
            numerical_cols = numerical_cols.append(pd.Index([col]))
    # --- End Data Type Handling ---

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )

    X = preprocessor.fit_transform(X)
    feature_names_after_preprocessing = preprocessor.get_feature_names_out()

    # Convert back to DataFrame
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names_after_preprocessing)
    else:
        X = pd.DataFrame(X.toarray(), columns=feature_names_after_preprocessing)

    # --- Quantile Transformer ---
    if len(numerical_cols) > 1:
        qt = QuantileTransformer(output_distribution='normal',
                                n_quantiles=min(len(df), 100))
        numerical_features_after_ohe = [col for col in feature_names_after_preprocessing
                                       if col in numerical_cols]
        if numerical_features_after_ohe:
            X_transformed = qt.fit_transform(X[numerical_features_after_ohe])
            X_transformed_numerical = pd.DataFrame(X_transformed,
                                                 columns=numerical_features_after_ohe)
            X_combined = pd.concat(
                [X_transformed_numerical, X.drop(columns=numerical_cols,
                                                errors='ignore')], axis=1)
        else:
            print("No numerical columns left after OHE. Skipping QuantileTransformer.")
            X_combined = X.copy()
    else:
        print("Not enough numerical features for QuantileTransformer, skipping.")
        X_combined = X.copy()
    # --- End Quantile Transformer ---

    models = {
           
        'XGBoost': XGBRegressor(random_state=42),
         'Ridge Regression': Ridge(alpha=1.0),  # alpha is the regularization strength
        # 'Extra Trees Regressor': ExtraTreesRegressor(random_state=42),
         
    }

    best_model = None
    best_r2 = -float('inf')
    best_feature_set = ''

    # --- Train-Test Split (No Cross-Validation) ---
    X_train, X_test, y_train, y_test = train_test_split(   X, y, test_size=0.2, random_state=42)  # Adjust test_size as needed
    # X_combined_train, X_combined_test, y_train_combined, y_test_combined = train_test_split( X_combined, y, test_size=0.2, random_state=42)

    for name, model in models.items():
        print(f"\nTraining and evaluating {name}...")

        # Evaluate on original features
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        print(f"  R2 (Original Features): {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_feature_set = 'Original'

        # Evaluate on combined features
        # model.fit(X_combined_train, y_train_combined)
        # y_pred_combined = model.predict(X_combined_test)
        # r2_combined = r2_score(y_test_combined, y_pred_combined)
        # print(f"  R2 (Combined Features): {r2_combined:.4f}")
        # if r2_combined > best_r2:
        #     best_r2 = r2_combined
        #     best_model = model
        #     best_feature_set = 'Combined'
    # --- End Train-Test Split ---

    print(f"\nBest Model: {type(best_model).__name__}, Feature Set: {best_feature_set}, Best R2: {best_r2:.4f}")

    # Get feature importances from the best model
    if hasattr(best_model, "feature_importances_"):
        if best_feature_set == "Original":
            best_feature_importances = dict(zip(X.columns, best_model.feature_importances_))
        else:
            best_feature_importances = dict(zip(X_combined.columns, best_model.feature_importances_))

    elif hasattr(best_model, "coef_"):
        if best_feature_set == "Original":
            best_feature_importances = dict(zip(X.columns, best_model.coef_))  # For LinearRegression
        else:
            best_feature_importances = dict(zip(X_combined.columns, best_model.coef_))
    else:
        best_feature_importances = {}
        print("Best model does not support feature importance.")
        
    # Map feature importance names back to original column names
    original_feature_importances = {}
    if best_feature_importances:
        feature_name_mapping = dict(zip(feature_names_after_preprocessing, X.columns if best_feature_set == "Original" else X_combined.columns))
        for processed_feature, importance in best_feature_importances.items():
            # Remove 'num__' or 'cat__' prefixes
            processed_feature = processed_feature.replace("num__", "").replace("cat__", "")
            
            # Find the original column(s) that contributed to this processed feature
            mapped_original_names = []
            for original_col in original_columns:
                if processed_feature.startswith(original_col):  # Check if the processed feature name starts with the original column name
                    mapped_original_names.append(original_col)
            
            if mapped_original_names:
                # If the processed feature maps to multiple original columns (e.g., from OneHotEncoding),
                # you might want to sum/average the importances or handle them differently.
                # For simplicity, we'll average here. You might need a more sophisticated approach.
                original_feature_importances[", ".join(mapped_original_names)] = importance / len(mapped_original_names)
            else:
                original_feature_importances[processed_feature] = importance  # Keep the processed name if no mapping is found

    # model_filename = os.path.join(output_dir,f"best_model_{target_variable.replace(' ', '_')}.joblib")
    model_filename = os.path.join(output_dir,f"best_model_{target_variable}.joblib")

    print(f"Saving best model to: {model_filename}")
    joblib.dump(best_model, model_filename)
    
    # Save preprocessor
    preprocessor_filename = os.path.join(output_dir, f"preprocessor_{target_variable}.joblib")
    print(f"Saving preprocessor to: {preprocessor_filename}")
    joblib.dump(preprocessor, preprocessor_filename)

    feature_names = X.columns if best_feature_set == 'Original' else X_combined.columns
    features_filename = os.path.join(output_dir,
                                     f"model_features_{target_variable.replace(' ', '_')}.joblib")
    print(f"Saving feature names to: {features_filename}")
    joblib.dump(feature_names, features_filename)
    accuracy =best_r2
    return model_filename, features_filename, accuracy, original_feature_importances