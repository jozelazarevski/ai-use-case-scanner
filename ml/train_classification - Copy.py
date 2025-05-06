import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, QuantileTransformer, OneHotEncoder, FunctionTransformer

from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import joblib
import os
import chardet
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from ml.read_file import read_data_flexible

# target_variable = "y"  # Default target variable


import pandas as pd
import chardet
import io
import csv
import json

"""
filename = 'pricerunner_aggregate.csv'
filename = "UCI CBM Dataset.txt"
filename = "crx.data"
filename = "bank-full.csv"
target_variable = "StockCode"
filename="Online_Retail.xlsx"
filepath = f"../uploads/{filename}"
"""


def train_classification_model(filepath, target_variable, output_dir="models"):
    """
    Predicts the values of a specified target variable in a dataset using multiple models
    and saves the best performing one along with preprocessing artifacts.

    Args:
        filepath (str): The path to the CSV or Excel file containing the data.
        target_variable (str): The name of the column to be predicted.
        output_dir (str, optional): Directory to save the best model and related files.
                                     Defaults to "models".

    Returns:
        tuple: A tuple containing:
            - model_filename (str): Path to the saved best model.
            - features_filename (str): Path to the saved feature names used by the model.
            - best_accuracy (float): Accuracy of the best performing model on the validation set.
            - original_feature_importances (dict): Dictionary of feature importances for the
              original features (averaged if one-hot encoded). Returns None if the best
              model doesn't support feature importance.
    """
    try:
        if filepath.endswith('.xlsx'):
            df = pd.read_excel(filepath)
        elif filepath.endswith('.csv'):
            df = pd.read_csv(filepath)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None, None, None

    if target_variable not in df.columns:
        print(f"Error: Target variable '{target_variable}' not found in the DataFrame.")
        return None, None, None, None

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    df_processed = df.copy()
    y = df_processed[target_variable]
    X = df_processed.drop(columns=[target_variable])

    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns
    numerical_features = X.select_dtypes(include=np.number).columns
    original_columns = X.columns.tolist()

    # Preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'  # Keep other columns (if any)
    )

    X_processed = preprocessor.fit_transform(X)
    X_for_model = pd.DataFrame(X_processed, columns=preprocessor.get_feature_names_out())

    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X_for_model, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

    # Define models
    models = {
        'GaussianNB': GaussianNB(),
        'LogisticRegression': LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'GradientBoosting': GradientBoostingClassifier(random_state=42)
    }

    best_model = None
    best_accuracy = 0.0
    best_auc_roc = 0.0
    best_report = None

    for name, model in models.items():
        print(f"\nTraining and evaluating {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred, target_names=label_encoder.classes_, zero_division=0)

        # Calculate AUC-ROC if it's a binary or multi-class problem (ovr for multi-class)
        if len(label_encoder.classes_) > 2:
            y_pred_proba = model.predict_proba(X_val)
            auc_roc = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
        elif len(label_encoder.classes_) == 2:
            try:
                y_pred_proba = model.predict_proba(X_val)[:, 1]
                auc_roc = roc_auc_score(y_val, y_pred_proba)
            except AttributeError:
                auc_roc = 0.0 # Some models might not have predict_proba
        else:
            auc_roc = 0.0

        print(f"{name} - Accuracy: {accuracy:.4f}, AUC-ROC: {auc_roc:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_auc_roc = auc_roc
            best_model = model
            best_report = report

    print(f"\nBest Model: {type(best_model).__name__}, Accuracy: {best_accuracy:.4f}, AUC-ROC: {best_auc_roc:.4f}")
    print(f"\nBest Model Classification Report:\n{best_report}")

    # Feature Importance (Corrected)
    best_feature_importances = None
    if hasattr(best_model, "feature_importances_"):
        best_feature_importances = dict(
            zip(X_for_model.columns, best_model.feature_importances_))
    elif hasattr(best_model, "coef_"):
        # For Logistic Regression, handle multi-class by averaging coefficients
        if len(best_model.coef_.shape) > 1:
            avg_coef = np.mean(np.abs(best_model.coef_), axis=0)
            best_feature_importances = dict(zip(X_for_model.columns, avg_coef))
        else:
            best_feature_importances = dict(zip(X_for_model.columns, best_model.coef_[0]))
    else:
        print("Best model does not support feature importance.")

    original_feature_importances = {}
    if best_feature_importances:
        for processed_feature, importance in best_feature_importances.items():
            # Find the original column(s) that contributed to this processed feature
            mapped_original_names = []
            for original_col in original_columns:
                if processed_feature.startswith(f"{original_col}_") or processed_feature == original_col:
                    mapped_original_names.append(original_col)
                elif processed_feature == original_col: # For numerical features
                    mapped_original_names.append(original_col)

            if mapped_original_names:
                original_feature_importances[", ".join(mapped_original_names)] = \
                    importance / len(mapped_original_names)  # Average importance
            else:
                original_feature_importances[processed_feature] = importance

    # Save model, preprocessor, and feature info
    model_filename = os.path.join(output_dir,
                                    f"best_model_{target_variable.replace(' ', '_')}.joblib")
    joblib.dump(best_model, model_filename)
    preprocessor_filename = os.path.join(output_dir,
                                         f"preprocessor_{target_variable.replace(' ', '_')}.joblib")  # Save preprocessor
    joblib.dump(preprocessor, preprocessor_filename)
    features_filename = os.path.join(output_dir,
                                     f"model_features_{target_variable.replace(' ', '_')}.joblib")
    joblib.dump(X_for_model.columns, features_filename)  # Save feature names

    return model_filename, features_filename, best_accuracy, original_feature_importances

if __name__ == "__main__":
    filename = "Online_Retail.xlsx"
    filepath = f"../uploads/{filename}"
    target_variable = "StockCode"
    output_directory = "stock_code_models" # Specify an output directory

    # Example usage to predict StockCode
    model_path, features_path, accuracy, feature_importances = train_classification_model(
        filepath, target_variable, output_dir=output_directory
    )

    if model_path:
        print(f"\nBest model saved to: {model_path}")
        print(f"Model features saved to: {features_path}")
        print(f"Best model accuracy on validation set: {accuracy:.4f}")
        if feature_importances:
            print("\nOriginal Feature Importances:")
            for feature, importance in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True):
                print(f"{feature}: {importance:.4f}")

    # Example of predicting a different target variable (if it exists)
    # target_variable_country = "Country"
    # output_directory_country = "country_models"
    # model_path_country, features_path_country, accuracy_country, feature_importances_country = predict_target_variable(
    #     filepath, target_variable_country, output_dir=output_directory_country
    # )
    #
    # if model_path_country:
    #     print(f"\nBest model for {target_variable_country} saved to: {model_path_country}")
    #     print(f"Model features saved to: {features_path_country}")
    #     print(f"Best model accuracy on validation set: {accuracy_country:.4f}")
    #     if feature_importances_country:
    #         print("\nOriginal Feature Importances for {target_variable_country}:")
    #         for feature, importance in sorted(feature_importances_country.items(), key=lambda item: item[1], reverse=True):
    #             print(f"{feature}: {importance:.4f}")