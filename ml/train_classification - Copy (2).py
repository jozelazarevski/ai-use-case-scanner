import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib
import os
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Import preprocessing modules (try different import approaches)
try:
    # Try absolute import
    from ml.preprocess_data import preprocess_data, save_preprocessor, get_output_dir, get_model_filename
    from ml.read_file import read_data_flexible
except ImportError:
    try:
        # Try direct import
        from preprocess_data import preprocess_data, save_preprocessor, get_output_dir, get_model_filename
        from read_file import read_data_flexible
    except ImportError:
        # Try relative import
        from .preprocess_data import preprocess_data, save_preprocessor, get_output_dir, get_model_filename
        from .read_file import read_data_flexible

"""
target_variable="pets_allowed"
filepath="../uploads/apartments_for_rent_classified_10K.csv"
user_id='1'
use_case=target_variable
train_classification_model(target_variable, filepath, user_id, use_case) 


"""

def train_classification_model(target_variable, filepath, user_id, use_case):
    """



    Trains a classification model on the given dataset and calculates feature importance for the best model.

    Args:
        target_variable (str): The name of the target variable column.
        filepath (str): The path to the data file.
        user_id (str): User ID for organization.
        use_case (str): Use case identifier.

    Returns:
        tuple: A tuple containing the model filename, feature names,
               the accuracy of the best performing model, and feature importances
               of the best model using original column names.
    """
    # Extract filename from filepath
    filename = os.path.basename(filepath)
    
    # Get output directory for this model under databases/trained_models/{user_id}/
    output_dir = get_output_dir(user_id, use_case, target_variable, filename)
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate model filename
    model_filename = get_model_filename(user_id, use_case, target_variable, filename)
    model_path = os.path.join(output_dir, model_filename)
    
    # Initialize variables
    accuracy = 0
    best_feature_importances = {}  # Initialize for best model feature importances

    print(f"Starting training process for target: '{target_variable}' on file: '{filepath}'")
    print(f"Model will be saved to: {model_path}")
    print(f"Using directory structure: databases/trained_models/{user_id}/{use_case}_{target_variable}_{os.path.basename(filepath)}.joblib")

      
    # Read the data using the flexible reader function
    df = read_data_flexible(filepath)
    
    if df is None:
        print("Error: Failed to read the data. Training aborted.")
        return None, None, accuracy, best_feature_importances
    
    # Store original column names
    original_columns = list(df.columns)
    
    # Ensure target variable is properly encoded for classification
    if df[target_variable].dtype == 'object' or df[target_variable].dtype.name == 'category':
        print(f"Target variable '{target_variable}' is categorical. Converting to numerical.")
        # Use LabelEncoder for consistent encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        df[target_variable] = le.fit_transform(df[target_variable])
        print(f"Target variable '{target_variable}' encoded categories:", le.classes_)
        print(f"Target variable '{target_variable}' unique values after encoding:", df[target_variable].unique())
    elif df[target_variable].dtype in ['int64', 'float64']:
        # Check if there are negative values or non-sequential integers
        unique_values = df[target_variable].unique()
        min_value = unique_values.min()
        max_value = unique_values.max()
        if min_value < 0 or not np.array_equal(np.sort(unique_values), np.arange(min_value, min_value + len(unique_values))):
            print(f"Target variable has non-standard values. Current range: {min_value} to {max_value}")
            # Remap to ensure values start from 0 and are sequential
            le = LabelEncoder()
            df[target_variable] = le.fit_transform(df[target_variable])
            print(f"Target variable remapped to range: 0 to {len(unique_values)-1}")
            print(f"Value mapping: {dict(zip(sorted(unique_values), le.transform(sorted(unique_values))))}")
    
    print(f"Final target variable '{target_variable}' unique values:", sorted(df[target_variable].unique()))

    # Separate features (X) and target (y)
    X_df = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Identify numerical and categorical features
    numerical_features = X_df.select_dtypes(include=['number']).columns
    categorical_features = X_df.select_dtypes(include=['object', 'category']).columns

    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    # Use our preprocessing function
    print("Preprocessing data...")
    X, preprocessor, feature_names = preprocess_data(X_df)
    print("Shape of preprocessed data:", X.shape)

    # Apply quantile transform to get a second version of the data
    print("Creating quantile-transformed version of the data...")
    X_combined, _, _ = preprocess_data(X_df, preprocessor=preprocessor, apply_quantile_transform=True)
    print("Shape of quantile-transformed data:", X_combined.shape)

    best_accuracy = 0
    best_model = None
    best_report = None
    best_auc_roc = 0
    best_feature_set = None

    # Split data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        random_state=42,
                                                        stratify=y)  # Stratified split
    X_combined_train, X_combined_test, y_combined_train, y_combined_test = \
        train_test_split(X_combined, y, test_size=0.2, random_state=42,
                         stratify=y)  # Stratified split

    # Define the classification models to train
    models = {
         'Logistic Regression': LogisticRegression(random_state=42,solver='liblinear'),
        # 'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
        # 'LightGBM': LGBMClassifier(random_state=42),
        # 'HistGradientBoostingClassifier': HistGradientBoostingClassifier(random_state=42),
         'Random Forest': RandomForestClassifier(random_state=42),
        # 'Gaussian Naive Bayes': GaussianNB(),
        # 'Decision Tree': DecisionTreeClassifier(random_state=42)
    }

    # Train and evaluate models on original features
    print("Training models on original features...")
    for name, model in models.items():
        print(f"\nTraining {name}...")
        try:
            # Set LOKY_MAX_CPU_COUNT here, before LightGBM is initialized.  Pick a number appropriate for your system.
            os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4') # Set default, or use what is already set
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            current_accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {current_accuracy:.4f}")
            print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")

            # Calculate AUC-ROC (if applicable)
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc_roc = roc_auc_score(y_test, y_pred_proba)
                print(f"{name} AUC-ROC: {auc_roc:.4f}")
            except AttributeError:
                auc_roc = 0  # Or some other default value
                print(f"{name} does not support probability predictions.")

            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                best_model = model
                best_feature_set = "original"
                best_report = classification_report(y_test, y_pred)
                best_auc_roc = auc_roc
                print(f"Best model updated to {name} with Accuracy: {best_accuracy:.4f}")
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue  # Skip to the next model

    # Train and evaluate models on combined features
    print("\nTraining models on combined features...")
    for name, model in models.items():
        print(f"\nTraining {name} on combined features...")
        try:
            #  LOKY_MAX_CPU_COUNT should already be set, but we'll leave this here for clarity
            os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
            model.fit(X_combined_train, y_combined_train)
            y_pred_combined = model.predict(X_combined_test)
            accuracy_combined = accuracy_score(y_combined_test, y_pred_combined)
            print(f"{name} (Combined) Accuracy: {accuracy_combined:.4f}")
            print(f"{name} (Combined) Classification Report:\n{classification_report(y_combined_test, y_pred_combined)}")

            # Calculate AUC-ROC (if applicable)
            try:
                y_pred_proba_combined = model.predict_proba(X_combined_test)[:, 1]
                auc_roc_combined = roc_auc_score(y_combined_test, y_pred_proba_combined)
                print(f"{name} (Combined) AUC-ROC: {auc_roc_combined:.4f}")
            except AttributeError:
                auc_roc_combined = 0
                print(f"{name} (Combined) does not support probability predictions.")

            if accuracy_combined > best_accuracy:
                best_accuracy = accuracy_combined
                best_model = model
                best_feature_set = "combined"
                best_report = classification_report(y_combined_test, y_pred_combined)
                best_auc_roc = auc_roc_combined
                print(
                    f"Best model updated to {name} (Combined) with Accuracy: {best_accuracy:.4f}")
        except Exception as e:
            print(f"Error training {name} (Combined): {e}")
            continue  # Skip to the next model

    accuracy = best_accuracy
    print(
        f"\nBest model: {type(best_model).__name__}, Feature Set: {best_feature_set}, Best Accuracy: {accuracy:.4f}, Best AUC-ROC: {best_auc_roc:.4f}")
    print(f"\nBest Model Classification Report:\n{best_report}")

    # Get feature importances from the best model
    if hasattr(best_model, "feature_importances_"):
        if best_feature_set == "original":
            best_feature_importances = dict(zip(X.columns, best_model.feature_importances_))
        else:
            best_feature_importances = dict(zip(X_combined.columns, best_model.feature_importances_))

    elif hasattr(best_model, "coef_"):
        if best_feature_set == "original":
            best_feature_importances = dict(zip(X.columns, best_model.coef_[0]))
        else:
            best_feature_importances = dict(zip(X_combined.columns, best_model.coef_[0]))
    else:
        best_feature_importances = {}
        print("Best model does not support feature importance.")

    # Map feature importance names back to original column names
    original_feature_importances = {}
    if best_feature_importances:
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

    # Save the best model
    print(f"Saving best model to: {model_path}")
    joblib.dump(best_model, model_path)
    
    # Save preprocessor
    preprocessor_path = save_preprocessor(preprocessor, user_id, use_case, target_variable, filename, output_dir)
    print(f"Saved preprocessor to: {preprocessor_path}")

    # Save feature names
    feature_names = X.columns if best_feature_set == "original" else X_combined.columns
    features_filename = model_path.replace(model_filename, f"model_features_{model_filename}")
    print(f"Saving feature names to: {features_filename}")
    joblib.dump(feature_names, features_filename)

    return model_path, features_filename, accuracy, original_feature_importances