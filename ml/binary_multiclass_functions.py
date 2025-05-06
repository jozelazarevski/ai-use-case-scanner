import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, 
                           f1_score, precision_score, recall_score, confusion_matrix,
                           precision_recall_curve, average_precision_score)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import label_binarize

# Import custom modules (with multiple import approaches)
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


def train_binary_classification(target_variable, filepath, user_id, use_case, 
                               cv_folds=5, use_grid_search=True, threshold=0.5,
                               calibrate_probabilities=True):
    """
    Trains and optimizes binary classification models on the given dataset.
    
    Args:
        target_variable (str): The name of the target variable column.
        filepath (str): The path to the data file.
        user_id (str): User ID for organization.
        use_case (str): Use case identifier.
        cv_folds (int): Number of cross-validation folds.
        use_grid_search (bool): Whether to use grid search for hyperparameter tuning.
        threshold (float): Classification threshold for binary predictions.
        calibrate_probabilities (bool): Whether to calibrate probability estimates.
        
    Returns:
        tuple: A tuple containing:
            - model_path (str): Path to the saved model.
            - features_filename (str): Path to the saved feature names.
            - metrics (dict): Dictionary of performance metrics.
            - feature_importances (dict): Feature importance scores.
    """
    print(f"Starting binary classification training for target: '{target_variable}'")
    print(f"Reading data from: '{filepath}'")
    
    # Output directories setup
    output_dir = get_output_dir(user_id, use_case, target_variable, filename=os.path.basename(filepath))
    os.makedirs(output_dir, exist_ok=True)
    model_filename = get_model_filename(user_id, use_case, target_variable, filename=os.path.basename(filepath))
    model_path = os.path.join(output_dir, model_filename)
    
    # Read and preprocess data
    df = read_data_flexible(filepath)
    if df is None:
        print("Error: Failed to read the data. Training aborted.")
        return None, None, {}, {}
    
    # Store original column names
    original_columns = list(df.columns)
    
    # Process target variable
    label_encoder = LabelEncoder()
    if df[target_variable].dtype == 'object' or df[target_variable].dtype.name == 'category':
        print(f"Target variable '{target_variable}' is categorical. Converting to numerical.")
        df[target_variable] = label_encoder.fit_transform(df[target_variable])
    elif df[target_variable].dtype in ['int64', 'float64']:
        unique_values = df[target_variable].unique()
        if len(unique_values) > 2:
            print(f"Warning: Target variable has {len(unique_values)} unique values. Ensuring binary encoding.")
            # Map to binary if needed
            if not np.array_equal(np.sort(unique_values), np.array([0, 1])):
                df[target_variable] = label_encoder.fit_transform(df[target_variable])
                print(f"Mapped target values to: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    # Verify binary classification
    unique_target_values = np.unique(df[target_variable])
    if len(unique_target_values) != 2:
        print(f"Warning: Expected 2 classes for binary classification, found {len(unique_target_values)}")
        if len(unique_target_values) > 2:
            print(f"Converting to binary problem. Classes {unique_target_values[1:]} will be treated as positive class.")
            # Convert to binary (0 for first class, 1 for all others)
            binary_target = np.zeros(len(df[target_variable]))
            binary_target[df[target_variable] > 0] = 1
            df[target_variable] = binary_target
            print(f"Converted to binary classification with distribution: {np.bincount(df[target_variable].astype(int))}")
    
    # Check class imbalance
    class_counts = np.bincount(df[target_variable].astype(int))
    class_ratio = class_counts.min() / class_counts.max()
    is_imbalanced = class_ratio < 0.25  # Arbitrary threshold
    if is_imbalanced:
        print(f"Warning: Detected class imbalance. Class ratio: {class_ratio:.3f}")
        print(f"Class distribution: {class_counts}")
        print("Will use balanced class weights and F1-score for evaluation.")
    
    # Split features and target
    X_df = df.drop(columns=[target_variable])
    y = df[target_variable].astype(int)
    
    # Preprocess features
    print("Preprocessing features...")
    X, preprocessor, feature_names = preprocess_data(X_df)
    
    # Create a version with quantile transform
    X_quantile, _, _ = preprocess_data(X_df, preprocessor=preprocessor, apply_quantile_transform=True)
    
    # Split data with stratification
    X_train, X_test, X_quantile_train, X_quantile_test, y_train, y_test = train_test_split(
        X, X_quantile, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models optimized for binary classification
    class_weight = 'balanced' if is_imbalanced else None
    
    binary_models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'params': {
                'C': [0.01, 0.1, 1.0, 10.0],
                'class_weight': [class_weight, None],
                'solver': ['liblinear', 'saga']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'class_weight': [class_weight, None]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 6],
                'scale_pos_weight': [1, class_counts[0]/class_counts[1] if class_counts[1] > 0 else 1]
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 127],
                'class_weight': [class_weight, None]
            }
        }
    }
    
    # Add SVM for smaller datasets
    if len(y) < 10000:  # Only add SVM for smaller datasets
        binary_models['SVM'] = {
            'model': SVC(probability=True, random_state=42),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['rbf', 'linear'],
                'class_weight': [class_weight, None]
            }
        }
    
    best_score = 0
    best_model = None
    best_model_name = None
    best_feature_set = None
    best_metrics = {}
    feature_importances = {}
    
    # Function to evaluate a model
    def evaluate_binary_model(model, X_train, X_test, y_train, y_test, model_name, feature_set='original'):
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # For calibrated probabilities
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            # Threshold tuning if needed
            if threshold != 0.5:
                y_pred = (y_prob >= threshold).astype(int)
        else:
            y_prob = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # ROC-AUC if probabilities available
        auc_roc = roc_auc_score(y_test, y_prob) if y_prob is not None else 0
        
        # For imbalanced datasets, F1 is more important
        primary_score = f1 if is_imbalanced else accuracy
        
        # Print results
        print(f"\n{model_name} ({feature_set}) Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        if y_prob is not None:
            print(f"ROC AUC: {auc_roc:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        # Detailed classification report
        print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm.tolist(),
            'auc_roc': auc_roc if y_prob is not None else None
        }
        
        return model, primary_score, metrics
    
    # Train and evaluate models
    for feature_set, (X_trn, X_tst) in [
        ('original', (X_train, X_test)),
        ('quantile', (X_quantile_train, X_quantile_test))
    ]:
        print(f"\n{'='*50}")
        print(f"Training on {feature_set.upper()} feature set")
        print(f"{'='*50}")
        
        for name, model_config in binary_models.items():
            print(f"\nTraining {name}...")
            base_model = model_config['model']
            
            try:
                if use_grid_search:
                    # GridSearch with cross-validation
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    grid = GridSearchCV(
                        estimator=base_model,
                        param_grid=model_config['params'],
                        cv=cv,
                        scoring='f1' if is_imbalanced else 'accuracy',
                        n_jobs=-1 if 'LightGBM' not in name else 1  # Avoid parallel issues with LightGBM
                    )
                    grid.fit(X_trn, y_trn)
                    model = grid.best_estimator_
                    print(f"Best parameters: {grid.best_params_}")
                else:
                    model = base_model
                
                # Calibrate probabilities if needed
                if calibrate_probabilities and hasattr(model, "predict_proba"):
                    print("Calibrating probability estimates...")
                    model = CalibratedClassifierCV(
                        model, method='sigmoid', cv=3
                    )
                
                # Evaluate model
                trained_model, score, metrics = evaluate_binary_model(
                    model, X_trn, X_tst, y_trn, y_tst, name, feature_set
                )
                
                # Update best model if needed
                if score > best_score:
                    best_score = score
                    best_model = trained_model
                    best_model_name = name
                    best_feature_set = feature_set
                    best_metrics = metrics
                    
                    # Get feature importances if available
                    if hasattr(trained_model, 'feature_importances_'):
                        importances = trained_model.feature_importances_
                    elif hasattr(trained_model, 'coef_'):
                        importances = np.abs(trained_model.coef_[0])
                    else:
                        importances = None
                    
                    if importances is not None:
                        features = feature_names
                        feature_importances = dict(zip(features, importances))
            
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
    
    # Final evaluation of best model
    print(f"\n{'='*50}")
    print(f"Best model: {best_model_name} with {best_feature_set} features")
    print(f"Best score: {best_score:.4f}")
    print(f"{'='*50}")
    
    # Save the best model
    print(f"Saving best model to: {model_path}")
    joblib.dump(best_model, model_path)
    
    # Save preprocessor
    preprocessor_path = save_preprocessor(preprocessor, user_id, use_case, target_variable, 
                                        os.path.basename(filepath), output_dir)
    print(f"Saved preprocessor to: {preprocessor_path}")
    
    # Save feature names
    features_filename = os.path.join(output_dir, f"features_{model_filename}")
    print(f"Saving feature names to: {features_filename}")
    joblib.dump(feature_names, features_filename)
    
    # Save label encoder if used
    if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
        encoder_path = os.path.join(output_dir, f"label_encoder_{model_filename}")
        joblib.dump(label_encoder, encoder_path)
        print(f"Saved label encoder to: {encoder_path}")
    
    # Map feature importances back to original column names if possible
    original_feature_importances = {}
    if feature_importances:
        for processed_feature, importance in feature_importances.items():
            # Remove prefixes like 'num__' or 'cat__'
            processed_feature = processed_feature.replace("num__", "").replace("cat__", "")
            
            # Find the original column that contributed to this processed feature
            mapped_original_names = []
            for original_col in original_columns:
                if processed_feature.startswith(original_col):
                    mapped_original_names.append(original_col)
            
            if mapped_original_names:
                original_feature_importances[", ".join(mapped_original_names)] = importance / len(mapped_original_names)
            else:
                original_feature_importances[processed_feature] = importance
    
    # Save feature importances visualization if available
    if original_feature_importances:
        # Sort features by importance
        sorted_features = dict(sorted(original_feature_importances.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        # Save top N features to file
        top_n = min(20, len(sorted_features))
        top_features = {k: sorted_features[k] for k in list(sorted_features.keys())[:top_n]}
        
        importances_path = os.path.join(output_dir, f"feature_importance_{os.path.basename(filepath)}.csv")
        pd.DataFrame({
            'Feature': list(sorted_features.keys()),
            'Importance': list(sorted_features.values())
        }).to_csv(importances_path, index=False)
        print(f"Saved feature importances to: {importances_path}")
    
    # Return paths and metrics
    return model_path, features_filename, best_metrics, original_feature_importances


def train_multiclass_classification(target_variable, filepath, user_id, use_case,
                                  cv_folds=5, use_grid_search=True,
                                  multiclass_strategy='ovr', balance_classes=True):
    """
    Trains and optimizes multi-class classification models on the given dataset.
    
    Args:
        target_variable (str): The name of the target variable column.
        filepath (str): The path to the data file.
        user_id (str): User ID for organization.
        use_case (str): Use case identifier.
        cv_folds (int): Number of cross-validation folds.
        use_grid_search (bool): Whether to use grid search for hyperparameter tuning.
        multiclass_strategy (str): Strategy for multi-class classification ('ovr' or 'ovo').
        balance_classes (bool): Whether to apply class balancing techniques.
        
    Returns:
        tuple: A tuple containing:
            - model_path (str): Path to the saved model.
            - features_filename (str): Path to the saved feature names.
            - metrics (dict): Dictionary of performance metrics.
            - feature_importances (dict): Feature importance scores.
    """
    print(f"Starting multi-class classification training for target: '{target_variable}'")
    print(f"Reading data from: '{filepath}'")
    
    # Output directories setup
    output_dir = get_output_dir(user_id, use_case, target_variable, filename=os.path.basename(filepath))
    os.makedirs(output_dir, exist_ok=True)
    model_filename = get_model_filename(user_id, use_case, target_variable, filename=os.path.basename(filepath))
    model_path = os.path.join(output_dir, model_filename)
    
    # Read and preprocess data
    df = read_data_flexible(filepath)
    if df is None:
        print("Error: Failed to read the data. Training aborted.")
        return None, None, {}, {}
    
    # Store original column names
    original_columns = list(df.columns)
    
    # Process target variable
    label_encoder = LabelEncoder()
    if df[target_variable].dtype == 'object' or df[target_variable].dtype.name == 'category':
        print(f"Target variable '{target_variable}' is categorical. Converting to numerical.")
        df[target_variable] = label_encoder.fit_transform(df[target_variable])
        print(f"Target variable '{target_variable}' encoded categories:", label_encoder.classes_)
    elif df[target_variable].dtype in ['int64', 'float64']:
        unique_values = df[target_variable].unique()
        min_value = unique_values.min()
        max_value = unique_values.max()
        # Ensure sequential integers starting from 0
        if min_value != 0 or not np.array_equal(np.sort(unique_values), np.arange(min_value, min_value + len(unique_values))):
            print(f"Target variable has non-standard values. Current range: {min_value} to {max_value}")
            df[target_variable] = label_encoder.fit_transform(df[target_variable])
            print(f"Target variable remapped to range: 0 to {len(unique_values)-1}")
    
    # Verify multi-class classification
    unique_classes = np.unique(df[target_variable])
    num_classes = len(unique_classes)
    if num_classes < 3:
        print(f"Warning: Only {num_classes} classes detected. Consider using binary classification instead.")
    
    print(f"Number of classes: {num_classes}")
    print(f"Class values: {unique_classes}")
    
    # Check class imbalance
    class_counts = np.bincount(df[target_variable].astype(int))
    class_distribution = class_counts / class_counts.sum()
    min_class_ratio = class_counts.min() / class_counts.max()
    is_imbalanced = min_class_ratio < 0.3  # Arbitrary threshold for multiclass
    
    print(f"Class distribution: {class_distribution}")
    if is_imbalanced:
        print(f"Warning: Detected class imbalance. Min/Max class ratio: {min_class_ratio:.3f}")
        print("Will use balanced class weights and weighted F1-score for evaluation.")
    
    # Split features and target
    X_df = df.drop(columns=[target_variable])
    y = df[target_variable].astype(int)
    
    # Preprocess features
    print("Preprocessing features...")
    X, preprocessor, feature_names = preprocess_data(X_df)
    
    # Create a version with quantile transform
    X_quantile, _, _ = preprocess_data(X_df, preprocessor=preprocessor, apply_quantile_transform=True)
    
    # Split data with stratification
    X_train, X_test, X_quantile_train, X_quantile_test, y_train, y_test = train_test_split(
        X, X_quantile, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # One-hot encode for multiclass metrics
    y_test_bin = label_binarize(y_test, classes=np.unique(y))
    
    # Define class weights if needed
    if balance_classes and is_imbalanced:
        # Compute class weights
        n_samples = len(y)
        class_weight = {i: n_samples / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        print(f"Using class weights: {class_weight}")
    else:
        class_weight = None
    
    # Define models optimized for multi-class classification
    multiclass_models = {
        'LogisticRegression': {
            'model': LogisticRegression(random_state=42, max_iter=1000, multi_class='auto', solver='lbfgs'),
            'params': {
                'C': [0.1, 1.0, 10.0],
                'class_weight': [class_weight, 'balanced', None],
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'max_depth': [None, 15, 30],
                'min_samples_split': [2, 5],
                'class_weight': [class_weight, 'balanced', None]
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 5]
            }
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss'),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'max_depth': [3, 6]
            }
        }
    }
    
    # For larger number of classes, consider one-vs-rest or one-vs-one strategies
    if num_classes > 3 and multiclass_strategy == 'ovr':
        multiclass_models['LogisticRegression_OVR'] = {
            'model': OneVsRestClassifier(LogisticRegression(random_state=42, max_iter=1000)),
            'params': {
                'estimator__C': [0.1, 1.0, 10.0],
                'estimator__class_weight': [class_weight, 'balanced', None]
            }
        }
    elif num_classes > 3 and multiclass_strategy == 'ovo':
        multiclass_models['LogisticRegression_OVO'] = {
            'model': OneVsOneClassifier(LogisticRegression(random_state=42, max_iter=1000)),
            'params': {
                'estimator__C': [0.1, 1.0],
                'estimator__class_weight': [class_weight, 'balanced', None]
            }
        }
    
    # Add LightGBM for faster training with many classes
    if num_classes <= 10:  # For reasonable number of classes
        multiclass_models['LightGBM'] = {
            'model': LGBMClassifier(random_state=42, objective='multiclass'),
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1],
                'num_leaves': [31, 127],
                'class_weight': [class_weight, 'balanced', None]
            }
        }
    
    best_score = 0
    best_model = None
    best_model_name = None
    best_feature_set = None
    best_metrics = {}
    feature_importances = {}
    
    # Function to evaluate a multiclass model
    def evaluate_multiclass_model(model, X_train, X_test, y_train, y_test, model_name, feature_set='original'):
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Probability predictions if available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        else:
            y_prob = None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # For multiclass: macro averages
        precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        # For multiclass: weighted averages (better for imbalanced)
        precision_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # For multiclass ROC-AUC, we need one-vs-rest approach
        if y_prob is not None and num_classes > 2:
            try:
                # For multi-class, we use the OvR approach
                roc_auc_ovr = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
            except (ValueError, IndexError):
                roc_auc_ovr = None
        else:
            roc_auc_ovr = None
        
        # Primary score for model selection (weighted metrics better for imbalanced)
        primary_score = f1_weighted if is_imbalanced else accuracy
        
        # Print results
        print(f"\n{model_name} ({feature_set}) Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
        print(f"Weighted Precision: {precision_weighted:.4f}, Recall: {recall_weighted:.4f}, F1: {f1_weighted:.4f}")
        if roc_auc_ovr is not None:
            print(f"ROC AUC (OvR): {roc_auc_ovr:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        # Detailed classification report
        print(f"Classification Report:\n{classification_report(y_test, y_pred, zero_division=0)}")
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted, 
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm.tolist(),
            'roc_auc_ovr': roc_auc_ovr
        }
        
        return model, primary_score, metrics
    
    # Train and evaluate models
    for feature_set, (X_trn, X_tst) in [
        ('original', (X_train, X_test)),
        ('quantile', (X_quantile_train, X_quantile_test))
    ]:
        print(f"\n{'='*50}")
        print(f"Training on {feature_set.upper()} feature set")
        print(f"{'='*50}")
        
        for name, model_config in multiclass_models.items():
            print(f"\nTraining {name}...")
            base_model = model_config['model']
            
            try:
                if use_grid_search:
                    # GridSearch with cross-validation
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    scoring = 'f1_weighted' if is_imbalanced else 'accuracy'
                    
                    # Limit the parameter grid size for large datasets or many classes
                    if len(y) > 50000 or num_classes > 10:
                        # Simplified parameter grid for large datasets
                        params = {k: v[:1] for k, v in model_config['params'].items()}
                        print("Large dataset detected: Using simplified parameter grid")
                    else:
                        params = model_config['params']
                    
                    grid = GridSearchCV(
                        estimator=base_model,
                        param_grid=params,
                        cv=cv,
                        scoring=scoring,
                        n_jobs=-1 if 'LightGBM' not in name and 'XGBoost' not in name else 1
                    )
                    grid.fit(X_trn, y_trn)
                    model = grid.best_estimator_
                    print(f"Best parameters: {grid.best_params_}")
                else:
                    model = base_model
                    model.fit(X_trn, y_trn)
                
                # Evaluate model
                trained_model, score, metrics = evaluate_multiclass_model(
                    model, X_trn, X_tst, y_trn, y_tst, name, feature_set
                )
                
                # Update best model if needed
                if score > best_score:
                    best_score = score
                    best_model = trained_model
                    best_model_name = name
                    best_feature_set = feature_set
                    best_metrics = metrics
                    
                    # Get feature importances if available
                    if hasattr(trained_model, 'feature_importances_'):
                        importances = trained_model.feature_importances_
                    elif hasattr(trained_model, 'coef_'):
                        # For multiclass LogisticRegression, coef_ is 2D array with shape (n_classes, n_features)
                        # Take the average across all classes
                        importances = np.mean(np.abs(trained_model.coef_), axis=0)
                    elif hasattr(trained_model, 'estimators_') and hasattr(trained_model.estimators_[0], 'feature_importances_'):
                        # For ensemble of estimators like in OVR
                        importances = np.mean([est.feature_importances_ for est in trained_model.estimators_], axis=0)
                    else:
                        importances = None
                    
                    if importances is not None:
                        features = feature_names
                        feature_importances = dict(zip(features, importances))
                        
            except Exception as e:
                print(f"Error training {name}: {str(e)}")
                continue
    
    # Final evaluation of best model
    print(f"\n{'='*50}")
    print(f"Best model: {best_model_name} with {best_feature_set} features")
    print(f"Best score: {best_score:.4f}")
    print(f"{'='*50}")
    
    # Save the best model
    print(f"Saving best model to: {model_path}")
    joblib.dump(best_model, model_path)
    
    # Save preprocessor
    preprocessor_path = save_preprocessor(preprocessor, user_id, use_case, target_variable, 
                                         os.path.basename(filepath), output_dir)
    print(f"Saved preprocessor to: {preprocessor_path}")
    
    # Save feature names
    features_filename = os.path.join(output_dir, f"features_{model_filename}")
    print(f"Saving feature names to: {features_filename}")
    joblib.dump(feature_names, features_filename)
    
    # Save label encoder for class mapping
    if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
        encoder_path = os.path.join(output_dir, f"label_encoder_{model_filename}")
        joblib.dump(label_encoder, encoder_path)
        print(f"Saved label encoder to: {encoder_path}")
        
        # Save class mapping for reference
        class_mapping = {i: class_name for i, class_name in enumerate(label_encoder.classes_)}
        mapping_path = os.path.join(output_dir, f"class_mapping_{os.path.basename(filepath)}.csv")
        pd.DataFrame({
            'Class_ID': list(class_mapping.keys()),
            'Class_Name': list(class_mapping.values())
        }).to_csv(mapping_path, index=False)
        print(f"Saved class mapping to: {mapping_path}")
    
    # Map feature importances back to original column names if possible
    original_feature_importances = {}
    if feature_importances:
        for processed_feature, importance in feature_importances.items():
            # Remove prefixes like 'num__' or 'cat__'
            processed_feature = processed_feature.replace("num__", "").replace("cat__", "")
            
            # Find the original column that contributed to this processed feature
            mapped_original_names = []
            for original_col in original_columns:
                if processed_feature.startswith(original_col):
                    mapped_original_names.append(original_col)
            
            if mapped_original_names:
                original_feature_importances[", ".join(mapped_original_names)] = importance / len(mapped_original_names)
            else:
                original_feature_importances[processed_feature] = importance
    
    # Save feature importances visualization if available
    if original_feature_importances:
        # Sort features by importance
        sorted_features = dict(sorted(original_feature_importances.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        # Save top N features to file
        top_n = min(20, len(sorted_features))
        top_features = {k: sorted_features[k] for k in list(sorted_features.keys())[:top_n]}
        
        importances_path = os.path.join(output_dir, f"feature_importance_{os.path.basename(filepath)}.csv")
        pd.DataFrame({
            'Feature': list(sorted_features.keys()),
            'Importance': list(sorted_features.values())
        }).to_csv(importances_path, index=False)
        print(f"Saved feature importances to: {importances_path}")
        
        # Generate per-class importances for models that support it (like LogisticRegression)
        if hasattr(best_model, 'coef_') and len(best_model.coef_.shape) > 1:
            class_importances = {}
            for class_idx in range(best_model.coef_.shape[0]):
                class_name = label_encoder.classes_[class_idx] if hasattr(label_encoder, 'classes_') else f"Class_{class_idx}"
                class_importances[class_name] = dict(zip(feature_names, np.abs(best_model.coef_[class_idx])))
            
            # Save per-class importances
            for class_name, imps in class_importances.items():
                safe_class_name = str(class_name).replace(" ", "_").replace("/", "_")
                class_imp_path = os.path.join(output_dir, f"feature_importance_class_{safe_class_name}_{os.path.basename(filepath)}.csv")
                sorted_imps = dict(sorted(imps.items(), key=lambda x: abs(x[1]), reverse=True))
                pd.DataFrame({
                    'Feature': list(sorted_imps.keys()),
                    'Importance': list(sorted_imps.values())
                }).to_csv(class_imp_path, index=False)
                print(f"Saved feature importances for class '{class_name}' to: {class_imp_path}")
    
    # Return paths and metrics
    return model_path, features_filename, best_metrics, original_feature_importances


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
from sklearn.preprocessing import LabelEncoder

# Import the specialized classification functions
from binary_multiclass_functions import train_binary_classification, train_multiclass_classification

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
 
def train_classification_model(target_variable, filepath, user_id, use_case, auto_detect=True, force_binary=False, force_multiclass=False, **kwargs):
    """
    Smart classification model trainer that automatically detects whether to use
    binary or multi-class classification based on the target variable.
    
    Args:
        target_variable (str): The name of the target variable column.
        filepath (str): The path to the data file.
        user_id (str): User ID for organization.
        use_case (str): Use case identifier.
        auto_detect (bool): Whether to automatically detect binary vs multi-class.
        force_binary (bool): Force binary classification even for multi-class problems.
        force_multiclass (bool): Force multi-class classification even for binary problems.
        **kwargs: Additional arguments to pass to the specific classification functions.
    
    Returns:
        tuple: A tuple containing the model path, features filename, metrics, and feature importances.
    """
    print(f"Starting automated classification training for target: '{target_variable}'")
    
    # Read the data using the flexible reader function
    df = read_data_flexible(filepath)
    
    if df is None:
        print("Error: Failed to read the data. Training aborted.")
        return None, None, {}, {}
    
    # Process target variable to determine classification type
    label_encoder = LabelEncoder()
    
    if df[target_variable].dtype == 'object' or df[target_variable].dtype.name == 'category':
        print(f"Target variable '{target_variable}' is categorical. Converting to numerical.")
        df[target_variable] = label_encoder.fit_transform(df[target_variable])
        print(f"Target variable '{target_variable}' encoded categories:", label_encoder.classes_)
    
    # Count unique values in target
    unique_values = np.unique(df[target_variable])
    num_classes = len(unique_values)
    
    print(f"Target variable has {num_classes} unique classes: {unique_values}")
    
    # Determine whether to use binary or multi-class classification
    use_binary = ((num_classes == 2) and not force_multiclass) or force_binary
    
    if not auto_detect:
        if force_binary:
            use_binary = True
            if num_classes > 2:
                print("Warning: Forcing binary classification on a multi-class problem.")
                print("Classes other than 0 will be combined into a single positive class.")
        elif force_multiclass:
            use_binary = False
            if num_classes == 2:
                print("Warning: Forcing multi-class classification on a binary problem.")
    
    # Apply the appropriate classification function
    if use_binary:
        print("Using binary classification approach.")
        return train_binary_classification(target_variable, filepath, user_id, use_case, **kwargs)
    else:
        print("Using multi-class classification approach.")
        return train_multiclass_classification(target_variable, filepath, user_id, use_case, **kwargs)


# Example usage:
if __name__ == "__main__":
    # This will auto-detect whether to use binary or multi-class classification
    model_path, features_path, metrics, importances = train_classification_model(
        target_variable="pets_allowed",
        filepath="../uploads/apartments_for_rent_classified_10K.csv",
        user_id="1",
        use_case="rental_prediction",
        cv_folds=3,  # Fewer folds for faster execution
        use_grid_search=True
    )
    
    print("\nTraining complete!")
    print(f"Model saved to: {model_path}")
    print(f"Features saved to: {features_path}")
    print(f"Model performance metrics: {metrics}")
    
    # Print top 5 most important features
    if importances:
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True)[:5])
        print("\nTop 5 most important features:")
        for feature, importance in sorted_importances.items():
            print(f"{feature}: {importance:.4f}")