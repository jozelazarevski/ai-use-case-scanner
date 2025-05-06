# -*- coding: utf-8 -*-
"""
target_variable="pets_allowed"
filepath="../uploads/apartments_for_rent_classified_10K.csv"
user_id='1'
use_case=target_variable
model_path, features_path, accuracy, importances = train_classification_model( target_variable,  filepath,   user_id, use_case,
                                                                              threshold=0.60,
                                                                              use_f1_for_threshold=False )



target_variable="price_type"
filepath="../uploads/apartments_for_rent_classified_10K.csv"
user_id='1'
use_case=target_variable
model_path, features_path, accuracy, importances = train_classification_model( target_variable,  filepath,   user_id, use_case,
                                                                              threshold=0.60,
                                                                              use_f1_for_threshold=False )

 

target_variable="y"
filepath="../uploads/bank-full.csv"
user_id='1'
use_case=target_variable
model_path, features_path, accuracy, importances = train_classification_model( target_variable,  filepath,   user_id, use_case,
                                                                              threshold=0.60,
                                                                              use_f1_for_threshold=False )



target_variable="target"
filepath="../uploads/salesforce_lead_syntetic_data.csv"
user_id='1'
use_case=target_variable
model_path, features_path, accuracy, importances = train_classification_model( target_variable,  filepath,   user_id, use_case,
                                                                              threshold=0.90,
                                                                              use_f1_for_threshold=False )


 target_variable="median_house_value_tier"
 filepath="../uploads/california_housing_train_expanded.csv"
 user_id='1'
 use_case=target_variable
 model_path, features_path, accuracy, importances = train_classification_model( target_variable,  filepath,   user_id, use_case,
                                                                               threshold=0.90,
                                                                               use_f1_for_threshold=False )
 
 
 target_variable="StockCode"
 filepath="../uploads/Online_Retail.xlsx"
 user_id='1'
 use_case=target_variable
 model_path, features_path, accuracy, importances = train_classification_model( target_variable,  filepath,   user_id, use_case,
                                                                               threshold=0.90,
                                                                               use_f1_for_threshold=False )
 
 
 target_variable="ITEM TYPE"
 filepath="../uploads/Warehouse_and_Retail_Sales.csv"
 user_id='1'
 use_case=target_variable
 model_path, features_path, accuracy, importances = train_classification_model( target_variable,  filepath,   user_id, use_case,
                                                                               threshold=0.90,
                                                                               use_f1_for_threshold=False )
 
 target_variable="StockCode"
 filepath="../uploads/Online_Retail.xlsx"
 user_id='1'
 use_case=target_variable
 model_path, features_path, accuracy, importances = train_classification_model(
                                                                target_variable,
                                                                filepath,
                                                                user_id,
                                                                use_case,
                                                                threshold=0.85,                # Target accuracy/F1 score
                                                                use_sampling=True,             # Enable sampling
                                                                sample_size=1000,             # Use 50,000 samples
                                                                stratified_sampling=True,      # Maintain class distribution
                                                                quick_mode=True,               # Use simpler models
                                                                leakage_threshold=0.8          # Correlation threshold for leakage detection
                                                            )
 

 

"""

import pandas as pd
import numpy as np
import joblib
import os
import gc
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier, AdaBoostClassifier, BaggingClassifier
)
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from utils.feature_transformer import FeatureTransformer
from ml.read_file import read_data_flexible
from ml.preprocess_data import (
        preprocess_data, create_preprocessor,
        save_preprocessor, save_label_encoder, get_output_dir, get_model_filename,screen_features,add_screen_features_to_pipeline
    )
    
 

def configure_simple_models(is_binary: bool, num_classes: int, class_counts: np.array, is_imbalanced: bool, n_features: int, n_samples: int) -> dict:
    """
    Creates a simplified model configuration with Logistic Regression and Random Forest.

    Args:
        is_binary (bool): Whether this is a binary classification problem
        num_classes (int): Number of distinct classes in the target variable
        class_counts (np.array): Array of sample counts for each class
        is_imbalanced (bool): Whether class distribution is imbalanced
        n_features (int): Number of features in the dataset
        n_samples (int): Number of samples in the dataset

    Returns:
        dict: Dictionary of configured models (Logistic Regression and Random Forest)
    """
    base_models = {}
    common_params = {'random_state': 42}
    class_weight = None
    if is_imbalanced:
        n_samples_total = np.sum(class_counts)
        class_weight = {i: n_samples_total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        print(f"Using class weights: {class_weight}")

    complexity = "low"
    if n_samples < 10000:
        complexity = "medium"
    elif n_samples < 100000:
        complexity = "high"
    else:
        complexity = "very_high"

    # Logistic Regression
    lr_params = {
        **common_params,
        'solver': 'liblinear' if is_binary else 'lbfgs',
        'C': 0.1 if complexity in ["high", "very_high"] else 1.0,
        'penalty': 'l2' if is_binary else None,
        'max_iter': 1000,
        'class_weight': class_weight,
        'n_jobs': -1 if not is_binary else None,
        'multi_class': 'auto' if not is_binary else None
    }
    base_models['LogisticRegression'] = LogisticRegression(**{k: v for k, v in lr_params.items() if v is not None})

    # Random Forest
    rf_params = {
        **common_params,
        'n_estimators': 100 if complexity == "low" else 200,
        'max_depth': 10 if complexity == "low" else None,
        'min_samples_split': 5 if complexity == "low" else 2,
        'min_samples_leaf': 2 if complexity == "low" else 1,
        'max_features': 'sqrt',
        'bootstrap': True,
        'n_jobs': -1,
        'class_weight': class_weight
    }
    base_models['RandomForest'] = RandomForestClassifier(**rf_params)

    return base_models



def configure_advanced_models(is_binary, num_classes, class_counts, is_imbalanced, n_features, n_samples):
    """
    Creates advanced model configurations optimized for either binary or multi-class classification,
    prioritizing models with feature importance and computational efficiency.
    
    Args:
        is_binary (bool): Whether this is a binary classification problem
        num_classes (int): Number of distinct classes in the target variable
        class_counts (np.array): Array of sample counts for each class
        is_imbalanced (bool): Whether class distribution is imbalanced
        n_features (int): Number of features in the dataset
        n_samples (int): Number of samples in the dataset
        
    Returns:
        dict: Dictionary of configured models
    """
    # Determine class weights for imbalanced datasets
    if is_imbalanced:
        # Calculate balanced class weights (used where applicable)
        n_samples_total = np.sum(class_counts)
        class_weight_dict = {i: n_samples_total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        print(f"Using class weights: {class_weight_dict}")
        class_weight = class_weight_dict
    else:
        class_weight = None
    
    # Feature selection settings
    use_feature_selection = n_features > 20
    feature_selection_threshold = 'median' if n_features > 100 else '1.25*mean'
    
    # Model complexity settings
    if n_samples < 1000:
        complexity = "low"
    elif n_samples < 10000:
        complexity = "medium"
    elif n_samples < 100000:
        complexity = "high"
    else:
        complexity = "very_high"
    
    base_models = {}
    
    # Common parameters
    common_params = {
        'random_state': 42,
    }
    
    if is_binary:
        # Logistic Regression
        base_models['LogisticRegression'] = LogisticRegression(
            **common_params,
            solver='liblinear',
            C=0.1 if complexity in ["high", "very_high"] else 1.0,
            penalty='l2',
            max_iter=1000,
            class_weight=class_weight
        )
        
        # RandomForest
        base_models['RandomForest'] = RandomForestClassifier(
            **common_params,
            n_estimators=100 if complexity == "low" else 200,
            max_depth=10 if complexity == "low" else None,
            min_samples_split=5 if complexity == "low" else 2,
            min_samples_leaf=2 if complexity == "low" else 1,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
            class_weight=class_weight
        )
        
        # HistGradientBoosting (Fast Gradient Boosting)
        base_models['HistGradientBoosting'] = HistGradientBoostingClassifier(
            **common_params,
            max_iter=150 if complexity != "low" else 50,
            learning_rate=0.1,
            max_depth=5 if complexity != "low" else None,
            l2_regularization=0.1 if complexity in ["high", "very_high"] else 0.0
        )
        
        # XGBoost
        base_models['XGBoost'] = XGBClassifier(
            **common_params,
            eval_metric='logloss',
            n_estimators=100 if complexity == "low" else 150,
            learning_rate=0.1 if complexity == "low" else 0.05,
            max_depth=4 if complexity == "low" else 6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='auto',
            n_jobs=-1,
            scale_pos_weight=class_counts[0]/class_counts[1] if is_imbalanced and class_counts[1] > 0 else 1
        )
        
        # LightGBM
        if n_samples < 100000:
            base_models['LightGBM'] = LGBMClassifier(
                **common_params,
                n_estimators=100 if complexity == "low" else 150,
                learning_rate=0.1 if complexity == "low" else 0.05,
                max_depth=4 if complexity == "low" else 6,
                num_leaves=31 if complexity == "low" else 63,
                subsample=0.8,
                colsample_bytree=0.8,
                n_jobs=-1,
                class_weight=class_weight if class_weight else 'balanced'
            )
        
        # Ensemble Models
        if complexity in ["medium", "high", "very_high"]:
            estimators = [
                ('lr', base_models['LogisticRegression']),
                ('rf', base_models['RandomForest']),
                ('xgb', base_models['XGBoost'])
            ]
            if 'LightGBM' in base_models:
                estimators.append(('lgbm', base_models['LightGBM']))
            
            if len(estimators) >= 2:
                base_models['VotingClassifier'] = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    n_jobs=-1
                )
            
            base_models['BaggingClassifier'] = BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42),
                n_estimators=80,
                max_samples=0.8,
                max_features=0.8,
                bootstrap=True,
                bootstrap_features=False,
                random_state=42,
                n_jobs=-1
            )
            
            base_models['AdaBoostClassifier'] = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
                n_estimators=80,
                learning_rate=0.1,
                algorithm='SAMME.R',
                random_state=42
            )
            
        if use_feature_selection:
            # RandomForest-based feature selection
            rf_selector = RandomForestClassifier(
                n_estimators=50, random_state=42, class_weight=class_weight
            )
            base_models['RF_FeatureSelection'] = Pipeline([
                ('feature_selection', SelectFromModel(
                    rf_selector, threshold=feature_selection_threshold
                )),
                ('classifier', RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight=class_weight
                ))
            ])
            
            # RFE (Recursive Feature Elimination) with LogisticRegression
            if n_features < 100:
                base_models['RFE_LogisticRegression'] = Pipeline([
                    ('feature_selection', RFE(
                        LogisticRegression(random_state=42),
                        n_features_to_select=min(20, n_features // 2),
                        step=1
                    )),
                    ('classifier', LogisticRegression(
                        random_state=42, class_weight=class_weight, max_iter=1000
                    ))
                ])
    
    else:  # Multi-class
        # Logistic Regression
        base_models['LogisticRegression'] = LogisticRegression(
            **common_params,
            multi_class='auto',
            solver='lbfgs',
            C=0.1 if complexity in ["high", "very_high"] else 1.0,
            max_iter=1000,
            n_jobs=-1,
            class_weight=class_weight
        )
        
        # RandomForest
        base_models['RandomForest'] = RandomForestClassifier(
            **common_params,
            n_estimators=100 if complexity == "low" else 200,
            max_depth=10 if complexity == "low" else None,
            min_samples_split=5 if complexity == "low" else 2,
            min_samples_leaf=2 if complexity == "low" else 1,
            max_features='sqrt',
            bootstrap=True,
            n_jobs=-1,
            class_weight=class_weight
        )
        
        # XGBoost
        base_models['XGBoost'] = XGBClassifier(
            **common_params,
            eval_metric='mlogloss',
            n_estimators=100 if complexity == "low" else 150,
            learning_rate=0.1 if complexity == "low" else 0.05,
            max_depth=4 if complexity == "low" else 6,
            subsample=0.8,
            colsample_bytree=0.8,
            objective='multi:softprob',
            num_class=num_classes,
            tree_method='auto',
            n_jobs=-1
        )
        
        # LightGBM
        if n_samples < 100000 and num_classes <= 10:
            base_models['LightGBM'] = LGBMClassifier(
                **common_params,
                n_estimators=100 if complexity == "low" else 150,
                learning_rate=0.1 if complexity == "low" else 0.05,
                max_depth=4 if complexity == "low" else 6,
                num_leaves=31 if complexity == "low" else 63,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multiclass',
                num_class=num_classes,
                n_jobs=-1,
                class_weight=class_weight
            )
            
        # HistGradientBoosting
        if n_samples >= 10000:
            base_models['HistGradientBoosting'] = HistGradientBoostingClassifier(
                **common_params,
                max_iter=150 if complexity != "low" else 50,
                learning_rate=0.1,
                max_depth=5 if complexity != "low" else None,
                l2_regularization=0.1 if complexity in ["high", "very_high"] else 0.0
            )
            
        # Ensemble Models
        if complexity in ["medium", "high", "very_high"] and num_classes <= 10:
            estimators = [
                ('lr', base_models['LogisticRegression']),
                ('rf', base_models['RandomForest']),
                ('xgb', base_models['XGBoost'])
            ]
            if 'LightGBM' in base_models:
                estimators.append(('lgbm', base_models['LightGBM']))
            
            if len(estimators) >= 2:
                base_models['VotingClassifier'] = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    n_jobs=-1
                )
            
            base_models['BaggingClassifier'] = BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42),
                n_estimators=80,
                max_samples=0.8,
                max_features=0.8,
                bootstrap=True,
                bootstrap_features=False,
                random_state=42,
                n_jobs=-1
            )
            
            base_models['AdaBoostClassifier'] = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
                n_estimators=80,
                learning_rate=0.1,
                algorithm='SAMME.R',
                random_state=42
            )
        
        if use_feature_selection:
            # RandomForest-based feature selection
            rf_selector = RandomForestClassifier(
                n_estimators=50, random_state=42, class_weight=class_weight
            )
            base_models['RF_FeatureSelection'] = Pipeline([
                ('feature_selection', SelectFromModel(
                    rf_selector, threshold=feature_selection_threshold
                )),
                ('classifier', RandomForestClassifier(
                    n_estimators=100, random_state=42, class_weight=class_weight
                ))
            ])
            
            # RFE (Recursive Feature Elimination) with LogisticRegression
            if n_features < 100:
                base_models['RFE_LogisticRegression'] = Pipeline([
                    ('feature_selection', RFE(
                        LogisticRegression(random_state=42),
                        n_features_to_select=min(20, n_features // 2),
                        step=1
                    )),
                    ('classifier', LogisticRegression(
                        random_state=42, class_weight=class_weight, max_iter=1000
                    ))
                ])
    
    return base_models
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, is_binary, is_imbalanced, feature_set='original'):
    """
    Train and evaluate a machine learning model, calculating various performance metrics.
    
    Args:
        model: The machine learning model to evaluate
        X_train: Training features
        X_test: Testing features
        y_train: Training target values
        y_test: Testing target values
        model_name: Name of the model (for logging)
        is_binary: Whether this is a binary classification problem
        is_imbalanced: Whether the class distribution is imbalanced
        feature_set: Description of feature set being used (for logging)
    
    Returns:
        tuple: (trained model, primary score, f1 score, accuracy, AUC-ROC score, classification report)
    """
    # Train model
    print(f"Fitting {model_name} on {feature_set} features...")
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # Calculate F1 score with appropriate averaging for binary/multiclass
    if is_binary:
        f1 = f1_score(y_test, y_pred, average='binary')
    else:
        f1 = f1_score(y_test, y_pred, average='weighted')
    
    # ROC-AUC calculation if applicable
    auc_roc = 0
    try:
        if hasattr(model, "predict_proba"):
            if is_binary:
                # Binary classification
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                auc_roc = roc_auc_score(y_test, y_pred_proba)
            else:
                # Multi-class classification
                y_pred_proba = model.predict_proba(X_test)
                auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
    except (ValueError, IndexError) as e:
        print(f"Could not calculate ROC-AUC: {e}")
    
    # Print metrics
    print(f"{model_name} ({feature_set}) Accuracy: {accuracy:.4f}")
    print(f"{model_name} ({feature_set}) F1-Score: {f1:.4f}")
    if auc_roc > 0:
        print(f"{model_name} ({feature_set}) AUC-ROC: {auc_roc:.4f}")
    
    # Generate and print classification report
    report = classification_report(y_test, y_pred)
    print(f"{model_name} ({feature_set}) Classification Report:\n{report}")
    
    # For imbalanced datasets, F1 is more important than accuracy
    primary_score = accuracy #f1 if is_imbalanced else accuracy
    
    return model, primary_score, f1, accuracy, auc_roc, report


  

def create_fresh_model(name, is_binary, is_imbalanced, class_counts):
    """
    Create a fresh instance of a model with appropriate parameters.
    
    Args:
        name (str): Model name
        is_binary (bool): Whether this is a binary classification problem
        is_imbalanced (bool): Whether the class distribution is imbalanced
        class_counts (ndarray): Array of class counts
        
    Returns:
        A new model instance or None if model type not recognized
    """
    class_weight = 'balanced' if is_imbalanced else None
    
    if name == 'LogisticRegression':
        if is_binary:
            return LogisticRegression(
                random_state=42, solver='liblinear', class_weight=class_weight, max_iter=1000
            )
        else:
            return LogisticRegression(
                random_state=42, multi_class='auto', solver='lbfgs', 
                class_weight=class_weight, max_iter=1000
            )
    elif name == 'RandomForest':
        return RandomForestClassifier(
            random_state=42, class_weight=class_weight, n_estimators=100
        )
    elif name == 'GradientBoosting':
        return GradientBoostingClassifier(random_state=42, n_estimators=100)
    elif name == 'XGBoost':
        if is_binary:
            return XGBClassifier(
                eval_metric='logloss', random_state=42,
                scale_pos_weight=class_counts[0]/class_counts[1] if is_imbalanced and class_counts[1] > 0 else 1
            )
        else:
            return XGBClassifier(
                eval_metric='mlogloss', random_state=42
            )
    elif name == 'LightGBM':
        if is_binary:
            return LGBMClassifier(
                random_state=42, class_weight=class_weight if class_weight else 'balanced'
            )
        else:
            return LGBMClassifier(
                random_state=42, objective='multiclass', class_weight=class_weight
            )
    elif name == 'OVR_LogisticRegression':
        return OneVsRestClassifier(
            LogisticRegression(random_state=42, class_weight=class_weight, max_iter=1000)
        )
    else:
        return None  # Model type not recognized


def extract_feature_importances(model):
    """
    Extract feature importances from a trained model if available.
    
    Args:
        model: Trained model
        
    Returns:
        ndarray: Feature importances or None if not available
    """
    if hasattr(model, "feature_importances_"):
        return model.feature_importances_
    elif hasattr(model, "coef_"):
        if len(model.coef_.shape) > 1:
            # For multi-class, take average of absolute values across classes
            return np.mean(np.abs(model.coef_), axis=0)
        else:
            return np.abs(model.coef_[0])
    elif hasattr(model, "estimators_") and hasattr(model.estimators_[0], "feature_importances_"):
        # For ensemble of estimators like in OVR
        return np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
    else:
        print("Model does not support feature importance.")
        return None


def map_feature_importances_to_original(feature_importances, original_columns):
    """
    Map processed feature importances back to original column names.
    
    Args:
        feature_importances (dict): Dictionary of feature importances
        original_columns (list): List of original column names
        
    Returns:
        dict: Feature importances mapped to original columns
    """
    original_feature_importances = {}
    
    if not feature_importances:
        return original_feature_importances
    
    for processed_feature, importance in feature_importances.items():
        # Remove 'num__' or 'cat__' prefixes
        processed_feature = processed_feature.replace("num__", "").replace("cat__", "")
        
        # Find the original column(s) that contributed to this processed feature
        mapped_original_names = []
        for original_col in original_columns:
            if processed_feature.startswith(original_col):
                mapped_original_names.append(original_col)
        
        if mapped_original_names:
            original_feature_importances[", ".join(mapped_original_names)] = importance / len(mapped_original_names)
        else:
            original_feature_importances[processed_feature] = importance
    
    return original_feature_importances


 

def train_classification_model(target_variable, filepath, user_id, use_case, 
                             threshold=0.90, use_f1_for_threshold=False,
                             use_sampling=False, sample_size=100000, stratified_sampling=True,
                             use_rgs=False, n_rgs_iterations=20,
                             leakage_threshold=0.8, quick_mode=True):
    """
    Trains a classification model with advanced features for handling large datasets:
    - Efficient preprocessing with memory optimization
    - Data leakage prevention
    - Mixed data type handling
    - Randomized Grid Search for hyperparameter optimization
    - Stratified sampling for faster training
    - Early stopping when performance threshold is reached
    
    Args:
        target_variable (str): The name of the target variable column
        filepath (str): Path to the data file
        user_id (str): User ID for organization
        use_case (str): Use case identifier
        threshold (float): Stop training once a model achieves this score (default: 0.90)
        use_f1_for_threshold (bool): If True, use F1 for threshold; if False, use accuracy
        use_sampling (bool): Whether to use sampling for faster training (default: False)
        sample_size (int): Maximum number of samples to use if sampling is enabled (default: 100000)
        stratified_sampling (bool): Whether to use stratified sampling to maintain class distribution
        use_rgs (bool): Whether to use Randomized Grid Search for hyperparameter optimization
        n_rgs_iterations (int): Number of parameter combinations to try in RGS (default: 20)
        leakage_threshold (float): Correlation/MI threshold for detecting leaky features (default: 0.8)
        quick_mode (bool): Run in quick mode with fewer models and iterations (default: False)
    
    Returns:
        tuple: (model_path, features_path, best_score, feature_importances)
    """
    import pandas as pd
    import numpy as np
    import joblib
    import os
    import gc
    import time
    from sklearn.model_selection import train_test_split, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, f1_score
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    from scipy.stats import randint, uniform
    
    try:
        from xgboost import XGBClassifier
        xgboost_available = True
    except ImportError:
        xgboost_available = False
        print("XGBoost not available. Skipping XGBoost models.")
    
    try:
        from lightgbm import LGBMClassifier
        lightgbm_available = True
    except ImportError:
        lightgbm_available = False
        print("LightGBM not available. Skipping LightGBM models.")
    
    # Start timing
    start_time = time.time()
    
    # Import efficient preprocessing functions if available
    try:
        from efficient_preprocessing import efficient_preprocessing,save_preprocessor
        efficient_preproc_available = True
    except ImportError:
        print("Warning: Efficient preprocessing modules not found. Using default methods.")
        efficient_preproc_available = False
    
    # File and directory setup
    filename = os.path.basename(filepath)
    output_dir = get_output_dir(user_id, use_case, target_variable, filename)
    os.makedirs(output_dir, exist_ok=True)
    model_filename = get_model_filename(user_id, use_case, target_variable, filename)
    model_path = os.path.join(output_dir, model_filename)
    
    # Initialize tracking variables
    best_score = 0
    best_model = None
    best_report = None
    best_auc_roc = 0
    best_f1 = 0
    best_accuracy = 0
    early_stop = False
    
    # Define parameter distributions for RandomizedSearchCV
    def get_parameter_distributions(model_name, is_binary, quick_mode=False):
        """Define parameter distributions for RandomizedSearchCV"""
        if model_name == 'LogisticRegression':
            return {
                'C': uniform(0.01, 10),
                'solver': ['liblinear', 'saga'] if is_binary else ['lbfgs', 'saga'],
                'max_iter': randint(500, 2000)
            }
        elif model_name == 'RandomForest':
            # Reduced parameter space for quick mode
            if quick_mode:
                return {
                    'n_estimators': randint(50, 150),
                    'max_depth': [None, 10, 20],
                    'min_samples_split': randint(2, 10),
                    'min_samples_leaf': randint(1, 5)
                }
            else:
                return {
                    'n_estimators': randint(100, 300),
                    'max_depth': [None, 10, 20, 30, 40],
                    'min_samples_split': randint(2, 20),
                    'min_samples_leaf': randint(1, 10),
                    'max_features': ['sqrt', 'log2', None]
                }
        elif model_name == 'XGBoost':
            return {
                'n_estimators': randint(50, 300),
                'max_depth': randint(3, 10),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.5, 0.5),
                'colsample_bytree': uniform(0.5, 0.5)
            }
        elif model_name == 'LightGBM':
            return {
                'n_estimators': randint(50, 300),
                'num_leaves': randint(20, 100),
                'learning_rate': uniform(0.01, 0.3),
                'subsample': uniform(0.5, 0.5),
                'colsample_bytree': uniform(0.5, 0.5)
            }
        # Add more models as needed
        return {}

    # Helper function for class distribution
    def get_class_distribution(y):
        """Get class distribution regardless of whether labels are continuous"""
        classes, counts = np.unique(y, return_counts=True)
        return dict(zip(classes, counts))

    # Function to detect features with suspiciously high correlation with target
    def detect_leaky_features(df, target_variable, threshold=0.8):
        """
        Detect features that might be leaking target information due to high correlation.
        
        Args:
            df: DataFrame with features and target
            target_variable: Name of the target column
            threshold: Correlation threshold (default: 0.8)
            
        Returns:
            List of potentially leaky feature names with their scores
        """
        leaky_features = []
        
        # Get target variable
        y = df[target_variable]
        
        # Check if target is categorical
        target_is_categorical = y.dtype == 'object' or y.dtype.name == 'category'
        
        # For numerical target, check correlation directly
        if not target_is_categorical:
            numerical_features = df.select_dtypes(include=['number']).columns.tolist()
            if target_variable in numerical_features:
                numerical_features.remove(target_variable)
            
            if numerical_features:
                # Calculate correlation with target
                correlations = {}
                for col in numerical_features:
                    try:
                        corr = df[col].corr(y)
                        if abs(corr) > threshold:
                            correlations[col] = corr
                    except:
                        pass  # Skip features that cause errors
                
                # Sort by absolute correlation
                sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                
                # Add highly correlated features to the list
                for feature, corr in sorted_correlations:
                    leaky_features.append((feature, abs(corr)))
                    print(f"High correlation with target: {feature} ({corr:.4f})")
        
        # For categorical target or categorical features, use mutual information
        try:
            from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
            
            # Process categorical features
            categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if target_variable in categorical_features:
                categorical_features.remove(target_variable)
            
            if categorical_features:
                # Encode categorical features for MI calculation
                X_cat = df[categorical_features].copy()
                for col in X_cat.columns:
                    X_cat[col] = X_cat[col].astype('category').cat.codes
                
                # Calculate MI based on target type
                if target_is_categorical:
                    y_encoded = y.astype('category').cat.codes
                    mi_scores = mutual_info_classif(X_cat, y_encoded)
                else:
                    mi_scores = mutual_info_regression(X_cat, y)
                
                # Find features with high MI
                for i, col in enumerate(categorical_features):
                    if mi_scores[i] > threshold * 0.5:  # Adjust threshold for MI scale
                        leaky_features.append((col, mi_scores[i]))
                        print(f"High mutual information with target: {col} ({mi_scores[i]:.4f})")
            
            # Process numerical features for categorical target
            if target_is_categorical and numerical_features:
                # Encode target for MI calculation
                y_encoded = y.astype('category').cat.codes
                
                # Calculate MI
                mi_scores = mutual_info_classif(df[numerical_features], y_encoded)
                
                # Find features with high MI
                for i, col in enumerate(numerical_features):
                    if mi_scores[i] > threshold * 0.5:  # Adjust threshold for MI scale
                        # Only add if not already in the list
                        if col not in [f[0] for f in leaky_features]:
                            leaky_features.append((col, mi_scores[i]))
                            print(f"High mutual information with target: {col} ({mi_scores[i]:.4f})")
        except Exception as e:
            print(f"Could not calculate mutual information: {e}")
        
        return leaky_features
    
    # Function to preprocess target variable with mixed type handling
    def preprocess_target_variable(df, target_variable):
        """
        Preprocess the target variable using LabelEncoder
        Handles mixed data types by converting to string
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_variable (str): Name of the target variable column
        
        Returns:
            tuple: (processed target variable, LabelEncoder)
        """
        # Get the original target variable series
        y = df[target_variable].copy()
        
        # Check for mixed types in the target variable
        try:
            # Force values to be either all strings or all numeric for consistency
            sample_values = y.dropna().head(1000).tolist()
            types_present = set(type(val) for val in sample_values)
            
            if len(types_present) > 1:
                print(f"Warning: Mixed data types detected in target variable: {[t.__name__ for t in types_present]}")
                print("Converting all values to strings for consistent encoding...")
                # Convert all values to strings to ensure consistent types
                y = y.astype(str)
        except Exception as e:
            print(f"Warning when checking target types: {e}")
            # Force conversion to string as a fallback
            y = y.fillna("None").astype(str)
            print("Forced conversion of target to string type")
        
        # Initialize LabelEncoder
        le = LabelEncoder()
        
        try:
            # Try to fit transform directly
            y_encoded = le.fit_transform(y)
            
            # Get information about the encoded values
            print(f"Target variable '{target_variable}' encoded successfully.")
            print(f"Number of unique classes: {len(le.classes_)}")
            
            # Output more information based on data type
            if hasattr(y, 'dtype') and (y.dtype == 'object' or pd.api.types.is_categorical_dtype(y)):
                # For categorical data, show distribution
                categories = y.value_counts()
                print("Target variable category distribution (top 10):")
                print(categories.head(10))
                
                if len(le.classes_) > 10:
                    print(f"Note: {len(le.classes_)} total unique classes present")
            else:
                # For numerical data, show range
                try:
                    min_value = y.min()
                    max_value = y.max()
                    print(f"Original target variable range: {min_value} to {max_value}")
                except:
                    pass
            
            print(f"Encoded target variable unique values: {np.unique(y_encoded)}")
            
            return y_encoded, le
            
        except TypeError as te:
            # If we still get a type error, it means there might be complex mixed types
            # Convert to string and try again
            print(f"Error during encoding: {te}")
            print("Falling back to string conversion for problematic target variable")
            
            # Force string conversion
            y = y.fillna("None").astype(str)
            y_encoded = le.fit_transform(y)
            
            print(f"Successfully encoded target after type conversion")
            print(f"Number of unique classes: {len(le.classes_)}")
            print(f"Encoded target variable unique values: {np.unique(y_encoded)}")
            
            return y_encoded, le
        
        except Exception as e:
            # Handle any other errors
            print(f"Unexpected error during target encoding: {e}")
            
            # Create a simple integer encoding as a final fallback
            print("Using simple integer encoding as final fallback")
            unique_values = y.fillna("None").unique()
            value_to_int = {val: i for i, val in enumerate(unique_values)}
            y_encoded = np.array([value_to_int.get(val, 0) for val in y])
            
            # Create a minimal label encoder with the mapping
            le = LabelEncoder()
            le.classes_ = np.array(list(value_to_int.keys()))
            
            print(f"Fallback encoding created with {len(unique_values)} classes")
            print(f"Encoded target variable unique values: {np.unique(y_encoded)}")
            
            return y_encoded, le
    
    # Function to evaluate model performance
    def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
        """Evaluate a model and return performance metrics"""
        train_start = time.time()
        print(f"Training {model_name}...")
        
        model.fit(X_train, y_train)
        train_time = time.time() - train_start
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # F1 score with appropriate averaging
        if is_binary:
            f1 = f1_score(y_test, y_pred, average='binary')
        else:
            f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC-AUC if possible
        auc_roc = 0
        try:
            if hasattr(model, "predict_proba"):
                if is_binary:
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    auc_roc = roc_auc_score(y_test, y_pred_proba)
                else:
                    try:
                        y_pred_proba = model.predict_proba(X_test)
                        auc_roc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                    except ValueError as ve:
                        print(f"Could not calculate multi-class ROC-AUC: {ve}")
        except Exception as e:
            print(f"Could not calculate ROC-AUC: {e}")
        
        # Print metrics
        print(f"{model_name} - Trained in {train_time:.1f}s - Accuracy: {accuracy:.4f}, F1: {f1:.4f}" + 
              (f", AUC: {auc_roc:.4f}" if auc_roc > 0 else ""))
        
        # Generate classification report
        report = classification_report(y_test, y_pred)
        
        # Primary score for model selection
        primary_score = f1 if is_imbalanced else accuracy
        
        return model, primary_score, f1, accuracy, auc_roc, report

    # Print status
    print(f"Starting training for target: '{target_variable}' on file: '{filepath}'")
    if quick_mode:
        print("Running in quick mode with reduced complexity")
    if use_sampling:
        print(f"Using sampling: max {sample_size} samples with {'stratified' if stratified_sampling else 'random'} sampling")
    if use_rgs:
        print(f"Using Randomized Grid Search with {n_rgs_iterations} iterations for hyperparameter optimization")
    
    # Read the data
    print(f"Reading data from {filepath}...")
    original_df = read_data_flexible(filepath)
    if original_df is None:
        print("Error: Failed to read the data. Training aborted.")
        return None, None, 0, {}
    
    # Store original dataset size for reference
    original_size = len(original_df)
    print(f"Original dataset size: {original_size} rows, {len(original_df.columns)} columns")
    
    # Handle mixed types in target variable right after loading
    if target_variable in original_df.columns:
        try:
            # Check for mixed types in target
            sample_values = original_df[target_variable].dropna().head(1000).tolist()
            types_present = set(type(val) for val in sample_values)
            
            if len(types_present) > 1:
                print(f"Warning: Mixed data types detected in target variable: {[t.__name__ for t in types_present]}")
                print("Converting target to string type to ensure consistency...")
                original_df[target_variable] = original_df[target_variable].astype(str)
        except Exception as e:
            print(f"Warning: Could not check target variable types: {e}")
    
    # Apply sampling if requested and needed
    sampled_for_training = False
    if use_sampling and len(original_df) > sample_size:
        print(f"Sampling dataset from {len(original_df)} to {sample_size} rows...")
        
        # For stratified sampling, we need to encode the target first
        if stratified_sampling:
            y = original_df[target_variable]
            if y.dtype == 'object' or y.dtype.name == 'category':
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                try:
                    y_encoded = le.fit_transform(y)
                except TypeError:
                    # Handle mixed type error
                    y = y.astype(str)
                    y_encoded = le.fit_transform(y)
                
                # Get class distribution before sampling
                before_dist = get_class_distribution(y_encoded)
                
                # Perform stratified sampling
                from sklearn.model_selection import train_test_split
                _, sampled_idx = train_test_split(
                    np.arange(len(original_df)),
                    test_size=min(sample_size/len(original_df), 0.99),  # Ensure we don't exceed 99%
                    stratify=y_encoded,
                    random_state=42
                )
                df_for_training = original_df.iloc[sampled_idx].copy()
                
                # Check class distribution after sampling
                sampled_y = y_encoded[sampled_idx]
                after_dist = get_class_distribution(sampled_y)
                print(f"Class distribution maintained with stratified sampling:")
                for cls in before_dist:
                    if cls in after_dist:
                        before_pct = before_dist[cls] / len(y_encoded) * 100
                        after_pct = after_dist[cls] / len(sampled_y) * 100
                        print(f"  Class {cls}: {before_pct:.1f}% → {after_pct:.1f}%")
            else:
                # Simple random sampling for numerical targets
                df_for_training = original_df.sample(sample_size, random_state=42)
        else:
            # Simple random sampling
            df_for_training = original_df.sample(sample_size, random_state=42)
        
        sampled_for_training = True
        print(f"Sampled dataset size: {len(df_for_training)} rows")
    else:
        # Use the original dataset
        df_for_training = original_df
    
    # Store original column names for feature importance mapping
    original_columns = list(original_df.columns)
    
    # Check for high correlation with target variable
    print("Checking for features highly correlated with target variable...")
    # Use a sample if dataset is large
    if len(df_for_training) > 50000:
        print("Using a sample to detect leaky features...")
        sample_for_leakage = df_for_training.sample(50000, random_state=42)
        potentially_leaky_features = detect_leaky_features(sample_for_leakage, target_variable, leakage_threshold)
        del sample_for_leakage
        gc.collect()
    else:
        potentially_leaky_features = detect_leaky_features(df_for_training, target_variable, leakage_threshold)
    
    # Extract just the feature names from the leaky features list (without scores)
    leaky_feature_names = [f[0] for f in potentially_leaky_features]
    
    if leaky_feature_names:
        print(f"Removing {len(leaky_feature_names)} potentially leaky features: {leaky_feature_names}")
        # Remove leaky features from the dataset before preprocessing
        df_for_training = df_for_training.drop(columns=leaky_feature_names)
        # Record the removed features
        leakage_report_path = os.path.join(output_dir, f"leakage_report_{model_filename}.txt")
        with open(leakage_report_path, 'w') as f:
            f.write(f"Potentially leaky features for target '{target_variable}':\n")
            for feature, score in potentially_leaky_features:
                f.write(f"{feature}: {score:.4f}\n")
        print(f"Saved leakage report to: {leakage_report_path}")
    else:
        print("No potentially leaky features detected")
    
    # Apply preprocessing
    print("Preprocessing data...")
    try:
        # Use efficient preprocessing if available
        if efficient_preproc_available:
            # Configure preprocessing parameters
            max_categories = 10
            max_dummy_features = 500
            sample_for_analysis = df_for_training.shape[0] > 100000 or df_for_training.shape[1] > 50
            sample_size_analysis = min(50000, len(df_for_training) // 2) if sample_for_analysis else None
            
            preprocessing_start = time.time()
            # Process the data
            preprocessor, X_processed, feature_names = efficient_preprocessing(
                df=df_for_training,
                target_variable=target_variable,
                max_categories=max_categories,
                max_dummy_features=max_dummy_features,
                sample_for_analysis=sample_for_analysis,
                sample_size=sample_size_analysis,
                memory_efficient=True,
                feature_selection=True,
                max_features=100 if not quick_mode else 50,  # Use fewer features in quick mode
                categorical_encoding_method='auto',
                verbose=True
            )
            print(f"Preprocessing completed in {time.time() - preprocessing_start:.1f} seconds")
        else:
            # Fall back to traditional preprocessing
            raise ImportError("Efficient preprocessing not available")
            
    except Exception as e:
        print(f"Error in efficient preprocessing: {str(e)}")
        print("Using standard preprocessing...")
        
        preprocessing_start = time.time()
        # Screen features to remove problematic ones
        df, dropped_features, _ = screen_features(
            df_for_training, 
            target_variable,
            correlation_threshold=0.85,
            mi_threshold=0.8,
            verbose=True,
            plot=False
        )
        
        # Prepare data for preprocessing
        X_df = df.drop(columns=[target_variable])
        
        # Identify feature types
        numerical_features = X_df.select_dtypes(include=['number']).columns.tolist()
        categorical_features = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Identify high cardinality features
        high_cardinality_features = []
        low_cardinality_features = []
        
        for col in categorical_features:
            if X_df[col].nunique() > 10:
                high_cardinality_features.append(col)
            else:
                low_cardinality_features.append(col)
        
        print(f"Found {len(numerical_features)} numerical, {len(low_cardinality_features)} low-cardinality, "
              f"and {len(high_cardinality_features)} high-cardinality features")
        
        # Create preprocessing pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        
        transformers = []
        
        # Add numerical transformer
        if numerical_features:
            numerical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', numerical_transformer, numerical_features))
        
        # Add low-cardinality transformer
        if low_cardinality_features:
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
            ])
            transformers.append(('cat_low', categorical_transformer, low_cardinality_features))
        
        # Add high-cardinality transformer using ordinal encoding
        if high_cardinality_features:
            hc_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
            transformers.append(('cat_high', hc_transformer, high_cardinality_features))
        
        # Create and fit preprocessor
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        
        try:
            print("Fitting preprocessor...")
            preprocessor.fit(X_df)
            X_processed = preprocessor.transform(X_df)
            
            # Get feature names
            try:
                feature_names = preprocessor.get_feature_names_out()
            except AttributeError:
                feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
            
            print(f"Preprocessing completed in {time.time() - preprocessing_start:.1f} seconds")
                
        except Exception as inner_e:
            print(f"Error in standard preprocessing: {str(inner_e)}")
            print("Using simplified preprocessing...")
            
            # Ultimate fallback: only use numerical features
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', SimpleImputer(strategy='median'), numerical_features)
                ],
                remainder='drop'
            )
            
            preprocessor.fit(X_df)
            X_processed = preprocessor.transform(X_df)
            feature_names = numerical_features
            print(f"Fallback preprocessing completed in {time.time() - preprocessing_start:.1f} seconds")
    
    # Process target variable using the enhanced function
    print("Processing target variable...")
    try:
        # Process target with mixed type handling
        y_encoded, label_encoder = preprocess_target_variable(df_for_training, target_variable)
    except Exception as e:
        print(f"Error in target preprocessing: {str(e)}")
        # Fallback to simpler approach
        y = df_for_training[target_variable].astype(str)
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    
    # Ensure X_processed is a DataFrame if possible
    if not isinstance(X_processed, pd.DataFrame):
        try:
            if hasattr(X_processed, "toarray"):
                X_processed = pd.DataFrame(X_processed.toarray(), columns=feature_names)
            else:
                X_processed = pd.DataFrame(X_processed, columns=feature_names)
        except Exception as e:
            print(f"Could not convert to DataFrame: {str(e)}. Using array directly.")
    
    # Check class distribution before splitting
    full_distribution = get_class_distribution(y_encoded)
    min_class_count = min(full_distribution.values())
    num_classes = len(full_distribution)
    
    # Detect classification type and class imbalance
    unique_classes = np.unique(y_encoded)
    num_classes = len(unique_classes)
    is_binary = num_classes == 2
    
    print(f"{'Binary' if is_binary else 'Multi-class'} classification with {num_classes} classes")
    
    # Check for class imbalance
    max_class_count = max(full_distribution.values())
    class_ratio = min_class_count / max_class_count
    is_imbalanced = class_ratio < 0.25
    
    if is_imbalanced:
        print(f"Class imbalance detected. Min/Max ratio: {class_ratio:.3f}")
        print(f"Class distribution: {full_distribution}")
    
    # Determine evaluation metric
    if use_f1_for_threshold is None:
        use_f1_for_threshold = is_imbalanced
    
    metric_name = "F1 score" if use_f1_for_threshold else "Accuracy"
    print(f"Using {metric_name} with threshold {threshold:.2f} for early stopping")
    
    # Split data
    print("Splitting data...")
    # Decide whether to use stratification
    if min_class_count > 1:
        # Safe to use stratification
        print(f"Using stratified split (min class count: {min_class_count}, classes: {num_classes})")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
    else:
       
        # Cannot use stratification - some classes have only one example
        print(f"Warning: Cannot use stratified split - some classes have only one example. Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y_encoded, test_size=0.2, random_state=42
        )
        
        # Print distribution of classes in train and test sets
        train_distribution = get_class_distribution(y_train)
        test_distribution = get_class_distribution(y_test)
        
        print(f"Train set class distribution: {train_distribution}")
        print(f"Test set class distribution: {test_distribution}")
        
        # Check if any classes are missing from either set
        missing_in_train = [cls for cls in full_distribution if cls not in train_distribution]
        missing_in_test = [cls for cls in full_distribution if cls not in test_distribution]
        
        if missing_in_train:
            print(f"Warning: {len(missing_in_train)} classes are missing from training set: {missing_in_train}")
        if missing_in_test:
            print(f"Warning: {len(missing_in_test)} classes are missing from test set: {missing_in_test}")
    
    # Free memory
    del X_processed
    gc.collect()

    # Configure models
    n_samples, n_features = X_train.shape
    print(f"Training data shape: {n_samples} samples, {n_features} features")
    
    # Prepare class counts for model configuration
    from collections import Counter
    class_counts_array = np.zeros(num_classes, dtype=int)
    for k, v in Counter(y_encoded).items():
        if k < num_classes:  # Ensure index is within bounds
            class_counts_array[k] = v
    
    # Get model configurations based on mode
    if quick_mode:
        # Create a simplified model configuration
        models = {
            'LogisticRegression': LogisticRegression(
                solver='liblinear' if is_binary else 'lbfgs',
                C=1.0,
                max_iter=500,  # Reduced iterations
                random_state=42
            ),
            'RandomForest': RandomForestClassifier(
                n_estimators=50,  # Reduced trees
                max_depth=10,     # Limited depth
                random_state=42,
                n_jobs=-1
            )
        }
        
        # Add one more model based on availability
        if xgboost_available:
            # For small datasets, add XGBoost
            models['XGBoost'] = XGBClassifier(
                n_estimators=50,  # Reduced trees
                max_depth=4,      # Limited depth
                learning_rate=0.1,
                random_state=42
            )
    else:
        # Use full model configurations
        models = configure_advanced_models(
            is_binary=is_binary,
            num_classes=num_classes,
            class_counts=class_counts_array,
            is_imbalanced=is_imbalanced,
            n_features=n_features,
            n_samples=n_samples
        )

    # Train and evaluate models
    print("\nTraining models...")
    model_start_time = time.time()
    
    # Use randomized grid search if requested
    if use_rgs:
        print("\nUsing Randomized Grid Search for hyperparameter optimization...")
        
        # Train and evaluate models
        for name, model in list(models.items()):
            if early_stop:
                print(f"Stopping early: {metric_name} threshold of {threshold:.2f} reached.")
                break
                
            try:
                # Set environment variables for parallel processing
                os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
                
                # Get parameter distributions for this model
                param_distributions = get_parameter_distributions(name, is_binary, quick_mode)
                
                if not param_distributions:
                    print(f"Skipping RGS for {name} - no parameter distributions defined")
                    # Fallback to regular training
                    trained_model, primary_score, f1, acc, auc, report = evaluate_model(
                        model, X_train, X_test, y_train, y_test, name
                    )
                else:
                    print(f"Running RandomizedSearchCV for {name}...")
                    
                    # Define scoring metric based on problem type
                    scoring = 'f1' if is_binary else 'f1_weighted' if is_imbalanced else 'accuracy'
                    
                    # Create RandomizedSearchCV
                    rgs = RandomizedSearchCV(
                        model,
                        param_distributions=param_distributions,
                        n_iter=n_rgs_iterations,
                        cv=3,  # Use 3-fold CV for speed, increase for better results
                        scoring=scoring,
                        random_state=42,
                        n_jobs=-1,
                        verbose=1
                    )
                    
                    # Train model
                    rgs_start = time.time()
                    rgs.fit(X_train, y_train)
                    
                    # Get best model
                    best_rgs_model = rgs.best_estimator_
                    
                    # Evaluate on test set
                    y_pred = best_rgs_model.predict(X_test)
                    
                    # Calculate metrics
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='binary' if is_binary else 'weighted')
                    
                    # Calculate ROC-AUC if possible
                    auc = 0
                    try:
                        if hasattr(best_rgs_model, "predict_proba"):
                            if is_binary:
                                y_pred_proba = best_rgs_model.predict_proba(X_test)[:, 1]
                                auc = roc_auc_score(y_test, y_pred_proba)
                            else:
                                y_pred_proba = best_rgs_model.predict_proba(X_test)
                                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='macro')
                    except Exception as e:
                        print(f"Could not calculate ROC-AUC: {e}")
                    
                    # Generate classification report
                    report = classification_report(y_test, y_pred)
                    
                    # Determine primary score
                    primary_score = acc#f1 if is_imbalanced else acc
                    
                    # Print results
                    print(f"{name} with RGS - Time: {time.time() - rgs_start:.1f}s")
                    print(f"Best parameters: {rgs.best_params_}")
                    print(f"Accuracy: {acc:.4f}, F1: {f1:.4f}" + 
                          (f", AUC: {auc:.4f}" if auc > 0 else ""))
                    
                    # Set variables for return
                    trained_model = best_rgs_model
                
                # Check if this is the best model so far
                if primary_score > best_score:
                    best_score = primary_score
                    best_model = trained_model
                    best_report = report
                    best_auc_roc = auc
                    best_f1 = f1
                    best_accuracy = acc
                    print(f"New best model: {name} with score: {best_score:.4f}")
                    
                    # Check for early stopping
                    threshold_metric = f1 if use_f1_for_threshold else acc
                    if threshold_metric >= threshold:
                        print(f"{metric_name} threshold met: {threshold_metric:.4f} >= {threshold:.2f}")
                        early_stop = True
                        
            except Exception as e:
                print(f"Error in RGS for {name}: {e}")
                continue
    else:
        # Train without RGS
        for name, model in models.items():
            if early_stop:
                print(f"Stopping early: {metric_name} threshold of {threshold:.2f} reached.")
                break
                
            try:
                # Set environment variables for parallel processing
                os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
                
                # Train and evaluate
                trained_model, primary_score, f1, acc, auc, report = evaluate_model(
                    model, X_train, X_test, y_train, y_test, name
                )
                
                # Check if this is the best model so far
                if primary_score > best_score:
                    best_score = primary_score
                    best_model = trained_model
                    best_report = report
                    best_auc_roc = auc
                    best_f1 = f1
                    best_accuracy = acc
                    print(f"New best model: {name} with score: {best_score:.4f}")
                    
                    # Check for early stopping
                    threshold_metric = f1 if use_f1_for_threshold else acc
                    if threshold_metric >= threshold:
                        print(f"{metric_name} threshold met: {threshold_metric:.4f} >= {threshold:.2f}")
                        early_stop = True
                        
            except Exception as e:
                print(f"Error training {name}: {e}")
                continue
    
    total_model_time = time.time() - model_start_time
    print(f"Model training completed in {total_model_time:.1f} seconds")

    # Print final results
    print("\nTraining complete.")
    if best_model is None:
        print("No successful models were trained.")
        return None, None, 0, {}
        
    print(f"Best model: {type(best_model).__name__}")
    print(f"Metrics: F1={best_f1:.4f}, Accuracy={best_accuracy:.4f}, AUC-ROC={best_auc_roc:.4f}")
    print(f"\nClassification Report:\n{best_report}")

    # Extract feature importances
    feature_importances = None
    if hasattr(best_model, "feature_importances_"):
        feature_importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        if len(best_model.coef_.shape) > 1:
            feature_importances = np.mean(np.abs(best_model.coef_), axis=0)
        else:
            feature_importances = np.abs(best_model.coef_[0])
    elif hasattr(best_model, "estimators_") and hasattr(best_model.estimators_[0], "feature_importances_"):
        feature_importances = np.mean([est.feature_importances_ for est in best_model.estimators_], axis=0)
    
    # Map feature importances
    original_feature_importances = {}
    if feature_importances is not None:
        best_feature_importances = dict(zip(feature_names, feature_importances))
        
        # Map back to original column names
        for processed_feature, importance in best_feature_importances.items():
            # Extract original feature name by removing prefixes/suffixes
            processed_feature_clean = processed_feature.replace("num__", "").replace("cat_low__", "").replace("cat_high__", "")
            
            # Handle one-hot encoded features (format: feature__value)
            parts = processed_feature_clean.split('__')
            base_feature = parts[0] if len(parts) > 0 else processed_feature_clean
            
            # Find matching original column
            for original_col in original_columns:
                if base_feature.startswith(original_col):
                    if original_col in original_feature_importances:
                        original_feature_importances[original_col] += importance
                    else:
                        original_feature_importances[original_col] = importance
                    break
            else:
                # If no match found, use the processed feature name
                original_feature_importances[processed_feature] = importance
    
    # Add note about sampling, leakage and RGS to output
    processing_info = ""
    if sampled_for_training:
        processing_info += f"_sampled_{sample_size}"
    if leaky_feature_names:
        processing_info += f"_leakage_removed_{len(leaky_feature_names)}"
    if use_rgs:
        processing_info += f"_rgs_{n_rgs_iterations}"
    
    # Save model and artifacts
    print(f"Saving model to: {model_path}")
    joblib.dump(best_model, model_path)
    
    # Save preprocessor with additional info
    preprocessor_path = os.path.join(output_dir, f"preprocessor_{model_filename}{processing_info}")
    
    # Use save_preprocessor if available, otherwise use joblib.dump
    try:
        if efficient_preproc_available:
            save_preprocessor(
                preprocessor=preprocessor,
                filepath=preprocessor_path,
                additional_info={
                    'feature_names': feature_names,
                    'target_variable': target_variable,
                    'sampled_training': sampled_for_training,
                    'sample_size': sample_size if sampled_for_training else original_size,
                    'original_size': original_size,
                    'removed_leaky_features': leaky_feature_names,
                    'leakage_threshold': leakage_threshold,
                    'used_rgs': use_rgs,
                    'rgs_iterations': n_rgs_iterations if use_rgs else 0
                }
            )
        else:
            joblib.dump({
                'preprocessor': preprocessor, 
                'feature_names': feature_names,
                'target_variable': target_variable,
                'sampled_training': sampled_for_training,
                'sample_size': sample_size if sampled_for_training else original_size,
                'original_size': original_size,
                'removed_leaky_features': leaky_feature_names,
                'leakage_threshold': leakage_threshold,
                'used_rgs': use_rgs,
                'rgs_iterations': n_rgs_iterations if use_rgs else 0
            }, preprocessor_path)
    except Exception as e:
        print(f"Error saving preprocessor: {e}")
        # Basic fallback
        joblib.dump(preprocessor, preprocessor_path)
    
    print(f"Saved preprocessor to: {preprocessor_path}")

    # Save feature names
    features_filename = os.path.join(output_dir, f"model_features_{model_filename}{processing_info}")
    joblib.dump(feature_names, features_filename)
    
    # Save label encoder if used
    if label_encoder:
        encoder_path = os.path.join(output_dir, f"label_encoder_{use_case}_{target_variable}{processing_info}.joblib")
        joblib.dump(label_encoder, encoder_path)
        
        # Save class mapping
        if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
            class_mapping = {i: class_name for i, class_name in enumerate(label_encoder.classes_)}
            mapping_path = os.path.join(output_dir, f"class_mapping_{filename}{processing_info}.csv")
            pd.DataFrame({
                'Class_ID': list(class_mapping.keys()),
                'Class_Name': list(class_mapping.values())
            }).to_csv(mapping_path, index=False)
    
    # Save information about the training process
    training_info_path = os.path.join(output_dir, f"training_info_{model_filename}{processing_info}.txt")
    with open(training_info_path, 'w') as f:
        f.write(f"Training Information for {target_variable} model\n")
        f.write(f"====================================\n\n")
        f.write(f"Dataset: {filepath}\n")
        f.write(f"Original size: {original_size} rows\n")
        if sampled_for_training:
            f.write(f"Training sample: {sample_size} rows ({sample_size/original_size*100:.1f}%)\n")
        if leaky_feature_names:
            f.write(f"\nRemoved {len(leaky_feature_names)} leaky features:\n")
            for feature, score in potentially_leaky_features:
                f.write(f"  - {feature}: {score:.4f}\n")
        f.write(f"\nTraining parameters:\n")
        f.write(f"  - RGS: {'Yes' if use_rgs else 'No'}")
        if use_rgs:
            f.write(f" ({n_rgs_iterations} iterations)\n")
        else:
            f.write(f"\n")
        f.write(f"  - Quick mode: {'Yes' if quick_mode else 'No'}\n")
        f.write(f"  - Target threshold: {threshold} ({metric_name})\n")
        f.write(f"\nModel performance:\n")
        f.write(f"  - Model type: {type(best_model).__name__}\n")
        f.write(f"  - F1 Score: {best_f1:.4f}\n")
        f.write(f"  - Accuracy: {best_accuracy:.4f}\n")
        f.write(f"  - AUC-ROC: {best_auc_roc:.4f}\n\n")
        f.write(f"Classification Report:\n{best_report}\n")
    
    print(f"Saved training information to: {training_info_path}")
    
    # Print summary messages
    if leaky_feature_names:
        print(f"Note: {len(leaky_feature_names)} potentially leaky features were removed during training")
    if sampled_for_training:
        print(f"Note: Model was trained on a {sample_size}/{original_size} sample of the full dataset")
    if use_rgs:
        print(f"Note: Model was optimized using Randomized Grid Search with {n_rgs_iterations} iterations")
    if early_stop:
        print(f"Training stopped early: {metric_name} threshold of {threshold:.2f} was reached.")
    
    total_time = time.time() - start_time
    print(f"Total processing time: {total_time:.1f} seconds")

    return model_path, features_filename, best_score, original_feature_importances
