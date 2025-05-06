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



target_variable="marketing_channel"
filepath="../uploads/ecommerce_multiclass_data.csv"
user_id='1'
use_case=target_variable
model_path, features_path, accuracy, importances = train_classification_model( target_variable,  filepath,   user_id, use_case,
                                                                              threshold=0.60,
                                                                              use_f1_for_threshold=False )
 
 

"""

import pandas as pd
import numpy as np
import joblib
import os
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

# Import preprocessing modules (try different import approaches)
try:
    # Try absolute import
    from ml.preprocess_data import (
        preprocess_data, preprocess_target_variable, create_preprocessor,
        save_preprocessor, save_label_encoder, get_output_dir, get_model_filename
    )
    from ml.read_file import read_data_flexible
except ImportError:
    try:
        # Try direct import
        from preprocess_data import (
            preprocess_data, preprocess_target_variable, create_preprocessor,
            save_preprocessor, save_label_encoder, get_output_dir, get_model_filename
        )
        from read_file import read_data_flexible
    except ImportError:
        # Try relative import
        from .preprocess_data import (
            preprocess_data, preprocess_target_variable, create_preprocessor,
            save_preprocessor, save_label_encoder, get_output_dir, get_model_filename
        )
        from .read_file import read_data_flexible

def configure_advanced_models(is_binary, num_classes, class_counts, is_imbalanced, n_features, n_samples):
    """
    Creates advanced model configurations optimized for either binary or multi-class classification.
    
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
        # Calculate balanced class weights
        n_samples_total = np.sum(class_counts)
        class_weight_dict = {i: n_samples_total / (len(class_counts) * count) for i, count in enumerate(class_counts)}
        print(f"Using class weights: {class_weight_dict}")
        class_weight = class_weight_dict
    else:
        class_weight = None
    
    # Feature selection settings
    use_feature_selection = n_features > 20  # Only use feature selection with many features
    feature_selection_threshold = 'median' if n_features > 100 else '1.25*mean'
    
    # Model complexity settings based on dataset size
    if n_samples < 1000:
        complexity = "low"      # Simple models to avoid overfitting
    elif n_samples < 10000:
        complexity = "medium"   # Balanced complexity
    elif n_samples < 100000:
        complexity = "high"     # More complex models
    else:
        complexity = "very_high" # Very complex models for big data
    
    # Configure base models with complexity-appropriate parameters
    if is_binary:
        # Binary classification models
        base_models = {}
        
        # Logistic Regression variants
        if complexity in ["low", "medium"]:
            base_models['LogisticRegression'] = LogisticRegression(
                random_state=42, 
                solver='liblinear',
                C=1.0,
                class_weight=class_weight,
                max_iter=2000,
                tol=1e-4
            )
        else:
            # Stronger regularization for larger datasets
            base_models['LogisticRegression_L1'] = LogisticRegression(
                random_state=42, 
                solver='liblinear',
                penalty='l1',
                C=0.1,
                class_weight=class_weight,
                max_iter=2000
            )
            base_models['LogisticRegression_L2'] = LogisticRegression(
                random_state=42, 
                solver='liblinear',
                penalty='l2',
                C=0.1,
                class_weight=class_weight, 
                max_iter=2000
            )
            
        # Tree-based models with complexity-appropriate parameters
        if complexity == "low":
            # Simpler tree models for small datasets
            base_models['RandomForest'] = RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight,
                bootstrap=True
            )
            base_models['GradientBoosting'] = GradientBoostingClassifier(
                random_state=42,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                subsample=0.8,
                min_samples_split=5
            )
        elif complexity == "medium":
            # Medium complexity for medium-sized datasets
            base_models['RandomForest'] = RandomForestClassifier(
                random_state=42,
                n_estimators=200,
                max_depth=None,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight=class_weight,
                bootstrap=True
            )
            base_models['GradientBoosting'] = GradientBoostingClassifier(
                random_state=42,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                min_samples_split=3
            )
        else:
            # Higher complexity for large datasets
            base_models['RandomForest'] = RandomForestClassifier(
                random_state=42,
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight=class_weight,
                bootstrap=True,
                n_jobs=-1
            )
            base_models['GradientBoosting'] = GradientBoostingClassifier(
                random_state=42,
                n_estimators=300,
                learning_rate=0.01,
                max_depth=8,
                subsample=0.8,
                min_samples_split=2
            )
            
            # HistGradientBoosting is faster for large datasets
            base_models['HistGradientBoosting'] = HistGradientBoostingClassifier(
                random_state=42,
                max_iter=300,
                learning_rate=0.1,
                max_depth=None,
                min_samples_leaf=20,
                l2_regularization=0.1
            )
        
        # XGBoost with appropriate scale_pos_weight for imbalanced data
        if complexity == "low":
            base_models['XGBoost'] = XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=class_counts[0]/class_counts[1] if is_imbalanced and class_counts[1] > 0 else 1
            )
        elif complexity == "medium":
            base_models['XGBoost'] = XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                gamma=0.1,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=class_counts[0]/class_counts[1] if is_imbalanced and class_counts[1] > 0 else 1,
                tree_method='auto'
            )
        else:
            base_models['XGBoost'] = XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                n_estimators=300,
                learning_rate=0.01,
                max_depth=8,
                gamma=0.1,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                scale_pos_weight=class_counts[0]/class_counts[1] if is_imbalanced and class_counts[1] > 0 else 1,
                tree_method='auto',
                n_jobs=-1
            )
        
        # LightGBM for faster processing on larger datasets
        if n_samples < 100000:
            if complexity == "low":
                base_models['LightGBM'] = LGBMClassifier(
                    random_state=42,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight=class_weight if class_weight else 'balanced'
                )
            elif complexity == "medium":
                base_models['LightGBM'] = LGBMClassifier(
                    random_state=42,
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=63,
                    min_child_samples=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight=class_weight if class_weight else 'balanced'
                )
            else:
                base_models['LightGBM'] = LGBMClassifier(
                    random_state=42,
                    n_estimators=300,
                    learning_rate=0.01,
                    max_depth=8,
                    num_leaves=127,
                    min_child_samples=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    class_weight=class_weight if class_weight else 'balanced',
                    n_jobs=-1
                )
        
        # SVM models for smaller datasets
        if n_samples < 10000:
            base_models['SVM'] = SVC(
                random_state=42,
                probability=True,
                C=1.0,
                kernel='rbf',
                gamma='scale',
                class_weight=class_weight
            )
            
            # Linear SVM with calibration
            if n_samples < 5000:
                linear_svc = LinearSVC(
                    random_state=42,
                    C=1.0,
                    class_weight=class_weight,
                    max_iter=2000,
                    dual='auto'  # Updated for newer sklearn
                )
                base_models['CalibratedSVM'] = CalibratedClassifierCV(
                    linear_svc, method='sigmoid', cv=3
                )
        
        # Neural network for medium datasets
        if n_samples > 1000 and n_samples < 50000:
            base_models['NeuralNetwork'] = MLPClassifier(
                random_state=42,
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True
            )
        
        # Naive Bayes as a baseline
        base_models['GaussianNB'] = GaussianNB()
        
        # Add feature selection if appropriate
        if use_feature_selection:
            # RandomForest-based feature selection
            rf_selector = RandomForestClassifier(
                n_estimators=100, random_state=42, class_weight=class_weight
            )
            base_models['RF_FeatureSelection'] = Pipeline([
                ('feature_selection', SelectFromModel(
                    rf_selector, threshold=feature_selection_threshold
                )),
                ('classifier', RandomForestClassifier(
                    n_estimators=200, random_state=42, class_weight=class_weight
                ))
            ])
            
            # RFE (Recursive Feature Elimination) with LogisticRegression
            if n_features < 100:  # RFE can be slow with many features
                base_models['RFE_LogisticRegression'] = Pipeline([
                    ('feature_selection', RFE(
                        LogisticRegression(random_state=42), 
                        n_features_to_select=min(20, n_features // 2),
                        step=1
                    )),
                    ('classifier', LogisticRegression(
                        random_state=42, class_weight=class_weight, max_iter=2000
                    ))
                ])
        
        # Ensemble models for better performance
        if complexity in ["medium", "high", "very_high"]:
            # Bagging classifier with DecisionTrees
            base_models['BaggedTrees'] = BaggingClassifier(
                estimator=DecisionTreeClassifier(random_state=42),  # Changed from base_estimator to estimator
                n_estimators=100,
                max_samples=0.8,
                max_features=0.8,
                bootstrap=True,
                bootstrap_features=False,
                random_state=42,
                n_jobs=-1
            )
            
            # AdaBoost with DecisionTrees
            base_models['AdaBoost'] = AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3, random_state=42),  # Changed from base_estimator to estimator
                n_estimators=100,
                learning_rate=0.1,
                algorithm='SAMME.R',
                random_state=42
            )
            
            # Voting Classifier for model combination
            estimators = []
            if 'LogisticRegression' in base_models:
                estimators.append(('lr', base_models['LogisticRegression']))
            elif 'LogisticRegression_L2' in base_models:
                estimators.append(('lr', base_models['LogisticRegression_L2']))
                
            if 'RandomForest' in base_models:
                estimators.append(('rf', base_models['RandomForest']))
                
            if 'GradientBoosting' in base_models:
                estimators.append(('gb', base_models['GradientBoosting']))
                
            if 'XGBoost' in base_models:
                estimators.append(('xgb', base_models['XGBoost']))
            
            if len(estimators) >= 3:  # Need at least 3 estimators for a meaningful ensemble
                base_models['VotingClassifier'] = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    weights=[1, 2, 2, 3] if len(estimators) == 4 else [1, 2, 2],
                    n_jobs=-1
                )
            
            # Stacking Classifier for advanced ensemble
            if complexity in ["high", "very_high"] and len(estimators) >= 3:
                # Use different models for base estimators
                base_estimators = []
                if 'LogisticRegression' in base_models:
                    base_estimators.append(('lr', base_models['LogisticRegression']))
                elif 'LogisticRegression_L2' in base_models:
                    base_estimators.append(('lr', base_models['LogisticRegression_L2']))
                    
                if 'RandomForest' in base_models:
                    base_estimators.append(('rf', base_models['RandomForest']))
                    
                if 'GradientBoosting' in base_models:
                    base_estimators.append(('gb', base_models['GradientBoosting']))
                    
                if 'XGBoost' in base_models:
                    base_estimators.append(('xgb', base_models['XGBoost']))
                
                if 'LightGBM' in base_models:
                    base_estimators.append(('lgbm', base_models['LightGBM']))
                
                # Use a strong model as final estimator
                final_estimator = GradientBoostingClassifier(
                    n_estimators=100, 
                    learning_rate=0.05, 
                    random_state=42
                )
                
                base_models['StackingClassifier'] = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=final_estimator,
                    cv=5,
                    stack_method='predict_proba',
                    n_jobs=-1
                )
    
    else:
        # Multi-class models configurations
        base_models = {}
        
        # Logistic Regression with appropriate multi-class settings
        base_models['LogisticRegression'] = LogisticRegression(
            random_state=42,
            multi_class='multinomial' if num_classes <= 5 else 'ovr',
            solver='lbfgs',
            C=1.0,
            class_weight=class_weight,
            max_iter=2000,
            n_jobs=-1
        )
        
        # RandomForest with complexity-appropriate settings
        if complexity == "low":
            base_models['RandomForest'] = RandomForestClassifier(
                random_state=42,
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight=class_weight,
                bootstrap=True
            )
        elif complexity == "medium":
            base_models['RandomForest'] = RandomForestClassifier(
                random_state=42,
                n_estimators=200,
                max_depth=None,
                min_samples_split=3,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight=class_weight,
                bootstrap=True,
                n_jobs=-1
            )
        else:
            base_models['RandomForest'] = RandomForestClassifier(
                random_state=42,
                n_estimators=300,
                max_depth=None,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                class_weight=class_weight,
                bootstrap=True,
                n_jobs=-1
            )
        
        # XGBoost for multi-class
        if complexity == "low":
            base_models['XGBoost'] = XGBClassifier(
                random_state=42,
                eval_metric='mlogloss',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=num_classes
            )
        elif complexity == "medium":
            base_models['XGBoost'] = XGBClassifier(
                random_state=42,
                eval_metric='mlogloss',
                n_estimators=200,
                learning_rate=0.05,
                max_depth=6,
                gamma=0.1,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='multi:softprob',
                num_class=num_classes
            )
        else:
            base_models['XGBoost'] = XGBClassifier(
                random_state=42,
                eval_metric='mlogloss',
                n_estimators=300,
                learning_rate=0.01,
                max_depth=8,
                gamma=0.1,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                colsample_bylevel=0.8,
                objective='multi:softprob',
                num_class=num_classes,
                tree_method='auto',
                n_jobs=-1
            )
        
        # LightGBM for multi-class if not too many classes and dataset isn't too large
        if num_classes <= 10 and n_samples < 100000:
            if complexity == "low":
                base_models['LightGBM'] = LGBMClassifier(
                    random_state=42,
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multiclass',
                    class_weight=class_weight,
                    num_class=num_classes
                )
            elif complexity == "medium":
                base_models['LightGBM'] = LGBMClassifier(
                    random_state=42,
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=6,
                    num_leaves=63,
                    min_child_samples=5,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='multiclass',
                    class_weight=class_weight,
                    num_class=num_classes
                )
            else:
                base_models['LightGBM'] = LGBMClassifier(
                    random_state=42,
                    n_estimators=300,
                    learning_rate=0.01,
                    max_depth=8,
                    num_leaves=127,
                    min_child_samples=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    objective='multiclass',
                    class_weight=class_weight,
                    num_class=num_classes,
                    n_jobs=-1
                )
        
        # Gradient Boosting for multi-class
        if n_samples < 50000:  # GBM can be slow for large datasets
            base_models['GradientBoosting'] = GradientBoostingClassifier(
                random_state=42,
                n_estimators=200,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                min_samples_split=3
            )
            
        # HistGradientBoosting is faster for multi-class on large datasets
        if n_samples >= 10000:
            base_models['HistGradientBoosting'] = HistGradientBoostingClassifier(
                random_state=42,
                max_iter=200,
                learning_rate=0.1,
                max_depth=None,
                min_samples_leaf=20,
                l2_regularization=0.1,
                class_weight=class_weight
            )
        
        # SVM for multi-class if dataset isn't too large
        if n_samples < 10000 and num_classes <= 10:
            base_models['SVM'] = SVC(
                random_state=42,
                probability=True,
                C=1.0,
                kernel='rbf',
                gamma='scale',
                class_weight=class_weight,
                decision_function_shape='ovr'
            )
        
        # Neural Network for multi-class if dataset isn't too large
        if 1000 < n_samples < 50000:
            # Calculate appropriate hidden layer sizes based on number of features and classes
            h1_size = min(100, max(num_classes * 5, n_features))
            h2_size = max(num_classes * 2, h1_size // 2)
            
            base_models['NeuralNetwork'] = MLPClassifier(
                random_state=42,
                hidden_layer_sizes=(h1_size, h2_size),
                activation='relu',
                solver='adam',
                alpha=0.0001,
                batch_size='auto',
                learning_rate='adaptive',
                max_iter=500,
                early_stopping=True
            )
        
        # One-vs-Rest strategy for larger number of classes
        if num_classes > 3:
            base_models['OVR_LogisticRegression'] = OneVsRestClassifier(
                LogisticRegression(
                    random_state=42, 
                    solver='liblinear',
                    C=1.0,
                    class_weight=class_weight,
                    max_iter=2000
                )
            )
            
            # One-vs-One strategy if number of classes is reasonable
            if 3 < num_classes <= 10:
                base_models['OVO_LogisticRegression'] = OneVsOneClassifier(
                    LogisticRegression(
                        random_state=42,
                        solver='liblinear',
                        C=1.0, 
                        class_weight=class_weight,
                        max_iter=2000
                    )
                )
        
        # Ensemble models for multi-class
        if complexity in ["medium", "high", "very_high"] and num_classes <= 10:
            # Voting Classifier for model combination
            estimators = []
            if 'LogisticRegression' in base_models:
                estimators.append(('lr', base_models['LogisticRegression']))
                
            if 'RandomForest' in base_models:
                estimators.append(('rf', base_models['RandomForest']))
                
            if 'XGBoost' in base_models:
                estimators.append(('xgb', base_models['XGBoost']))
            
            if len(estimators) >= 3:  # Need at least 3 estimators for a meaningful ensemble
                base_models['VotingClassifier'] = VotingClassifier(
                    estimators=estimators,
                    voting='soft',
                    n_jobs=-1
                )
            
            # Stacking for multi-class if dataset isn't too large
            if complexity in ["high", "very_high"] and n_samples < 50000 and len(estimators) >= 3:
                # Use different models for base estimators
                base_estimators = []
                if 'LogisticRegression' in base_models:
                    base_estimators.append(('lr', base_models['LogisticRegression']))
                    
                if 'RandomForest' in base_models:
                    base_estimators.append(('rf', base_models['RandomForest']))
                    
                if 'XGBoost' in base_models:
                    base_estimators.append(('xgb', base_models['XGBoost']))
                
                if 'LightGBM' in base_models:
                    base_estimators.append(('lgbm', base_models['LightGBM']))
                
                final_estimator = XGBClassifier(
                    random_state=42,
                    eval_metric='mlogloss',
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=4,
                    objective='multi:softprob',
                    num_class=num_classes
                )
                
                base_models['StackingClassifier'] = StackingClassifier(
                    estimators=base_estimators,
                    final_estimator=final_estimator,
                    cv=5,
                    stack_method='predict_proba',
                    n_jobs=-1
                )
    
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
    primary_score = f1 if is_imbalanced else accuracy
    
    return model, primary_score, f1, accuracy, auc_roc, report


def train_classification_model(target_variable, filepath, user_id, use_case, threshold=0.90, use_f1_for_threshold=False):
    """
    Trains a classification model on the given dataset and calculates feature importance.
    Automatically detects and handles both binary and multi-class classification.
    Stops training if a model achieves primary score above the specified threshold.
    
    Args:
        target_variable (str): The name of the target variable column
        filepath (str): Path to the data file
        user_id (str): User ID for organization
        use_case (str): Use case identifier
        threshold (float): Stop training once a model achieves this score (default: 0.90)
        use_f1_for_threshold (bool): If True, use F1 for threshold; if False, use accuracy; 
                                     if None, determine based on class imbalance
    
    Returns:
        tuple: (model_path, features_path, best_score, feature_importances)
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
    best_score = 0
    best_model = None
    best_feature_importances = {}
    best_report = None
    best_auc_roc = 0
    best_feature_set = None
    best_f1 = 0
    best_accuracy = 0
    
    # Define flag for early stopping
    early_stop = False

    print(f"Starting training process for target: '{target_variable}' on file: '{filepath}'")
    print(f"Model will be saved to: {model_path}")
      
    # Read the data using the flexible reader function
    df = read_data_flexible(filepath)
    
    if df is None:
        print("Error: Failed to read the data. Training aborted.")
        return None, None, 0, {}
    
    # Store original column names
    original_columns = list(df.columns)
    
    # Preprocess the target variable using the function from preprocess_data.py
    y_encoded, label_encoder = preprocess_target_variable(df, target_variable)
    
    # Detect if binary or multi-class classification
    unique_classes = np.unique(y_encoded)
    num_classes = len(unique_classes)
    is_binary = num_classes == 2
    
    if is_binary:
        print("Detected binary classification problem.")
    else:
        print(f"Detected multi-class classification problem with {num_classes} classes.")

    # Check for class imbalance
    class_counts = np.bincount(y_encoded.astype(int))
    class_ratio = class_counts.min() / class_counts.max()
    is_imbalanced = class_ratio < 0.25  # Arbitrary threshold
    
    if is_imbalanced:
        print(f"Warning: Detected class imbalance. Min/Max class ratio: {class_ratio:.3f}")
        print(f"Class distribution: {class_counts}")
    
    # Determine which metric to use for early stopping
    if use_f1_for_threshold is None:
        use_f1_for_threshold = is_imbalanced
    
    metric_name = "F1 score" if use_f1_for_threshold else "Accuracy"
    print(f"Early stopping threshold: {threshold:.2f} (using {metric_name})")
        
    # Prepare features dataframe (without target variable)
    X_df = df.drop(columns=[target_variable])
    
    # Identify numerical and categorical features
    numerical_features = []
    categorical_features = []
    
    for col in X_df.select_dtypes(include=['number']).columns:
        if col in X_df.columns:  # Double-check column exists
            numerical_features.append(col)
    
    for col in X_df.select_dtypes(include=['object', 'category']).columns:
        if col in X_df.columns:  # Double-check column exists
            categorical_features.append(col)
    
    print("Verified numerical features:", numerical_features)
    print("Verified categorical features:", categorical_features)
    
        # IMPORTANT: Verify all columns exist in the dataframe
    all_columns = set(X_df.columns)
    valid_numerical = [col for col in numerical_features if col in all_columns]
    valid_categorical = [col for col in categorical_features if col in all_columns]
    
    if len(valid_numerical) != len(numerical_features) or len(valid_categorical) != len(categorical_features):
        missing_numerical = set(numerical_features) - set(valid_numerical)
        missing_categorical = set(categorical_features) - set(valid_categorical)
        
        if missing_numerical:
            print(f"Warning: Removing non-existent numerical features: {missing_numerical}")
        if missing_categorical:
            print(f"Warning: Removing non-existent categorical features: {missing_categorical}")
            
        numerical_features = valid_numerical
        categorical_features = valid_categorical
    
    print("Final numerical features:", numerical_features)
    print("Final categorical features:", categorical_features)
    
    print("Creating preprocessor directly to avoid column issues...")
    
    # Create preprocessor directly instead of calling the problematic create_preprocessor
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.impute import SimpleImputer
    
    # Ensure all features are actually in the dataframe
    all_columns = set(X_df.columns)
    valid_numerical = [col for col in numerical_features if col in all_columns]
    valid_categorical = [col for col in categorical_features if col in all_columns]
    
    # Create a robust preprocessing pipeline for numerical features
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Create a robust preprocessing pipeline for categorical features
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Create the column transformer with validated features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, valid_numerical),
            ('cat', categorical_transformer, valid_categorical)
        ],
        remainder='drop'
    )
    
    # Create preprocessor with verified features
    # preprocessor = create_preprocessor(X_df, numerical_features, categorical_features)
    
    # Fit and transform using the validated features
    print("Applying preprocessing...")
    preprocessor.fit(X_df)
     # Apply preprocessing directly to X_df (without the target variable)
    print("Preprocessing data...")
    X_processed = preprocessor.transform(X_df)
    
    # Get feature names after preprocessing
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions
        feature_names = (
            [f"num__{col}" for col in numerical_features] + 
            [f"cat__{col}__{cat}" for col in categorical_features 
             for cat in preprocessor.named_transformers_['cat'].categories_[preprocessor.named_transformers_['cat'].feature_names_in_.tolist().index(col)]]
        )
    
    # Convert to DataFrame
    X = pd.DataFrame(X_processed, columns=feature_names)
    print("Shape of preprocessed data:", X.shape)
    
    # Apply quantile transform to get a second version of the data
    print("Creating quantile-transformed version of the data...")
    # Create a copy of X for quantile transformation
    X_quantile = X.copy()
    
    # Find numerical features in processed data
    numerical_cols_processed = [col for col in X.columns if col.startswith('num__')]
    
    if numerical_cols_processed and len(numerical_cols_processed) > 1:
        qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(df), 100))
        X_numerical_transformed = qt.fit_transform(X_quantile[numerical_cols_processed])
        X_numerical_df = pd.DataFrame(X_numerical_transformed, columns=numerical_cols_processed)
        
        # Replace the columns
        X_quantile = X_quantile.drop(columns=numerical_cols_processed)
        X_quantile = pd.concat([X_numerical_df, X_quantile], axis=1)
    
    print("Shape of quantile-transformed data:", X_quantile.shape)

    # Split data into training and testing sets with stratification
    print("Splitting data into training and testing sets...")
    X_train, X_test, X_quantile_train, X_quantile_test, y_train, y_test = train_test_split(
        X, X_quantile, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Define models with appropriate parameters based on classification type
    class_weight = 'balanced' if is_imbalanced else None
    
    n_samples, n_features = X.shape

    # Configure advanced models
    models = configure_advanced_models(
        is_binary=is_binary,
        num_classes=num_classes,
        class_counts=class_counts,
        is_imbalanced=is_imbalanced,
        n_features=n_features,
        n_samples=n_samples
    ) 

    # Train and evaluate models on original features
    print("\nTraining models on original features...")
    for name, model in list(models.items()):
        # Check if we should stop early
        if early_stop:
            print(f"Stopping early: {metric_name} threshold of {threshold:.2f} reached.")
            break
            
        print(f"\nTraining {name}...")
        try:
            # Set environment variable for LOKY if needed
            os.environ.setdefault('LOKY_MAX_CPU_COUNT', '4')
            
            # Train and evaluate
            trained_model, primary_score, f1, acc, auc, report = evaluate_model(
                model, X_train, X_test, y_train, y_test, name, 
                is_binary, is_imbalanced, 'original'
            )
            
            # Save current model metrics
            threshold_metric = f1 if use_f1_for_threshold else acc
            
            # Update best model
            if primary_score > best_score:  # Using primary score for model selection
                best_score = primary_score
                best_model = trained_model
                best_feature_set = "original"
                best_report = report
                best_auc_roc = auc
                best_f1 = f1
                best_accuracy = acc
                print(f"Best model updated to {name} (original) with score: {best_score:.4f}")
                
                # Check if we've reached the threshold for early stopping
                if threshold_metric >= threshold:
                    print(f"{metric_name} threshold met: {threshold_metric:.4f} >= {threshold:.2f}")
                    early_stop = True
                    break
                    
        except Exception as e:
            print(f"Error training {name}: {e}")
            continue

    # Train and evaluate models on quantile-transformed features (if we haven't stopped early)
    if not early_stop:
        print("\nTraining models on quantile-transformed features...")
        for name, model in list(models.items()):
            # Check if we should stop early
            if early_stop:
                print(f"Stopping early: {metric_name} threshold of {threshold:.2f} reached.")
                break
                
            print(f"\nTraining {name} on quantile features...")
            try:
                # Create a fresh instance of the model to avoid parameter conflicts
                if name == 'LogisticRegression':
                    if is_binary:
                        model = LogisticRegression(
                            random_state=42, solver='liblinear', class_weight=class_weight, max_iter=1000
                        )
                    else:
                        model = LogisticRegression(
                            random_state=42, multi_class='auto', solver='lbfgs', 
                            class_weight=class_weight, max_iter=1000
                        )
                elif name == 'RandomForest':
                    model = RandomForestClassifier(
                        random_state=42, class_weight=class_weight, n_estimators=100
                    )
                elif name == 'GradientBoosting':
                    model = GradientBoostingClassifier(random_state=42, n_estimators=100)
                elif name == 'XGBoost':
                    if is_binary:
                        model = XGBClassifier(
                            eval_metric='logloss', random_state=42,
                            scale_pos_weight=class_counts[0]/class_counts[1] if is_imbalanced and class_counts[1] > 0 else 1
                        )
                    else:
                        model = XGBClassifier(
                            eval_metric='mlogloss', random_state=42
                        )
                elif name == 'LightGBM':
                    if is_binary:
                        model = LGBMClassifier(
                            random_state=42, class_weight=class_weight if class_weight else 'balanced'
                        )
                    else:
                        model = LGBMClassifier(
                            random_state=42, objective='multiclass', class_weight=class_weight
                        )
                elif name == 'OVR_LogisticRegression':
                    model = OneVsRestClassifier(
                        LogisticRegression(random_state=42, class_weight=class_weight, max_iter=1000)
                    )
                    
                # Train and evaluate
                trained_model, primary_score, f1, acc, auc, report = evaluate_model(
                    model, X_quantile_train, X_quantile_test, y_train, y_test, name, 
                    is_binary, is_imbalanced, 'quantile'
                )
                
                # Save current model metrics
                threshold_metric = f1 if use_f1_for_threshold else acc
                
                # Update best model
                if primary_score > best_score:
                    best_score = primary_score
                    best_model = trained_model
                    best_feature_set = "quantile"
                    best_report = report
                    best_auc_roc = auc
                    best_f1 = f1
                    best_accuracy = acc
                    print(f"Best model updated to {name} (quantile) with score: {best_score:.4f}")
                    
                    # Check if we've reached the threshold for early stopping
                    if threshold_metric >= threshold:
                        print(f"{metric_name} threshold met: {threshold_metric:.4f} >= {threshold:.2f}")
                        early_stop = True
                        break
                        
            except Exception as e:
                print(f"Error training {name} (quantile): {e}")
                continue

    # Final results
    primary_metric = "F1-Score" if is_imbalanced else "Accuracy"
    print(f"\nBest model: {type(best_model).__name__}, Feature Set: {best_feature_set}")
    print(f"Best {primary_metric}: {best_score:.4f}, F1: {best_f1:.4f}, Accuracy: {best_accuracy:.4f}, AUC-ROC: {best_auc_roc:.4f}")
    print(f"\nBest Model Classification Report:\n{best_report}")

    # Get feature importances from the best model
    if hasattr(best_model, "feature_importances_"):
        feature_importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        if len(best_model.coef_.shape) > 1:
            # For multi-class, take average of absolute values across classes
            feature_importances = np.mean(np.abs(best_model.coef_), axis=0)
        else:
            feature_importances = np.abs(best_model.coef_[0])
    elif hasattr(best_model, "estimators_") and hasattr(best_model.estimators_[0], "feature_importances_"):
        # For ensemble of estimators like in OVR
        feature_importances = np.mean([est.feature_importances_ for est in best_model.estimators_], axis=0)
    else:
        feature_importances = None
        print("Best model does not support feature importance.")
    
    # Store feature importances
    if feature_importances is not None:
        best_features = X_train.columns if best_feature_set == "original" else X_quantile_train.columns
        best_feature_importances = dict(zip(best_features, feature_importances))
    
    # Map feature importance names back to original column names
    original_feature_importances = {}
    if best_feature_importances:
        for processed_feature, importance in best_feature_importances.items():
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
    
    # Save the best model
    print(f"Saving best model to: {model_path}")
    joblib.dump(best_model, model_path)
    
    # Save preprocessor
    preprocessor_path = save_preprocessor(preprocessor, user_id, use_case, target_variable, filename, output_dir)
    print(f"Saved preprocessor to: {preprocessor_path}")

    # Save feature names
    features_filename = os.path.join(output_dir, f"model_features_{model_filename}")
    print(f"Saving feature names to: {features_filename}")
    joblib.dump(feature_names, features_filename)
    
    # Save label encoder
    if label_encoder:
        encoder_path = save_label_encoder(label_encoder, user_id, use_case, target_variable, filename, output_dir)
        print(f"Saved label encoder to: {encoder_path}")
        
        # Save class mapping for reference if the label encoder was used
        if hasattr(label_encoder, 'classes_') and len(label_encoder.classes_) > 0:
            class_mapping = {i: class_name for i, class_name in enumerate(label_encoder.classes_)}
            mapping_path = os.path.join(output_dir, f"class_mapping_{filename}.csv")
            pd.DataFrame({
                'Class_ID': list(class_mapping.keys()),
                'Class_Name': list(class_mapping.values())
            }).to_csv(mapping_path, index=False)
            print(f"Saved class mapping to: {mapping_path}")

    # If early stop was triggered, note that in the output
    if early_stop:
        print(f"Note: Training stopped early because {metric_name} threshold of {threshold:.2f} was reached.")

    return model_path, features_filename, best_score, original_feature_importances

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