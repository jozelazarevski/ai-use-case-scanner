#!/usr/bin/env python3
"""
Comprehensive AutoML Script for Supervised Learning (Classification/Regression) and Unsupervised Learning (Clustering)

This script:
1. Loads and preprocesses data
2. Performs feature engineering
3. Uses AutoML for classification/regression tasks
4. Uses AutoML for clustering tasks
5. Evaluates and visualizes results
6. Saves trained models

Dependencies:
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn
- optuna (for hyperparameter optimization)
- auto-sklearn (for AutoML classification/regression)
- pycaret (for simplified AutoML workflows)
"""

import os
import time
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Import AutoML libraries
try:
    import autosklearn.classification
    import autosklearn.regression
    HAS_AUTOSKLEARN = True
except ImportError:
    HAS_AUTOSKLEARN = False
    print("Warning: auto-sklearn not installed. Will use fallback methods.")

try:
    from pycaret.classification import setup as clf_setup, compare_models as clf_compare, create_model as clf_create_model, tune_model as clf_tune_model, save_model as clf_save_model
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, create_model as reg_create_model, tune_model as reg_tune_model, save_model as reg_save_model
    from pycaret.clustering import setup as clust_setup, create_model as clust_create_model, tune_model as clust_tune_model, save_model as clust_save_model
    HAS_PYCARET = True
except ImportError:
    HAS_PYCARET = False
    print("Warning: PyCaret not installed. Will use fallback methods.")

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("Warning: Optuna not installed. Will use default hyperparameters.")


class AutoMLTrainer:
    """
    A class to handle AutoML training for both supervised and unsupervised tasks.
    """
    def __init__(self, 
                 data_path=None, 
                 target_column=None,
                 task_type="auto",  # can be "auto", "classification", "regression", "clustering"
                 test_size=0.2,
                 random_state=42,
                 time_budget=120,  # in seconds
                 output_dir="models"):
        """
        Initialize the AutoML trainer.
        
        Args:
            data_path (str): Path to the data file (CSV, Excel, etc.)
            target_column (str): Name of the target column for supervised learning
            task_type (str): Type of task (auto, classification, regression, clustering)
            test_size (float): Fraction of data to use for testing
            random_state (int): Random seed for reproducibility
            time_budget (int): Time budget in seconds for AutoML training
            output_dir (str): Directory to save models and results
        """
        self.data_path = data_path
        self.target_column = target_column
        self.task_type = task_type
        self.test_size = test_size
        self.random_state = random_state
        self.time_budget = time_budget
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize instance variables
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.categorical_features = []
        self.numerical_features = []
        self.model = None
        self.task_detected = None
        self.feature_importances = None
        self.best_model = None
        
        # Settings
        warnings.filterwarnings('ignore')
        sns.set_style('whitegrid')
        
    def load_data(self, data=None):
        """
        Load data from file or directly from a DataFrame.
        
        Args:
            data (pd.DataFrame, optional): Data provided directly as DataFrame
                If None, loads from self.data_path
                
        Returns:
            pd.DataFrame: Loaded data
        """
        if data is not None:
            self.data = data
        elif self.data_path is not None:
            if self.data_path.endswith('.csv'):
                self.data = pd.read_csv(self.data_path)
            elif self.data_path.endswith(('.xls', '.xlsx')):
                self.data = pd.read_excel(self.data_path)
            elif self.data_path.endswith('.json'):
                self.data = pd.read_json(self.data_path)
            elif self.data_path.endswith('.parquet'):
                self.data = pd.read_parquet(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path}")
        else:
            raise ValueError("No data provided. Please provide data or a valid data path.")
        
        print(f"Data loaded with shape: {self.data.shape}")
        return self.data
    
    def detect_task_type(self):
        """
        Auto-detect the task type based on the target variable.
        
        Returns:
            str: Detected task type ("classification", "regression", "clustering")
        """
        if self.task_type != "auto":
            self.task_detected = self.task_type
            return self.task_type
        
        if self.target_column is None:
            print("No target column specified, assuming clustering task.")
            self.task_detected = "clustering"
            return "clustering"
        
        # Check if target column exists in the data
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in data.")
        
        # Check the number of unique values in the target
        n_unique = self.data[self.target_column].nunique()
        n_samples = len(self.data)
        
        if n_unique == 2:
            print(f"Detected binary classification task (target has 2 unique values).")
            self.task_detected = "classification"
        elif n_unique > 2 and n_unique <= 20:
            print(f"Detected multiclass classification task (target has {n_unique} unique values).")
            self.task_detected = "classification"
        elif n_unique > 20 and n_unique < n_samples * 0.5:
            print(f"Detected regression task (target has {n_unique} unique values).")
            self.task_detected = "regression"
        else:
            print(f"Detected regression task based on number of unique values ({n_unique}).")
            self.task_detected = "regression"
        
        return self.task_detected

    def analyze_data(self):
        """
        Perform initial data analysis and print summary statistics.
        """
        print("\n=== DATA ANALYSIS ===")
        print(f"Dataset shape: {self.data.shape}")
        
        # Display basic info and summary statistics
        print("\nData types:")
        print(self.data.dtypes.value_counts())
        
        print("\nMissing values:")
        missing = self.data.isnull().sum()
        print(missing[missing > 0])
        
        # Identify numeric and categorical columns
        self.numerical_features = list(self.data.select_dtypes(include=['int64', 'float64']).columns)
        self.categorical_features = list(self.data.select_dtypes(include=['object', 'category', 'bool']).columns)
        
        if self.target_column in self.numerical_features:
            self.numerical_features.remove(self.target_column)
        if self.target_column in self.categorical_features:
            self.categorical_features.remove(self.target_column)
        
        print(f"\nNumerical features ({len(self.numerical_features)}): {self.numerical_features}")
        print(f"Categorical features ({len(self.categorical_features)}): {self.categorical_features}")
        
        if self.task_detected in ["classification", "regression"]:
            print(f"\nTarget column: {self.target_column}")
            print(f"Target distribution:")
            if self.task_detected == "classification":
                print(self.data[self.target_column].value_counts())
            else:
                print(self.data[self.target_column].describe())
        
    def preprocess_data(self):
        """
        Preprocess the data, including:
        - Handling missing values
        - Encoding categorical variables
        - Scaling numerical features
        - Creating train/test split for supervised learning
        
        Returns:
            tuple: Processed data ready for modeling
        """
        print("\n=== DATA PREPROCESSING ===")
        
        if self.task_detected in ["classification", "regression"]:
            # Split into features and target
            self.y = self.data[self.target_column]
            self.X = self.data.drop(columns=[self.target_column])
            
            # Train-test split
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X, self.y, test_size=self.test_size, random_state=self.random_state
            )
            
            print(f"Training set: {self.X_train.shape}, Test set: {self.X_test.shape}")
            return self.X_train, self.X_test, self.y_train, self.y_test
        else:
            # For clustering, use all data
            self.X = self.data
            print(f"Clustering data shape: {self.X.shape}")
            return self.X, None, None, None
    
    def build_preprocessing_pipeline(self):
        """
        Build a scikit-learn preprocessing pipeline.
        
        Returns:
            sklearn.pipeline.Pipeline: Preprocessing pipeline
        """
        # Numeric preprocessor
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical preprocessor
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', pd.get_dummies)
        ])
        
        # Combine preprocessors
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numerical_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])
        
        return preprocessor
    
    def train_classification(self):
        """
        Train a classification model using AutoML.
        
        Returns:
            object: Trained classification model
        """
        print("\n=== CLASSIFICATION TRAINING ===")
        start_time = time.time()
        
        if HAS_PYCARET:
            print("Using PyCaret for AutoML classification...")
            # Setup PyCaret environment
            clf_setup(data=self.data, target=self.target_column, 
                      train_size=(1-self.test_size), session_id=self.random_state)
            
            # Compare models
            print("Comparing different models...")
            self.best_model = clf_compare(sort='Accuracy', n_select=3)
            
            # Tune the best model
            print(f"Tuning the best model...")
            self.model = clf_tune_model(self.best_model)
            
            # Save model
            clf_save_model(self.model, os.path.join(self.output_dir, 'classification_model'))
        
        elif HAS_AUTOSKLEARN:
            print("Using auto-sklearn for AutoML classification...")
            
            self.model = autosklearn.classification.AutoSklearnClassifier(
                time_left_for_this_task=self.time_budget,
                per_run_time_limit=int(self.time_budget/5),
                ensemble_size=1,
                memory_limit=None,
                seed=self.random_state
            )
            
            self.model.fit(self.X_train, self.y_train)
            print(self.model.sprint_statistics())
            
            # Get feature importances if possible
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        else:
            print("Using scikit-learn RandomForestClassifier as fallback...")
            
            # Use RandomForest as fallback with preprocessing
            preprocessor = self.build_preprocessing_pipeline()
            
            self.model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', RandomForestClassifier(n_estimators=100, random_state=self.random_state))
            ])
            
            self.model.fit(self.X_train, self.y_train)
            
            # Extract feature importances
            feature_names = (
                self.numerical_features + 
                list(pd.get_dummies(self.X_train[self.categorical_features]).columns)
            )
            self.feature_importances = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.named_steps['classifier'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        
        return self.model
    
    def train_regression(self):
        """
        Train a regression model using AutoML.
        
        Returns:
            object: Trained regression model
        """
        print("\n=== REGRESSION TRAINING ===")
        start_time = time.time()
        
        if HAS_PYCARET:
            print("Using PyCaret for AutoML regression...")
            # Setup PyCaret environment
            reg_setup(data=self.data, target=self.target_column, 
                      train_size=(1-self.test_size), session_id=self.random_state)
            
            # Compare models
            print("Comparing different models...")
            self.best_model = reg_compare(sort='MAE', n_select=3)
            
            # Tune the best model
            print(f"Tuning the best model...")
            self.model = reg_tune_model(self.best_model)
            
            # Save model
            reg_save_model(self.model, os.path.join(self.output_dir, 'regression_model'))
        
        elif HAS_AUTOSKLEARN:
            print("Using auto-sklearn for AutoML regression...")
            
            self.model = autosklearn.regression.AutoSklearnRegressor(
                time_left_for_this_task=self.time_budget,
                per_run_time_limit=int(self.time_budget/5),
                ensemble_size=1,
                memory_limit=None,
                seed=self.random_state
            )
            
            self.model.fit(self.X_train, self.y_train)
            print(self.model.sprint_statistics())
            
            # Get feature importances if possible
            if hasattr(self.model, 'feature_importances_'):
                self.feature_importances = pd.DataFrame({
                    'feature': self.X_train.columns,
                    'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        else:
            print("Using scikit-learn RandomForestRegressor as fallback...")
            
            # Use RandomForest as fallback with preprocessing
            preprocessor = self.build_preprocessing_pipeline()
            
            self.model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('regressor', RandomForestRegressor(n_estimators=100, random_state=self.random_state))
            ])
            
            self.model.fit(self.X_train, self.y_train)
            
            # Extract feature importances
            feature_names = (
                self.numerical_features + 
                list(pd.get_dummies(self.X_train[self.categorical_features]).columns)
            )
            self.feature_importances = pd.DataFrame({
                'feature': feature_names,
                'importance': self.model.named_steps['regressor'].feature_importances_
            }).sort_values('importance', ascending=False)
        
        elapsed_time = time.time() - start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        
        return self.model
    
    def train_clustering(self):
        """
        Train a clustering model using AutoML.
        
        Returns:
            object: Trained clustering model
        """
        print("\n=== CLUSTERING TRAINING ===")
        start_time = time.time()
        
        if HAS_PYCARET:
            print("Using PyCaret for AutoML clustering...")
            # Setup PyCaret environment
            clust_setup(data=self.data, session_id=self.random_state)
            
            # Create and evaluate multiple clustering models
            print("Evaluating different clustering algorithms...")
            
            # Try different clustering algorithms with PyCaret
            clustering_models = ['kmeans', 'hclust', 'dbscan', 'optics']
            best_score = -np.inf
            
            for model_name in clustering_models:
                print(f"Evaluating {model_name}...")
                # Create model with PyCaret
                model = clust_create_model(model_name)
                
                # Try to get silhouette score
                try:
                    score = silhouette_score(
                        self.X, 
                        model.labels_ if hasattr(model, 'labels_') else model.predict(self.X)
                    )
                    print(f"{model_name} silhouette score: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        self.model = model
                        self.best_model_name = model_name
                except:
                    print(f"Couldn't evaluate {model_name}")
            
            # Tune the best model if one was found
            if self.model is not None:
                print(f"Best model: {self.best_model_name} with silhouette score: {best_score:.4f}")
                print(f"Tuning the best model...")
                self.model = clust_tune_model(self.model)
                
                # Save model
                clust_save_model(self.model, os.path.join(self.output_dir, 'clustering_model'))
        
        else:
            print("Using scikit-learn KMeans clustering as fallback...")
            from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
            
            # Preprocess data
            preprocessor = self.build_preprocessing_pipeline()
            X_processed = preprocessor.fit_transform(self.X)
            
            # Try different numbers of clusters
            best_k = 2
            best_score = -np.inf
            
            # Try KMeans with different cluster numbers
            for k in range(2, 11):
                model = KMeans(n_clusters=k, random_state=self.random_state)
                model.fit(X_processed)
                
                # Evaluate using silhouette score
                try:
                    score = silhouette_score(X_processed, model.labels_)
                    print(f"KMeans with {k} clusters: silhouette score = {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_k = k
                except:
                    continue
            
            # Train final model with best k
            print(f"Training final KMeans model with {best_k} clusters (silhouette = {best_score:.4f})")
            
            # Create final pipeline
            self.model = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('clusterer', KMeans(n_clusters=best_k, random_state=self.random_state))
            ])
            
            self.model.fit(self.X)
        
        elapsed_time = time.time() - start_time
        print(f"Clustering completed in {elapsed_time:.2f} seconds")
        
        return self.model
    
    def evaluate_classification(self):
        """
        Evaluate the trained classification model.
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print("\n=== CLASSIFICATION EVALUATION ===")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, average='weighted'),
            'recall': recall_score(self.y_test, y_pred, average='weighted'),
            'f1': f1_score(self.y_test, y_pred, average='weighted')
        }
        
        # Print metrics
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.title()}: {metric_value:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = pd.crosstab(self.y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        
        # Plot feature importance if available
        if self.feature_importances is not None:
            plt.figure(figsize=(10, 8))
            top_n = min(20, len(self.feature_importances))
            sns.barplot(x='importance', y='feature', data=self.feature_importances.head(top_n))
            plt.title(f'Top {top_n} Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        
        return metrics
    
    def evaluate_regression(self):
        """
        Evaluate the trained regression model.
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print("\n=== REGRESSION EVALUATION ===")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'mae': mean_absolute_error(self.y_test, y_pred),
            'mse': mean_squared_error(self.y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, y_pred)),
            'r2': r2_score(self.y_test, y_pred)
        }
        
        # Print metrics
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name.upper()}: {metric_value:.4f}")
        
        # Plot actual vs predicted
        plt.figure(figsize=(8, 8))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], [self.y_test.min(), self.y_test.max()], 'r--')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs Predicted Values')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'actual_vs_predicted.png'))
        
        # Plot residuals
        residuals = self.y_test - y_pred
        plt.figure(figsize=(8, 6))
        sns.histplot(residuals, kde=True)
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title('Residual Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'residual_distribution.png'))
        
        # Plot feature importance if available
        if self.feature_importances is not None:
            plt.figure(figsize=(10, 8))
            top_n = min(20, len(self.feature_importances))
            sns.barplot(x='importance', y='feature', data=self.feature_importances.head(top_n))
            plt.title(f'Top {top_n} Feature Importances')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'feature_importance.png'))
        
        return metrics
    
    def evaluate_clustering(self):
        """
        Evaluate the trained clustering model.
        
        Returns:
            dict: Dictionary of evaluation metrics
        """
        print("\n=== CLUSTERING EVALUATION ===")
        
        # Get cluster labels
        if hasattr(self.model, 'labels_'):
            labels = self.model.labels_
        elif hasattr(self.model, 'predict'):
            labels = self.model.predict(self.X)
        elif hasattr(self.model, 'named_steps') and hasattr(self.model.named_steps['clusterer'], 'predict'):
            # For pipeline models
            labels = self.model.named_steps['clusterer'].predict(self.model.named_steps['preprocessor'].transform(self.X))
        else:
            print("Could not extract cluster labels from model")
            return {}
        
        # Get processed data for evaluation
        if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
            X_processed = self.model.named_steps['preprocessor'].transform(self.X)
        else:
            # Simple standardization as fallback
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(self.X.select_dtypes(include=['int64', 'float64']))
        
        # Calculate metrics
        metrics = {}
        
        try:
            metrics['silhouette'] = silhouette_score(X_processed, labels)
            print(f"Silhouette Score: {metrics['silhouette']:.4f}")
        except:
            print("Could not calculate Silhouette score")
        
        try:
            metrics['davies_bouldin'] = davies_bouldin_score(X_processed, labels)
            print(f"Davies-Bouldin Index: {metrics['davies_bouldin']:.4f}")
        except:
            print("Could not calculate Davies-Bouldin Index")
        
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(X_processed, labels)
            print(f"Calinski-Harabasz Score: {metrics['calinski_harabasz']:.4f}")
        except:
            print("Could not calculate Calinski-Harabasz Score")
        
        # Plot cluster distribution
        plt.figure(figsize=(8, 6))
        cluster_counts = pd.Series(labels).value_counts().sort_index()
        sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
        plt.xlabel('Cluster')
        plt.ylabel('Number of Samples')
        plt.title('Cluster Size Distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'cluster_distribution.png'))
        
        # Try to visualize clusters with PCA
        try:
            from sklearn.decomposition import PCA
            
            # Apply PCA to reduce dimensions to 2
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_processed)
            
            # Create DataFrame for plotting
            pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
            pca_df['Cluster'] = labels
            
            # Plot clusters
            plt.figure(figsize=(10, 8))
            sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=pca_df, palette='viridis')
            plt.title('Cluster Visualization using PCA')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'cluster_pca.png'))
            
            print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2f}")
        except Exception as e:
            print(f"Could not create PCA visualization: {str(e)}")
        
        return metrics
    
    def save_model(self):
        """
        Save the trained model and important artifacts.
        """
        print("\n=== SAVING MODEL AND ARTIFACTS ===")
        
        # If not already saved by PyCaret
        if not HAS_PYCARET:
            import pickle
            
            # Save model
            model_path = os.path.join(self.output_dir, f"{self.task_detected}_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {model_path}")
            
            # Save feature importances if available
            if self.feature_importances is not None:
                fi_path = os.path.join(self.output_dir, "feature_importances.csv")
                self.feature_importances.to_csv(fi_path, index=False)
                print(f"Feature importances saved to {fi_path}")
        
        # Save model summary
        summary = {
            'task_type': self.task_detected,
            'data_shape': self.data.shape,
            'features': list(self.X.columns) if hasattr(self.X, 'columns') else [],
            'target': self.target_column,
            'model_type': str(type(self.model)),
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(self.output_dir, "model_summary.txt"), "w") as f:
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
        
        print("Model summary saved.")
    
    def run(self):
        """
        Run the full AutoML pipeline.
        """
        print("\n===== STARTING AUTOML PIPELINE =====")
        
        # Load data
        self.load_data()
        
        # Detect task type
        self.detect_task_type()
        
        # Analyze data
        self.analyze_data()
        
        # Preprocess data
        self.preprocess_data()
        
        # Train model based on task type
        if self.task_detected == "classification":
            self.train_classification()
            self.evaluate_classification()
        elif self.task_detected == "regression":
            self.train_regression()
            self.evaluate_regression()
        elif self.task_detected == "clustering":
            self.train_clustering()
            self.evaluate_clustering()
        else:
            raise ValueError(f"Unsupported task type: {self.task_detected}")
        
        # Save model and artifacts
        self.save_model()
        
        print("\n===== AUTOML PIPELINE COMPLETED =====")
        
        return self.model


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="AutoML for classification, regression, and clustering")
    
    parser.add_argument("--data", type=str, help="Path to the data file (CSV, Excel, etc.)")
    parser.add_argument("--target", type=str, help="Name of the target column for supervised learning")
    parser.add_argument("--task", type=str, default="auto", 
                        choices=["auto", "classification", "regression", "clustering"],
                        help="Type of ML task to perform")
    parser.add_argument("--test-size", type=float, default=0.2,
                        help="Fraction of data to use for testing")
    parser.add_argument("--time-budget", type=int, default=120,
                        help="Time budget in seconds for AutoML training")
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save models and results")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    
    # Create and run AutoML trainer
    trainer = AutoMLTrainer(
        data_path=args.data,
        target_column=args.target,
        task_type=args.task,
        test_size=args.test_size,
        random_state=args.random_state,
        time_budget=args.time_budget,
        output_dir=args.output_dir
    )
    
    # Run the pipeline
    model = trainer.run()
