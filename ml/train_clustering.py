"""
filepath = '../uploads/bank-full.csv'
MODEL_SAVE_DIR = 'trained_models'
model_filename, features_filename, kmeans = train_clustering_model(filepath, n_clusters=None)

input_data = read_data_flexible(filepath)
model_path = 'trained_models\\kmeans_model_joze_usecase1_2_clusters.joblib'
predict_clustering(model_path, input_data)
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from ml.read_file import read_data_flexible
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.impute import SimpleImputer  # Import SimpleImputer for handling missing values
from sklearn.pipeline import Pipeline     # Import Pipeline to create preprocessing steps

USER = 'joze'
USE_CASE = "usecase1"


def find_optimal_clusters(X, max_clusters=20, methods=('elbow', 'silhouette', 'davies_bouldin')):
    """
    Finds the optimal number of clusters using multiple methods.

    Args:
        X (pd.DataFrame): The preprocessed data.
        max_clusters (int, optional): The maximum number of clusters to try. Defaults to 20.
        methods (tuple, optional): Tuple of methods to use.
                                  Options: 'elbow', 'silhouette', 'davies_bouldin'.
                                  Defaults to ('elbow', 'silhouette', 'davies_bouldin').

    Returns:
        dict: A dictionary containing the optimal number of clusters for each method.
    """

    optimal_clusters = {}

    if 'elbow' in methods:
        optimal_clusters['elbow'] = find_optimal_clusters_elbow(X, max_clusters)
    if 'silhouette' in methods:
        optimal_clusters['silhouette'] = find_optimal_clusters_silhouette(X, max_clusters)
    if 'davies_bouldin' in methods:
        optimal_clusters['davies_bouldin'] = find_optimal_clusters_davies_bouldin(X, max_clusters)

    return optimal_clusters


def find_optimal_clusters_elbow(X, max_clusters=20, elbow_method='acceleration'):
    """
    Finds the optimal number of clusters using the Elbow Method.

    Args:
        X (pd.DataFrame): The preprocessed data.
        max_clusters (int, optional): The maximum number of clusters to try. Defaults to 20.
        elbow_method (str, optional): The method to determine the elbow.
                                     Options: 'acceleration', 'curvature'.
                                     Defaults to 'acceleration'.

    Returns:
        int: The optimal number of clusters determined by the elbow method.
    """

    print(f"Finding optimal number of clusters using Elbow Method ({elbow_method})...")
    distortions = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X)
        distortions.append(kmeans.inertia_)  # Inertia: Sum of squared distances to closest centroid

  
    if elbow_method == 'acceleration':
        # Calculate the "acceleration" (second derivative)
        acceleration = np.diff(distortions, n=2)
        if len(acceleration) > 0:
            optimal_clusters = np.argmax(acceleration) + 2  # +2 because of the two diffs
            optimal_clusters = min(optimal_clusters, max_clusters)  # Ensure it's within the range
        else:
            optimal_clusters = 2  # Default if no clear elbow
    elif elbow_method == 'curvature':
        # Calculate the curvature
        x1 = np.array(range(1, len(distortions) - 1))
        x2 = x1 + 1
        x3 = x1 + 2

        y1 = np.array(distortions[:-2])
        y2 = np.array(distortions[1:-1])
        y3 = np.array(distortions[2:])

        curvature = np.abs((y1 - 2 * y2 + y3) / np.sqrt(((x1 - x3) ** 2) + ((y1 - y3) ** 2)))
        if len(curvature) > 0:
            optimal_clusters = np.argmax(curvature) + 2
            optimal_clusters = min(optimal_clusters, max_clusters)
        else:
            optimal_clusters = 2
    else:
        raise ValueError(f"Invalid elbow_method: {elbow_method}.  Choose 'acceleration' or 'curvature'.")

    print(f"Optimal number of clusters (Elbow Method - {elbow_method}): {optimal_clusters}")
    return optimal_clusters


def find_optimal_clusters_silhouette(X, max_clusters=20):
    """
    Finds the optimal number of clusters using the Silhouette Method.

    Args:
        X (pd.DataFrame): The preprocessed data.
        max_clusters (int, optional): The maximum number of clusters to try. Defaults to 20.

    Returns:
        int: The optimal number of clusters determined by the Silhouette Method.
    """

    print("Finding optimal number of clusters using Silhouette Method...")
    silhouette_scores = []
    for i in range(2, max_clusters + 1):  # Silhouette score requires at least 2 clusters
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X)
        silhouette_avg = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because we started from 2
    print(f"Optimal number of clusters (Silhouette Method): {optimal_clusters}")
    return optimal_clusters


def find_optimal_clusters_davies_bouldin(X, max_clusters=20):
    """
    Finds the optimal number of clusters using the Davies-Bouldin Index.

    Args:
        X (pd.DataFrame): The preprocessed data.
        max_clusters (int, optional): The maximum number of clusters to try. Defaults to 20.

    Returns:
        int: The optimal number of clusters determined by the Davies-Bouldin Index.
    """

    print("Finding optimal number of clusters using Davies-Bouldin Index...")
    db_scores = []
    for i in range(2, max_clusters + 1):  # Davies-Bouldin Index requires at least 2 clusters
        kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
        kmeans.fit(X)
        db_avg = davies_bouldin_score(X, kmeans.labels_)
        db_scores.append(db_avg)

    optimal_clusters = np.argmin(db_scores) + 2  # Optimal is the minimum score
    print(f"Optimal number of clusters (Davies-Bouldin Index): {optimal_clusters}")
    return optimal_clusters


def train_clustering_model(
    filepath,
    n_clusters=None,
    max_clusters_optimal=20,
    optimal_method='elbow',
    methods_optimal=('elbow', 'silhouette', 'davies_bouldin'),
):
    """
    Trains a KMeans clustering model on the given dataset, optionally finding the
    optimal number of clusters using the specified method(s).

    Args:
        filepath (str): The path to the data file.
        n_clusters (int, optional): The number of clusters to form. If None, the
                                  optimal method is used to find the optimal number.
        max_clusters_optimal (int, optional): Maximum clusters to try for optimal method.
                                            Defaults to 20.
        optimal_method (str, optional): The method to determine the optimal number of clusters
                                      if n_clusters is None. Options: 'elbow', 'silhouette',
                                      'davies_bouldin', or 'all'. Defaults to 'elbow'.
        methods_optimal (tuple, optional): Tuple of methods to use when optimal_method is 'all'.
                                          Defaults to ('elbow', 'silhouette', 'davies_bouldin').

    Returns:
        tuple: A tuple containing the model filename, feature names,
               and the trained KMeans model.
    """

    output_dir = "trained_models"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting clustering training process on file: '{filepath}'")

    # Read the data using the flexible reader function
    df = read_data_flexible(filepath)

    if df is None:
        print("Error: Failed to read the data. Training aborted.")
        return None, None, None

    # Store original column names
    original_columns = list(df.columns)

    # Identify numerical and categorical features
    numerical_features = df.select_dtypes(include=['number']).columns
    categorical_features = df.select_dtypes(include=['object', 'category']).columns

    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)

    # Create a ColumnTransformer for preprocessing
    # Create preprocessing pipelines with imputation for both numerical and categorical features
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),  # Replace NaN with mean for numerical features
        ('scaler', StandardScaler())  # Scale numerical features
    ])
    
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),  # Replace NaN with most frequent value for categorical
        ('encoder', OneHotEncoder(handle_unknown='ignore'))  # One-hot encode categorical features
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_features),  
            ('cat', cat_pipeline, categorical_features),
        ],
        remainder='passthrough',  # Keep other columns (if any)
    )

    print("Preprocessing data...")
    X = preprocessor.fit_transform(df)  # Apply preprocessing

    # Convert the preprocessed data back to a DataFrame
    feature_names_after_preprocessing = preprocessor.get_feature_names_out()
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=feature_names_after_preprocessing)
    else:
        X = pd.DataFrame(X.toarray(), columns=feature_names_after_preprocessing)

    print("Shape of preprocessed data:", X.shape)

    # Determine the number of clusters
    if n_clusters is None:
        if optimal_method == 'all':
            optimal_clusters = find_optimal_clusters(X, max_clusters=max_clusters_optimal, methods=methods_optimal)
            # You might want to implement a strategy to choose a single value from the dict
            # For simplicity, let's take the 'elbow' value if available, else the first value.
            n_clusters = optimal_clusters.get('elbow') or list(optimal_clusters.values())[0] if optimal_clusters else 3
        elif optimal_method in ('elbow', 'silhouette', 'davies_bouldin'):
            find_optimal_func = globals()[f'find_optimal_clusters_{optimal_method}']
            n_clusters = find_optimal_func(X, max_clusters=max_clusters_optimal)
        else:
            raise ValueError(
                f"Invalid optimal_method: {optimal_method}. Choose 'elbow', 'silhouette', 'davies_bouldin', or 'all'.")
        print(f"Optimal number of clusters found: {n_clusters}")
    else:
        print(f"Using provided number of clusters: {n_clusters}")

    # Train the KMeans clustering model
    print(f"Training KMeans model with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X)

    # Save the trained model, preprocessor, and feature names
    model_filename = os.path.join(output_dir,
                                 f"kmeans_model_{USER}_{USE_CASE}_{n_clusters}_clusters.joblib")  # Include USER and USE_CASE
    print(f"Saving KMeans model to: {model_filename}")
    joblib.dump(kmeans, model_filename)

    preprocessor_filename = os.path.join(output_dir,
                                        f"preprocessor_{USER}_{USE_CASE}_kmeans.joblib")  # Include USER and USE_CASE
    print(f"Saving preprocessor to: {preprocessor_filename}")
    joblib.dump(preprocessor, preprocessor_filename)

    features_filename = os.path.join(output_dir,
                                     f"model_features_{USER}_{USE_CASE}_kmeans.joblib")  # Include USER and USE_CASE
    print(f"Saving feature names to: {features_filename}")
    joblib.dump(feature_names_after_preprocessing, features_filename)

    return model_filename, features_filename, kmeans