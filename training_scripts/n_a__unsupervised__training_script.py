import pandas as pd
import numpy as np
import os
import json
import joblib
from sklearn.model_selection import train_test_split # Although not strictly needed for unsupervised, kept for potential future uses/consistency
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning) # DBSCAN might warn if no core samples found

# --- Configuration ---
# 1. Define File Paths and Parameters (as specified in the prompt)
DATA_FILE_PATH = "../uploads/bank-full.csv"
MODEL_DIR = "trained_models"
# Using a fixed name derived from the prompt example for consistency
METRICS_FILE = os.path.join(MODEL_DIR, f"joze_clustering_bank-fullcsv_metrics.json")
# Using a generic name for the best model, type specified by variable
MODEL_FILE = os.path.join(MODEL_DIR, "best_clustering_model.joblib")
# Report file might not be directly applicable for clustering like classification, but kept for structure
REPORT_FILE = os.path.join(MODEL_DIR, "clustering_evaluation_report.txt")
# Output file for prediction results (example)
OUTPUT_DIR = "results" # Assuming an output directory for predictions
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"joze_clustering_bank-fullcsv_results.json")

# 2. Define Features and Target
# Features based on the use case description
FEATURE_COLUMNS = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan', 'default', 'poutcome', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous'] # Expanded features based on common bank datasets
TARGET_COLUMN = None # Unsupervised learning

# Separate features by type for preprocessing
NUMERICAL_FEATURES = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Ensure all specified FEATURE_COLUMNS are covered
assert set(FEATURE_COLUMNS) == set(NUMERICAL_FEATURES) | set(CATEGORICAL_FEATURES), "Mismatch between FEATURE_COLUMNS and typed features"

# --- Core Functions ---

def load_and_prepare_data(file_path):
    """Loads data, handles encoding errors, and selects features."""
    try:
        df = pd.read_csv(file_path, sep=';', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, sep=';', encoding='latin-1')
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            raise
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        raise

    # Select only the features needed for clustering
    # Handle potential missing columns gracefully if needed, though erroring out is safer here
    missing_cols = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The following required columns are missing from the dataset: {missing_cols}")

    df_features = df[FEATURE_COLUMNS].copy() # Use copy to avoid SettingWithCopyWarning

    # Basic check for missing values (optional: add imputation if necessary)
    if df_features.isnull().sum().sum() > 0:
        print("Warning: Missing values detected. Consider imputation.")
        # Simple imputation example (replace with more sophisticated methods if needed)
        for col in NUMERICAL_FEATURES:
            if df_features[col].isnull().any():
                df_features[col].fillna(df_features[col].median(), inplace=True)
        for col in CATEGORICAL_FEATURES:
             if df_features[col].isnull().any():
                df_features[col].fillna(df_features[col].mode()[0], inplace=True) # Fill with mode

    return df_features

def create_preprocessor():
    """Creates the preprocessing pipeline for numerical and categorical features."""
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Use sparse=False if models downstream don't handle sparse well easily

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='passthrough' # Keep other columns if any, though we selected features
    )
    return preprocessor

def train_clustering_model():
    """
    Loads data, preprocesses it, trains multiple clustering models,
    evaluates them, selects the best one, saves it, and returns metrics.
    """
    print("Starting clustering model training...")

    # Ensure output directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True) # For prediction output example

    # 1. Load and Prepare Data
    try:
        features_df = load_and_prepare_data(DATA_FILE_PATH)
    except Exception as e:
        print(f"Failed to load or prepare data: {e}")
        return None, None # Indicate failure

    # 2. Create Preprocessor
    preprocessor = create_preprocessor()

    # 3. Define Candidate Models and Parameters
    # We need to evaluate K-Means for different K
    k_range = range(2, 9) # Example range for K, adjust as needed
    models_to_evaluate = {}

    print("Preprocessing data...")
    try:
        # Fit preprocessor and transform data *once*
        X_processed = preprocessor.fit_transform(features_df)
        print(f"Data processed. Shape: {X_processed.shape}")
        if X_processed.shape[0] < 2:
             print("Error: Not enough samples to perform clustering after preprocessing.")
             return None, None
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return None, None

    # --- K-Means Evaluation ---
    print("Evaluating K-Means models...")
    best_kmeans_score = -1
    best_k = -1
    best_kmeans_model = None
    kmeans_metrics = {}

    for k in k_range:
        print(f"  Training K-Means with k={k}...")
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            labels = kmeans.fit_predict(X_processed)

            # Check if clustering produced more than one cluster label (needed for silhouette)
            if len(set(labels)) > 1:
                score = silhouette_score(X_processed, labels)
                print(f"    k={k}, Silhouette Score: {score:.4f}")
                kmeans_metrics[k] = {'silhouette': score}
                if score > best_kmeans_score:
                    best_kmeans_score = score
                    best_k = k
                    # Store the *fitted* KMeans object (not the pipeline yet)
                    best_kmeans_model = kmeans
            else:
                print(f"    k={k}, Only one cluster found. Cannot calculate Silhouette Score.")
                kmeans_metrics[k] = {'silhouette': None}

        except Exception as e:
            print(f"    Error training/evaluating K-Means with k={k}: {e}")
            kmeans_metrics[k] = {'silhouette': None}

    if best_kmeans_model:
        models_to_evaluate['KMeans'] = {
            'model': best_kmeans_model, # Store the fitted model object
            'params': {'n_clusters': best_k},
            'score': best_kmeans_score,
            'pipeline': Pipeline([('preprocess', preprocessor), ('cluster', best_kmeans_model)]) # Store the final pipeline structure
        }
        print(f"Best K-Means: k={best_k} with Silhouette Score: {best_kmeans_score:.4f}")
    else:
        print("K-Means training did not yield a best model.")


    # --- DBSCAN Evaluation (Example) ---
    # DBSCAN parameters often need careful tuning (e.g., using NearestNeighbors)
    # Using default parameters here for demonstration
    print("Evaluating DBSCAN model...")
    dbscan_score = -1
    dbscan_model = None
    try:
        dbscan = DBSCAN(eps=0.5, min_samples=5) # Default parameters, likely need tuning!
        labels = dbscan.fit_predict(X_processed)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0) # Exclude noise label if present

        print(f"  DBSCAN found {n_clusters} clusters.")

        # Calculate Silhouette Score only if meaningful clusters are found
        # Exclude noise points (label -1) for silhouette calculation
        if n_clusters > 1:
            labels_valid = labels[labels != -1]
            X_processed_valid = X_processed[labels != -1]
            if len(labels_valid) > 1 and X_processed_valid.shape[0] > 1: # Need at least 2 valid points in >1 cluster
                 dbscan_score = silhouette_score(X_processed_valid, labels_valid)
                 print(f"  DBSCAN Silhouette Score (excluding noise): {dbscan_score:.4f}")
                 dbscan_model = dbscan
            else:
                 print("  Not enough valid points/clusters to calculate DBSCAN Silhouette Score.")
        else:
            print("  DBSCAN did not find enough clusters to calculate Silhouette Score.")

        if dbscan_model:
             models_to_evaluate['DBSCAN'] = {
                'model': dbscan_model,
                'params': {'eps': 0.5, 'min_samples': 5}, # Store used params
                'score': dbscan_score,
                'pipeline': Pipeline([('preprocess', preprocessor), ('cluster', dbscan_model)])
            }

    except Exception as e:
        print(f"  Error training/evaluating DBSCAN: {e}")


    # --- Agglomerative Clustering Evaluation (Example) ---
    print("Evaluating Agglomerative Clustering model...")
    agg_score = -1
    agg_model = None
    agg_k = best_k if best_k > 1 else 3 # Use best K from KMeans or a default if KMeans failed
    try:
        agg_clustering = AgglomerativeClustering(n_clusters=agg_k, linkage='ward')
        labels = agg_clustering.fit_predict(X_processed)

        if len(set(labels)) > 1:
            score = silhouette_score(X_processed, labels)
            print(f"  Agglomerative Clustering (k={agg_k}) Silhouette Score: {score:.4f}")
            if score > -1: # Basic check if score is valid
                 agg_score = score
                 agg_model = agg_clustering
        else:
             print(f"  Agglomerative Clustering (k={agg_k}) found only one cluster.")

        if agg_model:
            models_to_evaluate['Agglomerative'] = {
                'model': agg_model,
                'params': {'n_clusters': agg_k, 'linkage': 'ward'},
                'score': agg_score,
                'pipeline': Pipeline([('preprocess', preprocessor), ('cluster', agg_model)])
            }

    except Exception as e:
        print(f"  Error training/evaluating Agglomerative Clustering: {e}")


    # 4. Select the Best Model based on Silhouette Score
    best_model_name = None
    best_score = -1.1 # Initialize below minimum possible silhouette score
    best_model_pipeline = None
    final_metrics = {}

    if not models_to_evaluate:
        print("Error: No models were successfully trained and evaluated.")
        return None, None

    print("\n--- Model Comparison ---")
    for name, info in models_to_evaluate.items():
        print(f"Model: {name}, Params: {info['params']}, Silhouette Score: {info.get('score', 'N/A')}")
        # Check if score is valid and better than current best
        current_score = info.get('score', -1.1)
        if isinstance(current_score, (int, float)) and current_score > best_score:
            best_score = current_score
            best_model_name = name
            best_model_pipeline = info['pipeline'] # The pipeline associated with this model

    if best_model_pipeline is None:
        print("Error: Could not determine the best model.")
        return None, None

    print(f"\nBest Performing Model: {best_model_name} (Silhouette Score: {best_score:.4f})")

    # 5. Fit the *entire* best pipeline on the full dataset
    # Note: The model component (e.g., best_kmeans_model) was already fitted during evaluation.
    #       The pipeline bundles the fitted preprocessor and the *already fitted* best clusterer.
    #       No need to call fit() on best_model_pipeline again unless we rebuilt it with an unfitted model.
    #       Let's ensure the model within the pipeline *is* the fitted one.
    #       If we stored unfitted models and just parameters, we'd call fit() here.
    #       Since we stored fitted models (like best_kmeans_model), the pipeline is ready.

    # 6. Calculate Final Metrics for the Best Model
    try:
        best_model_clusterer = best_model_pipeline.named_steps['cluster']
        labels = best_model_clusterer.labels_ # Get labels from the fitted model inside pipeline

        n_clusters_final = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"Best model ({best_model_name}) assigned data to {n_clusters_final} clusters.")

        # Exclude noise for metric calculation if DBSCAN was chosen and produced noise
        if best_model_name == 'DBSCAN' and -1 in labels:
            valid_indices = labels != -1
            if np.sum(valid_indices) > 1 and len(set(labels[valid_indices])) > 1:
                X_processed_valid = X_processed[valid_indices]
                labels_valid = labels[valid_indices]
                final_silhouette = silhouette_score(X_processed_valid, labels_valid)
                final_davies_bouldin = davies_bouldin_score(X_processed_valid, labels_valid)
                final_calinski_harabasz = calinski_harabasz_score(X_processed_valid, labels_valid)
            else:
                print("Warning: Not enough valid points/clusters in DBSCAN result for detailed metrics.")
                final_silhouette = best_score # Use the score calculated earlier if possible
                final_davies_bouldin = None
                final_calinski_harabasz = None
        elif len(set(labels)) > 1 and X_processed.shape[0] > 1:
             # Calculate metrics for K-Means or Agglomerative, or DBSCAN if no noise
            final_silhouette = silhouette_score(X_processed, labels)
            final_davies_bouldin = davies_bouldin_score(X_processed, labels)
            final_calinski_harabasz = calinski_harabasz_score(X_processed, labels)
        else:
             print("Warning: Best model resulted in single cluster or insufficient data. Cannot calculate detailed metrics.")
             final_silhouette = best_score # Use evaluation score
             final_davies_bouldin = None
             final_calinski_harabasz = None


        final_metrics = {
            "model_type": "clustering",
            "best_algorithm": best_model_name,
            "parameters": models_to_evaluate[best_model_name]['params'],
            "n_clusters_found": n_clusters_final,
            "silhouette_score": final_silhouette if final_silhouette is not None else 'N/A',
            "davies_bouldin_index": final_davies_bouldin if final_davies_bouldin is not None else 'N/A',
            "calinski_harabasz_index": final_calinski_harabasz if final_calinski_harabasz is not None else 'N/A',
            "data_file": DATA_FILE_PATH,
            "n_samples_processed": X_processed.shape[0],
            "n_features_processed": X_processed.shape[1] # After preprocessing (OHE expands features)
        }
        print("\nFinal Metrics for Best Model:")
        print(json.dumps(final_metrics, indent=4))

    except Exception as e:
        print(f"Error calculating final metrics: {e}")
        # Use the score from evaluation as fallback
        final_metrics = {
            "model_type": "clustering",
            "best_algorithm": best_model_name,
            "parameters": models_to_evaluate[best_model_name]['params'],
            "silhouette_score": best_score,
            "error": f"Could not calculate detailed metrics: {e}"
        }

    # 7. Save the Best Model Pipeline and Metrics
    try:
        joblib.dump(best_model_pipeline, MODEL_FILE)
        print(f"Best model pipeline saved to: {MODEL_FILE}")

        with open(METRICS_FILE, 'w') as f:
            json.dump(final_metrics, f, indent=4)
        print(f"Detailed metrics saved to: {METRICS_FILE}")

        # Create a simple report file (optional, content may vary)
        with open(REPORT_FILE, 'w') as f:
             f.write("Clustering Evaluation Report\n")
             f.write("="*30 + "\n")
             f.write(f"Data Source: {DATA_FILE_PATH}\n")
             f.write(f"Best Model Found: {best_model_name}\n")
             f.write(f"Model saved to: {MODEL_FILE}\n")
             f.write("\nMetrics:\n")
             f.write(json.dumps(final_metrics, indent=4))
        print(f"Evaluation report saved to: {REPORT_FILE}")


    except Exception as e:
        print(f"Error saving model or metrics: {e}")
        return None, final_metrics # Return metrics even if saving failed

    # 8. Return results
    # Using Silhouette Score as the primary "Accuracy" metric as requested
    Accuracy = final_metrics.get("silhouette_score", None)
    if Accuracy == 'N/A': # Handle the case where it couldn't be calculated
        Accuracy = None

    print("Clustering model training completed.")
    return Accuracy, final_metrics

# --- Prediction Function ---

def predict_with_model(input_data_path, model_path=MODEL_FILE, output_path=OUTPUT_FILE):
    """
    Loads a trained clustering model and applies it to new data.

    Args:
        input_data_path (str): Path to the CSV file containing new data.
        model_path (str): Path to the saved model file (.joblib).
        output_path (str): Path to save the predictions (JSON format).

    Returns:
        pandas.DataFrame: DataFrame with original data and added 'cluster_label' column,
                          or None if prediction fails.
    """
    print(f"\nStarting prediction using model: {model_path}")
    print(f"Input data: {input_data_path}")

    # 1. Load the trained model pipeline
    try:
        model_pipeline = joblib.load(model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

    # 2. Load and prepare the new data (using the same steps as training)
    try:
        # Ensure output directory for prediction exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load data, handling encoding
        try:
            new_df = pd.read_csv(input_data_path, sep=';', encoding='utf-8')
        except UnicodeDecodeError:
            new_df = pd.read_csv(input_data_path, sep=';', encoding='latin-1')

        # Select the *exact same* feature columns the model was trained on
        missing_cols = [col for col in FEATURE_COLUMNS if col not in new_df.columns]
        if missing_cols:
             raise ValueError(f"Prediction data missing required columns: {missing_cols}")

        new_features_df = new_df[FEATURE_COLUMNS].copy()

        # IMPORTANT: Apply the *same* preprocessing steps
        # The pipeline handles this automatically (uses fit_transform logic internally if needed, but here just transform)
        # No need to manually preprocess if using the saved pipeline

        print("Predicting cluster labels...")
        # 3. Predict cluster labels using the loaded pipeline
        # The pipeline applies preprocessing and then predicts
        cluster_labels = model_pipeline.predict(new_features_df)
        print("Prediction complete.")

        # 4. Add predictions to the original DataFrame
        # Use the original new_df which might have more columns than just features
        output_df = new_df.copy()
        output_df['cluster_label'] = cluster_labels

        # 5. Save the results
        output_df.to_json(output_path, orient='records', indent=4)
        print(f"Prediction results saved to: {output_path}")

        return output_df

    except FileNotFoundError:
        print(f"Error: Input data file not found at {input_data_path}")
        return None
    except ValueError as ve:
         print(f"Error preparing prediction data: {ve}")
         return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None


# --- Main Execution Block ---
if __name__ == "__main__":
    print("Executing clustering script directly...")

    # Run the training process
    Accuracy, metrics = train_clustering_model()

    if Accuracy is not None and metrics is not None:
        print(f"\nTraining Summary:")
        print(f"  Primary Metric (Silhouette Score - 'Accuracy'): {Accuracy:.4f}")
        print(f"  Detailed metrics saved in: {METRICS_FILE}")
        print(f"  Best model saved in: {MODEL_FILE}")

        # Example: Run prediction on the *training* data itself
        # In a real scenario, you'd use a separate, unseen dataset here
        print("\nRunning prediction as an example (using training data as input)...")
        prediction_results = predict_with_model(
            input_data_path=DATA_FILE_PATH, # Using training data as example input
            model_path=MODEL_FILE,
            output_path=OUTPUT_FILE
        )

        if prediction_results is not None:
            print(f"\nExample Prediction Output Head (saved to {OUTPUT_FILE}):")
            print(prediction_results.head())
        else:
            print("\nPrediction example failed.")

    else:
        print("\nModel training failed.")

    print("\nScript finished.")