import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os
import json
from sklearn.metrics import silhouette_score
import joblib
import subprocess

# User-specific variable
USER = 'josip'

# Define the data file path
DATA_FILE_PATH = "..uploads/bank-full.csv"
MODEL_DIR = "trained_models"
OUTPUT_DIR = 'trained_models_outcome'
SCRIPT_NAME = os.path.splitext(os.path.basename(__file__))[0]
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"{USER}_{SCRIPT_NAME}_results.json")


FEATURE_COLUMNS = ['age', 'job', 'marital', 'education', 'balance', 'housing', 'loan']
TARGET_COLUMN = None  # No target variable for clustering

NUMERICAL_FEATURES = ['age', 'balance']
CATEGORICAL_FEATURES = ['job', 'marital', 'education', 'housing', 'loan']

def read_data_flexible(file_path):
    """
    Reads data from a CSV file, attempting different encodings.

    Args:
        file_path (str): The path to the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame, or None if reading fails.
    """
    encodings_to_try = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings_to_try:
        try:
            df = pd.read_csv(file_path, encoding=encoding, sep=';')
            print(f"Successfully read file using encoding: {encoding}")
            return df
        except UnicodeDecodeError:
            print(f"Failed to read with encoding: {encoding}")
        except FileNotFoundError:
            print(f"Error: File not found at {file_path}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during file reading: {e}")
            return None
    return None

def train_clustering_model(data_path=DATA_FILE_PATH, output_path=OUTPUT_FILE):
    """
    Trains and evaluates clustering models and saves the results to a JSON file.

    Args:
        data_path (str): The path to the input CSV file.
        output_path (str): The path to save the JSON output.

    Returns:
        dict: A dictionary containing evaluation metrics and model information.
    """
    results = {}
    try:
        df = read_data_flexible(data_path)
        if df is None:
            results["error"] = f"Failed to read data from {data_path}"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=4)
            return results
        X = df[FEATURE_COLUMNS].copy()

        # Preprocessing
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, NUMERICAL_FEATURES),
                ('cat', categorical_transformer, CATEGORICAL_FEATURES)])

        # Define the clustering model (K-Means as an example)
        # You can experiment with different numbers of clusters
        n_clusters = 5
        kmeans_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

        # Create a pipeline
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('clusterer', kmeans_model)])

        # Train the model
        pipeline.fit(X)

        # Get cluster labels
        labels = pipeline.predict(X)

        # Evaluate the clustering (using Silhouette Score)
        # Note: Silhouette Score is more meaningful when clusters are somewhat distinct.
        # Other metrics might be relevant depending on the data and goals.
        try:
            silhouette = silhouette_score(pipeline.named_steps['preprocessor'].transform(X), labels)
        except ValueError:
            silhouette = "Silhouette Score not well-defined for a single cluster"

        # Save the trained model
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_file_path = os.path.join(MODEL_DIR, "best_clustering_model.joblib")
        joblib.dump(pipeline, model_file_path)

        # Save metrics
        metrics = {
            "silhouette_score": silhouette,
            "n_clusters": n_clusters,
            "model_path": model_file_path
        }
        metrics_file_path = os.path.join(MODEL_DIR, "clustering_metrics.json")
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        results = {
            "model_name": "KMeans",
            "silhouette_score": silhouette,
            "n_clusters": n_clusters,
            "model_path": model_file_path,
            "metrics_path": metrics_file_path
        }

    except Exception as e:
        results["error"] = f"An error occurred during model training: {e}"

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save the results to a JSON file
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(json.dumps(results)) # Print the results to stdout
    return results

def run_training_script(training_script):
    """
    Runs a Python training script as a subprocess and reads its JSON output.

    Args:
        training_script (str): The path to the Python training script.

    Returns:
        dict: A dictionary containing the parsed JSON output from the script,
              or an error dictionary if the script fails or the output is not valid JSON.
    """
    command = ["python", training_script]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        return {"error": stderr}  # Return error in a dict

    try:
        return json.loads(stdout)  # Parse JSON
    except json.JSONDecodeError:
        return {"error": "Invalid JSON output from script"}

# Main execution block
if __name__ == "__main__":
    # Train the model and save the output to a file
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_clustering_model(DATA_FILE_PATH, OUTPUT_FILE)

    # # Now, simulate running this script as an external process
    results_from_subprocess = run_training_script(__file__)
    # print("Results from subprocess:")
    # print(json.dumps(results_from_subprocess, indent=4))