import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline
import joblib
import os
from datetime import datetime

# --- Configuration ---
INPUT_FILE_PATH = '../uploads/e-commerce.csv'
MODEL_SAVE_DIR = 'trained_models'
CLUSTERING_MODEL_TYPE = 'clustering' # Variable name indicating the model type

# --- Create Save Directory ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 1. Load Data ---
try:
    # Attempt to read with utf-8, fall back to latin1 if error
    try:
        df = pd.read_csv(INPUT_FILE_PATH, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8 decoding failed, trying latin1 encoding.")
        df = pd.read_csv(INPUT_FILE_PATH, encoding='latin1')
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_FILE_PATH}")
    exit()
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# --- 2. Feature Engineering ---
# Convert 'Transaction Date' to datetime
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])

# Calculate Recency, Frequency, Monetary Value (RFM) and other features per customer
# Find the most recent date in the dataset + 1 day for recency calculation
max_date = df['Transaction Date'].max() + pd.Timedelta(days=1)

customer_data = df.groupby('Customer ID').agg(
    Recency=('Transaction Date', lambda x: (max_date - x.max()).days),
    Frequency=('Order ID', 'nunique'),
    TotalSpending=('Order Total', 'sum'),
    TotalQuantity=('Quantity', 'sum')
)

# Calculate Average Order Value
customer_data['AverageOrderValue'] = customer_data['TotalSpending'] / customer_data['Frequency']

# Handle potential NaN/inf values if Frequency is 0 (though unlikely with nunique)
customer_data.replace([np.inf, -np.inf], np.nan, inplace=True)
customer_data.fillna(0, inplace=True) # Fill potential NaNs with 0, e.g., if a customer somehow had 0 frequency

# Select features for clustering
features_to_cluster = ['Recency', 'Frequency', 'TotalSpending', 'TotalQuantity', 'AverageOrderValue']
X = customer_data[features_to_cluster]

# --- 3. Data Preprocessing ---
# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4. Model Training and Selection ---
n_clusters_range = range(2, 11) # Range of clusters to try for K-Means and Agglomerative
best_score = -1  # Initialize with -1 for Silhouette Score
best_model = None
best_model_name = ""
best_n_clusters = 0
all_metrics = {}

# --- K-Means ---
print("Evaluating K-Means...")
kmeans_metrics = {}
best_kmeans_score = -1
best_kmeans_model = None
best_k_kmeans = 0

for k in n_clusters_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    
    # Avoid calculating metrics if only one cluster is formed (or invalid labels)
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)
        kmeans_metrics[k] = {'Silhouette': silhouette, 'Davies-Bouldin': db_score, 'Calinski-Harabasz': ch_score}
        print(f"  K-Means (k={k}): Silhouette={silhouette:.4f}, DB={db_score:.4f}, CH={ch_score:.4f}")
        
        if silhouette > best_kmeans_score:
            best_kmeans_score = silhouette
            best_kmeans_model = kmeans
            best_k_kmeans = k
    else:
         print(f"  K-Means (k={k}): Could not form more than one cluster.")

print(f"Best K-Means: k={best_k_kmeans}, Silhouette Score={best_kmeans_score:.4f}")
all_metrics['KMeans'] = {'best_k': best_k_kmeans, 'metrics': kmeans_metrics.get(best_k_kmeans, {})}

if best_kmeans_score > best_score:
    best_score = best_kmeans_score
    best_model = best_kmeans_model
    best_model_name = "KMeans"
    best_n_clusters = best_k_kmeans

# --- Agglomerative Clustering ---
print("\nEvaluating Agglomerative Clustering...")
agglo_metrics = {}
best_agglo_score = -1
best_agglo_model = None
best_k_agglo = 0

for k in n_clusters_range:
    # Linkage 'ward' requires Euclidean distance, which is default with pre-scaled data
    agglo = AgglomerativeClustering(n_clusters=k, linkage='ward') 
    labels = agglo.fit_predict(X_scaled)
    
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X_scaled, labels)
        db_score = davies_bouldin_score(X_scaled, labels)
        ch_score = calinski_harabasz_score(X_scaled, labels)
        agglo_metrics[k] = {'Silhouette': silhouette, 'Davies-Bouldin': db_score, 'Calinski-Harabasz': ch_score}
        print(f"  Agglomerative (k={k}): Silhouette={silhouette:.4f}, DB={db_score:.4f}, CH={ch_score:.4f}")

        if silhouette > best_agglo_score:
            best_agglo_score = silhouette
            best_agglo_model = agglo
            best_k_agglo = k
    else:
         print(f"  Agglomerative (k={k}): Could not form more than one cluster.")


print(f"Best Agglomerative: k={best_k_agglo}, Silhouette Score={best_agglo_score:.4f}")
all_metrics['Agglomerative'] = {'best_k': best_k_agglo, 'metrics': agglo_metrics.get(best_k_agglo, {})}

if best_agglo_score > best_score:
    best_score = best_agglo_score
    best_model = best_agglo_model
    best_model_name = "Agglomerative"
    best_n_clusters = best_k_agglo

# --- (Optional) DBSCAN --- 
# DBSCAN parameter tuning can be complex. Trying a default setting for comparison.
print("\nEvaluating DBSCAN...")
dbscan = DBSCAN(eps=0.5, min_samples=5) # Common starting point, might need tuning
labels = dbscan.fit_predict(X_scaled)
n_clusters_dbscan = len(set(labels)) - (1 if -1 in labels else 0) # Count clusters, exclude noise points (-1)
n_noise = list(labels).count(-1)

dbscan_metrics = {}
if n_clusters_dbscan > 1:
    silhouette = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    ch_score = calinski_harabasz_score(X_scaled, labels)
    dbscan_metrics = {'Silhouette': silhouette, 'Davies-Bouldin': db_score, 'Calinski-Harabasz': ch_score}
    print(f"  DBSCAN (eps=0.5, min_samples=5): Clusters={n_clusters_dbscan}, Noise={n_noise}, Silhouette={silhouette:.4f}, DB={db_score:.4f}, CH={ch_score:.4f}")
    
    # Note: Comparing DBSCAN Silhouette directly with k-based methods might be tricky
    # if the number of clusters found is very different. Let's stick to comparing K-Means/Agglo primarily based on Silhouette for simplicity.
    # However, if DBSCAN gives a good score and reasonable clusters, it could be considered.
    # For this script, we will only update the best model if DBSCAN beats the current best score AND finds a reasonable number of clusters (e.g. > 1)
    if silhouette > best_score:
       best_score = silhouette
       best_model = dbscan
       best_model_name = "DBSCAN"
       best_n_clusters = n_clusters_dbscan # Store the number of clusters found by DBSCAN
       
else:
    print(f"  DBSCAN (eps=0.5, min_samples=5): Did not find more than one cluster (found {n_clusters_dbscan} clusters, {n_noise} noise points).")

all_metrics['DBSCAN'] = {'params': {'eps': 0.5, 'min_samples': 5}, 'n_clusters': n_clusters_dbscan, 'n_noise': n_noise, 'metrics': dbscan_metrics}


# --- 5. Save Best Model and Scaler ---
if best_model:
    print(f"\nSelected Best Model: {best_model_name} (Number of clusters/Best k: {best_n_clusters}) with Silhouette Score: {best_score:.4f}")
    
    # Save the scaler
    scaler_path = os.path.join(MODEL_SAVE_DIR, f'{CLUSTERING_MODEL_TYPE}_scaler.joblib')
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to {scaler_path}")

    # Save the best model
    model_path = os.path.join(MODEL_SAVE_DIR, f'best_{CLUSTERING_MODEL_TYPE}_model.joblib')
    joblib.dump(best_model, model_path)
    print(f"Best model saved to {model_path}")

    # --- 6. Save Statistics ---
    # Using Silhouette score as the primary metric, stored in 'Accuracy' as requested
    Accuracy = best_score 
    
    # Store detailed metrics of the best performing model
    if best_model_name == "KMeans":
        detailed_metrics = all_metrics['KMeans']['metrics']
    elif best_model_name == "Agglomerative":
        detailed_metrics = all_metrics['Agglomerative']['metrics']
    elif best_model_name == "DBSCAN":
        detailed_metrics = all_metrics['DBSCAN']['metrics']
    else:
        detailed_metrics = {}
        
    print(f"\nModel Statistics:")
    print(f"  Best Model Type: {best_model_name}")
    if best_model_name != "DBSCAN":
         print(f"  Optimal Number of Clusters (k): {best_n_clusters}")
    else:
         print(f"  Number of Clusters Found: {best_n_clusters}")
         print(f"  Number of Noise Points: {all_metrics['DBSCAN']['n_noise']}")
         print(f"  Parameters Used (eps, min_samples): {all_metrics['DBSCAN']['params']}")

    print(f"  Silhouette Score (Accuracy Variable): {Accuracy:.4f}")
    if detailed_metrics:
      print(f"  Davies-Bouldin Index: {detailed_metrics.get('Davies-Bouldin', 'N/A'):.4f}")
      print(f"  Calinski-Harabasz Index: {detailed_metrics.get('Calinski-Harabasz', 'N/A'):.4f}")
    
else:
    print("\nNo suitable clustering model found or evaluation failed.")
    Accuracy = -1 # Indicate failure or no model
    detailed_metrics = {}


# --- 8. Prepare script for running the best model on other data ---

def predict_customer_segments(input_csv_path, scaler_path, model_path):
    """
    Loads new data, preprocesses it using the saved scaler, and predicts
    cluster assignments using the saved clustering model.

    Args:
        input_csv_path (str): Path to the new CSV data file.
        scaler_path (str): Path to the saved scaler (.joblib file).
        model_path (str): Path to the saved clustering model (.joblib file).

    Returns:
        pandas.DataFrame: DataFrame with Customer ID and their assigned Cluster.
                          Returns None if an error occurs.
    """
    print(f"\n--- Running Prediction on: {input_csv_path} ---")
    try:
        # Load the scaler and model
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        print("Loaded scaler and model successfully.")

        # Load new data
        try:
            new_df = pd.read_csv(input_csv_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8 decoding failed for new data, trying latin1.")
            new_df = pd.read_csv(input_csv_path, encoding='latin1')
        
        # --- Perform IDENTICAL Feature Engineering ---
        new_df['Transaction Date'] = pd.to_datetime(new_df['Transaction Date'])
        # Use the same max_date from training for consistency in Recency calculation
        # Or recalculate max_date based *only* on training data if that's desired
        # Using the previously calculated max_date ensures recency is comparable
        
        new_customer_data = new_df.groupby('Customer ID').agg(
            Recency=('Transaction Date', lambda x: (max_date - x.max()).days),
            Frequency=('Order ID', 'nunique'),
            TotalSpending=('Order Total', 'sum'),
            TotalQuantity=('Quantity', 'sum')
        )
        new_customer_data['AverageOrderValue'] = new_customer_data['TotalSpending'] / new_customer_data['Frequency']
        new_customer_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        new_customer_data.fillna(0, inplace=True)

        # Ensure the columns used for scaling match the training features
        features_to_cluster = ['Recency', 'Frequency', 'TotalSpending', 'TotalQuantity', 'AverageOrderValue']
        
        # Check if all required columns are present after grouping
        missing_cols = [col for col in features_to_cluster if col not in new_customer_data.columns]
        if missing_cols:
             print(f"Error: Missing required columns after feature engineering: {missing_cols}")
             return None
             
        X_new = new_customer_data[features_to_cluster]

        # Apply the *saved* scaler
        X_new_scaled = scaler.transform(X_new) # Use transform, not fit_transform

        # Predict clusters
        predictions = model.fit_predict(X_new_scaled) # Use fit_predict for clustering models

        # Create result DataFrame
        results_df = pd.DataFrame({'Customer ID': new_customer_data.index, 'Cluster': predictions})
        print("Prediction complete.")
        print(results_df.head())
        
        # Add cluster labels back to the aggregated customer data
        new_customer_data['Cluster'] = predictions
        print("\nAggregated customer data with clusters:")
        print(new_customer_data.head())


        return results_df

    except FileNotFoundError:
        print(f"Error: Scaler or Model file not found at {scaler_path} or {model_path}")
        return None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None

# --- Example of using the prediction function ---
# This will run prediction on the original data file as an example
if best_model:
    scaler_load_path = os.path.join(MODEL_SAVE_DIR, f'{CLUSTERING_MODEL_TYPE}_scaler.joblib')
    model_load_path = os.path.join(MODEL_SAVE_DIR, f'best_{CLUSTERING_MODEL_TYPE}_model.joblib')
    
    # Check if files were actually saved before attempting prediction
    if os.path.exists(scaler_load_path) and os.path.exists(model_load_path):
        prediction_results = predict_customer_segments(INPUT_FILE_PATH, scaler_load_path, model_load_path)
        # You can work with prediction_results DataFrame here if needed
    else:
        print("\nCould not run prediction example because model/scaler files were not saved.")
else:
    print("\nSkipping prediction example as no best model was determined.")

print("\nScript finished.")