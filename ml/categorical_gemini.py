import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Define the directory where the data files are located
data_dir = '../uploads'  # Update this if your data is in a different directory

# Function to read data files
def read_data(filename):
    """
    Reads a data file (CSV or Excel) into a pandas DataFrame.

    Args:
        filename (str): The name of the file to read.

    Returns:
        pandas.DataFrame: The DataFrame containing the data, or None if the file type is unsupported or an error occurs.
    """
    try:
        if filename.endswith('.csv'):
            delimiter = get_delimiter(os.path.join(data_dir, filename))
            return pd.read_csv(os.path.join(data_dir, filename), delimiter=delimiter)
        elif filename.endswith('.xlsx'):
            return pd.read_excel(os.path.join(data_dir, filename))
        else:
            print(f"Unsupported file type: {filename}")
            return None
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return None

def get_delimiter(file_path, max_lines=100):
    """
    Sniffs the delimiter of a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        max_lines (int, optional): Maximum number of lines to read for sniffing. Defaults to 100.

    Returns:
        str: The detected delimiter, or ';' if detection fails.
    """
    try:
        with open(file_path, 'r') as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                if ',' in line:
                    return ','
                elif ';' in line:
                    return ';'
        return ';'  # Default delimiter if none found
    except Exception:
        return ';'

# Function to predict classification for categorical variables
def predict_categorical(df, filename):
    """
    Predicts classification for all categorical variables in a DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame.
        filename (str): The name of the file being processed.

    Returns:
        list: A list of dictionaries, where each dictionary contains the results for one target variable.
              Returns an empty list if no suitable target variables are found or an error occurs.
    """
    results = []
    if df is None:
        return results

    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if not categorical_cols.any():
        print(f"No categorical columns found in {filename}")
        return results

    for target_variable in categorical_cols:
        # Avoid using the target variable as a feature
        features = [col for col in categorical_cols if col != target_variable]

        if not features:
            print(f"Not enough features to predict {target_variable} in {filename}")
            continue

        # Prepare data: Features (X) and Target (y)
        X = df[features]
        y = df[target_variable]

        # Check if the target variable has more than one unique class
        if len(y.unique()) <= 1:
            print(f"Target variable '{target_variable}' in {filename} has only one unique value, skipping.")
            continue

        # Encode categorical features and target variable
        X_encoded = pd.get_dummies(X, dummy_na=False) # handle missing values in features
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)


        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

        # Train a Random Forest Classifier model.  Handle potential errors during model training.
        try:
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"Error training model for target variable '{target_variable}' in {filename}: {e}")
            continue # Skip to the next target variable

        # Make predictions
        y_pred = model.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)

        # Get feature importances
        importances = model.feature_importances_.tolist()
        feature_names = X_encoded.columns.tolist()  # Get feature names *after* encoding
        feature_importances = dict(zip(feature_names, importances))

        # Save model and features (in memory paths for now)
        model_path = "in_memory_model"  # Placeholder for a real path
        features_path = "in_memory_features"  # Placeholder

        # Store the results
        results.append({
            'model_path': model_path,
            'features_path': features_path,
            'accuracy': accuracy,
            'importances': feature_importances,
            'filename': filename,
            'target_variable': target_variable
        })
    return results

def main():
    """
    Main function to process all files in the data directory and print the results.
    """
    # Create the data directory if it does not exist
    if not os.path.exists(data_dir):
        print(f"Error: The directory '{data_dir}' does not exist. Please create it and place the data files there.")
        return

    # Process each file in the data directory
    for filename in os.listdir(data_dir):
        df = read_data(filename)
        if df is not None: # only process if reading the file was successful
            results = predict_categorical(df.copy(), filename) # Pass a copy to avoid modifying original DataFrame
            if results:
                print(f"\nResults for {filename}:")
                for result in results:
                    print(f"  Target Variable: {result['target_variable']}")
                    print(f"    Model Path: {result['model_path']}")
                    print(f"    Features Path: {result['features_path']}")
                    print(f"    Accuracy: {result['accuracy']:.4f}")
                    print("    Feature Importances:")
                    for feature, importance in result['importances'].items():
                        print(f"      {feature}: {importance:.4f}")
                    print("-" * 40)
            else:
                print(f"\nNo suitable categorical target variables found in {filename}")

if __name__ == "__main__":
    main()

