import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from feature_transformer import FeatureTransformer

# Create necessary directories
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("results", exist_ok=True)

def load_and_explore_data(file_path):
    """
    Load and explore the dataset
    
    Parameters:
    file_path (str): Path to the data file
    
    Returns:
    pandas.DataFrame: Loaded dataframe
    """
    print(f"\nLoading data from {file_path}...")
    
    try:
        # Try different encodings if needed
        try:
            df = pd.read_csv(file_path)
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(file_path, encoding='latin-1')
                
        # Check for semicolon separator
        if df.shape[1] == 1 and ';' in df.iloc[0, 0]:
            df = pd.read_csv(file_path, sep=';')
            
        # Print basic info
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        print("\nColumn Data Types:")
        print(df.dtypes)
        
        print("\nSample Data:")
        print(df.head())
        
        print("\nMissing Values:")
        missing = df.isnull().sum()
        print(missing[missing > 0])
        
        return df
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

def generate_and_explore_features(df):
    """
    Generate features using the feature transformer and explore them
    
    Parameters:
    df (pandas.DataFrame): Input dataframe
    
    Returns:
    tuple: (transformed_df, transformer)
    """
    print("\nGenerating features using FeatureTransformer...")
    
    # Create and fit transformer
    transformer = FeatureTransformer(target_only=False)  # Generate ALL possible features
    transformed_df = transformer.fit_transform(df)
    
    # Show what new features were created
    original_cols = df.columns.tolist()
    new_cols = [col for col in transformed_df.columns if col not in original_cols]
    
    print(f"\nGenerated {len(new_cols)} new features.")
    
    # Group new features by type
    feature_types = {
        'Categorical': [col for col in new_cols if col.endswith('_category') or
                       col.endswith('_tier') or col.endswith('_level') or
                       col.endswith('_segment')],
        'Binary': [col for col in new_cols if col.endswith('_is_high') or
                  col.endswith('_is_outlier') or col.endswith('_above_median') or
                  col.endswith('_is_weekend')],
        'Numeric': [col for col in new_cols if col.endswith('_zscore') or
                   'per_' in col or col.endswith('_days')],
        'Time-based': [col for col in new_cols if col.endswith('_year') or
                      col.endswith('_month') or col.endswith('_quarter')]
    }
    
    # Print feature counts by group
    print("\nFeature counts by type:")
    for feature_type, cols in feature_types.items():
        print(f"- {feature_type}: {len(cols)} features")
        if len(cols) > 0:
            print(f"  Example: {cols[0]}")
    
    # Analyze potential target variables
    potential_targets = feature_types['Categorical']
    
    print(f"\nPotential target variables for classification ({len(potential_targets)}):")
    for i, target in enumerate(potential_targets[:5]):  # Show top 5
        value_counts = transformed_df[target].value_counts()
        print(f"{i+1}. {target}: {len(value_counts)} classes - {', '.join(value_counts.index.astype(str))}")
    
    # Save the transformed data
    transformed_df.to_csv("data/transformed_data.csv", index=False)
    print("\nTransformed data saved to 'data/transformed_data.csv'")
    
    return transformed_df, transformer

def train_model(transformed_df, transformer):
    """
    Train a classification model using the transformed data
    
    Parameters:
    transformed_df (pandas.DataFrame): Transformed dataframe
    transformer (FeatureTransformer): The fitted transformer
    
    Returns:
    tuple: (model, target_col, feature_cols)
    """
    print("\nPreparing for model training...")
    
    # Identify original and new columns
    original_cols = [col for col in transformed_df.columns if col in transformer.column_types]
    new_cols = [col for col in transformed_df.columns if col not in original_cols]
    
    # Identify potential target columns (categorical features)
    target_cols = [col for col in new_cols if col.endswith('_category') or
                  col.endswith('_tier') or col.endswith('_level') or
                  col.endswith('_segment')]
    
    if not target_cols:
        print("No suitable target variables found.")
        return None, None, None
    
    # Select a target column (you could use a different one)
    target_col = target_cols[0]
    print(f"Using '{target_col}' as the target variable")
    
    # Select features (use both original numeric and generated features)
    feature_cols = [col for col in transformed_df.columns
                   if col != target_col and
                   pd.api.types.is_numeric_dtype(transformed_df[col])]
    
    print(f"Using {len(feature_cols)} numeric features for training")
    
    # Prepare training data
    X = transformed_df[feature_cols]
    y = transformed_df[target_col]
    
    # Drop rows with missing target values
    valid_rows = ~y.isna()
    X = X[valid_rows]
    y = y[valid_rows]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Train the model
    print("\nTraining classification model...")
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # Save model artifacts
    print("\nSaving model artifacts...")
    
    # Save transformer
    transformer.save("models/feature_transformer.pkl")
    
    # Save model
    with open("models/classification_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # Save metadata
    with open("models/model_metadata.txt", "w") as f:
        f.write(f"Target column: {target_col}\n")
        f.write(f"Number of features: {len(feature_cols)}\n")
        # Save top 10 feature names (could be many)
        f.write(f"Feature sample: {','.join(feature_cols[:10])}\n")
    
    # Save feature list separately (could be large)
    with open("models/feature_columns.txt", "w") as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    print("Model and artifacts saved to 'models/' directory")
    
    return model, target_col, feature_cols

def apply_to_new_data(new_data_path):
    """
    Apply the trained model to new data
    
    Parameters:
    new_data_path (str): Path to new data file
    
    Returns:
    pandas.DataFrame: DataFrame with predictions
    """
    print(f"\nApplying model to new data: {new_data_path}")
    
    # Check if model exists
    if not os.path.exists("models/feature_transformer.pkl") or not os.path.exists("models/classification_model.pkl"):
        print("Error: Model files not found. Please train the model first.")
        return None
    
    # Load new data
    try:
        new_df = pd.read_csv(new_data_path)
        print(f"Loaded new data: {new_df.shape[0]} rows, {new_df.shape[1]} columns")
    except Exception as e:
        print(f"Error loading new data: {str(e)}")
        return None
    
    # Load transformer and model
    print("Loading transformer and model...")
    transformer = FeatureTransformer.load("models/feature_transformer.pkl")
    
    with open("models/classification_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    # Get target and feature information
    with open("models/model_metadata.txt", "r") as f:
        lines = f.readlines()
        target_col = lines[0].split("Target column: ")[1].strip()
    
    # Load feature columns
    with open("models/feature_columns.txt", "r") as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    # Transform new data
    print("Transforming new data...")
    transformed_new_df = transformer.transform(new_df)
    
    # Check for missing features
    missing_features = [col for col in feature_cols if col not in transformed_new_df.columns]
    if missing_features:
        print(f"Warning: {len(missing_features)} features are missing in the transformed data")
        # Add missing features as NaN
        for col in missing_features:
            transformed_new_df[col] = np.nan
    
    # Select features for prediction
    X_new = transformed_new_df[feature_cols]
    
    # Make predictions
    print("Making predictions...")
    predictions = model.predict(X_new)
    
    # Add predictions to original data
    new_df['predicted_' + target_col] = predictions
    
    # Add prediction probabilities
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_new)
        class_names = model.classes_
        
        for i, class_name in enumerate(class_names):
            new_df[f'probability_{class_name}'] = probas[:, i]
    
    # Save predictions
    output_path = f"results/predictions_{os.path.basename(new_data_path)}"
    new_df.to_csv(output_path, index=False)
    print(f"Predictions saved to '{output_path}'")
    
    return new_df

def main():
    """Main function demonstrating the complete workflow"""
    print("=" * 60)
    print("COMPLETE MACHINE LEARNING WORKFLOW WITH FEATURE TRANSFORMER")
    print("=" * 60)
    
    # Step 1: Load and explore data
    data_path = "C:/data_sets/california_housing_train.csv"  # Replace with your actual data path
    df = load_and_explore_data(data_path)
    
    if df is None:
        return
    
    # Step 2: Generate and explore features
    transformed_df, transformer = generate_and_explore_features(df)
    
    # Step 3: Train model
    model, target_col, feature_cols = train_model(transformed_df, transformer)
    
    if model is None:
        return
    
    # Step 4: Apply to new data
    new_data_path = "C:/data_sets/california_housing_test.csv"   # Replace with your new data path
    result_df = apply_to_new_data(new_data_path)
    
    if result_df is not None:
        print("\nSample predictions:")
        print(result_df.head())
    
    print("\nWorkflow complete!")


# df = pd.read_csv("your_data.csv")

# # Create a transformer that generates ALL features
# transformer = FeatureTransformer(target_only=False)

# # Generate features
# transformed_df = transformer.fit_transform(df)

# # Save the transformer for later use
# transformer.save("models/feature_transformer.pkl")

# # When you have new data to predict:
# new_df = pd.read_csv("new_data.csv")
# transformer = FeatureTransformer.load("models/feature_transformer.pkl")
# transformed_new_df = transformer.transform(new_df)

if __name__ == "__main__":
    main()
