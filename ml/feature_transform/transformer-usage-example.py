import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from feature_transformer import FeatureTransformer

# Example workflow for training a model and predicting on new data

def train_and_save_pipeline(data_path):
    """
    Train a model and save the feature transformer and model
    
    Parameters:
    data_path (str): Path to the training dataset CSV
    """
    print("Loading training data...")
    df = pd.read_csv(data_path)
    
    # 1. Create and fit the feature transformer (generate all possible features)
    print("Creating feature transformations...")
    transformer = FeatureTransformer(target_only=False)
    transformed_df = transformer.fit_transform(df)
    
    # 2. Identify potential target variables from the generated features
    generated_cols = [col for col in transformed_df.columns if col not in df.columns]
    target_cols = [col for col in generated_cols 
                  if col.endswith('_category') or col.endswith('_tier') 
                  or col.endswith('_segment') or col.endswith('_level')]
    
    if not target_cols:
        print("No suitable target variables were created. Check your data.")
        return
    
    # Select the first target variable (you could choose a different one)
    target_col = target_cols[0]
    print(f"Using {target_col} as target variable for training.")
    
    # 3. Prepare features for the model - use both original and generated features
    # Filter out categorical columns and the target
    feature_cols = [col for col in transformed_df.columns 
                   if col != target_col 
                   and not (isinstance(transformed_df[col].dtype, pd.CategoricalDtype) 
                           or transformed_df[col].dtype == 'object')]
    
    print(f"Using {len(feature_cols)} features for training.")
    
    # 4. Split data for training and validation
    X = transformed_df[feature_cols]
    y = transformed_df[target_col]
    
    # Drop rows with missing target values
    valid_rows = ~y.isna()
    X = X[valid_rows]
    y = y[valid_rows]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # 5. Train a model
    print("Training classification model...")
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    # 6. Evaluate the model
    y_pred = model.predict(X_test)
    print("\nModel Evaluation:")
    print(classification_report(y_test, y_pred))
    
    # 7. Save both the transformer and model
    print("\nSaving pipeline components...")
    import os
    os.makedirs("models", exist_ok=True)
    
    transformer.save("models/feature_transformer.pkl")
    
    import pickle
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    # 8. Save metadata for future reference
    with open("models/model_metadata.txt", "w") as f:
        f.write(f"Target column: {target_col}\n")
        f.write(f"Feature columns: {','.join(feature_cols)}\n")
    
    print("Training and saving complete.")
    return transformer, model, target_col, feature_cols


def predict_on_new_data(new_data_path):
    """
    Load saved transformer and model to make predictions on new data
    
    Parameters:
    new_data_path (str): Path to the new data CSV
    
    Returns:
    pandas.DataFrame: Original data with predictions added
    """
    # 1. Load the new data
    print(f"Loading new data from {new_data_path}...")
    new_df = pd.read_csv(new_data_path)
    
    # 2. Load the saved transformer and model
    print("Loading transformer and model...")
    try:
        transformer = FeatureTransformer.load("models/feature_transformer.pkl")
        
        import pickle
        with open("models/model.pkl", "rb") as f:
            model = pickle.load(f)
            
        # Read metadata
        with open("models/model_metadata.txt", "r") as f:
            lines = f.readlines()
            target_col = lines[0].split(": ")[1].strip()
            feature_cols = lines[1].split(": ")[1].strip().split(",")
            
    except FileNotFoundError:
        print("Error: Saved model or transformer not found. Run training first.")
        return None
    
    # 3. Apply the same transformations to new data
    print("Applying transformations to new data...")
    transformed_df = transformer.transform(new_df)
    
    # 4. Ensure we have all the required features
    missing_cols = [col for col in feature_cols if col not in transformed_df.columns]
    if missing_cols:
        print(f"Warning: Missing {len(missing_cols)} feature columns. These will be filled with NaN.")
        for col in missing_cols:
            transformed_df[col] = np.nan
    
    # 5. Make predictions
    print("Making predictions...")
    X_new = transformed_df[feature_cols]
    predictions = model.predict(X_new)
    
    # 6. Add predictions to the original data
    new_df['predicted_' + target_col] = predictions
    
    # 7. Calculate prediction probabilities if it's a classifier
    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(X_new)
        class_names = model.classes_
        
        # Add probability for each class
        for i, class_name in enumerate(class_names):
            new_df[f'probability_{class_name}'] = probas[:, i]
            
    print("Prediction complete.")
    return new_df


# Example usage
if __name__ == "__main__":
    import os
    
    # Create output directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Example 1: Train and save pipeline
    print("=" * 50)
    print("EXAMPLE 1: TRAINING THE MODEL")
    print("=" * 50)
    train_data_path = "C:/data_sets/california_housing_train.csv"  # Replace with your training file
    transformer, model, target_col, feature_cols = train_and_save_pipeline(train_data_path)
    
    # Example 2: Use saved pipeline to predict on new data
    print("\n" + "=" * 50)
    print("EXAMPLE 2: PREDICTING ON NEW DATA")
    print("=" * 50)
    new_data_path = "california_housing_test.csv"  # Replace with your new data file
    result_df = predict_on_new_data(new_data_path)
    
    if result_df is not None:
        print("\nSample of predictions:")
        print(result_df.head())
        
        # Save predictions
        result_df.to_csv("predictions.csv", index=False)
        print("Predictions saved to predictions.csv")

