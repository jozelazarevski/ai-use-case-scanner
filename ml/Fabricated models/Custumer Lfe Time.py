import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # For saving and loading models and transformers
from sklearn.preprocessing import StandardScaler
from sklearn.compose import make_column_transformer


def train_and_save_model(csv_path='C:/data_sets/bank-full.csv'):
    """
    Trains a CLV prediction model, saves the model and preprocessor to disk.

    Args:
        csv_path (str): Path to the CSV file.  Defaults to 'bank-full.csv'.
    """
    # Load the dataset
    df = pd.read_csv(csv_path, delimiter=';')

    # Convert 'y' to binary (1 for 'yes', 0 for 'no')
    df['y'] = df['y'].map({'yes': 1, 'no': 0})

    # Calculate Recency, Frequency and MonetaryValue
    latest_date = df['day'].max()
    rfm_df = df.groupby('age').agg(
        Recency=('day', lambda x: latest_date - x.max()),
        Frequency=('y', 'sum'),
        MonetaryValue=('balance', 'mean')
    ).reset_index()

    # Split the data into training and testing sets
    X = rfm_df[['Recency', 'Frequency', 'MonetaryValue']]
    y = rfm_df['MonetaryValue']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocess the numerical data
    numerical_features = ['Recency', 'Frequency', 'MonetaryValue']  # Define numerical features
    preprocessor = make_column_transformer(
        (StandardScaler(), numerical_features),  # Scale numerical features
        remainder='passthrough'  # Keep other columns as is (if any)
    )

    # Fit and transform the training data
    X_train_transformed = preprocessor.fit_transform(X_train)
    X_test_transformed = preprocessor.transform(X_test)  # Apply to test data

    # Train a linear regression model
    model = LinearRegression()
    model.      (X_train_transformed, y_train)

    # Evaluate the model
    mse = mean_squared_error(y_test, model.predict(X_test_transformed))
    r2 = r2_score(y_test, model.predict(X_test_transformed))
    print(f"Trained Model: Mean Squared Error: {mse:.2f}, R-squared: {r2:.2f}")

    # Save the model and preprocessor
    joblib.dump(model, 'clv_prediction_model.joblib')
    joblib.dump(preprocessor, 'clv_preprocessor.joblib')
    print("Model and preprocessor saved to disk.")
    return preprocessor, model  # Return for use in this session


def predict_clv(new_data, preprocessor=None, model=None):
    """
    Predicts CLV for new, unseen data using the saved model and preprocessor.

    Args:
        new_data (pd.DataFrame): DataFrame containing the new data with 'Recency',
                          'Frequency', and 'MonetaryValue' columns.
        preprocessor (object, optional): Preprocessor object. If None, it will be loaded
            from 'clv_preprocessor.joblib'.
        model (object, optional): Trained model object. If None, it will be loaded
            from 'clv_prediction_model.joblib'.

    Returns:
        pd.Series: Predicted CLV values for the new data.
    """
    if preprocessor is None:
        preprocessor = joblib.load('clv_preprocessor.joblib')
    if model is None:
        model = joblib.load('clv_prediction_model.joblib')

    # Transform the new data using the saved preprocessor
    new_data_transformed = preprocessor.transform(new_data)

    # Make predictions
    predicted_clv = model.predict(new_data_transformed)
    return predicted_clv



if __name__ == "__main__":
    # Train and save the model
    preprocessor, model = train_and_save_model() #capture the return

    # Example of predicting CLV for new data
    new_data = pd.DataFrame({
        'Recency': [10, 20, 5],
        'Frequency': [2, 5, 1],
        'MonetaryValue': [1500, 2500, 1000]
    })
    predictions = predict_clv(new_data, preprocessor, model) # Pass preprocessor and model
    print("Predictions for new data:")
    print(predictions)

    # Visualize the predictions (optional)
    plt.figure(figsize=(10, 6))
    sns.histplot(predictions, kde=True)
    plt.title('Predicted CLV Distribution for New Data')
    plt.xlabel('CLV')
    plt.ylabel('Frequency')
    plt.show()
