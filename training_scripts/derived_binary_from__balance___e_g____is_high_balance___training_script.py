import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.exceptions import NotFittedError
import warnings
from sklearn.exceptions import ConvergenceWarning

# Suppress specific warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
# --- Configuration ---
INPUT_FILE_PATH = '../uploads/bank-full.csv'
MODEL_SAVE_DIR = 'trained_models'
TARGET_VARIABLE = 'is_high_balance'
BALANCE_THRESHOLD = 10000
MODEL_TYPE = 'classification' # Clear variable name for model type
RANDOM_STATE = 42
TEST_SIZE = 0.2

# --- Ensure model directory exists ---
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# --- 1. Load Data ---
df = pd.read_csv(INPUT_FILE_PATH, sep=';')

try:
    # Try reading with default utf-8 encoding
    df = pd.read_csv(INPUT_FILE_PATH, sep=';')
except UnicodeDecodeError:
    try:
        # Fallback to latin1 if utf-8 fails
        df = pd.read_csv(INPUT_FILE_PATH, sep=';', encoding='latin1')
    except Exception as e:
        print(f"Error reading CSV file: {e}")
       
except FileNotFoundError:
    print(f"Error: Input file not found at {INPUT_FILE_PATH}")
   

# --- 2. Define Target Variable ---
df[TARGET_VARIABLE] = (df['balance'] > BALANCE_THRESHOLD).astype(int)

# --- 3. Feature Engineering & Preprocessing ---

# Identify features (excluding original balance and the other target 'y')
# Including 'duration' might leak information if prediction happens before the call ends.
# Exclude 'duration' for a more realistic pre-call prediction scenario.
# However, for predicting *propensity* based on past interactions, it could be included.
# We will exclude 'duration' as per typical use cases for proactive outreach.
# Also exclude 'pdays' if -1 means no previous contact, as it might correlate strongly with 'previous' == 0.
# Keeping 'pdays' for now, but its high negative value might need specific handling if not using tree-based models well.

features = ['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'day', 'month', 'campaign', 'pdays', 'previous', 'poutcome']
X = df[features]
y = df[TARGET_VARIABLE]

# Identify categorical and numerical features
numerical_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

# Create preprocessing pipelines for numerical and categorical features
# Handle unknown values in categorical features by treating them as a separate category
# Scale numerical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False) # Use sparse_output=False for easier handling later if needed

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='passthrough' # Keep other columns (if any) - should be none here
)

# --- 4. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y # Stratify due to potential imbalance
)

# --- 5. Model Training & Selection ---

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(random_state=RANDOM_STATE, max_iter=1000, class_weight='balanced'),
    # 'Random Forest': RandomForestClassifier(random_state=RANDOM_STATE, class_weight='balanced'),
    # 'Gradient Boosting': GradientBoostingClassifier(random_state=RANDOM_STATE)
}

# Define hyperparameter grids for tuning (optional, but good practice)
# Keep grids small for faster execution in this example
param_grids = {
    'Logistic Regression': {
        'classifier__C': [0.1, 1.0, 10.0]
    },
    'Random Forest': {
        'classifier__n_estimators': [50, 100],
        'classifier__max_depth': [None, 10]
    },
    'Gradient Boosting': {
        'classifier__n_estimators': [50, 100],
        'classifier__learning_rate': [0.05, 0.1]
    }
}

best_model = None
best_model_name = ""
best_accuracy = 0.0
best_pipeline = None
results = {}

print("Starting model training and evaluation...")

for name, model in models.items():
    print(f"\nTraining {name}...")
    # Create the full pipeline: Preprocessing + Classifier
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # Parameter Tuning (optional but recommended)
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    print(f"Best parameters for {name}: {grid_search.best_params_}")
    current_pipeline = grid_search.best_estimator_ # Use the best estimator found by GridSearchCV

    # Evaluate on the test set
    y_pred = current_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Low Balance', 'High Balance'])
    conf_matrix = confusion_matrix(y_test, y_pred)

    results[name] = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': conf_matrix,
        'pipeline': current_pipeline # Store the fitted pipeline
    }

    print(f"Accuracy for {name}: {accuracy:.4f}")
    # print(f"Classification Report for {name}:\n{report}")
    # print(f"Confusion Matrix for {name}:\n{conf_matrix}")

    # Update best model if current model is better
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_pipeline = current_pipeline # This pipeline includes preprocessing

print(f"\nBest performing model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# --- 6. Save Best Model & Preprocessor ---
# The best_pipeline already contains the fitted preprocessor and the best model
model_filename = os.path.join(MODEL_SAVE_DIR, f'{MODEL_TYPE}_best_model_pipeline.joblib')
joblib.dump(best_pipeline, model_filename)
print(f"Best model pipeline saved to {model_filename}")

# --- 7. Return Statistics ---
Accuracy = best_accuracy # Store accuracy in the specified variable

# Save detailed metrics for the best model
metrics_filename = os.path.join(MODEL_SAVE_DIR, f'{MODEL_TYPE}_best_model_metrics.txt')
best_model_results = results[best_model_name]
with open(metrics_filename, 'w') as f:
    f.write(f"Best Model: {best_model_name}\n")
    f.write(f"Accuracy: {best_model_results['accuracy']:.4f}\n\n")
    f.write("Classification Report:\n")
    f.write(best_model_results['classification_report'])
    f.write("\n\nConfusion Matrix:\n")
    f.write(np.array2string(best_model_results['confusion_matrix']))

print(f"Detailed metrics saved to {metrics_filename}")

# --- 8. Prepare for running on other data ---
# Define a function to load the model and predict on new data

def predict_high_balance(new_data_df, model_path=model_filename):
    """
    Loads the saved pipeline (preprocessor + model) and predicts
    the 'is_high_balance' status for new customer data.

    Args:
        new_data_df (pd.DataFrame): DataFrame containing new customer data
                                     with the same features used for training.
        model_path (str): Path to the saved .joblib model pipeline file.

    Returns:
        np.ndarray: Array of predictions (1 for high balance, 0 for low balance).
                    Returns None if an error occurs.
    """
    try:
        # Load the pipeline
        loaded_pipeline = joblib.load(model_path)

        # Ensure the input dataframe has the correct columns in the correct order
        # (important if the original training features were selected explicitly)
        # The preprocessor within the pipeline expects columns it was trained on.
        # Reorder/select columns based on the features used during training.
        training_features = loaded_pipeline.named_steps['preprocessor'].feature_names_in_
        new_data_df_ordered = new_data_df[training_features]


        # Make predictions
        # The pipeline handles preprocessing and prediction
        predictions = loaded_pipeline.predict(new_data_df_ordered)
        probabilities = None
        if hasattr(loaded_pipeline, "predict_proba"):
             # Get probabilities for the positive class (high balance)
             probabilities = loaded_pipeline.predict_proba(new_data_df_ordered)[:, 1]


        return predictions, probabilities

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None, None
    except NotFittedError:
         print("Error: The loaded pipeline appears not to be fitted. Please ensure the model was trained correctly.")
         return None, None
    except KeyError as e:
        print(f"Error: Missing expected column in input data: {e}")
        print(f"Expected columns: {list(training_features)}")
        return None, None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None, None

# --- Example Usage of Prediction Function (commented out) ---
# print("\n--- Prediction Example ---")
# # Create some sample new data (should match the structure of X)
# sample_data = {
#     'age': [45, 62, 31],
#     'job': ['management', 'retired', 'technician'],
#     'marital': ['married', 'divorced', 'single'],
#     'education': ['tertiary', 'primary', 'secondary'],
#     'default': ['no', 'no', 'no'],
#     'housing': ['yes', 'no', 'yes'],
#     'loan': ['no', 'no', 'yes'],
#     'contact': ['cellular', 'unknown', 'telephone'],
#     'day': [15, 2, 28],
#     'month': ['jun', 'sep', 'feb'],
#     'campaign': [1, 3, 1],
#     'pdays': [-1, 180, -1],
#     'previous': [0, 1, 0],
#     'poutcome': ['unknown', 'success', 'unknown']
# }
# new_customers_df = pd.DataFrame(sample_data)

# # Make predictions
# predictions, probabilities = predict_high_balance(new_customers_df)

# if predictions is not None:
#     print("Prediction Results:")
#     new_customers_df['predicted_is_high_balance'] = predictions
#     if probabilities is not None:
#         new_customers_df['predicted_high_balance_probability'] = probabilities
#     print(new_customers_df[['job', 'age', 'predicted_is_high_balance'] + (['predicted_high_balance_probability'] if probabilities is not None else [])])


print(json.dumps(best_model_results))  # Output JSON results