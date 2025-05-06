from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""
Use Case: Sales Campaign Optimization

Description: Analyze which factors (e.g., contact type, campaign frequency) influence the success of marketing campaigns to improve their effectiveness.
Target Variable: y (yes/no) or duration (to analyze engagement)
"""

# Load the dataset
df = pd.read_csv('C:/data_sets/bank-full.csv', delimiter=';')

# Preprocessing: One-hot encode categorical variables
categorical_features = df.select_dtypes(include=['object']).columns.tolist()
categorical_features.remove('y')  # Exclude the target variable
transformer = make_column_transformer(
    (OneHotEncoder(handle_unknown='ignore'), categorical_features),
    remainder='passthrough'  # Keep numerical columns
)

# Split data into training and testing sets
X = df.drop('y', axis=1)
y = df['y'].map({'yes': 1, 'no': 0})  # Convert target variable to binary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply the preprocessing transformer
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

# Train a logistic regression model
model = LogisticRegression(solver='liblinear', random_state=42)
model.fit(X_train_transformed, y_train)

# Make predictions
y_pred = model.predict(X_test_transformed)
y_proba = model.predict_proba(X_test_transformed)[:, 1]  # Probabilities for the positive class

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
