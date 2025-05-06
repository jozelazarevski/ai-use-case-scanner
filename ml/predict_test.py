"""# Classification prediction
from predict_model_clasification import predict_classification
from read_file import read_data_flexible

model_path = "../trained_models/best_model_y.joblib"
filepath='../uploads/bank-full.csv'
input_data = read_data_flexible(filepath)
prediction_results = predict_classification(model_path, input_data)
print(prediction_results)




# Regression prediction
# Classification prediction
from predict_model_regression import predict_regression
from read_file import read_data_flexible

model_path = "../trained_models/best_model_y.joblib"
filepath='../uploads/bank-full.csv'
input_data = read_data_flexible(filepath)
prediction_results = predict_regression(model_path, input_data)
print(prediction_results)
"""