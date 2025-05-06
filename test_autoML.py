from automl_script import AutoMLTrainer

# Create trainer instance
trainer = AutoMLTrainer(
    data_path='data_sets/website_visits.csv',
    target_column='customer_id',
    task_type='auto',  # or 'classification', 'regression', 'clustering'
    time_budget=120,  # seconds
    output_dir='models'
)

# Run the full pipeline
model = trainer.run()

# Or run individual steps manually
trainer.load_data()
trainer.detect_task_type()
trainer.analyze_data()
trainer.preprocess_data()

# Task-specific training
if trainer.task_detected == 'classification':
    trainer.train_classification()
    trainer.evaluate_classification()
elif trainer.task_detected == 'regression':
    trainer.train_regression()
    trainer.evaluate_regression()
else:  # clustering
    trainer.train_clustering()
    trainer.evaluate_clustering()

# Save model
trainer.save_model()

# Use the model for predictions on new data
# Example for classification/regression
new_data = pd.read_csv('bank-full.csv')
predictions = model.predict(new_data)