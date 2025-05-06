#!/usr/bin/env python3
"""
AutoML Demo Script - Example usage for classification, regression, and clustering tasks

This script demonstrates how to use the AutoMLTrainer class for:
1. Classification
2. Regression
3. Clustering

It also shows how to load custom datasets and handle various data formats.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_diabetes, load_wine, make_blobs
from automl_script import AutoMLTrainer  # Import the AutoMLTrainer class

def demo_classification():
    """
    Demonstrate AutoML for classification using the Iris dataset.
    """
    print("\n\n========== CLASSIFICATION DEMO ==========\n")
    
    # Load Iris dataset
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['target'] = iris.target
    
    # Map numeric target to class names for better interpretability
    target_names = {i: name for i, name in enumerate(iris.target_names)}
    iris_df['species'] = iris_df['target'].map(target_names)
    
    # Create and run AutoML trainer
    trainer = AutoMLTrainer(
        target_column='species',  # Use the string class names
        task_type='classification',
        time_budget=60,  # 1 minute for this demo
        output_dir='models/classification_demo'
    )
    
    # Load data and run
    trainer.load_data(data=iris_df)
    model = trainer.run()
    
    print("Classification demo completed successfully!")
    return model

def demo_regression():
    """
    Demonstrate AutoML for regression using the Diabetes dataset.
    """
    print("\n\n========== REGRESSION DEMO ==========\n")
    
    # Load Diabetes dataset
    diabetes = load_diabetes()
    diabetes_df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
    diabetes_df['target'] = diabetes.target
    
    # Create and run AutoML trainer
    trainer = AutoMLTrainer(
        target_column='target',
        task_type='regression',
        time_budget=60,  # 1 minute for this demo
        output_dir='models/regression_demo'
    )
    
    # Load data and run
    trainer.load_data(data=diabetes_df)
    model = trainer.run()
    
    print("Regression demo completed successfully!")
    return model

def demo_clustering():
    """
    Demonstrate AutoML for clustering using the Wine dataset.
    """
    print("\n\n========== CLUSTERING DEMO ==========\n")
    
    # Load Wine dataset (but exclude the target for unsupervised learning)
    wine = load_wine()
    wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    
    # Create and run AutoML trainer
    trainer = AutoMLTrainer(
        task_type='clustering',
        time_budget=60,  # 1 minute for this demo
        output_dir='models/clustering_demo'
    )
    
    # Load data and run
    trainer.load_data(data=wine_df)
    model = trainer.run()
    
    print("Clustering demo completed successfully!")
    return model

def demo_custom_dataset(csv_path, target_column=None, task_type='auto'):
    """
    Demonstrate AutoML using a custom CSV dataset.
    
    Args:
        csv_path (str): Path to CSV file
        target_column (str): Name of target column (None for clustering)
        task_type (str): Type of task ('auto', 'classification', 'regression', 'clustering')
    """
    print(f"\n\n========== CUSTOM DATASET DEMO ({task_type}) ==========\n")
    
    # Create and run AutoML trainer
    trainer = AutoMLTrainer(
        data_path=csv_path,
        target_column=target_column,
        task_type=task_type,
        time_budget=120,  # 2 minutes for this demo
        output_dir=f'models/custom_{task_type}_demo'
    )
    
    # Run the pipeline
    model = trainer.run()
    
    print(f"Custom dataset ({task_type}) demo completed successfully!")
    return model

def generate_synthetic_dataset():
    """
    Generate a synthetic dataset for demo purposes.
    
    Returns:
        str: Path to the generated CSV file
    """
    print("\n\n========== GENERATING SYNTHETIC DATASET ==========\n")
    
    # Create features
    np.random.seed(42)
    n_samples = 1000
    
    # Numeric features
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.normal(5, 2, n_samples)
    feature3 = np.random.uniform(-10, 10, n_samples)
    
    # Categorical features
    categories = ['A', 'B', 'C', 'D']
    cat_feature1 = np.random.choice(categories, n_samples)
    cat_feature2 = np.random.choice(['Yes', 'No', 'Maybe'], n_samples)
    
    # Create target (classification)
    # Target is influenced by features
    logits = 0.5 * feature1 - 0.2 * feature2 + 0.1 * feature3 + np.random.normal(0, 0.5, n_samples)
    probabilities = 1 / (1 + np.exp(-logits))
    classification_target = (probabilities > 0.5).astype(str)
    
    # Create target (regression)
    regression_target = 2 * feature1 + 0.5 * feature2 - feature3 + np.random.normal(0, 2, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2, 
        'feature3': feature3,
        'categorical1': cat_feature1,
        'categorical2': cat_feature2,
        'classification_target': classification_target,
        'regression_target': regression_target
    })
    
    # Add some missing values
    mask = np.random.random(data.shape) < 0.05
    data = data.mask(mask)
    
    # Save to CSV
    csv_path = 'synthetic_data.csv'
    data.to_csv(csv_path, index=False)
    
    print(f"Synthetic dataset created with {n_samples} samples and saved to {csv_path}")
    print(f"Dataset shape: {data.shape}")
    print(f"First few rows:\n{data.head()}")
    
    return csv_path

def main():
    """
    Run all demos.
    """
    print("===== AUTOML COMPREHENSIVE DEMO =====")
    
    # Uncomment the demos you want to run
    
    # 1. Classification demo with Iris dataset
    demo_classification()
    
    # 2. Regression demo with Diabetes dataset
    demo_regression()
    
    # 3. Clustering demo with Wine dataset
    demo_clustering()
    
    # 4. Custom dataset demos with synthetic data
    csv_path = generate_synthetic_dataset()
    demo_custom_dataset(csv_path, 'classification_target', 'classification')
    demo_custom_dataset(csv_path, 'regression_target', 'regression')
    demo_custom_dataset(csv_path, None, 'clustering')
    
    print("\n===== ALL DEMOS COMPLETED SUCCESSFULLY =====")


if __name__ == "__main__":
    main()
