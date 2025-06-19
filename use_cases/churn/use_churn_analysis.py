# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 19:56:18 2025

@author: joze_
"""

# 1. Basic usage with synthetic data
from churn_analysis import ChurnAnalysisPipeline, SyntheticDataGenerator

# Generate sample data
generator = SyntheticDataGenerator()
df = generator.generate_ecommerce_data(n_samples=5000)

file="D:/data_sets/Churn/data/ecom-user-churn-data.csv"
 
# df.to_csv(file, index=False)

# Run analysis
pipeline = ChurnAnalysisPipeline()
results = pipeline.analyze_file(file,'target_class')

# 2. Analyze your own data
# results = pipeline.analyze_file(file, target_col='churn')

# 3. Access results
print(f"Churn Rate: {results['churn_rate']:.1f}%")
print(f"Model Accuracy: {results['model_accuracy']:.1f}%")
print(f"Revenue at Risk: ${results['revenue_at_risk']:,.2f}")