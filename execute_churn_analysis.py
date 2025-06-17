# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 11:19:10 2025

@author: joze_
"""

# Save the code as churn_analysis.py
from churn_analysis import ChurnAnalysisPipeline

# Create pipeline
pipeline = ChurnAnalysisPipeline()

# Analyze any CSV file
results = pipeline.analyze_file('your_data.csv')

# Results are automatically saved and visualized"""
Complete Example: How to Use the Churn Analysis System
======================================================
"""

import pandas as pd
import numpy as np
from churn_analysis import (
    ChurnAnalysisPipeline, 
    SyntheticDataGenerator,
    ChurnColumnMatrix,
    StatisticalEDA
)

# Example 1: Quick Start with Synthetic Data
# ==========================================
def example_synthetic_data():
    """Example using synthetic e-commerce data"""
    print("=" * 60)
    print("Example 1: Analyzing Synthetic E-commerce Data")
    print("=" * 60)
    
    # Generate synthetic data
    generator = SyntheticDataGenerator()
    df = generator.generate_shopify_data(n_samples=1000)
    
    # Save to CSV
    df.to_csv('example_shopify_data.csv', index=False)
    print(f"Generated synthetic data: {df.shape}")
    print(f"Columns: {', '.join(df.columns)}")
    print(f"\nFirst few rows:")
    print(df.head())
    
    # Run analysis
    pipeline = ChurnAnalysisPipeline()
    results = pipeline.analyze_file('example_shopify_data.csv')
    
    # Print results summary
    print("\n" + "=" * 40)
    print("ANALYSIS RESULTS")
    print("=" * 40)
    print(f"Target column identified: {results['column_analysis']['classification']['target']}")
    print(f"Number of features: {len(results['column_analysis']['classification']['features'])}")
    print(f"Model accuracy: {results['model_results'].get('accuracy', 0) * 100:.2f}%")
    
    return results


# Example 2: Analyzing Your Own Data
# ==================================
def example_custom_data(filepath):
    """Example using your own CSV file"""
    print("\n" + "=" * 60)
    print(f"Example 2: Analyzing {filepath}")
    print("=" * 60)
    
    # Load and preview data
    df = pd.read_csv(filepath)
    print(f"Data shape: {df.shape}")
    print(f"Columns: {', '.join(df.columns[:10])}...")  # Show first 10 columns
    
    # Initialize pipeline
    pipeline = ChurnAnalysisPipeline()
    
    # Run analysis
    results = pipeline.analyze_file(filepath)
    
    return results


# Example 3: Step-by-Step Analysis
# ================================
def example_step_by_step():
    """Example showing individual components"""
    print("\n" + "=" * 60)
    print("Example 3: Step-by-Step Analysis")
    print("=" * 60)
    
    # Create sample data
    np.random.seed(42)
    df = pd.DataFrame({
        'customer_id': [f'CUST{i:04d}' for i in range(100)],
        'tenure_days': np.random.randint(1, 1000, 100),
        'monthly_spend': np.random.uniform(10, 500, 100),
        'support_calls': np.random.poisson(2, 100),
        'last_login_days': np.random.exponential(10, 100),
        'subscription_type': np.random.choice(['Basic', 'Premium'], 100),
        'churn': np.random.choice([0, 1], 100, p=[0.8, 0.2])
    })
    
    print("Sample data created:")
    print(df.head())
    
    # Step 1: Column Classification
    print("\n1. Classifying columns...")
    matrix = ChurnColumnMatrix()
    classification = matrix.classify_columns(df)
    print(f"   Target: {classification.target_column}")
    print(f"   Must-have: {classification.must_have}")
    print(f"   Good-to-have: {classification.good_to_have}")
    
    # Step 2: Statistical Analysis
    print("\n2. Statistical Analysis...")
    eda = StatisticalEDA(df, classification.target_column)
    summary = eda.generate_summary()
    print(f"   Churn rate: {summary['target_distribution']['percentage']}")
    print(f"   Missing data: {summary['missing_data']['total_missing_pct']:.2f}%")
    
    # Step 3: Generate visualizations
    print("\n3. Generating visualizations...")
    eda.generate_visualizations('example_plots/')
    print("   Plots saved to example_plots/")
    
    return df, classification, summary


# Example 4: Working with Column Info (fixing your error)
# =======================================================
def example_column_analysis():
    """Example showing how to analyze columns properly"""
    print("\n" + "=" * 60)
    print("Example 4: Column Analysis")
    print("=" * 60)
    
    # Create sample dataframe
    df = pd.DataFrame({
        'customer_id': ['C001', 'C002', 'C003', 'C004', 'C005'],
        'age': [25, 35, 45, 28, 52],
        'tenure_months': [12, 24, 6, 18, 36],
        'monthly_charges': [50.5, 75.0, 120.0, 65.5, 95.0],
        'total_charges': [606, 1800, 720, 1179, 3420],
        'churn': [0, 0, 1, 0, 1]
    })
    
    print("Sample DataFrame:")
    print(df)
    
    # Now we can use the column analysis code
    sample_size = 3
    sample_data = df.head(sample_size).to_dict()
    column_info = {
        col: {
            'dtype': str(df[col].dtype),
            'unique_values': df[col].nunique(),
            'null_count': df[col].isnull().sum(),
            'sample_values': df[col].dropna().head(3).tolist()
        }
        for col in df.columns
    }
    
    print("\nColumn Information:")
    for col, info in column_info.items():
        print(f"\n{col}:")
        print(f"  Type: {info['dtype']}")
        print(f"  Unique values: {info['unique_values']}")
        print(f"  Missing: {info['null_count']}")
        print(f"  Samples: {info['sample_values']}")
    
    return df, column_info


# Example 5: Handling Different File Types
# ========================================
def example_analyze_multiple_files():
    """Example analyzing multiple file types"""
    print("\n" + "=" * 60)
    print("Example 5: Analyzing Multiple Files")
    print("=" * 60)
    
    # List of files to analyze (use your actual file paths)
    files_to_analyze = [
        'internet_service_churn.csv',
        'WA_FnUseC_HREmployeeAttrition.csv',
        'Churn_Modelling.csv'
    ]
    
    pipeline = ChurnAnalysisPipeline()
    all_results = {}
    
    for filepath in files_to_analyze:
        try:
            print(f"\nAnalyzing {filepath}...")
            results = pipeline.analyze_file(filepath)
            all_results[filepath] = results
            
            # Print summary
            target = results['column_analysis']['classification']['target']
            accuracy = results['model_results'].get('accuracy', 0) * 100
            print(f"  Target: {target}")
            print(f"  Accuracy: {accuracy:.2f}%")
            
        except FileNotFoundError:
            print(f"  File not found: {filepath}")
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    return all_results


# Example 6: Custom Analysis with Gemini API
# ==========================================
def example_with_gemini(api_key):
    """Example using Gemini API for intelligent column analysis"""
    print("\n" + "=" * 60)
    print("Example 6: Analysis with Gemini AI")
    print("=" * 60)
    
    # Generate test data
    df = pd.DataFrame({
        'user_id': range(1000),
        'days_since_signup': np.random.randint(1, 365, 1000),
        'num_logins_last_month': np.random.poisson(10, 1000),
        'subscription_cancelled': np.random.choice([True, False], 1000, p=[0.2, 0.8]),
        'customer_satisfaction_score': np.random.uniform(1, 5, 1000),
        'total_revenue': np.random.exponential(100, 1000)
    })
    df.to_csv('gemini_test_data.csv', index=False)
    
    # Initialize with Gemini
    pipeline = ChurnAnalysisPipeline(gemini_api_key=api_key)
    results = pipeline.analyze_file('gemini_test_data.csv')
    
    print("\nGemini Analysis Results:")
    gemini_results = results['column_analysis']['gemini_analysis']
    print(f"Target identified: {gemini_results.get('target_column')}")
    print(f"Reasoning: {gemini_results.get('reasoning')}")
    
    return results


# Main execution
if __name__ == "__main__":
    # Run examples
    
    # Example 1: Synthetic data
    results1 = example_synthetic_data()
    
    # Example 3: Step by step
    df, classification, summary = example_step_by_step()
    
    # Example 4: Column analysis (fixes your error)
    df_sample, column_info = example_column_analysis()
    
    # To analyze your specific files, uncomment and modify:
    # results = example_custom_data('path/to/your/file.csv')
    
    # To use with Gemini API:
    # results = example_with_gemini('your-gemini-api-key')
    
    print("\n" + "=" * 60)
    print("✅ All examples completed!")
    print("Check the generated files:")
    print("  - example_shopify_data.csv")
    print("  - churn_analysis_results.json")
    print("  - churn_analysis_report.html")
    print("  - example_plots/ (folder with visualizations)")
    print("=" * 60)