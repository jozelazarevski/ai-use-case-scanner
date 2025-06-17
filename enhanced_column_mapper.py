# enhanced_column_mapper.py
"""
Enhanced Column mapping and validation module with comprehensive EDA capabilities
Handles intelligent column identification, validation, and data analysis
"""

import json
import re
import os
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
import traceback
from scipy import stats
import math

# Configure logger
logger = logging.getLogger(__name__)

def clean_json_data(data):
    """
    Recursively clean data to remove NaN values and make it JSON-serializable
    """
    if isinstance(data, dict):
        return {key: clean_json_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data) or math.isinf(data):
            return None
        return data
    elif isinstance(data, np.floating):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, np.integer):
        return int(data)
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif pd.isna(data):  # Handle pandas NA/NaN
        return None
    else:
        return data

def safe_json_dumps(data):
    """
    Safely convert data to JSON string, handling NaN values
    """
    cleaned_data = clean_json_data(data)
    return json.dumps(cleaned_data, ensure_ascii=False, default=str)

def convert_numpy_types(obj):
    """Convert numpy types to JSON serializable Python types"""
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_numpy_types(item) for item in obj)
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

class EnhancedColumnMapper:
    """Enhanced column mapper with comprehensive EDA capabilities"""
    
    def __init__(self, gemini_model=None):
        """Initialize the Enhanced ColumnMapper"""
        self.gemini_model = gemini_model
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize the column mappings database tables in PostgreSQL"""
        try:
            # Import get_db_connection from the correct location
            from utils.user_auth import get_db_connection
            
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Enhanced column_mappings table with EDA results
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS column_mappings (
                            id VARCHAR(255) PRIMARY KEY DEFAULT gen_random_uuid()::text,
                            user_id VARCHAR(255) NOT NULL,
                            filename VARCHAR(255) NOT NULL,
                            mappings JSONB NOT NULL,
                            eda_results JSONB,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(user_id, filename)
                        )
                    ''')
                    
                    # Add eda_results column if it doesn't exist
                    cursor.execute('''
                        ALTER TABLE column_mappings 
                        ADD COLUMN IF NOT EXISTS eda_results JSONB
                    ''')
                    
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Error initializing database tables: {str(e)}")
    
    def perform_comprehensive_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive exploratory data analysis
        """
        try:
            eda_results = {
                'basic_stats': self._get_basic_statistics(df),
                'data_quality': self._assess_data_quality(df),
                'statistical_summary': self._get_statistical_summary(df),
                'correlations': self._analyze_correlations(df),
                'distributions': self._analyze_distributions(df),
                'patterns': self._detect_patterns(df),
                'business_insights': self._generate_business_insights(df),
                'anomalies': self._detect_anomalies(df),
                'relationships': self._analyze_relationships(df)
            }
            
            # Clean the results to ensure JSON serialization
            return convert_numpy_types(eda_results)
        except Exception as e:
            self.logger.error(f"Error in perform_comprehensive_eda: {str(e)}")
            return {
                'basic_stats': {'n_rows': len(df), 'n_columns': len(df.columns)},
                'data_quality': {'quality_score': 0},
                'error': str(e)
            }
    
    def _get_basic_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic statistics about the dataset"""
        try:
            return {
                'n_rows': int(len(df)),
                'n_columns': int(len(df.columns)),
                'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
                'duplicates': int(df.duplicated().sum()),
                'duplicate_percentage': float((df.duplicated().sum() / len(df)) * 100) if len(df) > 0 else 0
            }
        except Exception as e:
            self.logger.error(f"Error in _get_basic_statistics: {str(e)}")
            return {'n_rows': len(df), 'n_columns': len(df.columns)}
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess data quality metrics"""
        try:
            total_cells = df.shape[0] * df.shape[1]
            total_missing = df.isnull().sum().sum()
            
            # Missing data by column
            missing_by_column = {}
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    missing_by_column[col] = float((missing_count / len(df)) * 100)
            
            # Data type counts
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            datetime_cols = df.select_dtypes(include=['datetime']).columns
            
            # Identify potential issues
            issues = []
            if total_cells > 0 and total_missing / total_cells > 0.2:
                issues.append("High missing data rate (>20%)")
            
            if df.duplicated().sum() > len(df) * 0.05:
                issues.append("Significant duplicate records (>5%)")
            
            # Check for high cardinality categorical columns
            for col in categorical_cols:
                if df[col].nunique() > len(df) * 0.5:
                    issues.append(f"High cardinality in {col}")
            
            quality_score = self._calculate_quality_score(df)
            
            return {
                'missing_percentage': float((total_missing / total_cells) * 100) if total_cells > 0 else 0,
                'missing_by_column': missing_by_column,
                'completeness_score': float(100 - ((total_missing / total_cells) * 100)) if total_cells > 0 else 100,
                'numeric_count': int(len(numeric_cols)),
                'categorical_count': int(len(categorical_cols)),
                'date_count': int(len(datetime_cols)),
                'text_count': int(len(df.columns) - len(numeric_cols) - len(categorical_cols) - len(datetime_cols)),
                'quality_issues': issues,
                'quality_score': float(quality_score)
            }
        except Exception as e:
            self.logger.error(f"Error in _assess_data_quality: {str(e)}")
            return {'quality_score': 0, 'missing_percentage': 0}
    
    def _calculate_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score"""
        try:
            scores = []
            
            # Completeness score
            completeness = 100 - ((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100)
            scores.append(completeness)
            
            # Uniqueness score (inverse of duplicate percentage)
            uniqueness = 100 - ((df.duplicated().sum() / len(df)) * 100) if len(df) > 0 else 100
            scores.append(uniqueness)
            
            # Consistency score (check for mixed data types, outliers)
            consistency = 100  # Start with perfect score
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if len(df[col].dropna()) > 0:
                    # Check for outliers using IQR
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    outliers = ((df[col] < (Q1 - 3 * IQR)) | (df[col] > (Q3 + 3 * IQR))).sum()
                    if outliers > len(df) * 0.05:  # More than 5% outliers
                        consistency -= 10
            scores.append(max(0, consistency))
            
            return float(np.mean(scores))
        except Exception as e:
            self.logger.error(f"Error in _calculate_quality_score: {str(e)}")
            return 50.0
    
    def _get_statistical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get detailed statistical summary"""
        summary = {}
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    summary[col] = {
                        'mean': float(col_data.mean()),
                        'median': float(col_data.median()),
                        'std': float(col_data.std()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'q1': float(col_data.quantile(0.25)),
                        'q3': float(col_data.quantile(0.75)),
                        'skewness': float(col_data.skew()),
                        'kurtosis': float(col_data.kurtosis()),
                        'cv': float(col_data.std() / col_data.mean()) if col_data.mean() != 0 else 0
                    }
        except Exception as e:
            self.logger.error(f"Error in _get_statistical_summary: {str(e)}")
        
        return summary
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numeric features"""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {}
            
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Convert to dict format and clean NaN values
            corr_dict = {}
            for col in corr_matrix.columns:
                corr_dict[col] = {}
                for row in corr_matrix.index:
                    value = corr_matrix.loc[row, col]
                    corr_dict[col][row] = float(value) if not pd.isna(value) else 0.0
            
            # Find significant correlations
            significant_correlations = []
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:  # Upper triangle only
                        corr_value = corr_matrix.iloc[i, j]
                        if not pd.isna(corr_value) and abs(corr_value) > 0.5:
                            significant_correlations.append({
                                'feature1': col1,
                                'feature2': col2,
                                'correlation': float(corr_value),
                                'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                            })
            
            return {
                'correlation_matrix': corr_dict,
                'significant_correlations': significant_correlations,
                'highly_correlated_features': [c for c in significant_correlations if abs(c['correlation']) > 0.8]
            }
        except Exception as e:
            self.logger.error(f"Error in _analyze_correlations: {str(e)}")
            return {}
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric features"""
        distributions = {}
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols[:10]:  # Limit to first 10 columns
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # Create histogram data
                    hist, bin_edges = np.histogram(col_data, bins=20)
                    
                    distributions[col] = {
                        'bins': [float(x) for x in bin_edges[:-1]],
                        'counts': [int(x) for x in hist],
                        'type': self._identify_distribution_type(col_data)
                    }
        except Exception as e:
            self.logger.error(f"Error in _analyze_distributions: {str(e)}")
        
        return distributions
    
    def _identify_distribution_type(self, data: pd.Series) -> str:
        """Identify the type of distribution"""
        try:
            skewness = float(data.skew())
            kurtosis = float(data.kurtosis())
            
            if abs(skewness) < 0.5 and abs(kurtosis) < 0.5:
                return "normal"
            elif skewness > 1:
                return "right-skewed"
            elif skewness < -1:
                return "left-skewed"
            elif kurtosis > 1:
                return "leptokurtic"
            elif kurtosis < -1:
                return "platykurtic"
            else:
                return "unknown"
        except:
            return "unknown"
    
    def _detect_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns in the data"""
        patterns = {
            'seasonal_patterns': [],
            'trends': [],
            'cyclic_patterns': [],
            'has_temporal_data': False
        }
        
        try:
            # Check for date columns
            date_cols = df.select_dtypes(include=['datetime']).columns
            if len(date_cols) > 0:
                patterns['has_temporal_data'] = True
                
                # Analyze temporal patterns
                for date_col in date_cols:
                    try:
                        # Check for seasonality
                        if hasattr(df[date_col].dt, 'month'):
                            month_counts = df[date_col].dt.month.value_counts()
                            if month_counts.std() > 10:
                                patterns['seasonal_patterns'].append({
                                    'column': date_col,
                                    'type': 'monthly_variation'
                                })
                    except:
                        pass
        except Exception as e:
            self.logger.error(f"Error in _detect_patterns: {str(e)}")
        
        return patterns
    
    def _generate_business_insights(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate business insights from the data"""
        insights = []
        
        try:
            # Check for customer-related data
            customer_cols = [col for col in df.columns if any(term in col.lower() 
                            for term in ['customer', 'client', 'user', 'account'])]
            
            if customer_cols:
                insights.append({
                    'type': 'customer_analytics',
                    'title': 'Customer Data Available',
                    'description': f'Found {len(customer_cols)} customer-related columns',
                    'recommendations': [
                        'Customer segmentation analysis',
                        'Customer lifetime value prediction',
                        'Churn prediction modeling'
                    ]
                })
            
            # Check for financial data
            financial_cols = [col for col in df.columns if any(term in col.lower() 
                             for term in ['revenue', 'cost', 'price', 'amount', 'payment', 'sales'])]
            
            if financial_cols:
                insights.append({
                    'type': 'financial_analytics',
                    'title': 'Financial Data Detected',
                    'description': f'Found {len(financial_cols)} financial columns',
                    'recommendations': [
                        'Revenue forecasting',
                        'Cost optimization analysis',
                        'Profitability analysis'
                    ]
                })
            
            # Check for time series potential
            date_cols = df.select_dtypes(include=['datetime']).columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(date_cols) > 0 and len(numeric_cols) > 0:
                insights.append({
                    'type': 'time_series',
                    'title': 'Time Series Analysis Possible',
                    'description': 'Dataset contains both temporal and numeric data',
                    'recommendations': [
                        'Trend analysis',
                        'Seasonal decomposition',
                        'Forecasting models'
                    ]
                })
            
            # Check for categorical analysis potential
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            if len(categorical_cols) > 2:
                insights.append({
                    'type': 'categorical_analysis',
                    'title': 'Rich Categorical Data',
                    'description': f'Found {len(categorical_cols)} categorical features',
                    'recommendations': [
                        'Market basket analysis',
                        'Association rule mining',
                        'Category performance comparison'
                    ]
                })
        except Exception as e:
            self.logger.error(f"Error in _generate_business_insights: {str(e)}")
        
        return insights
    
    def _detect_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect anomalies in the dataset"""
        anomalies = {
            'outliers': {},
            'unusual_patterns': [],
            'data_quality_issues': []
        }
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    # IQR method for outlier detection
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
                    
                    if len(outliers) > 0:
                        anomalies['outliers'][col] = {
                            'count': int(len(outliers)),
                            'percentage': float((len(outliers) / len(col_data)) * 100),
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound)
                        }
                    
                    # Check for unusual patterns
                    if col_data.nunique() == 1:
                        anomalies['unusual_patterns'].append({
                            'column': col,
                            'issue': 'constant_value',
                            'description': f'Column {col} has only one unique value'
                        })
                    
                    # Check for potential data entry errors
                    if col_data.std() == 0 and len(col_data) > 1:
                        anomalies['data_quality_issues'].append({
                            'column': col,
                            'issue': 'zero_variance',
                            'description': f'Column {col} has zero variance'
                        })
        except Exception as e:
            self.logger.error(f"Error in _detect_anomalies: {str(e)}")
        
        return anomalies
    
    def _analyze_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze relationships between features"""
        relationships = {
            'feature_interactions': [],
            'dependencies': [],
            'redundant_features': []
        }
        
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Check for potential redundant features (very high correlation)
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j:
                            corr_value = corr_matrix.iloc[i, j]
                            if not pd.isna(corr_value) and abs(corr_value) > 0.95:
                                relationships['redundant_features'].append({
                                    'feature1': col1,
                                    'feature2': col2,
                                    'correlation': float(corr_value)
                                })
            
            # Check for potential categorical-numeric relationships
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            for cat_col in categorical_cols[:5]:  # Limit to prevent long computation
                if df[cat_col].nunique() < 20:  # Only for low cardinality
                    for num_col in numeric_cols[:5]:
                        try:
                            # Simple ANOVA-style check
                            groups = df.groupby(cat_col)[num_col].agg(['mean', 'std', 'count'])
                            if len(groups) > 1 and groups['count'].min() > 5:
                                # Check if means vary significantly
                                mean_variance = groups['mean'].var()
                                if mean_variance > 0:
                                    relationships['feature_interactions'].append({
                                        'categorical': cat_col,
                                        'numeric': num_col,
                                        'type': 'categorical_numeric_relationship'
                                    })
                        except:
                            pass
        except Exception as e:
            self.logger.error(f"Error in _analyze_relationships: {str(e)}")
        
        return relationships
    
    def analyze_columns_with_eda(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Analyze DataFrame columns using Gemini AI with comprehensive EDA
        """
        # Perform EDA
        eda_results = self.perform_comprehensive_eda(df)
        
        # Get sample data for analysis
        sample_data = self._get_sample_data(df, n_samples=10)
        
        # Get data statistics
        data_stats = self._get_column_statistics(df)
        
        # Create enhanced prompt for Gemini
        prompt = self._create_enhanced_analysis_prompt(
            df.columns.tolist(), 
            sample_data, 
            data_stats, 
            filename,
            eda_results
        )
        
        try:
            # Call Gemini API
            if self.gemini_model:
                response = self.gemini_model.generate_content(prompt)
                
                # Parse response
                analysis = self._parse_gemini_response(response.text)
            else:
                # Fallback to basic analysis
                analysis = self._get_basic_analysis(df, filename)
            
            # Add additional metadata
            analysis['filename'] = filename
            analysis['total_rows'] = len(df)
            analysis['total_columns'] = len(df.columns)
            analysis['eda_results'] = eda_results
            
            # Add data quality insights
            analysis['data_quality'] = eda_results.get('data_quality', {})
            analysis['correlation_matrix'] = eda_results.get('correlations', {}).get('correlation_matrix', {})
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in Gemini analysis: {str(e)}")
            # Return enhanced basic analysis as fallback
            return self._get_enhanced_basic_analysis(df, filename, eda_results)
    
    def _create_enhanced_analysis_prompt(self, columns: List[str], sample_data: Dict, 
                                       stats: Dict, filename: str, 
                                       eda_results: Dict) -> str:
        """Create an enhanced prompt for Gemini with EDA insights"""
        
        # Summarize key EDA findings
        quality_score = eda_results.get('data_quality', {}).get('quality_score', 0)
        missing_percentage = eda_results.get('data_quality', {}).get('missing_percentage', 0)
        significant_correlations = eda_results.get('correlations', {}).get('significant_correlations', [])
        business_insights = eda_results.get('business_insights', [])
        
        eda_summary = f"""
EDA SUMMARY:
- Data Quality Score: {quality_score:.1f}%
- Missing Data: {missing_percentage:.1f}%
- Significant Correlations: {len(significant_correlations)}
- Business Insights Found: {len(business_insights)}
"""
        
        # Clean sample data and stats for JSON serialization
        sample_data_clean = convert_numpy_types(sample_data)
        stats_clean = convert_numpy_types(stats)
        correlations_clean = convert_numpy_types(significant_correlations[:5]) if significant_correlations else []
        
        prompt = f"""Analyze the following dataset columns and provide intelligent mapping suggestions with business context.

FILENAME: {filename}
{eda_summary}

COLUMNS AND SAMPLE DATA:
{json.dumps(sample_data_clean, indent=2)}

COLUMN STATISTICS:
{json.dumps(stats_clean, indent=2)}

KEY CORRELATIONS:
{json.dumps(correlations_clean, indent=2) if correlations_clean else "None found"}

Please analyze each column and provide:
1. A standardized, business-friendly column name
2. A detailed description of what the column likely represents
3. Whether it could be a target variable (and if so, what it might predict)
4. Confidence score (0-1) for your analysis
5. Business relevance and potential use cases
6. For target variables, provide comprehensive prediction possibilities

Consider the EDA results and correlations when making suggestions.

IMPORTANT: Return ONLY a valid JSON object with the following structure:
{{
  "columns": [
    {{
      "original_name": "original column name",
      "suggested_name": "meaningful business name",
      "data_type": "numeric/categorical/date/text/identifier/boolean",
      "description": "detailed description of what this column represents",
      "is_target": true/false,
      "target_meaning": "if is_target is true, explain what it predicts",
      "prediction_use_cases": ["use case 1", "use case 2", "use case 3"],
      "business_value": "explanation of business value",
      "model_type": "classification/regression",
      "confidence": 0.95,
      "reasoning": "brief explanation of your analysis"
    }}
  ],
  "dataset_summary": "Comprehensive description of the dataset with business context",
  "suggested_use_cases": ["use case 1", "use case 2", "use case 3"],
  "data_recommendations": ["recommendation 1", "recommendation 2"]
}}"""
        
        return prompt
    
    def _get_enhanced_basic_analysis(self, df: pd.DataFrame, filename: str, 
                                    eda_results: Dict) -> Dict[str, Any]:
        """Provide enhanced basic analysis without AI"""
        basic_analysis = self._get_basic_analysis(df, filename)
        
        # Enhance with EDA insights
        basic_analysis['eda_results'] = eda_results
        basic_analysis['data_quality'] = eda_results.get('data_quality', {})
        basic_analysis['correlation_matrix'] = eda_results.get('correlations', {}).get('correlation_matrix', {})
        
        # Add data-driven recommendations
        recommendations = []
        
        quality_score = eda_results.get('data_quality', {}).get('quality_score', 0)
        if quality_score > 80:
            recommendations.append("High data quality suitable for machine learning")
        else:
            recommendations.append("Consider data cleaning and preprocessing")
        
        if len(eda_results.get('business_insights', [])) > 0:
            recommendations.append("Rich business data with multiple analytics opportunities")
        
        basic_analysis['data_recommendations'] = recommendations
        
        return basic_analysis
    
    def _get_sample_data(self, df: pd.DataFrame, n_samples: int = 10) -> Dict[str, List]:
        """Get sample data from each column"""
        sample_data = {}
        
        for col in df.columns:
            # Get non-null samples
            non_null_data = df[col].dropna()
            
            if len(non_null_data) > 0:
                # Get random samples
                n = min(n_samples, len(non_null_data))
                samples = non_null_data.sample(n=n, random_state=42).tolist()
                
                # Convert to native Python types
                samples = [self._convert_to_native_type(s) for s in samples]
                sample_data[col] = samples
            else:
                sample_data[col] = []
        
        return sample_data
    
    def _convert_to_native_type(self, value):
        """Convert numpy/pandas types to native Python types"""
        if pd.isna(value):
            return None
        elif isinstance(value, (np.int64, np.int32, np.int16)):
            return int(value)
        elif isinstance(value, (np.float64, np.float32)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        else:
            return str(value)
    
    def _get_column_statistics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Get statistical information about each column"""
        stats = {}
        
        for col in df.columns:
            col_stats = {
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'unique_count': int(df[col].nunique()),
                'dtype': str(df[col].dtype)
            }
            
            # Add numeric statistics if applicable
            if pd.api.types.is_numeric_dtype(df[col]):
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    col_stats.update({
                        'mean': float(col_data.mean()),
                        'min': float(col_data.min()),
                        'max': float(col_data.max()),
                        'std': float(col_data.std())
                    })
                else:
                    col_stats.update({
                        'mean': None,
                        'min': None,
                        'max': None,
                        'std': None
                    })
            
            # Check if it might be categorical
            if df[col].nunique() < 20 and len(df) > 50:
                col_stats['possible_categorical'] = True
                value_counts = df[col].value_counts().head(10)
                col_stats['value_counts'] = {str(k): int(v) for k, v in value_counts.to_dict().items()}
            
            stats[col] = col_stats
        
        return stats
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate Gemini's response"""
        try:
            # Remove any markdown formatting
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```'):
                cleaned_text = re.sub(r'^```.*?\n', '', cleaned_text)
                cleaned_text = re.sub(r'\n```', '', cleaned_text)
            
            # Parse JSON
            analysis = json.loads(cleaned_text)
            
            # Validate structure
            if 'columns' not in analysis:
                raise ValueError("Missing 'columns' in response")
            
            return analysis
            
        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"Error parsing Gemini response: {str(e)}")
            # Return a basic structure
            return {
                'columns': [],
                'dataset_summary': 'Unable to fully analyze dataset',
                'error': 'Failed to parse AI response'
            }
    
    def _get_basic_analysis(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Provide basic analysis without AI"""
        columns = []
        
        for col in df.columns:
            # Basic heuristics for column types
            if pd.api.types.is_numeric_dtype(df[col]):
                data_type = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                data_type = 'date'
            elif df[col].nunique() < 20:
                data_type = 'categorical'
            else:
                data_type = 'text'
            
            # Enhanced target variable detection
            is_target = False
            target_meaning = None
            prediction_use_cases = []
            business_value = None
            model_type = None
            
            col_lower = col.lower()
            
            # Common target variable patterns
            target_patterns = {
                'churn': {
                    'meaning': 'Customer churn prediction - whether a customer will stop using the service',
                    'use_cases': [
                        'Identify at-risk customers for retention campaigns',
                        'Calculate customer lifetime value predictions',
                        'Optimize customer service resource allocation'
                    ],
                    'business_value': 'Reducing churn by even 5% can increase profits by 25-95%',
                    'model_type': 'classification'
                },
                'fraud': {
                    'meaning': 'Fraud detection - identify potentially fraudulent transactions or activities',
                    'use_cases': [
                        'Real-time transaction fraud detection',
                        'Account takeover prevention',
                        'Risk-based authentication decisions'
                    ],
                    'business_value': 'Prevent financial losses and protect customer trust',
                    'model_type': 'classification'
                },
                'sale': {
                    'meaning': 'Sales prediction - forecast sales amount or probability of sale',
                    'use_cases': [
                        'Revenue forecasting and planning',
                        'Inventory optimization',
                        'Sales team performance prediction'
                    ],
                    'business_value': 'Improve revenue predictability and resource allocation',
                    'model_type': 'regression' if data_type == 'numeric' else 'classification'
                },
                'price': {
                    'meaning': 'Price prediction - estimate optimal pricing or future price movements',
                    'use_cases': [
                        'Dynamic pricing optimization',
                        'Competitive price monitoring',
                        'Demand-based pricing strategies'
                    ],
                    'business_value': 'Maximize revenue through optimal pricing strategies',
                    'model_type': 'regression'
                }
            }
            
            # Check for target patterns
            for pattern, info in target_patterns.items():
                if pattern in col_lower:
                    is_target = True
                    target_meaning = info['meaning']
                    prediction_use_cases = info['use_cases']
                    business_value = info['business_value']
                    model_type = info['model_type']
                    break
            
            # Check for generic target names
            if not is_target and col_lower in ['target', 'label', 'y', 'outcome', 'result', 'class']:
                is_target = True
                if data_type == 'categorical' or (data_type == 'numeric' and df[col].nunique() < 10):
                    target_meaning = f'Classification target with {df[col].nunique()} classes'
                    model_type = 'classification'
                else:
                    target_meaning = f'Regression target for predicting numeric values'
                    model_type = 'regression'
                
                prediction_use_cases = [
                    'Predict target variable based on input features',
                    'Identify key factors influencing the outcome',
                    'Build automated decision-making system'
                ]
                business_value = 'Automate predictions and improve decision-making efficiency'
            
            column_info = {
                'original_name': col,
                'suggested_name': self._clean_column_name(col),
                'data_type': data_type,
                'description': f'Column {col} of type {data_type}',
                'is_target': is_target,
                'confidence': 0.7 if is_target else 0.5,
                'reasoning': 'Basic heuristic analysis'
            }
            
            # Add target-specific fields if applicable
            if is_target:
                column_info.update({
                    'target_meaning': target_meaning,
                    'prediction_use_cases': prediction_use_cases,
                    'business_value': business_value,
                    'model_type': model_type
                })
            
            columns.append(column_info)
        
        return {
            'columns': columns,
            'dataset_summary': f'Dataset from {filename} with {len(df)} rows and {len(df.columns)} columns',
            'suggested_use_cases': ['Data analysis', 'Predictive modeling', 'Business intelligence']
        }
    
    def _clean_column_name(self, name: str) -> str:
        """Clean and standardize column names"""
        # Replace underscores with spaces and title case
        cleaned = name.replace('_', ' ').replace('-', ' ')
        cleaned = ' '.join(word.capitalize() for word in cleaned.split())
        return cleaned
    
    def save_mappings_with_eda(self, user_id: str, filename: str, 
                              mappings: List[Dict], eda_results: Dict) -> bool:
        """
        Save column mappings with EDA results to PostgreSQL database
        """
        from utils.user_auth import get_db_connection
        import uuid
        
        try:
            # Clean EDA results to ensure JSON serialization
            eda_results_clean = convert_numpy_types(eda_results) if eda_results else {}
            
            # Prepare mappings data
            mappings_data = {
                'columns': mappings,
                'timestamp': datetime.now().isoformat()
            }
            
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Delete existing mappings for this user and file
                    cursor.execute(
                        """
                        DELETE FROM column_mappings 
                        WHERE user_id = %s AND filename = %s
                        """,
                        (user_id, filename)
                    )
                    
                    # Insert new mapping with EDA results
                    cursor.execute(
                        """
                        INSERT INTO column_mappings 
                        (id, user_id, filename, mappings, eda_results, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            str(uuid.uuid4()),
                            user_id,
                            filename,
                            json.dumps(mappings_data),
                            json.dumps(eda_results_clean),
                            datetime.now(),
                            datetime.now()
                        )
                    )
                    
                    conn.commit()
                    self.logger.info(f"Successfully saved mappings with EDA for user {user_id}, file {filename}")
                    return True
                    
        except Exception as e:
            self.logger.error(f"Error saving mappings with EDA: {str(e)}")
            self.logger.error(traceback.format_exc())
            return False

    def get_saved_mappings_with_eda(self, user_id: str, filename: str) -> Tuple[Optional[List], Optional[Dict]]:
        """
        Retrieve saved mappings and EDA results for a user and filename
        """
        from utils.user_auth import get_db_connection
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT mappings, eda_results 
                        FROM column_mappings 
                        WHERE user_id = %s AND filename = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (user_id, filename)
                    )
                    result = cursor.fetchone()
                    
                    if result:
                        mappings_data = result.get('mappings', {})
                        eda_results = result.get('eda_results', {})
                        
                        # Handle both old and new formats
                        if isinstance(mappings_data, dict) and 'columns' in mappings_data:
                            columns = mappings_data['columns']
                        elif isinstance(mappings_data, list):
                            columns = mappings_data
                        else:
                            return None, None
                        
                        # Ensure all mappings have required fields
                        for mapping in columns:
                            mapping.setdefault('suggested_name', mapping.get('mapped_name', mapping['original_name']))
                            mapping.setdefault('mapped_name', mapping.get('suggested_name', mapping['original_name']))
                            mapping.setdefault('data_type', 'text')
                            mapping.setdefault('description', '')
                            mapping.setdefault('is_target', False)
                            mapping.setdefault('confidence', 0.5)
                            
                            if mapping['is_target']:
                                mapping.setdefault('target_meaning', '')
                                mapping.setdefault('prediction_use_cases', [])
                                mapping.setdefault('business_value', '')
                                mapping.setdefault('model_type', 'classification')
                        
                        return columns, eda_results
                            
            return None, None
            
        except Exception as e:
            self.logger.error(f"Error retrieving mappings with EDA: {str(e)}")
            self.logger.error(traceback.format_exc())
            return None, None
    
    def apply_mappings(self, df: pd.DataFrame, mappings: List[Dict]) -> pd.DataFrame:
        """Apply column mappings to dataframe"""
        df_mapped = df.copy()
        
        # Create mapping dictionary
        rename_dict = {m['original_name']: m['mapped_name'] 
                      for m in mappings 
                      if m['original_name'] != m['mapped_name']}
        
        # Rename columns
        if rename_dict:
            df_mapped.rename(columns=rename_dict, inplace=True)
        
        return df_mapped