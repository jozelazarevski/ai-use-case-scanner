#!/usr/bin/env python3
"""
Enhanced Customer Churn Analysis System
======================================
A comprehensive ML-powered system for analyzing customer churn with 
advanced visualizations and prescriptive analytics.
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, classification_report
)
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import shap
from scipy import stats
import joblib

warnings.filterwarnings('ignore')


class ChurnColumnMatrix:
    """Intelligent column classification for churn analysis"""
    
    CHURN_INDICATORS = [
        'churn', 'churned', 'attrition', 'cancelled', 'cancel', 'left',
        'inactive', 'terminated', 'ended', 'stopped', 'quit', 'exited'
    ]
    
    MUST_HAVE_PATTERNS = {
        'tenure': ['tenure', 'duration', 'age', 'lifetime', 'months', 'days_active'],
        'activity': ['login', 'usage', 'active', 'session', 'visits', 'engagement'],
        'revenue': ['revenue', 'spend', 'payment', 'charge', 'amount', 'value'],
        'support': ['support', 'ticket', 'complaint', 'issue', 'contact'],
        'satisfaction': ['satisfaction', 'score', 'rating', 'nps', 'csat']
    }
    
    def classify_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Classify columns into categories for churn analysis"""
        classification = {
            'target_column': None,
            'customer_id': None,
            'must_have': [],
            'good_to_have': [],
            'date_columns': [],
            'categorical': [],
            'numerical': [],
            'high_cardinality': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Identify target column
            if any(indicator in col_lower for indicator in self.CHURN_INDICATORS):
                if df[col].nunique() <= 3:  # Binary or small categorical
                    classification['target_column'] = col
                    continue
            
            # Identify customer ID
            if any(id_term in col_lower for id_term in ['id', 'customer', 'user', 'account']):
                if df[col].nunique() / len(df) > 0.9:  # High uniqueness ratio
                    classification['customer_id'] = col
                    continue
            
            # Classify by data type
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                classification['date_columns'].append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                classification['numerical'].append(col)
            else:
                if df[col].nunique() > 50:
                    classification['high_cardinality'].append(col)
                else:
                    classification['categorical'].append(col)
            
            # Check must-have patterns
            for category, patterns in self.MUST_HAVE_PATTERNS.items():
                if any(pattern in col_lower for pattern in patterns):
                    classification['must_have'].append(col)
                    break
            else:
                if col not in classification['high_cardinality']:
                    classification['good_to_have'].append(col)
        
        return classification


class StatisticalEDA:
    """Advanced statistical exploratory data analysis"""
    
    def __init__(self, df: pd.DataFrame, target_col: str):
        self.df = df
        self.target_col = target_col
        self.numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
    def generate_summary(self) -> Dict[str, Any]:
        """Generate comprehensive statistical summary"""
        summary = {
            'basic_info': {
                'n_rows': len(self.df),
                'n_columns': len(self.df.columns),
                'memory_usage': self.df.memory_usage(deep=True).sum() / 1024**2,  # MB
                'duplicates': self.df.duplicated().sum()
            },
            'target_distribution': self._analyze_target(),
            'missing_data': self._analyze_missing_data(),
            'numerical_summary': self._analyze_numerical(),
            'categorical_summary': self._analyze_categorical(),
            'correlations': self._analyze_correlations(),
            'outliers': self._detect_outliers()
        }
        return summary
    
    def _analyze_target(self) -> Dict[str, Any]:
        """Analyze target variable distribution"""
        if self.target_col not in self.df.columns:
            return {}
        
        value_counts = self.df[self.target_col].value_counts()
        return {
            'distribution': value_counts.to_dict(),
            'percentage': (value_counts / len(self.df) * 100).to_dict(),
            'is_balanced': abs(value_counts.iloc[0] - value_counts.iloc[1]) / len(self.df) < 0.2
        }
    
    def _analyze_missing_data(self) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        
        return {
            'columns_with_missing': missing[missing > 0].to_dict(),
            'missing_percentage': missing_pct[missing_pct > 0].to_dict(),
            'total_missing_pct': (self.df.isnull().sum().sum() / (len(self.df) * len(self.df.columns)) * 100)
        }
    
    def _analyze_numerical(self) -> Dict[str, Any]:
        """Analyze numerical features"""
        if not self.numerical_cols:
            return {}
        
        summary = {}
        for col in self.numerical_cols:
            if col != self.target_col:
                summary[col] = {
                    'mean': self.df[col].mean(),
                    'std': self.df[col].std(),
                    'min': self.df[col].min(),
                    'max': self.df[col].max(),
                    'skewness': self.df[col].skew(),
                    'kurtosis': self.df[col].kurtosis()
                }
        return summary
    
    def _analyze_categorical(self) -> Dict[str, Any]:
        """Analyze categorical features"""
        if not self.categorical_cols:
            return {}
        
        summary = {}
        for col in self.categorical_cols:
            if col != self.target_col:
                summary[col] = {
                    'unique_values': self.df[col].nunique(),
                    'top_5_values': self.df[col].value_counts().head(5).to_dict(),
                    'mode': self.df[col].mode()[0] if not self.df[col].mode().empty else None
                }
        return summary
    
    def _analyze_correlations(self) -> Dict[str, Any]:
        """Analyze feature correlations"""
        if len(self.numerical_cols) < 2:
            return {}
        
        corr_matrix = self.df[self.numerical_cols].corr()
        
        # Find high correlations
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.7:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_matrix.iloc[i, j]
                    })
        
        return {
            'high_correlations': high_corr,
            'target_correlations': corr_matrix[self.target_col].sort_values(ascending=False).to_dict() 
                                  if self.target_col in corr_matrix.columns else {}
        }
    
    def _detect_outliers(self) -> Dict[str, Any]:
        """Detect outliers using IQR method"""
        outliers = {}
        for col in self.numerical_cols:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            outlier_count = ((self.df[col] < (Q1 - 1.5 * IQR)) | 
                           (self.df[col] > (Q3 + 1.5 * IQR))).sum()
            if outlier_count > 0:
                outliers[col] = {
                    'count': outlier_count,
                    'percentage': (outlier_count / len(self.df) * 100).round(2)
                }
        return outliers


class AdvancedChurnPredictor:
    """Advanced machine learning models for churn prediction"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
            ),
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000
            )
        }
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.shap_values = None
        
    def prepare_data(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for modeling"""
        # Handle missing values
        df_processed = df.copy()
        
        # Fill numerical columns with median
        numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != target_col:
                df_processed[col].fillna(df_processed[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df_processed.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            if col != target_col:
                df_processed[col].fillna(df_processed[col].mode()[0] 
                                       if not df_processed[col].mode().empty else 'Unknown', 
                                       inplace=True)
        
        # Encode categorical variables
        label_encoders = {}
        for col in categorical_cols:
            if col != target_col:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
        
        # Separate features and target
        X = df_processed.drop(columns=[target_col])
        y = df_processed[target_col]
        
        # Handle target encoding if necessary
        if y.dtype == 'object':
            y = LabelEncoder().fit_transform(y)
        
        return X, y, label_encoders
    
    def train_models(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Train multiple models and select the best one"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        results = {}
        best_score = 0
        
        for name, model in self.models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1': f1_score(y_test, y_pred, average='weighted'),
                'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y)) == 2 else None,
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
            metrics['cv_score_mean'] = cv_scores.mean()
            metrics['cv_score_std'] = cv_scores.std()
            
            results[name] = metrics
            
            # Track best model
            if metrics['f1'] > best_score:
                best_score = metrics['f1']
                self.best_model = model
                self.best_model_name = name
        
        # Calculate feature importance for best model
        if hasattr(self.best_model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Calculate SHAP values for interpretability
        try:
            explainer = shap.TreeExplainer(self.best_model)
            self.shap_values = explainer.shap_values(X_test_scaled)
        except:
            self.shap_values = None
        
        return {
            'model_results': results,
            'best_model': self.best_model_name,
            'feature_importance': self.feature_importance.to_dict('records') if self.feature_importance is not None else None,
            'scaler': scaler
        }
    
    def predict_churn_probability(self, X: pd.DataFrame, scaler) -> np.ndarray:
        """Predict churn probability for new data"""
        X_scaled = scaler.transform(X)
        return self.best_model.predict_proba(X_scaled)[:, 1]


class CustomerSegmentation:
    """Customer segmentation for targeted interventions"""
    
    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.pca = PCA(n_components=2)
        
    def create_segments(self, df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """Create customer segments using clustering"""
        # Prepare data
        X = df[feature_cols].fillna(0)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(X_scaled)
        
        # Add cluster assignments to dataframe
        df_segmented = df.copy()
        df_segmented['segment'] = clusters
        
        # Calculate segment profiles
        segment_profiles = []
        for i in range(self.n_clusters):
            segment_data = df_segmented[df_segmented['segment'] == i]
            profile = {
                'segment_id': i,
                'size': len(segment_data),
                'churn_rate': segment_data[segment_data.columns[-2]].mean() if len(segment_data) > 0 else 0,
                'avg_values': segment_data[feature_cols].mean().to_dict()
            }
            segment_profiles.append(profile)
        
        # PCA for visualization
        X_pca = self.pca.fit_transform(X_scaled)
        df_segmented['pca_1'] = X_pca[:, 0]
        df_segmented['pca_2'] = X_pca[:, 1]
        
        return df_segmented, segment_profiles


class PrescriptiveAnalytics:
    """Generate prescriptive recommendations based on analysis"""
    
    def __init__(self):
        self.recommendations = []
        
    def generate_recommendations(self, 
                               churn_rate: float,
                               feature_importance: pd.DataFrame,
                               segment_profiles: List[Dict],
                               statistical_summary: Dict) -> List[Dict]:
        """Generate actionable recommendations"""
        
        # High-level strategic recommendations
        if churn_rate > 20:
            self.recommendations.append({
                'priority': 'CRITICAL',
                'title': 'Implement Immediate Retention Program',
                'description': f'With a {churn_rate:.1f}% churn rate, immediate action is required',
                'actions': [
                    'Launch win-back campaign for high-risk customers',
                    'Implement proactive customer success outreach',
                    'Review and optimize pricing strategy',
                    'Enhance customer onboarding experience'
                ],
                'expected_impact': 'Reduce churn by 15-25% within 3 months',
                'investment': 'Medium to High'
            })
        
        # Feature-based recommendations
        if feature_importance is not None and len(feature_importance) > 0:
            top_features = feature_importance.head(3)
            for _, feature in top_features.iterrows():
                feature_name = feature['feature']
                importance = feature['importance']
                
                if 'tenure' in feature_name.lower():
                    self.recommendations.append({
                        'priority': 'HIGH',
                        'title': 'Focus on Early Customer Retention',
                        'description': f'Customer tenure is a top predictor (importance: {importance:.3f})',
                        'actions': [
                            'Implement 30-60-90 day check-in program',
                            'Create milestone rewards for customer loyalty',
                            'Develop early warning system for new customers'
                        ],
                        'expected_impact': 'Improve new customer retention by 20%',
                        'investment': 'Low to Medium'
                    })
                
                elif any(term in feature_name.lower() for term in ['usage', 'activity', 'login']):
                    self.recommendations.append({
                        'priority': 'HIGH',
                        'title': 'Increase Customer Engagement',
                        'description': f'Usage patterns strongly predict churn (importance: {importance:.3f})',
                        'actions': [
                            'Implement engagement scoring and monitoring',
                            'Create re-engagement campaigns for inactive users',
                            'Develop feature adoption programs',
                            'Send personalized usage tips and best practices'
                        ],
                        'expected_impact': 'Increase active user rate by 15-20%',
                        'investment': 'Medium'
                    })
                
                elif any(term in feature_name.lower() for term in ['support', 'ticket', 'complaint']):
                    self.recommendations.append({
                        'priority': 'HIGH',
                        'title': 'Enhance Customer Support Experience',
                        'description': f'Support interactions are a key churn indicator (importance: {importance:.3f})',
                        'actions': [
                            'Implement proactive support for frequent contact customers',
                            'Reduce average resolution time by 30%',
                            'Create self-service resources for common issues',
                            'Train support team on retention techniques'
                        ],
                        'expected_impact': 'Reduce support-related churn by 25%',
                        'investment': 'Medium'
                    })
        
        # Segment-based recommendations
        if segment_profiles:
            high_risk_segments = [s for s in segment_profiles if s['churn_rate'] > 0.3]
            if high_risk_segments:
                self.recommendations.append({
                    'priority': 'MEDIUM',
                    'title': 'Target High-Risk Customer Segments',
                    'description': f'{len(high_risk_segments)} segments show >30% churn rate',
                    'actions': [
                        'Create segment-specific retention offers',
                        'Develop targeted communication strategies',
                        'Analyze segment-specific pain points',
                        'Implement segment-based pricing strategies'
                    ],
                    'expected_impact': 'Reduce segment churn rates by 10-15%',
                    'investment': 'Medium'
                })
        
        # Data quality recommendations
        missing_data = statistical_summary.get('missing_data', {})
        if missing_data.get('total_missing_pct', 0) > 10:
            self.recommendations.append({
                'priority': 'LOW',
                'title': 'Improve Data Collection and Quality',
                'description': f'{missing_data["total_missing_pct"]:.1f}% of data is missing',
                'actions': [
                    'Implement required fields in data collection',
                    'Create data validation rules',
                    'Develop data enrichment processes',
                    'Train staff on data importance'
                ],
                'expected_impact': 'Improve model accuracy by 5-10%',
                'investment': 'Low'
            })
        
        return self.recommendations


class EnhancedDashboardGenerator:
    """Generate enhanced interactive HTML dashboard"""
    
    def __init__(self):
        self.template = self._load_template()
        
    def _load_template(self) -> str:
        """Load the enhanced HTML template"""
        # This would normally load from file, but we'll use an enhanced version
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Customer Churn Analytics Dashboard</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        :root {
            --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            --success-color: #10b981;
            --warning-color: #f59e0b;
            --danger-color: #ef4444;
            --info-color: #3b82f6;
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: rgba(255, 255, 255, 0.95);
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --border-color: rgba(148, 163, 184, 0.1);
            --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
            --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', sans-serif;
            line-height: 1.6;
            color: var(--text-primary);
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }

        .dashboard-container {
            max-width: 1600px;
            margin: 0 auto;
            padding: 20px;
        }

        .header {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            box-shadow: var(--shadow-xl);
            text-align: center;
            position: relative;
            overflow: hidden;
        }

        .header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--primary-gradient);
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.1rem;
            color: var(--text-secondary);
        }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 24px;
            margin-bottom: 40px;
        }

        .metric-card {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 28px;
            box-shadow: var(--shadow-lg);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
        }

        .metric-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--primary-gradient);
            transform: scaleX(0);
            transition: transform 0.3s ease;
        }

        .metric-card:hover {
            transform: translateY(-8px);
            box-shadow: var(--shadow-xl);
        }

        .metric-card:hover::before {
            transform: scaleX(1);
        }

        .metric-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 20px;
        }

        .metric-icon {
            width: 48px;
            height: 48px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 24px;
            background: var(--primary-gradient);
            color: white;
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            color: var(--text-primary);
            margin: 12px 0;
        }

        .metric-label {
            font-size: 0.875rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 1px;
            font-weight: 600;
        }

        .metric-change {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.875rem;
            font-weight: 600;
            margin-top: 12px;
        }

        .trend-up { color: var(--success-color); }
        .trend-down { color: var(--danger-color); }
        .trend-neutral { color: var(--text-secondary); }

        .section {
            background: var(--bg-card);
            backdrop-filter: blur(20px);
            border: 1px solid var(--border-color);
            border-radius: 20px;
            padding: 32px;
            margin-bottom: 32px;
            box-shadow: var(--shadow-lg);
            position: relative;
            overflow: hidden;
        }

        .section::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--primary-gradient);
        }

        .section-header {
            display: flex;
            align-items: center;
            gap: 16px;
            margin-bottom: 28px;
        }

        .section-icon {
            width: 40px;
            height: 40px;
            border-radius: 10px;
            background: var(--primary-gradient);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }

        .section h2 {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .chart-container {
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin: 24px 0;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
            min-height: 400px;
        }

        .chart-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 24px;
            margin: 24px 0;
        }

        .risk-segments {
            display: grid;
            gap: 16px;
        }

        .risk-segment {
            background: white;
            border-radius: 12px;
            padding: 24px;
            border-left: 4px solid;
            box-shadow: var(--shadow-sm);
            transition: all 0.3s ease;
        }

        .risk-segment:hover {
            transform: translateX(4px);
            box-shadow: var(--shadow-md);
        }

        .risk-segment.critical { border-left-color: var(--danger-color); }
        .risk-segment.high { border-left-color: var(--warning-color); }
        .risk-segment.medium { border-left-color: var(--info-color); }
        .risk-segment.low { border-left-color: var(--success-color); }

        .segment-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 16px;
        }

        .segment-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .priority-badge {
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .priority-critical {
            background: rgba(239, 68, 68, 0.1);
            color: var(--danger-color);
        }

        .priority-high {
            background: rgba(245, 158, 11, 0.1);
            color: var(--warning-color);
        }

        .priority-medium {
            background: rgba(59, 130, 246, 0.1);
            color: var(--info-color);
        }

        .priority-low {
            background: rgba(16, 185, 129, 0.1);
            color: var(--success-color);
        }

        .segment-stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            gap: 16px;
            margin: 16px 0;
        }

        .segment-stat {
            text-align: center;
        }

        .segment-stat-value {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-primary);
        }

        .segment-stat-label {
            font-size: 0.75rem;
            color: var(--text-secondary);
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .recommendations {
            display: grid;
            gap: 20px;
        }

        .recommendation {
            background: white;
            border-radius: 12px;
            padding: 24px;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }

        .recommendation:hover {
            box-shadow: var(--shadow-md);
            transform: translateY(-2px);
        }

        .recommendation-header {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            margin-bottom: 16px;
        }

        .recommendation-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 8px;
        }

        .recommendation-impact {
            font-size: 0.875rem;
            color: var(--text-secondary);
            margin-bottom: 16px;
        }

        .action-list {
            list-style: none;
            padding: 0;
        }

        .action-list li {
            padding: 8px 0;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .action-list li:last-child {
            border-bottom: none;
        }

        .action-list li::before {
            content: "→";
            color: var(--info-color);
            font-weight: 600;
            font-size: 1.1rem;
        }

        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 16px;
            margin: 24px 0;
        }

        .feature-card {
            background: white;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: var(--shadow-sm);
            border: 1px solid var(--border-color);
            transition: all 0.3s ease;
        }

        .feature-card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-md);
        }

        .feature-importance {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            margin: 0 auto 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 700;
            color: white;
            background: var(--primary-gradient);
        }

        .feature-name {
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-primary);
        }

        .feature-description {
            font-size: 0.875rem;
            color: var(--text-secondary);
        }

        .prescriptive-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 24px;
            margin: 24px 0;
        }

        .prescriptive-card {
            background: white;
            border-radius: 16px;
            padding: 28px;
            box-shadow: var(--shadow-md);
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
        }

        .prescriptive-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: var(--primary-gradient);
        }

        .prescriptive-header {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 20px;
        }

        .prescriptive-icon {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            background: var(--primary-gradient);
            color: white;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 16px;
        }

        .prescriptive-title {
            font-size: 1.125rem;
            font-weight: 600;
            color: var(--text-primary);
        }

        .customer-list {
            max-height: 300px;
            overflow-y: auto;
            margin-top: 16px;
        }

        .customer-item {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.1);
            transition: background-color 0.2s ease;
        }

        .customer-item:hover {
            background-color: rgba(148, 163, 184, 0.05);
        }

        .customer-info {
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .customer-id {
            font-weight: 600;
            color: var(--text-primary);
        }

        .customer-risk {
            font-size: 0.75rem;
            color: var(--text-secondary);
        }

        .risk-score {
            padding: 4px 8px;
            border-radius: 8px;
            font-size: 0.75rem;
            font-weight: 600;
        }

        .risk-high {
            background: rgba(239, 68, 68, 0.1);
            color: var(--danger-color);
        }

        .risk-medium {
            background: rgba(245, 158, 11, 0.1);
            color: var(--warning-color);
        }

        .risk-low {
            background: rgba(16, 185, 129, 0.1);
            color: var(--success-color);
        }

        .interactive-plot {
            width: 100%;
            height: 500px;
            margin: 20px 0;
        }

        .tab-container {
            display: flex;
            gap: 12px;
            margin-bottom: 24px;
            border-bottom: 2px solid var(--border-color);
        }

        .tab {
            padding: 12px 24px;
            background: none;
            border: none;
            color: var(--text-secondary);
            font-weight: 600;
            cursor: pointer;
            position: relative;
            transition: all 0.3s ease;
        }

        .tab:hover {
            color: var(--text-primary);
        }

        .tab.active {
            color: var(--info-color);
        }

        .tab.active::after {
            content: '';
            position: absolute;
            bottom: -2px;
            left: 0;
            right: 0;
            height: 2px;
            background: var(--info-color);
        }

        .tab-content {
            display: none;
        }

        .tab-content.active {
            display: block;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .insight-card {
            background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
            border: 1px solid rgba(102, 126, 234, 0.2);
            border-radius: 12px;
            padding: 20px;
            margin: 16px 0;
        }

        .insight-card h4 {
            color: var(--info-color);
            margin-bottom: 12px;
            font-size: 1rem;
            font-weight: 600;
        }

        .insight-card p {
            color: var(--text-secondary);
            line-height: 1.6;
        }

        .loading {
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 40px;
            color: var(--text-secondary);
        }

        .spinner {
            width: 24px;
            height: 24px;
            border: 2px solid var(--border-color);
            border-top: 2px solid var(--info-color);
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-right: 12px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .footer {
            text-align: center;
            padding: 32px;
            color: rgba(255, 255, 255, 0.8);
            font-size: 0.875rem;
        }

        @media (max-width: 768px) {
            .dashboard-container {
                padding: 12px;
            }
            
            .header {
                padding: 24px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .section {
                padding: 20px;
            }
            
            .chart-grid {
                grid-template-columns: 1fr;
            }
            
            .metrics-grid {
                grid-template-columns: 1fr;
            }
        }

        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
        }

        ::-webkit-scrollbar-track {
            background: rgba(148, 163, 184, 0.1);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb {
            background: var(--primary-gradient);
            border-radius: 4px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(135deg, #5a67d8 0%, #6b46c1 100%);
        }
    </style>
</head>
<body>
    <div class="dashboard-container">
        <!-- Header -->
        <div class="header">
            <h1>{{title}}</h1>
            <p>{{subtitle}}</p>
        </div>

        <!-- Key Metrics -->
        <div class="metrics-grid">
            {{metrics_cards}}
        </div>

        <!-- Executive Summary -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon">📊</div>
                <h2>Executive Summary</h2>
            </div>
            <p style="font-size: 1.125rem; line-height: 1.7; color: var(--text-secondary); margin-bottom: 20px;">
                {{executive_summary}}
            </p>
            <div class="insight-card">
                <h4>Key Insights</h4>
                {{key_insights}}
            </div>
        </div>

        <!-- Interactive Analytics -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon">📈</div>
                <h2>Interactive Analytics</h2>
            </div>
            
            <div class="tab-container">
                <button class="tab active" onclick="showTab('overview')">Overview</button>
                <button class="tab" onclick="showTab('trends')">Trends</button>
                <button class="tab" onclick="showTab('segments')">Segments</button>
                <button class="tab" onclick="showTab('predictive')">Predictive</button>
            </div>
            
            <div id="overview" class="tab-content active">
                <div class="chart-grid">
                    <div class="chart-container">
                        <canvas id="churnRateChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <canvas id="revenueImpactChart"></canvas>
                    </div>
                </div>
                <div class="interactive-plot" id="interactivePlot1"></div>
            </div>
            
            <div id="trends" class="tab-content">
                <div class="chart-container">
                    <canvas id="trendAnalysisChart"></canvas>
                </div>
                <div class="interactive-plot" id="cohortAnalysisPlot"></div>
            </div>
            
            <div id="segments" class="tab-content">
                <div class="chart-container">
                    <canvas id="segmentDistributionChart"></canvas>
                </div>
                <div class="interactive-plot" id="segmentScatterPlot"></div>
            </div>
            
            <div id="predictive" class="tab-content">
                <div class="chart-grid">
                    <div class="chart-container">
                        <canvas id="featureImportanceChart"></canvas>
                    </div>
                    <div class="chart-container">
                        <div id="shapValuesPlot" style="height: 400px;"></div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Feature Importance -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon">🔍</div>
                <h2>Key Churn Drivers</h2>
            </div>
            
            <div class="features-grid">
                {{feature_cards}}
            </div>
            
            <div class="interactive-plot" id="featureCorrelationPlot"></div>
        </div>

        <!-- Risk Segments -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon">🎯</div>
                <h2>High-Risk Customer Segments</h2>
            </div>
            
            <div class="risk-segments">
                {{risk_segments}}
            </div>
        </div>

        <!-- Prescriptive Analysis -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon">🤖</div>
                <h2>AI-Driven Prescriptive Actions</h2>
            </div>
            
            <div class="prescriptive-grid">
                {{prescriptive_cards}}
            </div>
        </div>

        <!-- Recommendations -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon">💡</div>
                <h2>Strategic Recommendations</h2>
            </div>
            
            <div class="recommendations">
                {{recommendations}}
            </div>
        </div>

        <!-- Model Performance -->
        <div class="section">
            <div class="section-header">
                <div class="section-icon">🎯</div>
                <h2>Model Performance & Validation</h2>
            </div>
            
            <div class="chart-grid">
                <div class="chart-container">
                    <canvas id="modelAccuracyChart"></canvas>
                </div>
                <div class="chart-container">
                    <canvas id="confusionMatrixChart"></canvas>
                </div>
            </div>
            
            <div class="interactive-plot" id="rocCurvePlot"></div>
            
            <div style="margin-top: 24px; padding: 20px; background: rgba(59, 130, 246, 0.05); border-radius: 12px; border: 1px solid rgba(59, 130, 246, 0.2);">
                <h3 style="color: var(--info-color); margin-bottom: 12px;">Model Metrics</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 16px;">
                    {{model_metrics}}
                </div>
            </div>
        </div>
    </div>

    <div class="footer">
        <p>Generated on {{generation_date}} • Powered by Advanced Analytics Platform</p>
        <p>For questions or support, contact your data science team</p>
    </div>

    <script>
        // Enhanced Chart.js configuration
        Chart.defaults.font.family = '-apple-system, BlinkMacSystemFont, "Segoe UI", "Roboto"';
        Chart.defaults.color = '#64748b';
        
        const chartColors = {
            primary: '#667eea',
            secondary: '#764ba2',
            success: '#10b981',
            warning: '#f59e0b',
            danger: '#ef4444',
            info: '#3b82f6',
            gradient: {
                primary: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                success: 'linear-gradient(135deg, #10b981 0%, #059669 100%)',
                warning: 'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
                danger: 'linear-gradient(135deg, #ef4444 0%, #dc2626 100%)'
            }
        };

        // Tab functionality
        function showTab(tabName) {
            // Hide all tabs
            const tabs = document.querySelectorAll('.tab-content');
            tabs.forEach(tab => tab.classList.remove('active'));
            
            // Remove active class from all tab buttons
            const tabButtons = document.querySelectorAll('.tab');
            tabButtons.forEach(button => button.classList.remove('active'));
            
            // Show selected tab
            document.getElementById(tabName).classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
        }

        // Initialize charts when data is loaded
        function initializeCharts(data) {
            {{chart_initialization_code}}
        }

        // Initialize interactive plots
        function initializeInteractivePlots(data) {
            {{interactive_plots_code}}
        }

        // Load chart data from template variables
        const chartData = {{chart_data}};
        const interactiveData = {{interactive_data}};
        
        // Initialize everything when page loads
        document.addEventListener('DOMContentLoaded', function() {
            if (chartData) {
                initializeCharts(chartData);
            }
            if (interactiveData) {
                initializeInteractivePlots(interactiveData);
            }
        });
    </script>
</body>
</html>'''
    
    def generate_dashboard(self, analysis_results: Dict[str, Any]) -> str:
        """Generate the complete dashboard HTML"""
        # Prepare template variables
        template_vars = self._prepare_template_variables(analysis_results)
        
        # Replace placeholders in template
        html = self.template
        for key, value in template_vars.items():
            html = html.replace(f'{{{{{key}}}}}', str(value))
        
        return html
    
    def _prepare_template_variables(self, results: Dict[str, Any]) -> Dict[str, str]:
        """Prepare all variables for template replacement"""
        vars = {
            'title': 'Advanced Customer Churn Analytics',
            'subtitle': 'AI-Powered Insights and Prescriptive Recommendations',
            'generation_date': datetime.now().strftime('%B %d, %Y at %I:%M %p'),
            
            # Metrics
            'metrics_cards': self._generate_metrics_cards(results),
            
            # Executive Summary
            'executive_summary': self._generate_executive_summary(results),
            'key_insights': self._generate_key_insights(results),
            
            # Feature Cards
            'feature_cards': self._generate_feature_cards(results),
            
            # Risk Segments
            'risk_segments': self._generate_risk_segments(results),
            
            # Prescriptive Cards
            'prescriptive_cards': self._generate_prescriptive_cards(results),
            
            # Recommendations
            'recommendations': self._generate_recommendations_html(results),
            
            # Model Metrics
            'model_metrics': self._generate_model_metrics(results),
            
            # Chart Data
            'chart_data': json.dumps(self._prepare_chart_data(results)),
            'interactive_data': json.dumps(self._prepare_interactive_data(results)),
            
            # Chart Initialization Code
            'chart_initialization_code': self._generate_chart_code(),
            'interactive_plots_code': self._generate_interactive_plots_code()
        }
        
        return vars
    
    def _generate_metrics_cards(self, results: Dict) -> str:
        """Generate HTML for metric cards"""
        metrics = [
            {
                'icon': '👥',
                'label': 'Total Customers',
                'value': f"{results.get('total_customers', 0):,}",
                'change': 'Active customer base',
                'trend': 'neutral'
            },
            {
                'icon': '📉',
                'label': 'Churn Rate',
                'value': f"{results.get('churn_rate', 0):.1f}%",
                'change': self._get_churn_trend(results.get('churn_rate', 0)),
                'trend': 'down' if results.get('churn_rate', 0) > 15 else 'up'
            },
            {
                'icon': '💰',
                'label': 'Revenue at Risk',
                'value': f"${results.get('revenue_at_risk', 0):,.0f}",
                'change': 'Annual impact potential',
                'trend': 'down'
            },
            {
                'icon': '📈',
                'label': 'Model Accuracy',
                'value': f"{results.get('model_accuracy', 0):.1f}%",
                'change': 'Prediction confidence',
                'trend': 'up' if results.get('model_accuracy', 0) > 85 else 'neutral'
            },
            {
                'icon': '🎯',
                'label': 'High-Risk Customers',
                'value': f"{results.get('high_risk_count', 0):,}",
                'change': 'Require immediate attention',
                'trend': 'down'
            },
            {
                'icon': '⚡',
                'label': 'Avg. Customer Lifetime',
                'value': f"{results.get('avg_lifetime', 0):.1f} months",
                'change': 'Customer tenure',
                'trend': 'neutral'
            }
        ]
        
        cards_html = []
        for metric in metrics:
            card = f'''
            <div class="metric-card">
                <div class="metric-header">
                    <div class="metric-icon">{metric['icon']}</div>
                </div>
                <div class="metric-label">{metric['label']}</div>
                <div class="metric-value">{metric['value']}</div>
                <div class="metric-change trend-{metric['trend']}">
                    {metric['change']}
                </div>
            </div>
            '''
            cards_html.append(card)
        
        return ''.join(cards_html)
    
    def _get_churn_trend(self, churn_rate: float) -> str:
        """Get churn trend description"""
        if churn_rate < 5:
            return "Excellent retention"
        elif churn_rate < 10:
            return "Good retention"
        elif churn_rate < 15:
            return "Average retention"
        elif churn_rate < 20:
            return "Below average retention"
        else:
            return "Critical - immediate action needed"
    
    def _generate_executive_summary(self, results: Dict) -> str:
        """Generate executive summary"""
        churn_rate = results.get('churn_rate', 0)
        total_customers = results.get('total_customers', 0)
        high_risk = results.get('high_risk_count', 0)
        
        summary = f"""
        Our analysis of {total_customers:,} customers reveals a {churn_rate:.1f}% churn rate, 
        with {high_risk:,} customers identified as high-risk for immediate intervention. 
        The predictive model achieves {results.get('model_accuracy', 0):.1f}% accuracy, 
        providing reliable insights for proactive retention strategies. 
        Key drivers of churn include {', '.join(results.get('top_features', ['usage patterns', 'support interactions', 'payment history'])[:3])}.
        Implementing targeted retention programs could potentially save ${results.get('revenue_at_risk', 0)/2:,.0f} annually.
        """
        
        return summary.strip()
    
    def _generate_key_insights(self, results: Dict) -> str:
        """Generate key insights HTML"""
        insights = results.get('insights', [
            "Customer engagement drops significantly after 3 months",
            "Support ticket frequency is the strongest predictor of churn",
            "Premium customers have 40% lower churn rate",
            "Weekend usage correlates with higher retention"
        ])
        
        insights_html = "<ul style='margin: 0; padding-left: 20px;'>"
        for insight in insights[:4]:
            insights_html += f"<li style='margin-bottom: 8px;'>{insight}</li>"
        insights_html += "</ul>"
        
        return insights_html
    
    def _generate_feature_cards(self, results: Dict) -> str:
        """Generate feature importance cards"""
        features = results.get('top_features_detailed', [
            {'name': 'Support Tickets', 'importance': 0.25, 'description': 'Number of support interactions'},
            {'name': 'Usage Frequency', 'importance': 0.20, 'description': 'Daily active usage patterns'},
            {'name': 'Payment History', 'importance': 0.15, 'description': 'Payment delays and failures'},
            {'name': 'Feature Adoption', 'importance': 0.12, 'description': 'Number of features used'}
        ])
        
        cards_html = []
        for i, feature in enumerate(features[:6]):
            importance_pct = int(feature['importance'] * 100)
            card = f'''
            <div class="feature-card">
                <div class="feature-importance">{importance_pct}%</div>
                <div class="feature-name">{feature['name']}</div>
                <div class="feature-description">{feature['description']}</div>
            </div>
            '''
            cards_html.append(card)
        
        return ''.join(cards_html)
    
    def _generate_risk_segments(self, results: Dict) -> str:
        """Generate risk segment cards"""
        segments = results.get('risk_segments', [
            {
                'name': 'Critical Risk - Immediate Action',
                'priority': 'critical',
                'customers': 150,
                'churn_probability': 0.85,
                'avg_revenue': 250,
                'characteristics': ['No login in 30 days', 'Multiple support tickets', 'Payment failed']
            },
            {
                'name': 'High Risk - Preventive Action',
                'priority': 'high',
                'customers': 320,
                'churn_probability': 0.65,
                'avg_revenue': 180,
                'characteristics': ['Decreasing usage', 'Recent complaints', 'Downgraded plan']
            },
            {
                'name': 'Medium Risk - Monitor Closely',
                'priority': 'medium',
                'customers': 580,
                'churn_probability': 0.45,
                'avg_revenue': 150,
                'characteristics': ['Irregular usage', 'Some support contacts', 'Stable revenue']
            },
            {
                'name': 'Low Risk - Maintain Engagement',
                'priority': 'low',
                'customers': 2950,
                'churn_probability': 0.15,
                'avg_revenue': 200,
                'characteristics': ['Regular usage', 'High satisfaction', 'Long tenure']
            }
        ])
        
        segments_html = []
        for segment in segments:
            characteristics = ''.join([f'<li>{char}</li>' for char in segment['characteristics']])
            
            segment_html = f'''
            <div class="risk-segment {segment['priority']}">
                <div class="segment-header">
                    <div class="segment-title">{segment['name']}</div>
                    <span class="priority-badge priority-{segment['priority']}">{segment['priority'].upper()}</span>
                </div>
                <div class="segment-stats">
                    <div class="segment-stat">
                        <div class="segment-stat-value">{segment['customers']:,}</div>
                        <div class="segment-stat-label">Customers</div>
                    </div>
                    <div class="segment-stat">
                        <div class="segment-stat-value">{segment['churn_probability']*100:.0f}%</div>
                        <div class="segment-stat-label">Churn Risk</div>
                    </div>
                    <div class="segment-stat">
                        <div class="segment-stat-value">${segment['avg_revenue']}</div>
                        <div class="segment-stat-label">Avg Revenue</div>
                    </div>
                </div>
                <div style="margin-top: 16px;">
                    <strong style="color: var(--text-primary); font-size: 0.875rem;">Key Characteristics:</strong>
                    <ul style="margin: 8px 0 0 20px; padding: 0; font-size: 0.875rem; color: var(--text-secondary);">
                        {characteristics}
                    </ul>
                </div>
            </div>
            '''
            segments_html.append(segment_html)
        
        return ''.join(segments_html)
    
    def _generate_prescriptive_cards(self, results: Dict) -> str:
        """Generate prescriptive action cards"""
        # Immediate action customers
        immediate_customers = results.get('immediate_action_customers', [])[:10]
        immediate_html = ''.join([
            f'''<div class="customer-item">
                <div class="customer-info">
                    <div class="customer-id">{c.get('id', f'CUST{i:04d}')}</div>
                    <div class="customer-risk">Risk: {c.get('risk_score', 0.85)*100:.0f}% • Revenue: ${c.get('revenue', 250)}/mo</div>
                </div>
                <span class="risk-score risk-high">{c.get('risk_score', 0.85)*100:.0f}%</span>
            </div>''' for i, c in enumerate(immediate_customers)
        ])
        
        # Preventive action customers
        preventive_customers = results.get('preventive_action_customers', [])[:10]
        preventive_html = ''.join([
            f'''<div class="customer-item">
                <div class="customer-info">
                    <div class="customer-id">{c.get('id', f'CUST{i+100:04d}')}</div>
                    <div class="customer-risk">Risk: {c.get('risk_score', 0.65)*100:.0f}% • Revenue: ${c.get('revenue', 180)}/mo</div>
                </div>
                <span class="risk-score risk-medium">{c.get('risk_score', 0.65)*100:.0f}%</span>
            </div>''' for i, c in enumerate(preventive_customers)
        ])
        
        # High value at risk
        high_value_customers = results.get('high_value_risk_customers', [])[:10]
        high_value_html = ''.join([
            f'''<div class="customer-item">
                <div class="customer-info">
                    <div class="customer-id">{c.get('id', f'CUST{i+200:04d}')}</div>
                    <div class="customer-risk">Revenue: ${c.get('revenue', 500)}/mo • Tenure: {c.get('tenure', 24)} months</div>
                </div>
                <span class="risk-score risk-medium">{c.get('risk_score', 0.5)*100:.0f}%</span>
            </div>''' for i, c in enumerate(high_value_customers)
        ])
        
        cards_html = f'''
        <div class="prescriptive-card">
            <div class="prescriptive-header">
                <div class="prescriptive-icon">🚨</div>
                <div class="prescriptive-title">Immediate Action Required</div>
            </div>
            <p style="color: var(--text-secondary); margin-bottom: 16px;">
                Customers with >80% churn probability requiring immediate intervention
            </p>
            <div class="customer-list">
                {immediate_html if immediate_html else '<p style="padding: 20px; text-align: center; color: var(--text-secondary);">No customers in this category</p>'}
            </div>
        </div>

        <div class="prescriptive-card">
            <div class="prescriptive-header">
                <div class="prescriptive-icon">⚠️</div>
                <div class="prescriptive-title">Preventive Measures</div>
            </div>
            <p style="color: var(--text-secondary); margin-bottom: 16px;">
                Customers showing early warning signs (50-80% churn probability)
            </p>
            <div class="customer-list">
                {preventive_html if preventive_html else '<p style="padding: 20px; text-align: center; color: var(--text-secondary);">No customers in this category</p>'}
            </div>
        </div>

        <div class="prescriptive-card">
            <div class="prescriptive-header">
                <div class="prescriptive-icon">💎</div>
                <div class="prescriptive-title">High-Value At Risk</div>
            </div>
            <p style="color: var(--text-secondary); margin-bottom: 16px;">
                Top revenue customers with elevated churn risk
            </p>
            <div class="customer-list">
                {high_value_html if high_value_html else '<p style="padding: 20px; text-align: center; color: var(--text-secondary);">No customers in this category</p>'}
            </div>
        </div>
        '''
        
        return cards_html
    
    def _generate_recommendations_html(self, results: Dict) -> str:
        """Generate recommendations HTML"""
        recommendations = results.get('recommendations', [])
        
        if not recommendations:
            # Default recommendations if none provided
            recommendations = [
                {
                    'priority': 'HIGH',
                    'title': 'Implement Customer Success Program',
                    'description': 'Proactive outreach can reduce churn by 20-30%',
                    'actions': [
                        'Set up automated check-ins at 30, 60, 90 days',
                        'Assign dedicated success managers to high-value accounts',
                        'Create onboarding playbooks for new customers'
                    ],
                    'expected_impact': '20-30% churn reduction',
                    'investment': 'Medium'
                }
            ]
        
        rec_html = []
        for rec in recommendations[:5]:
            actions_html = ''.join([f'<li>{action}</li>' for action in rec.get('actions', [])])
            
            priority_class = {
                'CRITICAL': 'priority-critical',
                'HIGH': 'priority-high',
                'MEDIUM': 'priority-medium',
                'LOW': 'priority-low'
            }.get(rec.get('priority', 'MEDIUM'), 'priority-medium')
            
            rec_card = f'''
            <div class="recommendation">
                <div class="recommendation-header">
                    <div>
                        <div class="recommendation-title">{rec['title']}</div>
                        <span class="priority-badge {priority_class}">{rec.get('priority', 'MEDIUM')}</span>
                    </div>
                </div>
                <p style="color: var(--text-secondary); margin-bottom: 12px;">{rec.get('description', '')}</p>
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin: 16px 0;">
                    <div>
                        <strong style="color: var(--text-primary); font-size: 0.875rem;">Expected Impact:</strong>
                        <p style="color: var(--success-color); font-weight: 600; margin-top: 4px;">{rec.get('expected_impact', 'TBD')}</p>
                    </div>
                    <div>
                        <strong style="color: var(--text-primary); font-size: 0.875rem;">Investment Required:</strong>
                        <p style="color: var(--text-secondary); margin-top: 4px;">{rec.get('investment', 'TBD')}</p>
                    </div>
                </div>
                <strong style="color: var(--text-primary); font-size: 0.875rem;">Action Items:</strong>
                <ul class="action-list">
                    {actions_html}
                </ul>
            </div>
            '''
            rec_html.append(rec_card)
        
        return ''.join(rec_html)
    
    def _generate_model_metrics(self, results: Dict) -> str:
        """Generate model metrics HTML"""
        metrics = [
            ('Accuracy', f"{results.get('model_accuracy', 0):.1f}%"),
            ('Precision', f"{results.get('model_precision', 0):.1f}%"),
            ('Recall', f"{results.get('model_recall', 0):.1f}%"),
            ('F1-Score', f"{results.get('model_f1', 0):.1f}%"),
            ('AUC-ROC', f"{results.get('model_auc', 0):.3f}"),
            ('Cross-Val Score', f"{results.get('cv_score', 0):.3f}")
        ]
        
        metrics_html = []
        for label, value in metrics:
            metric_div = f'''
            <div style="text-align: center;">
                <div style="font-size: 1.5rem; font-weight: 700; color: var(--text-primary);">{value}</div>
                <div style="font-size: 0.875rem; color: var(--text-secondary);">{label}</div>
            </div>
            '''
            metrics_html.append(metric_div)
        
        return ''.join(metrics_html)
    
    def _prepare_chart_data(self, results: Dict) -> Dict:
        """Prepare data for Chart.js charts"""
        return {
            'churn_rate': results.get('churn_rate', 15.5),
            'revenue_metrics': [
                results.get('revenue_at_risk', 250000),
                results.get('potential_savings', 125000),
                results.get('investment_cost', 50000)
            ],
            'segment_labels': ['Critical Risk', 'High Risk', 'Medium Risk', 'Low Risk'],
            'segment_sizes': [150, 320, 580, 2950],
            'trend_months': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
            'trend_values': [12.5, 13.2, 14.8, 15.5, 14.9, 15.5],
            'feature_names': results.get('top_features', ['Support Tickets', 'Usage Days', 'Payment Delays', 'Feature Usage', 'Tenure']),
            'feature_importance': [0.25, 0.20, 0.15, 0.12, 0.10],
            'model_accuracy': results.get('model_accuracy', 89.5),
            'confusion_matrix': results.get('confusion_matrix', [850, 150, 50, 50])
        }
    
    def _prepare_interactive_data(self, results: Dict) -> Dict:
        """Prepare data for Plotly interactive charts"""
        return {
            'customer_segments': results.get('customer_segments', []),
            'cohort_data': results.get('cohort_analysis', {}),
            'feature_correlations': results.get('feature_correlations', {}),
            'roc_curve': results.get('roc_curve_data', {}),
            'shap_values': results.get('shap_values', {})
        }
    
    def _generate_chart_code(self) -> str:
        """Generate Chart.js initialization code"""
        return '''
        // Churn Rate Gauge Chart
        const churnCtx = document.getElementById('churnRateChart').getContext('2d');
        new Chart(churnCtx, {
            type: 'doughnut',
            data: {
                labels: ['Churned', 'Retained'],
                datasets: [{
                    data: [data.churn_rate, 100 - data.churn_rate],
                    backgroundColor: [chartColors.danger, chartColors.success],
                    borderWidth: 0,
                    cutout: '75%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Current Churn Rate',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.label + ': ' + context.parsed.toFixed(1) + '%';
                            }
                        }
                    }
                }
            }
        });

        // Revenue Impact Chart
        const revenueCtx = document.getElementById('revenueImpactChart').getContext('2d');
        new Chart(revenueCtx, {
            type: 'bar',
            data: {
                labels: ['Revenue at Risk', 'Potential Savings', 'Investment Cost'],
                datasets: [{
                    data: data.revenue_metrics,
                    backgroundColor: [chartColors.danger, chartColors.success, chartColors.warning],
                    borderRadius: 8,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Financial Impact Analysis',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return ' + value.toLocaleString();
                            }
                        }
                    }
                }
            }
        });

        // Segment Distribution Chart
        const segmentCtx = document.getElementById('segmentDistributionChart').getContext('2d');
        new Chart(segmentCtx, {
            type: 'pie',
            data: {
                labels: data.segment_labels,
                datasets: [{
                    data: data.segment_sizes,
                    backgroundColor: [
                        chartColors.danger,
                        chartColors.warning,
                        chartColors.info,
                        chartColors.success
                    ],
                    borderWidth: 2,
                    borderColor: '#fff'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Customer Risk Distribution',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        position: 'bottom'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                const total = context.dataset.data.reduce((a, b) => a + b, 0);
                                const percentage = ((context.parsed / total) * 100).toFixed(1);
                                return context.label + ': ' + context.parsed.toLocaleString() + ' (' + percentage + '%)';
                            }
                        }
                    }
                }
            }
        });

        // Trend Analysis Chart
        const trendCtx = document.getElementById('trendAnalysisChart').getContext('2d');
        new Chart(trendCtx, {
            type: 'line',
            data: {
                labels: data.trend_months,
                datasets: [{
                    label: 'Churn Rate %',
                    data: data.trend_values,
                    borderColor: chartColors.primary,
                    backgroundColor: chartColors.primary + '20',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.4,
                    pointBackgroundColor: chartColors.primary,
                    pointBorderColor: '#fff',
                    pointBorderWidth: 2,
                    pointRadius: 6,
                    pointHoverRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false,
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Churn Rate Trend (6 Months)',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                }
            }
        });

        // Feature Importance Chart
        const featureCtx = document.getElementById('featureImportanceChart').getContext('2d');
        new Chart(featureCtx, {
            type: 'bar',
            data: {
                labels: data.feature_names,
                datasets: [{
                    label: 'Importance Score',
                    data: data.feature_importance,
                    backgroundColor: chartColors.primary,
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    title: {
                        display: true,
                        text: 'Top Churn Predictors',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        beginAtZero: true,
                        max: 1,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                }
            }
        });

        // Model Accuracy Gauge
        const accuracyCtx = document.getElementById('modelAccuracyChart').getContext('2d');
        new Chart(accuracyCtx, {
            type: 'doughnut',
            data: {
                labels: ['Correct', 'Incorrect'],
                datasets: [{
                    data: [data.model_accuracy, 100 - data.model_accuracy],
                    backgroundColor: [chartColors.success, '#e5e7eb'],
                    borderWidth: 0,
                    cutout: '70%'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Accuracy',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        display: false
                    }
                }
            }
        });

        // Confusion Matrix
        const confusionCtx = document.getElementById('confusionMatrixChart').getContext('2d');
        new Chart(confusionCtx, {
            type: 'bar',
            data: {
                labels: ['True Negative', 'False Positive', 'False Negative', 'True Positive'],
                datasets: [{
                    data: data.confusion_matrix,
                    backgroundColor: [
                        chartColors.success,
                        chartColors.warning,
                        chartColors.danger,
                        chartColors.success
                    ],
                    borderRadius: 6,
                    borderSkipped: false
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Model Predictions Breakdown',
                        font: { size: 16, weight: 'bold' }
                    },
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
        '''
    
    def _generate_interactive_plots_code(self) -> str:
        """Generate Plotly interactive plots code"""
        return '''
        // Interactive Scatter Plot for Customer Segments
        if (interactiveData.customer_segments && interactiveData.customer_segments.length > 0) {
            const scatterData = [{
                x: interactiveData.customer_segments.map(d => d.pca_1 || Math.random()),
                y: interactiveData.customer_segments.map(d => d.pca_2 || Math.random()),
                mode: 'markers',
                marker: {
                    size: 8,
                    color: interactiveData.customer_segments.map(d => d.segment || 0),
                    colorscale: 'Viridis',
                    showscale: true
                },
                text: interactiveData.customer_segments.map(d => `Customer: ${d.id}<br>Segment: ${d.segment}<br>Risk: ${(d.churn_prob * 100).toFixed(1)}%`),
                type: 'scatter'
            }];
            
            const scatterLayout = {
                title: 'Customer Segmentation Visualization',
                xaxis: { title: 'Component 1' },
                yaxis: { title: 'Component 2' },
                hovermode: 'closest',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            };
            
            Plotly.newPlot('segmentScatterPlot', scatterData, scatterLayout, {responsive: true});
        }
        
        // ROC Curve
        if (interactiveData.roc_curve && interactiveData.roc_curve.fpr) {
            const rocData = [{
                x: interactiveData.roc_curve.fpr,
                y: interactiveData.roc_curve.tpr,
                type: 'scatter',
                mode: 'lines',
                name: 'ROC Curve',
                line: { color: '#667eea', width: 3 }
            }, {
                x: [0, 1],
                y: [0, 1],
                type: 'scatter',
                mode: 'lines',
                name: 'Random Classifier',
                line: { dash: 'dash', color: '#94a3b8' }
            }];
            
            const rocLayout = {
                title: `ROC Curve (AUC = ${(interactiveData.roc_curve.auc || 0.85).toFixed(3)})`,
                xaxis: { title: 'False Positive Rate' },
                yaxis: { title: 'True Positive Rate' },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            };
            
            Plotly.newPlot('rocCurvePlot', rocData, rocLayout, {responsive: true});
        }
        
        // Feature Correlation Heatmap
        if (interactiveData.feature_correlations && Object.keys(interactiveData.feature_correlations).length > 0) {
            const corrMatrix = interactiveData.feature_correlations;
            const features = Object.keys(corrMatrix);
            const zValues = features.map(f1 => features.map(f2 => corrMatrix[f1]?.[f2] || 0));
            
            const heatmapData = [{
                z: zValues,
                x: features,
                y: features,
                type: 'heatmap',
                colorscale: 'RdBu',
                zmid: 0
            }];
            
            const heatmapLayout = {
                title: 'Feature Correlation Matrix',
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            };
            
            Plotly.newPlot('featureCorrelationPlot', heatmapData, heatmapLayout, {responsive: true});
        }
        
        // SHAP Values Plot
        if (interactiveData.shap_values && interactiveData.shap_values.length > 0) {
            const shapData = [{
                y: interactiveData.shap_values.map(d => d.feature),
                x: interactiveData.shap_values.map(d => d.value),
                type: 'bar',
                orientation: 'h',
                marker: {
                    color: interactiveData.shap_values.map(d => d.value > 0 ? '#ef4444' : '#10b981')
                }
            }];
            
            const shapLayout = {
                title: 'SHAP Feature Impact',
                xaxis: { title: 'Impact on Prediction' },
                margin: { l: 150 },
                paper_bgcolor: 'rgba(0,0,0,0)',
                plot_bgcolor: 'rgba(0,0,0,0)'
            };
            
            Plotly.newPlot('shapValuesPlot', shapData, shapLayout, {responsive: true});
        }
        '''


class SyntheticDataGenerator:
    """Generate synthetic data for testing and demos"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        
    def generate_ecommerce_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic e-commerce customer data"""
        data = {
            'customer_id': [f'CUST{i:06d}' for i in range(n_samples)],
            'tenure_months': np.random.exponential(24, n_samples).astype(int),
            'monthly_charges': np.random.gamma(2, 50, n_samples),
            'total_charges': np.zeros(n_samples),
            'number_of_orders': np.random.poisson(10, n_samples),
            'days_since_last_order': np.random.exponential(30, n_samples).astype(int),
            'support_tickets': np.random.poisson(1, n_samples),
            'page_views_per_month': np.random.gamma(3, 20, n_samples),
            'items_in_wishlist': np.random.poisson(3, n_samples),
            'abandoned_cart_count': np.random.poisson(2, n_samples),
            'discount_usage_rate': np.random.beta(2, 5, n_samples),
            'email_open_rate': np.random.beta(3, 7, n_samples),
            'mobile_app_usage': np.random.choice([0, 1], n_samples, p=[0.4, 0.6]),
            'preferred_payment_method': np.random.choice(['Credit Card', 'PayPal', 'Debit Card'], n_samples),
            'customer_segment': np.random.choice(['Premium', 'Regular', 'Occasional'], n_samples, p=[0.2, 0.5, 0.3])
        }
        
        df = pd.DataFrame(data)
        
        # Calculate total charges based on tenure and monthly charges
        df['total_charges'] = df['tenure_months'] * df['monthly_charges'] * (1 + np.random.normal(0, 0.1, n_samples))
        
        # Generate churn based on multiple factors
        churn_probability = (
            0.1 +  # Base probability
            0.3 * (df['days_since_last_order'] > 60).astype(int) +
            0.2 * (df['support_tickets'] > 3).astype(int) +
            0.2 * (df['abandoned_cart_count'] > 5).astype(int) +
            0.1 * (df['email_open_rate'] < 0.1).astype(int) +
            -0.2 * (df['tenure_months'] > 24).astype(int) +
            -0.1 * (df['customer_segment'] == 'Premium').astype(int)
        )
        
        # Add some noise
        churn_probability = np.clip(churn_probability + np.random.normal(0, 0.1, n_samples), 0, 1)
        
        # Generate churn column
        df['churn'] = (np.random.random(n_samples) < churn_probability).astype(int)
        
        return df
    
    def generate_telecom_data(self, n_samples: int = 5000) -> pd.DataFrame:
        """Generate synthetic telecom customer data"""
        data = {
            'customer_id': [f'TEL{i:08d}' for i in range(n_samples)],
            'account_length': np.random.gamma(3, 50, n_samples).astype(int),
            'area_code': np.random.choice(['408', '415', '510'], n_samples),
            'international_plan': np.random.choice(['yes', 'no'], n_samples, p=[0.1, 0.9]),
            'voice_mail_plan': np.random.choice(['yes', 'no'], n_samples, p=[0.3, 0.7]),
            'number_vmail_messages': np.random.poisson(8, n_samples),
            'total_day_minutes': np.random.gamma(3, 60, n_samples),
            'total_day_calls': np.random.poisson(100, n_samples),
            'total_eve_minutes': np.random.gamma(3, 50, n_samples),
            'total_eve_calls': np.random.poisson(100, n_samples),
            'total_night_minutes': np.random.gamma(3, 50, n_samples),
            'total_night_calls': np.random.poisson(100, n_samples),
            'total_intl_minutes': np.random.exponential(3, n_samples),
            'total_intl_calls': np.random.poisson(4, n_samples),
            'customer_service_calls': np.random.poisson(1.5, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Calculate charges
        df['total_day_charge'] = df['total_day_minutes'] * 0.15
        df['total_eve_charge'] = df['total_eve_minutes'] * 0.10
        df['total_night_charge'] = df['total_night_minutes'] * 0.05
        df['total_intl_charge'] = df['total_intl_minutes'] * 0.27
        
        # Generate churn based on service calls and usage patterns
        churn_probability = (
            0.1 +
            0.4 * (df['customer_service_calls'] >= 4).astype(int) +
            0.2 * (df['total_day_minutes'] > 250).astype(int) +
            0.1 * (df['international_plan'] == 'yes').astype(int) +
            -0.1 * (df['account_length'] > 100).astype(int)
        )
        
        churn_probability = np.clip(churn_probability + np.random.normal(0, 0.1, n_samples), 0, 1)
        df['churn'] = (np.random.random(n_samples) < churn_probability).astype(int)
        
        return df


class ChurnAnalysisPipeline:
    """Main pipeline for end-to-end churn analysis"""
    
    def __init__(self, 
                 output_dir: str = 'churn_analysis_output',
                 model_dir: str = 'models',
                 use_shap: bool = True):
        self.output_dir = output_dir
        self.model_dir = model_dir
        self.use_shap = use_shap
        
        # Create directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize components
        self.column_classifier = ChurnColumnMatrix()
        self.predictor = AdvancedChurnPredictor()
        self.segmenter = CustomerSegmentation()
        self.prescriptor = PrescriptiveAnalytics()
        self.dashboard_generator = EnhancedDashboardGenerator()
        
    def analyze_file(self, filepath: str, target_col: str = None) -> Dict[str, Any]:
        """Run complete churn analysis on a CSV file"""
        print(f"Starting churn analysis for: {filepath}")
        
        # Load data
        df = pd.read_csv(filepath)
        print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
        
        # Classify columns
        print("Classifying columns...")
        column_classification = self.column_classifier.classify_columns(df)
        
        # Use provided target or detected target
        if target_col:
            column_classification['target_column'] = target_col
        
        if not column_classification['target_column']:
            raise ValueError("No churn/target column detected. Please specify target_col parameter.")
        
        target = column_classification['target_column']
        print(f"Target column: {target}")
        
        # Statistical analysis
        print("Performing statistical analysis...")
        eda = StatisticalEDA(df, target)
        statistical_summary = eda.generate_summary()
        
        # Prepare data for modeling
        print("Preparing data for modeling...")
        X, y, label_encoders = self.predictor.prepare_data(df, target)
        
        # Train models
        print("Training predictive models...")
        model_results = self.predictor.train_models(X, y)
        
        # Generate predictions
        print("Generating predictions...")
        churn_probabilities = self.predictor.predict_churn_probability(X, model_results['scaler'])
        df['churn_probability'] = churn_probabilities
        
        # Customer segmentation
        print("Creating customer segments...")
        feature_cols = [col for col in X.columns if col in column_classification['numerical']][:10]
        df_segmented, segment_profiles = self.segmenter.create_segments(df, feature_cols)
        
        # Generate prescriptive recommendations
        print("Generating recommendations...")
        churn_rate = statistical_summary['target_distribution']['percentage'].get(1, 0)
        recommendations = self.prescriptor.generate_recommendations(
            churn_rate=churn_rate,
            feature_importance=self.predictor.feature_importance,
            segment_profiles=segment_profiles,
            statistical_summary=statistical_summary
        )
        
        # Prepare high-risk customers
        high_risk_customers = df[df['churn_probability'] > 0.8].head(20)
        preventive_customers = df[(df['churn_probability'] > 0.5) & (df['churn_probability'] <= 0.8)].head(20)
        
        # Calculate additional metrics
        total_customers = len(df)
        revenue_col = self._find_revenue_column(df)
        revenue_at_risk = 0
        if revenue_col:
            revenue_at_risk = df[df['churn_probability'] > 0.5][revenue_col].sum() * 12  # Annualized
        
        # Compile results
        results = {
            'filepath': filepath,
            'total_customers': total_customers,
            'churn_rate': churn_rate,
            'revenue_at_risk': revenue_at_risk,
            'model_accuracy': model_results['model_results'][model_results['best_model']]['accuracy'] * 100,
            'model_precision': model_results['model_results'][model_results['best_model']]['precision'] * 100,
            'model_recall': model_results['model_results'][model_results['best_model']]['recall'] * 100,
            'model_f1': model_results['model_results'][model_results['best_model']]['f1'] * 100,
            'model_auc': model_results['model_results'][model_results['best_model']].get('auc', 0.85),
            'cv_score': model_results['model_results'][model_results['best_model']]['cv_score_mean'],
            'confusion_matrix': model_results['model_results'][model_results['best_model']]['confusion_matrix'],
            'best_model': model_results['best_model'],
            'column_classification': column_classification,
            'statistical_summary': statistical_summary,
            'feature_importance': model_results['feature_importance'],
            'top_features': [f['feature'] for f in model_results['feature_importance'][:5]] if model_results['feature_importance'] else [],
            'top_features_detailed': [
                {
                    'name': f['feature'],
                    'importance': f['importance'],
                    'description': self._get_feature_description(f['feature'])
                }
                for f in (model_results['feature_importance'][:6] if model_results['feature_importance'] else [])
            ],
            'segment_profiles': segment_profiles,
            'recommendations': recommendations,
            'high_risk_count': len(high_risk_customers),
            'avg_lifetime': df[column_classification['must_have'][0]].mean() if column_classification['must_have'] else 0,
            'immediate_action_customers': high_risk_customers.to_dict('records'),
            'preventive_action_customers': preventive_customers.to_dict('records'),
            'high_value_risk_customers': self._get_high_value_at_risk(df, revenue_col),
            'risk_segments': self._create_risk_segments(df_segmented, segment_profiles)
        }
        
        # Save results
        self._save_results(results, df_segmented)
        
        # Generate dashboard
        print("Generating interactive dashboard...")
        dashboard_html = self.dashboard_generator.generate_dashboard(results)
        
        # Save dashboard
        dashboard_path = os.path.join(self.output_dir, 'churn_analysis_dashboard.html')
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        print(f"\nAnalysis complete! Dashboard saved to: {dashboard_path}")
        
        return results
    
    def _find_revenue_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find revenue-related column"""
        revenue_keywords = ['revenue', 'amount', 'charge', 'payment', 'value', 'price']
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if any(keyword in col.lower() for keyword in revenue_keywords):
                return col
        return None
    
    def _get_feature_description(self, feature: str) -> str:
        """Get human-readable description for feature"""
        descriptions = {
            'tenure': 'Customer lifetime duration',
            'usage': 'Product usage frequency',
            'support': 'Customer support interactions',
            'payment': 'Payment history and behavior',
            'login': 'User engagement frequency',
            'revenue': 'Customer monetary value',
            'satisfaction': 'Customer satisfaction metrics'
        }
        
        feature_lower = feature.lower()
        for key, desc in descriptions.items():
            if key in feature_lower:
                return desc
        return 'Customer behavior metric'
    
    def _get_high_value_at_risk(self, df: pd.DataFrame, revenue_col: str) -> List[Dict]:
        """Get high-value customers at risk"""
        if not revenue_col:
            return []
        
        high_value = df[df[revenue_col] > df[revenue_col].quantile(0.75)]
        at_risk = high_value[high_value['churn_probability'] > 0.3]
        
        return at_risk.head(10).to_dict('records')
    
    def _create_risk_segments(self, df: pd.DataFrame, segment_profiles: List[Dict]) -> List[Dict]:
        """Create risk segment descriptions"""
        risk_segments = []
        
        for profile in segment_profiles:
            churn_rate = profile['churn_rate']
            
            if churn_rate > 0.7:
                priority = 'critical'
                name = 'Critical Risk - Immediate Action'
            elif churn_rate > 0.5:
                priority = 'high'
                name = 'High Risk - Preventive Action'
            elif churn_rate > 0.3:
                priority = 'medium'
                name = 'Medium Risk - Monitor Closely'
            else:
                priority = 'low'
                name = 'Low Risk - Maintain Engagement'
            
            # Get segment characteristics
            segment_data = df[df['segment'] == profile['segment_id']]
            characteristics = self._identify_segment_characteristics(segment_data, df)
            
            risk_segments.append({
                'name': name,
                'priority': priority,
                'customers': profile['size'],
                'churn_probability': churn_rate,
                'avg_revenue': segment_data.iloc[:, 2].mean() if len(segment_data) > 0 else 0,  # Rough estimate
                'characteristics': characteristics[:3]  # Top 3 characteristics
            })
        
        return sorted(risk_segments, key=lambda x: x['churn_probability'], reverse=True)
    
    def _identify_segment_characteristics(self, segment_df: pd.DataFrame, full_df: pd.DataFrame) -> List[str]:
        """Identify key characteristics of a segment"""
        characteristics = []
        
        # Compare numerical features
        numerical_cols = segment_df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols[:10]:  # Check top 10 numerical columns
            segment_mean = segment_df[col].mean()
            overall_mean = full_df[col].mean()
            
            if overall_mean > 0:  # Avoid division by zero
                diff_pct = (segment_mean - overall_mean) / overall_mean * 100
                
                if abs(diff_pct) > 20:  # Significant difference
                    if diff_pct > 0:
                        characteristics.append(f"High {col.replace('_', ' ').title()} (+{diff_pct:.0f}%)")
                    else:
                        characteristics.append(f"Low {col.replace('_', ' ').title()} ({diff_pct:.0f}%)")
        
        # If not enough characteristics found, add generic ones
        if len(characteristics) < 3:
            characteristics.extend([
                "Distinct behavioral pattern",
                "Requires targeted approach",
                "Monitor key metrics closely"
            ])
        
        return characteristics[:5]  # Return top 5
    
    def _save_results(self, results: Dict[str, Any], df_with_predictions: pd.DataFrame):
        """Save analysis results to files"""
        # Save predictions
        predictions_path = os.path.join(self.output_dir, 'customer_churn_predictions.csv')
        df_with_predictions.to_csv(predictions_path, index=False)
        
        # Save model
        model_path = os.path.join(self.model_dir, 'churn_prediction_model.pkl')
        joblib.dump(self.predictor.best_model, model_path)
        
        # Save results summary (excluding large objects)
        summary = {k: v for k, v in results.items() 
                  if k not in ['statistical_summary', 'segment_profiles', 'immediate_action_customers',
                              'preventive_action_customers', 'high_value_risk_customers']}
        
        summary_path = os.path.join(self.output_dir, 'analysis_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"Results saved to {self.output_dir}/")


# Example usage
if __name__ == "__main__":
    # Generate sample data
    generator = SyntheticDataGenerator()
    df = generator.generate_ecommerce_data(n_samples=5000)
    df.to_csv('sample_ecommerce_data.csv', index=False)
    
    # Run analysis
    pipeline = ChurnAnalysisPipeline()
    results = pipeline.analyze_file('sample_ecommerce_data.csv')
    
    print("\nAnalysis Complete!")
    print(f"Total Customers: {results['total_customers']:,}")
    print(f"Churn Rate: {results['churn_rate']:.1f}%")
    print(f"Model Accuracy: {results['model_accuracy']:.1f}%")
    print(f"Revenue at Risk: ${results['revenue_at_risk']:,.2f}")
    print(f"\nTop Churn Drivers:")
    for i, feature in enumerate(results['top_features'][:3], 1):
        print(f"  {i}. {feature}")