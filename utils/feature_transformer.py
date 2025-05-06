import pandas as pd
import numpy as np
import pickle
import os
from sklearn.base import BaseEstimator, TransformerMixin
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FeatureTransformer:
    """
    A transformer class that can be fitted on training data and then
    used to transform new data consistently for prediction.
    """
    
    def __init__(self, target_only=False, categorical_threshold=10):
        """
        Initialize the FeatureTransformer
        
        Parameters:
        target_only (bool): If True, only generate categorical features suitable for target variables
                            If False, generate all possible feature transformations
        categorical_threshold (int): Maximum number of unique values for a column to be considered categorical
        """
        self.target_only = target_only
        self.categorical_threshold = categorical_threshold
        
        # Configuration will be stored here during fit
        self.column_types = {}
        self.column_stats = {}
        self.quantile_bins = {}
        self.category_mappings = {}
        self.generated_columns = []
        
    def fit(self, df):
        """
        Analyze the dataframe and store transformation parameters
        
        Parameters:
        df (pandas.DataFrame): Input dataframe to analyze
        
        Returns:
        self: The fitted transformer
        """
        print(f"Analyzing dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Store column information
        for col in df.columns:
            # Store data type
            self.column_types[col] = str(df[col].dtype)
            
            # For numeric columns, store statistics
            if pd.api.types.is_numeric_dtype(df[col]):
                self.column_stats[col] = {
                    'mean': df[col].mean(),
                    'median': df[col].median(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'std': df[col].std(),
                    'q1': df[col].quantile(0.25),
                    'q3': df[col].quantile(0.75)
                }
                
                # Store quantile bins for bucketing
                try:
                    if df[col].nunique() > 3:
                        # Store 4-quantile bins (quartiles)
                        quantiles = [0, 0.25, 0.5, 0.75, 1.0]
                        self.quantile_bins[f"{col}_category"] = [df[col].quantile(q) for q in quantiles]
                        
                        # Store 3-tier bins for trend
                        trend_quantiles = [0, 0.33, 0.67, 1.0]
                        self.quantile_bins[f"{col}_trend"] = [df[col].quantile(q) for q in trend_quantiles]
                        
                        # Store standard deviation bins
                        mean = self.column_stats[col]['mean']
                        std = self.column_stats[col]['std']
                        if std > 0:
                            self.quantile_bins[f"{col}_std_category"] = [
                                float('-inf'), mean - std, mean - 0.5*std, 
                                mean + 0.5*std, mean + std, float('inf')
                            ]
                except Exception as e:
                    print(f"Error creating bins for {col}: {str(e)}")
            
            # For categorical columns, store value mappings
            elif df[col].nunique() <= self.categorical_threshold:
                self.category_mappings[col] = df[col].value_counts().to_dict()
                
        # Track which columns will be generated
        self._identify_features_to_generate(df)
        
        return self
    
    def _identify_features_to_generate(self, df):
        """Identify which features will be generated"""
        features = []
        
        # For each numeric column, we'll create various derived features
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Standard features for all numeric columns (always include regardless of target_only)
                features.extend([
                    f"{col}_zscore",
                    f"{col}_is_outlier",
                    f"{col}_is_high",
                    f"{col}_above_median"
                ])
                
                # Categorical features for all numeric columns
                if df[col].nunique() > 3:
                    features.extend([
                        f"{col}_category",
                        f"{col}_trend"
                    ])
                    
                    if self.column_stats[col]['std'] > 0:
                        features.append(f"{col}_std_category")
                    
                    # Price-specific features
                    if 'price' in col.lower() or 'cost' in col.lower() or 'value' in col.lower():
                        features.append(f"{col}_tier")
                        features.append(f"{col}_price_strategy") 
                    
                    # Quantity-specific features
                    if 'quantity' in col.lower() or 'count' in col.lower() or 'stock' in col.lower():
                        features.append(f"{col}_stock_level")
                        
                    # Rating-specific features
                    if 'rating' in col.lower() or 'score' in col.lower() or 'satisfaction' in col.lower():
                        features.append(f"{col}_satisfaction")
                        if df[col].max() <= 10 and df[col].min() >= 0:
                            features.append(f"{col}_nps_category")
            
            # Date-specific features if not target_only
            if not self.target_only:
                # Check if column looks like a date
                if 'date' in col.lower() or 'time' in col.lower():
                    try:
                        # Try to convert to datetime
                        pd.to_datetime(df[col], errors='raise')
                        features.extend([
                            f"{col}_year",
                            f"{col}_month",
                            f"{col}_quarter",
                            f"{col}_is_weekend",
                            f"{col}_recency_days",
                            f"{col}_recency_category"
                        ])
                    except:
                        pass
        
        # Generate ratio features if not target_only
        if not self.target_only:
            # Find price and quantity columns for price-to-quantity ratios
            price_cols = [col for col in df.columns 
                         if pd.api.types.is_numeric_dtype(df[col]) and
                         ('price' in col.lower() or 'cost' in col.lower() or 'value' in col.lower())]
            
            qty_cols = [col for col in df.columns 
                       if pd.api.types.is_numeric_dtype(df[col]) and
                       ('quantity' in col.lower() or 'qty' in col.lower() or 'count' in col.lower())]
            
            # Create ratio features
            for p_col in price_cols:
                for q_col in qty_cols:
                    ratio_col = f"{p_col}_per_{q_col}"
                    features.append(ratio_col)
                    features.append(f"{ratio_col}_category")
        
        # Store the list of features we'll generate
        self.generated_columns = features
        print(f"Identified {len(features)} features to generate")
        
    def transform(self, df, return_merged=True):
        """
        Transform new data using the fitted parameters
        
        Parameters:
        df (pandas.DataFrame): New data to transform
        return_merged (bool): If True, returns the original df merged with new features
        
        Returns:
        pandas.DataFrame: Transformed data
        """
        # Make a copy to avoid modifying the input
        input_df = df.copy()
        
        # Check if the transformer has been fitted
        if not hasattr(self, 'generated_columns') or not self.generated_columns:
            print("Warning: Transformer has not been fitted yet. Call fit() first.")
            return input_df if return_merged else pd.DataFrame(index=input_df.index)
            
        # Generate features manually
        feature_generator = FeatureGeneratorTransformer(self)
        generated_features = feature_generator.transform(input_df)
        
        # If return_merged is True, merge with original data
        if return_merged:
            # Combine original data with generated features
            result_df = pd.concat([input_df, generated_features], axis=1)
            return result_df
        else:
            # Return only the generated features
            return generated_features
    
    def fit_transform(self, df, return_merged=True):
        """
        Fit the transformer to the data and then transform it
        
        Parameters:
        df (pandas.DataFrame): Data to fit and transform
        return_merged (bool): If True, returns the original df merged with new features
        
        Returns:
        pandas.DataFrame: Transformed data
        """
        # First fit the transformer
        self.fit(df)
        
        # Then transform the data
        # Important: Only call transform after fit is complete
        return self.transform(df, return_merged=return_merged)
    
    def save(self, path):
        """
        Save the fitted transformer to disk
        
        Parameters:
        path (str): Path to save the transformer
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save to disk
        with open(path, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Transformer saved to {path}")
    
    @classmethod
    def load(cls, path):
        """
        Load a saved transformer from disk
        
        Parameters:
        path (str): Path to the saved transformer
        
        Returns:
        FeatureTransformer: Loaded transformer
        """
        with open(path, 'rb') as f:
            transformer = pickle.load(f)
        
        print(f"Transformer loaded from {path}")
        return transformer


class FeatureGeneratorTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for generating features from preprocessed data"""
    
    def __init__(self, feature_transformer):
        self.feature_transformer = feature_transformer
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        """Generate features based on the feature_transformer configuration"""
        # Create a dataframe from X
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            # Assume it's a numpy array - need to recreate column names
            df = pd.DataFrame(X)
            
        # Apply transformations based on the feature_transformer configuration
        result = pd.DataFrame(index=df.index)
        
        # Generate each feature that was identified during fit
        for feature in self.feature_transformer.generated_columns:
            # Parse the feature name to identify the source column and transformation
            parts = feature.split('_')
            if len(parts) >= 2:
                # The source column could have underscores in it
                transform_type = parts[-1]
                source_col = '_'.join(parts[:-1])
                
                # Special case for ratio features
                if 'per' in parts:
                    per_index = parts.index('per')
                    if per_index > 0 and per_index < len(parts) - 1:
                        # This is a ratio feature
                        p_col = '_'.join(parts[:per_index])
                        q_col = '_'.join(parts[per_index+1:])
                        if p_col in df.columns and q_col in df.columns:
                            if transform_type == 'category':
                                result[feature] = self._create_ratio_category(df, p_col, q_col)
                            else:
                                result[feature] = self._create_ratio(df, p_col, q_col)
                        continue
                
                # Check if the source column exists
                if source_col in df.columns:
                    # Apply the appropriate transformation
                    if transform_type == 'category':
                        result[feature] = self._create_category(df, source_col)
                    elif transform_type == 'tier':
                        result[feature] = self._create_tier(df, source_col)
                    elif transform_type == 'zscore':
                        result[feature] = self._create_zscore(df, source_col)
                    elif transform_type == 'is_outlier':
                        result[feature] = self._create_is_outlier(df, source_col)
                    elif transform_type == 'is_high':
                        result[feature] = self._create_is_high(df, source_col)
                    elif transform_type == 'above_median':
                        result[feature] = self._create_above_median(df, source_col)
                    elif transform_type == 'trend':
                        result[feature] = self._create_trend(df, source_col)
                    elif transform_type == 'std_category':
                        result[feature] = self._create_std_category(df, source_col)
                    elif transform_type == 'price_strategy':
                        result[feature] = self._create_price_strategy(df, source_col)
                    elif transform_type == 'stock_level':
                        result[feature] = self._create_stock_level(df, source_col)
                    elif transform_type == 'satisfaction':
                        result[feature] = self._create_satisfaction(df, source_col)
                    elif transform_type == 'nps_category':
                        result[feature] = self._create_nps_category(df, source_col)
                    elif transform_type == 'year' and self._is_date_column(df, source_col):
                        result[feature] = self._create_date_year(df, source_col)
                    elif transform_type == 'month' and self._is_date_column(df, source_col):
                        result[feature] = self._create_date_month(df, source_col)
                    elif transform_type == 'quarter' and self._is_date_column(df, source_col):
                        result[feature] = self._create_date_quarter(df, source_col)
                    elif transform_type == 'is_weekend' and self._is_date_column(df, source_col):
                        result[feature] = self._create_date_is_weekend(df, source_col)
                    elif transform_type == 'days' and 'recency' in feature and self._is_date_column(df, source_col):
                        result[feature] = self._create_date_recency_days(df, source_col)
                    elif transform_type == 'category' and 'recency' in feature and self._is_date_column(df, source_col):
                        result[feature] = self._create_date_recency_category(df, source_col)
                
        return result
    
    def _is_date_column(self, df, col):
        """Check if a column can be converted to datetime"""
        try:
            pd.to_datetime(df[col], errors='raise')
            return True
        except:
            return False
    
    def _create_category(self, df, col):
        """Create category based on quartiles"""
        if f"{col}_category" in self.feature_transformer.quantile_bins:
            bins = self.feature_transformer.quantile_bins[f"{col}_category"]
            return pd.cut(df[col], bins=bins, labels=['Low', 'Medium', 'High', 'Very High'], include_lowest=True)
        return pd.Series(index=df.index)
    
    def _create_tier(self, df, col):
        """Create price tiers"""
        if col in self.feature_transformer.column_stats:
            # Use min/max to create 4 tiers
            min_val = self.feature_transformer.column_stats[col]['min']
            max_val = self.feature_transformer.column_stats[col]['max']
            step = (max_val - min_val) / 4
            bins = [min_val, min_val + step, min_val + 2*step, min_val + 3*step, max_val]
            return pd.cut(df[col], bins=bins, labels=['Low', 'Medium', 'High', 'Premium'], include_lowest=True)
        return pd.Series(index=df.index)
    
    def _create_zscore(self, df, col):
        """Create z-score based on stored mean and std"""
        if col in self.feature_transformer.column_stats:
            mean = self.feature_transformer.column_stats[col]['mean']
            std = self.feature_transformer.column_stats[col]['std']
            if std > 0:
                return (df[col] - mean) / std
        return pd.Series(index=df.index)
    
    def _create_is_outlier(self, df, col):
        """Create outlier flag based on z-score"""
        zscore = self._create_zscore(df, col)
        return (abs(zscore) > 2).astype(int)
    
    def _create_is_high(self, df, col):
        """Create high-value flag based on median"""
        if col in self.feature_transformer.column_stats:
            median = self.feature_transformer.column_stats[col]['median']
            return (df[col] > median).astype(int)
        return pd.Series(index=df.index)
    
    def _create_above_median(self, df, col):
        """Create above-median flag"""
        return self._create_is_high(df, col)
    
    def _create_trend(self, df, col):
        """Create trend indicator"""
        if f"{col}_trend" in self.feature_transformer.quantile_bins:
            bins = self.feature_transformer.quantile_bins[f"{col}_trend"]
            return pd.cut(df[col], bins=bins, labels=['Decreasing', 'Stable', 'Increasing'], include_lowest=True)
        return pd.Series(index=df.index)
    
    def _create_std_category(self, df, col):
        """Create category based on standard deviations"""
        if f"{col}_std_category" in self.feature_transformer.quantile_bins:
            bins = self.feature_transformer.quantile_bins[f"{col}_std_category"]
            return pd.cut(df[col], bins=bins, labels=['Very Low', 'Low', 'Average', 'High', 'Very High'], include_lowest=True)
        return pd.Series(index=df.index)
    
    def _create_price_strategy(self, df, col):
        """Create price strategy categories"""
        if col in self.feature_transformer.column_stats:
            # Use percentiles for price strategy
            p = [0, 25, 50, 75, 90, 100]
            stats = self.feature_transformer.column_stats[col]
            
            # Create bins based on the percentiles
            min_val = stats['min']
            max_val = stats['max']
            q1 = stats['q1']
            median = stats['median']
            q3 = stats['q3']
            p90 = min_val + 0.9 * (max_val - min_val)  # Approximate 90th percentile
            
            bins = [min_val, q1, median, q3, p90, max_val]
            return pd.cut(df[col], bins=bins, labels=['Budget', 'Value', 'Standard', 'Premium', 'Luxury'], include_lowest=True)
        return pd.Series(index=df.index)
    
    def _create_stock_level(self, df, col):
        """Create stock level categories"""
        if col in self.feature_transformer.column_stats:
            # Use percentiles for stock levels
            stats = self.feature_transformer.column_stats[col]
            
            # Create bins based on percentiles
            min_val = stats['min']
            max_val = stats['max']
            p20 = min_val + 0.2 * (max_val - min_val)
            p50 = min_val + 0.5 * (max_val - min_val)
            p80 = min_val + 0.8 * (max_val - min_val)
            
            bins = [min_val, p20, p50, p80, max_val]
            return pd.cut(df[col], bins=bins, labels=['Critical', 'Low', 'Adequate', 'Excess'], include_lowest=True)
        return pd.Series(index=df.index)
    
    def _create_satisfaction(self, df, col):
        """Create satisfaction categories"""
        if col in self.feature_transformer.column_stats:
            stats = self.feature_transformer.column_stats[col]
            min_val = stats['min']
            max_val = stats['max']
            
            # For 5-star or 10-point scales
            if max_val <= 5:
                bins = [0, 1.5, 2.5, 3.5, 4.5, 5]
                labels = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
                return pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
            elif max_val <= 10:
                bins = [0, 2, 4, 6, 8, 10]
                labels = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
                return pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
            else:
                # For other scales, use quantile-based approach
                step = (max_val - min_val) / 5
                bins = [min_val, min_val + step, min_val + 2*step, min_val + 3*step, min_val + 4*step, max_val]
                labels = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
                return pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        return pd.Series(index=df.index)
    
    def _create_nps_category(self, df, col):
        """Create NPS categories (Detractor, Passive, Promoter)"""
        if col in self.feature_transformer.column_stats and self.feature_transformer.column_stats[col]['max'] <= 10:
            # Standard NPS bucketing
            bins = [0, 6, 8, 10]
            labels = ['Detractor', 'Passive', 'Promoter']
            return pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
        return pd.Series(index=df.index)
    
    def _create_ratio(self, df, p_col, q_col):
        """Create ratio between two columns"""
        if p_col in df.columns and q_col in df.columns:
            # Avoid division by zero
            return df[p_col] / df[q_col].replace(0, np.nan)
        return pd.Series(index=df.index)
    
    def _create_ratio_category(self, df, p_col, q_col):
        """Create categories for ratios"""
        ratio = self._create_ratio(df, p_col, q_col)
        
        # Use quartiles for categories
        try:
            # Using np.nanquantile to handle NaN values
            q1 = np.nanquantile(ratio, 0.25)
            median = np.nanquantile(ratio, 0.5)
            q3 = np.nanquantile(ratio, 0.75)
            
            bins = [float('-inf'), q1, median, q3, float('inf')]
            return pd.cut(ratio, bins=bins, labels=['Low', 'Medium', 'High', 'Very High'])
        except:
            return pd.Series(index=df.index)
    
    def _create_date_year(self, df, col):
        """Extract year from date column"""
        try:
            dates = pd.to_datetime(df[col])
            return dates.dt.year
        except:
            return pd.Series(index=df.index)
    
    def _create_date_month(self, df, col):
        """Extract month from date column"""
        try:
            dates = pd.to_datetime(df[col])
            return dates.dt.month
        except:
            return pd.Series(index=df.index)
    
    def _create_date_quarter(self, df, col):
        """Extract quarter from date column"""
        try:
            dates = pd.to_datetime(df[col])
            return dates.dt.quarter
        except:
            return pd.Series(index=df.index)
    
    def _create_date_is_weekend(self, df, col):
        """Create weekend flag from date column"""
        try:
            dates = pd.to_datetime(df[col])
            return dates.dt.dayofweek.isin([5, 6]).astype(int)
        except:
            return pd.Series(index=df.index)
    
    def _create_date_recency_days(self, df, col):
        """Calculate days since most recent date"""
        try:
            dates = pd.to_datetime(df[col])
            if 'reference_date' in self.feature_transformer.column_stats.get(col, {}):
                reference_date = self.feature_transformer.column_stats[col]['reference_date']
            else:
                # Use current date as reference
                reference_date = pd.Timestamp.now()
            
            return (reference_date - dates).dt.days
        except:
            return pd.Series(index=df.index)
    
    def _create_date_recency_category(self, df, col):
        """Create recency categories from days"""
        recency_days = self._create_date_recency_days(df, col)
        bins = [0, 30, 90, 180, float('inf')]
        labels = ['Very Recent', 'Recent', 'Older', 'Historical']
        return pd.cut(recency_days, bins=bins, labels=labels)