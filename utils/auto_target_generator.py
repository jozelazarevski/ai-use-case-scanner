import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
import re
from datetime import datetime
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

class AutoTargetFeatureGenerator:
    """
    Automatically generates target features from datasets for machine learning models.
    The generator is column-agnostic and uses pattern detection to identify appropriate transformations.
    
    This class properly handles missing values throughout the feature generation process
    and can create classification targets from numerical data to support business decision-making.
    """
    
    def __init__(self):
        # Column type identification patterns
        self.patterns = {
            'id': r'(?i)(\b|_)(id|identifier|key)(\b|_)',
            'date': r'(?i)(\b|_)(date|time|timestamp|dt|day|month|year)(\b|_)',
            'price': r'(?i)(\b|_)(price|cost|amount|value|salary|income|spend|budget|balance)(\b|_)',
            'quantity': r'(?i)(\b|_)(quantity|qty|count|number|units|rooms|population|households)(\b|_)',
            'rating': r'(?i)(\b|_)(rating|score|satisfaction|quality|performance|review)(\b|_)',
            'duration': r'(?i)(\b|_)(duration|time|period|days|hours|seconds|minutes)(\b|_)',
            'location': r'(?i)(\b|_)(location|address|city|country|latitude|longitude|geo)(\b|_)',
            'stage': r'(?i)(\b|_)(stage|phase|status|step|level|progress)(\b|_)',
            'method': r'(?i)(\b|_)(method|channel|type|category|device|browser|source)(\b|_)',
            'text': r'(?i)(\b|_)(name|text|description|title|comment|label|audience)(\b|_)'
        }
        
    def identify_column_types(self, df):
        """Identify the likely type of each column based on name and content."""
        column_types = {}
        
        for col in df.columns:
            # Check the column name against patterns
            col_type = None
            for type_name, pattern in self.patterns.items():
                if re.search(pattern, col):
                    col_type = type_name
                    break
            
            # If no pattern match, infer from data
            if col_type is None:
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check value ranges to guess type
                    if df[col].nunique() < 10 and df[col].min() >= 0 and df[col].max() <= 5:
                        col_type = 'rating'
                    elif df[col].min() >= 0 and df[col].max() < 1000:
                        col_type = 'quantity'
                    else:
                        col_type = 'numeric'
                elif pd.api.types.is_datetime64_dtype(df[col]) or self._looks_like_date(df[col]):
                    col_type = 'date'
                elif df[col].nunique() / len(df) < 0.01:  # Less than 1% unique values
                    col_type = 'category'
                else:
                    col_type = 'text'
                    
            column_types[col] = col_type
            
        return column_types
    
    def _looks_like_date(self, series):
        """Check if a series looks like it contains dates."""
        if series.dtype == 'object':
            # Sample a few values to check if they're date-like
            sample = series.dropna().sample(min(5, len(series.dropna()))).astype(str)
            date_patterns = [
                r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
                r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
                r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
                r'\d{4}/\d{2}/\d{2}'   # YYYY/MM/DD
            ]
            
            for value in sample:
                if any(re.match(pattern, value) for pattern in date_patterns):
                    return True
        return False
        
    def generate_features(self, df, output_file=None):
        """
        Main method to generate new features from the dataframe
        
        Parameters:
        df (pandas.DataFrame): Input dataframe
        output_file (str, optional): Path to save transformed dataframe
        
        Returns:
        pandas.DataFrame: Original dataframe with added features
        """
        print(f"Analyzing dataset with {df.shape[0]} rows and {df.shape[1]} columns")
        
        # Make a copy to avoid modifying the original
        transformed_df = df.copy()
        
        # Clean the dataframe
        transformed_df = self._clean_dataframe(transformed_df)
        
        # Identify column types
        column_types = self.identify_column_types(transformed_df)
        print("\nIdentified column types:")
        for col_type in set(column_types.values()):
            cols = [col for col, type_name in column_types.items() if type_name == col_type]
            print(f"- {col_type}: {len(cols)} columns")
        
        # Create features based on column types
        features_added = []
        
        # Process numeric columns that aren't already identified as specific types
        generic_numeric_cols = [col for col in transformed_df.columns 
                               if pd.api.types.is_numeric_dtype(transformed_df[col]) 
                               and col not in column_types]
        
        # Add these as numeric type
        for col in generic_numeric_cols:
            column_types[col] = 'numeric'
        
        print(f"- Found {len(generic_numeric_cols)} additional numeric columns")
        
        # Process price/monetary columns
        price_cols = [col for col, type_name in column_types.items() 
                     if type_name == 'price' and pd.api.types.is_numeric_dtype(transformed_df[col])]
        for col in price_cols:
            features_added.extend(self._create_price_features(transformed_df, col))
            
        # Process quantity columns
        qty_cols = [col for col, type_name in column_types.items() 
                   if type_name == 'quantity' and pd.api.types.is_numeric_dtype(transformed_df[col])]
        for col in qty_cols:
            features_added.extend(self._create_quantity_features(transformed_df, col))
            
        # Process rating columns
        rating_cols = [col for col, type_name in column_types.items() 
                      if type_name == 'rating' and pd.api.types.is_numeric_dtype(transformed_df[col])]
        for col in rating_cols:
            features_added.extend(self._create_rating_features(transformed_df, col))
        
        # Process generic numeric columns (those that weren't categorized as price, quantity, rating)
        for col in generic_numeric_cols:
            # Convert to quartile-based categories 
            if transformed_df[col].nunique() > 3 and not transformed_df[col].isnull().all():
                try:
                    col_no_missing = transformed_df[col].dropna()
                    if len(col_no_missing) > 10:  # Only if we have enough data
                        transformed_df[f"{col}_category"] = pd.qcut(
                            transformed_df[col], 
                            4, 
                            labels=['Low', 'Medium', 'High', 'Very High'],
                            duplicates='drop'
                        )
                        features_added.append(f"{col}_category")
                        
                        # Create binary version (above/below median)
                        median_val = transformed_df[col].median()
                        transformed_df[f"{col}_above_median"] = (transformed_df[col] > median_val).astype(int)
                        features_added.append(f"{col}_above_median")
                        
                        # Create trend indicator (could be calculated against previous period in time series)
                        transformed_df[f"{col}_trend"] = pd.cut(
                            transformed_df[col],
                            bins=[float('-inf'), transformed_df[col].quantile(0.33),
                                  transformed_df[col].quantile(0.67), float('inf')],
                            labels=['Decreasing', 'Stable', 'Increasing']
                        )
                        features_added.append(f"{col}_trend")
                except Exception as e:
                    print(f"  Could not create categories for {col}: {str(e)}")
            
        # Process date columns
        date_cols = [col for col, type_name in column_types.items() if type_name == 'date']
        for col in date_cols:
            # Convert to datetime if it's not already
            if not pd.api.types.is_datetime64_dtype(transformed_df[col]):
                try:
                    transformed_df[col] = pd.to_datetime(transformed_df[col], errors='coerce')
                    features_added.extend(self._create_date_features(transformed_df, col))
                except Exception as e:
                    print(f"  Could not convert {col} to datetime: {str(e)}")
            else:
                features_added.extend(self._create_date_features(transformed_df, col))
        
        # Create ratio features between related columns
        features_added.extend(self._create_ratio_features(transformed_df, column_types))
        
        # Create aggregate features for related columns
        features_added.extend(self._create_aggregate_features(transformed_df, column_types))
        
        # Create text-based features if any text columns exist
        text_cols = [col for col, type_name in column_types.items() if type_name == 'text']
        if text_cols:
            features_added.extend(self._create_text_features(transformed_df, text_cols))
        
        # Create location-based features
        loc_cols = [col for col, type_name in column_types.items() if type_name == 'location']
        if len(loc_cols) >= 2:
            features_added.extend(self._create_location_features(transformed_df, loc_cols))
            
        # Apply statistical bucketing for segment identification
        numeric_cols = [col for col in transformed_df.columns 
                        if pd.api.types.is_numeric_dtype(transformed_df[col]) 
                        and col not in ['id', 'identifier', 'key']]
        if len(numeric_cols) >= 2:
            features_added.extend(self._create_statistical_segments(transformed_df, numeric_cols))
            
        # Create a sample ML-ready dataset with potentially useful model targets
        self._create_ml_ready_examples(transformed_df, features_added)
        
        print(f"\nAdded {len(features_added)} new target features:")
        for feature in features_added:
            print(f"- {feature}")
            
        # Save to file if requested
        if output_file:
            transformed_df.to_csv(output_file, index=False)
            print(f"\nTransformed dataset saved to {output_file}")
            
        return transformed_df
        
    def _create_ml_ready_examples(self, df, features_added):
        """Create examples of ready-to-use ML models with the generated features."""
        
        # Find categorical target columns we've created
        category_cols = [col for col in features_added if col.endswith('_category') 
                        or col.endswith('_tier') or col.endswith('_segment')]
        
        if not category_cols:
            return
            
        print("\nExample ML model targets ready for training:")
        print("These targets can be used for classification models instead of regression")
        
        for i, target_col in enumerate(category_cols[:5]):  # Show up to 5 examples
            # Count classes to ensure it's a meaningful target
            class_counts = df[target_col].value_counts()
            
            # Only show if we have a reasonably balanced distribution
            if len(class_counts) >= 2 and class_counts.min() >= 5:
                print(f"{i+1}. '{target_col}' - Classification target with {len(class_counts)} classes:")
                
                # Show class distribution
                for cls, count in class_counts.items():
                    pct = count / len(df) * 100
                    print(f"   - {cls}: {count} instances ({pct:.1f}%)")
                
                # Suggest model type
                if len(class_counts) == 2:
                    print("   Suggested model: Binary classification with HistGradientBoostingClassifier")
                else:
                    print("   Suggested model: Multi-class classification with HistGradientBoostingClassifier")
                
                print("   Sample code:")
                print(f"""   ```python
   # Define features and target
   X = df.drop('{target_col}', axis=1)
   y = df['{target_col}']
   
   # Create a pipeline with imputer for missing values
   from sklearn.pipeline import Pipeline
   from sklearn.impute import SimpleImputer
   from sklearn.ensemble import HistGradientBoostingClassifier
   
   model = Pipeline([
       ('imputer', SimpleImputer(strategy='median')),
       ('classifier', HistGradientBoostingClassifier(random_state=42))
   ])
   
   # Train the model
   model.fit(X, y)
   ```""")
    
    def _clean_dataframe(self, df):
        """Clean the dataframe by handling missing values and data types."""
        # Handle potentially problematic column names
        df.columns = [col.strip() for col in df.columns]
        
        # Convert obvious date columns to datetime
        for col in df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                try:
                    df[col] = pd.to_datetime(df[col], errors='ignore')
                except:
                    pass
                    
        return df
    
    def _create_price_features(self, df, col):
        """Create features from price/monetary columns using robust statistical methods."""
        features_added = []
        
        try:
            # Handle missing values safely
            col_data = df[col].dropna()
            
            # Skip if not enough data
            if len(col_data) < 5:
                return features_added
                
            # Create price tiers with robust handling of edge cases
            tier_col = f"{col}_tier"
            try:
                # Try quantile-based bucketing first
                df[tier_col] = pd.qcut(
                    df[col], 
                    4, 
                    labels=['Low', 'Medium', 'High', 'Premium'],
                    duplicates='drop'
                )
            except ValueError as e:
                # Fall back to equal-width bins if quantile-based fails
                min_val = df[col].min()
                max_val = df[col].max()
                if max_val > min_val:
                    df[tier_col] = pd.cut(
                        df[col],
                        bins=[min_val, min_val + (max_val-min_val)*0.25, 
                              min_val + (max_val-min_val)*0.5, 
                              min_val + (max_val-min_val)*0.75, 
                              max_val],
                        labels=['Low', 'Medium', 'High', 'Premium'],
                        include_lowest=True
                    )
                else:
                    # If all values are the same, just create a single category
                    df[tier_col] = 'Medium'
            
            features_added.append(tier_col)
            
            # Create binary features using robust statistics (median instead of mean)
            median = df[col].median()
            df[f"{col}_is_high"] = (df[col] > median).astype(int)
            features_added.append(f"{col}_is_high")
            
            # Use quantile-based bucketing for more nuanced categorization
            try:
                quartiles = df[col].quantile([0.25, 0.5, 0.75])
                df[f"{col}_quantile_segment"] = pd.cut(
                    df[col],
                    bins=[float('-inf'), quartiles[0.25], quartiles[0.5], quartiles[0.75], float('inf')],
                    labels=['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%']
                )
                features_added.append(f"{col}_quantile_segment")
            except:
                pass
                
            # Create business growth indicators
            try:
                # Calculate percentiles for business-relevant thresholds
                p90 = df[col].quantile(0.9)
                p10 = df[col].quantile(0.1)
                
                # Create high-value flag (top 10%)
                df[f"{col}_high_value"] = (df[col] >= p90).astype(int)
                features_added.append(f"{col}_high_value")
                
                # Create low-value flag (bottom 10%)
                df[f"{col}_low_value"] = (df[col] <= p10).astype(int)
                features_added.append(f"{col}_low_value")
            except:
                pass
                
            # Create z-score with robust handling of outliers
            if df[col].std() > 0:  # Only if there's variation
                # Use more robust Z-score calculation
                df[f"{col}_zscore"] = stats.zscore(df[col], nan_policy='omit')
                df[f"{col}_is_outlier"] = (abs(df[f"{col}_zscore"]) > 2).astype(int)
                features_added.append(f"{col}_is_outlier")
                
                # Create categorical feature based on standard deviations
                df[f"{col}_std_category"] = pd.cut(
                    df[f"{col}_zscore"],
                    bins=[-float('inf'), -2, -1, 1, 2, float('inf')],
                    labels=['Very Low', 'Low', 'Average', 'High', 'Very High']
                )
                features_added.append(f"{col}_std_category")
        except Exception as e:
            print(f"  Error creating price features for {col}: {str(e)}")
            
        return features_added
    
    def _create_quantity_features(self, df, col):
        """Create features from quantity columns using robust statistical methods."""
        features_added = []
        
        try:
            # Skip if column has all missing values
            if df[col].isna().all():
                return features_added
                
            # Create quantity categories with robust error handling
            try:
                df[f"{col}_category"] = pd.qcut(
                    df[col], 
                    3, 
                    labels=['Low', 'Medium', 'High'],
                    duplicates='drop'
                )
                features_added.append(f"{col}_category")
            except ValueError:
                # Fallback to custom bins if quantile-based approach fails
                if df[col].nunique() > 1:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[f"{col}_category"] = pd.cut(
                        df[col],
                        bins=[min_val, min_val + (max_val-min_val)/3, min_val + 2*(max_val-min_val)/3, max_val],
                        labels=['Low', 'Medium', 'High'],
                        include_lowest=True
                    )
                    features_added.append(f"{col}_category")
                else:
                    # If all values are the same, just assign 'Medium'
                    df[f"{col}_category"] = 'Medium'
                    features_added.append(f"{col}_category")
            
            # Create binary flags using median (more robust than mean)
            median_val = df[col].median()
            df[f"{col}_above_median"] = (df[col] > median_val).astype(int)
            features_added.append(f"{col}_above_median")
            
            # Create more granular quantity segments using statistical percentiles
            try:
                # Define business-relevant percentiles
                percentiles = [0, 20, 40, 60, 80, 100]
                bins = [df[col].quantile(p/100) for p in percentiles]
                
                # Check if we have enough distinct values
                if len(set(bins)) >= 3:
                    df[f"{col}_percentile_segment"] = pd.cut(
                        df[col],
                        bins=bins,
                        labels=['Minimal', 'Low', 'Moderate', 'High', 'Extensive'],
                        include_lowest=True
                    )
                    features_added.append(f"{col}_percentile_segment")
            except Exception as e:
                print(f"  Couldn't create percentile segments for {col}: {str(e)}")
                
            # Create a business-relevant "stock level" indicator
            if 'stock' in col.lower() or 'inventory' in col.lower() or 'qty' in col.lower():
                # Define threshold for "low stock" - bottom 15%
                low_threshold = df[col].quantile(0.15)
                high_threshold = df[col].quantile(0.85)
                
                df[f"{col}_stock_level"] = pd.cut(
                    df[col],
                    bins=[0, low_threshold, high_threshold, float('inf')],
                    labels=['Low Stock', 'Adequate', 'Excess'],
                    include_lowest=True
                )
                features_added.append(f"{col}_stock_level")
                
                # Create binary flags for inventory management
                df[f"{col}_restock_needed"] = (df[col] <= low_threshold).astype(int)
                features_added.append(f"{col}_restock_needed")
                
        except Exception as e:
            print(f"  Error creating quantity features for {col}: {str(e)}")
            
        return features_added
    
    def _create_rating_features(self, df, col):
        """Create features from rating columns."""
        features_added = []
        
        # Normalize ratings to 0-1 scale
        min_val, max_val = df[col].min(), df[col].max()
        if min_val != max_val:  # Avoid division by zero
            df[f"{col}_normalized"] = (df[col] - min_val) / (max_val - min_val)
            features_added.append(f"{col}_normalized")
        
        # Create satisfaction categories
        if df[col].nunique() > 3:
            # Use quantiles for more diverse ratings
            df[f"{col}_satisfaction"] = pd.qcut(df[col], 4, 
                                               labels=['Poor', 'Fair', 'Good', 'Excellent'])
        else:
            # For binary or tertiary ratings, use simple mapping
            mapping = {
                1: 'Poor',
                2: 'Fair',
                3: 'Good',
                4: 'Good',
                5: 'Excellent'
            }
            df[f"{col}_satisfaction"] = df[col].map(lambda x: mapping.get(x, 'Unknown'))
        
        features_added.append(f"{col}_satisfaction")
        
        return features_added
    
    def _create_date_features(self, df, col):
        """Create features from date columns."""
        features_added = []
        
        # Extract basic components
        df[f"{col}_year"] = df[col].dt.year
        df[f"{col}_month"] = df[col].dt.month
        df[f"{col}_quarter"] = df[col].dt.quarter
        df[f"{col}_is_weekend"] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
        
        features_added.extend([f"{col}_year", f"{col}_month", f"{col}_quarter", f"{col}_is_weekend"])
        
        # Create recency feature (days since most recent date in dataset)
        if not df[col].isna().all():
            max_date = df[col].max()
            df[f"{col}_recency_days"] = (max_date - df[col]).dt.days
            features_added.append(f"{col}_recency_days")
            
            # Create recency categories
            df[f"{col}_recency_category"] = pd.cut(df[f"{col}_recency_days"], 
                                                 bins=[0, 30, 90, 180, float('inf')],
                                                 labels=['Very Recent', 'Recent', 'Older', 'Historical'])
            features_added.append(f"{col}_recency_category")
        
        return features_added
    
    def _create_ratio_features(self, df, column_types):
        """Create ratio features between related columns with proper handling of missing values."""
        features_added = []
        
        # Find price and quantity columns for price-to-quantity ratios
        price_cols = [col for col, type_name in column_types.items() 
                     if type_name == 'price' and pd.api.types.is_numeric_dtype(df[col])]
        qty_cols = [col for col, type_name in column_types.items() 
                   if type_name == 'quantity' and pd.api.types.is_numeric_dtype(df[col])]
        
        # Create price-to-quantity ratios (e.g., price per unit, cost per item)
        for price_col in price_cols:
            for qty_col in qty_cols:
                # Skip if columns don't seem related (based on name similarity)
                price_base = price_col.split('_')[0] if '_' in price_col else price_col
                qty_base = qty_col.split('_')[0] if '_' in qty_col else qty_col
                
                try:
                    # Create ratio with proper handling of zeros and NaNs
                    ratio_col = f"{price_col}_per_{qty_col}"
                    
                    # Safely create the ratio, handling both NaNs and zero divisions
                    df[ratio_col] = df[price_col].div(df[qty_col].replace(0, np.nan))
                    
                    # Handle NaN values - use median but only if we have enough data
                    if df[ratio_col].notna().sum() > 0:
                        median_value = df[ratio_col].median()
                        df[ratio_col] = df[ratio_col].fillna(median_value)
                        
                        # Create ratio categories with proper NaN handling
                        try:
                            df[f"{ratio_col}_category"] = pd.qcut(
                                df[ratio_col].fillna(median_value), 
                                4, 
                                labels=['Low', 'Medium', 'High', 'Premium']
                            )
                            features_added.extend([ratio_col, f"{ratio_col}_category"])
                        except ValueError as e:
                            # Handle case where qcut fails (e.g., not enough distinct values)
                            print(f"  Could not create categories for {ratio_col}: {str(e)}")
                            
                            # Try a simpler approach with custom bins
                            median = df[ratio_col].median()
                            q1 = df[ratio_col].quantile(0.25)
                            q3 = df[ratio_col].quantile(0.75)
                            
                            df[f"{ratio_col}_category"] = pd.cut(
                                df[ratio_col],
                                bins=[float('-inf'), q1, median, q3, float('inf')],
                                labels=['Low', 'Medium', 'High', 'Premium']
                            )
                            features_added.extend([ratio_col, f"{ratio_col}_category"])
                except Exception as e:
                    print(f"  Error creating ratio feature {price_col}/{qty_col}: {str(e)}")
        
        return features_added
    
    def _create_aggregate_features(self, df, column_types):
        """Create aggregate features by grouping related data."""
        features_added = []
        
        # Identify potential grouping columns (id columns that might represent entities)
        id_cols = [col for col, type_name in column_types.items() if type_name == 'id']
        
        # Skip if no suitable ID columns found
        if not id_cols:
            return features_added
        
        # Find price and quantity columns for aggregation
        numeric_cols = [col for col in df.columns 
                       if pd.api.types.is_numeric_dtype(df[col]) 
                       and not col.endswith('_id') 
                       and col not in id_cols]
        
        # Select a primary ID column (customer, product, etc.)
        for id_col in id_cols:
            # Skip if this ID has only unique values (primary keys)
            if df[id_col].nunique() == len(df):
                continue
                
            # For each numeric column, create aggregations
            for num_col in numeric_cols[:5]:  # Limit to 5 numeric columns to avoid explosion of features
                # Skip columns that are likely IDs
                if num_col.endswith('_id') or df[num_col].nunique() > 0.8 * len(df):
                    continue
                
                # Create aggregations
                agg_funcs = ['mean', 'sum']
                
                for agg_func in agg_funcs:
                    # Create the aggregation
                    agg_values = df.groupby(id_col)[num_col].transform(agg_func)
                    agg_col = f"{num_col}_by_{id_col}_{agg_func}"
                    df[agg_col] = agg_values
                    
                    # Create categories from the aggregation
                    df[f"{agg_col}_category"] = pd.qcut(df[agg_col], 4, 
                                                      labels=['Low', 'Medium', 'High', 'Very High'])
                    
                    features_added.extend([agg_col, f"{agg_col}_category"])
            
            # Limit to one ID column to avoid feature explosion
            break
                    
        return features_added
    
    def _create_text_features(self, df, text_cols):
        """Create features from text columns."""
        features_added = []
        
        # For now, just create simple text length features
        for col in text_cols[:3]:  # Limit to 3 text columns
            if df[col].dtype == 'object':
                df[f"{col}_length"] = df[col].astype(str).apply(len)
                df[f"{col}_length_category"] = pd.qcut(df[f"{col}_length"], 3, 
                                                     labels=['Short', 'Medium', 'Long'])
                features_added.extend([f"{col}_length", f"{col}_length_category"])
        
        return features_added
    
    def _create_location_features(self, df, loc_cols):
        """Create features from location columns."""
        features_added = []
        
        # Check if we have latitude and longitude
        lat_col = next((col for col in loc_cols if 'lat' in col.lower()), None)
        lng_col = next((col for col in loc_cols if 'lon' in col.lower() or 'lng' in col.lower()), None)
        
        if lat_col and lng_col:
            # Create simple distance from center feature
            center_lat = df[lat_col].mean()
            center_lng = df[lng_col].mean()
            
            df['distance_from_center'] = np.sqrt(
                (df[lat_col] - center_lat)**2 + (df[lng_col] - center_lng)**2
            )
            df['location_zone'] = pd.qcut(df['distance_from_center'], 3, 
                                         labels=['Central', 'Midrange', 'Peripheral'])
            
            features_added.extend(['distance_from_center', 'location_zone'])
        
        return features_added
    
    def _create_statistical_segments(self, df, numeric_cols):
        """Create segments using statistical bucketing methods instead of clustering."""
        features_added = []
        
        # Select the most relevant numeric columns for segmentation
        # Prioritize columns that might represent business value
        value_keywords = ['income', 'revenue', 'sales', 'profit', 'price', 'spend', 'cost', 'value']
        
        # Find columns that might represent value
        value_cols = [col for col in numeric_cols if any(keyword in col.lower() for keyword in value_keywords)]
        
        # If no value columns found, use the first few numeric columns
        if not value_cols and len(numeric_cols) > 0:
            value_cols = numeric_cols[:min(3, len(numeric_cols))]
        
        # Create combined value score if multiple value columns exist
        if len(value_cols) > 1:
            try:
                # Handle missing values first
                value_data = df[value_cols].copy()
                
                # Use SimpleImputer to handle missing values
                imputer = SimpleImputer(strategy='median')
                value_data_imputed = pd.DataFrame(
                    imputer.fit_transform(value_data),
                    columns=value_data.columns,
                    index=value_data.index
                )
                
                # Normalize each column to 0-1 scale
                for col in value_data_imputed.columns:
                    col_min = value_data_imputed[col].min()
                    col_max = value_data_imputed[col].max()
                    if col_max > col_min:  # Avoid division by zero
                        value_data_imputed[col] = (value_data_imputed[col] - col_min) / (col_max - col_min)
                
                # Create a combined value score (average of normalized values)
                df['value_score'] = value_data_imputed.mean(axis=1)
                features_added.append('value_score')
                
                # Create value segments using quantile-based bucketing
                try:
                    df['value_segment'] = pd.qcut(
                        df['value_score'], 
                        4, 
                        labels=['Low Value', 'Medium Value', 'High Value', 'Premium']
                    )
                    features_added.append('value_segment')
                except ValueError:
                    # Fall back to simpler bucketing if qcut fails
                    df['value_segment'] = pd.cut(
                        df['value_score'],
                        bins=[0, 0.25, 0.5, 0.75, 1],
                        labels=['Low Value', 'Medium Value', 'High Value', 'Premium'],
                        include_lowest=True
                    )
                    features_added.append('value_segment')
            except Exception as e:
                print(f"  Could not create combined value score: {str(e)}")
        
        # Create individual statistical segments for key columns
        for col in value_cols[:3]:  # Limit to top 3 value columns
            try:
                # Create segments based on standard deviations from mean
                col_mean = df[col].mean()
                col_std = df[col].std()
                
                if col_std > 0:  # Only if there's variation in the data
                    df[f"{col}_stddev_segment"] = pd.cut(
                        df[col],
                        bins=[float('-inf'), col_mean - col_std, col_mean, col_mean + col_std, float('inf')],
                        labels=['Low', 'Below Average', 'Above Average', 'High']
                    )
                    features_added.append(f"{col}_stddev_segment")
                    
                    # Create percentile-based segments
                    percentiles = [0, 25, 50, 75, 100]
                    bins = [df[col].quantile(p/100) for p in percentiles]
                    
                    # Ensure bins are unique
                    if len(set(bins)) > 1:
                        df[f"{col}_percentile_segment"] = pd.cut(
                            df[col],
                            bins=bins,
                            labels=['Bottom 25%', 'Lower Middle', 'Upper Middle', 'Top 25%'],
                            include_lowest=True
                        )
                        features_added.append(f"{col}_percentile_segment")
            except Exception as e:
                print(f"  Could not create segments for {col}: {str(e)}")
        
        # Create a combined behavior segment if we have behavior-related columns
        behavior_keywords = ['frequency', 'recency', 'duration', 'visits', 'orders', 'purchases']
        behavior_cols = [col for col in numeric_cols if any(keyword in col.lower() for keyword in behavior_keywords)]
        
        if len(behavior_cols) >= 2:
            try:
                # Handle missing values
                behavior_data = df[behavior_cols].copy()
                behavior_data = behavior_data.fillna(behavior_data.median())
                
                # Normalize each column
                for col in behavior_data.columns:
                    col_min = behavior_data[col].min()
                    col_max = behavior_data[col].max()
                    if col_max > col_min:
                        behavior_data[col] = (behavior_data[col] - col_min) / (col_max - col_min)
                
                # Create behavior score by averaging normalized values
                df['behavior_score'] = behavior_data.mean(axis=1)
                
                # Create behavior segments
                df['behavior_segment'] = pd.qcut(
                    df['behavior_score'],
                    4,
                    labels=['Infrequent', 'Occasional', 'Regular', 'Frequent'],
                    duplicates='drop'
                )
                
                features_added.extend(['behavior_score', 'behavior_segment'])
            except Exception as e:
                print(f"  Could not create behavior segments: {str(e)}")
        
        # Create multi-dimensional segments using a combination of value and recency (if available)
        recency_cols = [col for col in df.columns if 'recency' in col.lower() and pd.api.types.is_numeric_dtype(df[col])]
        
        if 'value_score' in df.columns and recency_cols:
            try:
                recency_col = recency_cols[0]
                
                # Create a binary value segment (high/low)
                df['value_binary'] = np.where(df['value_score'] > df['value_score'].median(), 'High', 'Low')
                
                # Create a binary recency segment (recent/old)
                df['recency_binary'] = np.where(df[recency_col] < df[recency_col].median(), 'Recent', 'Old')
                
                # Combine into a customer lifecycle segment
                df['lifecycle_segment'] = df['recency_binary'] + ' ' + df['value_binary']
                
                # Map to business-friendly names
                segment_map = {
                    'Recent High': 'Active High Value',
                    'Recent Low': 'Active Low Value',
                    'Old High': 'Lapsed High Value',
                    'Old Low': 'Lapsed Low Value'
                }
                
                df['customer_lifecycle'] = df['lifecycle_segment'].map(segment_map)
                features_added.append('customer_lifecycle')
                
                # Clean up temporary columns
                df.drop(['value_binary', 'recency_binary', 'lifecycle_segment'], axis=1, inplace=True)
            except Exception as e:
                print(f"  Could not create lifecycle segments: {str(e)}")
        
        return features_added


# import pandas as pd
# from auto_target_generator import AutoTargetFeatureGenerator
 

# Example usage
if __name__ == "__main__":
    # Specify the path to your CSV file
    file_path = "../uploads/california_housing_train.csv"

    # Read the data
    df = pd.read_csv(file_path)
   
        # Create the feature generator
    generator = AutoTargetFeatureGenerator()
    
    # Generate features - by default returns merged dataframe with original and new features
    merged_df = generator.generate_features(df)
    
    merged_df.to_csv("../uploads/california_housing_train_expanded.csv")
    
    # Now merged_df contains both original columns and new target features
    print(f"Original columns: {len(df.columns)}")
    print(f"Total columns after transformation: {len(merged_df.columns)}")
    print(f"New features added: {len(merged_df.columns) - len(df.columns)}")
     