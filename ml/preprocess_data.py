import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, QuantileTransformer, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy.stats import chi2_contingency

def screen_features(df, target_variable, correlation_threshold=0.85, 
                    mi_threshold=0.8, temporal_columns=None, id_threshold=0.9,
                    cardinality_threshold=0.9, verbose=True, plot=True):
    """
    Screen features to detect and optionally remove high correlation, leaking variables, 
    and other problematic features. Handles both categorical and numerical target variables.
    
    Args:
        df (pd.DataFrame): Input dataframe including the target variable
        target_variable (str): Name of the target variable column
        correlation_threshold (float): Threshold for correlation between features (default: 0.85)
        mi_threshold (float): Threshold for mutual information with target (default: 0.8)
        temporal_columns (list): List of datetime columns to check for data leakage (default: None)
        id_threshold (float): Threshold for unique values ratio to detect ID columns (default: 0.9)
        cardinality_threshold (float): Threshold for categorical cardinality (default: 0.9)
        verbose (bool): Whether to print detailed information (default: True)
        plot (bool): Whether to create correlation and MI plots (default: True)
    
    Returns:
        tuple: (cleaned_df, dropped_features, feature_info)
            - cleaned_df: DataFrame with problematic features removed
            - dropped_features: Dictionary with reasons for dropped features
            - feature_info: Dictionary with information about features
    """
    if verbose:
        print(f"Starting feature screening for target: '{target_variable}'")
        print(f"Initial dataframe shape: {df.shape}")
    
    # Make a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Initialize output dictionaries
    dropped_features = {
        'high_correlation': [],
        'high_mutual_info': [],
        'temporal_leakage': [],
        'id_like': [],
        'high_cardinality': [],
        'constant': [],
        'null_heavy': []
    }
    
    feature_info = {
        'correlation_with_target': {},
        'mutual_info_with_target': {},
        'correlation_matrix': None,
        'unique_ratio': {},
        'null_ratio': {}
    }
    
    # Get the target variable
    y = df_copy[target_variable]
    
    # Determine target type (categorical or numerical)
    target_is_categorical = (y.dtype == 'object' or 
                             y.dtype.name == 'category' or 
                             y.nunique() < 10)  # Heuristic for categorical
    
    if verbose:
        print(f"Target variable type: {'Categorical' if target_is_categorical else 'Numerical'}")
    
    # Identify categorical and numerical features
    categorical_features = df_copy.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df_copy.select_dtypes(include=['number']).columns.tolist()
    
    if target_variable in numerical_features:
        numerical_features.remove(target_variable)
    
    # Step 1: Check for constant or near-constant features
    for col in df_copy.columns:
        if col == target_variable:
            continue
            
        # Check for constant features
        unique_values = df_copy[col].nunique(dropna=False)
        if unique_values <= 1:
            if verbose:
                print(f"Dropping constant feature: {col} (nunique={unique_values})")
            dropped_features['constant'].append(col)
            continue
        
        # Calculate unique value ratio
        unique_ratio = df_copy[col].nunique() / len(df_copy)
        feature_info['unique_ratio'][col] = unique_ratio
        
        # Check for ID-like columns
        if unique_ratio > id_threshold:
            if verbose:
                print(f"Dropping ID-like feature: {col} (unique ratio={unique_ratio:.4f})")
            dropped_features['id_like'].append(col)
            continue
        
        # Check for high-cardinality categorical features
        if col in categorical_features and unique_ratio > cardinality_threshold:
            if verbose:
                print(f"Dropping high-cardinality categorical feature: {col} (unique ratio={unique_ratio:.4f})")
            dropped_features['high_cardinality'].append(col)
            continue
        
        # Calculate null ratio
        null_ratio = df_copy[col].isna().mean()
        feature_info['null_ratio'][col] = null_ratio
        
        # Drop columns with too many nulls (e.g., > 50%)
        if null_ratio > 0.5:
            if verbose:
                print(f"Dropping null-heavy feature: {col} (null ratio={null_ratio:.4f})")
            dropped_features['null_heavy'].append(col)
            continue
    
    # Create a list of features to drop
    features_to_drop = []
    for reason, cols in dropped_features.items():
        features_to_drop.extend(cols)
    
    # Create a temporary dataframe without already identified problematic features
    temp_df = df_copy.drop(columns=features_to_drop + [target_variable], errors='ignore')
    
    # Step 2: Check for temporal leakage
    if temporal_columns is not None:
        # Validate temporal_columns exist in the dataframe
        valid_temporal_columns = [col for col in temporal_columns if col in temp_df.columns]
        
        for col in valid_temporal_columns:
            # Convert to datetime if not already
            if not pd.api.types.is_datetime64_dtype(temp_df[col]):
                try:
                    temp_df[col] = pd.to_datetime(temp_df[col])
                except Exception as e:
                    if verbose:
                        print(f"Could not convert {col} to datetime: {e}")
                    continue
            
            # Calculate time-based features
            temp_df[f"{col}_year"] = temp_df[col].dt.year
            temp_df[f"{col}_month"] = temp_df[col].dt.month
            temp_df[f"{col}_day"] = temp_df[col].dt.day
            temp_df[f"{col}_dayofweek"] = temp_df[col].dt.dayofweek
            
            # For categorical targets, use chi-square test instead of correlation
            if target_is_categorical:
                for time_col in [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"]:
                    if time_col in temp_df.columns:
                        # Create contingency table
                        try:
                            contingency = pd.crosstab(temp_df[time_col], y)
                            chi2, p, _, _ = chi2_contingency(contingency)
                            
                            # Lower p-value indicates stronger relationship
                            if p < 0.01:  # Significant relationship
                                if verbose:
                                    print(f"Possible temporal leakage detected: {time_col} (chi2={chi2:.2f}, p={p:.4f})")
                                dropped_features['temporal_leakage'].append(col)
                        except Exception as e:
                            if verbose:
                                print(f"Error in chi-square test for {time_col}: {e}")
            else:
                # For numerical targets, use correlation
                for time_col in [f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"]:
                    if time_col in temp_df.columns:
                        try:
                            correlation = temp_df[time_col].corr(y)
                            if abs(correlation) > 0.3:  # Using a lower threshold for temporal features
                                if verbose:
                                    print(f"Possible temporal leakage detected: {time_col} (correlation with target: {correlation:.4f})")
                                dropped_features['temporal_leakage'].append(col)
                        except Exception as e:
                            if verbose:
                                print(f"Error calculating correlation for {time_col}: {e}")
            
            # Drop the temporary time columns
            temp_df = temp_df.drop(columns=[f"{col}_year", f"{col}_month", f"{col}_day", f"{col}_dayofweek"], 
                                  errors='ignore')
    
    # Update features to drop
    features_to_drop = []
    for reason, cols in dropped_features.items():
        features_to_drop.extend(cols)
    
    # Create a temporary dataframe without already identified problematic features
    temp_df = df_copy.drop(columns=features_to_drop + [target_variable], errors='ignore')
    
    # Step 3: Correlation analysis for numerical features
    numerical_cols = [col for col in numerical_features if col in temp_df.columns]
    if numerical_cols:
        # Calculate correlation matrix
        correlation_matrix = temp_df[numerical_cols].corr().abs()
        feature_info['correlation_matrix'] = correlation_matrix
        
        # Calculate correlation with target for numerical features
        if not target_is_categorical:
            for col in numerical_cols:
                try:
                    correlation = df_copy[col].corr(y)
                    feature_info['correlation_with_target'][col] = correlation
                except Exception as e:
                    if verbose:
                        print(f"Error calculating correlation for {col}: {e}")
                    feature_info['correlation_with_target'][col] = 0
        else:
            # For categorical target, we'll use mutual information instead of correlation
            # But initialize with zeros for now
            for col in numerical_cols:
                feature_info['correlation_with_target'][col] = 0
        
        # Plot correlation matrix
        if plot and len(numerical_cols) > 1:
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm', 
                        vmin=0, vmax=1, linewidths=0.5)
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.savefig("feature_correlation_matrix.png")
            plt.close()
            
            # Plot correlation with target if numerical
            if not target_is_categorical and feature_info['correlation_with_target']:
                plt.figure(figsize=(10, len(numerical_cols) * 0.3))
                target_corr = pd.Series(feature_info['correlation_with_target']).sort_values(ascending=False)
                sns.barplot(x=target_corr.values, y=target_corr.index)
                plt.title(f"Numerical Features Correlation with {target_variable}")
                plt.tight_layout()
                plt.savefig("target_correlation.png")
                plt.close()
        
        # Find pairs of highly correlated features
        highly_correlated_pairs = []
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                col1, col2 = numerical_cols[i], numerical_cols[j]
                corr_value = correlation_matrix.loc[col1, col2]
                
                if corr_value >= correlation_threshold:
                    highly_correlated_pairs.append((col1, col2, corr_value))
        
        # Sort pairs by correlation value (descending)
        highly_correlated_pairs.sort(key=lambda x: x[2], reverse=True)
        
        # Process pairs to decide which feature to drop
        if highly_correlated_pairs:
            if verbose:
                print("\nHighly correlated feature pairs:")
                for col1, col2, corr in highly_correlated_pairs:
                    print(f"  {col1} <-> {col2}: {corr:.4f}")
            
            # Greedy approach to drop features with highest correlations
            # For each pair, keep the one with higher correlation/MI with target
            for col1, col2, corr in highly_correlated_pairs:
                # Skip if either column is already marked for removal
                if col1 in dropped_features['high_correlation'] or col2 in dropped_features['high_correlation']:
                    continue
                
                # Get correlation or MI value with target
                val1 = abs(feature_info['correlation_with_target'].get(col1, 0))
                val2 = abs(feature_info['correlation_with_target'].get(col2, 0))
                
                # If target is categorical, we'll update these values with MI later
                
                # Drop the feature with lower correlation/MI with target
                if val1 >= val2:
                    dropped_features['high_correlation'].append(col2)
                    if verbose:
                        print(f"Dropping {col2} due to high correlation with {col1} " + 
                              f"({corr:.4f}). Their target relationships: {col1}={val1:.4f}, {col2}={val2:.4f}")
                else:
                    dropped_features['high_correlation'].append(col1)
                    if verbose:
                        print(f"Dropping {col1} due to high correlation with {col2} " + 
                              f"({corr:.4f}). Their target relationships: {col1}={val1:.4f}, {col2}={val2:.4f}")
    
    # Update features to drop
    features_to_drop = []
    for reason, cols in dropped_features.items():
        features_to_drop.extend(cols)
    
    # Create a temporary dataframe without already identified problematic features
    temp_df = df_copy.drop(columns=features_to_drop + [target_variable], errors='ignore')
    
    # Step 4: Mutual Information analysis
    X = temp_df.copy()
    
    # Handle categorical features for MI calculation
    for col in X.columns:
        if col in categorical_features:
            # One-hot encode categorical features
            X[col] = X[col].astype('category').cat.codes
    
    # Prepare target for MI calculation
    y_for_mi = y
    if target_is_categorical:
        # Convert categorical target to numeric
        try:
            y_for_mi = y.astype('category').cat.codes
        except:
            # If conversion fails, create a simple mapping
            unique_values = y.unique()
            mapping = {val: idx for idx, val in enumerate(unique_values)}
            y_for_mi = y.map(mapping)
    
    # Calculate mutual information
    try:
        if target_is_categorical:
            mi_scores = mutual_info_classif(X, y_for_mi, random_state=42)
        else:
            mi_scores = mutual_info_regression(X, y_for_mi, random_state=42)
        
        mi_dict = dict(zip(X.columns, mi_scores))
        feature_info['mutual_info_with_target'] = mi_dict
        
        # Normalize MI scores to [0, 1]
        if mi_scores.max() > 0:
            normalized_mi = mi_scores / mi_scores.max()
            
            # Plot Mutual Information
            if plot:
                plt.figure(figsize=(10, len(X.columns) * 0.3))
                mi_df = pd.DataFrame({'Feature': X.columns, 'MI': normalized_mi})
                mi_df = mi_df.sort_values('MI', ascending=False)
                sns.barplot(x='MI', y='Feature', data=mi_df)
                plt.title(f"Normalized Mutual Information with {target_variable}")
                plt.tight_layout()
                plt.savefig("mutual_information.png")
                plt.close()
            
            # If target is categorical, update correlation_with_target with MI values
            # This will help with the decision of which feature to keep in highly correlated pairs
            if target_is_categorical:
                feature_info['correlation_with_target'] = mi_dict
            
            # Check for suspiciously high MI (potential leakage)
            for col, mi in zip(X.columns, normalized_mi):
                if mi > mi_threshold:
                    if verbose:
                        print(f"Feature with suspiciously high MI: {col} (MI={mi:.4f})")
                    dropped_features['high_mutual_info'].append(col)
    except Exception as e:
        if verbose:
            print(f"Error calculating mutual information: {e}")
    
    # Update final list of features to drop
    all_features_to_drop = []
    for reason, cols in dropped_features.items():
        all_features_to_drop.extend(cols)
    
    # Remove duplicates while preserving order
    all_features_to_drop = list(dict.fromkeys(all_features_to_drop))
    
    # Create final cleaned dataframe
    cleaned_df = df_copy.drop(columns=all_features_to_drop, errors='ignore')
    
    if verbose:
        print(f"\nFeature screening complete.")
        print(f"Original features: {df.shape[1]}")
        print(f"Features after cleaning: {cleaned_df.shape[1]}")
        print(f"Total features dropped: {len(all_features_to_drop)}")
        
        # Print summary by reason
        print("\nFeatures dropped by reason:")
        for reason, cols in dropped_features.items():
            if cols:
                print(f"  {reason}: {len(cols)} features")
                if verbose and len(cols) <= 10:
                    print(f"    {', '.join(cols)}")
    
    return cleaned_df, dropped_features, feature_info

def add_screen_features_to_pipeline(df, target_variable, correlation_threshold=0.85, 
                                   mi_threshold=0.8, temporal_columns=None, 
                                   id_threshold=0.9, cardinality_threshold=0.9, 
                                   verbose=True):
    """
    Screen features and return a tuple of (cleaned_df, dropped_features, feature_info)
    that can be integrated into a data preprocessing pipeline.
    
    Args:
        Same as screen_features function
    
    Returns:
        tuple: (cleaned_df, dropped_features, feature_info)
    """
    return screen_features(df, target_variable, 
                          correlation_threshold=correlation_threshold,
                          mi_threshold=mi_threshold, 
                          temporal_columns=temporal_columns,
                          id_threshold=id_threshold,
                          cardinality_threshold=cardinality_threshold,
                          verbose=verbose, 
                          plot=False)  # No plotting in pipeline mode


def preprocess_target_variable(df, target_variable):
    """
    Preprocess the target variable using LabelEncoder
    Handles mixed data types by converting to string
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_variable (str): Name of the target variable column
    
    Returns:
        tuple: (processed target variable, LabelEncoder)
    """
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
    
    # Get the original target variable series
    y = df[target_variable].copy()
    
    # Check for mixed types in the target variable
    try:
        # Force values to be either all strings or all numeric for consistency
        sample_values = y.dropna().head(1000).tolist()
        types_present = set(type(val) for val in sample_values)
        
        if len(types_present) > 1:
            print(f"Warning: Mixed data types detected in target variable: {[t.__name__ for t in types_present]}")
            print("Converting all values to strings for consistent encoding...")
            # Convert all values to strings to ensure consistent types
            y = y.astype(str)
    except Exception as e:
        print(f"Warning when checking target types: {e}")
        # Force conversion to string as a fallback
        y = y.fillna("None").astype(str)
        print("Forced conversion of target to string type")
    
    # Initialize LabelEncoder
    le = LabelEncoder()
    
    try:
        # Try to fit transform directly
        y_encoded = le.fit_transform(y)
        
        # Get information about the encoded values
        print(f"Target variable '{target_variable}' encoded successfully.")
        print(f"Number of unique classes: {len(le.classes_)}")
        
        # Output more information based on data type
        if hasattr(y, 'dtype') and (y.dtype == 'object' or pd.api.types.is_categorical_dtype(y)):
            # For categorical data, show distribution
            categories = y.value_counts()
            print("Target variable category distribution (top 10):")
            print(categories.head(10))
            
            if len(le.classes_) > 10:
                print(f"Note: {len(le.classes_)} total unique classes present")
        else:
            # For numerical data, show range
            try:
                min_value = y.min()
                max_value = y.max()
                print(f"Original target variable range: {min_value} to {max_value}")
            except:
                pass
        
        print(f"Encoded target variable unique values: {np.unique(y_encoded)}")
        
        return y_encoded, le
        
    except TypeError as te:
        # If we still get a type error, it means there might be complex mixed types
        # Convert to string and try again
        print(f"Error during encoding: {te}")
        print("Falling back to string conversion for problematic target variable")
        
        # Force string conversion
        y = y.fillna("None").astype(str)
        y_encoded = le.fit_transform(y)
        
        print(f"Successfully encoded target after type conversion")
        print(f"Number of unique classes: {len(le.classes_)}")
        print(f"Encoded target variable unique values: {np.unique(y_encoded)}")
        
        return y_encoded, le
    
    except Exception as e:
        # Handle any other errors
        print(f"Unexpected error during target encoding: {e}")
        
        # Create a simple integer encoding as a final fallback
        print("Using simple integer encoding as final fallback")
        unique_values = y.fillna("None").unique()
        value_to_int = {val: i for i, val in enumerate(unique_values)}
        y_encoded = np.array([value_to_int.get(val, 0) for val in y])
        
        # Create a minimal label encoder with the mapping
        le = LabelEncoder()
        le.classes_ = np.array(list(value_to_int.keys()))
        
        print(f"Fallback encoding created with {len(unique_values)} classes")
        print(f"Encoded target variable unique values: {np.unique(y_encoded)}")
        
        return y_encoded, le

def create_preprocessor(df, numerical_features=None, categorical_features=None):
    """
    Create a preprocessing pipeline for the given dataframe with robust feature validation.
    
    Args:
        df (pd.DataFrame): Input dataframe (should be features only, without target)
        numerical_features (list): List of numerical feature column names. If None, auto-detected.
        categorical_features (list): List of categorical feature column names. If None, auto-detected.
    
    Returns:
        ColumnTransformer: The fitted preprocessor
    """
    # Verify df is not None
    if df is None:
        raise ValueError("Input dataframe cannot be None")
    
    # Check if df is empty
    if len(df.columns) == 0:
        raise ValueError("Input dataframe has no columns")
    
    # Get actual columns in the dataframe
    available_columns = set(df.columns)
    
    # Auto-detect features if not provided
    if numerical_features is None:
        numerical_features = df.select_dtypes(include=['number']).columns.tolist()
    
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Filter features to only include those present in the dataframe
    valid_numerical = [col for col in numerical_features if col in available_columns]
    valid_categorical = [col for col in categorical_features if col in available_columns]
    
    # Log any removed features
    if len(valid_numerical) != len(numerical_features):
        removed = set(numerical_features) - set(valid_numerical)
        print(f"Warning: Removed {len(removed)} non-existent numerical features: {removed}")
    
    if len(valid_categorical) != len(categorical_features):
        removed = set(categorical_features) - set(valid_categorical)
        print(f"Warning: Removed {len(removed)} non-existent categorical features: {removed}")
    
    # Create transformers based on what's available
    transformers = []
    
    # Only add numerical transformer if we have valid numerical features
    if valid_numerical:
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        
        # Create a pipeline for numerical features with imputation as a safety measure
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        transformers.append(('num', numerical_pipeline, valid_numerical))
    
    # Only add categorical transformer if we have valid categorical features
    if valid_categorical:
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        
        # Create a pipeline for categorical features with imputation as a safety measure
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(
                handle_unknown='ignore',
                max_categories=10,  # Limit to prevent explosion with high cardinality
                sparse_output=False  # Dense output for better compatibility
            ))
        ])
        
        transformers.append(('cat', categorical_pipeline, valid_categorical))
    
    # Create the ColumnTransformer with validated features
    from sklearn.compose import ColumnTransformer
    
    # Handle the case where we have no valid features
    if not transformers:
        print("Warning: No valid features found. Creating a pass-through preprocessor.")
        # Create a pass-through preprocessor that doesn't transform anything
        preprocessor = ColumnTransformer(
            transformers=[],
            remainder='passthrough'  # Pass through all columns
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop'  # Drop columns not specified
        )
    
    # Fit the preprocessor on the dataframe
    try:
        preprocessor.fit(df)
        print(f"Preprocessor created successfully with {len(valid_numerical)} numerical and {len(valid_categorical)} categorical features")
    except Exception as e:
        print(f"Error fitting preprocessor: {e}")
        raise
    
    return preprocessor
    """
    Create a preprocessing pipeline for the given dataframe
    
    Args:
        df (pd.DataFrame): Input dataframe
        numerical_features (list): List of numerical feature column names. If None, auto-detected.
        categorical_features (list): List of categorical feature column names. If None, auto-detected.
    
    Returns:
        ColumnTransformer: The fitted preprocessor
    """
    # Determine the target variable (assumed to be the last column)
    target_variable = df.columns[-1]
    
    # Auto-detect features if not provided
    if numerical_features is None:
        # Exclude potential target variable from numerical features
        numerical_features = df.select_dtypes(include=['number']).columns.tolist()
        
        # Remove target variable from numerical features if present
        if target_variable in numerical_features:
            numerical_features.remove(target_variable)
    
    if categorical_features is None:
        categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Prepare data without target variable
    X = df.drop(columns=[target_variable])
    
    # Create a ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),  # Scale numerical features
            ('cat', OneHotEncoder(
                handle_unknown='ignore', 
                max_categories=10,  # Limit number of categories to prevent explosion
                sparse_output=False  # Updated from sparse=False to sparse_output=False
            ), categorical_features)  # OHE categorical features
        ],
        remainder='drop'  # Drop columns not specified
    )
    
    # Fit the preprocessor
    preprocessor.fit(X)
    
def preprocess_data(df, preprocessor=None, apply_quantile_transform=False):
    """
    Preprocess the data using the given preprocessor or create a new one
    
    Args:
        df (pd.DataFrame): Input dataframe
        preprocessor (ColumnTransformer, optional): Preprocessor to use. If None, a new one is created.
        apply_quantile_transform (bool): Whether to apply QuantileTransformer to numerical features
    
    Returns:
        tuple: (preprocessed DataFrame, preprocessor, feature names after preprocessing)
    """
    # Create a copy to avoid modifying the original dataframe
    df_copy = df.copy()
    
    # Identify the target variable (assuming it's the last column)
    target_variable = df_copy.columns[-1]
    
    # Preprocess the target variable
    y, label_encoder = preprocess_target_variable(df_copy, target_variable)
    
    # Remove target variable from features
    X_copy = df_copy.drop(columns=[target_variable])
    
    # Identify numerical and categorical features
    numerical_features = X_copy.select_dtypes(include=['number']).columns.tolist()
    categorical_features = X_copy.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Create or use preprocessor
    if preprocessor is None:
        preprocessor = create_preprocessor(df_copy)
    
    # Apply preprocessing
    X_processed = preprocessor.transform(X_copy)
    
    # Get feature names after preprocessing
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions
        feature_names = (
            [f"num__{col}" for col in numerical_features] + 
            [f"cat__{col}__{cat}" for col in categorical_features 
             for cat in preprocessor.named_transformers_['cat'].categories_[preprocessor.named_transformers_['cat'].feature_names_in_.tolist().index(col)]]
        )
    
    # Convert to DataFrame
    if isinstance(X_processed, np.ndarray):
        # Ensure the number of columns matches feature names
        if X_processed.shape[1] != len(feature_names):
            print(f"Warning: Processed data shape {X_processed.shape} does not match feature names length {len(feature_names)}")
            
            # Truncate feature names or pad processed data
            if X_processed.shape[1] < len(feature_names):
                feature_names = feature_names[:X_processed.shape[1]]
            else:
                # Pad processed data with zeros
                padding = np.zeros((X_processed.shape[0], len(feature_names) - X_processed.shape[1]))
                X_processed = np.hstack([X_processed, padding])
        
        X = pd.DataFrame(X_processed, columns=feature_names)
    else:
        # For sparse matrices or other formats
        X = pd.DataFrame(X_processed.toarray(), columns=feature_names)
    
    # Apply QuantileTransformer to numerical features if requested
    if apply_quantile_transform and len(numerical_features) > 1:
        # Find numerical features in processed data
        numerical_cols_processed = [col for col in X.columns if col.startswith('num__')]
        
        if numerical_cols_processed:
            qt = QuantileTransformer(output_distribution='normal', n_quantiles=min(len(df_copy), 100))
            X_numerical_transformed = qt.fit_transform(X[numerical_cols_processed])
            X_numerical_df = pd.DataFrame(X_numerical_transformed, columns=numerical_cols_processed)
            
            # Replace the columns
            X = X.drop(columns=numerical_cols_processed)
            X = pd.concat([X_numerical_df, X], axis=1)
    
    return X, preprocessor, feature_names

def get_output_dir(user_id, use_case=None, target_variable=None, filename=None):
    """
    Generate the output directory path for saving models
    
    Args:
        user_id (str): User ID for organization
        use_case (str, optional): Use case identifier
        target_variable (str, optional): Target variable name
        filename (str, optional): Original filename
        
    Returns:
        str: Path to the output directory
    """
    # Get the absolute path to the Flask app's root directory
    import os
    
    # Find the flask app root directory by looking for specific marker files/dirs
    # Start from current directory and go up until we find 'databases' folder
    current_dir = os.path.abspath(os.getcwd())
    app_root = current_dir
    
    # Try to find databases directory by going up to 3 levels
    for _ in range(3):
        if os.path.isdir(os.path.join(app_root, 'databases')):
            break
        parent = os.path.dirname(app_root)
        if parent == app_root:  # We've reached the filesystem root
            break
        app_root = parent
    
    # Base directory using the new structure: databases/trained_models/{user_id}/
    user_dir = os.path.join(app_root, "databases", "trained_models", user_id)
    
    # Create directory if it doesn't exist
    os.makedirs(user_dir, exist_ok=True)
    
    return user_dir

def get_model_filename(user_id, use_case, target_variable, filename=None):
    """
    Generate a model filename based on metadata
    
    Args:
        user_id (str): User ID for organization
        use_case (str): Use case identifier
        target_variable (str): Target variable name
        filename (str, optional): Original filename
    
    Returns:
        str: Model filename
    """
    # Clean up variables for safe filename
    safe_use_case = ''.join(c if c.isalnum() else '_' for c in use_case)
    safe_target = ''.join(c if c.isalnum() else '_' for c in target_variable)
    
    # Base name
    model_filename = f"{safe_use_case}_{safe_target}"
    
    # Add filename if available
    if filename:
        safe_filename = ''.join(c if c.isalnum() else '_' for c in os.path.splitext(filename)[0])
        model_filename = f"{model_filename}_{safe_filename}"
    
    # Add extension
    model_filename = f"{model_filename}.joblib"
    
    return model_filename

def save_preprocessor(preprocessor, user_id, use_case, target_variable, filename=None, output_dir=None):
    """
    Save the preprocessor to disk
    
    Args:
        preprocessor: The preprocessor to save
        user_id (str): User ID for organization
        use_case (str): Use case identifier
        target_variable (str): Target variable name
        filename (str, optional): Original filename
        output_dir (str, optional): Directory to save the preprocessor to
    
    Returns:
        str: Path to the saved preprocessor file
    """
    # Determine output directory
    if output_dir is None:
        output_dir = get_output_dir(user_id, use_case, target_variable, filename)
    else:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    # Clean up variables for safe filename
    safe_use_case = ''.join(c if c.isalnum() else '_' for c in use_case)
    safe_target = ''.join(c if c.isalnum() else '_' for c in target_variable)
    
    # Base name
    preprocessor_filename = f"preprocessor_{safe_use_case}_{safe_target}"
    
    # Add filename if available
    if filename:
        safe_filename = ''.join(c if c.isalnum() else '_' for c in os.path.splitext(filename)[0])
        preprocessor_filename = f"{preprocessor_filename}_{safe_filename}"
    
    # Add extension
    preprocessor_filename = f"{preprocessor_filename}.joblib"
    
    # Full path
    preprocessor_path = os.path.join(output_dir, preprocessor_filename)
    
    # Save preprocessor
    joblib.dump(preprocessor, preprocessor_path)
    
    return preprocessor_path

def save_label_encoder(label_encoder, user_id, use_case, target_variable, filename=None, output_dir=None):
    """
    Save the LabelEncoder to disk
    
    Args:
        label_encoder (LabelEncoder): The LabelEncoder to save
        user_id (str): User ID for organization
        use_case (str): Use case identifier
        target_variable (str): Target variable name
        filename (str, optional): Original filename
        output_dir (str, optional): Directory to save the encoder to
    
    Returns:
        str: Path to the saved LabelEncoder file
    """
    # If no label encoder, return None
    if label_encoder is None:
        return None
    
    # Determine output directory
    if output_dir is None:
        output_dir = get_output_dir(user_id, use_case, target_variable, filename)
    else:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    # Clean up variables for safe filename
    safe_use_case = ''.join(c if c.isalnum() else '_' for c in use_case)
    safe_target = ''.join(c if c.isalnum() else '_' for c in target_variable)
    
    # Base name
    encoder_filename = f"label_encoder_{safe_use_case}_{safe_target}"
    
    # Add filename if available
    if filename:
        safe_filename = ''.join(c if c.isalnum() else '_' for c in os.path.splitext(filename)[0])
        encoder_filename = f"{encoder_filename}_{safe_filename}"
    
    # Add extension
    encoder_filename = f"{encoder_filename}.joblib"
    
    # Full path
    encoder_path = os.path.join(output_dir, encoder_filename)
    
    # Save LabelEncoder
    joblib.dump(label_encoder, encoder_path)
    
    return encoder_path

def load_label_encoder(user_id, use_case, target_variable, filename=None, output_dir=None):
    """
    Load a LabelEncoder from disk
    
    Args:
        user_id (str): User ID for organization
        use_case (str): Use case identifier
        target_variable (str): Target variable name
        filename (str, optional): Original filename
        output_dir (str, optional): Directory where the encoder is stored
    
    Returns:
        LabelEncoder or None
    """
    # Determine output directory
    if output_dir is None:
        output_dir = get_output_dir(user_id, use_case, target_variable, filename)
    
    # Clean up variables for safe filename
    safe_use_case = ''.join(c if c.isalnum() else '_' for c in use_case)
    safe_target = ''.join(c if c.isalnum() else '_' for c in target_variable)
    
    # Base name
    encoder_filename = f"label_encoder_{safe_use_case}_{safe_target}"
    
    # Add filename if available
    if filename:
        safe_filename = ''.join(c if c.isalnum() else '_' for c in os.path.splitext(filename)[0])
        encoder_filename = f"{encoder_filename}_{safe_filename}"
    
    # Add extension
    encoder_filename = f"{encoder_filename}.joblib"
    
    # Full path
    encoder_path = os.path.join(output_dir, encoder_filename)
    
    try:
        # Try to load the LabelEncoder
        label_encoder = joblib.load(encoder_path)
        return label_encoder
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error loading LabelEncoder: {e}")
        return None
    
def load_preprocessor(user_id, use_case, target_variable, filename=None, output_dir=None):
    """
    Load a preprocessor from disk
    
    Args:
        user_id (str): User ID for organization
        use_case (str): Use case identifier
        target_variable (str): Target variable name
        filename (str, optional): Original filename
        output_dir (str, optional): Directory where the preprocessor is stored
    
    Returns:
        ColumnTransformer or None: The loaded preprocessor
    """
    # Determine output directory
    if output_dir is None:
        output_dir = get_output_dir(user_id, use_case, target_variable, filename)
    
    # Clean up variables for safe filename
    safe_use_case = ''.join(c if c.isalnum() else '_' for c in use_case)
    safe_target = ''.join(c if c.isalnum() else '_' for c in target_variable)
    
    # Base name
    preprocessor_filename = f"preprocessor_{safe_use_case}_{safe_target}"
    
    # Add filename if available
    if filename:
        safe_filename = ''.join(c if c.isalnum() else '_' for c in os.path.splitext(filename)[0])
        preprocessor_filename = f"{preprocessor_filename}_{safe_filename}"
    
    # Add extension
    preprocessor_filename = f"{preprocessor_filename}.joblib"
    
    # Full path
    preprocessor_path = os.path.join(output_dir, preprocessor_filename)
    
    try:
        # Try to load the preprocessor
        preprocessor = joblib.load(preprocessor_path)
        return preprocessor
    except (FileNotFoundError, ValueError, Exception) as e:
        print(f"Error loading preprocessor: {e}")
        return None
    
    
#### EFFICIENT PREPROCESSOR

 
from sklearn.compose import ColumnTransformer

def save_preprocessor(preprocessor, filepath, additional_info=None):
    """
    Save a fitted preprocessor and additional information to disk.
    
    Parameters:
    -----------
    preprocessor : ColumnTransformer or Pipeline
        The fitted preprocessor to save
    
    filepath : str
        Path where the preprocessor will be saved
    
    additional_info : dict, default=None
        Additional information to save alongside the preprocessor,
        such as feature names, target encoder, etc.
    
    Returns:
    --------
    str : Path to the saved preprocessor
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    
    # Prepare data to save
    data_to_save = {
        'preprocessor': preprocessor,
        'additional_info': additional_info or {}
    }
    
    # Save to disk
    joblib.dump(data_to_save, filepath)
    
    print(f"Preprocessor saved to {filepath}")
    return filepath


def load_preprocessor(filepath):
    """
    Load a preprocessor and additional information from disk.
    
    Parameters:
    -----------
    filepath : str
        Path to the saved preprocessor
    
    Returns:
    --------
    tuple : (preprocessor, additional_info)
        - preprocessor: The loaded preprocessor
        - additional_info: Dictionary containing additional information
    """
    # Load from disk
    try:
        data = joblib.load(filepath)
        
        # Extract components
        preprocessor = data['preprocessor']
        additional_info = data['additional_info']
        
        print(f"Preprocessor loaded from {filepath}")
        return preprocessor, additional_info
    
    except Exception as e:
        print(f"Error loading preprocessor: {str(e)}")
        return None, None


def preprocess_new_data(preprocessor, df, target_variable=None, feature_names=None):
    """
    Apply preprocessing to new (unseen) data using a saved preprocessor.
    
    Parameters:
    -----------
    preprocessor : ColumnTransformer or Pipeline
        The fitted preprocessor to apply
    
    df : pandas.DataFrame
        Input dataframe containing features to preprocess
    
    target_variable : str, default=None
        Name of the target column, if present in the data.
        If provided, it will be excluded from preprocessing.
    
    feature_names : list, default=None
        Feature names after preprocessing.
        If provided, the output will be converted to a DataFrame with these column names.
    
    Returns:
    --------
    processed_data : pandas.DataFrame or numpy.ndarray
        The preprocessed data
    """
    # Check if target variable needs to be excluded
    if target_variable and target_variable in df.columns:
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
    else:
        X = df
        y = None
    
    # Apply preprocessing
    try:
        X_processed = preprocessor.transform(X)
        
        # Convert to DataFrame if feature names are provided
        if feature_names is not None:
            if hasattr(X_processed, "toarray"):
                X_processed_array = X_processed.toarray()
            else:
                X_processed_array = X_processed
                
            # Ensure dimensions match
            if len(feature_names) == X_processed_array.shape[1]:
                X_processed = pd.DataFrame(X_processed_array, columns=feature_names)
            else:
                print(f"Warning: Feature names length ({len(feature_names)}) doesn't match "
                      f"processed data columns ({X_processed_array.shape[1]}).")
                # Still create a DataFrame, but with generic column names
                X_processed = pd.DataFrame(X_processed_array, 
                                         columns=[f'feature_{i}' for i in range(X_processed_array.shape[1])])
        
        # Return processed data and target if available
        if y is not None:
            return X_processed, y
        else:
            return X_processed
        
    except Exception as e:
        print(f"Error preprocessing data: {str(e)}")
        # Try batch processing if the error might be memory-related
        if "memory" in str(e).lower():
            print("Attempting batch processing...")
            return _preprocess_in_batches(preprocessor, X, feature_names)
        return None


def _preprocess_in_batches(preprocessor, X, feature_names=None, batch_size=5000):
    """
    Apply preprocessing in batches for very large datasets.
    
    Parameters:
    -----------
    preprocessor : ColumnTransformer or Pipeline
        The fitted preprocessor to apply
    
    X : pandas.DataFrame
        Input dataframe containing features to preprocess
    
    feature_names : list, default=None
        Feature names for the preprocessed data
    
    batch_size : int, default=5000
        Size of batches to process
    
    Returns:
    --------
    processed_data : pandas.DataFrame or numpy.ndarray
        The preprocessed data
    """
    try:
        # Process first batch to get output shape
        first_batch = preprocessor.transform(X.iloc[:min(batch_size, len(X))])
        
        # Convert to dense if sparse
        if hasattr(first_batch, "toarray"):
            first_batch = first_batch.toarray()
        
        # Create output array
        output = np.zeros((len(X), first_batch.shape[1]), dtype=first_batch.dtype)
        
        # Copy first batch results
        output[:min(batch_size, len(X))] = first_batch
        
        # Process remaining batches
        for start_idx in range(batch_size, len(X), batch_size):
            end_idx = min(start_idx + batch_size, len(X))
            
            batch = preprocessor.transform(X.iloc[start_idx:end_idx])
            if hasattr(batch, "toarray"):
                batch = batch.toarray()
            
            output[start_idx:end_idx] = batch
            
            # Print progress
            if start_idx % (10 * batch_size) == 0:
                print(f"Processed {end_idx}/{len(X)} rows...")
        
        # Convert to DataFrame if feature names are provided
        if feature_names is not None:
            if len(feature_names) == output.shape[1]:
                return pd.DataFrame(output, columns=feature_names)
            else:
                return pd.DataFrame(output, columns=[f'feature_{i}' for i in range(output.shape[1])])
        
        return output
    
    except Exception as e:
        print(f"Error in batch processing: {str(e)}")
        return None


# Example of how to use these functions

# 1. After training and obtaining your preprocessor:
def example_save_process(df, target_variable):
    """Example showing the complete workflow."""
    from efficient_preprocessing import efficient_preprocessing
    
    # First, preprocess your training data
    preprocessor, X_processed, feature_names = efficient_preprocessing(
        df=df,
        target_variable=target_variable, 
        max_categories=10,
        max_dummy_features=500
    )
    
    # Save the preprocessor and feature names
    save_path = "models/my_preprocessor.joblib"
    save_preprocessor(
        preprocessor=preprocessor,
        filepath=save_path,
        additional_info={
            'feature_names': feature_names,
            'target_variable': target_variable
        }
    )
    
    # Later, when you have new data to preprocess:
    new_data = pd.read_csv("new_data.csv")
    
    # Load the saved preprocessor
    loaded_preprocessor, additional_info = load_preprocessor(save_path)
    
    # Extract the saved feature names
    saved_feature_names = additional_info.get('feature_names')
    
    # Preprocess the new data
    X_new_processed = preprocess_new_data(
        preprocessor=loaded_preprocessor,
        df=new_data,
        target_variable=additional_info.get('target_variable'),
        feature_names=saved_feature_names
    )
    
    return X_new_processed

# If this file is run directly
if __name__ == "__main__":
    # Create a simple example
    import numpy as np
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample DataFrame
    df = pd.DataFrame({
        'num1': np.random.normal(0, 1, n_samples),
        'num2': np.random.normal(5, 2, n_samples),
        'cat1': np.random.choice(['A', 'B', 'C'], n_samples),
        'cat2': np.random.choice(['X', 'Y', 'Z', 'W'], n_samples),
        'target': np.random.randint(0, 2, n_samples)
    })
    
    # Create a preprocessor
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, ['num1', 'num2']),
            ('cat', categorical_transformer, ['cat1', 'cat2'])
        ]
    )
    
    # Fit preprocessor
    X = df.drop(columns=['target'])
    preprocessor.fit(X)
    
    # Get feature names
    feature_names = (
        ['num__num1', 'num__num2'] +
        ['cat__cat1_A', 'cat__cat1_B', 'cat__cat1_C'] +
        ['cat__cat2_X', 'cat__cat2_Y', 'cat__cat2_Z', 'cat__cat2_W']
    )
    
    # Save preprocessor
    save_path = "temp_preprocessor.joblib"
    save_preprocessor(
        preprocessor=preprocessor,
        filepath=save_path,
        additional_info={'feature_names': feature_names}
    )
    
    # Create new data
    new_df = pd.DataFrame({
        'num1': np.random.normal(0, 1, 10),
        'num2': np.random.normal(5, 2, 10),
        'cat1': np.random.choice(['A', 'B', 'C', 'D'], 10),  # Note 'D' is new
        'cat2': np.random.choice(['X', 'Y', 'Z'], 10)
    })
    
    # Load preprocessor
    loaded_preprocessor, additional_info = load_preprocessor(save_path)
    
    # Preprocess new data
    X_new_processed = preprocess_new_data(
        loaded_preprocessor, 
        new_df,
        feature_names=additional_info['feature_names']
    )
    
    print(f"Processed new data shape: {X_new_processed.shape}")
    print(X_new_processed.head())
    
    # Clean up
    if os.path.exists(save_path):
        os.remove(save_path)