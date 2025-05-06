import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.feature_selection import VarianceThreshold
from category_encoders import TargetEncoder, BinaryEncoder, HashingEncoder
import warnings
from sklearn.exceptions import ConvergenceWarning
import gc

# Suppress excessive warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def efficient_preprocessing(df, target_variable, 
                           max_categories=20, 
                           max_dummy_features=1000,
                           sample_for_analysis=True,
                           sample_size=50000,
                           memory_efficient=True,
                           feature_selection=True,
                           max_features=100,
                           categorical_encoding_method='auto',
                           numerical_features=None, 
                           categorical_features=None,
                           high_cardinality_features=None,
                           verbose=True):
    """
    Efficiently preprocess large datasets with high-cardinality categorical features.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing all features and the target variable
    
    target_variable : str
        Name of the target column
    
    max_categories : int, default=20
        Maximum number of categories to one-hot encode. Features with more categories
        will be handled using alternate encoding methods
    
    max_dummy_features : int, default=1000
        Maximum total number of dummy variables to create after one-hot encoding
    
    sample_for_analysis : bool, default=True
        Whether to use sampling for initial feature analysis to save memory
    
    sample_size : int, default=50000
        Number of rows to sample for analysis if sample_for_analysis is True
    
    memory_efficient : bool, default=True
        Whether to use memory optimizations like float32 instead of float64
    
    feature_selection : bool, default=True
        Whether to perform feature selection
    
    max_features : int, default=100
        Maximum number of features to select if feature_selection is True
    
    categorical_encoding_method : str, default='auto'
        Method to encode high-cardinality categorical features:
        - 'auto': Automatically choose based on data
        - 'hashing': Use hashing encoder
        - 'target': Use target encoder
        - 'binary': Use binary encoder
        - 'ordinal': Use ordinal encoder
    
    numerical_features : list, default=None
        List of numerical feature columns. If None, will be detected automatically.
    
    categorical_features : list, default=None
        List of categorical feature columns. If None, will be detected automatically.
    
    high_cardinality_features : list, default=None
        List of high cardinality feature columns to handle specially.
        If None, will be detected automatically based on max_categories.
    
    verbose : bool, default=True
        Whether to print details about the preprocessing steps
    
    Returns:
    --------
    tuple : (preprocessor, X_processed, feature_names)
        - preprocessor: Fitted ColumnTransformer for preprocessing new data
        - X_processed: Preprocessed features as a DataFrame
        - feature_names: List of feature names after preprocessing
    """
    if verbose:
        print(f"Starting efficient preprocessing for dataset with {df.shape[0]} rows and {df.shape[1]} columns")
    
    # Make a working copy to avoid modifying the original
    df_copy = df.copy()
    
    # Memory optimization: Convert float64 to float32 if requested
    if memory_efficient:
        for col in df_copy.select_dtypes(include=['float64']).columns:
            df_copy[col] = df_copy[col].astype('float32')
        if verbose:
            print("Converted float64 columns to float32 for memory efficiency")
    
    # Sample the data for analysis if requested
    if sample_for_analysis and df_copy.shape[0] > sample_size:
        df_analysis = df_copy.sample(sample_size, random_state=42)
        if verbose:
            print(f"Using {sample_size} sample rows for feature analysis")
    else:
        df_analysis = df_copy
    
    # Auto-detect features if not provided
    if numerical_features is None:
        numerical_features = df_analysis.select_dtypes(include=['number']).columns.tolist()
        if target_variable in numerical_features:
            numerical_features.remove(target_variable)
    
    if categorical_features is None:
        categorical_features = df_analysis.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_variable in categorical_features:
            categorical_features.remove(target_variable)
    
    if verbose:
        print(f"Detected {len(numerical_features)} numerical features and {len(categorical_features)} categorical features")
    
    # Identify high cardinality features
    if high_cardinality_features is None:
        high_cardinality_features = []
        low_cardinality_features = []
        
        for col in categorical_features:
            n_unique = df_analysis[col].nunique()
            if n_unique > max_categories:
                high_cardinality_features.append(col)
            else:
                low_cardinality_features.append(col)
        
        if verbose:
            print(f"Identified {len(high_cardinality_features)} high-cardinality features (>{max_categories} categories)")
            print(f"Identified {len(low_cardinality_features)} low-cardinality features")
    else:
        low_cardinality_features = [f for f in categorical_features if f not in high_cardinality_features]
    
    # Check potential feature explosion from one-hot encoding
    total_potential_dummies = 0
    for col in low_cardinality_features:
        total_potential_dummies += df_analysis[col].nunique()
    
    if verbose:
        print(f"One-hot encoding would generate approximately {total_potential_dummies} dummy features")
    
    # If too many dummy features would be created, move some to high-cardinality group
    if total_potential_dummies > max_dummy_features:
        if verbose:
            print(f"Too many potential dummy features ({total_potential_dummies} > {max_dummy_features})")
            print("Moving some low-cardinality features to high-cardinality group...")
        
        # Sort features by cardinality (descending)
        feature_cardinality = [(col, df_analysis[col].nunique()) for col in low_cardinality_features]
        feature_cardinality.sort(key=lambda x: x[1], reverse=True)
        
        # Move features to high-cardinality until below threshold
        current_total = total_potential_dummies
        for col, card in feature_cardinality:
            if current_total <= max_dummy_features:
                break
            low_cardinality_features.remove(col)
            high_cardinality_features.append(col)
            current_total -= card
            if verbose:
                print(f"  Moved '{col}' with {card} categories to high-cardinality group")
        
        if verbose:
            print(f"After adjustment: {len(low_cardinality_features)} low-cardinality and {len(high_cardinality_features)} high-cardinality features")
            print(f"Reduced potential dummy features to approximately {current_total}")
    
    # Determine optimal encoding for high-cardinality features
    if categorical_encoding_method == 'auto':
        # Determine the target type (categorical or numerical)
        is_classification = df_analysis[target_variable].dtype == 'object' or df_analysis[target_variable].nunique() < 10
        
        if is_classification:
            # For classification with many categories, hashing is often better
            if len(high_cardinality_features) > 0 and df_analysis[target_variable].nunique() > 10:
                categorical_encoding_method = 'hashing'
            else:
                categorical_encoding_method = 'target'  # Target encoding works well for classification
        else:
            # For regression, target encoding usually works best
            categorical_encoding_method = 'target'
        
        if verbose:
            print(f"Auto-selected '{categorical_encoding_method}' encoding for high-cardinality features")
    
    # Create transformers based on what's available
    transformers = []
    
    # Only add numerical transformer if we have valid numerical features
    if numerical_features:
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        transformers.append(('num', numerical_pipeline, numerical_features))
    
    # Only add low-cardinality transformer if we have valid features
    if low_cardinality_features:
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=True))
        ])
        
        transformers.append(('cat_low', categorical_pipeline, low_cardinality_features))
    
    # Only add high-cardinality transformer if we have valid features
    if high_cardinality_features:
        if categorical_encoding_method == 'hashing':
            high_card_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', HashingEncoder(n_components=min(50, len(high_cardinality_features) * 3)))
            ])
        elif categorical_encoding_method == 'target':
            high_card_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', TargetEncoder())
            ])
        elif categorical_encoding_method == 'binary':
            high_card_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', BinaryEncoder())
            ])
        elif categorical_encoding_method == 'ordinal':
            high_card_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
            ])
        else:
            raise ValueError(f"Unknown encoding method: {categorical_encoding_method}")
        
        transformers.append(('cat_high', high_card_pipeline, high_cardinality_features))
    
    # Create the ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder='drop'  # Drop columns not specified
    )
    
    # Prepare data for preprocessing
    X = df_copy.drop(columns=[target_variable])
    y = df_copy[target_variable]
    
    # Fit the preprocessor
    if verbose:
        print("Fitting preprocessor on data...")
    
    # Try to fit preprocessor with error handling
    try:
        preprocessor.fit(X, y)
    except MemoryError:
        if verbose:
            print("MemoryError: Unable to fit preprocessor due to memory constraints")
            print("Trying with a smaller sample...")
        
        # Try with a smaller sample for fitting
        sample_size_reduced = min(10000, df_copy.shape[0] // 10)
        df_small_sample = df_copy.sample(sample_size_reduced, random_state=42)
        X_small = df_small_sample.drop(columns=[target_variable])
        y_small = df_small_sample[target_variable]
        
        # Clear memory
        del df_small_sample
        gc.collect()
        
        preprocessor.fit(X_small, y_small)
        
        # Clear more memory
        del X_small, y_small
        gc.collect()
    
    # Transform the data
    if verbose:
        print("Transforming data...")
    
    try:
        X_processed = preprocessor.transform(X)
    except MemoryError:
        if verbose:
            print("MemoryError: Unable to transform all data at once")
            print("Trying batch processing...")
        
        batch_size = 10000  # Process in batches of 10,000 rows
        n_batches = (X.shape[0] // batch_size) + (1 if X.shape[0] % batch_size != 0 else 0)
        
        # Process first batch to get shape
        first_batch = preprocessor.transform(X.iloc[:batch_size])
        
        # Initialize result array with proper shape
        if hasattr(first_batch, "toarray"):  # For sparse matrices
            first_batch = first_batch.toarray()
        
        X_processed = np.zeros((X.shape[0], first_batch.shape[1]), 
                              dtype=np.float32 if memory_efficient else np.float64)
        
        # Copy first batch results
        X_processed[:batch_size] = first_batch
        
        # Process remaining batches
        for i in range(1, n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, X.shape[0])
            
            batch_result = preprocessor.transform(X.iloc[start_idx:end_idx])
            if hasattr(batch_result, "toarray"):
                batch_result = batch_result.toarray()
            
            X_processed[start_idx:end_idx] = batch_result
            
            if verbose and i % 10 == 0:
                print(f"  Processed batch {i}/{n_batches}")
        
        if verbose:
            print("Batch processing complete")
    
    # Get feature names
    try:
        feature_names = preprocessor.get_feature_names_out()
    except AttributeError:
        # Fallback for older sklearn versions or custom transformers
        feature_names = []
        for name, _, cols in transformers:
            for col in cols:
                feature_names.append(f"{name}__{col}")
        
        if verbose:
            print("Warning: Could not get exact feature names, using approximate names")
    
    # Feature selection if requested
    if feature_selection and len(feature_names) > max_features:
        if verbose:
            print(f"Performing feature selection to reduce from {len(feature_names)} to max {max_features} features")
        
        # First remove zero-variance features
        if hasattr(X_processed, "toarray"):
            selector = VarianceThreshold()
            X_selected = selector.fit_transform(X_processed.toarray())
            variance_mask = selector.get_support()
        else:
            selector = VarianceThreshold()
            X_selected = selector.fit_transform(X_processed)
            variance_mask = selector.get_support()
        
        # Update feature names after variance threshold
        feature_names_variance = [f for f, m in zip(feature_names, variance_mask) if m]
        
        if verbose:
            print(f"Removed {sum(~variance_mask)} zero-variance features, {len(feature_names_variance)} remaining")
        
        # If still too many features, use SelectKBest
        if len(feature_names_variance) > max_features:
            # Determine the target type again
            is_classification = y.dtype == 'object' or y.nunique() < 10
            
            # Sample for feature selection if dataset is large
            if X_selected.shape[0] > 50000:
                indices = np.random.choice(X_selected.shape[0], 50000, replace=False)
                X_for_selection = X_selected[indices]
                y_for_selection = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
            else:
                X_for_selection = X_selected
                y_for_selection = y
            
            # Convert target to numeric if it's categorical
            if is_classification and (y_for_selection.dtype == 'object' or hasattr(y_for_selection, 'cat')):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                y_for_selection = le.fit_transform(y_for_selection)
            
            # Select top features
            if is_classification:
                selector_k = SelectKBest(mutual_info_classif, k=min(max_features, len(feature_names_variance)))
            else:
                # For regression, use f_regression
                from sklearn.feature_selection import f_regression
                selector_k = SelectKBest(f_regression, k=min(max_features, len(feature_names_variance)))
            
            try:
                X_final = selector_k.fit_transform(X_for_selection, y_for_selection)
                kbest_mask = selector_k.get_support()
                
                # Update feature names and data
                feature_names_final = [f for f, m in zip(feature_names_variance, kbest_mask) if m]
                
                if verbose:
                    print(f"Selected top {len(feature_names_final)} features using {selector_k.__class__.__name__}")
                
                # Apply selection to the full dataset
                X_processed = selector_k.transform(X_selected)
                feature_names = feature_names_final
            except Exception as e:
                if verbose:
                    print(f"Feature selection failed: {str(e)}")
                    print("Using features after variance thresholding")
                X_processed = X_selected
                feature_names = feature_names_variance
        else:
            X_processed = X_selected
            feature_names = feature_names_variance
    
    # Convert to DataFrame
    try:
        if hasattr(X_processed, "toarray"):
            X_df = pd.DataFrame(X_processed.toarray(), columns=feature_names)
        else:
            X_df = pd.DataFrame(X_processed, columns=feature_names)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not convert to DataFrame: {str(e)}")
            print("Returning numpy array instead")
        X_df = X_processed  # Return the numpy array
    
    if verbose:
        print(f"Preprocessing complete: {X_processed.shape[0]} samples with {X_processed.shape[1]} features")
    
    return preprocessor, X_df, feature_names


def apply_efficient_preprocess(df, target_variable, preprocessor=None, **kwargs):
    """
    Apply preprocessing to new data or create a new preprocessor and apply it.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe containing all features and the target variable
    
    target_variable : str
        Name of the target column
    
    preprocessor : ColumnTransformer, default=None
        Fitted preprocessor to use. If None, a new one will be created.
    
    **kwargs : additional arguments
        Additional arguments to pass to efficient_preprocessing if creating a new preprocessor
    
    Returns:
    --------
    tuple : (preprocessor, X_processed, y, feature_names)
        - preprocessor: Fitted ColumnTransformer
        - X_processed: Preprocessed features
        - y: Target variable values
        - feature_names: List of feature names after preprocessing
    """
    # Get the target variable
    y = df[target_variable]
    
    # If no preprocessor is provided, create a new one
    if preprocessor is None:
        preprocessor, X_processed, feature_names = efficient_preprocessing(
            df, target_variable, **kwargs
        )
    else:
        # Use the provided preprocessor to transform the data
        X = df.drop(columns=[target_variable])
        
        try:
            X_processed = preprocessor.transform(X)
        except MemoryError:
            # Use batch processing for large datasets
            batch_size = 10000
            n_batches = (X.shape[0] // batch_size) + (1 if X.shape[0] % batch_size != 0 else 0)
            
            # Process first batch to get shape
            first_batch = preprocessor.transform(X.iloc[:batch_size])
            
            # Initialize result array with proper shape
            if hasattr(first_batch, "toarray"):  # For sparse matrices
                first_batch = first_batch.toarray()
            
            X_processed = np.zeros((X.shape[0], first_batch.shape[1]), 
                                  dtype=np.float32 if kwargs.get('memory_efficient', True) else np.float64)
            
            # Copy first batch results
            X_processed[:batch_size] = first_batch
            
            # Process remaining batches
            for i in range(1, n_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, X.shape[0])
                
                batch_result = preprocessor.transform(X.iloc[start_idx:end_idx])
                if hasattr(batch_result, "toarray"):
                    batch_result = batch_result.toarray()
                
                X_processed[start_idx:end_idx] = batch_result
        
        # Get feature names
        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f"feature_{i}" for i in range(X_processed.shape[1])]
    
    # Convert to DataFrame if possible
    try:
        if hasattr(X_processed, "toarray"):
            X_df = pd.DataFrame(X_processed.toarray(), columns=feature_names)
        else:
            X_df = pd.DataFrame(X_processed, columns=feature_names)
    except:
        X_df = X_processed
    
    return preprocessor, X_df, y, feature_names


# Example usage:
if __name__ == "__main__":
    # Example with synthetic data
    from sklearn.datasets import make_classification
    
    # Create a synthetic dataset with high-cardinality categorical features
    n_samples = 100000
    n_features = 10
    
    # Generate numerical features
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                              n_informative=5, n_redundant=2, random_state=42)
    
    # Create a pandas DataFrame
    df = pd.DataFrame(X, columns=[f'num_feature_{i}' for i in range(n_features)])
    
    # Add some categorical features with different cardinalities
    df['cat_low'] = np.random.choice(['A', 'B', 'C'], size=n_samples)
    df['cat_medium'] = np.random.choice([f'val_{i}' for i in range(30)], size=n_samples)
    df['cat_high'] = np.random.choice([f'category_{i}' for i in range(500)], size=n_samples)
    df['cat_extreme'] = np.random.choice([f'item_{i}' for i in range(5000)], size=n_samples)
    
    # Add target
    df['target'] = y
    
    # Apply efficient preprocessing
    preprocessor, X_processed, feature_names = efficient_preprocessing(
        df, 'target', 
        max_categories=20,
        max_dummy_features=1000,
        sample_for_analysis=True,
        sample_size=10000,
        memory_efficient=True,
        feature_selection=True,
        max_features=50,
        categorical_encoding_method='auto',
        verbose=True
    )
    
    print(f"Final output shape: {X_processed.shape}")
    print(f"Number of features after preprocessing: {len(feature_names)}")
    
import pandas as pd
import numpy as np
import joblib
import os
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