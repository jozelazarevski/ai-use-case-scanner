import pandas as pd
import os

# Set the directory where the data files are located
data_directory = 'data_sets'  # Change this if your directory is different

# Load the datasets
try:
    df_apartments = pd.read_csv(os.path.join(data_directory, "apartments_for_rent_classified_10K.csv"), sep=';')
    df_bank = pd.read_csv(os.path.join(data_directory, "bank-full.csv"), sep=';')
    df_california = pd.read_csv(os.path.join(data_directory, "california_housing_train.csv"))
    df_customer = pd.read_csv(os.path.join(data_directory, "customer.csv"))
    df_ecom = pd.read_csv(os.path.join(data_directory, "e-commerce.csv"))
    df_ecom_flat = pd.read_csv(os.path.join(data_directory, "ecommerce_data_flat.csv"))
    df_maternal = pd.read_csv(os.path.join(data_directory, "Maternal Health Risk Data Set.csv"))
    df_online_retail_1 = pd.read_csv(os.path.join(data_directory, "Online Retail.xlsx - Online Retail.csv"), encoding='unicode_escape')
    df_online_retail_2009_2010 = pd.read_csv(os.path.join(data_directory, "online_retail_II.xlsx - Year 2009-2010.csv"))
    df_online_retail_2010_2011 = pd.read_csv(os.path.join(data_directory, "online_retail_II.xlsx - Year 2010-2011.csv"))
    df_pricerunner = pd.read_csv(os.path.join(data_directory, "pricerunner_aggregate.csv"))
except FileNotFoundError:
    print("Error: One or more of the CSV files were not found. Please make sure the file names are correct and they are in the specified directory.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the data: {e}")
    exit()

# --- Universal Target Variable Dictionary ---
# This dictionary defines target variables that can be created across different datasets,
# even with different column names.  It's designed to be flexible and extensible.
#
# Each target variable is defined by:
#     type: 'binary', 'categorical', or 'regression'
#     source_columns: A *list* of potential source columns.  The code will check if any of these columns
#                     exist in the DataFrame.  This handles variations in column names.
#     method:  (optional) 'median' for binary, 'quantiles' for categorical.  If None, assumes regression.
#     num_quantiles: (optional, for categorical) Number of quantiles to use.
#     labels: (optional, for categorical) List of labels for the categories.
#
# The code will automatically select the first available source column from the list.
#
# This dictionary can be extended with more target variables as needed.
universal_target_config = {
    'HighValue': {
        'type': 'binary',
        'source_columns': ['price', 'Price', 'SalePrice', 'Rent'],  # Add more aliases as needed
        'method': 'median'
    },
    'QuantityCategory': {
        'type': 'categorical',
        'source_columns': ['Quantity', 'quantity', 'Units', 'Amount'],
        'method': 'quantiles',
        'num_quantiles': 3,
        'labels': ['Low', 'Medium', 'High']
    },
    'CustomerAgeGroup': {
        'type': 'categorical',
        'source_columns': ['age', 'Age', 'Customer_Age'],
        'method': 'quantiles',
        'num_quantiles': 4,
        'labels': ['Very Young', 'Young', 'Adult', 'Senior']
    },
    'TotalPrice': {
        'type': 'regression',
        'source_columns': ['Total', 'TotalPrice', 'total_price', 'Amount'],
    },
    'HighRisk': {
        'type': 'binary',
        'source_columns': ['Risk', 'HighRisk', 'high_risk', ' বিপদ'],  # Include non-English
        'method': 'median'
    },
    'RatingCategory':{
        'type':'categorical',
        'source_columns': ['Rating', 'rating'],
        'method': 'quantiles',
        'num_quantiles': 5,
        'labels': ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    },
    'Duration':{
        'type':'regression',
        'source_columns': ['duration', 'Duration', 'time'],
    }
}


def create_target_variables(df, target_config):
    """
    Creates new target variables in a DataFrame based on a universal configuration dictionary.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target_config (dict): A universal dictionary specifying how to create the target variables.

    Returns:
        pd.DataFrame: The modified DataFrame with the new target variables.
    """
    for target_name, config in target_config.items():
        source_columns = config['source_columns']
        target_type = config['type']
        method = config.get('method')
        available_source_column = None

        # Find the first available source column
        for col in source_columns:
            if col in df.columns:
                available_source_column = col
                break  # Stop searching once a valid column is found

        if available_source_column is None:
            print(f"Warning: None of the source columns {source_columns} found in DataFrame for target variable '{target_name}'. Skipping.")
            continue  # Skip to the next target variable

        if target_type == 'binary':
            if method == 'median':
                median_value = df[available_source_column].median()
                df[target_name] = (df[available_source_column] > median_value).astype(int)
            else:
                raise ValueError("For binary targets, 'method' must be 'median'.")

        elif target_type == 'categorical':
            if method == 'quantiles':
                num_quantiles = config.get('num_quantiles', 3)
                labels = config.get('labels', [f'Category_{i}' for i in range(num_quantiles)])
                if len(labels) != num_quantiles:
                    raise ValueError(f"Number of labels must match number of quantiles ({num_quantiles}).")
                try:
                    df[target_name] = pd.qcut(df[available_source_column], q=np.linspace(0, 1, num_quantiles + 1), labels=labels)
                    df[target_name] = df[target_name].astype('category').cat.codes
                except:
                    print(f"Error creating categorical target {target_name} from column {available_source_column}.  Likely not enough unique values.")
                    continue

            else:
                raise ValueError("For categorical targets, 'method' must be 'quantiles'.")

        elif target_type == 'regression':
            df[target_name] = df[available_source_column]
        else:
            raise ValueError(f"Invalid target type: {target_type}. Must be 'binary', 'categorical', or 'regression'.")

    return df

# Example usage:
all_dataframes = [
    df_apartments, df_bank, df_california, df_customer, df_ecom, df_ecom_flat,
    df_maternal, df_online_retail_1, df_online_retail_2009_2010, df_online_retail_2010_2011, df_pricerunner
]

for df in all_dataframes:
    df_name = str(df).split('.')[0].split(' ')[0] # Get dataframe name for printing
    print(f"\nProcessing DataFrame: {df_name}")
    df_processed = create_target_variables(df.copy(), universal_target_config) # Create a copy to avoid modifying original
    print(df_processed.head())  # Display the first few rows with the new target variables
