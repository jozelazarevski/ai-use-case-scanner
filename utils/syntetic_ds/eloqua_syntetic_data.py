import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
import os

def generate_synthetic_eloqua_data(dataset_type, num_samples=1000, missing_data_percentage=0.1):
    """
    Generates synthetic data resembling Eloqua data for marketing automation tasks.
    The data includes features like contact information, campaign engagement,
    and lead scoring. The target variable depends on the dataset_type.

    Args:
        dataset_type (str): The type of dataset to generate.
            Must be one of: 'lead_scoring', 'campaign_response', 'segmentation'.
        num_samples (int, optional): The number of data points to generate.
            Defaults to 1000.
        missing_data_percentage (float, optional): The percentage of missing data to introduce.
            Defaults to 0.1 (10%).

    Returns:
        pd.DataFrame: A Pandas DataFrame containing the generated data.

    Raises:
        ValueError: If an invalid dataset_type is provided.
    """
    if dataset_type not in ['lead_scoring', 'campaign_response', 'segmentation']:
        raise ValueError("Invalid dataset_type. Must be 'lead_scoring', 'campaign_response', or 'segmentation'.")

    # --- Feature Definitions ---
    num_contacts = 2000  # Total number of unique contacts
    num_campaigns = 50  # Total number of marketing campaigns
    num_segments = 10 # Total number of segments

    # Define features
    features = {
        'contact_id': {'type': 'categorical', 'categories': range(num_contacts)},
        'first_name': {'type': 'string'},
        'last_name': {'type': 'string'},
        'email': {'type': 'string'},
        'company': {'type': 'string'},
        'job_title': {'type': 'string'},
        'industry': {'type': 'categorical', 'categories': ['Technology', 'Finance', 'Healthcare', 'Retail', 'Manufacturing', 'Education']},
        'location': {'type': 'categorical', 'categories': ['North America', 'Europe', 'Asia', 'LATAM', 'ANZ']},
        'campaign_id': {'type': 'categorical', 'categories': range(num_campaigns)},
        'campaign_name': {'type': 'string'},
        'email_open_rate': {'type': 'numeric', 'min': 0, 'max': 1},
        'email_click_rate': {'type': 'numeric', 'min': 0, 'max': 1},
        'form_completion_count': {'type': 'numeric', 'min': 0, 'max': 10},
        'website_visits': {'type': 'numeric', 'min': 0, 'max': 100},
        'time_on_site': {'type': 'numeric', 'min': 0, 'max': 3600},  # in seconds
        'page_views': {'type': 'numeric', 'min': 1, 'max': 50},
        'last_activity_date': {'type': 'datetime', 'start': '2023-01-01', 'end': '2024-12-31'},
        'lead_source': {'type': 'categorical', 'categories': ['Web', 'Referral', 'Phone', 'Trade Show', 'Advertisement', 'Social Media']},
        'salutation': {'type': 'categorical', 'categories': ['Mr.', 'Ms.', 'Mrs.', 'Dr.']},
        'phone': {'type': 'string'},
        'fax': {'type': 'string'},
        'address': {'type': 'string'},
        'city': {'type': 'string'},
        'state': {'type': 'string'},
        'zip': {'type': 'string'},
        'country': {'type': 'string'},
        'annual_revenue': {'type': 'numeric', 'min': 10000, 'max': 100000000},
        'employee_count': {'type': 'numeric', 'min': 1, 'max': 10000},
        'industry_vertical': {'type': 'string'},
        'segment': {'type': 'categorical', 'categories': range(num_segments)},
        'score': {'type': 'numeric', 'min': 0, 'max': 100},
        'engagement_level': {'type': 'categorical', 'categories': ['High', 'Medium', 'Low']},
        'marketing_channel': {'type': 'categorical', 'categories': ['Email', 'Paid Search', 'Organic Search', 'Social Media', 'Display Ads']},
        'device_type': {'type': 'categorical', 'categories': ['Desktop', 'Mobile', 'Tablet']},
    }

    # --- Data Generation ---
    data = {}
    for feature_name, feature_params in features.items():
        if feature_name == 'first_name':
            data[feature_name] = [random.choice(['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Fiona', 'George', 'Hannah', 'Isaac', 'Jack']) for _ in range(num_samples)]
        elif feature_name == 'last_name':
            data[feature_name] = [random.choice(['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis', 'Rodriguez', 'Wilson']) for _ in range(num_samples)]
        elif feature_name == 'email':
            data[feature_name] = [f"{data['first_name'][i].lower()}.{data['last_name'][i].lower()}@{random.choice(['example.com', 'company.com', 'domain.net'])}" for i in range(num_samples)]
        elif feature_name == 'company':
            data[feature_name] = [f"Company {i+1}" for i in range(num_samples)]
        elif feature_name == 'job_title':
            data[feature_name] = [random.choice(['Marketing Manager', 'Sales Director', 'CEO', 'CFO', 'CTO', 'Analyst', 'Consultant', 'Engineer']) for _ in range(num_samples)]
        elif feature_name == 'phone':
            data[feature_name] = [f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}" for _ in range(num_samples)]
        elif feature_name == 'fax':
            data[feature_name] = [f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}" if random.random() < 0.5 else None for _ in range(num_samples)] # 50% have fax
        elif feature_name == 'address':
            data[feature_name] = [f"{random.randint(1, 1000)} {random.choice(['Main', 'First', 'Second', 'Third'])} St." for _ in range(num_samples)]
        elif feature_name == 'city':
            data[feature_name] = [random.choice(['New York', 'London', 'Tokyo', 'Paris', 'Berlin', 'Sydney', 'Toronto', 'Mumbai', 'Shanghai', 'Sao Paulo']) for _ in range(num_samples)]
        elif feature_name == 'state':
            data[feature_name] = [random.choice(['NY', 'CA', 'TX', 'FL', 'GA', 'IL', 'PA', 'OH', 'MI', 'NC']) for _ in range(num_samples)]
        elif feature_name == 'zip':
            data[feature_name] = [f"{random.randint(10000, 99999)}" for _ in range(num_samples)]
        elif feature_name == 'country':
            data[feature_name] = [random.choice(['USA', 'UK', 'Japan', 'France', 'Germany', 'Australia', 'Canada', 'India', 'China', 'Brazil']) for _ in range(num_samples)]
        elif feature_params['type'] == 'numeric':
            data[feature_name] = np.random.randint(feature_params['min'], feature_params['max'] + 1, num_samples)
        elif feature_params['type'] == 'categorical':
            data[feature_name] = np.random.choice(feature_params['categories'], num_samples)
        elif feature_params['type'] == 'datetime':
            start_date = pd.to_datetime(feature_params['start'])
            end_date = pd.to_datetime(feature_params['end'])
            time_between_dates = end_date - start_date
            days_between_dates = time_between_dates.days
            random_days = np.random.randint(0, days_between_dates, num_samples)
            data[feature_name] = start_date + pd.to_timedelta(random_days, unit='d')

    df = pd.DataFrame(data)

    # --- Introduce Missing Data ---
    for col in df.columns:
        mask = np.random.choice([True, False], size=num_samples, p=[1 - missing_data_percentage, missing_data_percentage])
        df.loc[mask, col] = np.nan

    # --- Target Variable Generation ---
    if dataset_type == 'lead_scoring':
        # Target: Lead score (numeric)
        df['target'] = (df['email_open_rate'].fillna(0) * 20 + df['form_completion_count'].fillna(0) * 10 + df['website_visits'].fillna(0) * 5 + df['time_on_site'].fillna(0) / 60).clip(0, 100).astype(int)
    elif dataset_type == 'campaign_response':
        # Target: Campaign response (binary)
        df['target'] = np.where(
            (df['email_open_rate'].fillna(0) > 0.2) & (df['email_click_rate'].fillna(0) > 0.1) | (df['form_completion_count'].fillna(0) > 0),
            1,
            np.random.choice([0, 1], num_samples, p=[0.6, 0.4])  # 40% response rate
        )
    elif dataset_type == 'segmentation':
        # Target: Customer segment (multiclass)
        df['target'] = np.select(
            [
                (df['annual_revenue'].fillna(0) > 1000000) & (df['employee_count'].fillna(0) > 100),  # Enterprise
                (df['annual_revenue'].fillna(0) > 100000) & (df['employee_count'].fillna(0) > 10),    # Mid-Market
                (df['annual_revenue'].fillna(0) <= 100000) | (df['employee_count'].fillna(0) <= 10)     # Small Business
            ],
            ['Enterprise', 'Mid-Market', 'Small Business'],
            default='Small Business'
        )
        df['target'] = df['target'].astype('category')
    return df




def save_dataset(df, dataset_name, directory="synthetic_data_set"):
    """
    Saves the generated dataset to a CSV file in a specified directory.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        dataset_name (str): The name of the dataset.
        directory (str, optional): The directory to save the dataset in.
            Defaults to "synthetic_data_set".
    """
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    filepath = os.path.join(directory, f"{dataset_name}.csv")
    df.to_csv(filepath, index=False)
    print(f"Dataset '{dataset_name}.csv' saved successfully to '{directory}' directory.")



if __name__ == "__main__":
    # --- Generate and Save Datasets ---
    for dataset_type in ['lead_scoring', 'campaign_response', 'segmentation']:
        print(f"Generating {dataset_type} dataset...")
        df = generate_synthetic_eloqua_data(dataset_type, num_samples=1100, missing_data_percentage=0.2) # Added missing data
        save_dataset(df, f"eloqua_{dataset_type}_synthetic")
  