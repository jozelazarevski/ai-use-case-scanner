# -*- coding: utf-8 -*-
"""
Created on Thu Apr 17 13:38:14 2025

@author: joze_
"""

import pandas as pd
import numpy as np

def generate_synthetic_sales_data(num_rows=100):
    """Generates synthetic sales data based on the provided columns."""

    data = {
        'SKU': [f'SKU{i+1:03d}' for i in range(num_rows)],
        'Description': np.random.choice(['T-Shirt', 'Coffee Mug', 'Notebook', 'Sneakers', 'Desk Lamp', 'Pen Set', 'Backpack', 'Water Bottle', 'Headphones', 'Tablet Case'], num_rows),
        'Property': np.random.choice(['Color-Blue', 'Size-Large', 'Type-Lined', 'Size-9', 'Style-Modern', 'Color-Black', 'Size-Medium', 'Material-Steel', 'Wireless', 'Fit-10 inch'], num_rows),
        'Product': np.random.choice(['Apparel', 'Home Goods', 'Stationery', 'Footwear', 'Furniture', 'Office Supplies', 'Accessories', 'Electronics'], num_rows),
        'Territory': np.random.choice(['USA', 'Europe', 'Asia', 'Canada', 'Australia'], num_rows),
        'Channel': np.random.choice(['Online', 'Retail', 'Wholesale', 'Distributor'], num_rows),
        'Sales Type': np.random.choice(['Direct', 'Reseller', 'Subscription'], num_rows),
        'Units': np.random.randint(1, 200, num_rows),
        'Price': np.random.uniform(5, 100, num_rows).round(2),
        'Return Units': np.random.randint(0, 10, num_rows),
        'Royalty %': np.random.uniform(0.01, 0.10, num_rows).round(4),
        'Royalty Pay': np.random.choice(['Paid', 'Pending', 'Processing', 'Not Applicable'], num_rows),
        'MONTH': np.random.randint(1, 13, num_rows),
        'YEAR': np.random.randint(2023, 2026, num_rows)
    }

    df = pd.DataFrame(data)

    # Calculate derived columns
    df['Gross Sales'] = (df['Units'] * df['Price']).round(2)
    df['Returns'] = (df['Return Units'] * df['Price']).round(2)
    df['Net Units'] = df['Units'] - df['Return Units']
    df['Net Sales'] = (df['Gross Sales'] - df['Returns']).round(2)
    df['Royalty Payable'] = (df['Net Sales'] * df['Royalty %']).round(2)

    return df

if __name__ == "__main__":
    num_rows_to_generate = 5000  # You can change the number of rows here
    synthetic_data_df = generate_synthetic_sales_data(num_rows=num_rows_to_generate)
    print(synthetic_data_df)

    # You can optionally save the data to a CSV file:
    synthetic_data_df.to_csv('Rolex_sales_data.csv', index=False)
    # print("\nSynthetic data saved to synthetic_sales_data.csv")