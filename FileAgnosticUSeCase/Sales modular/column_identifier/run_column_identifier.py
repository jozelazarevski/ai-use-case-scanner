from column_identifier import ColumnIdentifier
import pandas as pd


df = pd.read_csv("D:/AI Use Case App_GOOD/uploads/Rolex_sales_data.csv")
# df = pd.read_csv("D:/AI Use Case App_GOOD/uploads/california_housing_test.csv")

 
# Process the file with ColumnIdentifier
data_processor = ColumnIdentifier(df)


expected_columns=[  'date' , 'product', 'price','kickoff', 'units_sold']
manual_mapping={'kickoff':'Royalty Payable'} #Format: {'expected_col_type': 'original_col_name', ...}

# First run the automatic mapping
column_mapping, mapped_df = data_processor.identify_all_columns(expected_columns,manual_mapping)


# Display the result
print("\nMapped DataFrame:")
print(f"  Columns: {', '.join(mapped_df.columns)}")
print(f"  Rows: {len(mapped_df)}")
print("\nMapped DataFrame preview:")
print(mapped_df.head(5))

# Display the column mapping
print("\nColumn Mapping:")
print(column_mapping)



# Now apply corrections if the automatic mapping wasn't correct
corrections = {
    'sales': 'Net Sales',     # Override the date column
    
}


# Apply the corrections
updated_mapping, updated_df = data_processor.correct_column_mapping(corrections)



# Display the result
print("\nMapped DataFrame:")
print(f"  Columns: {', '.join(mapped_df.columns)}")
print(f"  Rows: {len(mapped_df)}")
print("\nMapped DataFrame preview:")
print(updated_df.head(5))

# Display the column mapping
print("\nColumn Mapping:")
print(updated_mapping)

df.columns
