 
from sales_forecast import SalesForecaster

# Create a forecaster instance
file="D:/AI Use Case App_GOOD/data_sets/ecommerce_data_flat.csv"
file="D:/AI Use Case App_GOOD/data_sets/RetailDS/Online Retail.csv"
file="C:/Users/joze_/Downloads/RetailDS/Online_store.csv"
file="C:/Users/joze_/Downloads/RetailDS/NeluxTech Proprietary Retail Dataset.csv"

file = "C:/Users/joze_/Downloads/RetailDS/Global Superstore.txt"
forecaster = ProductSalesForecaster(
    file_path=file,
    # date_col="OrderDate",  # Optional
    # sales_col="Revenue",   # Optional
    forecast_periods=3
)

# Run the complete forecasting pipeline
forecaster.run_pipeline()

# Or run individual steps for more control
forecaster.load_data()
forecaster.identify_columns()
forecaster.preprocess_data()
forecaster.split_data(test_size=0.2)
forecaster.build_model()
forecaster.visualize_results()
forecaster.export_results(output_format='csv')


"""
#### TO DO:
    1. specify the period I want to predict. 3 months instead of 3 days - there should be aggregation
    2. im not sure the formulas are right: Discount value column found: discount_value

         FORMULA USED: Original Sales = Discounted Sales + Discount Amount
         Original Price = sales + discount_value
        
        i want you to explecitly print the values and what columns is ussed as target column
    3. 
      Cell In[116], line 1
        forecaster.identify_columns()

      File D:\AI Use Case App_GOOD\FileAgnosticUSeCase\Sales\sales_forecast.py:939 in identify_columns
        column_scores[col_type][col] = 100

    KeyError: 'date'
"""            



# Import the forecaster class
from sales_forecast import ProductSalesForecaster

# Create a forecaster instance with your data file
forecaster = ProductSalesForecaster(
    file_path=file,
    forecast_months=3
)

# Run the complete pipeline
success = forecaster.run_pipeline()

if success:
    print("Forecasting completed successfully")
else:
    print("Forecasting failed")