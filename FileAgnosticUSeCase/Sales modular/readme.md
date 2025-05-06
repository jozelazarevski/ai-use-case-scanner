# Product Sales Forecaster

A comprehensive tool for forecasting product sales with improved handling of sparse data and extensive EDA insights.

## Features

- Automatic identification of date, product, and sales columns
- Exploratory Data Analysis (EDA) with rich visualizations and insights
- Intelligent handling of products with limited data
- Multiple forecasting models (SARIMA, XGBoost, fallback methods)
- Product level forecasting with configurable horizon
- Output in CSV, Excel, or JSON formats

## File Structure

```
sales_forecaster/
├── config.py                  # Configuration settings
├── column_identifier.py       # Column detection logic
├── eda_analyzer.py            # EDA functionality
├── logger.py                  # Logging configuration
├── main.py                    # Main entry point
├── product_forecaster.py      # Main forecasting functionality
├── README.md                  # This file
└── logs/                      # Log files directory (created on run)
```

## Requirements

- Python 3.6+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
- xgboost
- pmdarima
- fuzzywuzzy (optional, for better column detection)

Install the required packages:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels xgboost pmdarima fuzzywuzzy python-Levenshtein
```

## Usage

### Basic Usage

Run the forecaster with automatic column detection:

```bash
python main.py --file your_sales_data.csv
```

This will:
1. Load your data
2. Identify date, product, and sales columns
3. Perform EDA with visualizations
4. Preprocess the data
5. Build forecasting models for each product
6. Generate a 3-month forecast
7. Export results to CSV and create visualizations

### Advanced Options

Specify column names explicitly:

```bash
python main.py --file your_sales_data.csv --date_column "Order Date" --product_column "Product ID" --sales_column "Revenue"
```

Configure forecast parameters:

```bash
python main.py --file your_sales_data.csv --forecast_months 6 --min_data_points 5 --export_format excel
```

Skip EDA for faster processing:

```bash
python main.py --file your_sales_data.csv --skip_eda
```

Enable product grouping and pooled data modeling:

```bash
python main.py --file your_sales_data.csv --group_similar --use_pooled_data
```

Customize output location:

```bash
python main.py --file your_sales_data.csv --output_dir custom_output
```

### Full Options List

```
  --file FILE           Path to the data file
  --date_column DATE_COLUMN
                        Name of the date column
  --sales_column SALES_COLUMN
                        Name of the sales column
  --product_column PRODUCT_COLUMN
                        Name of the product column
  --forecast_months FORECAST_MONTHS
                        Number of months to forecast (default: 3)
  --min_data_points MIN_DATA_POINTS
                        Minimum number of data points required (default: 3)
  --group_similar       Group similar products for better forecasting
  --use_pooled_data     Use pooled data for products with insufficient data
  --export_format {csv,excel,json}
                        Export format for results (default: csv)
  --skip_eda            Skip the EDA step
  --output_dir OUTPUT_DIR
                        Directory to store output files (default: output)
  --log_level {DEBUG,INFO,WARNING,ERROR,CRITICAL}
                        Set the logging level (default: INFO)
```

## Output Files

The forecaster generates the following outputs:

### EDA Output
- `output/eda/eda_summary.txt` - Text summary of EDA findings
- `output/eda/monthly_sales_trend.png` - Overall sales trend visualization
- `output/eda/top_products.png` - Top performing products visualization
- `output/eda/monthly_pattern.png` - Monthly sales patterns
- Various other EDA visualizations and data files

### Forecast Output
- `output/product_forecast_summary.csv` - Summary table of all product forecasts
- `output/figures/forecast_*.png` - Individual product forecast visualizations
- `output/figures/top_products_forecast.png` - Forecast for top products
- `output/*_forecast.csv` - Individual CSV files for each product's forecast
- `output/model_performance_metrics.csv` - Model evaluation metrics

## Working with Products with Limited Data

The forecaster handles products with limited data through:

1. **Grouping Similar Products** - Products with similar sales patterns are grouped together
2. **Synthetic Data Generation** - Creates synthetic data based on patterns from similar products
3. **Model Selection** - Uses simpler models for products with limited data
4. **Pooled Modeling** - Leverages patterns from all products to create fallback models

This means you can forecast even for products with very few historical data points.

## License

[MIT License](LICENSE)
