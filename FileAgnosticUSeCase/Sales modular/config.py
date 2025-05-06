"""
Configuration settings for the Product Sales Forecaster

This module contains default configuration settings and constants
used throughout the application.
"""

# Default configuration values
DEFAULT_CONFIG = {
    # Forecasting parameters
    'forecast_months': 3,
    'min_data_points': 3,
    'group_similar_products': True,
    'use_pooled_data': True,
    
    # File handling
    'export_format': 'csv',
    'output_dir': 'output',
    
    # Date and sales column patterns for identification
    'date_patterns': [
        'date', 'dt', 'time', 'day', 'month', 'year', 'period',
        'order_date', 'orderdate', 'ord_date', 'sale_date', 'saledate',
        'transaction_date', 'transdate', 'invoice_date', 'invoicedate',
        'ship_date', 'shipdate', 'delivery_date', 'purchase_date'
    ],
    
    'product_patterns': [
        'product', 'prod', 'item', 'goods', 'merchandise', 'stock',
        'product_name', 'productname', 'product_id', 'productid',
        'product_code', 'productcode', 'item_name', 'itemname', 'sku',
        'article', 'commodity', 'material', 'part', 'part_number'
    ],
    
    'sales_patterns': [
        'sales', 'sale', 'revenue', 'rev', 'income', 'turnover',
        'total_sales', 'totalsales', 'sales_amount', 'salesamt',
        'gross_sales', 'net_sales', 'sales_value', 'salesval'
    ],
    
    'quantity_patterns': [
        'quantity', 'qty', 'order_qty', 'orderqty', 'ord_qty',
        'units', 'unit_count', 'count', 'volume', 'vol',
        'num_items', 'numitems', 'item_count', 'pieces', 'pcs'
    ],
    
    'price_patterns': [
        'price', 'unit_price', 'unitprice', 'price_per_unit', 'ppu',
        'rate', 'cost', 'unit_cost', 'unitcost', 'amount', 'amt',
        'item_price', 'item_cost', 'price_per_item'
    ]
}
