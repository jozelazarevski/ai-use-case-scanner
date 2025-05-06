# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 19:51:27 2025

@author: joze_
"""

# column_name_mappings.py

"""
Comprehensive column name mappings for sales and marketing analytics.
This module contains dictionaries for column name identification across various use cases including:
- Sales Forecasting (by region, product, channel, etc.)
- Customer Analytics (churn, lifetime value, segmentation)
- Product Analytics (popularity, recommendation, market basket analysis)
- Territory/Region Optimization
- Seasonal Analysis and more.
"""

# Comprehensive column name mappings for sales and marketing analytics use cases
COLUMN_NAME_MAPPINGS = {
    # Core sales data columns - expanded to include region and product for forecasting
    'quantity': ['units', 'unit', 'qty', 'quantity', 'volume', 'count', 'number_of_items', 'item_count', 'pieces', 'sales_units', 'sold_units'],
    'saled_units': ['units', 'unit', 'qty', 'quantity', 'sold_units', 'sold_qty', 'sold_count', 'sales_units', 'sales_volume', 'sold_volume'],
    'revenue': ['revenue', 'sales', 'gross_sales', 'net_sales', 'gross_revenue', 'net_revenue', 'income', 'turnover', 'sales_amount', 'total_sales'],
    'cost': ['cost', 'price', 'unit_price', 'unit_cost', 'cost_price', 'purchase_price', 'buying_price', 'cogs', 'cost_of_goods_sold', 'product_cost', 'wholesale_price'],
    'profit': ['profit', 'margin', 'profit_margin', 'gross_profit', 'net_profit', 'earnings', 'gain', 'contribution', 'royalty', 'royalty_pay'],
    'price': ['price', 'selling_price', 'sale_price', 'retail_price', 'list_price', 'msrp', 'unit_price', 'price_point'],
    'discount': ['discount', 'discount_amount', 'discount_value', 'promotion', 'promo', 'reduction', 'markdown', 'discount_pct', 'discount_percentage', 'discount_rate'],
    
    # Region-specific columns for regional forecasting
    'region': ['region', 'territory', 'area', 'zone', 'market', 'district', 'division', 'sector', 'sales_region', 'geo', 'geographic_area'],
    'region_id': ['region_id', 'territory_id', 'area_id', 'zone_id', 'market_id', 'district_id', 'division_id'],
    'region_name': ['region_name', 'territory_name', 'area_name', 'zone_name', 'market_name', 'district_name', 'division_name'],
    'region_manager': ['region_manager', 'territory_manager', 'area_manager', 'zone_manager', 'district_manager', 'regional_head'],
    'region_type': ['region_type', 'territory_type', 'area_type', 'zone_type', 'market_type', 'district_type'],
    'region_level': ['region_level', 'territory_level', 'area_level', 'zone_level', 'market_level', 'district_level', 'hierarchy_level'],
    
    # Product-specific columns for product forecasting
    'product': ['product', 'product_name', 'item', 'sku', 'product_id', 'item_id', 'product_code', 'article', 'merchandise', 'good', 'material', 'model', 'product_sku'],
    'product_id': ['product_id', 'item_id', 'sku', 'upc', 'ean', 'product_code', 'article_number', 'material_number', 'model_number', 'product_ref'],
    'product_name': ['product_name', 'item_name', 'description', 'product_description', 'item_description', 'product_title', 'item_title', 'product_desc'],
    'product_category': ['category', 'product_category', 'item_category', 'product_type', 'product_group', 'product_family', 'product_line', 'department', 'product_segment'],
    'product_subcategory': ['subcategory', 'product_subcategory', 'sub_category', 'item_subcategory', 'sub_type', 'sub_group', 'product_class'],
    'product_brand': ['brand', 'product_brand', 'manufacturer', 'vendor', 'make', 'label', 'producer', 'brand_name', 'item_brand'],
    'product_size': ['size', 'product_size', 'item_size', 'dimension', 'weight', 'volume', 'capacity', 'length', 'width', 'height'],
    'product_color': ['color', 'product_color', 'item_color', 'color_code', 'color_name', 'colour', 'product_colour'],
    'product_style': ['style', 'product_style', 'item_style', 'design', 'fashion', 'variant', 'version', 'model'],
    'product_generation': ['generation', 'gen', 'version', 'release', 'edition', 'model_year', 'product_generation'],
    'product_lifecycle': ['lifecycle', 'product_lifecycle', 'lifecycle_stage', 'product_stage', 'maturity', 'introduction', 'growth', 'decline', 'product_age'],
    
    # Seasonal/trend indicators for forecasting
    'season': ['season', 'seasonal', 'seasonality', 'seasonal_flag', 'seasonal_indicator', 'season_name', 'season_code'],
    'trend': ['trend', 'trend_indicator', 'trend_factor', 'trend_direction', 'growth_trend', 'trend_coefficient'],
    'seasonality_index': ['seasonality_index', 'seasonal_index', 'seasonal_factor', 'seasonal_coefficient', 'seasonal_multiplier'],
    'is_holiday': ['is_holiday', 'holiday', 'holiday_flag', 'special_day', 'is_event', 'event_day', 'festival', 'holiday_indicator'],
    'is_promotion': ['is_promotion', 'promotion', 'promo_flag', 'promotional', 'on_promotion', 'has_promotion', 'special_offer'],
    'is_weekend': ['is_weekend', 'weekend', 'weekend_flag', 'is_weekday', 'weekday', 'business_day', 'working_day'],
    
    # Time-related columns
    'date': ['date', 'order_date', 'transaction_date', 'purchase_date', 'sale_date', 'invoice_date', 'timestamp', 'time', 'datetime'],
    'year': ['year', 'yr', 'fiscal_year', 'fy', 'sales_year', 'order_year', 'transaction_year', 'calendar_year'],
    'month': ['month', 'mon', 'mo', 'sales_month', 'order_month', 'transaction_month', 'calendar_month', 'month_name', 'month_num'],
    'quarter': ['quarter', 'qtr', 'q', 'fiscal_quarter', 'sales_quarter', 'order_quarter', 'transaction_quarter', 'calendar_quarter', 'quarter_num'],
    'week': ['week', 'wk', 'sales_week', 'order_week', 'transaction_week', 'calendar_week', 'week_num', 'week_of_year', 'iso_week'],
    'day': ['day', 'day_of_month', 'dom', 'sales_day', 'order_day', 'transaction_day', 'calendar_day', 'day_num'],
    'dayofweek': ['dayofweek', 'weekday', 'day_of_week', 'dow', 'weekday_name', 'weekday_num'],
    
    # Transaction-related columns
    'transaction': ['transaction', 'transaction_id', 'order', 'order_id', 'invoice', 'invoice_id', 'sale', 'sale_id', 'purchase', 'purchase_id', 'receipt'],
    'order_id': ['order_id', 'order_number', 'order_no', 'order_ref', 'invoice_id', 'invoice_number', 'transaction_id', 'transaction_number'],
    'order_date': ['order_date', 'order_time', 'order_timestamp', 'transaction_date', 'transaction_time', 'purchase_date', 'sale_date', 'invoice_date'],
    'order_status': ['order_status', 'status', 'transaction_status', 'fulfillment_status', 'purchase_status', 'process_status', 'state'],
    'order_value': ['order_value', 'transaction_value', 'invoice_value', 'purchase_value', 'sale_value', 'order_amount', 'transaction_amount'],
    'order_quantity': ['order_quantity', 'transaction_quantity', 'order_items', 'transaction_items', 'order_units', 'order_volume'],
    
    # Customer-related columns
    'customer': ['customer', 'customer_id', 'customer_name', 'client', 'client_id', 'client_name', 'buyer', 'account', 'account_id', 'account_name', 'user', 'user_id'],
    'customer_id': ['customer_id', 'client_id', 'account_id', 'buyer_id', 'user_id', 'member_id', 'customer_number', 'client_number'],
    'customer_name': ['customer_name', 'client_name', 'account_name', 'buyer_name', 'full_name', 'person_name'],
    'customer_segment': ['segment', 'customer_segment', 'client_segment', 'customer_type', 'client_type', 'customer_group', 'customer_class', 'category'],
    'customer_since': ['customer_since', 'client_since', 'member_since', 'account_open_date', 'first_purchase_date', 'acquisition_date', 'join_date'],
    
    # Channel/Sales-related columns
    'channel': ['channel', 'sales_channel', 'distribution_channel', 'purchase_channel', 'order_channel', 'platform', 'marketplace', 'source', 'medium'],
    'sales_type': ['sales_type', 'order_type', 'transaction_type', 'sale_type', 'order_category', 'purchase_type', 'business_type'],
    'payment_method': ['payment_method', 'payment_type', 'payment_mode', 'pay_method', 'pay_type', 'payment', 'payment_option'],
    'promotion': ['promotion', 'promo', 'campaign', 'offer', 'deal', 'special', 'promotion_id', 'promo_id', 'campaign_id', 'offer_id'],
    'seller': ['seller', 'sales_rep', 'salesperson', 'sales_agent', 'sales_associate', 'seller_id', 'sales_rep_id', 'agent_id'],
    
    # Location/geography-related columns
    'territory': ['territory', 'sales_territory', 'sales_area', 'sales_zone', 'region', 'district', 'division', 'area', 'zone', 'sector', 'market'],
    'country': ['country', 'nation', 'country_name', 'country_code', 'nationality', 'country_id', 'origin_country', 'destination_country'],
    'state': ['state', 'province', 'state_name', 'state_code', 'province_name', 'province_code', 'administrative_area'],
    'city': ['city', 'town', 'municipality', 'urban_area', 'locality', 'settlement', 'township', 'city_name', 'city_code'],
    'zipcode': ['zipcode', 'zip', 'zip_code', 'postal_code', 'post_code', 'pin_code', 'pin', 'postal'],
    'store': ['store', 'shop', 'outlet', 'branch', 'location', 'site', 'retail_location', 'store_id', 'shop_id', 'branch_id', 'store_code'],
    
    # Returns/refunds columns
    'returns': ['returns', 'returned', 'return_units', 'return_qty', 'return_count', 'return_amount', 'return_quantity', 'refund', 'refund_amount'],
    'return_rate': ['return_rate', 'return_percentage', 'return_ratio', 'refund_rate', 'return_pct', 'return_%', 'returns_to_sales_ratio'],
    'net_units': ['net_units', 'net_quantity', 'net_qty', 'net_count', 'net_volume', 'effective_units', 'remaining_units'],
    
    # Inventory-related columns
    'stock': ['stock', 'inventory', 'on_hand', 'in_stock', 'available', 'stock_qty', 'inventory_qty', 'stock_level', 'inventory_level'],
    'stockout': ['stockout', 'out_of_stock', 'stock_out', 'oos', 'stock_status', 'availability_issue', 'inventory_issue'],
    'inventory_turnover': ['inventory_turnover', 'stock_turnover', 'inventory_turns', 'stock_rotation', 'inventory_movement'],
    
    # Performance/KPI columns
    'margin_pct': ['margin_pct', 'margin_percentage', 'margin_%', 'profit_percentage', 'profit_rate', 'profit_ratio', 'profit_pct', 'profit_%', 'margin_rate'],
    'sales_growth': ['sales_growth', 'growth', 'growth_rate', 'revenue_growth', 'sales_increase', 'year_over_year', 'yoy', 'year_on_year'],
    'forecast': ['forecast', 'prediction', 'projection', 'estimate', 'expected_sales', 'expected_revenue', 'expected_units', 'predicted_value'],
    'target': ['target', 'goal', 'quota', 'plan', 'budget', 'forecast_target', 'sales_target', 'revenue_target', 'unit_target'],
    'actual_vs_target': ['actual_vs_target', 'achievement', 'attainment', 'performance', 'variance', 'deviation', 'achievement_percentage', 'vs_plan'],
    
    # Gross/Net columns
    'gross_sales': ['gross_sales', 'gross_revenue', 'total_sales', 'total_revenue', 'sales_before_returns', 'gross_income', 'total_income'],
    'net_sales': ['net_sales', 'net_revenue', 'effective_sales', 'actual_sales', 'final_sales', 'net_income', 'sales_after_returns']
}

# Column name match scores (default is 15, override for specific types)
COLUMN_NAME_SCORES = {
    'saled_units': 12,
    'revenue': 12,
    'cost': 12,
    'profit': 10,
    'customer': 12,
    'region': 15,       # Higher priority for regional forecasting
    'product': 15,      # Higher priority for product forecasting
    'forecast': 16,     # Higher priority for forecasting indicators
    'target': 14,
    'sales_growth': 13,
    'customer_churn': 16,
    'customer_lifetime_value': 16,
    'basket_analysis': 16,
    'recommendation': 16,
    'territory_optimization': 16
}

# Define fallback relationships between column types (if primary not found, use these alternatives)
COLUMN_TYPE_FALLBACKS = {
    'revenue': ['sales', 'gross_sales', 'net_sales', 'income', 'turnover'],
    'sales': ['revenue', 'gross_sales', 'net_sales', 'income', 'turnover'],
    'saled_units': ['quantity', 'units', 'net_units', 'sales_volume', 'volume'],
    'quantity': ['saled_units', 'units', 'net_units', 'volume', 'count'],
    'cost': ['price', 'unit_cost', 'unit_price', 'buying_price', 'wholesale_price'],
    'price': ['cost', 'unit_price', 'selling_price', 'retail_price', 'list_price'],
    'profit': ['margin', 'earnings', 'royalty', 'net_profit', 'gross_profit'],
    'customer': ['client', 'account', 'buyer', 'user', 'member'],
    'product': ['item', 'sku', 'article', 'merchandise', 'good', 'material'],
    'date': ['order_date', 'transaction_date', 'purchase_date', 'sale_date', 'timestamp'],
    'territory': ['region', 'area', 'zone', 'district', 'market'],
    'region': ['territory', 'area', 'zone', 'district', 'market', 'state', 'country'],
    'return_units': ['returns', 'return_qty', 'return_count', 'returned_items'],
    'forecast': ['prediction', 'projection', 'estimate', 'expected_sales', 'expected_revenue'],
    # Multi-step fallbacks (if A not found, try B, if B not found, try C)
    'regional_sales': ['region_sales', 'territory_sales', 'area_sales', 'geographic_sales', 'market_sales'],
    'product_sales': ['product_revenue', 'item_sales', 'sku_sales', 'product_wise_sales', 'per_product_sales']
}

# Define composite column relationships for advanced analytics
COMPOSITE_COLUMNS = {
    'regional_forecast': {
        'required': ['region', 'date'],
        'optional': ['sales', 'quantity'],
        'description': 'Forecast sales by region'
    },
    'product_forecast': {
        'required': ['product', 'date'],
        'optional': ['sales', 'quantity'],
        'description': 'Forecast sales by product'
    },
    'product_region_forecast': {
        'required': ['product', 'region', 'date'],
        'optional': ['sales', 'quantity'],
        'description': 'Forecast sales by product and region'
    },
    'customer_churn_analysis': {
        'required': ['customer', 'date'],
        'optional': ['recency', 'frequency', 'monetary', 'churn'],
        'description': 'Predict customer churn'
    },
    'market_basket_analysis': {
        'required': ['transaction', 'product'],
        'optional': ['quantity', 'date'],
        'description': 'Analyze product associations'
    },
    'customer_lifetime_value': {
        'required': ['customer', 'sales', 'date'],
        'optional': ['frequency', 'recency'],
        'description': 'Calculate customer lifetime value'
    },
    'product_recommendation': {
        'required': ['customer', 'product', 'transaction'],
        'optional': ['quantity', 'date'],
        'description': 'Generate product recommendations'
    },
    'territory_optimization': {
        'required': ['territory', 'sales'],
        'optional': ['profit', 'cost', 'product'],
        'description': 'Optimize sales territories'
    },
    'promotional_impact': {
        'required': ['date', 'sales', 'is_promotion'],
        'optional': ['product', 'territory'],
        'description': 'Measure promotion effectiveness'
    },
    'seasonal_sales_patterns': {
        'required': ['date', 'sales'],
        'optional': ['product', 'territory', 'season'],
        'description': 'Analyze seasonal patterns in sales'
    }
}