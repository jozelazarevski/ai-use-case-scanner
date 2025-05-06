"""
Exploratory Data Analysis (EDA) Module

This module contains functionality for performing exploratory data analysis
on sales data, generating insights and visualizations.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

logger = logging.getLogger('sales_forecaster.eda_analyzer')

class EDAAnalyzer:
    """Class for performing exploratory data analysis on sales data"""
    
    def __init__(self, data, date_col, product_col, sales_col, output_dir='output'):
        """
        Initialize the EDA analyzer
        
        Args:
            data (DataFrame): The pandas DataFrame containing the sales data
            date_col (str): Name of the date column
            product_col (str): Name of the product column
            sales_col (str): Name of the sales column
            output_dir (str): Directory to save output files
        """
        self.data = data.copy()
        self.date_col = date_col
        self.product_col = product_col
        self.sales_col = sales_col
        self.output_dir = output_dir
        self.eda_dir = os.path.join(output_dir, 'eda')
        self.eda_results = {}
        
        # Create output directory
        os.makedirs(self.eda_dir, exist_ok=True)
        os.makedirs(os.path.join(self.eda_dir, 'data'), exist_ok=True)
        
        # Set up plotting style
        sns.set(style="whitegrid")
        
    def prepare_data(self):
        """Prepare data for EDA by adding time-based columns"""
        # Convert date column to datetime
        self.data[self.date_col] = pd.to_datetime(self.data[self.date_col], errors='coerce')
        
        # Drop rows with invalid dates
        self.data = self.data.dropna(subset=[self.date_col])
        
        # Ensure sales column is numeric
        self.data[self.sales_col] = pd.to_numeric(self.data[self.sales_col], errors='coerce')
        
        # Drop rows with missing sales values
        self.data = self.data.dropna(subset=[self.sales_col])
        
        # Sort by date
        self.data = self.data.sort_values(by=self.date_col)
        
        # Add time-based columns for analysis
        self.data['year'] = self.data[self.date_col].dt.year
        self.data['month'] = self.data[self.date_col].dt.month
        self.data['quarter'] = self.data[self.date_col].dt.quarter
        self.data['dayofweek'] = self.data[self.date_col].dt.dayofweek
        self.data['month_name'] = self.data[self.date_col].dt.month_name()
        
        return True
        
    def run_full_eda(self):
        """Run the complete EDA process"""
        logger.info("Performing Exploratory Data Analysis (EDA)...")
        
        try:
            # Prepare the data
            self.prepare_data()
            
            # Run all analysis functions
            self.analyze_sales_trend()
            self.analyze_top_products()
            self.analyze_monthly_patterns()
            self.analyze_quarterly_patterns()
            self.analyze_seasonal_products()
            self.analyze_yearly_growth()
            self.analyze_pareto()
            self.analyze_product_growth()
            
            # Generate summary report
            self.generate_summary_report()
            
            logger.info("EDA analysis completed successfully")
            return True
        except Exception as e:
            logger.error(f"Error performing EDA: {e}")
            return False
            
    def analyze_sales_trend(self):
        """Analyze and visualize overall sales trend"""
        logger.info("Analyzing overall sales trend...")
        
        # Group by month and calculate total sales
        monthly_sales = self.data.groupby(pd.Grouper(key=self.date_col, freq='M'))[self.sales_col].sum()
        
        # Store results
        self.eda_results['monthly_sales'] = monthly_sales.to_dict()
        
        # Create visualization
        plt.figure(figsize=(14, 7))
        plt.plot(monthly_sales.index, monthly_sales.values, marker='o')
        plt.title('Monthly Sales Trend')
        plt.xlabel('Date')
        plt.ylabel('Total Sales')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        trend_file = os.path.join(self.eda_dir, 'monthly_sales_trend.png')
        plt.savefig(trend_file)
        plt.close()
        
        logger.info(f"Overall sales trend visualization saved as {trend_file}")
        return True
        
    def analyze_top_products(self):
        """Analyze and visualize top and bottom performing products"""
        logger.info("Analyzing top and bottom products...")
        
        # Group by product and calculate total sales
        top_products = self.data.groupby(self.product_col)[self.sales_col].sum().sort_values(ascending=False)
        top_products_count = min(20, len(top_products))
        
        # Store results
        self.eda_results['top_products'] = top_products.head(top_products_count).to_dict()
        self.eda_results['bottom_products'] = top_products.tail(top_products_count).to_dict()
        
        # Create top products visualization
        plt.figure(figsize=(14, 8))
        top_products.head(top_products_count).plot(kind='bar')
        plt.title(f'Top {top_products_count} Products by Total Sales')
        plt.xlabel('Product')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        top_prod_file = os.path.join(self.eda_dir, 'top_products.png')
        plt.savefig(top_prod_file)
        plt.close()
        
        # Create bottom products visualization
        plt.figure(figsize=(14, 8))
        top_products.tail(top_products_count).plot(kind='bar')
        plt.title(f'Bottom {top_products_count} Products by Total Sales')
        plt.xlabel('Product')
        plt.ylabel('Total Sales')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save the plot
        bottom_prod_file = os.path.join(self.eda_dir, 'bottom_products.png')
        plt.savefig(bottom_prod_file)
        plt.close()
        
        # Export to CSV
        top_products_df = pd.DataFrame({
            'product': top_products.head(top_products_count).index,
            'total_sales': top_products.head(top_products_count).values
        })
        top_products_df.to_csv(os.path.join(self.eda_dir, 'data', 'top_products.csv'), index=False)
        
        bottom_products_df = pd.DataFrame({
            'product': top_products.tail(top_products_count).index,
            'total_sales': top_products.tail(top_products_count).values
        })
        bottom_products_df.to_csv(os.path.join(self.eda_dir, 'data', 'bottom_products.csv'), index=False)
        
        logger.info(f"Top and bottom products analysis saved")
        return True
        
    def analyze_monthly_patterns(self):
        """Analyze and visualize monthly sales patterns"""
        logger.info("Analyzing monthly sales patterns...")
        
        # Group by month and calculate total sales
        monthly_pattern = self.data.groupby('month_name')[self.sales_col].sum()
        
        # Reorder months chronologically
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                     'July', 'August', 'September', 'October', 'November', 'December']
        monthly_pattern = monthly_pattern.reindex(month_order)
        
        # Store results
        self.eda_results['monthly_pattern'] = monthly_pattern.to_dict()
        
        # Create visualization
        plt.figure(figsize=(14, 7))
        monthly_pattern.plot(kind='bar')
        plt.title('Monthly Sales Pattern (Seasonality)')
        plt.xlabel('Month')
        plt.ylabel('Total Sales')
        plt.tight_layout()
        
        # Save the plot
        monthly_pattern_file = os.path.join(self.eda_dir, 'monthly_pattern.png')
        plt.savefig(monthly_pattern_file)
        plt.close()
        
        # Export to CSV
        monthly_pattern_df = pd.DataFrame({
            'month': monthly_pattern.index,
            'total_sales': monthly_pattern.values
        })
        monthly_pattern_df.to_csv(os.path.join(self.eda_dir, 'data', 'monthly_pattern.csv'), index=False)
        
        logger.info(f"Monthly sales pattern analysis saved")
        return True
        
    def analyze_quarterly_patterns(self):
        """Analyze and visualize quarterly sales patterns"""
        logger.info("Analyzing quarterly sales patterns...")
        
        # Group by quarter and calculate total sales
        quarterly_pattern = self.data.groupby('quarter')[self.sales_col].sum()
        
        # Store results
        self.eda_results['quarterly_pattern'] = quarterly_pattern.to_dict()
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        quarterly_pattern.plot(kind='bar')
        plt.title('Quarterly Sales Pattern')
        plt.xlabel('Quarter')
        plt.ylabel('Total Sales')
        plt.xticks(ticks=range(4), labels=['Q1', 'Q2', 'Q3', 'Q4'])
        plt.tight_layout()
        
        # Save the plot
        quarterly_pattern_file = os.path.join(self.eda_dir, 'quarterly_pattern.png')
        plt.savefig(quarterly_pattern_file)
        plt.close()
        
        logger.info(f"Quarterly sales pattern analysis saved")
        return True
        
    def analyze_seasonal_products(self):
        """Analyze and visualize seasonally popular products"""
        logger.info("Analyzing seasonally popular products...")
        
        # Reorder months chronologically
        month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                     'July', 'August', 'September', 'October', 'November', 'December']
        
        # For each month, identify the top 3 products
        seasonal_top_products = {}
        seasonal_products_data = []
        
        for month in month_order:
            month_data = self.data[self.data['month_name'] == month]
            if len(month_data) > 0:
                top3 = month_data.groupby(self.product_col)[self.sales_col].sum().sort_values(ascending=False).head(3)
                seasonal_top_products[month] = top3.to_dict()
                
                # Add to data for export
                for product, sales in top3.items():
                    seasonal_products_data.append({
                        'month': month,
                        'product': product,
                        'sales': sales
                    })
        
        # Store results
        self.eda_results['seasonal_top_products'] = seasonal_top_products
        
        # Visualize the top product for each month
        plt.figure(figsize=(14, 8))
        top_monthly_products = {}
        
        for month in month_order:
            if month in seasonal_top_products and len(seasonal_top_products[month]) > 0:
                top_product = max(seasonal_top_products[month].items(), key=lambda x: x[1])[0]
                top_monthly_products[month] = top_product
        
        if top_monthly_products:
            months = list(top_monthly_products.keys())
            products = list(top_monthly_products.values())
            
            plt.bar(months, [1] * len(months))
            plt.title('Top Selling Product by Month')
            plt.xlabel('Month')
            plt.xticks(rotation=45)
            plt.ylabel('Top Product')
            # Add product labels to each bar
            for i, product in enumerate(products):
                plt.text(i, 0.5, str(product), ha='center', rotation=90, color='white')
            plt.tight_layout()
            
            # Save the plot
            seasonal_prod_file = os.path.join(self.eda_dir, 'seasonal_top_products.png')
            plt.savefig(seasonal_prod_file)
            plt.close()
            
            # Export to CSV
            if seasonal_products_data:
                seasonal_df = pd.DataFrame(seasonal_products_data)
                seasonal_df.to_csv(os.path.join(self.eda_dir, 'data', 'seasonal_top_products.csv'), index=False)
            
            logger.info(f"Seasonal top products analysis saved")
        
        return True
        
    def analyze_yearly_growth(self):
        """Analyze and visualize year-over-year growth"""
        logger.info("Analyzing year-over-year growth...")
        
        # Get unique years
        years = sorted(self.data['year'].unique())
        
        # Only proceed if we have data for multiple years
        if len(years) > 1:
            # Group by year and calculate total sales
            yearly_sales = self.data.groupby('year')[self.sales_col].sum()
            
            # Calculate year-over-year growth rates
            yoy_growth = yearly_sales.pct_change() * 100
            
            # Store results
            self.eda_results['yearly_sales'] = yearly_sales.to_dict()
            self.eda_results['yoy_growth'] = yoy_growth.to_dict()
            
            # Create visualization
            plt.figure(figsize=(12, 6))
            ax1 = plt.subplot(111)
            ax1.bar(yearly_sales.index, yearly_sales.values, color='blue', alpha=0.7)
            ax1.set_xlabel('Year')
            ax1.set_ylabel('Total Sales')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            # Add growth rate line
            ax2 = ax1.twinx()
            ax2.plot(yoy_growth.index, yoy_growth.values, color='red', marker='o', linestyle='-', linewidth=2)
            ax2.set_ylabel('YoY Growth (%)')
            ax2.tick_params(axis='y', labelcolor='red')
            
            plt.title('Year-over-Year Sales Growth')
            plt.tight_layout()
            
            # Save the plot
            yoy_file = os.path.join(self.eda_dir, 'yoy_growth.png')
            plt.savefig(yoy_file)
            plt.close()
            
            logger.info(f"Year-over-Year growth analysis saved")
        
        return True
        
    def analyze_pareto(self):
        """Analyze and visualize Pareto (80/20) principle for product sales"""
        logger.info("Analyzing sales concentration (Pareto principle)...")
        
        # Sort products by sales and calculate cumulative percentage
        product_sales = self.data.groupby(self.product_col)[self.sales_col].sum().sort_values(ascending=False)
        total_sales = product_sales.sum()
        product_sales_pct = product_sales / total_sales * 100
        cumulative_pct = product_sales_pct.cumsum()
        
        # Find the number of products that make up 80% of sales
        products_for_80pct = sum(cumulative_pct <= 80) + 1
        total_products = len(cumulative_pct)
        pareto_ratio = products_for_80pct / total_products
        
        # Store results
        self.eda_results['pareto_analysis'] = {
            'total_products': total_products,
            'products_for_80pct_sales': products_for_80pct,
            'pareto_ratio': pareto_ratio
        }
        
        # Create visualization
        plt.figure(figsize=(14, 7))
        ax1 = plt.subplot(111)
        ax1.bar(range(len(product_sales_pct)), product_sales_pct.values, alpha=0.7)
        ax1.set_xlabel('Products (Ranked by Sales)')
        ax1.set_ylabel('Sales Percentage (%)')
        
        ax2 = ax1.twinx()
        ax2.plot(range(len(cumulative_pct)), cumulative_pct.values, color='red', marker='', linestyle='-', linewidth=2)
        ax2.axhline(y=80, color='green', linestyle='--')
        ax2.axvline(x=products_for_80pct, color='green', linestyle='--')
        ax2.set_ylabel('Cumulative Percentage (%)')
        
        plt.title(f'Pareto Analysis: {products_for_80pct} Products ({pareto_ratio:.1%} of total) Account for 80% of Sales')
        plt.tight_layout()
        
        # Save the plot
        pareto_file = os.path.join(self.eda_dir, 'pareto_analysis.png')
        plt.savefig(pareto_file)
        plt.close()
        
        logger.info(f"Pareto analysis saved")
        return True
        
    def analyze_product_growth(self):
        """Analyze and visualize products with significant growth or decline"""
        logger.info("Analyzing product growth and decline...")
        
        # Get unique years
        years = sorted(self.data['year'].unique())
        
        # Only proceed if we have data for multiple years
        if len(years) > 1:
            # Get last two years
            last_two_years = years[-2:]
            
            growth_data = []
            for product in self.data[self.product_col].unique():
                product_yearly_sales = self.data[self.data[self.product_col] == product].groupby('year')[self.sales_col].sum()
                
                if all(year in product_yearly_sales.index for year in last_two_years):
                    prev_year_sales = product_yearly_sales[last_two_years[0]]
                    current_year_sales = product_yearly_sales[last_two_years[1]]
                    
                    if prev_year_sales > 0:  # Avoid division by zero
                        growth_pct = (current_year_sales - prev_year_sales) / prev_year_sales * 100
                        growth_data.append({
                            'product': product,
                            'previous_year': last_two_years[0],
                            'previous_sales': prev_year_sales,
                            'current_year': last_two_years[1],
                            'current_sales': current_year_sales,
                            'growth_pct': growth_pct
                        })
            
            if growth_data:
                growth_df = pd.DataFrame(growth_data)
                
                # Identify products with significant growth or decline
                # Use 1.5 standard deviations as the threshold for "significant"
                growth_mean = growth_df['growth_pct'].mean()
                growth_std = growth_df['growth_pct'].std()
                significant_threshold = 1.5 * growth_std
                
                high_growth_products = growth_df[growth_df['growth_pct'] > growth_mean + significant_threshold].sort_values('growth_pct', ascending=False)
                high_decline_products = growth_df[growth_df['growth_pct'] < growth_mean - significant_threshold].sort_values('growth_pct')
                
                # Store results
                self.eda_results['high_growth_products'] = high_growth_products.to_dict('records')
                self.eda_results['high_decline_products'] = high_decline_products.to_dict('records')
                
                # Create visualization
                plt.figure(figsize=(14, 10))
                
                # Plot top growth products
                top_growth = min(10, len(high_growth_products))
                if top_growth > 0:
                    plt.subplot(2, 1, 1)
                    bars = plt.barh(high_growth_products['product'].head(top_growth), 
                            high_growth_products['growth_pct'].head(top_growth),
                            color='green')
                    plt.xlabel('Growth (%)')
                    plt.title(f'Top {top_growth} Products with Highest Growth')
                    plt.gca().invert_yaxis()  # Highest growth at the top
                    
                    # Add percentage labels
                    for bar in bars:
                        width = bar.get_width()
                        plt.text(width + 5, bar.get_y() + bar.get_height()/2, 
                                f'{width:.1f}%', ha='left', va='center')
                
                # Plot top decline products
                top_decline = min(10, len(high_decline_products))
                if top_decline > 0:
                    plt.subplot(2, 1, 2)
                    bars = plt.barh(high_decline_products['product'].head(top_decline), 
                            high_decline_products['growth_pct'].head(top_decline),
                            color='red')
                    plt.xlabel('Decline (%)')
                    plt.title(f'Top {top_decline} Products with Highest Decline')
                    plt.gca().invert_yaxis()  # Highest decline at the top
                    
                    # Add percentage labels
                    for bar in bars:
                        width = bar.get_width()
                        plt.text(width - 5, bar.get_y() + bar.get_height()/2, 
                                f'{width:.1f}%', ha='right', va='center')
                
                plt.tight_layout()
                
                # Save the plot
                growth_file = os.path.join(self.eda_dir, 'product_growth_decline.png')
                plt.savefig(growth_file)
                plt.close()
                
                logger.info(f"Product growth/decline analysis saved")
                
                # Export to CSV
                growth_df.to_csv(os.path.join(self.eda_dir, 'data', 'product_growth.csv'), index=False)
        
        return True
        
    def generate_summary_report(self):
        """Generate a comprehensive EDA summary report"""
        logger.info("Generating EDA summary report...")
        
        with open(os.path.join(self.eda_dir, 'eda_summary.txt'), 'w') as f:
            f.write("SALES DATA EXPLORATORY ANALYSIS SUMMARY\n")
            f.write("=======================================\n\n")
            
            f.write(f"Data Period: {self.data[self.date_col].min().strftime('%Y-%m-%d')} to {self.data[self.date_col].max().strftime('%Y-%m-%d')}\n")
            f.write(f"Total Products: {self.data[self.product_col].nunique()}\n")
            f.write(f"Total Sales: {self.data[self.sales_col].sum():.2f}\n\n")
            
            # Top Products
            if 'top_products' in self.eda_results:
                f.write("TOP PERFORMING PRODUCTS\n")
                f.write("----------------------\n")
                for i, (product, sales) in enumerate(list(self.eda_results['top_products'].items())[:10]):
                    f.write(f"{i+1}. {product}: {sales:.2f}\n")
                f.write("\n")
            
            # Monthly Pattern
            if 'monthly_pattern' in self.eda_results:
                f.write("MONTHLY SALES PATTERN\n")
                f.write("--------------------\n")
                for month, sales in self.eda_results['monthly_pattern'].items():
                    f.write(f"{month}: {sales:.2f}\n")
                f.write("\n")
            
            # Pareto Analysis
            if 'pareto_analysis' in self.eda_results:
                f.write("SALES CONCENTRATION (PARETO ANALYSIS)\n")
                f.write("-----------------------------------\n")
                pareto = self.eda_results['pareto_analysis']
                f.write(f"{pareto['products_for_80pct_sales']} products ({pareto['pareto_ratio']:.1%} of all products) account for 80% of total sales\n\n")
            
            # Seasonal Top Products
            if 'seasonal_top_products' in self.eda_results:
                f.write("SEASONAL TOP PRODUCTS\n")
                f.write("--------------------\n")
                for month, products_dict in self.eda_results['seasonal_top_products'].items():
                    f.write(f"{month}:\n")
                    for i, (product, sales) in enumerate(list(products_dict.items())[:3]):
                        f.write(f"  {i+1}. {product}: {sales:.2f}\n")
                f.write("\n")
            
            # Year-over-Year Growth
            if 'yoy_growth' in self.eda_results:
                f.write("YEAR-OVER-YEAR GROWTH\n")
                f.write("--------------------\n")
                for year, growth in self.eda_results['yoy_growth'].items():
                    if not pd.isna(growth):
                        f.write(f"{year}: {growth:.2f}%\n")
                f.write("\n")
            
            # Products with High Growth/Decline
            if 'high_growth_products' in self.eda_results:
                f.write("PRODUCTS WITH SIGNIFICANT GROWTH\n")
                f.write("------------------------------\n")
                for product in self.eda_results['high_growth_products'][:10]:  # Top 10
                    f.write(f"{product['product']}: {product['growth_pct']:.2f}%\n")
                f.write("\n")
            
            if 'high_decline_products' in self.eda_results:
                f.write("PRODUCTS WITH SIGNIFICANT DECLINE\n")
                f.write("-------------------------------\n")
                for product in self.eda_results['high_decline_products'][:10]:  # Top 10
                    f.write(f"{product['product']}: {product['growth_pct']:.2f}%\n")
                f.write("\n")
        
        logger.info("EDA summary report generated successfully")
        return True
