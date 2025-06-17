# gemini_insights.py - Gemini AI Integration for Business Insights

import google.generativeai as genai
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any
import os
from datetime import datetime
import asyncio

class GeminiInsightsGenerator:
    def __init__(self, api_key: str):
        """
        Initialize Gemini AI for business insights generation
        
        Args:
            api_key: Google AI API key for Gemini
        """
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    def generate_chart_insights(self, column_name: str, data_summary: Dict, chart_type: str) -> List[Dict]:
        """
        Generate AI-powered insights for a specific chart using Gemini
        
        Args:
            column_name: Name of the column being analyzed
            data_summary: Statistical summary of the data
            chart_type: Type of chart (histogram, bar_chart, etc.)
        
        Returns:
            List of insights with type, icon, title, and content
        """
        
        # Prepare context for Gemini
        prompt = self._build_chart_analysis_prompt(column_name, data_summary, chart_type)
        
        try:
            response = self.model.generate_content(prompt)
            insights = self._parse_gemini_response(response.text)
            return insights
        except Exception as e:
            print(f"Error generating insights with Gemini: {e}")
            return self._fallback_insights(column_name, data_summary, chart_type)
    
    def generate_overall_insights(self, dataset_summary: Dict) -> List[Dict]:
        """
        Generate overall dataset insights using Gemini AI
        
        Args:
            dataset_summary: Complete dataset statistical summary
        
        Returns:
            List of high-level business insights
        """
        
        prompt = self._build_dataset_analysis_prompt(dataset_summary)
        
        try:
            response = self.model.generate_content(prompt)
            insights = self._parse_gemini_response(response.text)
            return insights
        except Exception as e:
            print(f"Error generating dataset insights: {e}")
            return []
    
    def generate_recommendations(self, dataset_summary: Dict, insights: List[Dict]) -> List[Dict]:
        """
        Generate actionable business recommendations using Gemini AI
        
        Args:
            dataset_summary: Complete dataset summary
            insights: Previously generated insights
        
        Returns:
            List of prioritized recommendations
        """
        
        prompt = self._build_recommendations_prompt(dataset_summary, insights)
        
        try:
            response = self.model.generate_content(prompt)
            recommendations = self._parse_gemini_recommendations(response.text)
            return recommendations
        except Exception as e:
            print(f"Error generating recommendations: {e}")
            return []
    
    def _build_chart_analysis_prompt(self, column_name: str, data_summary: Dict, chart_type: str) -> str:
        """Build detailed prompt for individual chart analysis"""
        
        prompt = f"""
You are a senior business intelligence analyst and data scientist. Analyze this chart data and provide actionable business insights.

CHART INFORMATION:
- Column Name: {column_name}
- Chart Type: {chart_type}
- Data Summary: {json.dumps(data_summary, indent=2)}

ANALYSIS REQUIREMENTS:
1. Statistical Analysis: Interpret the statistical patterns (distribution, variability, outliers)
2. Business Context: Infer business meaning from column name and data patterns
3. Actionable Insights: Provide specific, actionable business recommendations
4. Risk Assessment: Identify potential risks or opportunities
5. Strategic Implications: How this data impacts business strategy

OUTPUT FORMAT (JSON):
Return exactly 3-5 insights in this JSON format:
[
  {{
    "type": "insight|action|warning",
    "icon": "relevant emoji",
    "title": "Clear, specific title",
    "content": "Detailed explanation with specific actions. Include numbers and thresholds."
  }}
]

BUSINESS CONTEXT GUIDELINES:
- If column contains 'age': Focus on demographic strategies, generational marketing
- If column contains 'revenue/sales/income': Focus on customer segmentation, pricing strategy
- If column contains 'score/rating': Focus on performance improvement, quality control
- If column contains 'time/duration': Focus on engagement, retention, process efficiency
- If column contains 'segment/category': Focus on market positioning, targeting
- If column contains 'product/service': Focus on portfolio management, cross-selling
- If column contains 'region/location': Focus on geographic expansion, localization
- If column contains 'channel/source': Focus on marketing mix, attribution

INSIGHT QUALITY CRITERIA:
- Be specific with numbers and thresholds
- Provide 3-4 concrete action items per insight
- Consider industry best practices
- Address both opportunities and risks
- Make insights immediately actionable

Generate insights that a business executive could act on today.
"""
        return prompt
    
    def _build_dataset_analysis_prompt(self, dataset_summary: Dict) -> str:
        """Build prompt for overall dataset analysis"""
        
        prompt = f"""
You are a chief data officer analyzing a business dataset. Provide high-level strategic insights.

DATASET SUMMARY:
{json.dumps(dataset_summary, indent=2)}

ANALYSIS FOCUS:
1. Data Quality Assessment: Overall completeness, consistency, reliability
2. Business Health Indicators: Key metrics that indicate business performance
3. Strategic Opportunities: Areas for growth, optimization, or expansion
4. Risk Factors: Potential threats or weaknesses in the data
5. Competitive Positioning: How the data suggests market position

OUTPUT FORMAT (JSON):
[
  {{
    "type": "insight|opportunity|risk|quality",
    "icon": "relevant emoji",
    "title": "Strategic insight title",
    "content": "Executive-level analysis with business implications"
  }}
]

Focus on insights that would matter to C-level executives and board members.
"""
        return prompt
    
    def _build_recommendations_prompt(self, dataset_summary: Dict, insights: List[Dict]) -> str:
        """Build prompt for actionable recommendations"""
        
        prompt = f"""
You are a management consultant creating an action plan based on data analysis.

DATASET CONTEXT:
{json.dumps(dataset_summary, indent=2)}

PREVIOUS INSIGHTS:
{json.dumps(insights, indent=2)}

TASK: Generate prioritized, actionable business recommendations

OUTPUT FORMAT (JSON):
[
  {{
    "category": "Strategy|Operations|Marketing|Product|Technology",
    "title": "Specific recommendation title",
    "description": "Detailed implementation guidance",
    "priority": "high|medium|low",
    "impact": "Description of expected business impact",
    "timeline": "Suggested implementation timeframe",
    "resources": "Required resources or team"
  }}
]

RECOMMENDATION CRITERIA:
- Directly actionable within 30-90 days
- Clear business impact and ROI potential
- Realistic resource requirements
- Measurable outcomes
- Risk-balanced approach

Prioritize recommendations by potential business impact and implementation feasibility.
"""
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> List[Dict]:
        """Parse Gemini response into structured insights"""
        
        try:
            # Try to extract JSON from response
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                insights = json.loads(json_str)
                
                # Validate and clean insights
                validated_insights = []
                for insight in insights:
                    if all(key in insight for key in ['type', 'icon', 'title', 'content']):
                        validated_insights.append({
                            'type': insight['type'],
                            'icon': insight['icon'],
                            'title': insight['title'][:100],  # Limit title length
                            'content': insight['content']
                        })
                
                return validated_insights
            
        except json.JSONDecodeError:
            pass
        
        # Fallback: Parse as text and structure
        return self._parse_text_response(response_text)
    
    def _parse_gemini_recommendations(self, response_text: str) -> List[Dict]:
        """Parse Gemini recommendations response"""
        
        try:
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = response_text[start_idx:end_idx]
                recommendations = json.loads(json_str)
                
                # Validate recommendations
                validated_recs = []
                for rec in recommendations:
                    required_keys = ['category', 'title', 'description', 'priority']
                    if all(key in rec for key in required_keys):
                        validated_recs.append(rec)
                
                return validated_recs
                
        except json.JSONDecodeError:
            pass
        
        return []
    
    def _parse_text_response(self, text: str) -> List[Dict]:
        """Fallback parser for non-JSON responses"""
        
        insights = []
        lines = text.split('\n')
        
        current_insight = {}
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith('Title:') or line.startswith('**'):
                if current_insight:
                    insights.append(current_insight)
                current_insight = {
                    'type': 'insight',
                    'icon': '📊',
                    'title': line.replace('Title:', '').replace('**', '').strip(),
                    'content': ''
                }
            elif current_insight and line:
                current_insight['content'] += line + ' '
        
        if current_insight:
            insights.append(current_insight)
        
        return insights[:5]  # Limit to 5 insights
    
    def _fallback_insights(self, column_name: str, data_summary: Dict, chart_type: str) -> List[Dict]:
        """Fallback insights when Gemini API fails"""
        
        return [{
            'type': 'insight',
            'icon': '📊',
            'title': f'{column_name} Analysis',
            'content': f'This chart shows the distribution of {column_name}. The data contains {data_summary.get("count", "N/A")} records. Consider deeper analysis to uncover patterns and business opportunities.'
        }]

# Flask Integration Class
class FlaskGeminiIntegration:
    def __init__(self, app, gemini_api_key: str):
        """
        Flask integration for Gemini insights
        
        Args:
            app: Flask application instance
            gemini_api_key: Google AI API key
        """
        self.app = app
        self.insights_generator = GeminiInsightsGenerator(gemini_api_key)
        
    def analyze_chart_data(self, column_name: str, data: List, column_type: str) -> List[Dict]:
        """
        Analyze individual chart data using Gemini
        
        Args:
            column_name: Name of the column
            data: Column data values
            column_type: 'numeric' or 'categorical'
        
        Returns:
            List of AI-generated insights
        """
        
        # Prepare data summary
        if column_type == 'numeric':
            data_summary = self._prepare_numeric_summary(data)
            chart_type = 'histogram'
        else:
            data_summary = self._prepare_categorical_summary(data)
            chart_type = 'bar_chart'
        
        # Add column metadata
        data_summary['column_name'] = column_name
        data_summary['column_type'] = column_type
        data_summary['sample_size'] = len(data)
        
        # Generate insights using Gemini
        insights = self.insights_generator.generate_chart_insights(
            column_name, data_summary, chart_type
        )
        
        return insights
    
    def analyze_full_dataset(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive dataset analysis using Gemini
        
        Args:
            df: Complete dataset DataFrame
        
        Returns:
            Dictionary with overall insights and recommendations
        """
        
        # Prepare dataset summary
        dataset_summary = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': len(df.select_dtypes(exclude=[np.number]).columns),
            'missing_data_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
            'column_names': df.columns.tolist(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024
        }
        
        # Generate insights and recommendations
        overall_insights = self.insights_generator.generate_overall_insights(dataset_summary)
        recommendations = self.insights_generator.generate_recommendations(dataset_summary, overall_insights)
        
        return {
            'overall_insights': overall_insights,
            'recommendations': recommendations,
            'dataset_summary': dataset_summary
        }
    
    def _prepare_numeric_summary(self, data: List) -> Dict:
        """Prepare statistical summary for numeric data"""
        
        clean_data = [x for x in data if x is not None and not pd.isna(x)]
        
        if not clean_data:
            return {'error': 'No valid numeric data'}
        
        clean_data = np.array(clean_data)
        
        return {
            'count': len(clean_data),
            'mean': float(np.mean(clean_data)),
            'median': float(np.median(clean_data)),
            'std': float(np.std(clean_data)),
            'min': float(np.min(clean_data)),
            'max': float(np.max(clean_data)),
            'q25': float(np.percentile(clean_data, 25)),
            'q75': float(np.percentile(clean_data, 75)),
            'skewness': float(self._calculate_skewness(clean_data)),
            'coefficient_of_variation': float(np.std(clean_data) / np.mean(clean_data)) if np.mean(clean_data) != 0 else 0,
            'missing_count': len(data) - len(clean_data)
        }
    
    def _prepare_categorical_summary(self, data: List) -> Dict:
        """Prepare summary for categorical data"""
        
        clean_data = [x for x in data if x is not None and not pd.isna(x) and str(x).strip() != '']
        
        if not clean_data:
            return {'error': 'No valid categorical data'}
        
        value_counts = pd.Series(clean_data).value_counts()
        
        return {
            'count': len(clean_data),
            'unique_values': len(value_counts),
            'top_values': value_counts.head(10).to_dict(),
            'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
            'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
            'most_frequent_percentage': (value_counts.iloc[0] / len(clean_data) * 100) if len(value_counts) > 0 else 0,
            'cardinality_ratio': len(value_counts) / len(clean_data),
            'missing_count': len(data) - len(clean_data)
        }
    
    def _calculate_skewness(self, data: np.array) -> float:
        """Calculate skewness of the data"""
        try:
            from scipy.stats import skew
            return skew(data)
        except ImportError:
            # Manual calculation if scipy not available
            mean = np.mean(data)
            std = np.std(data)
            if std == 0:
                return 0
            return np.mean(((data - mean) / std) ** 3)

# Usage in Flask App
"""
# In your Flask application:

from gemini_insights import FlaskGeminiIntegration
import os

# Initialize Gemini integration
gemini_api_key = os.getenv('GEMINI_API_KEY')  # Set your API key
gemini_insights = FlaskGeminiIntegration(app, gemini_api_key)

@app.route('/business_insights')
def business_insights():
    # Your existing data loading code...
    
    # Generate AI insights for each chart
    chart_insights = {}
    for column in numeric_columns:
        insights = gemini_insights.analyze_chart_data(
            column, 
            df[column].tolist(), 
            'numeric'
        )
        chart_insights[column] = insights
    
    for column in categorical_columns:
        insights = gemini_insights.analyze_chart_data(
            column, 
            df[column].tolist(), 
            'categorical'
        )
        chart_insights[column] = insights
    
    # Generate overall analysis
    full_analysis = gemini_insights.analyze_full_dataset(df)
    
    return render_template('business_insights.html', 
                         data=data,
                         chart_insights=chart_insights,
                         overall_insights=full_analysis['overall_insights'],
                         recommendations=full_analysis['recommendations'])
"""