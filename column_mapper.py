# column_mapper.py
"""
Column mapping and validation module using Gemini AI
Handles intelligent column identification, validation, and user confirmation
"""

import json
import re
import os
from datetime import datetime
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import google.generativeai as genai
from config import Config
from utils.user_auth import get_db_connection
import logging

class ColumnMapper:
    """Handles column mapping and validation using Gemini AI"""
    
    
    def __init__(self, gemini_model):
        """Initialize the ColumnMapper with proper logging"""
        self.gemini_model = gemini_model
        self.logger = logging.getLogger(__name__)  # Add this line to create the logger
        self._init_database()   # This calls your existing database initialization
            
    def _init_database(self):
        """Initialize the column mappings database tables in PostgreSQL"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Create or update the column_mappings table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS column_mappings (
                            id VARCHAR(255) PRIMARY KEY,
                            user_id VARCHAR(255) NOT NULL,
                            filename VARCHAR(255) NOT NULL,
                            mappings JSONB NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(user_id, filename)
                        )
                    ''')
                    
                    # Add updated_at column if it doesn't exist
                    cursor.execute('''
                        ALTER TABLE column_mappings 
                        ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    ''')
                    
                    # Table for saved mapping templates (keep this as is)
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS mapping_templates (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(255) NOT NULL,
                            template_name VARCHAR(255) NOT NULL,
                            mappings TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Error initializing database tables: {str(e)}")
    
        """Initialize the column mappings database tables in PostgreSQL"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Table for column mappings
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS column_mappings (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(255) NOT NULL,
                            filename VARCHAR(255) NOT NULL,
                            original_column VARCHAR(255) NOT NULL,
                            mapped_column VARCHAR(255) NOT NULL,
                            column_type VARCHAR(50),
                            column_description TEXT,
                            is_target BOOLEAN DEFAULT FALSE,
                            target_meaning TEXT,
                            prediction_use_cases TEXT,
                            business_value TEXT,
                            model_type VARCHAR(50),
                            confidence_score REAL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(user_id, filename, original_column)
                        )
                    ''')
                    
                    # Table for saved mapping templates
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS mapping_templates (
                            id SERIAL PRIMARY KEY,
                            user_id VARCHAR(255) NOT NULL,
                            template_name VARCHAR(255) NOT NULL,
                            mappings TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    conn.commit()
        except Exception as e:
            print(f"Error initializing database tables: {str(e)}")
    
    def analyze_columns(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """
        Analyze DataFrame columns using Gemini AI
        
        Returns:
            Dict containing column analysis and mapping suggestions
        """
        # Get sample data for analysis
        sample_data = self._get_sample_data(df, n_samples=10)
        
        # Get data statistics
        data_stats = self._get_column_statistics(df)
        
        # Create prompt for Gemini
        prompt = self._create_analysis_prompt(df.columns.tolist(), sample_data, data_stats, filename)
        
        try:
            # Call Gemini API
            response = self.gemini_model.generate_content(prompt)
            
            # Parse response
            analysis = self._parse_gemini_response(response.text)
            
            # Add additional metadata
            analysis['filename'] = filename
            analysis['total_rows'] = len(df)
            analysis['total_columns'] = len(df.columns)
            
            return analysis
            
        except Exception as e:
            print(f"Error in Gemini analysis: {str(e)}")
            # Return basic analysis as fallback
            return self._get_basic_analysis(df, filename)
    
    def _get_sample_data(self, df: pd.DataFrame, n_samples: int = 10) -> Dict[str, List]:
        """Get sample data from each column"""
        sample_data = {}
        
        for col in df.columns:
            # Get non-null samples
            non_null_data = df[col].dropna()
            
            if len(non_null_data) > 0:
                # Get random samples
                n = min(n_samples, len(non_null_data))
                samples = non_null_data.sample(n=n, random_state=42).tolist()
                
                # Convert to native Python types
                samples = [self._convert_to_native_type(s) for s in samples]
                sample_data[col] = samples
            else:
                sample_data[col] = []
        
        return sample_data
    
    def _convert_to_native_type(self, value):
        """Convert numpy/pandas types to native Python types"""
        if pd.isna(value):
            return None
        elif isinstance(value, (np.int64, np.int32, np.int16)):
            return int(value)
        elif isinstance(value, (np.float64, np.float32)):
            return float(value)
        elif isinstance(value, np.bool_):
            return bool(value)
        else:
            return str(value)
    
    def _get_column_statistics(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Get statistical information about each column"""
        stats = {}
        
        for col in df.columns:
            col_stats = {
                'null_count': int(df[col].isnull().sum()),
                'null_percentage': float(df[col].isnull().sum() / len(df) * 100),
                'unique_count': int(df[col].nunique()),
                'dtype': str(df[col].dtype)
            }
            
            # Add numeric statistics if applicable
            if pd.api.types.is_numeric_dtype(df[col]):
                col_stats.update({
                    'mean': float(df[col].mean()) if not df[col].empty else None,
                    'min': float(df[col].min()) if not df[col].empty else None,
                    'max': float(df[col].max()) if not df[col].empty else None,
                    'std': float(df[col].std()) if not df[col].empty else None
                })
            
            # Check if it might be categorical
            if df[col].nunique() < 20 and len(df) > 50:
                col_stats['possible_categorical'] = True
                col_stats['value_counts'] = df[col].value_counts().head(10).to_dict()
            
            stats[col] = col_stats
        
        return stats
    
    def _create_analysis_prompt(self, columns: List[str], sample_data: Dict, 
                               stats: Dict, filename: str) -> str:
        """Create a detailed prompt for Gemini to analyze columns"""
        
        prompt = f"""Analyze the following dataset columns and provide intelligent mapping suggestions.

FILENAME: {filename}

COLUMNS AND SAMPLE DATA:
{json.dumps(sample_data, indent=2)}

COLUMN STATISTICS:
{json.dumps(stats, indent=2)}

Please analyze each column and provide:
1. A standardized, business-friendly column name
2. The data type (numeric, categorical, date, text, identifier, etc.)
3. A description of what the column likely represents
4. Whether it could be a target variable (and if so, what it might predict)
5. Confidence score (0-1) for your analysis
6. For target variables, provide detailed prediction possibilities and use cases

IMPORTANT TARGET VARIABLE ANALYSIS:
- Look for columns that could be prediction targets (outcomes, results, labels, classifications, scores, etc.)
- Common target patterns: 'churn', 'sale', 'fraud', 'approved', 'success', 'status', 'category', 'price', 'revenue', 'score', 'rating', binary yes/no columns
- For each potential target, explain:
  * What specific predictions could be made
  * All tangable concrete use cases for this prediction
  * Business value of predicting this variable
  * Expected model type (classification for categories/binary, regression for numeric)

For columns with ambiguous names (like 'V1', 'col1', 'feature_1', etc.), analyze the content to determine their meaning.

IMPORTANT: Return ONLY a valid JSON object with the following structure:
{{
  "columns": [
    {{
      "original_name": "original column name",
      "suggested_name": "meaningful business name",
      "data_type": "numeric/categorical/date/text/identifier/boolean",
      "description": "what this column represents",
      "is_target": true/false,
      "target_meaning": "if is_target is true, explain what it predicts",
      "prediction_use_cases": ["use case 1", "use case 2", "use case 3"] (only if is_target is true),
      "business_value": "explanation of business value from predicting this" (only if is_target is true),
      "model_type": "classification/regression" (only if is_target is true),
      "confidence": 0.95,
      "reasoning": "brief explanation of your analysis"
    }}
  ],
  "dataset_summary": "Brief description of what this dataset appears to be about",
  "suggested_use_cases": ["use case 1", "use case 2", "use case 3"]
}}

Do not include any markdown formatting or explanatory text outside the JSON."""
        
        return prompt
    
    def _parse_gemini_response(self, response_text: str) -> Dict[str, Any]:
        """Parse and validate Gemini's response"""
        try:
            # Remove any markdown formatting
            cleaned_text = response_text.strip()
            if cleaned_text.startswith('```'):
                cleaned_text = re.sub(r'^```.*?\n', '', cleaned_text)
                cleaned_text = re.sub(r'\n```$', '', cleaned_text)
            
            # Parse JSON
            analysis = json.loads(cleaned_text)
            
            # Validate structure
            if 'columns' not in analysis:
                raise ValueError("Missing 'columns' in response")
            
            return analysis
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error parsing Gemini response: {str(e)}")
            print(f"Response text: {response_text[:500]}...")
            
            # Try to extract any useful information
            return self._extract_partial_analysis(response_text)
    
    def _extract_partial_analysis(self, text: str) -> Dict[str, Any]:
        """Try to extract partial information from malformed response"""
        # This is a fallback method - implement as needed
        return {
            'columns': [],
            'dataset_summary': 'Unable to fully analyze dataset',
            'error': 'Failed to parse AI response'
        }
    
    def _get_basic_analysis(self, df: pd.DataFrame, filename: str) -> Dict[str, Any]:
        """Provide basic analysis without AI"""
        columns = []
        
        for col in df.columns:
            # Basic heuristics for column types
            if pd.api.types.is_numeric_dtype(df[col]):
                data_type = 'numeric'
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                data_type = 'date'
            elif df[col].nunique() < 20:
                data_type = 'categorical'
            else:
                data_type = 'text'
            
            # Enhanced target variable detection
            is_target = False
            target_meaning = None
            prediction_use_cases = []
            business_value = None
            model_type = None
            
            col_lower = col.lower()
            
            # Common target variable patterns
            target_patterns = {
                'churn': {
                    'meaning': 'Customer churn prediction - whether a customer will stop using the service',
                    'use_cases': [
                        'Identify at-risk customers for retention campaigns',
                        'Calculate customer lifetime value predictions',
                        'Optimize customer service resource allocation'
                    ],
                    'business_value': 'Reducing churn by even 5% can increase profits by 25-95%',
                    'model_type': 'classification'
                },
                'fraud': {
                    'meaning': 'Fraud detection - identify potentially fraudulent transactions or activities',
                    'use_cases': [
                        'Real-time transaction fraud detection',
                        'Account takeover prevention',
                        'Risk-based authentication decisions'
                    ],
                    'business_value': 'Prevent financial losses and protect customer trust',
                    'model_type': 'classification'
                },
                'sale': {
                    'meaning': 'Sales prediction - forecast sales amount or probability of sale',
                    'use_cases': [
                        'Revenue forecasting and planning',
                        'Inventory optimization',
                        'Sales team performance prediction'
                    ],
                    'business_value': 'Improve revenue predictability and resource allocation',
                    'model_type': 'regression' if data_type == 'numeric' else 'classification'
                },
                'price': {
                    'meaning': 'Price prediction - estimate optimal pricing or future price movements',
                    'use_cases': [
                        'Dynamic pricing optimization',
                        'Competitive price monitoring',
                        'Demand-based pricing strategies'
                    ],
                    'business_value': 'Maximize revenue through optimal pricing strategies',
                    'model_type': 'regression'
                }
            }
            
            # Check for target patterns
            for pattern, info in target_patterns.items():
                if pattern in col_lower:
                    is_target = True
                    target_meaning = info['meaning']
                    prediction_use_cases = info['use_cases']
                    business_value = info['business_value']
                    model_type = info['model_type']
                    break
            
            # Check for generic target names
            if not is_target and col_lower in ['target', 'label', 'y', 'outcome', 'result', 'class']:
                is_target = True
                # Analyze data to determine target type
                if data_type == 'categorical' or (data_type == 'numeric' and df[col].nunique() < 10):
                    target_meaning = f'Classification target with {df[col].nunique()} classes'
                    model_type = 'classification'
                else:
                    target_meaning = f'Regression target for predicting numeric values'
                    model_type = 'regression'
                
                prediction_use_cases = [
                    'Predict target variable based on input features',
                    'Identify key factors influencing the outcome',
                    'Build automated decision-making system'
                ]
                business_value = 'Automate predictions and improve decision-making efficiency'
            
            column_info = {
                'original_name': col,
                'suggested_name': self._clean_column_name(col),
                'data_type': data_type,
                'description': f'Column {col} of type {data_type}',
                'is_target': is_target,
                'confidence': 0.7 if is_target else 0.5,
                'reasoning': 'Basic heuristic analysis'
            }
            
            # Add target-specific fields if applicable
            if is_target:
                column_info.update({
                    'target_meaning': target_meaning,
                    'prediction_use_cases': prediction_use_cases,
                    'business_value': business_value,
                    'model_type': model_type
                })
            
            columns.append(column_info)
        
        return {
            'columns': columns,
            'dataset_summary': f'Dataset from {filename} with {len(df)} rows and {len(df.columns)} columns',
            'suggested_use_cases': ['Data analysis', 'Predictive modeling', 'Business intelligence']
        }
    
    def _clean_column_name(self, name: str) -> str:
        """Clean and standardize column names"""
        # Replace underscores with spaces and title case
        cleaned = name.replace('_', ' ').replace('-', ' ')
        cleaned = ' '.join(word.capitalize() for word in cleaned.split())
        return cleaned
    
    # In your column_mapper.py file, update the save_mappings method:
    
    def save_mappings(self, user_id, filename, mappings):
        """
        Save column mappings to database
        
        Args:
            user_id (str): User ID
            filename (str): Source filename
            mappings (list): List of column mapping dictionaries
            
        Returns:
            bool: True if successful, False otherwise
        """
        import uuid
        import json
        from datetime import datetime
        from utils.user_auth import get_db_connection
        
        try:
            mapping_id = str(uuid.uuid4())
            
            # Prepare the complete mappings data structure
            # Store everything in the JSONB mappings column
            mappings_data = {
                'columns': mappings,  # All column mappings including prediction_use_cases
                'version': '1.0',     # For future schema updates
                'metadata': {
                    'total_columns': len(mappings),
                    'target_columns': sum(1 for m in mappings if m.get('is_target', False)),
                    'mapped_at': datetime.now().isoformat()
                }
            }
            
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # First check if a mapping already exists for this user and file
                    cursor.execute(
                        """
                        SELECT id FROM column_mappings 
                        WHERE user_id = %s AND filename = %s
                        """,
                        (user_id, filename)
                    )
                    existing = cursor.fetchone()
                    
                    if existing:
                        # Update existing mapping
                        cursor.execute(
                            """
                            UPDATE column_mappings 
                            SET mappings = %s, updated_at = %s
                            WHERE user_id = %s AND filename = %s
                            """,
                            (
                                json.dumps(mappings_data),
                                datetime.now(),
                                user_id,
                                filename
                            )
                        )
                    else:
                        # Insert new mapping
                        cursor.execute(
                            """
                            INSERT INTO column_mappings (id, user_id, filename, mappings, created_at)
                            VALUES (%s, %s, %s, %s, %s)
                            """,
                            (
                                mapping_id,
                                user_id,
                                filename,
                                json.dumps(mappings_data),
                                datetime.now()
                            )
                        )
                    
                    conn.commit()
                    
            self.logger.info(f"Saved mappings for {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving mappings: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
        
    def get_saved_mappings(self, user_id, filename):
        """
        Retrieve saved mappings for a user and filename
        
        Args:
            user_id (str): User ID
            filename (str): Source filename
            
        Returns:
            list: List of column mappings or None if not found
        """
        from utils.user_auth import get_db_connection
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT mappings 
                        FROM column_mappings 
                        WHERE user_id = %s AND filename = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (user_id, filename)
                    )
                    result = cursor.fetchone()
                    
                    if result and result.get('mappings'):
                        mappings_data = result['mappings']
                        
                        # Handle both old and new formats
                        if isinstance(mappings_data, dict) and 'columns' in mappings_data:
                            # New format with metadata
                            columns = mappings_data['columns']
                            
                            # Ensure all mappings have the required fields
                            for mapping in columns:
                                # Map the keys to match what the template expects
                                if 'mapped_name' in mapping and 'suggested_name' not in mapping:
                                    mapping['suggested_name'] = mapping['mapped_name']
                                elif 'suggested_name' in mapping and 'mapped_name' not in mapping:
                                    mapping['mapped_name'] = mapping['suggested_name']
                                
                                # Ensure all required fields exist
                                mapping.setdefault('data_type', 'text')
                                mapping.setdefault('description', '')
                                mapping.setdefault('is_target', False)
                                mapping.setdefault('target_meaning', '')
                                mapping.setdefault('confidence', 0.5)
                                mapping.setdefault('prediction_use_cases', [])
                                mapping.setdefault('business_value', '')
                                mapping.setdefault('model_type', 'classification')
                            
                            return columns
                        elif isinstance(mappings_data, list):
                            # Old format - direct list of mappings
                            # Convert to expected format
                            for mapping in mappings_data:
                                if 'mapped_name' in mapping and 'suggested_name' not in mapping:
                                    mapping['suggested_name'] = mapping['mapped_name']
                                elif 'suggested_name' in mapping and 'mapped_name' not in mapping:
                                    mapping['mapped_name'] = mapping['suggested_name']
                                
                                mapping.setdefault('data_type', 'text')
                                mapping.setdefault('description', '')
                                mapping.setdefault('is_target', False)
                                mapping.setdefault('target_meaning', '')
                                mapping.setdefault('confidence', 0.5)
                                mapping.setdefault('prediction_use_cases', [])
                                mapping.setdefault('business_value', '')
                                mapping.setdefault('model_type', 'classification')
                            
                            return mappings_data
                        else:
                            self.logger.warning(f"Unknown mappings format for {filename}")
                            return None
                            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving mappings: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    

        """
        Retrieve saved mappings for a user and filename
        
        Args:
            user_id (str): User ID
            filename (str): Source filename
            
        Returns:
            list: List of column mappings or None if not found
        """
        from utils.user_auth import get_db_connection
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT mappings 
                        FROM column_mappings 
                        WHERE user_id = %s AND filename = %s
                        ORDER BY created_at DESC
                        LIMIT 1
                        """,
                        (user_id, filename)
                    )
                    result = cursor.fetchone()
                    
                    if result and 'mappings' in result:
                        mappings_data = result['mappings']
                        
                        # Handle both old and new formats
                        if isinstance(mappings_data, dict) and 'columns' in mappings_data:
                            # New format
                            return mappings_data['columns']
                        elif isinstance(mappings_data, list):
                            # Old format - direct list of mappings
                            return mappings_data
                        else:
                            self.logger.warning(f"Unknown mappings format for {filename}")
                            return None
                            
            return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving mappings: {str(e)}")
            return None
    
        """Retrieve saved mappings for a file from PostgreSQL"""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        SELECT original_column, mapped_column, column_type,
                               column_description, is_target, target_meaning,
                               confidence_score, prediction_use_cases, 
                               business_value, model_type
                        FROM column_mappings
                        WHERE user_id = %s AND filename = %s
                    ''', (user_id, filename))
                    
                    rows = cursor.fetchall()
                    
                    mappings = []
                    for row in rows:
                        mapping = {
                            'original_name': row['original_column'],
                            'mapped_name': row['mapped_column'],
                            'data_type': row['column_type'],
                            'description': row['column_description'],
                            'is_target': bool(row['is_target']),
                            'target_meaning': row['target_meaning'],
                            'confidence': row['confidence_score']
                        }
                        
                        # Add target-specific fields if applicable
                        if mapping['is_target']:
                            if row['prediction_use_cases']:
                                mapping['prediction_use_cases'] = json.loads(row['prediction_use_cases'])
                            mapping['business_value'] = row['business_value']
                            mapping['model_type'] = row['model_type']
                        
                        mappings.append(mapping)
                    
                    return mappings
                    
        except Exception as e:
            print(f"Error retrieving mappings: {str(e)}")
            return []
    
    def apply_mappings(self, df: pd.DataFrame, mappings: List[Dict]) -> pd.DataFrame:
        """Apply column mappings to dataframe"""
        df_mapped = df.copy()
        
        # Create mapping dictionary
        rename_dict = {m['original_name']: m['mapped_name'] 
                      for m in mappings 
                      if m['original_name'] != m['mapped_name']}
        
        # Rename columns
        if rename_dict:
            df_mapped.rename(columns=rename_dict, inplace=True)
        
        return df_mapped
    
    def save_template(self, user_id: str, template_name: str, mappings: List[Dict]) -> str:
        """
        Save column mappings as a reusable template
        
        Args:
            user_id (str): User ID
            template_name (str): Name for the template
            mappings (List[Dict]): Column mappings to save
            
        Returns:
            str: Template ID if successful, None otherwise
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Serialize mappings to JSON
                    mappings_json = json.dumps(mappings)
                    
                    cursor.execute('''
                        INSERT INTO mapping_templates (user_id, template_name, mappings)
                        VALUES (%s, %s, %s)
                        RETURNING id
                    ''', (user_id, template_name, mappings_json))
                    
                    template_id = cursor.fetchone()['id']
                    conn.commit()
                    
                    return str(template_id)
                    
        except Exception as e:
            print(f"Error saving template: {str(e)}")
            return None
    
    def get_user_templates(self, user_id: str) -> List[Dict]:
        """
        Get all saved templates for a user
        
        Args:
            user_id (str): User ID
            
        Returns:
            List[Dict]: List of template dictionaries
        """
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        SELECT id, template_name, created_at
                        FROM mapping_templates
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                    ''', (user_id,))
                    
                    rows = cursor.fetchall()
                    
                    templates = []
                    for row in rows:
                        templates.append({
                            'id': row['id'],
                            'name': row['template_name'],
                            'created_at': row['created_at'].isoformat() if row['created_at'] else None
                        })
                    
                    return templates
                    
        except Exception as e:
            print(f"Error retrieving templates: {str(e)}")
            return []