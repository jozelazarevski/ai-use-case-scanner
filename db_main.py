# -*- coding: utf-8 -*-
"""
Main database interface module for the AI Use Case Generator application.
Integrates with user_auth.py for authentication and session management.
"""

import re
import tempfile
import csv
import logging
import traceback
import os
import json
import requests
import time
import random
import pandas as pd
import numpy as np
import uuid
import shutil
import sys
from werkzeug.utils import secure_filename
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session, g
from flask_session import Session
from functools import wraps
import importlib
from ml_trainer import train_model_with_robust_error_handling
from utils.ml_utils import predict_with_preprocessor
from enhanced_column_mapper import EnhancedColumnMapper

 
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)   

# Import config
from config import Config, encodings

# Import user authentication module
from utils.user_auth import (
    get_db_connection, login_required, get_user_by_id, init_database,
    save_user_model, get_user_models, get_model_by_id, delete_user_model,
    save_use_cases, get_user_use_cases, delete_use_case,
    get_user_embeddings, create_model_embedding, get_embedding_by_id, delete_embedding, 
    init_auth_routes
)
from utils.read_file import read_data_flexible
 
gemini_insights=None

def make_json_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    import numpy as np
    
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                         np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, 'item'):  # numpy scalar
        return obj.item()
    else:
        return obj

# Dynamically handle ML module imports
def import_ml_module(module_name, default_value=None):
    """
    Safely import optional modules, returning a default value if import fails
    
    Args:
        module_name (str): Module to import
        default_value: Value to return if import fails
        
    Returns:
        module or default_value: The imported module or the default value
    """
    try:
        return importlib.import_module(module_name)
    except ImportError:
        logger.warning(f"Could not import module: {module_name}")
        return default_value

# Try to import ML modules
try:
    # Add parent directory to path to import modules from ml folder
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    
    # Import ML trainer modules
    ml_trainer = import_ml_module('ml_trainer')
    execute_LLM_model = import_ml_module('execute_LLM_model')
    read_file = import_ml_module('utils.read_file')
    
    # Import ML prediction modules
    predict_classification = import_ml_module('ml.predict_model_classification')
    predict_regression = import_ml_module('ml.predict_model_regression')
    # read_data_flexible = import_ml_module('ml.read_file')
    
    # Only try importing Google's libraries if configured to use them
    if Config.ACTIVE_MODEL.lower() == 'gemini':
        import google.generativeai as genai
        HAS_GEMINI = True
        logger.info("Google Generative AI (Gemini) library loaded successfully")
    else:
        HAS_GEMINI = False
except ImportError as e:
    logger.error(f"Error importing ML modules: {e}")
    # Set default placeholders for missing modules
    ml_trainer = None
    execute_LLM_model = None
    read_file = None
    predict_classification = None
    predict_regression = None
    read_data_flexible = None
    HAS_GEMINI = False

# Initialize Flask app
def create_app():
    """
    Create and configure the Flask application
    
    Returns:
        Flask: Configured Flask app
    """
    app = Flask(__name__, static_folder='static')
    
    # Create directories if they don't exist
    if not os.path.exists('static'):
        os.makedirs('static')
        
    # Create database directories if they don't exist
    if not os.path.exists(Config.DATABASE_DIR):
        os.makedirs(Config.DATABASE_DIR)
    
    # Create user models directory if it doesn't exist
    user_models_dir = os.path.join(Config.DATABASE_DIR, 'user_models')
    if not os.path.exists(user_models_dir):
        os.makedirs(user_models_dir)
    
    # Load configuration from Config class
    app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
    app.config['ALLOWED_EXTENSIONS'] = Config.ALLOWED_EXTENSIONS
    app.config['SECRET_KEY'] = Config.SECRET_KEY or os.urandom(24)
    
    # Session configuration from Config class
    app.config['SESSION_TYPE'] = Config.SESSION_TYPE
    app.config['SESSION_FILE_DIR'] = Config.SESSION_FILE_DIR
    app.config['SESSION_PERMANENT'] = Config.SESSION_PERMANENT
    app.config['SESSION_USE_SIGNER'] = Config.SESSION_USE_SIGNER
    app.config['SESSION_COOKIE_MAX_SIZE'] = Config.SESSION_COOKIE_MAX_SIZE
    app.config['SESSION_COOKIE_SECURE'] = Config.SESSION_COOKIE_SECURE
    app.config['SESSION_COOKIE_HTTPONLY'] = Config.SESSION_COOKIE_HTTPONLY
    app.config['SESSION_COOKIE_SAMESITE'] = Config.SESSION_COOKIE_SAMESITE
    
    # Initialize Flask-Session
    Session(app)
    
    # Initialize the database
    init_database()
    
    # Verify EDA database schema
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Check if EDA columns exist in column_mappings table
                cursor.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name = 'column_mappings' AND column_name = 'eda_results'
                """)
                eda_column_exists = cursor.fetchone()
                
                if not eda_column_exists:
                    logger.info("Adding EDA support to column_mappings table")
                    cursor.execute("""
                        ALTER TABLE column_mappings 
                        ADD COLUMN IF NOT EXISTS eda_results JSONB
                    """)
                    conn.commit()
                    logger.info("EDA column added successfully")
    except Exception as e:
        logger.error(f"Error updating database schema for EDA: {str(e)}")
    
    # Initialize authentication routes and middleware
    init_auth_routes(app)
    
    # Get active model from config
    ACTIVE_MODEL = Config.ACTIVE_MODEL.lower()
    if ACTIVE_MODEL not in ['claude', 'gemini']:
        logger.warning(f"Unknown model '{ACTIVE_MODEL}' specified. Defaulting to Claude.")
        ACTIVE_MODEL = 'claude'
    
    # Claude API configuration
    CLAUDE_API_KEY = Config.CLAUDE_API_KEY
    CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
    CLAUDE_MODEL = Config.CLAUDE_MODEL
    
    # Gemini API configuration
    GOOGLE_API_KEY = Config.GOOGLE_API_KEY
    GEMINI_MODEL_NAME = Config.GEMINI_MODEL
    
    # Initialize Gemini if it's the active model and API key is available
    HAS_GEMINI_CONFIG = False
    
    
    if ACTIVE_MODEL == 'gemini' and HAS_GEMINI and GOOGLE_API_KEY:
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            HAS_GEMINI_CONFIG = True
            logger.info(f"Gemini model '{GEMINI_MODEL_NAME}' configured successfully.")
        except Exception as e:
            logger.error(f"Error configuring Gemini API: {str(e)}")
            logger.info("Falling back to Claude due to Gemini configuration error.")
            ACTIVE_MODEL = 'claude'
    
    # Create necessary directories if they don't exist
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    
    # Register template filters and global functions
    app.jinja_env.filters['nl2br'] = lambda text: text.replace('\n', '<br>') if text else ''
    
    def read_script_file(script_path):
        """
        Read the content of a script file
        
        Args:
            script_path (str): Path to the script file
            
        Returns:
            str: Content of the script file
        """
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"
    
    app.jinja_env.globals.update(read_script_file=read_script_file)
    
    # Custom template filters
    @app.template_filter('datetimeformat')
    def datetimeformat(value, format='%Y-%m-%d %H:%M'):
        """
        Custom Jinja2 filter to format datetime strings
        
        Args:
            value (str): ISO formatted datetime string
            format (str, optional): Desired output format. Defaults to '%Y-%m-%d %H:%M'
        
        Returns:
            str: Formatted datetime string
        """
        try:
            # Parse the ISO formatted datetime string
            dt = datetime.fromisoformat(value)
            return dt.strftime(format)
        except (ValueError, TypeError):
            # If parsing fails, return the original value
            return value
    
    @app.template_filter('to_json_safe')
    def to_json_safe(obj):
        """
        Convert NumPy types to Python native types for JSON serialization
        """
        import numpy as np
        import json
        
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, 
                                  np.uint8, np.uint16, np.uint32, np.uint64)):
                    return int(obj)
                if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, np.bool_):
                    return bool(obj)
                return super().default(obj)
        
        return json.dumps(obj, cls=NumpyEncoder)
    
    return app, ACTIVE_MODEL, HAS_GEMINI_CONFIG, CLAUDE_API_KEY, CLAUDE_API_URL, CLAUDE_MODEL, GEMINI_MODEL_NAME

# Helper functions
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    app = current_app()
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_unique_filename_timestamp(file="", model_type="", user_id=None):
    """
    Generates a unique filename using a timestamp.
    
    Args:
        file (str): Original filename
        model_type (str): Type of model
        user_id (str): User ID
        
    Returns:
        str: Unique filename
    """
    timestamp = int(time.time() * 1000)  # Get current timestamp in milliseconds
    
    # Check if user is logged in and use their ID
    if user_id is None and 'user_id' in session:
        user_id = session['user_id'][:8]  # Use first 8 chars of user ID
    
    # Default user id if none provided
    if not user_id:
        user_id = 'anonymous'
        
    filename = f"{user_id}_{model_type}_{file}_{timestamp}"
    # Remove special characters that could cause issues in filenames
    filename = re.sub(r'[^a-zA-Z0-9_\-]', "", filename)

    return filename


     
def ensure_target_coverage(proposals, target_columns):
    """
    Ensure that there's at least one proposal for each identified target column
    
    Args:
        proposals (List[Dict]): List of AI proposals
        target_columns (List[Dict]): List of identified target columns
        
    Returns:
        List[Dict]: Enhanced proposals list
    """
    # Check which targets are covered
    covered_targets = set()
    for proposal in proposals:
        target_var = proposal.get('target_variable', '').lower()
        for target in target_columns:
            if target['name'].lower() == target_var:
                covered_targets.add(target['name'].lower())
    
    # Add proposals for missing targets
    for target in target_columns:
        if target['name'].lower() not in covered_targets:
            logger.info(f"Adding missing proposal for target: {target['name']}")
            new_proposal = create_proposal_for_target(target)
            proposals.append(new_proposal)
    
    return proposals


def create_proposal_for_target(target_info):
    """
    Create a proposal for a specific target variable
    
    Args:
        target_info (Dict): Information about the target column
        
    Returns:
        Dict: A complete proposal for this target
    """
    # Determine model type
    model_type = target_info.get('model_type', 'auto')
    if model_type == 'auto':
        # Infer from data type
        data_type = target_info.get('data_type', 'unknown').lower()
        if data_type in ['boolean', 'categorical'] or 'class' in target_info['name'].lower():
            model_type = 'classification'
        elif data_type in ['numeric', 'float', 'integer']:
            model_type = 'regression'
        else:
            model_type = 'classification'  # default
    
    # Create title based on target
    title = f"Predict {target_info['name']}"
    if 'churn' in target_info['name'].lower():
        title = "Customer Churn Prediction"
    elif 'fraud' in target_info['name'].lower():
        title = "Fraud Detection System"
    elif 'sale' in target_info['name'].lower():
        title = "Sales Prediction Model"
    
    # Generate description
    description = f"Develop a {model_type} model to predict {target_info['name']}. "
    description += f"{target_info.get('meaning', '')} "
    description += f"This AI solution will analyze the available data features to make accurate predictions, "
    description += f"enabling proactive decision-making and operational efficiency. "
    description += f"The model will provide actionable insights that can be integrated into business workflows."
    
    # Generate KPIs
    kpis = []
    if target_info.get('use_cases'):
        # Use suggested use cases as KPIs
        kpis = [f"KPI: {uc}" for uc in target_info['use_cases'][:3]]
    else:
        # Generate generic KPIs based on model type
        if model_type == 'classification':
            kpis = [
                f"Achieve >85% accuracy in {target_info['name']} prediction",
                f"Reduce false positive rate to below 10%",
                f"Improve decision-making speed by 50%"
            ]
        else:
            kpis = [
                f"Achieve R² score above 0.8 for {target_info['name']} prediction",
                f"Reduce prediction error (RMSE) by 30%",
                f"Enable accurate forecasting for business planning"
            ]
    
    # Business value
    business_value = target_info.get('business_value', '')
    if not business_value:
        business_value = f"By accurately predicting {target_info['name']}, the organization can make data-driven decisions, "
        business_value += f"optimize resource allocation, and improve operational efficiency. "
        business_value += f"This will lead to cost savings and better business outcomes."
    
    # Prediction interpretation
    prediction_interpretation = f"The model will output predictions for {target_info['name']}. "
    if model_type == 'classification':
        prediction_interpretation += f"Each prediction will indicate the predicted class/category with an associated confidence score. "
        prediction_interpretation += f"Higher confidence scores indicate more reliable predictions."
    else:
        prediction_interpretation += f"The predicted values represent the expected {target_info['name']} based on the input features. "
        prediction_interpretation += f"Prediction intervals can be provided to indicate uncertainty."
    
    return {
        "title": title,
        "description": description,
        "kpis": kpis,
        "business_value": business_value,
        "target_variable": target_info['name'],
        "model_type": model_type,
        "use_case_implementation_complexity": "medium",
        "prediction_interpretation": prediction_interpretation,
        "target_variable_understanding": target_info.get('meaning', f"The target variable {target_info['name']} represents the outcome we want to predict.")
    }


def create_proposals_from_targets(target_columns):
    """
    Create proposals directly from identified target columns when API is not available
    
    Args:
        target_columns (List[Dict]): List of target column information
        
    Returns:
        List[Dict]: List of proposals
    """
    proposals = []
    for target in target_columns:
        proposal = create_proposal_for_target(target)
        proposals.append(proposal)
    
    # If no targets were identified, create a generic proposal
    if not proposals:
        proposals.append({
            "title": "Data Analysis and Insights",
            "description": "Analyze the dataset to discover patterns and insights.",
            "kpis": ["Identify key patterns", "Generate insights", "Support decision-making"],
            "business_value": "Transform data into actionable intelligence.",
            "target_variable": "Unknown",
            "model_type": "auto",
            "use_case_implementation_complexity": "easy",
            "prediction_interpretation": "Analysis results will provide insights into data patterns.",
            "target_variable_understanding": "No specific target variable identified."
        })
    
    return proposals
def get_claude_proposals(prompt_text, claude_api_key, claude_api_url, claude_model, column_mappings=None, target_columns=None):
    """
    Get proposals from Claude API
    
    Args:
        prompt_text (str): Prompt to send to Claude
        claude_api_key (str): Claude API key
        claude_api_url (str): Claude API URL
        claude_model (str): Claude model identifier
        column_mappings (list): List of column mapping dictionaries (optional)
        target_columns (list): List of identified target columns (optional)
        
    Returns:
        List[Dict]: List of AI use case proposal dictionaries
    """
    # Prepare the headers for Claude API request
    headers = {
        "x-api-key": claude_api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json"
    }
    
    # Prepare the request data
    data = {
        "model": claude_model,
        "max_tokens": 4000,
        "messages": [
            {"role": "user", "content": prompt_text}
        ]
    }
    
    # Make the API request with retries
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Calling Claude API - attempt {attempt+1}")
            response = requests.post(claude_api_url, headers=headers, json=data, timeout=60)
            
            # Log response status for debugging
            logger.info(f"Status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                # Extract text from Claude's response
                if result.get("content"):
                    content_blocks = result["content"]
                    text = ""
                    for block in content_blocks:
                        if block.get("type") == "text":
                            text += block.get("text", "")
                    
                    # Try to parse the response as JSON first
                    try:
                        # Extract JSON content from text (in case there's surrounding text)
                        json_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
                        if json_match:
                            json_text = json_match.group(0)
                        else:
                            json_text = text
                            
                        proposals = json.loads(json_text)
                        
                        # Validate structure of proposals
                        if isinstance(proposals, list) and len(proposals) > 0:
                            proposals = sanitize_proposals(proposals)
                            
                            # Ensure coverage of all identified targets
                            if target_columns:
                                proposals = ensure_target_coverage(proposals, target_columns)
                            
                            return proposals
                        else:
                            raise ValueError("Invalid JSON structure: expected a list of proposal objects")
                    
                    except (json.JSONDecodeError, ValueError) as json_error:
                        logger.error(f"JSON parsing failed: {str(json_error)}. Falling back to text parsing.")
                        # Fall back to the original text parsing method
                        proposals = parse_proposals(text)
                        if proposals:
                            if target_columns:
                                proposals = ensure_target_coverage(proposals, target_columns)
                            return proposals
                        else:
                            raise ValueError("Failed to parse proposals from Claude's response")
                    
            # Handle rate limiting (429) with exponential backoff
            if response.status_code == 429:
                wait_time = (2 ** attempt) + 1  # Exponential backoff: 1, 3, 7 seconds
                logger.warning(f"Rate limited. Waiting {wait_time} seconds before retrying...")
                time.sleep(wait_time)
                continue
                
            # Log other error details
            error_message = f"API Error: Status {response.status_code}, Response: {response.text}"
            logger.error(error_message)
            last_error = ValueError(error_message)
            
        except Exception as e:
            error_message = f"Exception during API call: {str(e)}"
            logger.error(error_message)
            last_error = e
            
            # Wait before retrying
            time.sleep(2)
    
    # If we get here, all attempts failed
    if last_error:
        raise last_error
    else:
        raise ValueError("Failed to get proposals from Claude API after multiple attempts")          
# Function to get AI use case proposals using the selected model with column metadata
def get_ai_use_case_proposals(file_content, filename, active_model, has_gemini_config, 
                             claude_api_key, claude_api_url, claude_model, gemini_model=None,
                             column_mappings=None):
    """
    Generate AI use case proposals using the selected AI model based on file content.
    
    Args:
        file_content (str): Content of the uploaded file
        filename (str): Original filename
        active_model (str): Selected AI model ('claude' or 'gemini')
        has_gemini_config (bool): Whether Gemini is properly configured
        claude_api_key (str): Claude API key
        claude_api_url (str): Claude API URL
        claude_model (str): Claude model identifier
        gemini_model: Gemini model instance (optional)
        column_mappings (list): List of column mapping dictionaries with user validations (optional)
        
    Returns:
        List[Dict]: List of AI use case proposal dictionaries
    """
    # JSON structure example
    json_example = '''
{
  "title": "Title of the use case",
  "description": "Detailed description of the use case (2-3 paragraphs)",
  "kpis": ["Business KPI 1 : one sentence description", "Business KPI 2 : one sentence description", "Business KPI 3 : one sentence description"],
  "business_value": "Comprehensive explanation of business value",
  "target_variable": "exact_column_name_from_dataset",
  "model_type": "classification/regression/clustering/sentiment analysis",
  "use_case_implementation_complexity": "hard/medium/easy",
  "prediction_interpretation": "comprehensive explanation how to interpret the AI prediction with examples",
  "target_variable_understanding": "analysis of the target variable and its meaning for the use case"
}'''
    
    # Build detailed column information if mappings are provided
    column_info_section = ""
    target_columns_section = ""
    all_columns_list = ""
    
    if column_mappings:
        # Create lists for different types of columns
        column_descriptions = []
        target_columns = []
        all_column_names = []
        
        for mapping in column_mappings:
            # Collect all column names
            all_column_names.append(mapping['mapped_name'])
            
            # Build detailed column description
            col_desc = f"- **{mapping['mapped_name']}**"
            if mapping['original_name'] != mapping['mapped_name']:
                col_desc += f" (originally: {mapping['original_name']})"
            col_desc += f"\n  - Type: {mapping.get('data_type', 'unknown')}"
            col_desc += f"\n  - Description: {mapping.get('description', 'No description provided')}"
            
            column_descriptions.append(col_desc)
            
            # Collect target columns with full details
            if mapping.get('is_target', False):
                target_info = {
                    'name': mapping['mapped_name'],
                    'original_name': mapping['original_name'],
                    'meaning': mapping.get('target_meaning', ''),
                    'use_cases': mapping.get('prediction_use_cases', []),
                    'business_value': mapping.get('business_value', ''),
                    'model_type': mapping.get('model_type', 'auto'),
                    'data_type': mapping.get('data_type', 'unknown'),
                    'description': mapping.get('description', '')
                }
                target_columns.append(target_info)
        
        # Build column information section
        all_columns_list = f"\nAVAILABLE COLUMNS IN DATASET: {', '.join(all_column_names)}\n"
        
        column_info_section = f"""
VALIDATED COLUMN INFORMATION:
{chr(10).join(column_descriptions)}
"""
        
        # Build detailed target columns section
        if target_columns:
            target_columns_section = f"""
PRIORITY TARGET VARIABLES IDENTIFIED BY USER:
You MUST create at least one use case for each of these target variables:

"""
            for i, target in enumerate(target_columns, 1):
                target_columns_section += f"""
{i}. Target Column: "{target['name']}"
   - User-Validated Meaning: {target['meaning']}
   - Data Type: {target['data_type']}
   - Description: {target['description']}
   - Recommended Model Type: {target['model_type']}
   - Business Value: {target['business_value']}
   - User Suggested Use Cases: {'; '.join(target['use_cases']) if target['use_cases'] else 'Create relevant use cases'}
"""
    
    # Create the prompt with strong emphasis on using validated information
    prompt_text = f"""You are analyzing a dataset to create AI/ML use case proposals. The user has carefully validated and corrected the column information below. You MUST use this validated information in your proposals.

FILENAME: {filename}
{all_columns_list}
{column_info_section}
{target_columns_section}

FILE CONTENT SAMPLE (first 100 rows):
{file_content[:20000]}

CRITICAL INSTRUCTIONS:
1. You MUST create at least one use case for EACH identified target variable listed above
2. Use ONLY the exact column names provided (the mapped/validated names, NOT original names)
3. For target variables, use the user-validated meanings and descriptions
4. Do NOT suggest use cases that require columns not present in the dataset
5. Do NOT suggest use cases that require feature engineering
6. Each use case must be practical and implementable with the existing data

For each AI use case proposal, provide:
1. A clear, business-focused title
2. A detailed description (2-3 paragraphs) explaining:
   - What business problem this solves
   - How the AI/ML model would work
   - What insights or automation it provides
3. 3-5 specific, measurable KPIs
4. Clear business value proposition
5. The EXACT target variable name from the validated columns
6. Appropriate model type based on the target variable's data type
7. Implementation complexity assessment
8. Detailed prediction interpretation with business context
9. Target variable understanding based on the user's validation

Return ONLY a valid JSON array with this structure:
{json_example}

IMPORTANT: 
- If user identified "Churn" as a target, you MUST include a customer churn prediction use case
- If user identified any other specific targets, create use cases for those as well
- Each use case should be distinct and valuable
- Use the exact column names from the validated list above
"""
    
    # Log the prompt for debugging
    logger.info(f"Number of target columns identified: {len(target_columns) if column_mappings else 0}")
    if column_mappings and target_columns:
        logger.info(f"Target columns: {[t['name'] for t in target_columns]}")
    
    if active_model == "gemini" and has_gemini_config and gemini_model:
        # Use Gemini API
        try:
            logger.info("Using Gemini API to generate proposals")
            response = gemini_model.generate_content(prompt_text)
            if response and hasattr(response, 'text'):
                text = response.text
                
                # Try to parse JSON from response
                try:
                    # Extract JSON content from text
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
                    else:
                        json_text = text
                        
                    proposals = json.loads(json_text)
                    
                    # Validate and clean up proposals
                    if isinstance(proposals, list) and len(proposals) > 0:
                        proposals = sanitize_proposals(proposals)
                        
                        # Ensure we have proposals for all identified targets
                        if column_mappings and target_columns:
                            proposals = ensure_target_coverage(proposals, target_columns)
                        
                        return proposals
                    else:
                        raise ValueError("Invalid JSON structure: expected a list of proposal objects")
                    
                except (json.JSONDecodeError, ValueError) as json_error:
                    logger.error(f"Gemini JSON parsing failed: {str(json_error)}. Falling back to text parsing.")
                    # Fall back to the original text parsing method
                    proposals = parse_proposals(text)
                    if proposals:
                        if column_mappings and target_columns:
                            proposals = ensure_target_coverage(proposals, target_columns)
                        return proposals
                    else:
                        raise ValueError("Failed to parse proposals from Gemini's response")
            else:
                raise ValueError("Empty or invalid response from Gemini API")
                
        except Exception as e:
            error_message = f"Error with Gemini API: {str(e)}"
            logger.error(error_message)
            # Fall back to Claude if Gemini fails
            if claude_api_key:
                logger.info("Falling back to Claude API")
                return get_claude_proposals(prompt_text, claude_api_key, claude_api_url, claude_model, column_mappings, target_columns)
            else:
                raise ValueError(f"Error with Gemini API: {str(e)}. Claude API not configured as fallback.")
    
    else:
        # Use Claude API (default)
        if not claude_api_key:
            # Return proposals based on identified targets if no API key
            if column_mappings and target_columns:
                return create_proposals_from_targets(target_columns)
            else:
                logger.warning("Claude API key not configured. Returning dummy data.")
                return [{
                    "title": "Example AI Use Case",
                    "description": "This is a dummy use case proposal since no API key is configured.",
                    "kpis": ["Example KPI 1", "Example KPI 2"],
                    "business_value": "Configure a Claude or Gemini API key to get real use case proposals.",
                    "target_variable": "dummy_variable",
                    "model_type": "classification",
                    "use_case_implementation_complexity": "medium",
                    "prediction_interpretation": "This is a placeholder. Configure an API key for actual results.",
                    "target_variable_understanding": "Configure an API key to see real analysis."
                }]
        
        return get_claude_proposals(prompt_text, claude_api_key, claude_api_url, claude_model, column_mappings, target_columns)
    
    
def sanitize_proposals(proposals):
    """
    Sanitize and validate proposal structure
    
    Args:
        proposals (List[Dict]): Raw proposals list
        
    Returns:
        List[Dict]: Sanitized proposals
    """
    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("Proposals must be a list of objects")
        
        # Ensure all required fields are present with defaults if missing
        if 'title' not in proposal:
            proposal['title'] = "AI Use Case"
        if 'description' not in proposal:
            proposal['description'] = "Description not provided"
        if 'kpis' not in proposal or not isinstance(proposal['kpis'], list):
            proposal['kpis'] = []
        if 'target_variable' not in proposal:
            proposal['target_variable'] = "Unknown"
        if 'model_type' not in proposal:
            proposal['model_type'] = "auto"
        if 'business_value' not in proposal:
            proposal['business_value'] = "Not specified"
        if 'use_case_implementation_complexity' not in proposal:
            proposal['use_case_implementation_complexity'] = "medium"
        if 'prediction_interpretation' not in proposal:
            proposal['prediction_interpretation'] = "Predictions should be interpreted in the context of the business problem and validated by domain experts."
    
    return proposals

def parse_proposals(text):
    """Parse Claude's response into structured use case proposals."""
    logger.info("Parsing Claude's response text")
    # Clean up the text
    text = text.strip()
    
    # Remove any introductory or concluding text
    intro_pattern = r'^(.*?)(?=\b(?:Use Case|AI Use Case|Proposal|Title)\b)'
    intro_match = re.search(intro_pattern, text, re.IGNORECASE | re.DOTALL)
    if intro_match:
        intro_text = intro_match.group(1).strip()
        if intro_text and len(intro_text) < 150:  # Only remove if it looks like an intro
            text = text[len(intro_text):].strip()
    
    # Try to split text into separate use cases
    proposals = []
    
    # Method 1: Split by use case number or titles
    use_case_pattern = r'(?:\n|^)(?:Use Case|AI Use Case|Proposal)(?:\s*\d+)?(?:\s*[:–-])?\s*(.*?)(?=\n|$)'
    use_case_matches = re.finditer(use_case_pattern, text, re.IGNORECASE)
    
    start_positions = []
    for match in use_case_matches:
        start_positions.append(match.start())
    
    # Add end of string as final position
    if start_positions:
        start_positions.append(len(text))
        
        # Extract each use case section
        for i in range(len(start_positions) - 1):
            use_case_text = text[start_positions[i]:start_positions[i+1]].strip()
            proposal = parse_single_proposal(use_case_text)
            if proposal:
                proposals.append(proposal)
    
    # Method 2: If no matches found, try splitting by numbered items
    if not proposals:
        use_case_splits = re.split(r'\n\s*\d+[\.\)]\s+', '\n' + text)
        if len(use_case_splits) > 1:
            # Remove empty first element from split
            use_case_splits = [split for split in use_case_splits if split.strip()]
            
            for use_case_text in use_case_splits:
                proposal = parse_single_proposal(use_case_text)
                if proposal:
                    proposals.append(proposal)
    
    # Method 3: If still no matches, try splitting by double newlines
    if not proposals:
        use_case_splits = re.split(r'\n\n\n+', text)
        for use_case_text in use_case_splits:
            if len(use_case_text.strip()) > 50:  # Only consider substantial chunks
                proposal = parse_single_proposal(use_case_text)
                if proposal:
                    proposals.append(proposal)
    
    # If we couldn't parse any proposals, create one with the entire text
    if not proposals:
        proposals = [{
            "title": "AI Use Case Proposal",
            "description": text,
            "kpis": [],
            "target_variable": "Unknown",
            "model_type": "auto",
            "business_value": "Not specified",
            "use_case_implementation_complexity": "medium",
            "prediction_interpretation": "Predictions should be interpreted in the context of the business problem and validated by domain experts."
        }]
    
    logger.info(f"Found {len(proposals)} proposals")
    return proposals

def parse_single_proposal(text):
    """Parse a single use case proposal text into structured data."""
    if not text or len(text.strip()) < 10:
        return None
        
    lines = text.strip().split('\n')
    
    # Default values
    title = "AI Use Case"
    description = ""
    kpis = []
    target_variable = "Unknown"
    model_type = "auto"
    business_value = "Not specified"
    use_case_implementation_complexity = "medium"
    prediction_interpretation = "Predictions should be interpreted in the context of the business problem and validated by domain experts."
    
    # Extract title - typically the first line
    title_match = re.search(r'(?:Title|Use Case|AI Use Case)(?:\s*\d+)?(?:\s*[:–-])?\s*(.*?)(?=\n|$)', text, re.IGNORECASE)
    if title_match:
        title = title_match.group(1).strip()
    elif lines and lines[0]:
        # Use first line as title if it's not too long
        if len(lines[0]) < 100:
            title = lines[0].strip()
    
    # Look for target variable
    target_match = re.search(r'TARGET_VARIABLE:\s*([^\n]+)', text, re.IGNORECASE)
    if target_match:
        target_variable = target_match.group(1).strip()
    
    # Look for model type
    model_match = re.search(r'MODEL_TYPE:\s*([^\n]+)', text, re.IGNORECASE)
    if model_match:
        model_type_str = model_match.group(1).strip().lower()
        if model_type_str in ['classification', 'regression', 'clustering', 'sentiment_analysis']:
            model_type = model_type_str
    
    # Look for KPIs section
    kpi_section_start = -1
    for i, line in enumerate(lines):
        if re.search(r'\b(?:KPIs?|Key\s+Performance\s+Indicators?)\b', line, re.IGNORECASE):
            kpi_section_start = i
            break
    
    # Extract description (everything between title and KPIs)
    if kpi_section_start > 0:
        title_end = 1  # Skip the first line (title)
        description = '\n'.join(lines[title_end:kpi_section_start]).strip()
        
        # Extract KPIs
        kpi_section = '\n'.join(lines[kpi_section_start:]).strip()
        kpi_items = re.findall(r'(?:^|\n)\s*(?:\d+\.|\-|\*|\•)\s*(.*?)(?=\n\s*(?:\d+\.|\-|\*|\•)|$)', kpi_section, re.DOTALL)
        
        if kpi_items:
            kpis = [kpi.strip() for kpi in kpi_items if kpi.strip()]
        else:
            # If no bullet points found, try line by line after the KPI header
            for line in lines[kpi_section_start+1:]:
                line = line.strip()
                if line and not line.lower().startswith(('kpi', 'key performance')):
                    kpis.append(line)
    else:
        # No KPIs found, use everything after the title as description
        description = '\n'.join(lines[1:]).strip()
    
    # Clean up description - remove target_variable and model_type lines
    description = re.sub(r'^TARGET_VARIABLE:.*$', '', description, flags=re.MULTILINE).strip()
    description = re.sub(r'^MODEL_TYPE:.*$', '', description, flags=re.MULTILINE).strip()
    
    # Clean up any remaining markdown formatting
    title = re.sub(r'\*\*(.*?)\*\*', r'\1', title)
    description = re.sub(r'\*\*(.*?)\*\*', r'\1', description)
    kpis = [re.sub(r'\*\*(.*?)\*\*', r'\1', kpi) for kpi in kpis]
    
    return {
        "title": title,
        "description": description,
        "kpis": kpis,
        "target_variable": target_variable,
        "model_type": model_type,
        "business_value": business_value,
        "use_case_implementation_complexity": use_case_implementation_complexity,
        "prediction_interpretation": prediction_interpretation
    }

def generate_model_explanation(training_stats, active_model, has_gemini_config, 
                              claude_api_key, claude_api_url, claude_model, gemini_model=None):
    """
    Generate an explanation of the model results using LLM
    
    Args:
        training_stats (dict): Results from the training execution
        active_model (str): Selected AI model ('claude' or 'gemini')
        has_gemini_config (bool): Whether Gemini is properly configured
        claude_api_key (str): Claude API key
        claude_api_url (str): Claude API URL
        claude_model (str): Claude model identifier
        gemini_model: Gemini model instance (optional)
        
    Returns:
        str: Generated explanation
    """
    target_variable = training_stats.get('target_variable')
    title = training_stats.get('title')
    model_type = training_stats.get('model_type')
    accuracy = training_stats.get('accuracy')
    output = training_stats.get('output', '')
    
    prompt_text = f"""Write explanation that i can use on my web site for the following machine learning model results:
            
    Title: {title}
    Target Variable: {target_variable}
    Model Type: {model_type}
    Accuracy/R2(if model type is regression): {accuracy}
    
    Training Output:
    {output[:2000]}  # Limit content length
    
    Please provide:
    1. An explanation of what this accuracy means in business terms
    2. How good this result is compared to industry standards for this type of problem
    3. Why should i use the model and what are the business benefits?
    
    The explanation should be clear and understandable to non-technical stakeholders.
    The explanation is used in documentation.give the model explanation directly
    dont include the Title extra intro text like: "Okay, here is an explanation of the model results, designed for documentation and understandable by non-technical stakeholders" for example
    
    Format guidelines:
    - Use proper paragraphs with a single blank line between them
    - Use HTML tags like <h3>, <p>, <ul>, <li>, and <strong> for formatting
    - If you include code examples, use <pre> and <code> tags, NOT markdown triple backticks
    - Use numbered points for lists where appropriate
    - Organize your response into clear sections with headings
    
    The explanation should be clear and understandable to non-technical stakeholders.
    """
        
    # Use appropriate model based on what's available
    if active_model == "gemini" and has_gemini_config and gemini_model:
        try:
            response = gemini_model.generate_content(prompt_text)
            explanation = response.text
            explanation = re.sub(r'```\w*\n', '', explanation)  # Opening code fences with optional language
            explanation = re.sub(r'```', '', explanation)  # Closing code fences
        except Exception as e:
            logger.error(f"Error generating explanation with Gemini: {str(e)}")
            # Fall back to Claude if available
            if claude_api_key:
                logger.info("Falling back to Claude for model explanation")
                explanation = get_claude_explanation(prompt_text, claude_api_key, claude_api_url, claude_model)
            else:
                explanation = f"Error generating explanation: {str(e)}"
    else:
        # Use Claude API 
        explanation = get_claude_explanation(prompt_text, claude_api_key, claude_api_url, claude_model)
    
    return explanation

def get_claude_explanation(prompt_text, claude_api_key, claude_api_url, claude_model):
    """
    Get explanation from Claude API
    
    Args:
        prompt_text (str): Prompt to send to Claude
        claude_api_key (str): Claude API key
        claude_api_url (str): Claude API URL
        claude_model (str): Claude model identifier
        
    Returns:
        str: Generated explanation
    """
    # If Claude API key is not available, return default explanation
    if not claude_api_key:
        return "Model explanation not available. Configure an AI model API key to generate explanations."
    
    try:
        # Prepare headers for Claude API request
        headers = {
            "x-api-key": claude_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Prepare request data
        data = {
            "model": claude_model,
            "max_tokens": 2000,
            "messages": [
                {"role": "user", "content": prompt_text}
            ]
        }
        
        # Make API request
        response = requests.post(claude_api_url, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            # Extract text from Claude's response
            explanation = ""
            if result.get("content"):
                content_blocks = result["content"]
                for block in content_blocks:
                    if block.get("type") == "text":
                        explanation += block.get("text", "")
            return explanation
        else:
            return f"Error generating explanation: API Error ({response.status_code})"
    except Exception as e:
        logger.error(f"Error with Claude API for explanation: {str(e)}")
        return f"Error generating explanation: {str(e)}"

def get_embed_code(embed_id, base_url=None):
    """
    Get the HTML embed code for an embedding
    
    Args:
        embed_id (str): Embedding ID
        base_url (str, optional): Base URL for the application
        
    Returns:
        str: HTML embed code
    """
    # For absolute URL, combine base_url with relative path
    if base_url:
        embed_url = f"{base_url}/embedded/{embed_id}"
    else:
        # For relative URL
        embed_url = f"/embedded/{embed_id}"
    
    return f"""
    <iframe src="{embed_url}" width="100%" height="600px" frameborder="0"></iframe>
    """

# Main application and route handlers
app = None
ACTIVE_MODEL = None
HAS_GEMINI_CONFIG = False
CLAUDE_API_KEY = None
CLAUDE_API_URL = None
CLAUDE_MODEL = None
GEMINI_MODEL_NAME = None
gemini_model = None

def current_app():
    """Get the current application instance"""
    global app, ACTIVE_MODEL, HAS_GEMINI_CONFIG, CLAUDE_API_KEY, CLAUDE_API_URL, CLAUDE_MODEL, GEMINI_MODEL_NAME, gemini_model
    
    if app is None:
        app, ACTIVE_MODEL, HAS_GEMINI_CONFIG, CLAUDE_API_KEY, CLAUDE_API_URL, CLAUDE_MODEL, GEMINI_MODEL_NAME = create_app()
        
        # Initialize Gemini model if configured
        if ACTIVE_MODEL == 'gemini' and HAS_GEMINI_CONFIG:
            import google.generativeai as genai
            genai.configure(api_key=Config.GOOGLE_API_KEY)
            gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
    
    return app

def init_routes(app):
    """Initialize routes for the application"""
    
    
    from column_mapper import ColumnMapper
    
    # Then replace the existing upload_file route with this updated version:
    
    @app.route('/upload', methods=['GET', 'POST'])
    @login_required
    def upload_file():
        """Handle file upload and processing."""
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)
                
            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
                
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                try:
                    file.save(filepath)
                    
                    # Store file info in session for later use
                    session['uploaded_file'] = {
                        'filename': filename,
                        'filepath': filepath
                    }
                    
                    # Redirect to column validation page
                    return redirect(url_for('validate_columns'))
                    
                except Exception as e:
                    error_trace = traceback.format_exc()
                    print(f"Error processing file: {str(e)}\n{error_trace}")
                    flash(f"Error processing file: {str(e)}", 'error')
                    return render_template('error.html', error=str(e), trace=error_trace)
            else:
                flash(f"Invalid file type. Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}", 'error')
                return redirect(request.url)
        
        # For GET request, display the upload form
        return render_template('upload.html', allowed_extensions=app.config['ALLOWED_EXTENSIONS'],
                              active_model=ACTIVE_MODEL.capitalize())
         
 
    
    # Fixed implementation for db_main.py
    
    @app.route('/save-column-mappings', methods=['POST'])
    @login_required
    def save_column_mappings():
        """Save validated column mappings and proceed to business insights dashboard"""
        logger.info("save_column_mappings called")
        
        try:
            # Get form data
            filename = request.form.get('filename')
            file_path = request.form.get('file_path')
            mappings_json = request.form.get('mappings')
            
            logger.info(f"Processing mapping for file: {filename}")
            
            # Validate inputs
            if not all([filename, file_path, mappings_json]):
                return jsonify({'success': False, 'error': 'Missing required data'})
            
            # Parse mappings
            try:
                mappings = json.loads(mappings_json)
                logger.info(f"Parsed {len(mappings)} mappings")
            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {str(e)}")
                return jsonify({'success': False, 'error': f'Invalid JSON: {str(e)}'})
            
            # Get user ID
            user_id = session.get('user_id')
            if not user_id:
                return jsonify({'success': False, 'error': 'User not authenticated'}), 401
            
            # Find the file
            if not os.path.exists(file_path):
                # Try in upload folder
                upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                if os.path.exists(upload_path):
                    file_path = upload_path
                else:
                    return jsonify({'success': False, 'error': f'File not found: {filename}'})
            
            # Read the data
            try:
                df = read_data_flexible(file_path)
                if df is None or df.empty:
                    # Fallback to pandas
                    if file_path.lower().endswith('.csv'):
                        df = pd.read_csv(file_path, encoding='utf-8')
                    elif file_path.lower().endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(file_path)
                    else:
                        return jsonify({'success': False, 'error': 'Unsupported file format'})
                
                logger.info(f"Data loaded successfully: {df.shape}")
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
                return jsonify({'success': False, 'error': f'Error reading file: {str(e)}'})
            
            # Initialize enhanced column mapper and perform EDA
            eda_results = {}
            try:
                from enhanced_column_mapper import EnhancedColumnMapper
                column_mapper = EnhancedColumnMapper(gemini_model if HAS_GEMINI_CONFIG else None)
                
                # Perform EDA if not already done
                logger.info("Performing EDA analysis...")
                eda_results = column_mapper.perform_comprehensive_eda(df)
                
                # Save mappings with EDA results
                success = column_mapper.save_mappings_with_eda(user_id, filename, mappings, eda_results)
                
                if not success:
                    logger.warning("Failed to save mappings to database, but continuing...")
                
                # Apply mappings
                df_mapped = column_mapper.apply_mappings(df, mappings)
                
            except Exception as e:
                logger.warning(f"Enhanced mapper error: {str(e)}, using basic mapper")
                # Fallback to basic column mapper
                from column_mapper import ColumnMapper
                column_mapper = ColumnMapper(gemini_model if HAS_GEMINI_CONFIG else None)
                column_mapper.save_mappings(user_id, filename, mappings)
                
                # Simple column renaming
                rename_dict = {m['original_name']: m['mapped_name'] 
                              for m in mappings 
                              if m['original_name'] != m['mapped_name']}
                df_mapped = df.rename(columns=rename_dict) if rename_dict else df
            
            # Save mapped data
            try:
                base_path = file_path.rsplit('.', 1)[0]
                extension = file_path.rsplit('.', 1)[1] if '.' in file_path else 'csv'
                mapped_filepath = f"{base_path}_mapped.{extension}"
                
                if extension.lower() == 'csv':
                    df_mapped.to_csv(mapped_filepath, index=False)
                else:
                    df_mapped.to_excel(mapped_filepath, index=False)
                
                logger.info(f"Saved mapped data to: {mapped_filepath}")
                
            except Exception as e:
                logger.error(f"Error saving mapped file: {str(e)}")
                mapped_filepath = file_path
            
            # Store important data in session
            session['last_uploaded_file'] = filename
            session['mapped_file_path'] = mapped_filepath
            session['column_mappings'] = mappings
            session['original_file_path'] = file_path
            
            # Store EDA results summary if available
            if eda_results:
                session['eda_results'] = eda_results  # Store full EDA results
                session['has_eda'] = True
            
            # Identify target columns
            target_columns = [m for m in mappings if m.get('is_target')]
            if target_columns:
                session['target_variable'] = target_columns[0]['mapped_name']
            
            logger.info("Successfully processed column mappings, redirecting to business insights")
            
            # Return success - redirect to business insights WITHOUT filename in URL
            return jsonify({
                'success': True,
                'redirect': url_for('business_insights')  # No filename parameter
            })
            
        except Exception as e:
            logger.error(f"Unexpected error in save_column_mappings: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f'An unexpected error occurred: {str(e)}'
            })
    
    
  
    
    # Add a separate API endpoint for getting more data if needed
    @app.route('/api/business-insights-data')
    @login_required
    def get_business_insights_data():
        """API endpoint to get full dataset for business insights"""
        
        filename = session.get('last_uploaded_file')
        if not filename:
            return jsonify({'error': 'No data file found'}), 404
        
        file_path = session.get('mapped_file_path') or session.get('original_file_path')
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Data file not found'}), 404
        
        try:
            # Read data
            if file_path.lower().endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                df = pd.read_excel(file_path)
            
            # Get pagination parameters
            page = request.args.get('page', 1, type=int)
            per_page = request.args.get('per_page', 1000, type=int)
            
            # Calculate pagination
            start = (page - 1) * per_page
            end = start + per_page
            
            # Get data slice
            data_slice = df.iloc[start:end].to_dict('records')
            
            # Clean numpy types
            from enhanced_column_mapper import convert_numpy_types
            data_slice = convert_numpy_types(data_slice)
            
            return jsonify({
                'data': data_slice,
                'page': page,
                'per_page': per_page,
                'total_rows': len(df),
                'total_pages': (len(df) + per_page - 1) // per_page
            })
            
        except Exception as e:
            logger.error(f"Error in API: {str(e)}")
            return jsonify({'error': str(e)}), 500



    
    @app.route('/validate-columns')
    @login_required
    def validate_columns():
        """Display column validation and mapping interface with comprehensive EDA"""
        # Get uploaded file info from session
        file_info = session.get('uploaded_file')
        if not file_info:
            flash('No file uploaded. Please upload a file first.', 'error')
            return redirect(url_for('upload_file'))
        
        filename = file_info['filename']
        filepath = file_info['filepath']
        
        try:
            # Read the data
            df = read_data_flexible(filepath)
            
            # Initialize enhanced column mapper
            column_mapper = EnhancedColumnMapper(gemini_model if HAS_GEMINI_CONFIG else None)
            
            # Check if we have saved mappings for this file
            user_id = session.get('user_id')
            saved_mappings, saved_eda = column_mapper.get_saved_mappings_with_eda(user_id, filename)
            
            if saved_mappings and saved_eda:
                # Use saved mappings and EDA
                analysis = {
                    'columns': saved_mappings,
                    'dataset_summary': f'Previously analyzed dataset: {filename}',
                    'suggested_use_cases': [],
                    'eda_results': saved_eda
                }
            else:
                # Perform comprehensive analysis with EDA
                analysis = column_mapper.analyze_columns_with_eda(df, filename)
            
            # Prepare column data for template with enhanced target information
            columns = []
            for col_info in analysis.get('columns', []):
                # Get sample data for this column
                sample_data = df[col_info['original_name']].dropna().head(5).tolist() if col_info['original_name'] in df.columns else []
                
                column_data = {
                    'original_name': col_info['original_name'],
                    'suggested_name': col_info.get('suggested_name', col_info['original_name']),
                    'data_type': col_info.get('data_type', 'text'),
                    'description': col_info.get('description', ''),
                    'is_target': col_info.get('is_target', False),
                    'target_meaning': col_info.get('target_meaning', ''),
                    'confidence': col_info.get('confidence', 0.5),
                    'sample_data': sample_data
                }
                
                # Add target-specific fields if this is identified as a target
                if column_data['is_target']:
                    column_data['prediction_use_cases'] = col_info.get('prediction_use_cases', [])
                    column_data['business_value'] = col_info.get('business_value', '')
                    column_data['model_type'] = col_info.get('model_type', 'classification')
                
                columns.append(column_data)
            
            # Extract EDA results for template
            eda_results = analysis.get('eda_results', {})
            
            # Convert numpy types to JSON-serializable types
            eda_results = make_json_serializable(eda_results)
            data_quality = make_json_serializable(eda_results.get('data_quality', {}))
            correlations = make_json_serializable(eda_results.get('correlations', {}))
            business_insights = make_json_serializable(eda_results.get('business_insights', []))
            
            return render_template('enhanced_column_validation.html',
                                 filename=filename,
                                 file_path=filepath,
                                 total_rows=len(df),
                                 total_columns=len(df.columns),
                                 columns=columns,
                                 dataset_summary=analysis.get('dataset_summary', ''),
                                 suggested_use_cases=analysis.get('suggested_use_cases', []),
                                 eda_results=eda_results,
                                 data_quality=data_quality,
                                 correlations=correlations,
                                 business_insights=business_insights,
                                 saved_templates=[])
            
        except Exception as e:
            error_trace = traceback.format_exc()
            print(f"Error in column validation: {str(e)}\n{error_trace}")
            flash(f"Error analyzing columns: {str(e)}", 'error')
            return redirect(url_for('upload_file'))
                
        
        

    @app.route('/results')
    @login_required
    def show_results():
        """Show AI use case proposals after column validation"""
        filename = session.get('last_filename')
        proposals = session.get('proposals', [])
        target_variable = session.get('target_variable')
        
        if not proposals:
            flash('No proposals found. Please upload a file first.', 'error')
            return redirect(url_for('upload_file'))
        
        return render_template('results.html',
                             filename=filename,
                             proposals=proposals,
                             target_variable=target_variable)
        

        """Display comprehensive business insights screen"""
        
        # Get data from session
        filename = session.get('last_filename')
        eda_results = session.get('eda_results', {})
        column_mappings = session.get('column_mappings', [])
        
        if not filename:
            flash('No analysis data found. Please upload and analyze a file first.', 'error')
            return redirect(url_for('upload_file'))
        
        # Get target columns from mappings
        target_columns = [m for m in column_mappings if m.get('is_target')]
        
        # Extract data for template
        data_quality = eda_results.get('data_quality', {})
        correlations = eda_results.get('correlations', {})
        business_insights = eda_results.get('business_insights', [])
        
        # Get basic stats if available
        basic_stats = eda_results.get('basic_stats', {})
        total_rows = basic_stats.get('n_rows', 0)
        total_columns = basic_stats.get('n_columns', 0)
        
        return render_template('business_insights.html',
                             filename=filename,
                             total_rows=total_rows,
                             total_columns=total_columns,
                             data_quality=data_quality,
                             correlations=correlations,
                             business_insights=business_insights,
                             target_columns=target_columns)
        """Handle file upload and processing."""
        if request.method == 'POST':
            if 'file' not in request.files:
                flash('No file part', 'error')
                return redirect(request.url)
                
            file = request.files['file']
            if file.filename == '':
                flash('No selected file', 'error')
                return redirect(request.url)
                
            if file and allowed_file(file.filename):
                try:
                    filename = secure_filename(file.filename)
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    
                    file.save(filepath)
                    
                    # Try to determine the file encoding
                    file_content = None
                    
                    for encoding in encodings:
                        try:
                            with open(filepath, 'r', encoding=encoding, errors='strict') as f:
                                # Read and limit to 100 lines
                                lines = []
                                for i, line in enumerate(f):
                                    if i >= 100:  # Only read first 100 lines
                                        break
                                    lines.append(line)
                                file_content = ''.join(lines)
                                break
                        except UnicodeDecodeError:
                            continue
                    
                    if file_content is None:
                        # If all encodings fail, use 'ignore' to replace problematic characters
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            # Read and limit to 100 lines
                            lines = []
                            for i, line in enumerate(f):
                                if i >= 100:  # Only read first 100 lines
                                    break
                                lines.append(line)
                            file_content = ''.join(lines)
                    
                    # Limit file content length to prevent very large API requests
                    if len(file_content) > 100000:
                        file_content = file_content[:100000] + "\n\n[Content truncated due to size limitations]"
                    
                    # Get AI use case proposals using the active model
                    try:
                        ai_proposals = get_ai_use_case_proposals(
                            file_content, filename, 
                            ACTIVE_MODEL, HAS_GEMINI_CONFIG, 
                            CLAUDE_API_KEY, CLAUDE_API_URL, CLAUDE_MODEL,
                            gemini_model
                        )
                        
                        # Save use cases for the current user
                        if 'user_id' in session:
                           save_use_case_id = save_use_cases(
                               session['user_id'], 
                               filename, 
                               ai_proposals, 
                               metadata={
                                   'file_path': filepath,
                                   'proposal_count': len(ai_proposals)
                               }
                           )
                           session['current_use_case_id'] = save_use_case_id
                        
                        # Extract target variable from the first proposal if available
                        target_variable = None
                        if ai_proposals and len(ai_proposals) > 0:
                            target_variable = ai_proposals[0].get('target_variable', None)
                            
                        # Store the filename in session for later use in training
                        session['last_filename'] = filename
                        session['proposal_count'] = len(ai_proposals)
                        session['file_path'] = filepath
                        session['target_variable'] = target_variable
                        session['proposals'] = ai_proposals  # Store proposals for model training
                        
                        return render_template('results.html', 
                                               filename=filename, 
                                               proposals=ai_proposals,
                                               target_variable=target_variable)
                        
                    except Exception as api_error:
                        error_trace = traceback.format_exc()
                        logger.error(f"API Error: {str(api_error)}\n{error_trace}")
                        flash(f"Error generating AI use cases: {str(api_error)}", 'error')
                        return render_template('error.html', error=str(api_error), trace=error_trace)
                    
                except Exception as e:
                    error_trace = traceback.format_exc()
                    logger.error(f"Error processing file: {str(e)}\n{error_trace}")
                    flash(f"Error processing file: {str(e)}", 'error')
                    return render_template('error.html', error=str(e), trace=error_trace)
            else:
                flash(f"Invalid file type. Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}", 'error')
                return redirect(request.url)
        
        # For GET request, display the upload form
        return render_template('upload.html', 
                             allowed_extensions=app.config['ALLOWED_EXTENSIONS'],
                             active_model=ACTIVE_MODEL.capitalize())
    
    @app.route('/training_results', methods=['GET'])
    @login_required
    def training_results():
        """Show the training results after completion"""
        try:
            # Get training results from session
            training_results = session.get('training_results')
            
            if not training_results:
                # If no results in session, redirect to upload
                flash("No training results found. Please upload a file and train a model.", 'warning')
                return redirect(url_for('upload_file'))
            
            # Get feature importance directly from training_results
            feature_importance = training_results.get('feature_importance', {})
            
            # Convert NumPy types to Python native types
            if feature_importance:
                import numpy as np
                converted_feature_importance = {}
                for key, value in feature_importance.items():
                    # Convert NumPy float32/float64 to Python float
                    if isinstance(value, (np.float32, np.float64, np.float16, np.float_)):
                        converted_feature_importance[key] = float(value)
                    # Convert NumPy int types to Python int
                    elif isinstance(value, (np.int32, np.int64, np.int16, np.int8, np.int_, np.intc, np.intp)):
                        converted_feature_importance[key] = int(value)
                    # Convert NumPy bool to Python bool
                    elif isinstance(value, np.bool_):
                        converted_feature_importance[key] = bool(value)
                    # Convert NumPy arrays to lists
                    elif isinstance(value, np.ndarray):
                        converted_feature_importance[key] = value.tolist()
                    else:
                        converted_feature_importance[key] = value
                
                feature_importance = converted_feature_importance
            
            # Setup template context
            context = {
            'success': training_results.get('success', False),
            'accuracy': training_results.get('accuracy'),
            'output': training_results.get('output', ''),
            'error_message': training_results.get('error_message'),
            'error_trace': training_results.get('error_trace'),
            'model_type': training_results.get('model_type', 'unknown'),
            'explanation': training_results.get('explanation', ''),
            'proposal': training_results.get('proposal', {}),
            'script_path': training_results.get('script_path', ''),
            'feature_importance': feature_importance,
            'llm_explanation': training_results.get('llm_explanation'),
            'training_results': {
                'title': training_results.get('title'),
                'description': training_results.get('description'),
                'target_variable': training_results.get('target_variable'),
                'business_value': training_results.get('business_value'),
                'prediction_interpretation': training_results.get('prediction_interpretation'),
                'kpis': training_results.get('kpis', []),
                # Add this structure to match the template's expectations
                'metrics': {
                    'accuracy': training_results.get('accuracy'),
                    'precision': training_results.get('precision', 'N/A'),
                    'recall': training_results.get('recall', 'N/A'),
                    'f1_score': training_results.get('f1_score', 'N/A'),
                    'roc_auc': training_results.get('roc_auc', 'N/A')
                },
                'features': {
                    'importance': feature_importance
                }
            }
        }
        
            
            return render_template('training_results.html', **context)
        
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error in training_results: {str(e)}\n{error_trace}")
            flash(f"Error showing training results: {str(e)}", 'error')
            return render_template('error.html', error=str(e), trace=error_trace)
    
    @app.route('/error', methods=['GET'])
    def error_page():
        """Display error page with custom error message"""
        error = request.args.get('error', 'An unknown error occurred')
        trace = request.args.get('trace', '')
        return render_template('error.html', error=error, trace=trace)
    
    # Cleanup old files periodically
    @app.before_request
    def cleanup_old_files():
        """Clean up old upload files to prevent disk space issues"""
        # Only run occasionally to avoid overhead
        if random.random() < 0.05:  # 5% chance on each request
            try:
                now = time.time()
                for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    # Remove files older than 24 hours
                    if os.path.isfile(filepath) and now - os.path.getmtime(filepath) > 86400:
                        os.remove(filepath)
            except Exception as e:
                logger.error(f"Error cleaning up files: {str(e)}")
    
        

        """Verify and update the database schema if needed"""
        logger.info("Verifying database schema...")
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Check if the models table exists
                    cursor.execute("""
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = 'models'
                        );
                    """)
                    table_exists = cursor.fetchone()[0]
                    
                    if not table_exists:
                        logger.info("Models table does not exist. Creating it...")
                        cursor.execute("""
                            CREATE TABLE models (
                                id VARCHAR(255) PRIMARY KEY,
                                user_id VARCHAR(255) NOT NULL,
                                name VARCHAR(255) NOT NULL,
                                model_data BYTEA,
                                metadata JSONB,
                                created_at TIMESTAMP NOT NULL,
                                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                            )
                        """)
                        conn.commit()
                        logger.info("Models table created successfully")
                    else:
                        # Check if model_data column exists
                        cursor.execute("""
                            SELECT EXISTS (
                                SELECT FROM information_schema.columns 
                                WHERE table_name = 'models' AND column_name = 'model_data'
                            );
                        """)
                        column_exists = cursor.fetchone()[0]
                        
                        if not column_exists:
                            logger.info("model_data column does not exist. Adding it...")
                            cursor.execute("""
                                ALTER TABLE models ADD COLUMN model_data BYTEA;
                            """)
                            conn.commit()
                            logger.info("model_data column added successfully")
                    
                    logger.info("Database schema verification complete")
        except Exception as e:
            logger.error(f"Error verifying database schema: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
    
    def clean_metadata_for_json(metadata):
        """
        Clean metadata to ensure all values are JSON serializable
        
        Args:
            metadata (dict): Original metadata dictionary
            
        Returns:
            dict: Cleaned metadata with JSON-serializable values
        """
        import numpy as np
        
        def clean_value(value):
            if value is None:
                return None
            elif isinstance(value, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                                   np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(value)
            elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                return float(value)
            elif isinstance(value, np.bool_):
                return bool(value)
            elif isinstance(value, np.ndarray):
                return value.tolist()
            elif isinstance(value, dict):
                return {k: clean_value(v) for k, v in value.items()}
            elif isinstance(value, list):
                return [clean_value(item) for item in value]
            elif hasattr(value, 'tolist'):
                return value.tolist()
            elif hasattr(value, 'isoformat'):
                return value.isoformat()
            else:
                return value
        
        return {k: clean_value(v) for k, v in metadata.items()}
    

        
        
        
        """Start the model training process and save model to database"""
        try:
            # Get user information
            user_id = session.get('user_id')
            if not user_id:
                return jsonify({'error': 'User not authenticated'}), 401
                
            # Get information from the session
            filename = session.get('last_filename', 'Unknown file')
            proposal_index = int(request.form.get('proposal_index', 0))
            file_path = session.get('file_path')
            proposals = session.get('proposals', [])
            
            # Get the target variable and model type from the selected proposal
            target_variable = None
            claude_model_type = "auto"
            selected_proposal = None
            
            if proposals and len(proposals) > proposal_index:
                selected_proposal = proposals[proposal_index]
                target_variable = selected_proposal.get('target_variable')
                model_type = selected_proposal.get('model_type', "auto")
                
                # If target variable is "Unknown", set to None for auto-detection
                if target_variable == "Unknown":
                    target_variable = None
            
            # Call the training function from the ml module
            training_stats = ml_trainer.train_model_with_robust_error_handling(
                file_path, 
                model_type, 
                proposal_index, 
                target_variable,
                user_id  # Pass the user_id explicitly
            )
                        
            # Add proposal details to training stats
            if selected_proposal:
                training_stats['title'] = selected_proposal.get('title')
                training_stats['description'] = selected_proposal.get('description')
                training_stats['kpis'] = selected_proposal.get('kpis')
                training_stats['business_value'] = selected_proposal.get('business_value')
                training_stats['prediction_interpretation'] = selected_proposal.get('prediction_interpretation')
        
            # Generate explanation using LLM
            llm_explanation = generate_model_explanation(
                training_stats, 
                ACTIVE_MODEL, HAS_GEMINI_CONFIG, 
                CLAUDE_API_KEY, CLAUDE_API_URL, CLAUDE_MODEL,
                gemini_model
            )
            training_stats['llm_explanation'] = llm_explanation
            
            # Check if training was successful
            if not training_stats.get('success', False):
                return jsonify({
                    'error': training_stats.get('error_message', 'Training failed'),
                    'redirect': url_for('error_page', 
                                       error=training_stats.get('error_message', 'Training failed'),
                                       trace=training_stats.get('error_trace', ''))
                }), 400
            
            # Store training results in session
            session['training_results'] = training_stats
            
            # Find and save the model to database
            target_variable = training_stats.get('target_variable')
            logger.info(f"Looking for model file for target: {target_variable}")
            
            # Search for the model in the databases directory
            model_path = None
            model_files = []
            
            # Walk through the database directory to find the model file
            for root, dirs, files in os.walk('databases'):
                for file in files:
                    if file.endswith('.joblib') and target_variable in file and not file.startswith('preprocessor_') and not file.startswith('model_features_'):
                        model_path = os.path.join(root, file)
                        model_files.append(model_path)
            
            # If multiple files were found, use the most recently created one
            if len(model_files) > 1:
                model_files.sort(key=os.path.getmtime, reverse=True)
                model_path = model_files[0]
            
            logger.info(f"Found model file at: {model_path}")
            
            if model_path and os.path.exists(model_path):
                # Load the model
                import joblib
                model_object = joblib.load(model_path)
                logger.info(f"Loaded model from: {model_path}")
                
                # Also look for the preprocessor file
                preprocessor_path = model_path.replace('.joblib', '').replace(target_variable, f'preprocessor_{target_variable}')
                if not os.path.exists(preprocessor_path):
                    preprocessor_path = model_path.replace('.joblib', '_preprocessor.joblib')
                
                preprocessor = None
                if os.path.exists(preprocessor_path):
                    try:
                        preprocessor = joblib.load(preprocessor_path)
                        logger.info(f"Loaded preprocessor from: {preprocessor_path}")
                    except Exception as e:
                        logger.error(f"Error loading preprocessor: {str(e)}")
                
                # Create metadata for the database
                model_name = training_stats.get('title', f"Model for {target_variable}")
                
                # Get feature names
                feature_names = []
                if 'feature_names' in training_stats:
                    feature_names = training_stats['feature_names']
                elif hasattr(model_object, 'feature_names_in_'):
                    feature_names = model_object.feature_names_in_.tolist()
                
                # Create complete metadata
                metadata = {
                    'target_variable': training_stats.get('target_variable'),
                    'model_type': training_stats.get('model_type'),
                    'accuracy': training_stats.get('accuracy'),
                    'title': model_name,
                    'description': training_stats.get('description', ''),
                    'kpis': training_stats.get('kpis', []),
                    'business_value': training_stats.get('business_value', ''),
                    'prediction_interpretation': training_stats.get('prediction_interpretation', ''),
                    'llm_explanation': training_stats.get('llm_explanation', ''),
                    'source_file': filename,
                    'original_model_path': model_path,
                    'metrics': {
                        'accuracy': training_stats.get('accuracy'),
                        'precision': training_stats.get('precision', None),
                        'recall': training_stats.get('recall', None),
                        'f1_score': training_stats.get('f1_score', None)
                    },
                    'features': {
                        'trained_features': feature_names,
                        'importance': training_stats.get('feature_importance', {})
                    }
                }
                
                # Clean metadata for JSON serialization
                metadata = clean_metadata_for_json(metadata)
                
                # Generate a unique ID for the model
                model_id = str(uuid.uuid4())
                
                # Serialize the model
                import io
                buffer = io.BytesIO()
                joblib.dump(model_object, buffer)
                serialized_model = buffer.getvalue()
                
                # Save to database directly
                try:
                    with get_db_connection() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute(
                                """
                                INSERT INTO models (id, user_id, name, model_data, metadata, created_at)
                                VALUES (%s, %s, %s, %s, %s, %s)
                                RETURNING id
                                """,
                                (model_id, user_id, model_name, serialized_model, json.dumps(metadata), datetime.now())
                            )
                            result = cursor.fetchone()
                            conn.commit()
                    
                    logger.info(f"Model saved to database with ID: {model_id}")
                    
                    # Store the model ID in session
                    session['current_model_id'] = model_id
                    
                    # Delete the file from disk if successful
                    try:
                        os.remove(model_path)
                        logger.info(f"Removed model file from disk: {model_path}")
                        
                        # Also remove preprocessor if it exists
                        if preprocessor_path and os.path.exists(preprocessor_path):
                            os.remove(preprocessor_path)
                            logger.info(f"Removed preprocessor file from disk: {preprocessor_path}")
                    except Exception as e:
                        logger.warning(f"Could not remove files: {str(e)}")
                    
                except Exception as db_error:
                    logger.error(f"Database error saving model: {str(db_error)}")
                    import traceback
                    logger.error(traceback.format_exc())
            else:
                logger.error(f"Model file not found for target: {target_variable}")
            
            return jsonify({
                'success': True,
                'redirect': url_for('training_results')
            })
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error in train_model: {str(e)}\n{error_trace}")
            return jsonify({
                'success': False,
                'error': str(e),
                'redirect': url_for('error_page', error=str(e), trace=error_trace)
            }), 500

    @app.route('/my-models')
    @login_required
    def my_models():
        """Display the user's saved models."""
        user_id = session.get('user_id')
        if not user_id:
            flash('Please login to view your models', 'error')
            return redirect(url_for('login'))
        
        # Get user's models
        user_models_list = get_user_models(user_id)
        
        # Get user's embeddings
        user_embeddings_list = get_user_embeddings(user_id)
        
        return render_template('my_models.html', 
                              models=user_models_list,
                              embeddings=user_embeddings_list,
                              user_name=session.get('user_name', 'User'))
    
   
    @app.route('/delete-model/<model_id>', methods=['POST'])
    @login_required
    def delete_model_route(model_id):
        """Delete a model from the user's account."""
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        # Delete the model
        try:
            success = delete_user_model(user_id, model_id)
            
            if not success:
                return jsonify({'success': False, 'error': 'Failed to delete model or model not found'}), 404
            
            return jsonify({
                'success': True, 
                'message': 'Model deleted successfully'
            })
        
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error deleting model: {str(e)}\n{error_trace}")
            return jsonify({'success': False, 'error': str(e)}), 500
        
    @app.route('/load-model/<model_id>')
    @login_required
    def load_model(model_id):
        """Load a saved model for testing or further use."""
        import io
        import joblib
        
        user_id = session.get('user_id')
        if not user_id:
            flash('Please login to load models', 'error')
            return redirect(url_for('login'))
        
        # Get the model data
        model_data = get_model_by_id(model_id)
        
        if not model_data:
            flash('Model not found', 'error')
            return redirect(url_for('my_models'))
        
        # Check if user owns this model
        if model_data['user_id'] != user_id:
            flash('You do not have permission to access this model', 'error')
            return redirect(url_for('my_models'))
        
        # Check model availability - either from file or binary data
        has_model = False
        model_object = None
        
        # Case 1: Model exists as a file
        if model_data.get('model_path') and os.path.exists(model_data['model_path']):
            has_model = True
            session['current_model_path'] = model_data['model_path']
        # Case 2: Model exists as binary data in the database
        elif model_data.get('model_data'):
            has_model = True
            # We'll need to deserialize the model and possibly save it temporarily
            try:
                buffer = io.BytesIO(model_data['model_data'])
                model_object = joblib.load(buffer)
                
                # Optionally, save the model to a temporary file for compatibility
                temp_dir = os.path.join(os.getcwd(), "temp_models")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"{model_id}.joblib")
                joblib.dump(model_object, temp_path)
                
                session['current_model_path'] = temp_path
                model_data['model_path'] = temp_path  # Update for consistency
            except Exception as e:
                logger.error(f"Error deserializing model: {str(e)}")
                flash('Error loading model from database', 'error')
                return redirect(url_for('my_models'))
        
        if not has_model:
            flash('Model data not found', 'error')
            return redirect(url_for('my_models'))
        
        # Extract metadata and set session variables
        metadata = model_data.get('metadata', {})
        
        # Set feature names in session
        if 'features' in metadata and 'trained_features' in metadata['features']:
            session['feature_names'] = metadata['features']['trained_features']
        else:
            session['feature_names'] = []
        
        # Set model type
        session['model_type'] = metadata.get('model_type', 'unknown')
        
        # Store model ID
        session['current_model_id'] = model_id
        
        # Create a simplified training_results for consistency with the rest of the app
        training_results = {
            'success': True,
            'accuracy': metadata.get('accuracy'),
            'target_variable': metadata.get('target_variable'),
            'model_type': metadata.get('model_type', 'unknown'),
            'title': metadata.get('title', model_data.get('name', 'Model')),
            'description': metadata.get('description', ''),
            'feature_importance': metadata.get('features', {}).get('importance', {}),
            'model_path': model_data.get('model_path')
        }
        
        session['training_results'] = training_results
        
        # Redirect to model tester
        flash(f'Model "{model_data.get("name", "Model")}" loaded successfully', 'success')
        return redirect(url_for('model_tester'))
    
    @app.route('/create_embedding/<model_id>', methods=['POST'])
    @login_required
    def create_embedding_route(model_id):
        """Create an embeddable version of a model."""
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        # Get model data
        model_data = get_model_by_id(model_id)
        
        if not model_data:
            return jsonify({'success': False, 'error': 'Model not found'}), 404
        
        # Check if user owns this model
        if model_data['user_id'] != user_id:
            return jsonify({'success': False, 'error': 'You do not have permission to access this model'}), 403
        
        # Get embedding settings from form
        embed_name = request.form.get('embed_name', model_data['name'] + ' Embedding')
        embed_settings = {
            'show_confidence': request.form.get('show_confidence', 'on') == 'on',
            'allow_file_upload': request.form.get('allow_file_upload', 'on') == 'on',
            'custom_theme': request.form.get('custom_theme', 'default')
        }
        
        # Create the embedding
        try:
            embed_id = create_model_embedding(model_id, embed_name, embed_settings)
            
            if not embed_id:
                return jsonify({'success': False, 'error': 'Failed to create embedding'}), 500
            
            # Get embed code (include base URL for absolute path)
            request_base_url = request.url_root.rstrip('/')
            embed_code = get_embed_code(embed_id, request_base_url)
            
            return jsonify({
                'success': True, 
                'message': 'Embedding created successfully',
                'embed_id': embed_id,
                'embed_code': embed_code,
                'embed_url': url_for('view_embedding', embed_id=embed_id, _external=True)
            })
        
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error creating embedding: {str(e)}\n{error_trace}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/delete-embedding/<embed_id>', methods=['POST'])
    @login_required
    def delete_embedding_route(embed_id):
        """Delete an embedding."""
        user_id = session.get('user_id')
        if not user_id:
            return jsonify({'success': False, 'error': 'Not logged in'}), 401
        
        # Get embedding data
        embed_data = get_embedding_by_id(embed_id)
        
        if not embed_data:
            return jsonify({'success': False, 'error': 'Embedding not found'}), 404
        
        # Check if user owns this embedding
        if embed_data['user_id'] != user_id:
            return jsonify({'success': False, 'error': 'You do not have permission to delete this embedding'}), 403
        
        # Delete the embedding
        try:
            success = delete_embedding(embed_id)
            
            if not success:
                return jsonify({'success': False, 'error': 'Failed to delete embedding'}), 500
            
            return jsonify({
                'success': True, 
                'message': 'Embedding deleted successfully'
            })
        
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error deleting embedding: {str(e)}\n{error_trace}")
            return jsonify({'success': False, 'error': str(e)}), 500
    
    @app.route('/embedded/<embed_id>')
    def view_embedding(embed_id):
        """View an embedded model."""
        # Get embedding data
        embed_data = get_embedding_by_id(embed_id)
        
        if not embed_data:
            return render_template('error.html', error="Embedding not found", trace="The requested embedding does not exist or has been removed.")
        
        # Check if embedding HTML file exists
        embed_html_path = os.path.join(embed_data['embed_path'], "embed.html")
        
        if not os.path.exists(embed_html_path):
            return render_template('error.html', error="Embedding file not found", trace="The embedding HTML file could not be located.")
        
        # Serve the embedding HTML file
        try:
            with open(embed_html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            return html_content
        
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error serving embedding: {str(e)}\n{error_trace}")
            return render_template('error.html', error=f"Error serving embedding: {str(e)}", trace=error_trace)
    
    @app.route('/api/predict/<embed_id>', methods=['POST'])
    def api_predict(embed_id):
        """API endpoint for making predictions with an embedded model."""
        # Get embedding data
        embed_data = get_embedding_by_id(embed_id)
        
        if not embed_data:
            return jsonify({'error': 'Embedding not found'}), 404
        
        # Get model path from embedding data
        model_path = os.path.join(embed_data['embed_path'], "model.joblib")
        
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404
        
        # Get features from embedding metadata or from the features.json file
        features_path = os.path.join(embed_data['embed_path'], "features.json")
        if os.path.exists(features_path):
            try:
                with open(features_path, 'r') as f:
                    features = json.load(f)
            except Exception:
                features = embed_data['metadata'].get('features', [])
        else:
            features = embed_data['metadata'].get('features', [])
        
        # Get model type from metadata
        model_type = embed_data['metadata'].get('model_type', 'unknown')
        
        # Parse input data from request
        try:
            input_data = request.json
            
            # Convert input data to a DataFrame
            input_values = {}
            
            for feature in features:
                feature_key = f"feature_{feature}"
                if feature_key in input_data:
                    # Try to convert to numeric if possible
                    try:
                        value = float(input_data[feature_key])
                        # Check if it's actually an integer
                        if value.is_integer():
                            value = int(value)
                    except ValueError:
                        value = input_data[feature_key]
                    
                    input_values[feature] = value
                else:
                    return jsonify({'error': f'Missing required feature: {feature}'}), 400
            
            # Create a DataFrame with the input values
            input_df = pd.DataFrame([input_values])
            
            # Load the model
            import joblib
            model = joblib.load(model_path)
            
            # Make prediction
            if model_type == 'classification':
                # For classification, use predict method for the class
                prediction = model.predict(input_df)[0]
                
                # Get prediction probability if the model supports it
                probability = None
                if hasattr(model, 'predict_proba'):
                    try:
                        proba = model.predict_proba(input_df)[0]
                        # Get the probability of the predicted class
                        max_proba_index = proba.argmax()
                        probability = float(proba[max_proba_index])
                        probability_str = f"{probability * 100:.2f}%"
                    except Exception as e:
                        logger.error(f"Error getting prediction probability: {str(e)}")
                        probability_str = None
                
                return jsonify({
                    'prediction': str(prediction),
                    'probability': probability_str
                })
            
            elif model_type == 'regression':
                # For regression, use predict method for the value
                prediction = float(model.predict(input_df)[0])
                
                return jsonify({
                    'prediction': f"{prediction:.4f}"
                })
            
            else:
                return jsonify({'error': f'Unsupported model type: {model_type}'}), 400
        
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error making prediction: {str(e)}\n{error_trace}")
            return jsonify({'error': str(e)}), 500
        
    @app.route('/model_tester', methods=['GET'])
    @login_required
    def model_tester():
        """Display the model testing interface with the currently trained model."""
        user_id = session.get('user_id')
        if not user_id:
            flash('Please login to view your models', 'error')
            return redirect(url_for('login'))
            
        # Get model ID from session or request
        model_id = request.args.get('model_id') or session.get('current_model_id')
        
        # Load model info from database if we have an ID
        model_data = None
        if model_id:
            try:
                with get_db_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            """
                            SELECT id, name, metadata, created_at
                            FROM models 
                            WHERE id = %s AND user_id = %s
                            """, 
                            (model_id, user_id)
                        )
                        model_data = cursor.fetchone()
            except Exception as e:
                logger.error(f"Error retrieving model data: {str(e)}")
        
        # If we found model data, extract metadata
        if model_data:
            metadata = model_data.get('metadata', {})
            model_type = metadata.get('model_type', 'unknown')
            target_variable = metadata.get('target_variable', 'Unknown')
            accuracy = metadata.get('accuracy', 'N/A')
            feature_names = metadata.get('features', {}).get('trained_features', [])
            display_model_name = metadata.get('title', model_data.get('name', 'Model'))
            
            # Format accuracy for display
            if isinstance(accuracy, float):
                accuracy = f"{accuracy * 100:.2f}%" if accuracy < 1 else f"{accuracy:.2f}%"
            
            # Store model info in session
            session['current_model_id'] = model_id
            session['model_type'] = model_type
            session['feature_names'] = feature_names
            
            # Create training_results for context
            training_results = {
                'success': True,
                'target_variable': target_variable,
                'model_type': model_type,
                'accuracy': accuracy,
                'title': display_model_name,
                'description': metadata.get('description', ''),
                'kpis': metadata.get('kpis', []),
                'business_value': metadata.get('business_value', ''),
                'prediction_interpretation': metadata.get('prediction_interpretation', ''),
                'llm_explanation': metadata.get('llm_explanation', '')
            }
            session['training_results'] = training_results
            
            return render_template('model_tester.html',
                                 model_id=model_id,
                                 model_name=display_model_name,
                                 target_variable=target_variable,
                                 model_type=model_type,
                                 accuracy=str(accuracy),
                                 required_features=feature_names,
                                 prediction_results=None,
                                 training_results=training_results)
        
        # If no model data found, check session
        training_results = session.get('training_results')
        if not training_results:
            flash('No model found. Please train a model first.', 'error')
            return redirect(url_for('upload_file'))
        
        # Fallback to session data if no database model found
        target_variable = training_results.get('target_variable', 'Unknown')
        model_type = training_results.get('model_type', 'unknown')
        accuracy = training_results.get('accuracy', 'N/A')
        display_model_name = training_results.get('title', 'AI Model')
        
        # Format accuracy if needed
        if isinstance(accuracy, float):
            accuracy = f"{accuracy * 100:.2f}%" if accuracy < 1 else f"{accuracy:.2f}%"
        
        # Get feature names
        feature_names = session.get('feature_names', [])
        if not feature_names:
            feature_names = ["feature1", "feature2"]
            flash("No feature information available. Using placeholders.", 'warning')
        
        return render_template('model_tester.html',
                             model_id=model_id,
                             model_name=display_model_name,
                             target_variable=target_variable,
                             model_type=model_type,
                             accuracy=str(accuracy),
                             required_features=feature_names,
                             prediction_results=None,
                             training_results=training_results)

    
    @app.route('/train_model/<model_type>', methods=['POST'])
    @login_required
    def train_model(model_type):
        """Start the model training process and save model to database"""
        try:
            # Get user information
            user_id = session.get('user_id')
            if not user_id:
                return jsonify({'error': 'User not authenticated'}), 401
                    
            # Get information from the session
            filename = session.get('last_filename', 'Unknown file')
            proposal_index = int(request.form.get('proposal_index', 0))
            file_path = session.get('file_path')
            proposals = session.get('proposals', [])
            
            # Get the target variable and model type from the selected proposal
            target_variable = None
            claude_model_type = "auto"
            selected_proposal = None
            
            if proposals and len(proposals) > proposal_index:
                selected_proposal = proposals[proposal_index]
                target_variable = selected_proposal.get('target_variable')
                claude_model_type = selected_proposal.get('model_type', "auto")
                
                # If target variable is "Unknown", set to None for auto-detection
                if target_variable == "Unknown":
                    target_variable = None
            
            # Call the training function from the ml module
            training_stats = ml_trainer.train_model_with_robust_error_handling(
                file_path, 
                model_type, 
                proposal_index, 
                target_variable
            )
            
            # Add proposal details to training stats
            if selected_proposal:
                training_stats['title'] = selected_proposal.get('title')
                training_stats['description'] = selected_proposal.get('description')
                training_stats['kpis'] = selected_proposal.get('kpis')
                training_stats['business_value'] = selected_proposal.get('business_value')
                training_stats['prediction_interpretation'] = selected_proposal.get('prediction_interpretation')
            
            # Generate explanation using LLM
            llm_explanation = generate_model_explanation(
                training_stats, 
                ACTIVE_MODEL, HAS_GEMINI_CONFIG, 
                CLAUDE_API_KEY, CLAUDE_API_URL, CLAUDE_MODEL,
                gemini_model
            )
            training_stats['llm_explanation'] = llm_explanation
            
            # Check if training was successful
            if not training_stats.get('success', False):
                return jsonify({
                    'error': training_stats.get('error_message', 'Training failed'),
                    'redirect': url_for('error_page', 
                                       error=training_stats.get('error_message', 'Training failed'),
                                       trace=training_stats.get('error_trace', ''))
                }), 400
            
            # Store training results in session
            session['training_results'] = training_stats
            
            # Store the model ID in session for convenience
            if 'model_id' in training_stats:
                session['current_model_id'] = training_stats['model_id']
                
            return jsonify({
                'success': True,
                'redirect': url_for('training_results')
            })
            
        except Exception as e:
            error_trace = traceback.format_exc()
            logger.error(f"Error in train_model: {str(e)}\n{error_trace}")
            return jsonify({
                'success': False,
                'error': str(e),
                'redirect': url_for('error_page', error=str(e), trace=error_trace)
            }), 500
        
            
    
    @app.route('/test_model_with_file', methods=['POST'])
    @login_required
    def test_model_with_file():
        
        
        """Test the model using uploaded file data."""
        user_id = session.get('user_id')
        if not user_id:
            flash('Please login to test models', 'error')
            return redirect(url_for('login'))
        
        # Get model ID from form or session
        model_id = request.form.get('model_id') or session.get('current_model_id')
        
        # If no model ID provided, show error
        if not model_id:
            flash('No model ID provided. Please select a model first.', 'error')
            return redirect(url_for('model_tester'))
        
        # File upload handling
        if 'file' not in request.files:
            flash('No file part', 'error')
            return redirect(url_for('model_tester'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file', 'error')
            return redirect(url_for('model_tester'))
        
        try:
            # Save file temporarily
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_{user_id}_{filename}')
            file.save(filepath)
            
            # Get model metadata for the UI
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT name, metadata
                        FROM models 
                        WHERE id = %s
                        """,
                        (model_id,)
                    )
                    model_data = cursor.fetchone()
            
            metadata = model_data.get('metadata', {})
            model_type = metadata.get('model_type', 'unknown')
            target_variable = metadata.get('target_variable', 'Unknown')
            display_model_name = metadata.get('title', model_data.get('name', 'Model'))
            
            # Format accuracy for display
            accuracy = metadata.get('accuracy', 'N/A')
            if isinstance(accuracy, float):
                accuracy = f"{accuracy * 100:.2f}%" if accuracy < 1 else f"{accuracy:.2f}%"
            
            # Use our improved prediction function
            prediction_df = predict_with_preprocessor(model_id, filepath, user_id)
            
            if prediction_df is None:
                flash("Error making prediction. See logs for details.", 'error')
                return redirect(url_for('model_tester'))
            
            # Clean up temp file
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
            
            # Current timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Get feature names for the UI
            feature_names = prediction_df.columns.tolist()
            if f'predicted_{target_variable}' in feature_names:
                feature_names.remove(f'predicted_{target_variable}')
            if 'probability' in feature_names:
                feature_names.remove('probability')
            if 'error' in feature_names:
                feature_names.remove('error')
            
            # Return results to template
            return render_template('model_tester.html',
                                  model_id=model_id,
                                  model_name=display_model_name,
                                  target_variable=target_variable,
                                  model_type=model_type,
                                  accuracy=accuracy,
                                  required_features=feature_names,
                                  feature_names=feature_names,
                                  prediction_df=prediction_df,
                                  timestamp=timestamp)
            
        except Exception as e:
            error_trace = traceback.format_exc()
            flash(f'Error processing prediction: {str(e)}', 'error')
            logger.error(f"Prediction error: {str(e)}\n{error_trace}")
            
            # Clean up temp file on error
            if 'filepath' in locals() and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except:
                    pass
                    
            return redirect(url_for('model_tester'))
        
    @app.route('/test_model_manual', methods=['POST'])
    @login_required
    def test_model_manual():
        """Test a model using manually entered data"""
        user_id = session.get('user_id')
        if not user_id:
            flash('Please login to test models', 'error')
            return redirect(url_for('login'))
        
        # Get model ID from form or session
        model_id = request.form.get('model_id') or session.get('current_model_id')
        
        # Validate model ID
        if not model_id:
            flash('No model ID provided. Please select a model first.', 'error')
            return redirect(url_for('model_tester'))
        
        try:
            # Get model metadata for display
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT name, metadata 
                        FROM models 
                        WHERE id = %s AND user_id = %s
                        """,
                        (model_id, user_id)
                    )
                    model_data = cursor.fetchone()
            
            if not model_data:
                flash('Model not found', 'error')
                return redirect(url_for('model_tester'))
            
            metadata = model_data.get('metadata', {})
            model_type = metadata.get('model_type', 'unknown')
            target_variable = metadata.get('target_variable', 'Unknown')
            display_model_name = metadata.get('title', model_data.get('name', 'Model'))
            
            # Format accuracy for display
            accuracy = metadata.get('accuracy', 'N/A')
            if isinstance(accuracy, float):
                accuracy = f"{accuracy * 100:.2f}%" if accuracy < 1 else f"{accuracy:.2f}%"
            
            # Collect input values from form
            input_data = {}
            for key, value in request.form.items():
                if key.startswith('feature_'):
                    feature_name = key.replace('feature_', '')
                    
                    # Skip empty values
                    if not value.strip():
                        continue
                    
                    # Try to convert to numeric if possible
                    try:
                        input_data[feature_name] = int(value)
                    except ValueError:
                        try:
                            input_data[feature_name] = float(value)
                        except ValueError:
                            input_data[feature_name] = value
            
            # Check if we have any input data
            if not input_data:
                flash('No input data provided. Please enter values for at least one feature.', 'error')
                return redirect(url_for('model_tester'))
            
            # Convert to DataFrame with a single row
            input_df = pd.DataFrame([input_data])
            
            # Make predictions using our improved prediction function
            prediction_df = predict_with_preprocessor(model_id, input_df, user_id)
            
            if prediction_df is None:
                flash('Prediction failed. Please check the logs for details.', 'error')
                return redirect(url_for('model_tester'))
            
            # Current timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Check for error in prediction
            if 'error' in prediction_df.columns:
                error_message = prediction_df['error'].iloc[0]
                flash(f'Error in prediction: {error_message}', 'warning')
            
            # Get feature names for display
            
            # Return results to template
            return render_template('model_tester.html',
                                 model_id=model_id,
                                 model_name=display_model_name,
                                 target_variable=target_variable,
                                 model_type=model_type,
                                 accuracy=accuracy,
                                 required_features=feature_names,
                                 feature_names=feature_names,
                                 prediction_df=prediction_df,
                                 timestamp=timestamp)
            
        except Exception as e:
            error_trace = traceback.format_exc()
            flash(f'Error processing prediction: {str(e)}', 'error')
            logger.error(f"Prediction error: {str(e)}\n{error_trace}")
            return redirect(url_for('model_tester'))



    @app.route('/saved-use-cases')
    @login_required
    def saved_use_cases():
        """Display saved use cases for the current user"""
        user_id = session.get('user_id')
        
        # Get saved use cases
        use_cases = get_user_use_cases(user_id)
        
        return render_template('saved_use_cases.html', 
                               use_cases=use_cases,
                               user_name=session.get('user_name', 'User'))
    
    @app.route('/view-use-case/<use_case_id>')
    @login_required
    def view_use_case(use_case_id):
        """View details of a specific saved use case"""
        user_id = session.get('user_id')
        
        # Get the use case from the database
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM use_cases 
                    WHERE id = %s AND user_id = %s
                    """,
                    (use_case_id, user_id)
                )
                use_case = cursor.fetchone()
        
        if not use_case:
            flash('Use case not found', 'error')
            return redirect(url_for('saved_use_cases'))
        
        # Extract essential information
        filename = use_case['filename']
        proposals = use_case['proposals']
        metadata = use_case['metadata']
        
        # Prepare session variables
        session['last_filename'] = filename
        session['proposal_count'] = len(proposals)
        session['proposals'] = proposals
        
        # Handle file path reconstruction
        temp_file_path = None
        
        try:
            # First, try preserved file path
            if metadata and 'preserved_file_path' in metadata and os.path.exists(metadata['preserved_file_path']):
                temp_file_path = metadata['preserved_file_path']
            
            # If no preserved path, try original file path
            elif metadata and 'file_path' in metadata and os.path.exists(metadata['file_path']):
                temp_file_path = metadata['file_path']
            
            # If still no file, create a temporary CSV
            if not temp_file_path:
                import tempfile
                import csv
                
                # Create a temporary file
                temp_file_path = os.path.join(tempfile.gettempdir(), f"use_case_{use_case_id}.csv")
                
                # Attempt to create a CSV based on the first proposal
                if proposals:
                    first_proposal = proposals[0]
                    target_variable = first_proposal.get('target_variable', 'Unknown')
                    
                    # Attempt to extract potential feature names from description
                    features = []
                    if 'description' in first_proposal:
                        potential_features = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', first_proposal['description'])
                        features = list(set(potential_features))[:5]  # Limit to 5 unique features
                    
                    # Write CSV with target variable and placeholder features
                    with open(temp_file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        
                        # Write header
                        header = [target_variable] + features if features else [target_variable, 'feature1', 'feature2']
                        writer.writerow(header)
                        
                        # Write a single placeholder row
                        placeholder_row = ['0'] * len(header)
                        writer.writerow(placeholder_row)
                else:
                    # Fallback to minimal CSV
                    with open(temp_file_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(['target', 'feature1'])
                        writer.writerow(['0', '0'])
            
            # Set the file path in session
            session['file_path'] = temp_file_path
            
        except Exception as e:
            # Log the error, but continue with the process
            logger.error(f"Error creating temporary file: {str(e)}")
            flash('Could not recreate original file, using minimal placeholder', 'warning')
        
        # Include target variable in session if available
        if proposals and len(proposals) > 0:
            first_proposal = proposals[0]
            session['target_variable'] = first_proposal.get('target_variable', 'Unknown')
        
        return render_template('results.html', 
                               filename=filename,
                               proposals=proposals)
    
    @app.route('/delete-use-case/<use_case_id>', methods=['POST'])
    @login_required
    def delete_use_case_route(use_case_id):
        """Delete a specific use case"""
        user_id = session.get('user_id')
        
        try:
            success = delete_use_case(user_id, use_case_id)
            
            if success:
                flash('Use case deleted successfully', 'success')
            else:
                flash('Failed to delete use case', 'error')
            
            return redirect(url_for('saved_use_cases'))
        
        except Exception as e:
            flash(f'Error deleting use case: {str(e)}', 'error')
            return redirect(url_for('saved_use_cases'))
    
    @app.route('/')
    @app.route('/home')
    def home():
        """
        Home dashboard page that serves as a central hub for the application.
        Shows stats and provides navigation to all main sections.
        """
        # Check if user is logged in via session
        if not session.get('user_id'):
            # If not logged in, redirect to login page
            return redirect(url_for('login'))
        
        # Get stats for the dashboard
        user_id = session.get('user_id')
        
        # Query the database for actual statistics
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    # Get model count
                    cursor.execute("SELECT COUNT(*) AS count FROM models WHERE user_id = %s", (user_id,))
                    model_count = cursor.fetchone()['count']
                    
                    # Get use case count
                    cursor.execute("SELECT COUNT(*) AS count FROM use_cases WHERE user_id = %s", (user_id,))
                    use_case_count = cursor.fetchone()['count']
                    
                    # Get embedding count
                    cursor.execute("SELECT COUNT(*) AS count FROM embeddings WHERE user_id = %s", (user_id,))
                    embedding_count = cursor.fetchone()['count']
                    
                    # Get average model accuracy
                    cursor.execute("SELECT AVG(accuracy) AS avg_accuracy FROM models WHERE user_id = %s", (user_id,))
                    avg_accuracy = cursor.fetchone()['avg_accuracy']
                    
                    stats = {
                        'model_count': model_count,
                        'use_case_count': use_case_count,
                        'embedding_count': embedding_count,
                        'accuracy': f"{avg_accuracy * 100:.2f}%" if avg_accuracy else 'N/A'
                    }
        except Exception as e:
            logger.error(f"Error fetching dashboard stats: {str(e)}")
            stats = {
                'model_count': 0,
                'use_case_count': 0,
                'embedding_count': 0,
                'accuracy': 'N/A'
            }
        
        return render_template('home.html', stats=stats)
    
    # API endpoints
    @app.route('/api/get-file-preview', methods=['POST'])
    @login_required
    def api_get_file_preview():
        """API endpoint to get a preview of a file"""
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            try:
                # Save file temporarily
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'preview_' + filename)
                file.save(filepath)
                
                # Read data
                if read_data_flexible:
                    data = read_data_flexible(filepath)
                else:
                    # Fallback to pandas directly
                    if filepath.endswith('.csv'):
                        data = pd.read_csv(filepath)
                    elif filepath.endswith('.xlsx'):
                        data = pd.read_excel(filepath)
                    else:
                        return jsonify({'error': 'Unsupported file format'}), 400
                
                if data is None:
                    return jsonify({'error': 'Failed to read file'}), 400
                
                # Convert to dict for JSON response
                preview = {
                    'columns': list(data.columns),
                    'rows': data.head(10).to_dict(orient='records'),
                    'num_rows': len(data),
                    'num_columns': len(data.columns)
                }
                
                # Clean up temp file
                try:
                    os.remove(filepath)
                except:
                    pass
                
                return jsonify({'success': True, 'preview': preview})
                
            except Exception as e:
                logger.error(f"Error generating file preview: {str(e)}")
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    
    @app.route('/api/get-column-stats', methods=['POST'])
    @login_required
    def api_get_column_stats():
        """API endpoint to get statistics for a column in a file"""
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
            
        file = request.files['file']
        column = request.form.get('column')
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
            
        if not column:
            return jsonify({'error': 'No column specified'}), 400
            
        if file and allowed_file(file.filename):
            try:
                # Save file temporarily
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'stats_' + filename)
                file.save(filepath)
                
                # Read data
                if read_data_flexible:
                    data = read_data_flexible(filepath)
                else:
                    # Fallback to pandas directly
                    if filepath.endswith('.csv'):
                        data = pd.read_csv(filepath)
                    elif filepath.endswith('.xlsx'):
                        data = pd.read_excel(filepath)
                    else:
                        return jsonify({'error': 'Unsupported file format'}), 400
                
                if data is None:
                    return jsonify({'error': 'Failed to read file'}), 400
                
                if column not in data.columns:
                    return jsonify({'error': f'Column "{column}" not found'}), 400
                
                # Calculate stats
                stats = {}
                col_data = data[column]
                
                # Basic stats
                stats['count'] = len(col_data)
                stats['null_count'] = col_data.isnull().sum()
                stats['null_percent'] = f"{(col_data.isnull().sum() / len(col_data) * 100):.2f}%"
                
                # Check data type
                if pd.api.types.is_numeric_dtype(col_data):
                    # Numeric stats
                    stats['data_type'] = 'numeric'
                    stats['min'] = float(col_data.min()) if not pd.isnull(col_data.min()) else None
                    stats['max'] = float(col_data.max()) if not pd.isnull(col_data.max()) else None
                    stats['mean'] = float(col_data.mean()) if not pd.isnull(col_data.mean()) else None
                    stats['median'] = float(col_data.median()) if not pd.isnull(col_data.median()) else None
                    stats['std'] = float(col_data.std()) if not pd.isnull(col_data.std()) else None
                    
                    # Distribution data
                    try:
                        hist, bin_edges = np.histogram(col_data.dropna(), bins=10)
                        stats['histogram'] = {
                            'counts': hist.tolist(),
                            'bin_edges': bin_edges.tolist()
                        }
                    except:
                        stats['histogram'] = None
                else:
                    # Categorical stats
                    stats['data_type'] = 'categorical'
                    value_counts = col_data.value_counts().head(10)
                    stats['value_counts'] = {
                        'labels': value_counts.index.tolist(),
                        'counts': value_counts.values.tolist()
                    }
                    stats['unique_count'] = col_data.nunique()
                    stats['top_value'] = col_data.value_counts().index[0] if len(col_data.value_counts()) > 0 else None
                    stats['top_count'] = int(col_data.value_counts().values[0]) if len(col_data.value_counts()) > 0 else 0
                
                # Clean up temp file
                try:
                    os.remove(filepath)
                except:
                    pass
                
                return jsonify({'success': True, 'stats': stats})
                
            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"Error: {str(e)}\n{error_trace}")
                return jsonify({'error': str(e)}), 500
        else:
            return jsonify({'error': 'File type not allowed'}), 400
    
    @app.route('/api/log-prediction', methods=['POST'])
    @login_required
    def api_log_prediction():
        """API endpoint to log a prediction for analytics"""
        user_id = session.get('user_id')
        model_id = request.json.get('model_id')
        input_data = request.json.get('input_data')
        prediction_result = request.json.get('prediction_result')
        
        if not model_id or not input_data or not prediction_result:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        INSERT INTO prediction_logs (
                            user_id, model_id, input_data, prediction_result, timestamp
                        )
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (user_id, model_id, json.dumps(input_data), json.dumps(prediction_result), datetime.now())
                    )
                    log_id = cursor.fetchone()['id']
                    conn.commit()
            
            return jsonify({'success': True, 'log_id': str(log_id)})
        
        except Exception as e:
            logger.error(f"Error logging prediction: {str(e)}")
            return jsonify({'error': str(e)}), 500
        
    @app.route('/api/models', methods=['GET'])
    @login_required
    def api_models():
        """API endpoint to list all models for the current user"""
        if not session.get('user_id'):
            return jsonify({'error': 'Unauthorized'}), 401
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT id, name, metadata, created_at,
                               pg_column_size(model_data) as model_size
                        FROM models 
                        WHERE user_id = %s
                        ORDER BY created_at DESC
                        """,
                        (session['user_id'],)
                    )
                    models = cursor.fetchall()
                    
            # Format model data for JSON response
            formatted_models = []
            for model in models:
                formatted_model = {
                    'id': model['id'],
                    'name': model['name'],
                    'created_at': model['created_at'].isoformat(),
                    'model_size_kb': round(model['model_size'] / 1024, 2),
                    'target_variable': model['metadata'].get('target_variable', 'Unknown'),
                    'model_type': model['metadata'].get('model_type', 'Unknown'),
                    'metrics': {
                        'accuracy': model['metadata'].get('metrics', {}).get('accuracy', 'N/A'),
                        'recall': model['metadata'].get('metrics', {}).get('recall', 'N/A')
                    }
                }
                formatted_models.append(formatted_model)
            
            return jsonify({
                'count': len(formatted_models),
                'models': formatted_models
            })
        except Exception as e:
            logger.error(f"Error retrieving models: {str(e)}")
            return jsonify({'error': str(e)}), 500
        
    @app.route('/debug/model/<model_id>', methods=['GET'])
    @login_required
    def debug_model(model_id):
        """Debug view to examine a model's detailed metadata"""
        if not session.get('user_id'):
            return redirect(url_for('login'))
        
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        """
                        SELECT id, name, metadata, created_at,
                               pg_column_size(model_data) as model_size
                        FROM models 
                        WHERE id = %s AND user_id = %s
                        """,
                        (model_id, session['user_id'])
                    )
                    model = cursor.fetchone()
                    
            if not model:
                flash('Model not found', 'error')
                return redirect(url_for('my_models'))
            
            # Format model data for template
            model_details = {
                'id': model['id'],
                'name': model['name'],
                'created_at': model['created_at'].isoformat(),
                'model_size_kb': round(model['model_size'] / 1024, 2),
                'metadata': model['metadata']
            }
            
            return render_template('debug_model.html', model=model_details)
        except Exception as e:
            error_trace = traceback.format_exc()
            flash(f'Error examining model: {str(e)}', 'error')
            logger.error(f"Error examining model: {str(e)}\n{error_trace}")
            return redirect(url_for('my_models'))
        
    
        
     
    
        """Display the business insights dashboard - AI sampling, frontend pagination"""
        
        # Get filename from session (not URL)
        filename = session.get('last_uploaded_file')
        if not filename:
            flash('No data file found. Please upload and analyze a file first.', 'error')
            return redirect(url_for('upload_file'))
        
        # Get file path from session
        file_path = session.get('mapped_file_path') or session.get('original_file_path')
        if not file_path:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                original_path = session.get('original_file_path')
                if original_path and os.path.exists(original_path):
                    file_path = original_path
                else:
                    flash('Data file not found. Please re-upload.', 'error')
                    return redirect(url_for('upload_file'))
            
            # Read the FULL dataset
            logger.info(f"Loading complete dataset from: {file_path}")
            
            try:
                df = read_data_flexible(file_path)
                if df is None or df.empty:
                    if file_path.lower().endswith('.csv'):
                        df = pd.read_csv(file_path, encoding='utf-8')
                    elif file_path.lower().endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(file_path)
                    else:
                        return jsonify({'success': False, 'error': 'Unsupported file format'})
    
                logger.info(f"Complete dataset loaded: {df.shape}")
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
                return jsonify({'success': False, 'error': f'Error reading file: {str(e)}'})
            
            # Send FULL dataset to frontend (will handle pagination client-side)
            data = df.to_dict('records')
            
            # Clean numpy types for JSON serialization
            from enhanced_column_mapper import convert_numpy_types
            data = convert_numpy_types(data)
            
            logger.info(f"Sending {len(data)} rows to frontend for client-side pagination")
            
            # Generate AI insights using SAMPLED data for cost control
            chart_insights = {}
            overall_insights = []
            recommendations = []
            
            if gemini_insights:
                try:
                    # Sample data for AI analysis to control costs
                    ai_sample_size = min(5000, len(df))
                    if len(df) > ai_sample_size:
                        logger.info(f"Sampling {ai_sample_size} rows from {len(df)} for AI analysis (cost control)")
                        df_ai_sample = df.sample(n=ai_sample_size, random_state=42)
                    else:
                        df_ai_sample = df
                        logger.info(f"Using complete dataset ({len(df)} rows) for AI analysis")
                    
                    # Analyze using sampled data
                    numeric_columns = df_ai_sample.select_dtypes(include=['number']).columns.tolist()
                    categorical_columns = df_ai_sample.select_dtypes(exclude=['number']).columns.tolist()
                    
                    columns_to_analyze = (numeric_columns[:4] + categorical_columns[:2])[:6]
                    
                    for column in columns_to_analyze:
                        try:
                            if column in numeric_columns:
                                insights = gemini_insights.analyze_chart_data(
                                    column, 
                                    df_ai_sample[column].dropna().tolist(), 
                                    'numeric'
                                )
                            else:
                                insights = gemini_insights.analyze_chart_data(
                                    column, 
                                    df_ai_sample[column].dropna().tolist(), 
                                    'categorical'
                                )
                            
                            chart_insights[column] = insights
                            logger.info(f"Generated insights for {column} using {len(df_ai_sample)} sample rows")
                            
                        except Exception as e:
                            logger.error(f"Error generating insights for {column}: {e}")
                            chart_insights[column] = [{
                                'type': 'insight',
                                'icon': '📊',
                                'title': f'{column} Analysis',
                                'content': f'Analysis for {column} based on {ai_sample_size:,} representative records.'
                            }]
                    
                    # Generate overall analysis using sampled data
                    try:
                        full_analysis = gemini_insights.analyze_full_dataset(df_ai_sample)
                        overall_insights = full_analysis.get('overall_insights', [])
                        recommendations = full_analysis.get('recommendations', [])
                        
                        # Add sampling note to insights if data was sampled
                        if len(df) > ai_sample_size:
                            sample_note = f" (AI analysis based on {ai_sample_size:,} representative records for cost efficiency)"
                            for insight in overall_insights:
                                if 'content' in insight:
                                    insight['content'] += sample_note
                        
                        logger.info(f"Generated AI insights using {len(df_ai_sample)} rows for cost control")
                        
                    except Exception as e:
                        logger.error(f"Error generating overall analysis: {e}")
                        overall_insights = [{
                            'type': 'insight',
                            'icon': '🤖',
                            'title': 'AI Analysis',
                            'content': f'AI analysis processed {ai_sample_size:,} representative records for cost efficiency. Complete dataset has {len(df):,} rows.'
                        }]
                        
                except Exception as e:
                    logger.error(f"Error in AI insight generation: {e}")
                    overall_insights = [{
                        'type': 'warning',
                        'icon': '⚠️',
                        'title': 'AI Analysis Unavailable',
                        'content': f'AI insights temporarily unavailable. Statistical analysis available for all {len(df):,} rows.'
                    }]
            else:
                overall_insights = [{
                    'type': 'info',
                    'icon': 'ℹ️',
                    'title': 'Standard Analysis',
                    'content': f'Statistical analysis for {len(df):,} rows. Configure Gemini API for AI-powered insights.'
                }]
            
            # Get additional data
            eda_results = session.get('eda_results', {})
            if not eda_results and session.get('user_id'):
                try:
                    from enhanced_column_mapper import EnhancedColumnMapper
                    column_mapper = EnhancedColumnMapper()
                    _, eda_results = column_mapper.get_saved_mappings_with_eda(
                        session['user_id'], 
                        filename
                    )
                except Exception as e:
                    logger.warning(f"Could not retrieve EDA results: {str(e)}")
            
            column_mappings = session.get('column_mappings', [])
            
            # Prepare context for template
            context = {
                'data': data,  # Full dataset - frontend will handle pagination
                'filename': filename,
                'total_rows': len(df),
                'total_columns': len(df.columns),
                'displayed_rows': len(data),  # Same as total_rows
                'eda_results': eda_results,
                'column_mappings': column_mappings,
                'has_eda': bool(eda_results),
                'chart_insights': chart_insights,
                'overall_insights': overall_insights,
                'recommendations': recommendations
            }
            
            return render_template('business_insights.html', **context)
            
        except Exception as e:
            logger.error(f"Error loading data for business insights: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            flash(f'Error loading data: {str(e)}', 'error')
            return redirect(url_for('upload_file'))  
    
    @app.route('/business-insights')
    def business_insights():
        """Display the integrated dashboard with optional AI insights"""
        
        # Get filename from session (not URL)
        filename = session.get('last_uploaded_file')
        if not filename:
            flash('No data file found. Please upload and analyze a file first.', 'error')
            return redirect(url_for('upload_file'))
        
        # Get file path from session
        file_path = session.get('mapped_file_path') or session.get('original_file_path')
        if not file_path:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                original_path = session.get('original_file_path')
                if original_path and os.path.exists(original_path):
                    file_path = original_path
                else:
                    flash('Data file not found. Please re-upload.', 'error')
                    return redirect(url_for('upload_file'))
            
            # Read the data
            logger.info(f"Loading dataset from: {file_path}")
            
            try:
                df = read_data_flexible(file_path)
                if df is None or df.empty:
                    if file_path.lower().endswith('.csv'):
                        df = pd.read_csv(file_path, encoding='utf-8')
                    elif file_path.lower().endswith(('.xlsx', '.xls')):
                        df = pd.read_excel(file_path)
                    else:
                        flash('Unsupported file format', 'error')
                        return redirect(url_for('upload_file'))
    
                logger.info(f"Dataset loaded: {df.shape}")
            except Exception as e:
                logger.error(f"Error reading file: {str(e)}")
                flash(f'Error reading file: {str(e)}', 'error')
                return redirect(url_for('upload_file'))
            
            # Convert DataFrame to list of dictionaries for JavaScript
            data = df.to_dict('records')
            
            # Clean numpy types for JSON serialization
            def clean_for_json(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif pd.isna(obj):
                    return None
                else:
                    return obj
            
            # Clean all data values
            cleaned_data = []
            for row in data:
                cleaned_row = {}
                for key, value in row.items():
                    cleaned_row[key] = clean_for_json(value)
                cleaned_data.append(cleaned_row)
            
            logger.info(f"Sending {len(cleaned_data)} rows to integrated dashboard")
            
            # Optional: Generate AI insights if Gemini is configured
            ai_insights = {}
            if gemini_insights and len(df) < 10000:  # Only for smaller datasets to control costs
                try:
                    logger.info("Generating AI insights for dashboard...")
                    
                    # Sample columns for AI analysis
                    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()[:3]
                    categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()[:2]
                    
                    for column in numeric_columns + categorical_columns:
                        try:
                            if column in numeric_columns:
                                insights = gemini_insights.analyze_chart_data(
                                    column, 
                                    df[column].dropna().tolist()[:1000],  # Sample first 1000 values
                                    'numeric'
                                )
                            else:
                                insights = gemini_insights.analyze_chart_data(
                                    column, 
                                    df[column].dropna().tolist()[:1000], 
                                    'categorical'
                                )
                            ai_insights[column] = insights
                        except Exception as e:
                            logger.error(f"Error generating insights for {column}: {e}")
                    
                except Exception as e:
                    logger.error(f"Error in AI insight generation: {e}")
            
            # Get additional data from session
            eda_results = session.get('eda_results', {})
            column_mappings = session.get('column_mappings', [])
            
            # Render the integrated dashboard template
            return render_template('integrated_dashboard.html',
                                 filename=filename,
                                 data=cleaned_data,
                                 ai_insights=ai_insights,  # Optional AI insights
                                 eda_results=eda_results,  # Optional EDA results
                                 column_mappings=column_mappings)  # Optional column mappings
            
        except Exception as e:
            logger.error(f"Error loading data for integrated dashboard: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            flash(f'Error loading data: {str(e)}', 'error')
            return redirect(url_for('upload_file'))

 
def load_model_from_db(model_id, user_id=None):
    """
    Load a model and its preprocessor from the database
    
    Args:
        model_id (str): Database ID of the model
        user_id (str, optional): User ID for permission check
        
    Returns:
        tuple: (model_object, preprocessor_object, model_data)
    """
    try:
        import joblib
        import io
        import pickle
        
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Query with optional user_id check
                if user_id:
                    cursor.execute(
                        """
                        SELECT id, name, model_data, metadata, created_at 
                        FROM models 
                        WHERE id = %s AND user_id = %s
                        """,
                        (model_id, user_id)
                    )
                else:
                    cursor.execute(
                        """
                        SELECT id, name, model_data, metadata, created_at 
                        FROM models 
                        WHERE id = %s
                        """,
                        (model_id,)
                    )
                model_data = cursor.fetchone()
                
                if not model_data:
                    logger.warning(f"Model with ID {model_id} not found in database")
                    return None, None, None
                
                # Check if model_data contains binary data
                if model_data.get('model_data'):
                    logger.info(f"Model data size: {len(model_data['model_data'])} bytes")
                    
                    try:
                        # Try to deserialize as combined data package (new format)
                        combined_data = pickle.loads(model_data['model_data'])
                        
                        if isinstance(combined_data, dict) and 'model' in combined_data and 'preprocessor' in combined_data:
                            # Unpack model and preprocessor
                            model_bytes = combined_data['model']
                            preprocessor_bytes = combined_data['preprocessor']
                            
                            # Load model
                            model_buffer = io.BytesIO(model_bytes)
                            model_object = joblib.load(model_buffer)
                            
                            # Load preprocessor
                            preprocessor_buffer = io.BytesIO(preprocessor_bytes)
                            preprocessor_object = joblib.load(preprocessor_buffer)
                            
                            logger.info(f"Successfully loaded model and preprocessor from database for model: {model_id}")
                            return model_object, preprocessor_object, model_data
                        else:
                            # Legacy format - just the model
                            logger.info("Legacy format detected - model without preprocessor")
                            buffer = io.BytesIO(model_data['model_data'])
                            model_object = joblib.load(buffer)
                            return model_object, None, model_data
                    except Exception as e:
                        logger.error(f"Error deserializing model data: {str(e)}")
                        logger.error(traceback.format_exc())
                        
                        # Last resort - try direct load as a model
                        try:
                            buffer = io.BytesIO(model_data['model_data'])
                            model_object = joblib.load(buffer)
                            logger.warning("Model loaded without preprocessor - predictions may fail")
                            return model_object, None, model_data
                        except:
                            logger.error("Failed to load model as fallback")
                            return None, None, model_data
                else:
                    logger.error(f"Model data is empty for model ID: {model_id}")
                    return None, None, model_data
    except Exception as e:
        logger.error(f"Error loading model from database: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None
 
if __name__ == '__main__':
    app = current_app()
    init_routes(app)
    
    try:
       integrate_optimized_insights(app, Config.GOOGLE_API_KEY)
       logger.info("Optimized business insights integrated successfully")
    except Exception as e:
       logger.error(f"Failed to integrate optimized insights: {str(e)}")
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)