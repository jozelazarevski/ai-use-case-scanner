# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:47:21 2025

@author: joze_
"""
import re
import tempfile
import csv
import logging
import traceback
import os
import re
import json
import requests
import time
import random
import pandas as pd
from werkzeug.utils import secure_filename
from datetime import datetime
import joblib
import sys
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
from flask_session import Session

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)   

# Import config
from config import Config
from config import encodings

# Import ML modules
from ml_trainer import train_model_with_robust_error_handling
from execute_LLM_model import run_training_script
from utils.read_file import read_data_flexible

# Import user authentication module from utils
from utils.user_auth import init_auth_routes, init_database
from utils import user_models
# Add parent directory to path to import modules from ml folder
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# Import ML prediction modules
try:
    from ml.predict_model_classification import predict_classification
    from ml.predict_model_regression import predict_regression
    from ml.read_file import read_data_flexible
except ImportError:
    print("WARNING: Could not import prediction modules from ml folder")

# Import Google's Generative AI library
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("Warning: Google Generative AI library not found. Gemini API will not be available.")

USER = 'joze'
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count()) # Or any other number

# Initialize Flask app
app = Flask(__name__, static_folder='static')


 
if not os.path.exists('static'):
    os.makedirs('static')
    
# Create database directories if they don't exist
if not os.path.exists(Config.DATABASE_DIR):
    os.makedirs(Config.DATABASE_DIR)

# Create user models directory if it doesn't exist
user_models_dir = os.path.join(Config.DATABASE_DIR, 'user_models')
if not os.path.exists(user_models_dir):
    os.makedirs(user_models_dir)

# Initialize user database
init_database()

# Load configuration from Config class
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = Config.ALLOWED_EXTENSIONS
app.secret_key = Config.SECRET_KEY

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

# Initialize authentication routes and middleware
init_auth_routes(app)

# Get active model from config
ACTIVE_MODEL = Config.ACTIVE_MODEL.lower()
if ACTIVE_MODEL not in ['claude', 'gemini']:
    print(f"Warning: Unknown model '{ACTIVE_MODEL}' specified. Defaulting to Claude.")
    ACTIVE_MODEL = 'gemini'

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
        print(f"Gemini model '{GEMINI_MODEL_NAME}' configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {str(e)}")
        print("Falling back to Claude due to Gemini configuration error.")
        ACTIVE_MODEL = 'claude'

# Log which model will be used
print(f"Using {ACTIVE_MODEL.capitalize()} as the active AI model.")

# Create necessary directories if they don't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists('static'):
    os.makedirs('static')
    
# Create database directories if they don't exist
if not os.path.exists(Config.DATABASE_DIR):
    os.makedirs(Config.DATABASE_DIR)

# Initialize user database
init_database()

# Login required decorator - imported from utils.user_auth after initialization
from utils.user_auth import login_required

# Helper functions
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

# Register template filters and global functions
app.jinja_env.filters['nl2br'] = lambda text: text.replace('\n', '<br>') if text else ''
app.jinja_env.globals.update(read_script_file=read_script_file)

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_unique_filename_timestamp(file="", model_type="", user=USER):
    """
    Generates a unique filename using a timestamp.
    """
    timestamp = int(time.time() * 1000)  # Get current timestamp in milliseconds
    # Check if user is logged in and use their ID
    if 'user_id' in session:
        user = session['user_id'][:8]  # Use first 8 chars of user ID
        
    filename = f"{user}_{model_type}_{file}_{timestamp}"
    filename = re.sub(r'[^a-zA-Z0-9_\-]', "", filename)

    return filename

def limit_session_size(session, max_proposals=4):
    """Limit the number of proposals stored in the session."""
    if 'proposals' in session and len(session['proposals']) > max_proposals:
        session['proposals'] = session['proposals'][:max_proposals]
        session.modified = True

@app.route('/', methods=['GET', 'POST'])
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
                
                # Get AI use case proposals using the active model from config
                try:
                    ai_proposals = get_ai_use_case_proposals(file_content, filename)
                    
                    # Save use cases for the current user
                    if 'user_id' in session:
                       save_use_case_id = user_models.save_use_cases(
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
                    print(f"API Error: {str(api_error)}\n{error_trace}")
                    flash(f"Error generating AI use cases: {str(api_error)}", 'error')
                    return render_template('error.html', error=str(api_error), trace=error_trace)
                
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

# Function to get AI use case proposals using the selected model
def get_ai_use_case_proposals(file_content, filename):
    """Generate AI use case proposals using the selected AI model based on file content."""
    # JSON structure example - properly escaped for inclusion in a string
    json_example = '''
{
  "title": "Title of the use case",
  "description": "Detailed description of the use case",
  "kpis": ["Business KPI 1 : one sentence description on the KPI", "Business KPI 2 : one sentence description on the KPI", "Business KPI 3 : one sentence description on the KPI"],
  "business_value": "Comprehensive explanation of what business value will the Use case bring to the organization",
  "target_variable": "column_name",
  "model_type": "classification/regression/clustering/sentiment analysis",
  "use_case_implementation_complexity": "hard/medium/easy - judged by the target variable and model type",
  "prediction_interpretation": "comprehensive explanation how to interpret the AI prediction with examples. "
  "target variable understanding": "create analysis on the taret variable and explain the meaning for the usecase. expalin the number of unique predictions that will happen using this target variable."

}'''
    # dont propose different usecases that use the same target variable and Ml model.
    # Base prompt for all models
    prompt_text = f"""Based on the uploaded file content below, please generate comprehensive AI use case proposals. 
    Create usecases only on the available varibles. Do not create usecase where i have to do feature engeneering.
    Do not limit to only 4 use cases - provide as many distinct use cases as you can identify from the data.
   
    FILENAME: {filename}
    
    FILE CONTENT:
    {file_content[:20000]}  # Limit content length to avoid token limits
    
    For each AI use case proposal, please include:
    1. A clear, descriptive title
    2. A detailed description of how AI could be applied (2-3 paragraphs)
    3. 3-5 specific Key Performance Indicators (KPIs) that could measure success
    4. Business value description
    5. Explicitly identify a TARGET_VARIABLE from the dataset columns
    6. Specify a MODEL_TYPE (classification, regression, clustering, or sentiment analysis)
    7. Assess implementation complexity (easy, medium, or hard) based on the target variable and model type
    8. Provide a clear explanation of how to interpret predictions from this model with examples
    
    FORMAT YOUR RESPONSE AS A VALID JSON ARRAY OF OBJECTS. Each object should have the following structure:
    {json_example}
    
    Do not include any markdown formatting, explanatory text, or non-JSON content in your response.
    # dont propose different usecases that use the same target variable and Ml model.

    """
    
    if ACTIVE_MODEL == "gemini" and HAS_GEMINI_CONFIG:
        # Use Gemini API
        try:
            print("Using Gemini API to generate proposals")
            response = gemini_model.generate_content(prompt_text)
            if response and hasattr(response, 'text'):
                text = response.text
                
                # Try to parse JSON from response
                try:
                    # Extract JSON content from text (in case there's surrounding text)
                    json_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
                    if json_match:
                        json_text = json_match.group(0)
                    else:
                        json_text = text
                        
                    proposals = json.loads(json_text)
                    
                    # Validate and clean up proposals
                    if isinstance(proposals, list) and len(proposals) > 0:
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
                    else:
                        raise ValueError("Invalid JSON structure: expected a list of proposal objects")
                    
                except (json.JSONDecodeError, ValueError) as json_error:
                    print(f"Gemini JSON parsing failed: {str(json_error)}. Falling back to text parsing.")
                    # Fall back to the original text parsing method
                    proposals = parse_proposals(text)
                    if proposals:
                        return proposals
                    else:
                        raise ValueError("Failed to parse proposals from Gemini's response")
            else:
                raise ValueError("Empty or invalid response from Gemini API")
                
        except Exception as e:
            error_message = f"Error with Gemini API: {str(e)}"
            print(error_message)
            # Do not automatically fall back to Claude since model is specified in config
            raise ValueError(f"Error with Gemini API: {str(e)}. Check your configuration or try setting ACTIVE_MODEL to 'claude' in config.py.")
    
    else:
        # Use Claude API (default)
        if not CLAUDE_API_KEY or CLAUDE_API_KEY == "your_claude_api_key_here":
            # For demonstration purposes, return dummy data if no API key is configured
            return []
        
        # Prepare the headers for Claude API request
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        # Prepare the request data
        data = {
            "model": CLAUDE_MODEL,
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
                print(f"Calling Claude API - attempt {attempt+1}")
                response = requests.post(CLAUDE_API_URL, headers=headers, json=data, timeout=60)
                
                # Log response status and content for debugging
                print(f"Status code: {response.status_code}")
                
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
                                for proposal in proposals:
                                    if not isinstance(proposal, dict):
                                        raise ValueError("Proposals must be a list of objects")
                                    
                                    # Ensure all required fields are present
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
                                    # Add default values for new fields if missing
                                    if 'business_value' not in proposal:
                                        proposal['business_value'] = "Not specified"
                                    if 'use_case_implementation_complexity' not in proposal:
                                        proposal['use_case_implementation_complexity'] = "medium"
                                    if 'prediction_interpretation' not in proposal:
                                        proposal['prediction_interpretation'] = "Predictions should be interpreted in the context of the business problem and validated by domain experts."
                                
                                return proposals
                            else:
                                raise ValueError("Invalid JSON structure: expected a list of proposal objects")
                        
                        except (json.JSONDecodeError, ValueError) as json_error:
                            print(f"JSON parsing failed: {str(json_error)}. Falling back to text parsing.")
                            # Fall back to the original text parsing method
                            proposals = parse_proposals(text)
                            if proposals:
                                return proposals
                            else:
                                raise ValueError("Failed to parse proposals from Claude's response")
                        
                # Handle rate limiting (429) with exponential backoff
                if response.status_code == 429:
                    wait_time = (2 ** attempt) + 1  # Exponential backoff: 1, 3, 7 seconds
                    print(f"Rate limited. Waiting {wait_time} seconds before retrying...")
                    time.sleep(wait_time)
                    continue
                    
                # Log the error details
                error_message = f"API Error: Status {response.status_code}, Response: {response.text}"
                print(error_message)
                last_error = ValueError(error_message)
                
            except Exception as e:
                error_message = f"Exception during API call: {str(e)}"
                print(error_message)
                last_error = e
                
                # Wait before retrying
                time.sleep(2)
        
        # If we get here, all attempts failed
        if last_error:
            raise last_error
        else:
            raise ValueError("Failed to get proposals from Claude API after multiple attempts")

def parse_proposals(text):
    """Parse Claude's response into structured use case proposals."""
    print("Parsing Claude's response text")
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
    
    print(f"Found {len(proposals)} proposals")
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
            'feature_importance': feature_importance,  # Now contains Python native types
            'llm_explanation': training_results.get('llm_explanation'),
            'training_results': {
                'title': training_results.get('title'),
                'description': training_results.get('description'),
                'target_variable': training_results.get('target_variable'),
                'business_value': training_results.get('business_value'),
                'prediction_interpretation': training_results.get('prediction_interpretation'),
                'kpis': training_results.get('kpis', []),
            }
        }
        
        return render_template('training_results.html', **context)
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in training_results: {str(e)}\n{error_trace}")
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
            print(f"Error cleaning up files: {str(e)}")

# Function to generate model explanations using LLM
def generate_model_explanation(training_stats):
    """
    Generate an explanation of the model results using LLM
    
    Args:
        training_stats (dict): Results from the training execution
        
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
    if ACTIVE_MODEL == "gemini" and HAS_GEMINI_CONFIG:
        response = gemini_model.generate_content(prompt_text)
       
        explanation = response.text
        explanation = re.sub(r'```\w*\n', '', explanation)  # Opening code fences with optional language
        explanation = re.sub(r'```', '', explanation)  # Closing code fences
    else:
        # Use Claude API as fallback
        headers = {
            "x-api-key": CLAUDE_API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        
        data = {
            "model": CLAUDE_MODEL,
            "max_tokens": 2000,
            "messages": [
                {"role": "user", "content": prompt_text}
            ]
        }
        
        response = requests.post(CLAUDE_API_URL, headers=headers, json=data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            # Extract text from Claude's response
            explanation = ""
            if result.get("content"):
                content_blocks = result["content"]
                for block in content_blocks:
                    if block.get("type") == "text":
                        explanation += block.get("text", "")
        else:
            explanation = f"Error generating explanation: API Error ({response.status_code})"
    
    return explanation


@app.route('/my-models')
@login_required
def my_models():
    """Display the user's saved models."""
    user_id = session.get('user_id')
    if not user_id:
        flash('Please login to view your models', 'error')
        return redirect(url_for('login'))
    
    # Get user's models
    user_models_list = user_models.get_user_models(user_id)
    
    # Get user's embeddings
    user_embeddings = user_models.get_user_embeddings(user_id)
    
    return render_template('my_models.html', 
                          models=user_models_list,
                          embeddings=user_embeddings,
                          user_name=session.get('user_name', 'User'))

@app.route('/save-current-model', methods=['POST'])
@login_required
def save_current_model():
    """Save the currently trained model to the user's account."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    # Get model info from the form
    model_name = request.form.get('model_name', 'Untitled Model')
    description = request.form.get('description', '')
    
    # Get training results from session
    training_results = session.get('training_results')
    
    if not training_results:
        return jsonify({'success': False, 'error': 'No trained model found in session'}), 400
    
    # Get model path
    model_path = None
    if 'model_path' in training_results:
        model_path = training_results['model_path']
    else:
        # Find model path based on target variable
        target_variable = training_results.get('target_variable')
        if target_variable:
            safe_target = ''.join(c if c.isalnum() else '_' for c in target_variable)
            model_path = f"trained_models/best_model_{safe_target}.joblib"
            
            # Check if model file exists
            if not os.path.exists(model_path):
                return jsonify({'success': False, 'error': f'Model file not found: {model_path}'}), 404
    
    if not model_path:
        return jsonify({'success': False, 'error': 'Model path not found'}), 400
    
    # Prepare metadata
    metadata = {
        'target_variable': training_results.get('target_variable'),
        'model_type': training_results.get('model_type'),
        'accuracy': training_results.get('accuracy'),
        'description': description,
        'title': training_results.get('title', model_name),
        'features': session.get('feature_names', []),
        'feature_importance': training_results.get('feature_importance', {})
    }
    
    # Save model to user's account
    try:
        model_id = user_models.save_user_model(user_id, model_name, model_path, metadata)
        
        if not model_id:
            return jsonify({'success': False, 'error': 'Failed to save model'}), 500
        
        return jsonify({
            'success': True, 
            'message': 'Model saved successfully',
            'model_id': model_id,
            'redirect': url_for('my_models')
        })
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error saving model: {str(e)}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete-model/<model_id>', methods=['POST'])
@login_required
def delete_model(model_id):
    """Delete a model from the user's account."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    # Delete the model
    try:
        success = user_models.delete_user_model(user_id, model_id)
        
        if not success:
            return jsonify({'success': False, 'error': 'Failed to delete model or model not found'}), 404
        
        return jsonify({
            'success': True, 
            'message': 'Model deleted successfully'
        })
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error deleting model: {str(e)}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/load-model/<model_id>')
@login_required
def load_model(model_id):
    """Load a saved model for testing or further use."""
    user_id = session.get('user_id')
    if not user_id:
        flash('Please login to load models', 'error')
        return redirect(url_for('login'))
    
    # Get the model data
    model_data = user_models.get_model_by_id(model_id)
    
    if not model_data:
        flash('Model not found', 'error')
        return redirect(url_for('my_models'))
    
    # Check if user owns this model
    if model_data['user_id'] != user_id:
        flash('You do not have permission to access this model', 'error')
        return redirect(url_for('my_models'))
    
    # Check if model file exists
    if not os.path.exists(model_data['path']):
        flash('Model file not found', 'error')
        return redirect(url_for('my_models'))
    
    # Set model in session
    session['current_model_path'] = model_data['path']
    session['feature_names'] = model_data['metadata'].get('features', [])
    session['model_type'] = model_data['metadata'].get('model_type', 'unknown')
    
    # Create a simplified training_results for consistency with the rest of the app
    training_results = {
        'success': True,
        'accuracy': model_data['metadata'].get('accuracy'),
        'target_variable': model_data['metadata'].get('target_variable'),
        'model_type': model_data['metadata'].get('model_type'),
        'title': model_data['metadata'].get('title', model_data['name']),
        'description': model_data['metadata'].get('description', ''),
        'feature_importance': model_data['metadata'].get('feature_importance', {}),
        'model_path': model_data['path']
    }
    
    session['training_results'] = training_results
    
    # Redirect to model tester
    flash(f'Model "{model_data["name"]}" loaded successfully', 'success')
    return redirect(url_for('model_tester'))

@app.route('/create-embedding/<model_id>', methods=['POST'])
@login_required
def create_embedding(model_id):
    """Create an embeddable version of a model."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    # Get model data
    model_data = user_models.get_model_by_id(model_id)
    
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
        embed_id = user_models.create_model_embedding(model_id, embed_name, embed_settings)
        
        if not embed_id:
            return jsonify({'success': False, 'error': 'Failed to create embedding'}), 500
        
        # Get embed code
        embed_code = user_models.get_embed_code(embed_id)
        
        return jsonify({
            'success': True, 
            'message': 'Embedding created successfully',
            'embed_id': embed_id,
            'embed_code': embed_code,
            'embed_url': url_for('view_embedding', embed_id=embed_id, _external=True)
        })
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error creating embedding: {str(e)}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/delete-embedding/<embed_id>', methods=['POST'])
@login_required
def delete_embedding(embed_id):
    """Delete an embedding."""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'success': False, 'error': 'Not logged in'}), 401
    
    # Get embedding data
    embed_data = user_models.get_embedding_by_id(embed_id)
    
    if not embed_data:
        return jsonify({'success': False, 'error': 'Embedding not found'}), 404
    
    # Check if user owns this embedding
    if embed_data['user_id'] != user_id:
        return jsonify({'success': False, 'error': 'You do not have permission to delete this embedding'}), 403
    
    # Delete the embedding
    try:
        success = user_models.delete_embedding(embed_id)
        
        if not success:
            return jsonify({'success': False, 'error': 'Failed to delete embedding'}), 500
        
        return jsonify({
            'success': True, 
            'message': 'Embedding deleted successfully'
        })
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error deleting embedding: {str(e)}\n{error_trace}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/embedded/<embed_id>')
def view_embedding(embed_id):
    """View an embedded model."""
    # Get embedding data
    embed_data = user_models.get_embedding_by_id(embed_id)
    
    if not embed_data:
        return render_template('error.html', error="Embedding not found", trace="The requested embedding does not exist or has been removed.")
    
    # Check if embedding HTML file exists
    embed_html_path = os.path.join(embed_data['path'], "embed.html")
    
    if not os.path.exists(embed_html_path):
        return render_template('error.html', error="Embedding file not found", trace="The embedding HTML file could not be located.")
    
    # Serve the embedding HTML file
    try:
        with open(embed_html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        return html_content
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error serving embedding: {str(e)}\n{error_trace}")
        return render_template('error.html', error=f"Error serving embedding: {str(e)}", trace=error_trace)

@app.route('/api/predict/<embed_id>', methods=['POST'])
def api_predict(embed_id):
    """API endpoint for making predictions with an embedded model."""
    # Get embedding data
    embed_data = user_models.get_embedding_by_id(embed_id)
    
    if not embed_data:
        return jsonify({'error': 'Embedding not found'}), 404
    
    # Get model path from embedding data
    model_path = os.path.join(embed_data['path'], "model.joblib")
    
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 404
    
    # Get features from embedding metadata or from the features.json file
    features_path = os.path.join(embed_data['path'], "features.json")
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
                    print(f"Error getting prediction probability: {str(e)}")
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
        print(f"Error making prediction: {str(e)}\n{error_trace}")
        return jsonify({'error': str(e)}), 500
#### TRAIN MODEL
@app.route('/train_model/<model_type>', methods=['POST'])
@login_required

def train_model(model_type):
    """Start the model training process"""
    try:
        # Get information from the session
        filename = session.get('last_filename', 'Unknown file')
        proposal_index = int(request.form.get('proposal_index', 0))
        file_path = session.get('file_path')
        proposals = session.get('proposals', [])
        
        # Get the target variable and model type from the selected proposal
        target_variable = None
        claude_model_type = "auto"
        
        if proposals and len(proposals) > proposal_index:
            selected_proposal = proposals[proposal_index]
            target_variable = selected_proposal.get('target_variable')
            claude_model_type = selected_proposal.get('model_type', "auto")
            
            # If target variable is "Unknown", set to None for auto-detection
            if target_variable == "Unknown":
                target_variable = None
        
        
        """
        selected_proposal=proposals[0]
        filename='bank-full.csv'
        model_type='classification'
        proposal_index=0
        target_variable="y"
        file_path="uploads/bank-full.csv"
        
        filename='bank-full.csv'
        model_type='clustering'
        proposal_index=1
        target_variable="y"
        file_path="uploads/bank-full.csv"
        
        
        model_type='regression'
        filename = "online_retail_II_small.xls"
        target_variable = "Quantity"
        file_path="uploads/online_retail_II_small.xls"
        
        file_path="uploads/website_visits.csv"
        target_variable= "duration_seconds"
        model_type='regression'
        
        filename='bank-full.csv'
        model_type='regression'
        proposal_index=0
        target_variable="duration"
        file_path="uploads/bank-full.csv"

        """
       
        
        print(" Call the training function from the ml module")
        training_stats = train_model_with_robust_error_handling(
            file_path, 
            model_type, 
            proposal_index, 
            target_variable
        )
        # selected_proposal=proposal
        training_stats['title']=selected_proposal.get('title')
        training_stats['description']=selected_proposal.get('description')
        training_stats['kpis']=selected_proposal.get('kpis')
        training_stats['business_value']=selected_proposal.get('business_value')
        training_stats['prediction_interpretation']=selected_proposal.get('prediction_interpretation')

        #### GENERATE EXPLANATIONS USING LLM
        llm_explanation=generate_model_explanation(training_stats)
        
        training_stats['llm_explanation']=llm_explanation

        
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
        
        return jsonify({
            'success': True,
            'redirect': url_for('training_results')
        })
        
        # Store training results in session
        session['training_results'] = training_stats
        
        return jsonify({
            'success': True,
            'redirect': url_for('training_results')
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in train_model: {str(e)}\n{error_trace}")
        return jsonify({
            'success': False,
            'error': str(e),
            'redirect': url_for('error_page', error=str(e), trace=error_trace)
        }), 500  # Important: Return JSON even in error case
    
    """
    Route to train a model based on the selected proposal
    
    Args:
        model_type (str): Type of model to train (classification, regression, etc.)
    
    Returns:
        JSON response with training results or error
    """
    try:
        # Get the proposal index from the request
        proposal_index = int(request.form.get('proposal_index', 0))
        
        # Retrieve proposals from session
        proposals = session.get('proposals', [])
        
        if not proposals:
            return jsonify({
                'success': False, 
                'error': 'No proposals found. Please upload a file first.'
            }), 400
        
        # Validate proposal index
        if proposal_index < 0 or proposal_index >= len(proposals):
            return jsonify({
                'success': False, 
                'error': f'Invalid proposal index: {proposal_index}'
            }), 400
        
        # Get the selected proposal
        selected_proposal = proposals[proposal_index]
        
        # Prepare training parameters
        training_params = {
            'model_type': model_type,
            'proposal': selected_proposal,
            'file_path': session.get('file_path')
        }
        
        # Call the training script
        try:
            training_results = run_training_script(training_params)
            
            # Add the proposal details to the training results
            training_results.update({
                'title': selected_proposal.get('title'),
                'description': selected_proposal.get('description'),
                'business_value': selected_proposal.get('business_value'),
                'kpis': selected_proposal.get('kpis', []),
                'target_variable': selected_proposal.get('target_variable')
            })
            
            # Store training results in session
            session['training_results'] = training_results
            
            # Generate model explanation using LLM
            try:
                training_results['llm_explanation'] = generate_model_explanation(training_results)
            except Exception as explanation_error:
                print(f"Error generating model explanation: {str(explanation_error)}")
                training_results['llm_explanation'] = "Unable to generate detailed explanation."
            
            # Redirect to training results page
            return jsonify({
                'success': True,
                'redirect': url_for('training_results')
            })
        
        except Exception as training_error:
            error_trace = traceback.format_exc()
            print(f"Training Error: {str(training_error)}\n{error_trace}")
            
            return jsonify({
                'success': False, 
                'error': str(training_error),
                'trace': error_trace
            }), 500
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Unexpected Error: {str(e)}\n{error_trace}")
        
        return jsonify({
            'success': False, 
            'error': 'An unexpected error occurred during model training',
            'details': str(e)
        }), 500
    """Start the model training process"""
    try:
        # Get information from the session
        filename = session.get('last_filename', 'Unknown file')
        proposal_index = int(request.form.get('proposal_index', 0))
        file_path = session.get('file_path')
        proposals = session.get('proposals', [])
        
        # Get the target variable and model type from the selected proposal
        target_variable = None
        claude_model_type = "auto"
        
        if proposals and len(proposals) > proposal_index:
            selected_proposal = proposals[proposal_index]
            target_variable = selected_proposal.get('target_variable')
            claude_model_type = selected_proposal.get('model_type', "auto")
            
            # If target variable is "Unknown", set to None for auto-detection
            if target_variable == "Unknown":
                target_variable = None
        
        
        """
        filename='bank-full.csv'
        model_type='classification'
        proposal_index=0
        target_variable="y"
        file_path="uploads/bank-full.csv"
        
        filename='bank-full.csv'
        model_type='clustering'
        proposal_index=1
        target_variable="y"
        file_path="uploads/bank-full.csv"
        
        
        model_type='regression'
        filename = "online_retail_II_small.xls"
        target_variable = "Quantity"
        file_path="uploads/online_retail_II_small.xls"
        
        file_path="uploads/website_visits.csv"
        target_variable= "duration_seconds"
        model_type='regression'
        
        filename='bank-full.csv'
        model_type='regression'
        proposal_index=0
        target_variable="duration"
        file_path="uploads/bank-full.csv"

        """
        
        print(" Call the training function from the ml module")
        training_stats = train_model_with_robust_error_handling(
            file_path, 
            model_type, 
            proposal_index, 
            target_variable
        )
        # selected_proposal=proposal
        training_stats['title']=selected_proposal.get('title')
        training_stats['description']=selected_proposal.get('description')
        training_stats['kpis']=selected_proposal.get('kpis')
        training_stats['business_value']=selected_proposal.get('business_value')
        training_stats['prediction_interpretation']=selected_proposal.get('prediction_interpretation')

        #### GENERATE EXPLANATIONS USING LLM
        llm_explanation=generate_model_explanation(training_stats)
        
        training_stats['llm_explanation']=llm_explanation

        
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
        
        return jsonify({
            'success': True,
            'redirect': url_for('training_results')
        })
        
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error in train_model: {str(e)}\n{error_trace}")
        return jsonify({
            'error': str(e),
            'redirect': url_for('error_page', error=str(e), trace=error_trace)
        }), 500
#### MODEL TESTING

def predict_with_feature_alignment(model_path, input_data):
    """
    Make predictions with proper feature alignment for categorical variables
    
    Args:
        model_path (str): Path to the trained model file
        input_data (DataFrame): Input data for prediction
    
    Returns:
        DataFrame: Original data with predictions added
    """
    import joblib
    import pandas as pd
    import numpy as np
    import os
    
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Check if preprocessing pipeline is stored with the model
        model_dir = os.path.dirname(model_path)
        preprocessor_path = os.path.join(model_dir, os.path.basename(model_path).replace('best_model_', 'preprocessor_'))
        
        # Get the expected feature names from model if available
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            print(f"Model expects these features: {expected_features[:5]}... (total: {len(expected_features)})")
        else:
            expected_features = None
            print("Model does not have feature_names_in_ attribute")
        
        # Check for preprocessing pipeline
        if os.path.exists(preprocessor_path):
            print(f"Found preprocessor at {preprocessor_path}")
            preprocessor = joblib.load(preprocessor_path)
            
            # Apply the preprocessor to transform input data correctly
            transformed_data = preprocessor.transform(input_data)
            
            # If transform returns numpy array, convert back to DataFrame with correct column names
            if isinstance(transformed_data, np.ndarray):
                if expected_features is not None and transformed_data.shape[1] == len(expected_features):
                    transformed_df = pd.DataFrame(transformed_data, columns=expected_features)
                else:
                    # Use generic column names if we don't know the feature names
                    transformed_df = pd.DataFrame(transformed_data, 
                                              columns=[f'feature_{i}' for i in range(transformed_data.shape[1])])
            else:
                transformed_df = transformed_data
                
            # Make predictions with properly transformed data
            predictions = model.predict(transformed_df)
            
            # Add predictions to original input
            result_df = input_data.copy()
            result_df['predicted_value'] = predictions
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(transformed_df)
                    if len(probabilities) > 0:
                        if hasattr(model, 'classes_'):
                            # For each prediction, get the probability of the predicted class
                            for i, pred in enumerate(predictions):
                                class_idx = list(model.classes_).index(pred)
                                result_df.loc[result_df.index[i], 'probability'] = probabilities[i][class_idx]
                        else:
                            # If no classes attribute, just take max probability
                            result_df['probability'] = [max(probs) for probs in probabilities]
                except Exception as e:
                    print(f"Error calculating probabilities: {str(e)}")
            
            return result_df
        
        # If no preprocessor found, we need to manually align the features
        elif expected_features is not None:
            print("No preprocessor found. Attempting manual feature alignment...")
            
            # Identify categorical features (those with cat__ prefix)
            cat_features = [f for f in expected_features if f.startswith('cat__')]
            num_features = [f for f in expected_features if not f.startswith('cat__')]
            
            print(f"Model has {len(cat_features)} categorical features and {len(num_features)} numerical features")
            
            # For categorical features, we need to get the base column names
            cat_columns = set()
            for feature in cat_features:
                parts = feature.split('__')
                if len(parts) > 1:
                    col_parts = parts[1].split('_')
                    # The column name is everything before the last underscore in most cases
                    if len(col_parts) > 1:
                        cat_columns.add(col_parts[0])
                    else:
                        cat_columns.add(parts[1])
            
            print(f"Base categorical columns: {cat_columns}")
            
            # Create a new DataFrame to hold the aligned features
            aligned_df = pd.DataFrame(index=input_data.index)
            
            # Process numerical features - these should match by name
            for feature in num_features:
                if feature in input_data.columns:
                    aligned_df[feature] = input_data[feature]
                else:
                    print(f"Missing numerical feature: {feature}, filling with 0")
                    aligned_df[feature] = 0
            
            # Process categorical features using one-hot encoding
            for cat_col in cat_columns:
                if cat_col in input_data.columns:
                    # Get unique values in this column
                    values = input_data[cat_col].astype(str).unique()
                    
                    # Create one-hot columns for each value
                    for value in values:
                        feature_name = f"cat__{cat_col}_{value}"
                        aligned_df[feature_name] = (input_data[cat_col].astype(str) == value).astype(int)
                        
                        # If this feature is not expected, print a warning
                        if feature_name not in expected_features:
                            print(f"Warning: Generated feature {feature_name} not in model's expected features")
                else:
                    print(f"Missing categorical column: {cat_col}")
            
            # Fill in any missing expected features with zeros
            for feature in expected_features:
                if feature not in aligned_df.columns:
                    print(f"Adding missing feature: {feature}")
                    aligned_df[feature] = 0
            
            # Ensure we only have the expected features in the right order
            final_df = aligned_df[expected_features]
            
            # Make predictions
            predictions = model.predict(final_df)
            
            # Add predictions to original input
            result_df = input_data.copy()
            result_df['predicted_value'] = predictions
            
            # Get prediction probabilities if available
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(final_df)
                    if len(probabilities) > 0:
                        if hasattr(model, 'classes_'):
                            # For each prediction, get the probability of the predicted class
                            for i, pred in enumerate(predictions):
                                class_idx = list(model.classes_).index(pred)
                                result_df.loc[result_df.index[i], 'probability'] = probabilities[i][class_idx]
                        else:
                            # If no classes attribute, just take max probability
                            result_df['probability'] = [max(probs) for probs in probabilities]
                except Exception as e:
                    print(f"Error calculating probabilities: {str(e)}")
            
            return result_df
        
        else:
            # No feature information available, try direct prediction (may fail)
            print("No feature information available. Attempting direct prediction.")
            predictions = model.predict(input_data)
            result_df = input_data.copy()
            result_df['predicted_value'] = predictions
            return result_df
            
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Error in prediction with feature alignment: {str(e)}\n{error_trace}")
        
        # Create minimal result dataframe with error message
        result_df = input_data.copy()
        result_df['error'] = str(e)
        return result_df

 

# Update model_tester route to properly handle clustering models
@app.route('/model_tester', methods=['GET'])
@login_required
def model_tester():
    """Display the model testing interface with the currently trained model."""
    
    # Get model info from the training results session
    training_results = session.get('training_results')
    
    if not training_results:
        flash('No trained model found. Please train a model first.', 'error')
        return redirect(url_for('upload_file'))
    
    # Extract model information
    model_type = training_results.get('model_type', 'unknown')
    accuracy = training_results.get('accuracy', 'N/A')
    
    # Format accuracy for display
    if isinstance(accuracy, float):
        accuracy = f"{accuracy * 100:.2f}%" if accuracy < 1 else f"{accuracy:.2f}%"
    
    # Get current model path
    model_path = session.get('current_model_path')
    
    # Initialize variables
    target_variable = 'Unknown'
    display_model_name = 'Model'
    feature_names = []
    
    # Determine model name and target variable based on model type
    if model_path and os.path.exists(model_path):
        model_basename = os.path.basename(model_path)
        
        if model_type == "clustering" or "kmeans" in model_basename:
            # For clustering models
            parts = model_basename.split('_')
            if len(parts) >= 4:
                user = parts[2]
                use_case = parts[3]
                display_model_name = f"Cluster Analysis: {use_case}"
                target_variable = "cluster"  # Clusters don't have a target variable in the traditional sense
            else:
                display_model_name = "Cluster Analysis"
                target_variable = "cluster"
        else:
            # For classification/regression models
            target_variable = model_basename.replace("best_model_", "").replace(".joblib", "")
            display_model_name = target_variable.replace('_', ' ').title()
    else:
        # If no model path is available, try to get information from training results
        target_variable = training_results.get('target_variable', 'Unknown')
        if model_type == "clustering":
            display_model_name = "Cluster Analysis"
            target_variable = "cluster"
        else:
            display_model_name = target_variable.replace('_', ' ').title()
        
        # Find the model file based on model type
        models_dir = "trained_models"
        if model_type == "clustering":
            # Search for clustering model files
            if not os.path.exists(models_dir):
                flash('Models directory not found', 'error')
                return redirect(url_for('upload_file'))
            
            # Find the most recent kmeans model file
            kmeans_models = []
            for filename in os.listdir(models_dir):
                if filename.startswith("kmeans_model_") and filename.endswith(".joblib"):
                    file_path = os.path.join(models_dir, filename)
                    kmeans_models.append((file_path, os.path.getmtime(file_path)))
            
            if not kmeans_models:
                flash('No clustering model found. Please train a clustering model first.', 'error')
                return redirect(url_for('upload_file'))
            
            # Sort by modification time (most recent first)
            kmeans_models.sort(key=lambda x: x[1], reverse=True)
            model_path = kmeans_models[0][0]
        else:
            # For classification/regression models
            model_name = target_variable.replace(' ', '_')
            model_path = os.path.join(models_dir, f"best_model_{model_name}.joblib")
    
    # Verify model exists
    if not model_path or not os.path.exists(model_path):
        flash(f'Model file for "{target_variable}" not found. Please re-train the model.', 'error')
        return redirect(url_for('upload_file'))
    
    # Load the model to get feature information
    try:
        model = joblib.load(model_path)
        feature_names = []
        
        # Try to determine features based on model type
        if model_type == "clustering":
            # For clustering models, try to load preprocessor to get features
            model_dir = os.path.dirname(model_path)
            basename = os.path.basename(model_path)
            parts = basename.split('_')
            
            # Look for preprocessor
            preprocessor_path = os.path.join(model_dir, basename.replace("kmeans_model_", "preprocessor_"))
            if not preprocessor_path.endswith(".joblib"):
                preprocessor_path = preprocessor_path.replace(f"_{parts[-2]}_{parts[-1]}", "_kmeans.joblib")
            
            # Try alternate paths if not found
            if not os.path.exists(preprocessor_path):
                preprocessor_path = model_path.replace("kmeans_model_", "preprocessor_").replace(
                    f"_{parts[-2]}_{parts[-1]}", f"_kmeans.joblib")
            
            if os.path.exists(preprocessor_path):
                preprocessor = joblib.load(preprocessor_path)
                
                # Try to determine feature names from preprocessor
                if hasattr(preprocessor, 'feature_names_in_'):
                    feature_names = preprocessor.feature_names_in_
                elif hasattr(preprocessor, 'get_feature_names_out'):
                    feature_names = preprocessor.get_feature_names_out()
                else:
                    # Try to find features file
                    features_path = model_path.replace("kmeans_model_", "model_features_").replace(
                        f"_{parts[-2]}_{parts[-1]}", f"_kmeans.joblib")
                    
                    if os.path.exists(features_path):
                        loaded_features = joblib.load(features_path)
                        if loaded_features is not None:
                            if hasattr(loaded_features, '__iter__') and not isinstance(loaded_features, str):
                                feature_names = [str(f) for f in loaded_features]
                            else:
                                feature_names = [str(loaded_features)]
            
            # If still no features, use generic placeholders
            if len(feature_names) == 0:
                feature_names = ['feature1', 'feature2', 'feature3']
        else:
            # For classification/regression models, use existing feature loading logic
            features_filename = os.path.join(os.path.dirname(model_path), f"model_features_{target_variable.replace(' ', '_')}.joblib")
            
            if os.path.exists(features_filename):
                loaded_features = joblib.load(features_filename)
                
                if loaded_features is not None:
                    if hasattr(loaded_features, '__iter__') and not isinstance(loaded_features, str):
                        feature_names = [str(f) for f in loaded_features]
                    else:
                        feature_names = [str(loaded_features)]
        
        # If no feature names found, try model attributes
        if len(feature_names) == 0:
            if hasattr(model, 'feature_names_in_'):
                model_features = model.feature_names_in_
                # Convert to list of strings safely
                if model_features is not None:
                    if hasattr(model_features, '__iter__') and not isinstance(model_features, str):
                        feature_names = [str(f) for f in model_features]
                    else:
                        feature_names = [str(model_features)]
            elif hasattr(model, 'feature_names_'):
                model_features = model.feature_names_
                # Convert to list of strings safely
                if model_features is not None:
                    if hasattr(model_features, '__iter__') and not isinstance(model_features, str):
                        feature_names = [str(f) for f in model_features]
                    else:
                        feature_names = [str(model_features)]
        
        # Fall back to session features if still empty
        if len(feature_names) == 0:
            session_features = session.get('feature_names', [])
            if session_features is not None and (isinstance(session_features, list) or hasattr(session_features, '__len__')):
                if len(session_features) > 0:
                    if hasattr(session_features, '__iter__') and not isinstance(session_features, str):
                        feature_names = [str(f) for f in session_features]
                    else:
                        feature_names = [str(session_features)]
        
        # Ensure we have at least some placeholder features
        if len(feature_names) == 0:
            feature_names = ["feature1", "feature2"]
            flash("No feature information found. Using placeholder features.", 'warning')
            
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error loading model or features: {str(e)}\n{error_trace}")
        flash(f"Error loading model or features: {str(e)}", 'error')
        return redirect(url_for('upload_file'))
    
    # Convert NumPy arrays to Python lists for template safety
    if hasattr(feature_names, 'tolist'):
        # NumPy array
        feature_names = feature_names.tolist()
    elif hasattr(feature_names, '__iter__') and not isinstance(feature_names, (list, str)):
        # Other iterable
        feature_names = list(feature_names)
    
    # Handle scalar
    if not isinstance(feature_names, list):
        feature_names = [feature_names]
    
    # Store model path and features in session for prediction routes
    session['current_model_path'] = model_path
    session['feature_names'] = feature_names
    session['model_type'] = model_type
    
    return render_template('model_tester.html',
                          model_path=model_path,
                          model_name=display_model_name,
                          target_variable=target_variable,
                          model_type=model_type,
                          accuracy=str(accuracy),
                          required_features=feature_names,
                          prediction_results=None)

# Update the test_model_manual function to pass user_id and use_case
@app.route('/test_model_manual', methods=['POST'])
@login_required
def test_model_manual():
    """Test the model using manually entered data."""
    # Get model path from form or session
    model_path = request.form.get('model_path') or session.get('current_model_path')
    
    if not model_path or not os.path.exists(model_path):
        flash('Model not found. Please train a model first.', 'error')
        return redirect(url_for('model_tester'))
    
    try:
        # Extract model info
        model_basename = os.path.basename(model_path)
        model_type = session.get('model_type', 'unknown')
        
        # Set display info differently based on model type
        if "kmeans" in model_basename or model_type == "clustering":
            # For clustering models
            parts = model_basename.split('_')
            if len(parts) >= 4:
                user = parts[2]
                use_case = parts[3]
                display_model_name = f"Cluster Analysis: {use_case}"
                target_variable = "cluster"  # Clusters don't have a target variable in the traditional sense
            else:
                display_model_name = "Cluster Analysis"
                target_variable = "cluster"
        else:
            # For classification/regression models
            target_variable = model_basename.replace("best_model_", "").replace(".joblib", "")
            display_model_name = target_variable.replace('_', ' ').title()
        
        # Get accuracy from training results
        training_results = session.get('training_results', {})
        accuracy = training_results.get('accuracy', 'N/A')
        
        # Get user ID from session
        user_id = session.get('user_id')
        
        # Get use case from training results if available
        use_case = None
        if training_results and 'title' in training_results:
            use_case = training_results.get('title').replace(' ', '_').lower()
        
        # Collect input values from form
        input_data = {}
        for key, value in request.form.items():
            if key.startswith('feature_'):
                feature_name = key.replace('feature_', '')
                
                # Try to convert to numeric if possible
                try:
                    input_data[feature_name] = int(value)
                except ValueError:
                    try:
                        input_data[feature_name] = float(value)
                    except ValueError:
                        input_data[feature_name] = value
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make predictions based on model type
        if model_type == "classification":
            from ml.predict_model_clasification import predict_classification
            prediction_df = predict_classification(model_path, input_df)
        elif model_type == "regression":
            from ml.predict_model_regression import predict_regression
            prediction_df = predict_regression(model_path, input_df)
        elif model_type == "clustering":
            from ml.predict_model_clustering import predict_clustering
            # Pass user_id and use_case to the prediction function
            prediction_df = predict_clustering(model_path, input_df, user_id=user_id, use_case=use_case)
        else:
            # Try feature-aligned prediction as fallback
            prediction_df = predict_with_feature_alignment(model_path, input_df)
        
        # Current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get feature names for display
        feature_names = list(input_data.keys())
        
        # Return results to template
        return render_template('model_tester.html',
                             model_path=model_path,
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
        print(f"Prediction error: {str(e)}\n{error_trace}")
        return redirect(url_for('model_tester'))

# Update the test_model_with_file function similarly
@app.route('/test_model_with_file', methods=['POST'])
@login_required
def test_model_with_file():
    """Test the model using uploaded file data."""
    # Get model path from form or session
    model_path = request.form.get('model_path') or session.get('current_model_path')
    # Get user_id from session
    user_id = session.get('user_id', 'default_user')
     # Get use_case from training results or create a default
    training_results = session.get('training_results', {})
    use_case = training_results.get('title', 'default_use_case')
      
    if not model_path or not os.path.exists(model_path):
        flash('Model not found. Please train a model first.', 'error')
        return redirect(url_for('model_tester'))
    
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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_' + filename)
        file.save(filepath)
        
        # Extract model info
        model_basename = os.path.basename(model_path)
        model_type = session.get('model_type', 'unknown')
        
        # Set display info differently based on model type
        if "kmeans" in model_basename or model_type == "clustering":
            # For clustering models
            parts = model_basename.split('_')
            if len(parts) >= 4:
                user = parts[2]
                use_case = parts[3]
                display_model_name = f"Cluster Analysis: {use_case}"
                target_variable = "cluster"  # Clusters don't have a target variable in the traditional sense
            else:
                display_model_name = "Cluster Analysis"
                target_variable = "cluster"
        else:
            # For classification/regression models
            target_variable = model_basename.replace("best_model_", "").replace(".joblib", "")
            display_model_name = target_variable.replace('_', ' ').title()
        
        # Get model info from session
        training_results = session.get('training_results', {})
        accuracy = training_results.get('accuracy', 'N/A')
        
        # Get user ID from session
        user_id = session.get('user_id')
        
        # Get use case from training results if available
        use_case = None
        if training_results and 'title' in training_results:
            use_case = training_results.get('title').replace(' ', '_').lower()
        
        # Read data
        from utils.read_file import read_data_flexible
        input_data = read_data_flexible(filepath)
        
        # Make predictions based on model type
        if model_type == "classification":
            from ml.predict_model_classification import predict_classification
            prediction_df = predict_classification(model_path, input_data, user_id, use_case, target_variable)
        elif model_type == "regression":
            from ml.predict_model_regression import predict_regression
            prediction_df = predict_regression(model_path, input_data)
        elif model_type == "clustering":
            from ml.predict_model_clustering import predict_clustering
            # Pass user_id and use_case to the prediction function
            prediction_df = predict_clustering(model_path, input_data, user_id=user_id, use_case=use_case)
        else:
            # Try feature-aligned prediction as fallback
            prediction_df = predict_with_feature_alignment(model_path, input_data)
        
        # Current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Clean up temp file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Get feature names for display (convert to list for template safety)
        feature_names = list(input_data.columns)
        
        # Return results to template
        return render_template('model_tester.html',
                             model_path=model_path,
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
        print(f"Prediction error: {str(e)}\n{error_trace}")
        # Clean up temp file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return redirect(url_for('model_tester'))
    """Test the model using uploaded file data."""
    # Get model path from form or session
    model_path = request.form.get('model_path') or session.get('current_model_path')
    
    if not model_path or not os.path.exists(model_path):
        flash('Model not found. Please train a model first.', 'error')
        return redirect(url_for('model_tester'))
    
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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_' + filename)
        file.save(filepath)
        
        # Extract model info
        model_basename = os.path.basename(model_path)
        model_type = session.get('model_type', 'unknown')
        
        # Set display info differently based on model type
        if "kmeans" in model_basename or model_type == "clustering":
            # For clustering models
            parts = model_basename.split('_')
            if len(parts) >= 4:
                user = parts[2]
                use_case = parts[3]
                display_model_name = f"Cluster Analysis: {use_case}"
                target_variable = "cluster"  # Clusters don't have a target variable in the traditional sense
            else:
                display_model_name = "Cluster Analysis"
                target_variable = "cluster"
        else:
            # For classification/regression models
            target_variable = model_basename.replace("best_model_", "").replace(".joblib", "")
            display_model_name = target_variable.replace('_', ' ').title()
        
        # Get model info from session
        training_results = session.get('training_results', {})
        accuracy = training_results.get('accuracy', 'N/A')
        
        # Read data
        from utils.read_file import read_data_flexible
        input_data = read_data_flexible(filepath)
        
        # Make predictions based on model type
        if model_type == "classification":
            from ml.predict_model_clasification import predict_classification
            prediction_df = predict_classification(model_path, input_data)
        elif model_type == "regression":
            from ml.predict_model_regression import predict_regression
            prediction_df = predict_regression(model_path, input_data)
        elif model_type == "clustering":
            from ml.predict_model_clustering import predict_clustering
            prediction_df = predict_clustering(model_path, input_data)
        else:
            # Try feature-aligned prediction as fallback
            prediction_df = predict_with_feature_alignment(model_path, input_data)
        
        # Current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Clean up temp file
        if os.path.exists(filepath):
            os.remove(filepath)
        
        # Get feature names for display (convert to list for template safety)
        feature_names = list(input_data.columns)
        
        # Return results to template
        return render_template('model_tester.html',
                             model_path=model_path,
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
        print(f"Prediction error: {str(e)}\n{error_trace}")
        # Clean up temp file
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except:
                pass
        return redirect(url_for('model_tester'))
    """Test the model using manually entered data."""
    # Get model path from form or session
    model_path = request.form.get('model_path') or session.get('current_model_path')
    
    if not model_path or not os.path.exists(model_path):
        flash('Model not found. Please train a model first.', 'error')
        return redirect(url_for('model_tester'))
    
    try:
        # Extract model info
        model_basename = os.path.basename(model_path)
        model_type = session.get('model_type', 'unknown')
        
        # Set display info differently based on model type
        if "kmeans" in model_basename or model_type == "clustering":
            # For clustering models
            parts = model_basename.split('_')
            if len(parts) >= 4:
                user = parts[2]
                use_case = parts[3]
                display_model_name = f"Cluster Analysis: {use_case}"
                target_variable = "cluster"  # Clusters don't have a target variable in the traditional sense
            else:
                display_model_name = "Cluster Analysis"
                target_variable = "cluster"
        else:
            # For classification/regression models
            target_variable = model_basename.replace("best_model_", "").replace(".joblib", "")
            display_model_name = target_variable.replace('_', ' ').title()
        
        # Get accuracy from training results
        training_results = session.get('training_results', {})
        accuracy = training_results.get('accuracy', 'N/A')
        
        # Collect input values from form
        input_data = {}
        for key, value in request.form.items():
            if key.startswith('feature_'):
                feature_name = key.replace('feature_', '')
                
                # Try to convert to numeric if possible
                try:
                    input_data[feature_name] = int(value)
                except ValueError:
                    try:
                        input_data[feature_name] = float(value)
                    except ValueError:
                        input_data[feature_name] = value
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Make predictions based on model type
        if model_type == "classification":
            from ml.predict_model_clasification import predict_classification
            prediction_df = predict_classification(model_path, input_df)
        elif model_type == "regression":
            from ml.predict_model_regression import predict_regression
            prediction_df = predict_regression(model_path, input_df)
        elif model_type == "clustering":
            from ml.predict_model_clustering import predict_clustering
            prediction_df = predict_clustering(model_path, input_df)
        else:
            # Try feature-aligned prediction as fallback
            prediction_df = predict_with_feature_alignment(model_path, input_df)
        
        # Current timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Get feature names for display
        feature_names = list(input_data.keys())
        
        # Return results to template
        return render_template('model_tester.html',
                             model_path=model_path,
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
        print(f"Prediction error: {str(e)}\n{error_trace}")
        return redirect(url_for('model_tester'))

#### USE CASE MANAGEMENT
@app.route('/saved-use-cases')
@login_required
def saved_use_cases():
    """Display saved use cases for the current user"""
    user_id = session.get('user_id')
    
    # Get saved use cases
    use_cases = user_models.get_user_use_cases(user_id)
    
    return render_template('saved_use_cases.html', 
                           use_cases=use_cases,
                           user_name=session.get('user_name', 'User'))



@app.route('/view-use-case/<use_case_id>')
@login_required
def view_use_case(use_case_id):
    """View details of a specific saved use case"""
    user_id = session.get('user_id')
    
    # Find the use case file
    use_cases_dir = os.path.join(Config.DATABASE_DIR, 'use_cases', user_id)
    use_case_path = os.path.join(use_cases_dir, f"{use_case_id}.json")
    
    if not os.path.exists(use_case_path):
        flash('Use case not found', 'error')
        return redirect(url_for('saved_use_cases'))
    
    # Read the use case
    try:
        with open(use_case_path, 'r') as f:
            use_case = json.load(f)
    except Exception as e:
        flash(f'Error reading use case: {str(e)}', 'error')
        return redirect(url_for('saved_use_cases'))
    
    # Extract essential information
    filename = use_case.get('filename', 'Unknown File')
    proposals = use_case.get('proposals', [])
    metadata = use_case.get('metadata', {})
    
    # Prepare session variables
    session['last_filename'] = filename
    session['proposal_count'] = len(proposals)
    session['proposals'] = proposals
    
    # Handle file path reconstruction
    temp_file_path = None
    
    try:
        # First, try preserved file path
        if 'preserved_file_path' in metadata and os.path.exists(metadata['preserved_file_path']):
            temp_file_path = metadata['preserved_file_path']
        
        # If no preserved path, try original file path
        elif 'file_path' in metadata and os.path.exists(metadata['file_path']):
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
                
                # Attempt to extract feature names from the proposal
                features = []
                if 'description' in first_proposal:
                    # Try to extract potential feature names from description
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
        print(f"Error creating temporary file: {str(e)}")
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
def delete_use_case(use_case_id):
    """Delete a specific use case"""
    user_id = session.get('user_id')
    
    try:
        success = user_models.delete_use_case(user_id, use_case_id)
        
        if success:
            flash('Use case deleted successfully', 'success')
        else:
            flash('Failed to delete use case', 'error')
        
        return redirect(url_for('saved_use_cases'))
    
    except Exception as e:
        flash(f'Error deleting use case: {str(e)}', 'error')
        return redirect(url_for('saved_use_cases'))
    

def save_use_cases(user_id, filename, proposals, metadata=None):
    """
    Save AI use case proposals for a user
    
    Args:
        user_id (str): ID of the user
        filename (str): Original filename
        proposals (list): List of use case proposal dictionaries
        metadata (dict, optional): Additional metadata about the file/proposals
    
    Returns:
        str: ID of the saved use cases
    """
    from datetime import datetime
    import os
    import json
    import uuid
    import shutil
    
    # Ensure database directories exist
    use_cases_dir = os.path.join(Config.DATABASE_DIR, 'use_cases', user_id)
    os.makedirs(use_cases_dir, exist_ok=True)
    
    # Generate a unique ID for this set of use cases
    use_case_id = str(uuid.uuid4())
    
    # Prepare metadata with defaults
    if metadata is None:
        metadata = {}
    
    # If a file path is provided, try to preserve the file
    original_file_path = metadata.get('file_path')
    preserved_file_path = None
    
    if original_file_path and os.path.exists(original_file_path):
        # Create a directory to store the original file
        file_storage_dir = os.path.join(use_cases_dir, use_case_id)
        os.makedirs(file_storage_dir, exist_ok=True)
        
        # Copy the original file to the use case directory
        original_filename = os.path.basename(original_file_path)
        preserved_file_path = os.path.join(file_storage_dir, original_filename)
        
        try:
            shutil.copy2(original_file_path, preserved_file_path)
            # Update metadata with the new file path
            metadata['preserved_file_path'] = preserved_file_path
        except Exception as e:
            print(f"Error preserving file: {e}")
    
    # Prepare use case entry
    use_case_entry = {
        'id': use_case_id,
        'user_id': user_id,
        'filename': filename,
        'proposals': proposals,
        'created_at': datetime.now().isoformat(),
        'metadata': metadata or {}
    }
    
    # Save the use cases to a JSON file
    use_case_path = os.path.join(use_cases_dir, f"{use_case_id}.json")
    with open(use_case_path, 'w') as f:
        json.dump(use_case_entry, f, indent=2)
    
    return use_case_id
# Custom datetime formatting filter
from datetime import datetime

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


# Add this after the existing template filters
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

# Add this route to your app.py or routes.py file

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
    stats = {
        'model_count': 0,
        'use_case_count': 0,
        'embedding_count': 0,
        'accuracy': 'N/A'
    }
    
    # Query the database for actual statistics
    
    
    return render_template('home.html', stats=stats)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)