# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 18:47:21 2025

@author: joze_
"""
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)   
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
from flask_session import Session

import os
import re
import json
import requests
import time
import random
import traceback
import pandas as pd
from werkzeug.utils import secure_filename
# Import config for API keys
from config import Config
from ml_trainer import train_model_with_robust_error_handling


filepath='data_sets/bank-full.csv'
filename='bank-full.csv'
# Import Google's Generative AI library
try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False
    print("Warning: Google Generative AI library not found. Gemini API will not be available.")

# Initialize Flask app
app = Flask(__name__, static_folder='static')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'txt', 'pdf', 'csv', 'docx','xlsx','xsls','data'}
app.secret_key = Config.SECRET_KEY  # Use the key from config
app.config['SESSION_COOKIE_MAX_SIZE'] = 4093  # Set max cookie size
app.config['SESSION_COOKIE_SECURE'] = False  # For local development
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Session configuration
app.config["SESSION_TYPE"] = "filesystem"  # or "redis" if you have Redis
app.config["SESSION_FILE_DIR"] = "/tmp/flask_session"  # for filesystem storage
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_USE_SIGNER"] = True
Session(app)

# Get active model from config
ACTIVE_MODEL = getattr(Config, 'ACTIVE_MODEL', 'claude').lower()
if ACTIVE_MODEL not in ['claude', 'gemini']:
    print(f"Warning: Unknown model '{ACTIVE_MODEL}' specified. Defaulting to Claude.")
    ACTIVE_MODEL = 'claude'

# Claude API configuration
CLAUDE_API_KEY = Config.CLAUDE_API_KEY
CLAUDE_API_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODEL = getattr(Config, 'CLAUDE_MODEL', "claude-3-sonnet-20240229")

# Gemini API configuration
GOOGLE_API_KEY = getattr(Config, 'GOOGLE_API_KEY', None)
GEMINI_MODEL_NAME = getattr(Config, 'GEMINI_MODEL', "gemini-2.0-pro-exp")

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
    
# Limit the size of proposals in the session
def limit_session_size(session, max_proposals=4):
    """Limit the number of proposals stored in the session."""
    if 'proposals' in session and len(session['proposals']) > max_proposals:
        session['proposals'] = session['proposals'][:max_proposals]
        session.modified = True

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

 
# 
# selected_proposal=proposals[0]
import os

def generate_ml_with_llm(selected_proposal, file_content,filename):
    """
    Generate machine learning code based on the selected proposal and file content,
    and save it to a file in the models directory.
    
    Args:
        selected_proposal (dict): A dictionary containing proposal details
        file_content (str): Content of the uploaded file
        
    Returns:
        str: Path to the saved Python file
    """
    target_variable = selected_proposal.get('target_variable')
    description = selected_proposal.get('description')
    
    prompt_text = f"""Based on the uploaded file content below:
        1. create python code that will train the ML model for target variable:{target_variable} the Use Case:{description}  
            Use multiple ML algorytms and choose the best performing one
            do feature optimization and combination if you think it will improve the model
        2. for data input use DATA_FILE_PATH="../uploads/{filename}"
        3. Save the trained model in trained_models folder
        4. dont add additional explanations or descriptions that might break the model.
        5. check if the script is executable 
        6. Save the statistics of the model in Accuracy variable
        7. while reading the file ensure there are no errors in the encodings
        8. prepare script for running the best model on other data
            
        FILE CONTENT:
        {file_content[:2000]}  # Limit content length to avoid token limits
        return only the python code.
        """
    print('generating code')

    # Use appropriate model - assuming you have the model imported and initialized
    response = gemini_model.generate_content(prompt_text)
    
    # Extract the code from the response
    generated_code = response.text  # Adjust based on the actual response format
    # Remove markdown code block formatting if present
    generated_code = re.sub(r'^```python\s*', '', generated_code)
    generated_code = re.sub(r'\s*```$', '', generated_code)
    print(generated_code)
    
    # exec(generated_code)
    # Create models directory if it doesn't exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create a filename based on the target variable
    safe_target = target_variable.replace(" ", "_").lower()
    filename = f"{models_dir}/{safe_target}_model.py"
    
    # Save the generated code to the file
    with open(filename, "w") as 
        f.write(generated_code)
    
    print(f"ML code saved to {filename}")
    return filename

# Example usage:
# model_file = generate_ml_with_llm(selected_proposal, file_content,filename)
# If you want to execute it immediately:
# import subprocess
# subprocess.run(["python", model_file])
        


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
  "prediction_interpretation": "comprehensive explanation how to interpret the AI prediction with examples."
}'''
    
    # Base prompt for all models
    prompt_text = f"""Based on the uploaded file content below, please generate comprehensive AI use case proposals. Do not limit to only 4 use cases - provide as many distinct use cases as you can identify from the data.

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
                # if len(file_content) > 100000:
                #     file_content = file_content[:100000] + "\n\n[Content truncated due to size limitations]"
                
                
                # Get AI use case proposals using the active model from config
                try:
                    ai_proposals = get_ai_use_case_proposals(file_content, filename)
                    
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

 

"""
proposal=[{'title': 'Predicting Term Deposit Subscription Likelihood',
  'description': 'This use case focuses on predicting whether a bank client will subscribe to a term deposit product based on their demographic information, financial status, and previous marketing campaign interactions. By analyzing historical data (age, job, marital status, education, balance, existing loans, contact history, etc.), an AI model can learn patterns associated with successful subscriptions.\n\nThe model will generate a probability score for each customer, indicating their likelihood to subscribe during a future marketing campaign. This allows the bank to prioritize contacting customers with higher probabilities, optimizing resource allocation for marketing teams (e.g., call center agents, email marketers) and potentially increasing the overall campaign conversion rate.\n\nFurthermore, analyzing feature importance within the model can reveal key drivers of subscription decisions (e.g., certain job types, specific months of contact, previous campaign success), providing valuable insights for refining marketing strategies and product offerings.',
  'kpis': ['Campaign Conversion Rate : Increase in the percentage of contacted customers who subscribe to the term deposit.',
   'Marketing Cost Per Acquisition (CPA) : Reduction in the cost associated with acquiring a new term deposit subscriber.',
   "Customer List Prioritization Efficiency : Improvement in the ranking of leads, measured by the concentration of positive outcomes ('yes') in the top deciles/percentiles of the prediction list.",
   "Model Accuracy/AUC : Technical measure of the model's ability to distinguish between subscribers and non-subscribers."],
  'business_value': 'Improves marketing campaign efficiency by focusing efforts on high-propensity customers, leading to higher conversion rates, reduced marketing spend, and increased overall profitability from term deposit products. It also provides insights into customer behavior for strategic planning.',
  'target_variable': 'y',
  'model_type': 'classification',
  'use_case_implementation_complexity': 'medium',
  'prediction_interpretation': "The model outputs a probability score (e.g., 0.0 to 1.0) or a class label ('yes'/'no') for each customer. A higher score (e.g., > 0.75) indicates a high predicted likelihood of subscribing to the term deposit. For example, a customer with a prediction of 0.85 is considered much more likely to subscribe than a customer with a score of 0.15. Marketing teams can set a threshold (e.g., contact customers with scores > 0.6) to prioritize their outreach efforts."},
 {'title': 'Customer Segmentation for Personalized Marketing',
  'description': "This use case involves applying AI-driven clustering techniques to group customers into distinct segments based on shared characteristics found in the data. Instead of predicting a specific outcome, the goal is to uncover natural groupings within the customer base using variables like age, job, education, balance, housing/loan status, and potentially past campaign behavior (excluding the target 'y').\n\nOnce segments are identified (e.g., 'High-Balance Professionals', 'Young Renters with Personal Loans', 'Older Retired Homeowners'), the bank can analyze the characteristics and needs of each segment. This understanding allows for the development of highly targeted marketing messages, product recommendations, and communication strategies tailored to the specific preferences and financial situations of each group.\n\nFor example, one segment might respond better to digital communication about investment products, while another might prefer phone calls regarding loan consolidation options. This personalization can significantly improve customer engagement and marketing effectiveness.",
  'kpis': ['Segment Engagement Rate : Increase in interaction rates (e.g., click-through rates, response rates) for campaigns targeted at specific segments.',
   'Cross-Sell/Up-Sell Rate within Segments : Increase in the adoption of additional products by customers within identified segments.',
   'Customer Satisfaction Scores (Segment-Specific) : Improvement in satisfaction metrics for targeted segments.',
   'Marketing ROI per Segment : Measuring the return on investment for marketing campaigns tailored to different customer groups.'],
  'business_value': 'Enables personalized marketing strategies, leading to improved customer engagement, higher campaign response rates, increased cross-selling and up-selling opportunities, and enhanced customer loyalty by making customers feel better understood.',
  'target_variable': 'N/A (Unsupervised Learning)',
  'model_type': 'clustering',
  'use_case_implementation_complexity': 'medium',
  'prediction_interpretation': "The model assigns each customer to a specific cluster (e.g., Cluster 1, Cluster 2, etc.). Interpretation involves analyzing the defining characteristics of each cluster by examining the average or modal values of the input features (age, job, balance, etc.) for the customers within that cluster. For example, Cluster 3 might be characterized by customers who are typically 'management' or 'technician' job types, have 'tertiary' education, higher 'balance', and no existing 'loan'. This profile suggests a segment receptive to investment or premium banking services."},
 {'title': 'Predicting Existing Credit Default Status',
  'description': "This use case aims to predict whether a customer currently has credit in default ('yes' or 'no' in the 'default' column) based on their other available information, such as demographics, job type, balance, and other loan statuses. While this column represents past behavior, building a model to predict it can be useful for several reasons.\n\nFirstly, it can help identify potential data inconsistencies or serve as a proxy variable in situations where default status might not be readily available for a subset of customers. Secondly, understanding the factors strongly correlated with having credit in default (e.g., specific job categories, low balance coupled with other loans) provides valuable insights for risk assessment frameworks, even if it's not predicting *future* default.\n\nThe model could flag customer profiles that strongly resemble those historically marked as having credit in default, potentially triggering reviews or informing decisions about future creditworthiness assessments, although caution is needed as it reflects past, not future, status.",
  'kpis': ["Model Accuracy/Precision/Recall for Default : Measuring the model's ability to correctly identify customers with existing credit defaults.",
   'Feature Importance Analysis : Identification of key factors strongly associated with existing default status.',
   'Risk Profile Identification Accuracy : How well the model flags profiles matching known high-risk characteristics associated with default.'],
  'business_value': 'Provides insights into the characteristics associated with existing credit default, potentially helping to refine risk assessment profiles and identify data patterns related to credit risk. It can also act as a data validation or imputation tool in specific scenarios.',
  'target_variable': 'default',
  'model_type': 'classification',
  'use_case_implementation_complexity': 'medium',
  'prediction_interpretation': "The model predicts whether a customer is likely to have the 'default' flag set to 'yes' or 'no'. A prediction of 'yes' suggests the customer's profile (based on job, balance, age, etc.) strongly matches the profiles of other customers in the dataset who have credit in default. For instance, if the model predicts 'yes' for a customer with 90% confidence, it indicates a high statistical similarity to known defaulted customers based on the learned patterns. This doesn't predict they *will* default in the future, but that their current profile aligns with past defaults."},
 {'title': 'Optimizing Marketing Campaign Contact Strategy',
  'description': "This use case focuses on leveraging AI to determine the most effective way to contact different customer segments to maximize the likelihood of a positive outcome (term deposit subscription, 'y' = 'yes'). It analyzes the relationship between the target variable ('y') and various campaign-related features like contact method ('contact'), time of contact ('month', 'day'), and frequency ('campaign'). Note: 'duration' is excluded for pre-call prediction but can be analyzed post-campaign.\n\nThe model aims to predict the success probability ('y' = 'yes') under different hypothetical contact scenarios (e.g., contacting customer X via 'cellular' in 'aug' vs. 'telephone' in 'may'). By understanding which channels and timings work best for different customer profiles (potentially derived from segmentation or individual features like age, job, balance), the bank can create more effective, data-driven campaign plans.\n\nFor instance, the model might reveal that younger customers with high balances respond best to cellular calls in the later months of the year, while older, retired customers are more receptive to telephone calls earlier in the year. This allows for dynamic adjustment of campaign tactics.",
  'kpis': ['Contact Channel Effectiveness : Improvement in conversion rates attributed to using the AI-recommended contact channel.',
   'Campaign Timing Optimization : Increase in success rates achieved by contacting customers during AI-recommended months or days.',
   'Resource Allocation Efficiency : Better allocation of call center resources (cellular vs. telephone) based on predicted effectiveness.',
   "Overall Campaign Uplift : Increase in the overall success rate ('y' = 'yes') compared to baseline or non-optimized campaigns."],
  'business_value': 'Optimizes the use of different communication channels and timing for marketing campaigns, leading to higher success rates, improved resource utilization (e.g., call center staff, communication budgets), and a better customer experience through more relevant contact methods.',
  'target_variable': 'y',
  'model_type': 'classification',
  'use_case_implementation_complexity': 'medium',
  'prediction_interpretation': "The model predicts the probability of success ('y' = 'yes') given a customer's profile and specific contact strategy features (channel, month). To use it for optimization, you could simulate predictions for the same customer using different contact methods or months. For example, for Customer A, the model might predict P(y='yes' | contact='cellular', month='aug') = 0.65, while P(y='yes' | contact='telephone', month='may') = 0.30. This suggests using the cellular channel in August is the better strategy for this specific customer or segment."},
 {'title': 'Customer Balance Level Prediction',
  'description': "This use case involves building a regression model to predict a customer's likely account balance based on their demographic and employment information (age, job, marital status, education). This predictive capability can be valuable for various banking operations and strategic planning.\n\nUnderstanding the potential balance level associated with different customer profiles can help in tailoring product offerings, identifying potential high-value customers early on, and informing financial advisory services. For instance, predicting a high balance for a new customer with a 'management' job and 'tertiary' education might trigger an offer for premium banking services.\n\nFurthermore, comparing predicted balance with actual balance over time could potentially signal changes in a customer's financial situation, although this requires longitudinal data. The model provides a quantitative estimate of expected balance based on profile characteristics.",
  'kpis': ['Mean Absolute Error (MAE) / Root Mean Squared Error (RMSE) : Technical metrics measuring the average difference between predicted and actual balances.',
   'Accuracy in Predicting Balance Bands : Percentage of customers correctly classified into predefined balance categories (e.g., Low, Medium, High) based on the predicted value.',
   "High-Value Customer Identification Rate : Success rate in identifying customers who actually have high balances using the model's predictions.",
   'Correlation between Predicted Balance and Product Uptake : Assessing if customers predicted to have higher balances are more likely to adopt premium products.'],
  'business_value': 'Provides insights into the expected financial capacity of different customer segments, aids in identifying potential high-value customers, supports personalized product recommendations (e.g., investment vs. basic accounts), and can inform financial planning and advisory services.',
  'target_variable': 'balance',
  'model_type': 'regression',
  'use_case_implementation_complexity': 'hard',
  'prediction_interpretation': "The model outputs a continuous numerical value representing the predicted account balance for a customer. For example, for a 45-year-old 'management' professional with 'tertiary' education, the model might predict a balance of 8500. This prediction should be interpreted as an estimate based on the patterns learned from similar profiles in the dataset. It can be used directionally (e.g., this profile likely has a higher balance than a 'blue-collar' profile) or compared against thresholds to categorize customers (e.g., predicted balance > 10000 triggers a 'High Potential Value' flag)."},
 {'title': 'Identifying Profiles Similar to Housing Loan Holders',
  'description': "This use case utilizes AI to identify customers whose profiles closely resemble those who currently have a housing loan ('housing' = 'yes'). By training a classification model on demographic, financial, and employment data, the bank can predict the likelihood that a given customer (even one currently without a housing loan) fits the typical profile of a housing loan holder.\n\nWhile the model predicts the 'housing' status based on existing data, its primary value lies in identifying potential prospects for future housing-related products or services. Customers flagged by the model as having a high probability of fitting the 'housing loan holder' profile, but who don't currently have 'housing' = 'yes', might be good candidates for marketing campaigns related to mortgages, home equity lines of credit, or refinancing offers.\n\nUnderstanding the key features that differentiate housing loan holders (e.g., marital status, job stability inferred from job type, age group) also provides valuable market insights.",
  'kpis': ["Model Accuracy/AUC in Predicting 'housing' Status : Technical measure of the model's ability to identify current housing loan holders.",
   "Prospect Identification Rate : Percentage of customers identified by the model (predicted 'yes' but actual 'no') who subsequently inquire about or apply for housing products.",
   "Feature Importance for Housing Loan Profile : Identifying the key characteristics (e.g., 'married', 'management', age range) associated with having a housing loan.",
   'Conversion Rate of Targeted Campaigns : Success rate of housing loan marketing campaigns specifically targeting profiles identified by the model.'],
  'business_value': 'Helps identify potential leads for housing loan products by finding customers who fit the profile of existing borrowers. This enables targeted marketing for mortgages and related services, potentially increasing loan origination volume and improving marketing ROI.',
  'target_variable': 'housing',
  'model_type': 'classification',
  'use_case_implementation_complexity': 'medium',
  'prediction_interpretation': "The model predicts the probability (or class label 'yes'/'no') that a customer fits the profile of someone with a housing loan. A prediction of 'yes' with high confidence (e.g., probability > 0.8) for a customer who currently has 'housing' = 'no' means their profile (age, job, marital status, balance etc.) is very similar to customers who *do* have housing loans in the dataset. This customer could be flagged as a potential prospect for housing-related financial products."},
 {'title': 'Identifying Profiles Similar to Personal Loan Holders',
  'description': "Similar to the housing loan use case, this involves building an AI model to predict whether a customer has an existing personal loan ('loan' = 'yes') based on their other attributes. The objective is to identify customer profiles that are statistically similar to those who currently hold personal loans.\n\nThe model learns the characteristics (demographics, job, financial status like balance, potentially default status or housing loan status) associated with having a personal loan. By applying this model, the bank can score customers based on their likelihood of fitting this profile.\n\nCustomers who receive a high score (predicted 'yes') but do not currently have a personal loan ('loan' = 'no') represent potential leads for personal loan marketing campaigns. This allows the bank to proactively target individuals who might have a need or propensity for personal loan products, potentially increasing loan uptake.",
  'kpis': ["Model Accuracy/AUC in Predicting 'loan' Status : Technical measure of the model's ability to identify current personal loan holders.",
   "Prospect Identification Rate : Percentage of customers identified by the model (predicted 'yes' but actual 'no') who subsequently inquire about or apply for personal loans.",
   'Feature Importance for Personal Loan Profile : Identifying the key characteristics (e.g., specific job types, balance ranges, age groups) associated with having a personal loan.',
   'Conversion Rate of Targeted Campaigns : Success rate of personal loan marketing campaigns specifically targeting profiles identified by the model.'],
  'business_value': 'Identifies potential customers for personal loan products by matching their profiles to existing borrowers. This facilitates targeted marketing efforts, potentially increasing the volume of personal loans issued and improving the efficiency of marketing campaigns.',
  'target_variable': 'loan',
  'model_type': 'classification',
  'use_case_implementation_complexity': 'medium',
  'prediction_interpretation': "The model predicts the probability (or class label 'yes'/'no') that a customer fits the profile of someone with a personal loan. If the model predicts 'yes' with high confidence (e.g., probability > 0.7) for a customer who currently has 'loan' = 'no', it suggests this individual shares many characteristics (like age, job, balance, housing status) with those who do have personal loans. They can be considered a potential lead for personal loan offers."},
 {'title': 'Predicting Previous Marketing Campaign Outcome',
  'description': "This use case focuses on predicting the outcome of the *previous* marketing campaign ('poutcome': failure, success, other, unknown) based on customer attributes recorded *before* or *during* that previous campaign (using features like age, job, balance, etc., potentially excluding data from the *current* campaign like 'duration' or current 'campaign' number if aiming for a historical prediction).\n\nUnderstanding what factors led to success or failure in past interactions can provide valuable longitudinal insights into customer engagement patterns. For instance, identifying profiles that consistently resulted in 'failure' outcomes in previous campaigns might suggest a need for a fundamentally different approach or indicate low overall engagement potential.\n\nPredicting 'success' could highlight characteristics of customers who have been receptive in the past, making them potentially valuable targets for future campaigns, assuming their circumstances haven't drastically changed. Analyzing 'other' outcomes might reveal specific past campaign types or customer situations that need further investigation.",
  'kpis': ["Model Accuracy for Multi-Class Classification : Measuring the model's overall ability to correctly predict the previous outcome category.",
   "Precision/Recall for 'Success' Outcome : Measuring the model's ability to correctly identify customers who were successfully converted in the past.",
   "Precision/Recall for 'Failure' Outcome : Measuring the model's ability to correctly identify customers who were not receptive in the past.",
   'Feature Importance Analysis : Identifying key customer attributes correlated with specific past campaign outcomes.'],
  'business_value': 'Provides insights into long-term customer engagement patterns and factors driving historical campaign success or failure. This helps in refining customer understanding, identifying chronically unresponsive segments, and potentially prioritizing customers with a history of positive engagement.',
  'target_variable': 'poutcome',
  'model_type': 'classification',
  'use_case_implementation_complexity': 'medium',
  'prediction_interpretation': "The model predicts the most likely category for the 'poutcome' variable (failure, success, other, unknown) based on the input customer features. For example, the model might predict 'success' for a customer profile that strongly matches the characteristics of customers who subscribed in the previous campaign. Conversely, a prediction of 'failure' suggests the profile aligns with customers who did not respond positively in the past. This historical prediction informs current strategy; e.g., a predicted past 'success' might increase confidence in targeting that customer again, while a predicted past 'failure' might suggest caution or a different approach."}]
"""