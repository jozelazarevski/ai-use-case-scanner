# -*- coding: utf-8 -*-
"""
Rewritten version of the Flask application.
Created on Sun Mar 30 18:47:21 2025
"""

import os
import re
import json
import time
import random
import logging
import traceback
import pandas as pd
from datetime import datetime
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
from flask_session import Session
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import configurations and utilities
from config import Config
from utils.user_auth import init_auth_routes, init_database, login_required
from utils import user_models
from ml_trainer import train_model_with_robust_error_handling
from execute_LLM_model import run_training_script
from utils.read_file import read_data_flexible

# Initialize Flask app
app = Flask(__name__, static_folder='static')

# Load configurations
app.config.from_object(Config)

# Initialize Flask-Session
Session(app)

# Initialize authentication routes and database
init_auth_routes(app)
init_database()

# Helper functions
def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def generate_unique_filename_timestamp(file="", model_type="", user="user"):
    """Generate a unique filename using a timestamp."""
    timestamp = int(time.time() * 1000)
    filename = f"{user}_{model_type}_{file}_{timestamp}"
    return re.sub(r'[^a-zA-Z0-9_\-]', "", filename)

# Routes
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
            file.save(filepath)
            
            try:
                # Process the uploaded file
                file_content = read_data_flexible(filepath)
                ai_proposals = get_ai_use_case_proposals(file_content, filename)
                
                # Save proposals in session
                session['proposals'] = ai_proposals
                session['last_filename'] = filename
                session['file_path'] = filepath
                
                return render_template('results.html', filename=filename, proposals=ai_proposals)
            except Exception as e:
                error_trace = traceback.format_exc()
                logger.error(f"Error processing file: {str(e)}\n{error_trace}")
                flash(f"Error processing file: {str(e)}", 'error')
                return render_template('error.html', error=str(e), trace=error_trace)
        else:
            flash(f"Invalid file type. Allowed types: {', '.join(app.config['ALLOWED_EXTENSIONS'])}", 'error')
            return redirect(request.url)
    
    return render_template('upload.html', allowed_extensions=app.config['ALLOWED_EXTENSIONS'])

@app.route('/train_model/<model_type>', methods=['POST'])
@login_required
def train_model(model_type):
    """Train a model based on the selected proposal."""
    try:
        proposal_index = int(request.form.get('proposal_index', 0))
        proposals = session.get('proposals', [])
        if not proposals or proposal_index >= len(proposals):
            return jsonify({'error': 'Invalid proposal index'}), 400
        
        selected_proposal = proposals[proposal_index]
        target_variable = selected_proposal.get('target_variable')
        file_path = session.get('file_path')
        
        training_stats = train_model_with_robust_error_handling(file_path, model_type, proposal_index, target_variable)
        session['training_results'] = training_stats
        
        return jsonify({'success': True, 'redirect': url_for('training_results')})
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in train_model: {str(e)}\n{error_trace}")
        return jsonify({'error': str(e)}), 500

@app.route('/training_results', methods=['GET'])
@login_required
def training_results():
    """Display training results."""
    training_results = session.get('training_results')
    if not training_results:
        flash("No training results found.", 'warning')
        return redirect(url_for('upload_file'))
    
    return render_template('training_results.html', training_results=training_results)

@app.route('/saved-use-cases', methods=['GET'])
@login_required
def saved_use_cases():
    """Display saved use cases."""
    user_id = session.get('user_id')
    use_cases = user_models.get_user_use_cases(user_id)
    return render_template('saved_use_cases.html', use_cases=use_cases)

@app.route('/view-use-case/<use_case_id>', methods=['GET'])
@login_required
def view_use_case(use_case_id):
    """View details of a specific saved use case."""
    user_id = session.get('user_id')
    use_case = user_models.get_use_case_by_id(user_id, use_case_id)
    if not use_case:
        flash('Use case not found', 'error')
        return redirect(url_for('saved_use_cases'))
    return render_template('view_use_case.html', use_case=use_case)

# Main entry point
if __name__ == '__main__':
    app.run(debug=True)