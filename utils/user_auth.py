# -*- coding: utf-8 -*-
"""
Authentication module for the AI Use Case Generator application.
Handles user creation, authentication, and session management.
"""

import os
import json
import uuid
import time
import re
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session, redirect, url_for, flash, request, render_template
from functools import wraps
from config import Config

# Database configuration
DATABASE_DIR = Config.DATABASE_DIR
USERS_DB_FILE = os.path.join(DATABASE_DIR, 'users.json')
SESSIONS_DB_FILE = os.path.join(DATABASE_DIR, 'sessions.json')

# Ensure database directory exists
def init_database():
    """Initialize the database files if they don't exist."""
    os.makedirs(DATABASE_DIR, exist_ok=True)
    
    if not os.path.exists(USERS_DB_FILE):
        with open(USERS_DB_FILE, 'w') as f:
            json.dump({}, f)
    
    if not os.path.exists(SESSIONS_DB_FILE):
        with open(SESSIONS_DB_FILE, 'w') as f:
            json.dump({}, f)

# Initialize database on module import
init_database()

# User CRUD Operations
def create_user(first_name, last_name, email, password, organization=None):
    """
    Create a new user in the database.
    
    Args:
        first_name (str): User's first name
        last_name (str): User's last name
        email (str): User's email (used as unique identifier)
        password (str): User's password (will be hashed)
        organization (str, optional): User's organization
        
    Returns:
        str: User ID if created successfully, None if user already exists
    """
    # Load current users
    try:
        with open(USERS_DB_FILE, 'r') as f:
            users = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        users = {}
    
    # Check if email already exists
    email = email.lower().strip()
    for user_id, user_data in users.items():
        if user_data.get('email').lower() == email:
            return None  # User already exists
    
    # Create new user
    user_id = str(uuid.uuid4())
    users[user_id] = {
        'id': user_id,
        'first_name': first_name,
        'last_name': last_name,
        'email': email,
        'password_hash': generate_password_hash(password),
        'organization': organization,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat()
    }
    
    # Save to database
    with open(USERS_DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)
    
    return user_id

def get_user_by_email(email):
    """
    Get user data by email.
    
    Args:
        email (str): User's email
        
    Returns:
        dict: User data if found, None otherwise
    """
    try:
        with open(USERS_DB_FILE, 'r') as f:
            users = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None
    
    email = email.lower().strip()
    for user_id, user_data in users.items():
        if user_data.get('email').lower() == email:
            return user_data
    
    return None

def get_user_by_id(user_id):
    """
    Get user data by ID.
    
    Args:
        user_id (str): User's ID
        
    Returns:
        dict: User data if found, None otherwise
    """
    try:
        with open(USERS_DB_FILE, 'r') as f:
            users = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None
    
    return users.get(user_id)

def update_user(user_id, first_name=None, last_name=None, email=None, organization=None):
    """
    Update user information.
    
    Args:
        user_id (str): User's ID
        first_name (str, optional): New first name
        last_name (str, optional): New last name
        email (str, optional): New email
        organization (str, optional): New organization
        
    Returns:
        bool: True if update successful, False otherwise
    """
    try:
        with open(USERS_DB_FILE, 'r') as f:
            users = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return False
    
    if user_id not in users:
        return False
    
    # Update fields that were provided
    if first_name:
        users[user_id]['first_name'] = first_name
    
    if last_name:
        users[user_id]['last_name'] = last_name
    
    if email:
        # Check if email is already in use by another user
        email = email.lower().strip()
        for uid, user_data in users.items():
            if uid != user_id and user_data.get('email').lower() == email:
                return False  # Email already in use
        
        users[user_id]['email'] = email
    
    if organization is not None:  # Allow empty string to clear organization
        users[user_id]['organization'] = organization
    
    users[user_id]['updated_at'] = datetime.now().isoformat()
    
    # Save to database
    with open(USERS_DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)
    
    return True

def change_password(user_id, current_password, new_password):
    """
    Change a user's password.
    
    Args:
        user_id (str): User's ID
        current_password (str): Current password for verification
        new_password (str): New password to set
        
    Returns:
        bool: True if password changed successfully, False otherwise
    """
    try:
        with open(USERS_DB_FILE, 'r') as f:
            users = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return False
    
    if user_id not in users:
        return False
    
    # Verify current password
    if not check_password_hash(users[user_id]['password_hash'], current_password):
        return False
    
    # Update password
    users[user_id]['password_hash'] = generate_password_hash(new_password)
    users[user_id]['updated_at'] = datetime.now().isoformat()
    
    # Save to database
    with open(USERS_DB_FILE, 'w') as f:
        json.dump(users, f, indent=2)
    
    return True

def authenticate_user(email, password):
    """
    Authenticate a user by email and password.
    
    Args:
        email (str): User's email
        password (str): User's password
        
    Returns:
        dict: User data if authentication successful, None otherwise
    """
    user = get_user_by_email(email)
    
    if user and check_password_hash(user['password_hash'], password):
        return user
    
    return None

# Session Management
def create_session(user_id, remember=False):
    """
    Create a new session for a user.
    
    Args:
        user_id (str): User's ID
        remember (bool, optional): Whether to create a long-lived session
        
    Returns:
        str: Session token
    """
    try:
        with open(SESSIONS_DB_FILE, 'r') as f:
            sessions = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        sessions = {}
    
    # Create new session
    session_token = str(uuid.uuid4())
    # Use Config settings for session duration
    days = Config.SESSION_EXPIRY_DAYS if hasattr(Config, 'SESSION_EXPIRY_DAYS') else 30
    expiry = int(time.time()) + (days * 24 * 60 * 60 if remember else 24 * 60 * 60)
    
    sessions[session_token] = {
        'user_id': user_id,
        'created_at': int(time.time()),
        'expires_at': expiry
    }
    
    # Save to database
    with open(SESSIONS_DB_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)
    
    return session_token

def validate_session(session_token):
    """
    Validate a session token.
    
    Args:
        session_token (str): Session token to validate
        
    Returns:
        str: User ID if session is valid, None otherwise
    """
    try:
        with open(SESSIONS_DB_FILE, 'r') as f:
            sessions = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None
    
    session = sessions.get(session_token)
    
    if not session:
        return None
    
    # Check if session has expired
    if session.get('expires_at', 0) < int(time.time()):
        # Remove expired session
        del sessions[session_token]
        with open(SESSIONS_DB_FILE, 'w') as f:
            json.dump(sessions, f, indent=2)
        return None
    
    return session.get('user_id')

def delete_session(session_token):
    """
    Delete a session.
    
    Args:
        session_token (str): Session token to delete
        
    Returns:
        bool: True if session was deleted, False otherwise
    """
    try:
        with open(SESSIONS_DB_FILE, 'r') as f:
            sessions = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return False
    
    if session_token in sessions:
        del sessions[session_token]
        with open(SESSIONS_DB_FILE, 'w') as f:
            json.dump(sessions, f, indent=2)
        return True
    
    return False

def delete_user_sessions(user_id):
    """
    Delete all sessions for a user.
    
    Args:
        user_id (str): User's ID
        
    Returns:
        int: Number of sessions deleted
    """
    try:
        with open(SESSIONS_DB_FILE, 'r') as f:
            sessions = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return 0
    
    deleted_count = 0
    tokens_to_delete = []
    
    for token, session in sessions.items():
        if session.get('user_id') == user_id:
            tokens_to_delete.append(token)
            deleted_count += 1
    
    for token in tokens_to_delete:
        del sessions[token]
    
    with open(SESSIONS_DB_FILE, 'w') as f:
        json.dump(sessions, f, indent=2)
    
    return deleted_count

# Authentication Decorator
def login_required(view_function):
    """
    Decorator function to require login for views.
    
    Usage:
        @login_required
        def protected_route():
            ...
    """
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please login to access this page', 'error')
            return redirect(url_for('login', next=request.url))
        
        # Validate session token
        user_id = session.get('user_id')
        session_token = session.get('session_token')
        
        if not validate_session(session_token):
            session.clear()
            flash('Your session has expired. Please login again', 'error')
            return redirect(url_for('login'))
            
        return view_function(*args, **kwargs)
    
    return decorated_function

# Auth routes initialization function for Flask app
def init_auth_routes(app):
    """
    Initialize authentication routes for the Flask app.
    
    Args:
        app: Flask application instance
    """
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Handle user login."""
        # If user is already logged in, redirect to home
        if 'user_id' in session:
            return redirect(url_for('home'))
            
        if request.method == 'POST':
            email = request.form.get('email', '').lower()
            password = request.form.get('password', '')
            remember = request.form.get('remember') == 'on'
            
            if not email or not password:
                flash('Please enter both email and password', 'error')
                return render_template('login.html')
            
            # Authenticate user
            user = authenticate_user(email, password)
            
            if not user:
                flash('Invalid email or password', 'error')
                return render_template('login.html')
            
            # Create session
            session_token = create_session(user['id'], remember)
            
            # Set session cookies
            session['user_id'] = user['id']
            session['session_token'] = session_token
            session['user_email'] = email
            session['user_name'] = f"{user['first_name']} {user['last_name']}"
            
            # Redirect to next page or home
            next_page = request.args.get('next')
            if next_page and next_page.startswith('/'):
                return redirect(next_page)
            return redirect(url_for('home'))
        
        return render_template('login.html')

    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """Handle user registration."""
        # If user is already logged in, redirect to home
        if 'user_id' in session:
            return redirect(url_for('home'))
            
        if request.method == 'POST':
            first_name = request.form.get('firstName', '')
            last_name = request.form.get('lastName', '')
            email = request.form.get('email', '').lower()
            password = request.form.get('password', '')
            confirm_password = request.form.get('confirmPassword', '')
            organization = request.form.get('organization', '')
            terms = request.form.get('terms') == 'on'
            
            # Form validation
            if not first_name or not last_name or not email or not password:
                flash('All required fields must be filled', 'error')
                return render_template('register.html')
            
            if password != confirm_password:
                flash('Passwords do not match', 'error')
                return render_template('register.html')
            
            if not terms:
                flash('You must agree to the Terms of Service and Privacy Policy', 'error')
                return render_template('register.html')
            
            # Validate email format
            email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
            if not re.match(email_pattern, email):
                flash('Please enter a valid email address', 'error')
                return render_template('register.html')
            
            # Check if email already exists
            if get_user_by_email(email):
                flash('Email already exists. Please use a different email or login', 'error')
                return render_template('register.html')
            
            # Password strength validation
            min_length = Config.PASSWORD_MIN_LENGTH if hasattr(Config, 'PASSWORD_MIN_LENGTH') else 8
            if len(password) < min_length:
                flash(f'Password must be at least {min_length} characters long', 'error')
                return render_template('register.html')
            
            # Create user
            user_id = create_user(first_name, last_name, email, password, organization)
            
            if not user_id:
                flash('Error creating user. Please try again.', 'error')
                return render_template('register.html')
            
            # Create session
            session_token = create_session(user_id)
            
            # Set session cookies
            session['user_id'] = user_id
            session['session_token'] = session_token
            session['user_email'] = email
            session['user_name'] = f"{first_name} {last_name}"
            
            flash('Registration successful! Welcome to AI Use Case Generator.', 'success')
            return redirect(url_for('home'))
        
        return render_template('register.html')

    @app.route('/logout')
    def logout():
        """Handle user logout."""
        # Delete session from database
        if 'session_token' in session:
            delete_session(session['session_token'])
        
        # Clear session
        session.clear()
        
        flash('You have been logged out.', 'success')
        return redirect(url_for('login'))

    @app.route('/forgot-password', methods=['GET', 'POST'])
    def forgot_password():
        """Handle forgot password requests."""
        if request.method == 'POST':
            email = request.form.get('email', '').lower()
            
            if not email:
                flash('Please enter your email address', 'error')
                return render_template('forgot_password.html')
            
            # Check if email exists
            user = get_user_by_email(email)
            
            if not user:
                # Don't reveal that the email doesn't exist for security reasons
                flash('If your email is registered, you will receive password reset instructions.', 'success')
                return render_template('forgot_password.html')
            
            # In a real implementation, generate a reset token and send an email
            # For this example, we'll just display a success message
            flash('Password reset instructions have been sent to your email.', 'success')
            return render_template('forgot_password.html')
        
        return render_template('forgot_password.html')

    @app.route('/terms')
    def terms():
        """Display terms of service."""
        return render_template('terms.html')

    @app.route('/privacy')
    def privacy():
        """Display privacy policy."""
        return render_template('privacy.html')

    @app.route('/profile')
    @login_required
    def profile():
        """Display user profile."""
        # Get user data
        user_id = session['user_id']
        user = get_user_by_id(user_id)
        
        if not user:
            # User not found, log them out
            session.clear()
            flash('Your session has expired. Please login again.', 'error')
            return redirect(url_for('login'))
        
        return render_template('profile.html', user=user)

    # Update profile and change password routes
    @app.route('/update_profile', methods=['POST'])
    @login_required
    def update_profile():
        """Handle profile information updates."""
        first_name = request.form.get('firstName', '')
        last_name = request.form.get('lastName', '')
        email = request.form.get('email', '').lower()
        organization = request.form.get('organization', '')
        
        if not first_name or not last_name or not email:
            flash('First name, last name, and email are required', 'error')
            return redirect(url_for('profile'))
        
        # Validate email format
        email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        if not re.match(email_pattern, email):
            flash('Please enter a valid email address', 'error')
            return redirect(url_for('profile'))
        
        # Get user ID from session
        user_id = session.get('user_id')
        
        # Update user information
        if update_user(user_id, first_name, last_name, email, organization):
            # Update session data
            session['user_email'] = email
            session['user_name'] = f"{first_name} {last_name}"
            
            flash('Profile information successfully updated', 'success')
        else:
            flash('Error updating profile. Email may already be in use.', 'error')
        
        return redirect(url_for('profile'))

    @app.route('/change_password', methods=['POST'])
    @login_required
    def change_password_route():
        """Handle password change requests."""
        current_password = request.form.get('currentPassword', '')
        new_password = request.form.get('newPassword', '')
        confirm_password = request.form.get('confirmPassword', '')
        
        if not current_password or not new_password or not confirm_password:
            flash('All password fields are required', 'error')
            return redirect(url_for('profile'))
        
        if new_password != confirm_password:
            flash('New passwords do not match', 'error')
            return redirect(url_for('profile'))
        
        min_length = Config.PASSWORD_MIN_LENGTH if hasattr(Config, 'PASSWORD_MIN_LENGTH') else 8
        if len(new_password) < min_length:
            flash(f'Password must be at least {min_length} characters long', 'error')
            return redirect(url_for('profile'))
        
        # Get user ID from session
        user_id = session.get('user_id')
        
        # Change password
        if change_password(user_id, current_password, new_password):
            flash('Password successfully updated', 'success')
        else:
            flash('Current password is incorrect', 'error')
        
        return redirect(url_for('profile'))

    # Before request handler to validate session on every request
    @app.before_request
    def validate_user_session():
        """Validate user session before each request."""
        if 'user_id' in session and 'session_token' in session:
            # Only validate if accessing a protected route
            if request.endpoint and request.endpoint not in ['login', 'register', 'logout', 'forgot_password', 'terms', 'privacy', 'static']:
                user_id = session.get('user_id')
                session_token = session.get('session_token')
                
                # Validate session token
                valid_user_id = validate_session(session_token)
                
                if not valid_user_id or valid_user_id != user_id:
                    # Clear session and redirect to login
                    session.clear()
                    flash('Your session has expired. Please login again.', 'error')
                    return redirect(url_for('login'))

    # Add global template context for user info
    @app.context_processor
    def inject_user():
        """Inject user info into all templates."""
        user = None
        if 'user_id' in session:
            user_id = session['user_id']
            user = get_user_by_id(user_id)
        
        return {'user': user}
    
    # Add login_required decorator to app context
    app.login_required = login_required
