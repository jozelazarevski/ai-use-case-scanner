# -*- coding: utf-8 -*-
"""
User models management module for the AI Use Case Generator application.
Handles saving, retrieving, and deleting models using PostgreSQL database.
"""

import os
import json
import shutil
import uuid
import traceback
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Any, Union
import psycopg2
from psycopg2.extras import RealDictCursor
import contextlib
from werkzeug.security import generate_password_hash, check_password_hash
from flask import session, request, redirect, url_for, flash, render_template
from functools import wraps
import re
import numpy as np
# Import config
from config import Config
import uuid
import psycopg2.extensions

def adapt_uuid(uuid_value):
    return psycopg2.extensions.AsIs("'%s'" % uuid_value)

psycopg2.extensions.register_adapter(uuid.UUID, adapt_uuid)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database connection context manager
@contextlib.contextmanager
def get_db_connection():
    """
    Context manager for database connections.
    
    Usage:
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users")
            results = cursor.fetchall()
    """
    conn = None
    try:
        # Connect to the database with automatic UUID handling
        # Using values from Config instead of hardcoded values
        conn = psycopg2.connect(
            dbname=os.environ.get('DB_NAME', 'AIUseCase'),
            user=os.environ.get('DB_USER', 'postgres'),
            password=os.environ.get('DB_PASSWORD', 'root'),
            host=os.environ.get('DB_HOST', 'localhost'),
            port=os.environ.get('DB_PORT', '5432'),
            cursor_factory=RealDictCursor
        )
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Database connection error: {e}")
        raise e
    finally:
        if conn:
            conn.close()

def check_and_update_schema():
    """
    Comprehensive check and update of database schema to ensure all required tables
    and columns exist, preventing errors from missing columns.
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Schema definitions - tables and their required columns
                schema_definitions = {
                    'users': {
                        'id': "UUID PRIMARY KEY",
                        'username': "VARCHAR(50) UNIQUE NOT NULL",
                        'email': "VARCHAR(100) UNIQUE NOT NULL",
                        'password_hash': "VARCHAR(255) NOT NULL",
                        'first_name': "VARCHAR(50)",
                        'last_name': "VARCHAR(50)",
                        'organization': "VARCHAR(100)",
                        'created_at': "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
                        'updated_at': "TIMESTAMP",
                        'is_active': "BOOLEAN DEFAULT TRUE",
                        'last_login': "TIMESTAMP"
                    },
                    'sessions': {
                        'token': "UUID PRIMARY KEY",
                        'user_id': "UUID NOT NULL",
                        'created_at': "TIMESTAMP NOT NULL",
                        'expires_at': "TIMESTAMP",
                        'ip_address': "VARCHAR(45)",
                        'user_agent': "TEXT",
                        'is_active': "BOOLEAN DEFAULT TRUE"
                    },
                    'models': {
                        'id': "UUID PRIMARY KEY",
                        'user_id': "UUID NOT NULL",
                        'name': "VARCHAR(100) NOT NULL",
                        'description': "TEXT",
                        'model_path': "TEXT NOT NULL",
                        'model_type': "VARCHAR(50) NOT NULL",
                        'target_variable': "VARCHAR(100)",
                        'accuracy': "FLOAT",
                        'feature_names': "JSONB",
                        'feature_importance': "JSONB",
                        'created_at': "TIMESTAMP NOT NULL",
                        'updated_at': "TIMESTAMP"
                    },
                    'use_cases': {
                        'id': "UUID PRIMARY KEY",
                        'user_id': "UUID NOT NULL",
                        'filename': "VARCHAR(255)",
                        'file_path': "TEXT",
                        'proposals': "JSONB",
                        'created_at': "TIMESTAMP NOT NULL",
                        'metadata': "JSONB"
                    },
                    'embeddings': {
                        'id': "UUID PRIMARY KEY",
                        'user_id': "UUID NOT NULL",
                        'model_id': "UUID NOT NULL",
                        'name': "VARCHAR(100) NOT NULL",
                        'embed_path': "TEXT NOT NULL",
                        'settings': "JSONB",
                        'created_at': "TIMESTAMP NOT NULL"
                    },
                    'prediction_logs': {
                        'id': "UUID PRIMARY KEY DEFAULT gen_random_uuid()",
                        'user_id': "UUID NOT NULL",
                        'model_id': "UUID NOT NULL",
                        'input_data': "JSONB",
                        'prediction_result': "JSONB",
                        'timestamp': "TIMESTAMP NOT NULL",
                        'feedback': "JSONB"
                    }
                }
                
                # Iterate through all tables in the schema
                for table_name, columns in schema_definitions.items():
                    # Check if table exists
                    cursor.execute(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables 
                            WHERE table_name = %s
                        );
                        """, (table_name,)
                    )
                    table_exists = cursor.fetchone()['exists']
                    
                    if not table_exists:
                        logger.info(f"Table {table_name} does not exist. Creating it...")
                        
                        # Create the CREATE TABLE statement
                        create_stmt = f"CREATE TABLE {table_name} (\n"
                        for col_name, col_def in columns.items():
                            create_stmt += f"    {col_name} {col_def},\n"
                        # Remove the last comma and close the statement
                        create_stmt = create_stmt[:-2] + "\n);"
                        
                        # Execute the CREATE TABLE statement
                        try:
                            cursor.execute(create_stmt)
                            logger.info(f"Created table {table_name}")
                        except Exception as e:
                            logger.error(f"Error creating table {table_name}: {e}")
                        
                        # Skip column checking for newly created tables
                        continue
                        
                    # For existing tables, check if all required columns exist
                    cursor.execute(
                        """
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_name = %s;
                        """, (table_name,)
                    )
                    existing_columns = [row['column_name'] for row in cursor.fetchall()]
                    
                    # Check for missing columns
                    for col_name, col_def in columns.items():
                        if col_name not in existing_columns:
                            logger.info(f"Adding missing column {col_name} to table {table_name}")
                            alter_stmt = f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def};"
                            try:
                                cursor.execute(alter_stmt)
                                logger.info(f"Added column {col_name} to {table_name}")
                            except Exception as e:
                                logger.error(f"Error adding column {col_name} to {table_name}: {e}")
                
                # Commit all changes
                conn.commit()
                logger.info("Database schema check and update completed successfully")
                
    except Exception as e:
        logger.error(f"Error checking/updating schema: {str(e)}")
        logger.error(traceback.format_exc())

def init_database():
    """
    Initialize the database with required tables if they don't exist.
    
    Creates the following tables:
    - users: User authentication and profile information
    - sessions: User login sessions
    - models: ML models saved by users
    - use_cases: AI use case proposals
    - embeddings: Model embeddings for sharing
    - prediction_logs: Logs of model predictions
    """
    
    try:
        # Check and update schema first
        check_and_update_schema()
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Check if tables exist
                cursor.execute(
                    """
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_name = 'users'
                    );
                    """
                )
                tables_exist = cursor.fetchone()['exists']
                
                if not tables_exist:
                    logger.info("Creating database tables...")
                    
                    # Users table
                    cursor.execute(
                        """
                        CREATE TABLE users (
                            id UUID PRIMARY KEY,
                            username VARCHAR(50) UNIQUE NOT NULL,
                            email VARCHAR(100) UNIQUE NOT NULL,
                            password_hash VARCHAR(255) NOT NULL,
                            first_name VARCHAR(50),
                            last_name VARCHAR(50),
                            organization VARCHAR(100),
                            created_at TIMESTAMP NOT NULL,
                            updated_at TIMESTAMP,
                            is_active BOOLEAN DEFAULT TRUE,
                            last_login TIMESTAMP
                        );
                        """
                    )
                    
                    # Sessions table
                    cursor.execute(
                        """
                        CREATE TABLE sessions (
                            token UUID PRIMARY KEY,
                            user_id UUID NOT NULL,
                            created_at TIMESTAMP NOT NULL,
                            expires_at TIMESTAMP,
                            ip_address VARCHAR(45),
                            user_agent TEXT,
                            is_active BOOLEAN DEFAULT TRUE,
                            CONSTRAINT fk_user
                                FOREIGN KEY(user_id) 
                                REFERENCES users(id)
                                ON DELETE CASCADE
                        );
                        """
                    )
                    
                    # Models table
                    cursor.execute(
                        """
                        CREATE TABLE models (
                            id UUID PRIMARY KEY,
                            user_id UUID NOT NULL,
                            name VARCHAR(100) NOT NULL,
                            description TEXT,
                            model_path TEXT NOT NULL,
                            model_type VARCHAR(50) NOT NULL,
                            target_variable VARCHAR(100),
                            accuracy FLOAT,
                            feature_names JSONB,
                            feature_importance JSONB,
                            created_at TIMESTAMP NOT NULL,
                            updated_at TIMESTAMP,
                            CONSTRAINT fk_user
                                FOREIGN KEY(user_id) 
                                REFERENCES users(id)
                                ON DELETE CASCADE
                        );
                        """
                    )
                    
                    # Use cases table
                    cursor.execute(
                        """
                        CREATE TABLE use_cases (
                            id UUID PRIMARY KEY,
                            user_id UUID NOT NULL,
                            filename VARCHAR(255),
                            file_path TEXT,
                            proposals JSONB,
                            created_at TIMESTAMP NOT NULL,
                            metadata JSONB,
                            CONSTRAINT fk_user
                                FOREIGN KEY(user_id) 
                                REFERENCES users(id)
                                ON DELETE CASCADE
                        );
                        """
                    )
                    
                    # Embeddings table
                    cursor.execute(
                        """
                        CREATE TABLE embeddings (
                            id UUID PRIMARY KEY,
                            user_id UUID NOT NULL,
                            model_id UUID NOT NULL,
                            name VARCHAR(100) NOT NULL,
                            embed_path TEXT NOT NULL,
                            settings JSONB,
                            created_at TIMESTAMP NOT NULL,
                            CONSTRAINT fk_user
                                FOREIGN KEY(user_id) 
                                REFERENCES users(id)
                                ON DELETE CASCADE,
                            CONSTRAINT fk_model
                                FOREIGN KEY(model_id) 
                                REFERENCES models(id)
                                ON DELETE CASCADE
                        );
                        """
                    )
                    
                    # Prediction logs table
                    cursor.execute(
                        """
                        CREATE TABLE prediction_logs (
                            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                            user_id UUID NOT NULL,
                            model_id UUID NOT NULL,
                            input_data JSONB,
                            prediction_result JSONB,
                            timestamp TIMESTAMP NOT NULL,
                            feedback JSONB,
                            CONSTRAINT fk_user
                                FOREIGN KEY(user_id) 
                                REFERENCES users(id)
                                ON DELETE CASCADE,
                            CONSTRAINT fk_model
                                FOREIGN KEY(model_id) 
                                REFERENCES models(id)
                                ON DELETE CASCADE
                        );
                        """
                    )
                    
                    # Create admin user if it doesn't exist
                    admin_id = uuid.uuid4()
                    admin_password = os.environ.get('ADMIN_PASSWORD', 'admin123')  # Should be set in environment
                    password_hash = generate_password_hash(admin_password)
                    
                    cursor.execute(
                        """
                        INSERT INTO users (
                            id, username, email, password_hash, first_name, last_name,
                            created_at, is_active
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            admin_id, 'admin', 'admin@example.com', password_hash,
                            'Admin', 'User', datetime.now(), True
                        )
                    )
                    
                    conn.commit()
                    logger.info("Database tables created successfully")
                else:
                    logger.info("Database tables already exist")
    
    except Exception as e:
        logger.error(f"Database initialization error: {str(e)}")
        logger.error(traceback.format_exc())


# Add this function to your user_auth.py file, just after the init_database function


def create_users_table(cursor):
    """Create the users table with all required columns"""
    cursor.execute(
        """
        CREATE TABLE users (
            id UUID PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            first_name VARCHAR(50),
            last_name VARCHAR(50),
            organization VARCHAR(100),
            created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE,
            last_login TIMESTAMP
        );
        """
    )

 
def login_required(f):
    """
    Decorator to require login for certain routes.
    Redirects to login page if not authenticated.
    
    Usage:
    @app.route('/protected')
    @login_required
    def protected_route():
        return "This is a protected page"
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Get user information by ID.
    
    Args:
        user_id (str): User ID
        
    Returns:
        Optional[Dict]: User data or None if not found
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, username, email, first_name, last_name, organization,
                           created_at, updated_at, is_active, last_login
                    FROM users
                    WHERE id = %s
                    """,
                    (user_id,)
                )
                
                return cursor.fetchone()
    except Exception as e:
        logger.error(f"Error fetching user by ID: {str(e)}")
        return None

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """
    Get user information by email.
    
    Args:
        email (str): User email
        
    Returns:
        Optional[Dict]: User data or None if not found
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, username, email, password_hash, first_name, last_name, 
                           organization, created_at, updated_at, is_active, last_login
                    FROM users
                    WHERE email = %s
                    """,
                    (email.lower(),)  # Always use lowercase for email lookups
                )
                
                return cursor.fetchone()
    except Exception as e:
        logger.error(f"Error fetching user by email: {str(e)}")
        return None

def authenticate_user(email: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user with email and password.
    
    Args:
        email (str): User email
        password (str): User password
        
    Returns:
        Optional[Dict]: User data if authentication is successful, None otherwise
    """
    try:
        user = get_user_by_email(email.lower())  # Normalize email to lowercase
        
        if not user:
            return None
        
        if not check_password_hash(user['password_hash'], password):
            return None
        
        # Update last login time
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE users
                    SET last_login = %s
                    WHERE id = %s
                    """,
                    (datetime.now(), user['id'])
                )
                conn.commit()
        
        return user
    except Exception as e:
        logger.error(f"Authentication error: {str(e)}")
        return None

def create_user(first_name: str, last_name: str, email: str, password: str, 
               organization: Optional[str] = None) -> Optional[str]:
    """
    Create a new user.
    
    Args:
        first_name (str): User's first name
        last_name (str): User's last name
        email (str): User's email
        password (str): User's password
        organization (str, optional): User's organization
        
    Returns:
        Optional[str]: User ID if successful, None otherwise
    """
    try:
        # Normalize email to lowercase
        email = email.lower()
        
        # Check if email already exists
        existing_user = get_user_by_email(email)
        if existing_user:
            return None
        
        # Generate username from email
        username = email.split('@')[0]
        
        # Generate user ID
        user_id = uuid.uuid4()
        
        # Hash the password
        password_hash = generate_password_hash(password)
        
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO users (
                        id, username, email, password_hash, first_name, last_name,
                        organization, created_at, is_active
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        user_id, username, email, password_hash, first_name, last_name,
                        organization, datetime.now(), True
                    )
                )
                conn.commit()
                
                return str(user_id)
    except Exception as e:
        logger.error(f"Error creating user: {str(e)}")
        return None

def update_user(user_id: str, first_name: str, last_name: str, email: str, 
               organization: Optional[str] = None) -> bool:
    """
    Update user information.
    
    Args:
        user_id (str): User ID
        first_name (str): User's first name
        last_name (str): User's last name
        email (str): User's email
        organization (str, optional): User's organization
        
    Returns:
        bool: Success status
    """
    try:
        # Normalize email to lowercase
        email = email.lower()
        
        # Check if email is already in use by another user
        existing_user = get_user_by_email(email)
        if existing_user and str(existing_user['id']) != user_id:
            return False
        
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE users
                    SET first_name = %s, last_name = %s, email = %s, 
                        organization = %s, updated_at = %s
                    WHERE id = %s
                    """,
                    (first_name, last_name, email, organization, datetime.now(), user_id)
                )
                conn.commit()
                
                return True
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        return False

def change_password(user_id: str, current_password: str, new_password: str) -> bool:
    """
    Change user password.
    
    Args:
        user_id (str): User ID
        current_password (str): Current password
        new_password (str): New password
        
    Returns:
        bool: Success status
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get current password hash
                cursor.execute(
                    """
                    SELECT password_hash
                    FROM users
                    WHERE id = %s
                    """,
                    (user_id,)
                )
                
                user = cursor.fetchone()
                if not user:
                    return False
                
                # Verify current password
                if not check_password_hash(user['password_hash'], current_password):
                    return False
                
                # Validate password length
                min_length = Config.PASSWORD_MIN_LENGTH
                if len(new_password) < min_length:
                    return False
                
                # Hash new password
                new_password_hash = generate_password_hash(new_password)
                
                # Update password
                cursor.execute(
                    """
                    UPDATE users
                    SET password_hash = %s, updated_at = %s
                    WHERE id = %s
                    """,
                    (new_password_hash, datetime.now(), user_id)
                )
                conn.commit()
                
                return True
    except Exception as e:
        logger.error(f"Error changing password: {str(e)}")
        return False

def create_session(user_id: str, remember: bool, ip_address: Optional[str] = None, 
                  user_agent: Optional[str] = None) -> Optional[str]:
    """
    Create a new session for a user.
    
    Args:
        user_id (str): User ID
        remember (bool): Whether to remember the session
        ip_address (str, optional): Client IP address
        user_agent (str, optional): Client user agent
        
    Returns:
        Optional[str]: Session token if successful, None otherwise
    """
    try:
        # Generate session token
        token = uuid.uuid4()
        
        # Get current time with timezone
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        
        # Set expiration time based on remember flag
        if remember:
            # Expire in days set in config
            expires_at = now + timedelta(days=Config.SESSION_EXPIRY_DAYS)
        else:
            # Expire in 12 hours
            expires_at = now + timedelta(hours=12)
        
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO sessions (
                        token, user_id, created_at, expires_at, ip_address, user_agent, is_active
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING token
                    """,
                    (
                        token, user_id, now, expires_at, ip_address, user_agent, True
                    )
                )
                conn.commit()
                
                return str(token)
    except Exception as e:
        logger.error(f"Error creating session: {str(e)}")
        return None

def validate_session(token: str) -> Optional[str]:
    """
    Validate a session token.
    
    Args:
        token (str): Session token
        
    Returns:
        Optional[str]: User ID if session is valid, None otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT user_id, expires_at, is_active
                    FROM sessions
                    WHERE token = %s
                    """,
                    (token,)
                )
                
                session_data = cursor.fetchone()
                
                if not session_data:
                    return None
                
                # Check if session is active and not expired
                if not session_data['is_active']:
                    return None
                
                if session_data['expires_at']:
                    # Create timezone-aware current datetime to match PostgreSQL timestamp with timezone
                    from datetime import datetime, timezone
                    now = datetime.now(timezone.utc)
                    
                    # Make sure expires_at is timezone-aware too
                    expires_at = session_data['expires_at']
                    if expires_at.tzinfo is None:
                        # If expires_at is naive, assume it's in UTC
                        from datetime import timezone
                        expires_at = expires_at.replace(tzinfo=timezone.utc)
                    
                    if expires_at < now:
                        # Session expired, deactivate it
                        cursor.execute(
                            """
                            UPDATE sessions
                            SET is_active = FALSE
                            WHERE token = %s
                            """,
                            (token,)
                        )
                        conn.commit()
                        return None
                
                return str(session_data['user_id'])
    except Exception as e:
        logger.error(f"Error validating session: {str(e)}")
        return None

def delete_session(token: str) -> bool:
    """
    Delete a session.
    
    Args:
        token (str): Session token
        
    Returns:
        bool: Success status
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE sessions
                    SET is_active = FALSE
                    WHERE token = %s
                    """,
                    (token,)
                )
                conn.commit()
                
                return True
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        return False


def save_user_model(user_id, model_name, model_object, metrics, metadata=None):
    """
    Save a model and all its metrics directly to the database.
    
    Args:
        user_id (str): User ID
        model_name (str): Name for the model
        model_object: The trained model object
        metrics (dict): Dictionary containing model metrics (accuracy, recall, etc.)
        metadata (dict, optional): Additional metadata
    
    Returns:
        str: Model ID if successful, None otherwise
    """
    try:
        logger.info(f"Saving model: {model_name} for user {user_id}")
        
        # Generate a unique ID for the model
        model_id = str(uuid.uuid4())
        
        # Ensure metadata is a dictionary
        if metadata is None:
            metadata = {}
        
        # Add current timestamp
        current_time = datetime.now()
        metadata['created_at'] = current_time.isoformat()
        
        # Add metrics to metadata
        if 'metrics' not in metadata:
            metadata['metrics'] = {}
        
        # Add basic metrics if provided
        if metrics:
            for key, value in metrics.items():
                # Convert numpy types to Python native types
                if hasattr(value, 'item'):  # Check if it's a numpy scalar
                    value = value.item()
                elif isinstance(value, np.ndarray):
                    value = value.tolist()
                
                metadata['metrics'][key] = value
        
        # Extract feature importance if available
        if hasattr(model_object, 'feature_importances_'):
            feature_importances = model_object.feature_importances_
            if 'features' not in metadata:
                metadata['features'] = {}
            
            if 'trained_features' in metadata['features'] and len(metadata['features']['trained_features']) == len(feature_importances):
                # Create feature importance dictionary
                importance_dict = {}
                for i, feature in enumerate(metadata['features']['trained_features']):
                    importance_dict[feature] = float(feature_importances[i])
                
                metadata['features']['importance'] = importance_dict
        elif hasattr(model_object, 'coef_'):
            # For linear models
            if 'features' not in metadata:
                metadata['features'] = {}
                
            if hasattr(model_object.coef_, 'shape') and len(model_object.coef_.shape) > 1:
                # For multi-class models, average absolute coefficients
                feature_importances = np.mean(np.abs(model_object.coef_), axis=0)
            else:
                # For binary classification or regression
                feature_importances = np.abs(model_object.coef_)
                
            if 'trained_features' in metadata['features'] and len(metadata['features']['trained_features']) == len(feature_importances):
                # Create feature importance dictionary
                importance_dict = {}
                for i, feature in enumerate(metadata['features']['trained_features']):
                    importance_dict[feature] = float(feature_importances[i])
                
                metadata['features']['importance'] = importance_dict
        
        # Make all values JSON-serializable
        def make_json_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64,
                               np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (datetime, date)):
                return obj.isoformat()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            else:
                return obj
        
        # Process all metadata for JSON serialization
        sanitized_metadata = {}
        for k, v in metadata.items():
            if isinstance(v, dict):
                sanitized_metadata[k] = {
                    sk: make_json_serializable(sv) for sk, sv in v.items()
                }
            else:
                sanitized_metadata[k] = make_json_serializable(v)
        
        # Serialize the model to a binary format
        import io
        import joblib
        buffer = io.BytesIO()
        joblib.dump(model_object, buffer)
        serialized_model = buffer.getvalue()
        
        # Store in database
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO models (id, user_id, name, model_data, metadata, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (model_id, user_id, model_name, serialized_model, json.dumps(sanitized_metadata), current_time)
                )
                conn.commit()
                logger.info(f"Model saved successfully with ID: {model_id}")
        
        return model_id
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


def get_user_models(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all models for a user from the database
    
    Args:
        user_id (str): User ID
        
    Returns:
        List[Dict[str, Any]]: List of model data dictionaries
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Query models for this user
                cursor.execute(
                    """
                    SELECT id, user_id, name, model_path, metadata, created_at, model_data
                    FROM models
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (user_id,)
                )
                
                models = cursor.fetchall()
                result_models = []
                
                # Process each model
                for model in models:
                    # Handle case where metadata is already a dictionary
                    metadata = model['metadata'] if isinstance(model['metadata'], dict) else {}
                    
                    # Check if model files or binary data exist
                    model_exists = False
                    if model['model_path'] and os.path.exists(model['model_path']):
                        model_exists = True
                    elif model.get('model_data') is not None:
                        model_exists = True
                    
                    # Convert datetime to string to avoid truncate filter issues
                    created_at_str = model['created_at'].isoformat() if model['created_at'] else 'N/A'
                    
                    # Create standardized model entry
                    processed_model = {
                        'id': model['id'],
                        'user_id': model['user_id'],
                        'name': model['name'],
                        'model_path': model['model_path'],
                        'created_at': created_at_str,  # String format instead of datetime
                        'file_exists': model_exists,
                        
                        # Process metadata to ensure it has all required fields
                        'metadata': {
                            'features': metadata.get('features', {}).get('trained_features', []),
                            'feature_importance': metadata.get('features', {}).get('importance', {}),
                            'accuracy': metadata.get('accuracy', metadata.get('metrics', {}).get('accuracy', None)),
                            'description': metadata.get('description', ''),
                            'title': metadata.get('title', model['name']),
                            'target_variable': metadata.get('target_variable', 'Unknown'),
                            'model_type': metadata.get('model_type', 'unknown'),
                            'kpis': metadata.get('kpis', [])
                        }
                    }
                    
                    result_models.append(processed_model)
                
                return result_models
    except Exception as e:
        logger.error(f"Error getting user models: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []

def get_model_by_id(model_id):
    """
    Get a model by its ID
    
    Args:
        model_id (str): ID of the model
        
    Returns:
        dict: Model data or None if not found
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id, user_id, name, model_path, metadata, created_at, model_data
                    FROM models 
                    WHERE id = %s
                    """,
                    (model_id,)
                )
                model = cursor.fetchone()
                return model
    except Exception as e:
        logger.error(f"Error getting model by ID: {str(e)}")
        return None

def delete_user_model(user_id: str, model_id: str) -> bool:
    """
    Delete a model from the PostgreSQL database
    
    Args:
        user_id (str): User ID
        model_id (str): Model ID
        
    Returns:
        bool: Success status
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get the model to check if it exists and belongs to the user
                cursor.execute(
                    """
                    SELECT * FROM models 
                    WHERE id = %s AND user_id = %s
                    """,
                    (model_id, user_id)
                )
                model = cursor.fetchone()
                
                if not model:
                    return False
                
                # Delete model file if it exists
                if os.path.exists(model['model_path']):
                    try:
                        os.remove(model['model_path'])
                    except Exception as e:
                        logger.error(f"Error deleting model file: {str(e)}")
                
                # Delete the model from the database
                cursor.execute(
                    """
                    DELETE FROM models
                    WHERE id = %s
                    """,
                    (model_id,)
                )
                
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Error deleting user model: {str(e)}")
        return False

def save_use_cases(user_id: str, filename: str, proposals: List[Dict], metadata=None) -> str:
    """
    Save AI use case proposals for a user in the PostgreSQL database
    
    Args:
        user_id (str): ID of the user
        filename (str): Original filename
        proposals (list): List of use case proposal dictionaries
        metadata (dict, optional): Additional metadata about the file/proposals
    
    Returns:
        str: ID of the saved use cases
    """
    try:
        # First, verify the user exists in the database
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM users WHERE id = %s", (user_id,))
                user = cursor.fetchone()
                
                if not user:
                    logger.warning(f"User with ID {user_id} does not exist in the database")
                    # Option 1: Create a default user with this ID
                    try:
                        cursor.execute(
                            """
                            INSERT INTO users (id, username, email, password_hash, created_at, is_active)
                            VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                            (user_id, f"user_{user_id[:8]}", f"user_{user_id[:8]}@example.com", 
                             generate_password_hash("defaultpassword"), datetime.now(), True)
                        )
                        conn.commit()
                        logger.info(f"Created temporary user with ID {user_id}")
                    except Exception as e:
                        logger.error(f"Error creating user: {e}")
                        # Option 2: If we can't create a user, use a default admin user
                        cursor.execute("SELECT id FROM users WHERE username = 'admin'")
                        admin = cursor.fetchone()
                        if admin:
                            user_id = admin['id']
                            logger.info(f"Using admin user (ID: {user_id}) as fallback")
                        else:
                            raise ValueError(f"User with ID {user_id} does not exist and no admin fallback is available")
        
        # Create a unique ID for this set of use cases
        use_case_id = str(uuid.uuid4())
        
        # Handle the file path if provided
        file_path = None
        if metadata and 'file_path' in metadata and os.path.exists(metadata['file_path']):
            # Create a directory to store the original file
            use_cases_dir = os.path.join(Config.DATABASE_DIR, 'use_cases', str(user_id))
            os.makedirs(use_cases_dir, exist_ok=True)
            file_storage_dir = os.path.join(use_cases_dir, use_case_id)
            os.makedirs(file_storage_dir, exist_ok=True)
            
            # Copy the original file
            original_filename = os.path.basename(metadata['file_path'])
            preserved_file_path = os.path.join(file_storage_dir, original_filename)
            
            try:
                shutil.copy2(metadata['file_path'], preserved_file_path)
                file_path = preserved_file_path
                # Update metadata with the new file path
                if metadata:
                    metadata['preserved_file_path'] = preserved_file_path
            except Exception as e:
                logger.error(f"Error preserving file: {e}")
        
        # Save to PostgreSQL database
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO use_cases (
                        id, user_id, filename, file_path, proposals, created_at, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (use_case_id, user_id, filename, file_path, json.dumps(proposals), 
                     datetime.now(), json.dumps(metadata or {}))
                )
                conn.commit()
        
        return use_case_id
    except Exception as e:
        logger.error(f"Error saving use cases: {str(e)}")
        # Return a dummy ID to avoid breaking the application flow
        return str(uuid.uuid4())

def get_user_use_cases(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all use cases for a user from the PostgreSQL database
    
    Args:
        user_id (str): User ID
        
    Returns:
        List[Dict]: List of use cases
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT * FROM use_cases
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    """,
                    (user_id,)
                )
                
                use_cases = cursor.fetchall()
                
                # Process use cases
                for use_case in use_cases:
                    # Check if file exists
                    use_case['file_exists'] = use_case['file_path'] and os.path.exists(use_case['file_path'])
                    
                    # Get proposal count
                    if use_case['proposals']:
                        use_case['proposal_count'] = len(use_case['proposals'])
                    else:
                        use_case['proposal_count'] = 0
                
                return use_cases
    except Exception as e:
        logger.error(f"Error getting user use cases: {str(e)}")
        return []

def delete_use_case(user_id: str, use_case_id: str) -> bool:
    """
    Delete a use case from the PostgreSQL database
    
    Args:
        user_id (str): User ID
        use_case_id (str): Use case ID
        
    Returns:
        bool: Success status
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get the use case to check if it exists and belongs to the user
                cursor.execute(
                    """
                    SELECT * FROM use_cases 
                    WHERE id = %s AND user_id = %s
                    """,
                    (use_case_id, user_id)
                )
                use_case = cursor.fetchone()
                
                if not use_case:
                    return False
                
                # Delete preserved file if it exists
                if use_case['file_path'] and os.path.exists(use_case['file_path']):
                    try:
                        os.remove(use_case['file_path'])
                    except Exception as e:
                        logger.error(f"Error deleting preserved file: {str(e)}")
                
                # Delete the use case from the database
                cursor.execute(
                    """
                    DELETE FROM use_cases
                    WHERE id = %s
                    """,
                    (use_case_id,)
                )
                
                conn.commit()
                return True
    except Exception as e:
        logger.error(f"Error deleting use case: {str(e)}")
        return False

# Embedding functions
def get_user_embeddings(user_id: str) -> List[Dict[str, Any]]:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT e.*, m.name as model_name, m.model_type
                    FROM embeddings e
                    JOIN models m ON e.model_id::text = m.id::text
                    WHERE e.user_id = %s
                    ORDER BY e.created_at DESC
                    """,
                    (user_id,)
                )
                
                embeddings = cursor.fetchall()
                
                # Process embeddings
                for embedding in embeddings:
                    # Add null check here too
                    embedding['file_exists'] = embedding['embed_path'] is not None and os.path.exists(embedding['embed_path'])
                
                return embeddings
    except Exception as e:
        logger.error(f"Error getting user embeddings: {str(e)}")
        return []

def create_model_embedding(model_id: str, embed_name: str, settings: Dict[str, Any]) -> Optional[str]:
    """
    Create an embedding for a model and save to PostgreSQL
    
    Args:
        model_id (str): Model ID
        embed_name (str): Name for the embedding
        settings (Dict): Embedding settings
        
    Returns:
        Optional[str]: Embedding ID if successful, None otherwise
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get the model to check if it exists
                cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
                model = cursor.fetchone()
                
                if not model:
                    return None
                
                # Create a unique ID for the embedding
                embed_id = uuid.uuid4()
                
                # Create embedding directory in filesystem
                embed_dir = os.path.join('embeddings', str(embed_id))
                os.makedirs(embed_dir, exist_ok=True)
                
                # Copy model file to embedding directory
                if os.path.exists(model['model_path']):
                    model_dest = os.path.join(embed_dir, 'model.joblib')
                    shutil.copy2(model['model_path'], model_dest)
                    
                    # Create a basic HTML template for embedding
                    html_path = os.path.join(embed_dir, 'embed.html')
                    with open(html_path, 'w') as f:
                        f.write(f"""
                        <!DOCTYPE html>
                        <html>
                        <head>
                            <title>{embed_name}</title>
                            <meta charset="UTF-8">
                            <meta name="viewport" content="width=device-width, initial-scale=1.0">
                            <style>
                                body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; }}
                                h1 {{ color: #333; }}
                                .container {{ max-width: 800px; margin: 0 auto; }}
                                .form-group {{ margin-bottom: 15px; }}
                                label {{ display: block; margin-bottom: 5px; }}
                                input {{ width: 100%; padding: 8px; box-sizing: border-box; }}
                                button {{ background-color: #4CAF50; color: white; padding: 10px 15px; border: none; cursor: pointer; }}
                                .result {{ margin-top: 20px; padding: 15px; background-color: #f5f5f5; border-radius: 5px; }}
                            </style>
                        </head>
                        <body>
                            <div class="container">
                                <h1>{embed_name}</h1>
                                <p>Model Type: {model['model_type']}</p>
                                <p>Target Variable: {model['target_variable']}</p>
                                
                                <form id="predictionForm">
                                    <div id="featureInputs"></div>
                                    <button type="submit">Make Prediction</button>
                                </form>
                                
                                <div id="result" class="result" style="display: none;"></div>
                            </div>
                            
                            <script>
                                // Feature names
                                const features = {json.dumps(model['feature_names'])};
                                
                                // Create input fields dynamically
                                const featureInputs = document.getElementById('featureInputs');
                                
                                features.forEach(feature => {{
                                    const formGroup = document.createElement('div');
                                    formGroup.className = 'form-group';
                                    
                                    const label = document.createElement('label');
                                    label.textContent = feature;
                                    
                                    const input = document.createElement('input');
                                    input.type = 'text';
                                    input.name = 'feature_' + feature;
                                    input.placeholder = 'Enter value for ' + feature;
                                    input.required = true;
                                    
                                    formGroup.appendChild(label);
                                    formGroup.appendChild(input);
                                    featureInputs.appendChild(formGroup);
                                }});
                                
                                // Form submission
                                document.getElementById('predictionForm').addEventListener('submit', async function(e) {{
                                    e.preventDefault();
                                    
                                    const formData = new FormData(this);
                                    const data = {{}};
                                    
                                    for(const [key, value] of formData.entries()) {{
                                        data[key] = value;
                                    }}
                                    
                                    try {{
                                        const response = await fetch('/api/predict/{embed_id}', {{
                                            method: 'POST',
                                            headers: {{
                                                'Content-Type': 'application/json'
                                            }},
                                            body: JSON.stringify(data)
                                        }});
                                        
                                        const result = await response.json();
                                        
                                        const resultDiv = document.getElementById('result');
                                        resultDiv.style.display = 'block';
                                        
                                        if(result.error) {{
                                            resultDiv.innerHTML = '<p>Error: ' + result.error + '</p>';
                                        }} else {{
                                            resultDiv.innerHTML = '<h3>Prediction Result</h3>';
                                            resultDiv.innerHTML += '<p>Prediction: <strong>' + result.prediction + '</strong></p>';
                                            
                                            if(result.probability) {{
                                                resultDiv.innerHTML += '<p>Confidence: <strong>' + result.probability + '</strong></p>';
                                            }}
                                        }}
                                    }} catch(error) {{
                                        console.error('Error:', error);
                                        document.getElementById('result').innerHTML = '<p>Error making prediction. Please try again.</p>';
                                        document.getElementById('result').style.display = 'block';
                                    }}
                                }});
                            </script>
                        </body>
                        </html>
                        """)
                    
                    # Save features to a JSON file
                    features_path = os.path.join(embed_dir, 'features.json')
                    with open(features_path, 'w') as f:
                        json.dump(model['feature_names'], f)
                    
                    # Save to PostgreSQL database
                    cursor.execute(
                        """
                        INSERT INTO embeddings (
                            id, user_id, model_id, name, embed_path, settings, created_at
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            embed_id, model['user_id'], model_id, embed_name, embed_dir,
                            json.dumps(settings), datetime.now()
                        )
                    )
                    conn.commit()
                    
                    return str(embed_id)
                else:
                    logger.error(f"Model file not found: {model['model_path']}")
                    return None
                
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error creating embedding: {str(e)}\n{error_trace}")
        return None

def get_embedding_by_id(embed_id: str) -> Optional[Dict[str, Any]]:
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT e.*, m.feature_names, m.model_type, m.target_variable
                    FROM embeddings e
                    JOIN models m ON e.model_id::text = m.id::text
                    WHERE e.id = %s
                    """,
                    (embed_id,)
                )
                
                embedding = cursor.fetchone()
                
                if embedding:
                    # Create a metadata dictionary for the embedding
                    embedding['metadata'] = {
                        'features': embedding['feature_names'],
                        'model_type': embedding['model_type'],
                        'target_variable': embedding['target_variable']
                    }
                    
                    # Add null check here too
                    embedding['file_exists'] = embedding['embed_path'] is not None and os.path.exists(embedding['embed_path'])
                
                return embedding
    except Exception as e:
        logger.error(f"Error getting embedding by ID: {str(e)}")
        return None

def delete_embedding(embed_id: str) -> bool:
    """
    Delete an embedding from PostgreSQL database
    
    Args:
        embed_id (str): Embedding ID
        
    Returns:
        bool: Success status
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Get the embedding to check if it exists
                cursor.execute("SELECT * FROM embeddings WHERE id = %s", (embed_id,))
                embedding = cursor.fetchone()
                
                if not embedding:
                    return False
                
                # Delete embedding directory if it exists
                if embedding['embed_path'] and os.path.exists(embedding['embed_path']):
                    try:
                        shutil.rmtree(embedding['embed_path'])
                    except Exception as e:
                        logger.error(f"Error deleting embedding directory: {str(e)}")
                
                # Delete the embedding from the database
                cursor.execute("DELETE FROM embeddings WHERE id = %s", (embed_id,))
                conn.commit()
                
                return True
    except Exception as e:
        logger.error(f"Error deleting embedding: {str(e)}")
        return False

def get_embed_code(embed_id: str, embed_url: str = None) -> str:
    """
    Get the HTML embed code for an embedding
    
    Args:
        embed_id (str): Embedding ID
        embed_url (str, optional): Base URL for embedding, if None uses relative URL
        
    Returns:
        str: HTML embed code
    """
    # If no embed_url provided, use relative URL
    if not embed_url:
        embed_url = f"/embedded/{embed_id}"
    
    return f"""
    <iframe src="{embed_url}" width="100%" height="600px" frameborder="0"></iframe>
    """

def init_auth_routes(app):
    """
    Manual initialization of auth routes for applications that
    don't support the automatic init_auth_routes function.
    
    Args:
        app: Flask application instance
    """
    from flask import session, request, redirect, url_for, flash, render_template
    
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
            
            # Create session with client info
            ip_address = request.remote_addr
            user_agent = request.user_agent.string if request.user_agent else None
            session_token = create_session(user['id'], remember, ip_address, user_agent)
            
            if not session_token:
                flash('Error creating session. Please try again.', 'error')
                return render_template('login.html')
            
            # Set session cookies
            session['user_id'] = str(user['id'])
            session['session_token'] = session_token
            session['user_email'] = email
            session['user_name'] = f"{user['first_name'] or ''} {user['last_name'] or ''}".strip() or user['username']
            
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
            min_length = Config.PASSWORD_MIN_LENGTH
            if len(password) < min_length:
                flash(f'Password must be at least {min_length} characters long', 'error')
                return render_template('register.html')
            
            # Create user
            user_id = create_user(first_name, last_name, email, password, organization)
            
            if not user_id:
                flash('Error creating user. Please try again.', 'error')
                return render_template('register.html')
            
            # Create session with client info
            ip_address = request.remote_addr
            user_agent = request.user_agent.string if request.user_agent else None
            session_token = create_session(user_id, False, ip_address, user_agent)
            
            if not session_token:
                flash('Registration successful! Please log in.', 'success')
                return redirect(url_for('login'))
            
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
        
        min_length = Config.PASSWORD_MIN_LENGTH
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
    
    