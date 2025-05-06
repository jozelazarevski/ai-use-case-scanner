import psycopg2
import psycopg2.extras
import json
import uuid
import os
from datetime import datetime
from contextlib import contextmanager
from config import Config

# Register UUID adapter for psycopg2
psycopg2.extras.register_uuid()

# Database connection parameters
DB_PARAMS = {
    'dbname': 'ml_app',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': '5432'
}

# Extract connection params from SQLALCHEMY_DATABASE_URI if available
if hasattr(Config, 'SQLALCHEMY_DATABASE_URI'):
    uri = Config.SQLALCHEMY_DATABASE_URI
    if uri.startswith('postgresql://'):
        # Parse the URI
        parts = uri.replace('postgresql://', '').split('@')
        if len(parts) == 2:
            user_pass, host_port_db = parts
            user_parts = user_pass.split(':')
            if len(user_parts) == 2:
                DB_PARAMS['user'] = user_parts[0]
                DB_PARAMS['password'] = user_parts[1]
            
            host_parts = host_port_db.split('/')
            if len(host_parts) >= 2:
                DB_PARAMS['dbname'] = host_parts[1]
                host_port = host_parts[0].split(':')
                if len(host_port) == 2:
                    DB_PARAMS['host'] = host_port[0]
                    DB_PARAMS['port'] = host_port[1]
                else:
                    DB_PARAMS['host'] = host_port[0]

@contextmanager
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
        conn = psycopg2.connect(**DB_PARAMS, cursor_factory=psycopg2.extras.RealDictCursor)
        yield conn
    except Exception as e:
        if conn:
            conn.rollback()
        raise e
    finally:
        if conn:
            conn.close()

def initialize_database():
    """
    Create database tables if they don't exist.
    """
    try:
        # First, create the database if it doesn't exist
        create_database_if_not_exists()
        
        # Then create tables
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                # Read the SQL file with table definitions
                sql_file_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
                if not os.path.exists(sql_file_path):
                    # If schema.sql doesn't exist, use the hardcoded schema
                    sql_statements = """
                    -- Create extension for UUID support
                    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
                    
                    -- Users table
                    CREATE TABLE IF NOT EXISTS users (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        username VARCHAR(100) NOT NULL UNIQUE,
                        email VARCHAR(100) NOT NULL UNIQUE,
                        password_hash VARCHAR(256) NOT NULL,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP WITH TIME ZONE,
                        is_active BOOLEAN DEFAULT TRUE
                    );
                    
                    -- Create indexes on users table if they don't exist
                    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                    CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
                    CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);
                    
                    -- Models table
                    CREATE TABLE IF NOT EXISTS models (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        name VARCHAR(100) NOT NULL,
                        description TEXT,
                        model_path VARCHAR(255) NOT NULL,
                        model_type VARCHAR(50),
                        target_variable VARCHAR(100),
                        accuracy FLOAT,
                        feature_names JSONB,
                        feature_importance JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Create indexes on models table if they don't exist
                    CREATE INDEX IF NOT EXISTS idx_models_user_id ON models(user_id);
                    CREATE INDEX IF NOT EXISTS idx_models_model_type ON models(model_type);
                    CREATE INDEX IF NOT EXISTS idx_models_target_variable ON models(target_variable);
                    CREATE INDEX IF NOT EXISTS idx_models_created_at ON models(created_at);
                    
                    -- Use cases table
                    CREATE TABLE IF NOT EXISTS use_cases (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        filename VARCHAR(255) NOT NULL,
                        file_path VARCHAR(255),
                        proposals JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB
                    );
                    
                    -- Create indexes on use_cases table if they don't exist
                    CREATE INDEX IF NOT EXISTS idx_use_cases_user_id ON use_cases(user_id);
                    CREATE INDEX IF NOT EXISTS idx_use_cases_created_at ON use_cases(created_at);
                    
                    -- Embeddings table
                    CREATE TABLE IF NOT EXISTS embeddings (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
                        name VARCHAR(100) NOT NULL,
                        embed_path VARCHAR(255),
                        settings JSONB,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Create indexes on embeddings table if they don't exist
                    CREATE INDEX IF NOT EXISTS idx_embeddings_user_id ON embeddings(user_id);
                    CREATE INDEX IF NOT EXISTS idx_embeddings_model_id ON embeddings(model_id);
                    CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at);
                    
                    -- Prediction logs table
                    CREATE TABLE IF NOT EXISTS prediction_logs (
                        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                        user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        model_id UUID NOT NULL REFERENCES models(id) ON DELETE CASCADE,
                        input_data JSONB,
                        prediction_result JSONB,
                        timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Create indexes on prediction_logs table if they don't exist
                    CREATE INDEX IF NOT EXISTS idx_prediction_logs_user_id ON prediction_logs(user_id);
                    CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_id ON prediction_logs(model_id);
                    CREATE INDEX IF NOT EXISTS idx_prediction_logs_timestamp ON prediction_logs(timestamp);
                    
                    -- Create function to automatically update updated_at timestamp if it doesn't exist
                    CREATE OR REPLACE FUNCTION update_modified_column()
                    RETURNS TRIGGER AS $$
                    BEGIN
                        NEW.updated_at = CURRENT_TIMESTAMP;
                        RETURN NEW;
                    END;
                    $$ LANGUAGE plpgsql;
                    
                    -- Create trigger to update the updated_at column in the models table if it doesn't exist
                    DROP TRIGGER IF EXISTS update_models_updated_at ON models;
                    CREATE TRIGGER update_models_updated_at
                    BEFORE UPDATE ON models
                    FOR EACH ROW
                    EXECUTE FUNCTION update_modified_column();
                    """
                else:
                    with open(sql_file_path, 'r') as f:
                        sql_statements = f.read()
                
                cursor.execute(sql_statements)
                conn.commit()
                
        # Create admin user if it doesn't exist
        create_admin_user()
        
        print("Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
        raise

def create_database_if_not_exists():
    """
    Create the PostgreSQL database if it doesn't exist.
    """
    db_name = DB_PARAMS['dbname']
    
    # Temporarily connect to 'postgres' database to check if our database exists
    temp_params = DB_PARAMS.copy()
    temp_params['dbname'] = 'postgres'
    
    try:
        conn = psycopg2.connect(**temp_params)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Check if database exists
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (db_name,))
        exists = cursor.fetchone()
        
        if not exists:
            print(f"Creating database {db_name}...")
            cursor.execute(f"CREATE DATABASE {db_name}")
            print(f"Database {db_name} created successfully.")
        
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error creating database: {str(e)}")
        raise

def create_admin_user():
    """
    Create admin user if it doesn't exist.
    """
    from werkzeug.security import generate_password_hash
    
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Check if admin user exists
            cursor.execute("SELECT * FROM users WHERE username = %s", ('admin',))
            admin_user = cursor.fetchone()
            
            if not admin_user:
                # Create admin user
                cursor.execute(
                    """
                    INSERT INTO users (username, email, password_hash, created_at, is_active)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    ('admin', 'admin@example.com', generate_password_hash('admin123'), datetime.now(), True)
                )
                admin_id = cursor.fetchone()['id']
                conn.commit()
                print(f"Admin user created with ID: {admin_id}")

# User operations
def get_user_by_id(user_id):
    """Get user by ID"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            return cursor.fetchone()

def get_user_by_username(username):
    """Get user by username"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            return cursor.fetchone()

def create_user(username, email, password_hash):
    """Create a new user"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            try:
                cursor.execute(
                    """
                    INSERT INTO users (username, email, password_hash, created_at, is_active)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (username, email, password_hash, datetime.now(), True)
                )
                user_id = cursor.fetchone()['id']
                conn.commit()
                return user_id
            except psycopg2.errors.UniqueViolation:
                conn.rollback()
                return None

def update_user_last_login(user_id):
    """Update user's last login time"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                UPDATE users
                SET last_login = %s
                WHERE id = %s
                """,
                (datetime.now(), user_id)
            )
            conn.commit()

# Model operations
def save_model(user_id, name, description, model_path, model_type, target_variable, accuracy, feature_names, feature_importance):
    """Save a model to the database"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Convert Python objects to JSON
            feature_names_json = json.dumps(feature_names)
            feature_importance_json = json.dumps(feature_importance)
            
            # Generate a new UUID
            model_id = uuid.uuid4()
            
            cursor.execute(
                """
                INSERT INTO models (
                    id, user_id, name, description, model_path, model_type, 
                    target_variable, accuracy, feature_names, feature_importance,
                    created_at, updated_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    model_id, user_id, name, description, model_path, model_type, 
                    target_variable, accuracy, feature_names_json, feature_importance_json,
                    datetime.now(), datetime.now()
                )
            )
            
            conn.commit()
            return model_id

def get_user_models(user_id):
    """Get all models for a user"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT * FROM models
                WHERE user_id = %s
                ORDER BY created_at DESC
                """,
                (user_id,)
            )
            
            models = cursor.fetchall()
            
            # Check if model files exist
            for model in models:
                model['file_exists'] = os.path.exists(model['model_path'])
            
            return models

def get_model_by_id(model_id):
    """Get a model by ID"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM models WHERE id = %s", (model_id,))
            model = cursor.fetchone()
            
            if model:
                # Check if model file exists
                model['file_exists'] = os.path.exists(model['model_path'])
                
                # Create a metadata dictionary for compatibility with old code
                model['metadata'] = {
                    'features': model['feature_names'],
                    'feature_importance': model['feature_importance'],
                    'accuracy': model['accuracy'],
                    'description': model['description'],
                    'title': model['name'],
                    'target_variable': model['target_variable'],
                    'model_type': model['model_type']
                }
            
            return model

def delete_model(user_id, model_id):
    """Delete a model"""
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
                    print(f"Error deleting model file: {str(e)}")
            
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

# Use Case operations
def save_use_case(user_id, filename, proposals, file_path=None, metadata=None):
    """Save a use case to the database"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Convert Python objects to JSON
            proposals_json = json.dumps(proposals)
            metadata_json = json.dumps(metadata or {})
            
            # Generate a new UUID
            use_case_id = uuid.uuid4()
            
            cursor.execute(
                """
                INSERT INTO use_cases (
                    id, user_id, filename, file_path, proposals, created_at, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (use_case_id, user_id, filename, file_path, proposals_json, datetime.now(), metadata_json)
            )
            
            conn.commit()
            return use_case_id

def get_user_use_cases(user_id):
    """Get all use cases for a user"""
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
            
            # Process the use cases
            for use_case in use_cases:
                # Check if file exists
                use_case['file_exists'] = use_case['file_path'] and os.path.exists(use_case['file_path'])
                
                # Get proposal count
                if use_case['proposals']:
                    use_case['proposal_count'] = len(use_case['proposals'])
                else:
                    use_case['proposal_count'] = 0
            
            return use_cases

def get_use_case_by_id(use_case_id):
    """Get a use case by ID"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT * FROM use_cases WHERE id = %s", (use_case_id,))
            return cursor.fetchone()

def delete_use_case(user_id, use_case_id):
    """Delete a use case"""
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
                    print(f"Error deleting preserved file: {str(e)}")
            
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

# Embedding operations
def create_embedding(user_id, model_id, name, embed_path, settings):
    """Create an embedding for a model"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Convert Python objects to JSON
            settings_json = json.dumps(settings or {})
            
            # Generate a new UUID
            embed_id = uuid.uuid4()
            
            cursor.execute(
                """
                INSERT INTO embeddings (
                    id, user_id, model_id, name, embed_path, settings, created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (embed_id, user_id, model_id, name, embed_path, settings_json, datetime.now())
            )
            
            conn.commit()
            return embed_id

def get_user_embeddings(user_id):
    """Get all embeddings for a user"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT e.*, m.name as model_name, m.model_type
                FROM embeddings e
                JOIN models m ON e.model_id = m.id
                WHERE e.user_id = %s
                ORDER BY e.created_at DESC
                """,
                (user_id,)
            )
            
            embeddings = cursor.fetchall()
            
            # Process the embeddings
            for embed in embeddings:
                # Check if embedding files exist
                embed_exists = (embed['embed_path'] and os.path.exists(embed['embed_path']) and 
                               os.path.exists(os.path.join(embed['embed_path'], "embed.html")))
                embed['exists'] = embed_exists
                
                # Get embed code
                embed['embed_code'] = get_embed_code(embed['id'])
                embed['embed_url'] = f"/embedded/{embed['id']}"
            
            return embeddings

def get_embedding_by_id(embed_id):
    """Get an embedding by ID"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT e.*, m.model_type, m.target_variable, m.feature_names, m.accuracy
                FROM embeddings e
                JOIN models m ON e.model_id = m.id
                WHERE e.id = %s
                """,
                (embed_id,)
            )
            
            embed = cursor.fetchone()
            
            if embed:
                # Create a metadata dictionary for compatibility with old code
                embed['metadata'] = {
                    'model_type': embed['model_type'],
                    'target_variable': embed['target_variable'],
                    'features': embed['feature_names'],
                    'accuracy': embed['accuracy']
                }
            
            return embed

def delete_embedding(user_id, embed_id):
    """Delete an embedding"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Get the embedding to check if it exists and belongs to the user
            cursor.execute(
                """
                SELECT * FROM embeddings 
                WHERE id = %s AND user_id = %s
                """,
                (embed_id, user_id)
            )
            embed = cursor.fetchone()
            
            if not embed:
                return False
            
            # Delete embedding directory if it exists
            if embed['embed_path'] and os.path.exists(embed['embed_path']):
                try:
                    import shutil
                    shutil.rmtree(embed['embed_path'])
                except Exception as e:
                    print(f"Error deleting embedding directory: {str(e)}")
            
            # Delete the embedding from the database
            cursor.execute(
                """
                DELETE FROM embeddings
                WHERE id = %s
                """,
                (embed_id,)
            )
            
            conn.commit()
            return True

def get_embed_code(embed_id):
    """Get HTML embed code for an embedding"""
    embed_url = f"<iframe src='/embedded/{embed_id}' width='100%' height='600px' frameborder='0'></iframe>"
    return embed_url

# Prediction Log operations
def log_prediction(user_id, model_id, input_data, prediction_result):
    """Log a prediction"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            # Convert Python objects to JSON
            input_data_json = json.dumps(input_data)
            prediction_result_json = json.dumps(prediction_result)
            
            # Generate a new UUID
            log_id = uuid.uuid4()
            
            cursor.execute(
                """
                INSERT INTO prediction_logs (
                    id, user_id, model_id, input_data, prediction_result, timestamp
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (log_id, user_id, model_id, input_data_json, prediction_result_json, datetime.now())
            )
            
            conn.commit()
            return log_id

def get_user_prediction_logs(user_id, limit=100):
    """Get prediction logs for a user"""
    with get_db_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                SELECT pl.*, m.name as model_name, m.model_type
                FROM prediction_logs pl
                JOIN models m ON pl.model_id = m.id
                WHERE pl.user_id = %s
                ORDER BY pl.timestamp DESC
                LIMIT %s
                """,
                (user_id, limit)
            )
            
            return cursor.fetchall()

# If this file is run directly, initialize the database
if __name__ == "__main__":
    initialize_database()
