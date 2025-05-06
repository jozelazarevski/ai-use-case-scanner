import os
import json
import uuid
import time
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

# Database configuration
DATABASE_DIR = 'databases'
USERS_DB_FILE = os.path.join(DATABASE_DIR, 'users.json')
SESSIONS_DB_FILE = os.path.join(DATABASE_DIR, 'sessions.json')
MODELS_DB_FILE = os.path.join(DATABASE_DIR, 'user_models.json')
EMBEDDINGS_DB_FILE = os.path.join(DATABASE_DIR, 'model_embeddings.json')

# Ensure database directory exists
os.makedirs(DATABASE_DIR, exist_ok=True)

# Initialize database files if they don't exist
def init_database():
    """Initialize the database files if they don't exist."""
    if not os.path.exists(USERS_DB_FILE):
        with open(USERS_DB_FILE, 'w') as f:
            json.dump({}, f)
    
    if not os.path.exists(SESSIONS_DB_FILE):
        with open(SESSIONS_DB_FILE, 'w') as f:
            json.dump({}, f)
    
    if not os.path.exists(MODELS_DB_FILE):
        with open(MODELS_DB_FILE, 'w') as f:
            json.dump({}, f)
    
    if not os.path.exists(EMBEDDINGS_DB_FILE):
        with open(EMBEDDINGS_DB_FILE, 'w') as f:
            json.dump({}, f)

# Initialize databases on module import
init_database()

# User Functions
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
    expiry = int(time.time()) + (30 * 24 * 60 * 60 if remember else 24 * 60 * 60)  # 30 days or 1 day
    
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

# Model Functions
def save_user_model(user_id, model_name, model_path, metadata=None):
    """
    Save a user's model to the database.
    
    Args:
        user_id (str): User's ID
        model_name (str): Name of the model
        model_path (str): Path to the model file
        metadata (dict, optional): Additional model metadata
        
    Returns:
        str: Model ID if saved successfully, None otherwise
    """
    try:
        with open(MODELS_DB_FILE, 'r') as f:
            models = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        models = {}
    
    # Create new model entry
    model_id = str(uuid.uuid4())
    models[model_id] = {
        'id': model_id,
        'user_id': user_id,
        'name': model_name,
        'path': model_path,
        'metadata': metadata or {},
        'created_at': datetime.now().isoformat()
    }
    
    # Save to database
    with open(MODELS_DB_FILE, 'w') as f:
        json.dump(models, f, indent=2)
    
    return model_id

def get_user_models(user_id):
    """
    Get all models for a user.
    
    Args:
        user_id (str): User's ID
        
    Returns:
        list: List of model data dictionaries
    """
    try:
        with open(MODELS_DB_FILE, 'r') as f:
            models = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []
    
    user_models = []
    for model_id, model_data in models.items():
        if model_data.get('user_id') == user_id:
            user_models.append(model_data)
    
    # Sort by creation date, newest first
    user_models.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return user_models

def get_model_by_id(model_id):
    """
    Get model data by ID.
    
    Args:
        model_id (str): Model's ID
        
    Returns:
        dict: Model data if found, None otherwise
    """
    try:
        with open(MODELS_DB_FILE, 'r') as f:
            models = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None
    
    return models.get(model_id)

def delete_user_model(user_id, model_id):
    """
    Delete a user's model from the database.
    
    Args:
        user_id (str): User's ID
        model_id (str): Model's ID
        
    Returns:
        bool: True if model was deleted, False otherwise
    """
    try:
        with open(MODELS_DB_FILE, 'r') as f:
            models = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return False
    
    # Check if model exists and belongs to user
    if model_id in models and models[model_id].get('user_id') == user_id:
        # Delete model file if it exists
        model_path = models[model_id].get('path')
        if model_path and os.path.exists(model_path):
            try:
                os.remove(model_path)
            except:
                pass  # Continue even if file deletion fails
        
        # Delete model from database
        del models[model_id]
        with open(MODELS_DB_FILE, 'w') as f:
            json.dump(models, f, indent=2)
        
        # Also delete any embeddings for this model
        delete_model_embeddings(model_id)
        
        return True
    
    return False

# Embedding Functions
def create_model_embedding(model_id, embed_name, embed_settings=None):
    """
    Create an embeddable version of a model.
    
    Args:
        model_id (str): ID of the model to embed
        embed_name (str): Name for the embedding
        embed_settings (dict, optional): Settings for the embedding
        
    Returns:
        str: Embedding ID if created successfully, None otherwise
    """
    # First check if model exists
    model_data = get_model_by_id(model_id)
    if not model_data:
        return None
    
    try:
        with open(EMBEDDINGS_DB_FILE, 'r') as f:
            embeddings = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        embeddings = {}
    
    # Create embeddings directory if it doesn't exist
    embeddings_dir = 'model_embeddings'
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Create a unique embedding ID
    embed_id = str(uuid.uuid4())
    embed_path = os.path.join(embeddings_dir, embed_id)
    os.makedirs(embed_path, exist_ok=True)
    
    # Create embed.html file (simplified for example)
    embed_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{embed_name}</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    </head>
    <body>
        <div class="container mt-4">
            <h1>{embed_name}</h1>
            <div id="modelWidget">
                <!-- Model widget content would go here -->
                <p>This is an embedded model widget.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(os.path.join(embed_path, 'embed.html'), 'w') as f:
        f.write(embed_html)
    
    # Copy model file to embedding directory
    import shutil
    try:
        model_dest = os.path.join(embed_path, 'model.joblib')
        shutil.copy(model_data['path'], model_dest)
    except Exception as e:
        print(f"Error copying model file: {str(e)}")
    
    # Save embedding metadata
    with open(os.path.join(embed_path, 'features.json'), 'w') as f:
        json.dump(model_data.get('metadata', {}).get('features', []), f)
    
    # Record embedding in database
    embeddings[embed_id] = {
        'id': embed_id,
        'model_id': model_id,
        'user_id': model_data['user_id'],
        'name': embed_name,
        'path': embed_path,
        'settings': embed_settings or {},
        'created_at': datetime.now().isoformat()
    }
    
    with open(EMBEDDINGS_DB_FILE, 'w') as f:
        json.dump(embeddings, f, indent=2)
    
    return embed_id

def get_user_embeddings(user_id):
    """
    Get all model embeddings for a user.
    
    Args:
        user_id (str): User's ID
        
    Returns:
        list: List of embedding data dictionaries
    """
    try:
        with open(EMBEDDINGS_DB_FILE, 'r') as f:
            embeddings = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []
    
    user_embeddings = []
    for embed_id, embed_data in embeddings.items():
        if embed_data.get('user_id') == user_id:
            user_embeddings.append(embed_data)
    
    # Sort by creation date, newest first
    user_embeddings.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return user_embeddings

def get_embedding_by_id(embed_id):
    """
    Get embedding data by ID.
    
    Args:
        embed_id (str): Embedding's ID
        
    Returns:
        dict: Embedding data if found, None otherwise
    """
    try:
        with open(EMBEDDINGS_DB_FILE, 'r') as f:
            embeddings = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None
    
    return embeddings.get(embed_id)

def delete_embedding(embed_id):
    """
    Delete a model embedding from the database.
    
    Args:
        embed_id (str): Embedding's ID
        
    Returns:
        bool: True if embedding was deleted, False otherwise
    """
    try:
        with open(EMBEDDINGS_DB_FILE, 'r') as f:
            embeddings = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return False
    
    # Check if embedding exists
    if embed_id in embeddings:
        # Delete embedding files if they exist
        embed_path = embeddings[embed_id].get('path')
        if embed_path and os.path.exists(embed_path):
            try:
                import shutil
                shutil.rmtree(embed_path)
            except:
                pass  # Continue even if file deletion fails
        
        # Delete embedding from database
        del embeddings[embed_id]
        with open(EMBEDDINGS_DB_FILE, 'w') as f:
            json.dump(embeddings, f, indent=2)
        
        return True
    
    return False

def delete_model_embeddings(model_id):
    """
    Delete all embeddings for a model.
    
    Args:
        model_id (str): Model's ID
        
    Returns:
        int: Number of embeddings deleted
    """
    try:
        with open(EMBEDDINGS_DB_FILE, 'r') as f:
            embeddings = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return 0
    
    embed_ids_to_delete = []
    for embed_id, embed_data in embeddings.items():
        if embed_data.get('model_id') == model_id:
            embed_ids_to_delete.append(embed_id)
    
    deleted_count = 0
    for embed_id in embed_ids_to_delete:
        if delete_embedding(embed_id):
            deleted_count += 1
    
    return deleted_count

def get_embed_code(embed_id):
    """
    Generate HTML embed code for an embedding.
    
    Args:
        embed_id (str): Embedding's ID
        
    Returns:
        str: HTML embed code
    """
    embed_data = get_embedding_by_id(embed_id)
    if not embed_data:
        return None
    
    embed_url = f"/embedded/{embed_id}"
    
    # Generate HTML embed code
    embed_code = f"""
    <iframe 
        src="{embed_url}" 
        width="100%" 
        height="600px" 
        style="border:none;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.1);" 
        title="{embed_data.get('name', 'Embedded AI Model')}"
    ></iframe>
    """
    
    return embed_code