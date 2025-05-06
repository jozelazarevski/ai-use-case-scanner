import os
import json
import shutil
import datetime
import uuid
import sys
import argparse
from werkzeug.security import generate_password_hash

# Add app directory to path
sys.path.append('.')

# Import the database connection module
import db_connection as db

def migrate_users():
    """Migrate users from file-based system to database"""
    print("Migrating users...")
    
    users_dir = 'database/users'
    if not os.path.exists(users_dir):
        print("No users directory found. Skipping user migration.")
        return
    
    total_users = 0
    migrated_users = 0
    
    for filename in os.listdir(users_dir):
        if filename.endswith('.json'):
            user_path = os.path.join(users_dir, filename)
            total_users += 1
            
            try:
                with open(user_path, 'r') as f:
                    user_data = json.load(f)
                
                # Convert string UUID to UUID object for PostgreSQL
                try:
                    user_id = uuid.UUID(user_data.get('id'))
                except (ValueError, TypeError):
                    user_id = uuid.uuid4()
                
                # Check if user already exists in the database
                existing_user = db.get_user_by_username(user_data.get('username'))
                if existing_user:
                    print(f"User {user_data.get('username')} already exists in database. Skipping.")
                    continue
                
                # Insert user into database using SQL
                with db.get_db_connection() as conn:
                    with conn.cursor() as cursor:
                        # Prepare password hash
                        password_hash = user_data.get('password_hash') or generate_password_hash('password')
                        
                        # Prepare creation date
                        if 'created_at' in user_data:
                            try:
                                created_at = datetime.datetime.fromisoformat(user_data.get('created_at'))
                            except (ValueError, TypeError):
                                created_at = datetime.datetime.now()
                        else:
                            created_at = datetime.datetime.now()
                        
                        # Prepare last login date
                        if 'last_login' in user_data and user_data.get('last_login'):
                            try:
                                last_login = datetime.datetime.fromisoformat(user_data.get('last_login'))
                            except (ValueError, TypeError):
                                last_login = None
                        else:
                            last_login = None
                        
                        cursor.execute(
                            """
                            INSERT INTO users (id, username, email, password_hash, created_at, last_login, is_active)
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                            """,
                            (
                                user_id,
                                user_data.get('username'),
                                user_data.get('email', f"{user_data.get('username')}@example.com"),
                                password_hash,
                                created_at,
                                last_login,
                                True
                            )
                        )
                        conn.commit()
                
                migrated_users += 1
                print(f"Migrated user: {user_data.get('username')}")
                
            except Exception as e:
                print(f"Error migrating user from {user_path}: {str(e)}")
    
    print(f"Migrated {migrated_users} out of {total_users} users.")

def migrate_models():
    """Migrate models from file-based system to database"""
    print("Migrating models...")
    
    models_dir = 'database/user_models'
    if not os.path.exists(models_dir):
        print("No models directory found. Skipping model migration.")
        return
    
    total_models = 0
    migrated_models = 0
    
    # Iterate through user directories
    for user_dir in os.listdir(models_dir):
        user_models_dir = os.path.join(models_dir, user_dir)
        if not os.path.isdir(user_models_dir):
            continue
        
        # Convert string UUID to UUID object for PostgreSQL
        try:
            user_id = uuid.UUID(user_dir)
        except (ValueError, TypeError):
            print(f"Invalid user ID format: {user_dir}. Skipping models for this user.")
            continue
        
        # Get user from database
        user = db.get_user_by_id(user_id)
        if not user:
            print(f"User ID {user_dir} not found in database. Skipping models for this user.")
            continue
        
        # Iterate through model files
        for filename in os.listdir(user_models_dir):
            if filename.endswith('.json'):
                model_path = os.path.join(user_models_dir, filename)
                total_models += 1
                
                try:
                    with open(model_path, 'r') as f:
                        model_data = json.load(f)
                    
                    # Convert string UUID to UUID object for PostgreSQL
                    try:
                        model_id = uuid.UUID(model_data.get('id'))
                    except (ValueError, TypeError):
                        model_id = uuid.uuid4()
                    
                    # Check if model already exists
                    with db.get_db_connection() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT id FROM models WHERE id = %s", (model_id,))
                            existing_model = cursor.fetchone()
                    
                    if existing_model:
                        print(f"Model {model_data.get('id')} already exists in database. Skipping.")
                        continue
                    
                    # Prepare model data
                    metadata = model_data.get('metadata', {})
                    name = model_data.get('name', 'Untitled Model')
                    description = metadata.get('description', '')
                    model_type = metadata.get('model_type', 'unknown')
                    target_variable = metadata.get('target_variable', '')
                    accuracy = metadata.get('accuracy')
                    feature_names = metadata.get('features', [])
                    feature_importance = metadata.get('feature_importance', {})
                    model_path = model_data.get('path')
                    
                    # Prepare creation date
                    if 'created_at' in model_data:
                        try:
                            created_at = datetime.datetime.fromisoformat(model_data.get('created_at'))
                        except (ValueError, TypeError):
                            created_at = datetime.datetime.now()
                    else:
                        created_at = datetime.datetime.now()
                    
                    # Insert model into database using SQL
                    with db.get_db_connection() as conn:
                        with conn.cursor() as cursor:
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
                                    target_variable, accuracy, json.dumps(feature_names), json.dumps(feature_importance),
                                    created_at, created_at
                                )
                            )
                            conn.commit()
                    
                    migrated_models += 1
                    print(f"Migrated model: {name} for user {user['username']}")
                    
                except Exception as e:
                    print(f"Error migrating model from {model_path}: {str(e)}")
    
    print(f"Migrated {migrated_models} out of {total_models} models.")

def migrate_use_cases():
    """Migrate use cases from file-based system to database"""
    print("Migrating use cases...")
    
    use_cases_dir = 'database/use_cases'
    if not os.path.exists(use_cases_dir):
        print("No use cases directory found. Skipping use case migration.")
        return
    
    total_use_cases = 0
    migrated_use_cases = 0
    
    # Iterate through user directories
    for user_dir in os.listdir(use_cases_dir):
        user_use_cases_dir = os.path.join(use_cases_dir, user_dir)
        if not os.path.isdir(user_use_cases_dir):
            continue
        
        # Convert string UUID to UUID object for PostgreSQL
        try:
            user_id = uuid.UUID(user_dir)
        except (ValueError, TypeError):
            print(f"Invalid user ID format: {user_dir}. Skipping use cases for this user.")
            continue
        
        # Get user from database
        user = db.get_user_by_id(user_id)
        if not user:
            print(f"User ID {user_dir} not found in database. Skipping use cases for this user.")
            continue
        
        # Iterate through use case files
        for filename in os.listdir(user_use_cases_dir):
            if filename.endswith('.json'):
                use_case_path = os.path.join(user_use_cases_dir, filename)
                total_use_cases += 1
                
                try:
                    with open(use_case_path, 'r') as f:
                        use_case_data = json.load(f)
                    
                    # Convert string UUID to UUID object for PostgreSQL
                    try:
                        use_case_id = uuid.UUID(use_case_data.get('id'))
                    except (ValueError, TypeError):
                        use_case_id = uuid.uuid4()
                    
                    # Check if use case already exists
                    with db.get_db_connection() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute("SELECT id FROM use_cases WHERE id = %s", (use_case_id,))
                            existing_use_case = cursor.fetchone()
                    
                    if existing_use_case:
                        print(f"Use case {use_case_data.get('id')} already exists in database. Skipping.")
                        continue
                    
                    # Prepare use case data
                    filename = use_case_data.get('filename', 'Unknown file')
                    proposals = use_case_data.get('proposals', [])
                    metadata = use_case_data.get('metadata', {})
                    file_path = metadata.get('preserved_file_path', None)
                    
                    # Prepare creation date
                    if 'created_at' in use_case_data:
                        try:
                            created_at = datetime.datetime.fromisoformat(use_case_data.get('created_at'))
                        except (ValueError, TypeError):
                            created_at = datetime.datetime.now()
                    else:
                        created_at = datetime.datetime.now()
                    
                    # Insert use case into database using SQL
                    with db.get_db_connection() as conn:
                        with conn.cursor() as cursor:
                            cursor.execute(
                                """
                                INSERT INTO use_cases (
                                    id, user_id, filename, file_path, proposals, created_at, metadata
                                )
                                VALUES (%s, %s, %s, %s, %s, %s, %s)
                                RETURNING id
                                """,
                                (
                                    use_case_id, user_id, filename, file_path, 
                                    json.dumps(proposals), created_at, json.dumps(metadata)
                                )
                            )
                            conn.commit()
                    
                    migrated_use_cases += 1
                    print(f"Migrated use case for file: {filename} for user {user['username']}")
                    
                except Exception as e:
                    print(f"Error migrating use case from {use_case_path}: {str(e)}")
    
    print(f"Migrated {migrated_use_cases} out of {total_use_cases} use cases.")

def migrate_embeddings():
    """Migrate embeddings from file-based system to database"""
    print("Migrating embeddings...")
    
    embeddings_dir = 'database/embeddings'
    if not os.path.exists(embeddings_dir):
        print("No embeddings directory found. Skipping embedding migration.")
        return
    
    total_embeddings = 0
    migrated_embeddings = 0
    
    # Iterate through user directories
    for user_dir in os.listdir(embeddings_dir):
        user_embeddings_dir = os.path.join(embeddings_dir, user_dir)
        if not os.path.isdir(user_embeddings_dir):
            continue
        
        # Convert string UUID to UUID object for PostgreSQL
        try:
            user_id = uuid.UUID(user_dir)
        except (ValueError, TypeError):
            print(f"Invalid user ID format: {user_dir}. Skipping embeddings for this user.")
            continue
        
        # Get user from database
        user = db.get_user_by_id(user_id)
        if not user:
            print(f"User ID {user_dir} not found in database. Skipping embeddings for this user.")
            continue
        
        # Iterate through embedding directories
        for embed_dir_name in os.listdir(user_embeddings_dir):
            embed_dir = os.path.join(user_embeddings_dir, embed_dir_name)
            if not os.path.isdir(embed_dir):
                continue
            
            # Convert string UUID to UUID object for PostgreSQL
            try:
                embed_id = uuid.UUID(embed_dir_name)
            except (ValueError, TypeError):
                print(f"Invalid embedding ID format: {embed_dir_name}. Skipping.")
                continue
            
            # Look for metadata file
            metadata_path = os.path.join(embed_dir, 'metadata.json')
            if not os.path.exists(metadata_path):
                print(f"No metadata file found for embedding {embed_dir_name}. Skipping.")
                continue
            
            total_embeddings += 1
            
            try:
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Check if embedding already exists
                with db.get_db_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT id FROM embeddings WHERE id = %s", (embed_id,))
                        existing_embed = cursor.fetchone()
                
                if existing_embed:
                    print(f"Embedding {embed_dir_name} already exists in database. Skipping.")
                    continue
                
                # Convert model ID string to UUID
                try:
                    model_id = uuid.UUID(metadata.get('model_id'))
                except (ValueError, TypeError):
                    print(f"Invalid model ID format in embedding {embed_dir_name}. Skipping.")
                    continue
                
                # Get the model to verify it exists
                with db.get_db_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute("SELECT id FROM models WHERE id = %s", (model_id,))
                        model = cursor.fetchone()
                
                if not model:
                    print(f"Model {metadata.get('model_id')} not found for embedding {embed_dir_name}. Skipping.")
                    continue
                
                # Prepare embedding data
                name = metadata.get('name', 'Unnamed Embedding')
                settings = metadata.get('settings', {})
                
                # Prepare creation date
                if 'created_at' in metadata:
                    try:
                        created_at = datetime.datetime.fromisoformat(metadata.get('created_at'))
                    except (ValueError, TypeError):
                        created_at = datetime.datetime.now()
                else:
                    created_at = datetime.datetime.now()
                
                # Insert embedding into database using SQL
                with db.get_db_connection() as conn:
                    with conn.cursor() as cursor:
                        cursor.execute(
                            """
                            INSERT INTO embeddings (
                                id, user_id, model_id, name, embed_path, settings, created_at
                            )
                            VALUES (%s, %s, %s, %s, %s, %s, %s)
                            RETURNING id
                            """,
                            (
                                embed_id, user_id, model_id, name, 
                                embed_dir, json.dumps(settings), created_at
                            )
                        )
                        conn.commit()
                
                migrated_embeddings += 1
                print(f"Migrated embedding: {name} for user {user['username']}")
                
            except Exception as e:
                print(f"Error migrating embedding {embed_dir_name}: {str(e)}")
    
    print(f"Migrated {migrated_embeddings} out of {total_embeddings} embeddings.")

def create_admin_user():
    """Create an admin user if it doesn't exist"""
    print("Checking for admin user...")
    
    # Check if admin user exists
    admin_user = db.get_user_by_username('admin')
    if admin_user:
        print("Admin user already exists.")
        return
    
    # Create admin user
    with db.get_db_connection() as conn:
        with conn.cursor() as cursor:
            admin_id = uuid.uuid4()
            
            cursor.execute(
                """
                INSERT INTO users (id, username, email, password_hash, created_at, is_active)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    admin_id,
                    'admin',
                    'admin@example.com',
                    generate_password_hash('admin123'),  # Change in production
                    datetime.datetime.now(),
                    True
                )
            )
            conn.commit()
    
    print("Created admin user with username 'admin' and password 'admin123'")

def initialize_database():
    """Initialize the PostgreSQL database and schema"""
    print("Initializing PostgreSQL database...")
    
    # Initialize the database (create it if it doesn't exist)
    db.initialize_database()
    
    print("Database initialized successfully.")

def migrate_all():
    """Migrate all data from file-based system to database"""
    # Initialize the PostgreSQL database
    initialize_database()
    
    # Create admin user
    create_admin_user()
    
    # Migrate data
    migrate_users()
    migrate_models()
    migrate_use_cases()
    migrate_embeddings()
    
    print("Migration complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Migrate data from file-based system to PostgreSQL database')
    parser.add_argument('--init-db', action='store_true', help='Initialize PostgreSQL database')
    parser.add_argument('--users', action='store_true', help='Migrate users only')
    parser.add_argument('--models', action='store_true', help='Migrate models only')
    parser.add_argument('--use-cases', action='store_true', help='Migrate use cases only')
    parser.add_argument('--embeddings', action='store_true', help='Migrate embeddings only')
    parser.add_argument('--admin', action='store_true', help='Create admin user only')
    parser.add_argument('--all', action='store_true', help='Migrate all data')
    
    args = parser.parse_args()
    
    if args.init_db or args.all:
        initialize_database()
    
    if args.admin or args.all:
        create_admin_user()
    
    if args.users or args.all:
        migrate_users()
    
    if args.models or args.all:
        migrate_models()
    
    if args.use_cases or args.all:
        migrate_use_cases()
    
    if args.embeddings or args.all:
        migrate_embeddings()
    
    if not any(vars(args).values()):
        # No arguments provided, show help
        parser.print_help()
        print("\nNo arguments provided. Use --all to migrate all data.")
