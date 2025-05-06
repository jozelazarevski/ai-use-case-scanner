# Add this to utils/user_models.py or create this file if it doesn't exist

import os
import json
import shutil
import uuid
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any
from config import Config

def get_user_models_dir(user_id: str) -> str:
    """
    Get the directory where user models are stored
    
    Args:
        user_id (str): User ID
        
    Returns:
        str: Path to user models directory
    """
    user_dir = os.path.join(Config.DATABASE_DIR, 'user_models', user_id)
    os.makedirs(user_dir, exist_ok=True)
    return user_dir

def get_models_index_path(user_id: str) -> str:
    """
    Get the path to the user's models index file
    
    Args:
        user_id (str): User ID
        
    Returns:
        str: Path to models index file
    """
    return os.path.join(get_user_models_dir(user_id), 'models_index.json')

def load_models_index(user_id: str) -> List[Dict[str, Any]]:
    """
    Load the user's models index
    
    Args:
        user_id (str): User ID
        
    Returns:
        List[Dict]: List of model metadata
    """
    index_path = get_models_index_path(user_id)
    
    if os.path.exists(index_path):
        try:
            with open(index_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading models index: {str(e)}")
            return []
    else:
        return []

def save_models_index(user_id: str, models_list: List[Dict[str, Any]]) -> bool:
    """
    Save the user's models index
    
    Args:
        user_id (str): User ID
        models_list (List[Dict]): List of model metadata
        
    Returns:
        bool: Success status
    """
    index_path = get_models_index_path(user_id)
    
    try:
        with open(index_path, 'w') as f:
            json.dump(models_list, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving models index: {str(e)}")
        return False

def save_user_model(user_id: str, model_name: str, model_path: str, metadata: Dict[str, Any]) -> Optional[str]:
    """
    Save a model to the user's account
    
    Args:
        user_id (str): User ID
        model_name (str): Display name for the model
        model_path (str): Path to the model file
        metadata (Dict): Model metadata
        
    Returns:
        Optional[str]: Model ID if successful, None otherwise
    """
    try:
        print(f"Saving model for user {user_id}: {model_name}")
        print(f"Original model path: {model_path}")
        
        # Validate inputs
        if not os.path.exists(model_path):
            print(f"Error: Model file does not exist: {model_path}")
            return None
        
        # Create a unique ID for the model
        model_id = str(uuid.uuid4())
        
        # Get user models directory
        user_models_dir = get_user_models_dir(user_id)
        
        # Create model directory
        model_dir = os.path.join(user_models_dir, model_id)
        os.makedirs(model_dir, exist_ok=True)
        
        # Copy model file to user's directory
        dest_path = os.path.join(model_dir, 'model.joblib')
        shutil.copy2(model_path, dest_path)
        print(f"Copied model from {model_path} to {dest_path}")
        
        # Check if a preprocessor exists alongside the model
        preprocessor_path = model_path.replace('best_model_', 'preprocessor_')
        if os.path.exists(preprocessor_path):
            # Copy preprocessor file
            preprocessor_dest = os.path.join(model_dir, 'preprocessor.joblib')
            shutil.copy2(preprocessor_path, preprocessor_dest)
            print(f"Copied preprocessor from {preprocessor_path} to {preprocessor_dest}")
        
        # Save model features if they exist
        features_path = model_path.replace('best_model_', 'model_features_')
        if os.path.exists(features_path):
            # Copy features file
            features_dest = os.path.join(model_dir, 'features.joblib')
            shutil.copy2(features_path, features_dest)
            print(f"Copied features from {features_path} to {features_dest}")
        
        # If we have feature names in the metadata, save them as JSON too
        if 'features' in metadata and metadata['features']:
            features_json_path = os.path.join(model_dir, 'features.json')
            try:
                with open(features_json_path, 'w') as f:
                    json.dump(metadata['features'], f)
                print(f"Saved features JSON to {features_json_path}")
            except Exception as fe:
                print(f"Error saving features JSON: {str(fe)}")
        
        # Save metadata
        metadata_path = os.path.join(model_dir, 'metadata.json')
        with open(metadata_path, 'w') as f:
            # Add timestamp and model ID to metadata
            full_metadata = {
                **metadata,
                'model_id': model_id,
                'saved_at': datetime.now().isoformat(),
                'name': model_name
            }
            json.dump(full_metadata, f, indent=2)
        
        # Update models index
        models_list = load_models_index(user_id)
        models_list.append({
            'id': model_id,
            'name': model_name,
            'path': dest_path,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata,
            'user_id': user_id
        })
        save_models_index(user_id, models_list)
        
        return model_id
    
    except Exception as e:
        error_trace = traceback.format_exc()
        print(f"Error saving model: {str(e)}\n{error_trace}")
        return None

def get_user_models(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all models belonging to a user
    
    Args:
        user_id (str): User ID
        
    Returns:
        List[Dict]: List of model metadata
    """
    return load_models_index(user_id)

def get_model_by_id(model_id: str) -> Optional[Dict[str, Any]]:
    """
    Get a model by ID
    
    Args:
        model_id (str): Model ID
        
    Returns:
        Optional[Dict]: Model metadata or None if not found
    """
    # This is not efficient but works for small numbers of models
    # For larger systems, we would need a database
    model_dir = os.path.join(Config.DATABASE_DIR, 'user_models')
    
    for user_dir in os.listdir(model_dir):
        if not os.path.isdir(os.path.join(model_dir, user_dir)):
            continue
            
        models_list = load_models_index(user_dir)
        
        for model in models_list:
            if model.get('id') == model_id:
                return model
    
    return None

def delete_user_model(user_id: str, model_id: str) -> bool:
    """
    Delete a model from the user's account
    
    Args:
        user_id (str): User ID
        model_id (str): Model ID
        
    Returns:
        bool: Success status
    """
    try:
        # Get models list
        models_list = load_models_index(user_id)
        
        # Find the model to delete
        model_to_delete = None
        for model in models_list:
            if model.get('id') == model_id:
                model_to_delete = model
                break
        
        if not model_to_delete:
            return False
        
        # Remove model from list
        models_list = [m for m in models_list if m.get('id') != model_id]
        
        # Save updated list
        save_models_index(user_id, models_list)
        
        # Delete model directory
        model_dir = os.path.join(get_user_models_dir(user_id), model_id)
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            shutil.rmtree(model_dir)
        
        return True
    
    except Exception as e:
        print(f"Error deleting model: {str(e)}")
        return False

# Placeholder functions for embeddings
def get_user_embeddings(user_id: str) -> List[Dict[str, Any]]:
    """
    Get all embeddings belonging to a user
    
    Args:
        user_id (str): User ID
        
    Returns:
        List[Dict]: List of embedding metadata
    """
    # Embeddings implementation depends on your system design
    # This is a placeholder that returns an empty list
    return []

def create_model_embedding(model_id: str, embed_name: str, embed_settings: Dict[str, Any]) -> Optional[str]:
    """
    Create an embeddable version of a model
    
    Args:
        model_id (str): Model ID
        embed_name (str): Name for the embedding
        embed_settings (Dict): Embedding settings
        
    Returns:
        Optional[str]: Embedding ID if successful, None otherwise
    """
    # Placeholder function
    # In a real implementation, this would create the embedding
    return "embed-" + model_id

def get_embedding_by_id(embed_id: str) -> Optional[Dict[str, Any]]:
    """
    Get an embedding by ID
    
    Args:
        embed_id (str): Embedding ID
        
    Returns:
        Optional[Dict]: Embedding metadata or None if not found
    """
    # Placeholder function
    # In a real implementation, this would retrieve the embedding
    return None

def delete_embedding(embed_id: str) -> bool:
    """
    Delete an embedding
    
    Args:
        embed_id (str): Embedding ID
        
    Returns:
        bool: Success status
    """
    # Placeholder function
    # In a real implementation, this would delete the embedding
    return True

def get_embed_code(embed_id: str) -> str:
    """
    Get the HTML code for embedding the model
    
    Args:
        embed_id (str): Embedding ID
        
    Returns:
        str: HTML embed code
    """
    # Placeholder function
    # In a real implementation, this would generate the embed code
    return f"<iframe src='/embedded/{embed_id}' width='100%' height='500'></iframe>"

# Add these functions to your utils/user_models.py file

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
    
    # Ensure database directories exist
    use_cases_dir = os.path.join(Config.DATABASE_DIR, 'use_cases', user_id)
    os.makedirs(use_cases_dir, exist_ok=True)
    
    # Generate a unique ID for this set of use cases
    use_case_id = str(uuid.uuid4())
    
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

def get_user_use_cases(user_id):
    """
    Retrieve all saved use cases for a user
    
    Args:
        user_id (str): ID of the user
    
    Returns:
        list: List of saved use cases
    """
    import os
    import json
    
    use_cases_dir = os.path.join(Config.DATABASE_DIR, 'use_cases', user_id)
    
    # If directory doesn't exist, return empty list
    if not os.path.exists(use_cases_dir):
        return []
    
    # Read all use case files
    use_cases = []
    for filename in os.listdir(use_cases_dir):
        if filename.endswith('.json'):
            filepath = os.path.join(use_cases_dir, filename)
            try:
                with open(filepath, 'r') as f:
                    use_case = json.load(f)
                    uses = {
                        'id': use_case.get('id'),
                        'filename': use_case.get('filename'),
                        'created_at': use_case.get('created_at'),
                        'proposal_count': len(use_case.get('proposals', [])),
                        'proposals': use_case.get('proposals', [])
                    }
                    use_cases.append(uses)
            except Exception as e:
                print(f"Error reading use case file {filename}: {e}")
    
    # Sort use cases by creation time, most recent first
    use_cases.sort(key=lambda x: x.get('created_at', ''), reverse=True)
    
    return use_cases

def delete_use_case(user_id, use_case_id):
    """
    Delete a specific use case
    
    Args:
        user_id (str): ID of the user
        use_case_id (str): ID of the use case to delete
    
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    import os
    
    use_cases_dir = os.path.join(Config.DATABASE_DIR, 'use_cases', user_id)
    use_case_path = os.path.join(use_cases_dir, f"{use_case_id}.json")
    
    if os.path.exists(use_case_path):
        try:
            os.remove(use_case_path)
            return True
        except Exception as e:
            print(f"Error deleting use case {use_case_id}: {e}")
            return False
    
    return False