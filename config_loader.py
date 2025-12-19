"""
Configuration loader for multi-tenant system.
Loads client-specific configurations from MongoDB.
"""
import os
import json
from typing import Dict, Optional
from pathlib import Path
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import logging

logger = logging.getLogger(__name__)

# MongoDB connection for config storage
MONGODB_URI = os.getenv("MONGODB_URI", "")
ADMIN_DB_NAME = os.getenv("ADMIN_DB_NAME", "widget")

# Cache for loaded configs
_config_cache: Dict[str, dict] = {}
_mongo_client = None


def get_mongodb_client():
    """Get MongoDB client for config storage"""
    global _mongo_client
    if _mongo_client is not None:
        return _mongo_client
    
    if not MONGODB_URI:
        logger.error("MONGODB_URI not configured")
        return None
    
    try:
        _mongo_client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        _mongo_client.admin.command('ping')
        return _mongo_client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        return None


def load_client_config(client_id: str) -> dict:
    """
    Load client configuration from MongoDB.
    
    Args:
        client_id: Unique client identifier
        
    Returns:
        Client configuration dictionary
        
    Raises:
        FileNotFoundError: If client config doesn't exist in MongoDB
        ValueError: If config is invalid
    """
    # Normalize client_id
    client_id = client_id.lower().strip()
    
    # Check cache first
    if client_id in _config_cache:
        return _config_cache[client_id]
    
    # Load from MongoDB
    mongo_client = get_mongodb_client()
    if not mongo_client:
        raise FileNotFoundError(f"MongoDB connection failed. Cannot load config for client: {client_id}")
    
    try:
        admin_db = mongo_client[ADMIN_DB_NAME]
        clients_collection = admin_db["client_configs"]
        
        config = clients_collection.find_one({"client_id": client_id})
        
        if not config:
            raise FileNotFoundError(f"Client config not found in MongoDB: {client_id}")
        
        # Remove MongoDB _id
        config.pop('_id', None)
        
        # Validate required fields
        required_fields = ['client_id', 'client_name']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in client config")
        
        # Ensure client_id matches
        if config['client_id'] != client_id:
            raise ValueError(f"Client ID mismatch: expected {client_id}, got {config['client_id']}")
        
        # Cache the config
        _config_cache[client_id] = config
        
        return config
        
    except FileNotFoundError:
        raise
    except Exception as e:
        raise ValueError(f"Error loading client config from MongoDB: {e}")


def get_client_config(client_id: str) -> dict:
    """
    Get client configuration (cached).
    
    Args:
        client_id: Unique client identifier
        
    Returns:
        Client configuration dictionary
    """
    return load_client_config(client_id)


def get_mongodb_database_name(client_id: str) -> str:
    """
    Get MongoDB database name for a client.
    Uses client-specific database name from config, or defaults to client_id.
    
    Args:
        client_id: Unique client identifier
        
    Returns:
        MongoDB database name
    """
    config = get_client_config(client_id)
    return config.get('mongodb', {}).get('database_name', client_id.upper())


def get_s3_bucket_name(client_id: str) -> str:
    """
    Get S3 bucket name for a client.
    
    Args:
        client_id: Unique client identifier
        
    Returns:
        S3 bucket name
    """
    config = get_client_config(client_id)
    return config.get('s3', {}).get('bucket_name', f"{client_id}-storage")


def get_preprocessor_url(client_id: str) -> str:
    """
    Get preprocessor URL for a client.
    
    Args:
        client_id: Unique client identifier
        
    Returns:
        Preprocessor URL
    """
    config = get_client_config(client_id)
    return config.get('preprocessor', {}).get('url', os.getenv("PREPROCESSOR_URL", "http://localhost:8080"))


def get_postprocessor_url(client_id: str) -> str:
    """
    Get postprocessor URL for a client.
    
    Args:
        client_id: Unique client identifier
        
    Returns:
        Postprocessor URL
    """
    config = get_client_config(client_id)
    return config.get('postprocessor', {}).get('url', os.getenv("POSTPROCESSOR_URL", "http://localhost:8003"))


def get_openai_api_key(client_id: str) -> Optional[str]:
    """
    Get OpenAI API key for a client.
    Falls back to environment variable if not set in client config.
    
    Args:
        client_id: Unique client identifier
        
    Returns:
        OpenAI API key or None
    """
    try:
        config = get_client_config(client_id)
        api_key = config.get('openai', {}).get('api_key')
        if api_key:
            return api_key
    except Exception:
        pass
    
    # Fallback to environment variable
    return os.getenv("OPENAI_API_KEY")


def list_all_clients() -> list:
    """
    List all available client IDs from MongoDB.
    
    Returns:
        List of client IDs
    """
    mongo_client = get_mongodb_client()
    if not mongo_client:
        return []
    
    try:
        admin_db = mongo_client[ADMIN_DB_NAME]
        clients_collection = admin_db["client_configs"]
        
        clients = clients_collection.find({}, {"client_id": 1})
        client_ids = [client["client_id"] for client in clients if "client_id" in client]
        return sorted(client_ids)
    except Exception as e:
        logger.error(f"Error listing clients from MongoDB: {e}")
        return []


def clear_cache():
    """Clear the configuration cache."""
    global _config_cache
    _config_cache = {}


def get_client_id_from_request(request) -> str:
    """
    Extract client_id from FastAPI request (query params or headers).
    
    Args:
        request: FastAPI Request object
        
    Returns:
        Client ID string
        
    Raises:
        HTTPException: If client_id is missing or client not found
    """
    from fastapi import HTTPException
    
    # Try to get client_id from query params or headers
    client_id = None
    if hasattr(request, 'query_params'):
        client_id = request.query_params.get("client_id")
    if not client_id and hasattr(request, 'headers'):
        client_id = request.headers.get("X-Client-ID")
    
    if not client_id:
        raise HTTPException(
            status_code=400,
            detail="client_id is required (query param or X-Client-ID header)"
        )
    
    # Normalize client_id
    client_id = client_id.lower().strip()
    
    # Validate client exists
    try:
        get_client_config(client_id)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail=f"Client '{client_id}' not found"
        )
    
    return client_id

