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
from bson import ObjectId
from bson.errors import InvalidId
import logging

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, use environment variables directly

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MongoDB connection for config storage
MONGODB_URI = os.getenv("MONGODB_URI", "")
ADMIN_DB_NAME = os.getenv("ADMIN_DB_NAME", "widget")

# Cache for loaded configs
_config_cache: Dict[str, dict] = {}
_mongo_client = None


def get_mongodb_client():
    """Get MongoDB client for config storage with connection pooling"""
    global _mongo_client
    if _mongo_client is not None:
        return _mongo_client
    
    if not MONGODB_URI:
        logger.error("MONGODB_URI not configured")
        return None
    
    try:
        _mongo_client = MongoClient(
            MONGODB_URI,
            maxPoolSize=50,  # Maximum 50 connections in pool
            minPoolSize=10,  # Minimum 10 connections always ready
            maxIdleTimeMS=45000,  # Close idle connections after 45s
            serverSelectionTimeoutMS=5000,  # Timeout for server selection
            connectTimeoutMS=10000,  # Timeout for initial connection
            socketTimeoutMS=45000,  # Timeout for socket operations
            retryWrites=True,  # Retry writes on network errors
            retryReads=True,  # Retry reads on network errors
        )
        _mongo_client.admin.command('ping')
        logger.info("✅ MongoDB connection pool initialized for config_loader")
        return _mongo_client
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return None
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        return None


def load_client_config(id: str) -> dict:
    """
    Load client configuration from MongoDB by ObjectId.
    
    Args:
        id: MongoDB ObjectId as string
        
    Returns:
        Client configuration dictionary
        
    Raises:
        FileNotFoundError: If client config doesn't exist in MongoDB
        ValueError: If config is invalid or ObjectId format is invalid
    """
    # Check cache first
    if id in _config_cache:
        return _config_cache[id]
    
    # Load from MongoDB
    mongo_client = get_mongodb_client()
    if not mongo_client:
        raise FileNotFoundError(f"MongoDB connection failed. Cannot load config for client: {id}")
    
    try:
        # Convert string to ObjectId
        try:
            object_id = ObjectId(id)
        except (InvalidId, TypeError) as e:
            logger.error(f"Invalid ObjectId format: {id}")
            raise ValueError(f"Invalid ObjectId format: {id}")
        
        admin_db = mongo_client[ADMIN_DB_NAME]
        clients_collection = admin_db["client_configs"]
        
        logger.info(f"Looking for client with _id: '{id}' in database '{ADMIN_DB_NAME}', collection 'client_configs'")
        
        # Find by ObjectId
        config = clients_collection.find_one({"_id": object_id})
        
        if not config:
            # Try to list what's actually in the DB for debugging
            all_clients = list(clients_collection.find({}, {"_id": 1, "client_id": 1}))
            client_ids_in_db = [{"_id": str(c.get("_id")), "client_id": c.get("client_id")} for c in all_clients]
            logger.error(f"Client with id '{id}' not found in database '{ADMIN_DB_NAME}'. Available clients: {client_ids_in_db}")
            raise FileNotFoundError(f"Client config not found in MongoDB: {id}")
        
        # Convert ObjectId to string for JSON serialization
        if '_id' in config:
            config['_id'] = str(config['_id'])
        
        # Validate required fields
        required_fields = ['client_id', 'client_name']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field '{field}' in client config")
        
        # Cache the config using the ObjectId string
        _config_cache[id] = config
        logger.info(f"✅ Successfully loaded config for client: {id} (client_id: {config.get('client_id')})")
        
        return config
        
    except FileNotFoundError:
        raise
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error loading client config from MongoDB: {e}")
        raise FileNotFoundError(f"Failed to load client config: {str(e)}")


def get_client_config(id: str) -> dict:
    """
    Get client configuration by ObjectId (cached).

    Args:
        id: MongoDB ObjectId as string

    Returns:
        Client configuration dictionary
    """
    return load_client_config(id)


def get_mongodb_database_name(id: str) -> str:
    """
    Get MongoDB database name for a client by ObjectId.
    Uses client-specific database name from config, or defaults to client_id.

    Args:
        id: MongoDB ObjectId as string

    Returns:
        MongoDB database name
    """
    config = get_client_config(id)
    client_id = config.get('client_id', id)
    return config.get('mongodb', {}).get('database_name', client_id.upper())


def get_s3_bucket_name(id: str) -> str:
    """
    Get S3 bucket name for a client by ObjectId.

    Args:
        id: MongoDB ObjectId as string

    Returns:
        S3 bucket name
    """
    config = get_client_config(id)
    client_id = config.get('client_id', id)
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
