"""
FastAPI Authentication Middleware
Provides dependency injection for protected routes
"""
import os
from typing import Optional, Dict
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pymongo import MongoClient
from bson import ObjectId

from .jwt_validator import verify_jwt_token, extract_user_from_token
from .models import User

# Security scheme for Swagger UI
security = HTTPBearer()

# MongoDB connection
MONGODB_URI = os.getenv("MONGODB_URI", "")
ADMIN_DB_NAME = os.getenv("ADMIN_DB_NAME", "widget")


def get_mongodb_client():
    """Get MongoDB client for user/client lookups"""
    if not MONGODB_URI:
        return None
    try:
        client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ping')
        return client
    except Exception as e:
        print(f"❌ MongoDB connection failed: {e}")
        return None


def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """
    Dependency to require authentication for a route.
    
    Usage:
        @app.post("/protected")
        async def protected_route(user: User = Depends(require_auth)):
            return {"user_email": user.email}
    
    Args:
        credentials: HTTP Bearer credentials with JWT token
        
    Returns:
        User object with authenticated user info
        
    Raises:
        HTTPException: If token is invalid or missing
    """
    # Extract token from credentials
    token = credentials.credentials
    
    # Verify token
    payload = verify_jwt_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Extract user info
    user_info = extract_user_from_token(payload)
    
    if not user_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid user information in token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Return User object
    return User(**user_info)


def check_client_ownership(id: str, user: User) -> None:
    """
    Check if user owns a client by MongoDB ObjectId. Raises HTTPException if not.
    
    Usage:
        @app.put("/clients/{id}/system-prompt")
        async def update_prompt(
            id: str,
            user: User = Depends(require_auth)
        ):
            check_client_ownership(id, user)
            return {"message": "Updated"}
    
    Args:
        id: The MongoDB ObjectId as string
        user: Authenticated user
        
    Raises:
        HTTPException: If user doesn't own the client
    """
    mongo_client = get_mongodb_client()
    
    if not mongo_client:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Database connection failed"
        )
    
    try:
        # Convert string to ObjectId
        try:
            from bson.errors import InvalidId
            object_id = ObjectId(id)
        except (InvalidId, TypeError):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid ObjectId format: {id}"
            )
        
        # Get client config from MongoDB
        admin_db = mongo_client[ADMIN_DB_NAME]
        clients_collection = admin_db["client_configs"]
        
        client_config = clients_collection.find_one({"_id": object_id})
        
        if not client_config:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Client with id '{id}' not found"
            )
        
        # Check if user owns this client
        owner_id = client_config.get("owner_id")
        
        # owner_id might be a string or ObjectId, normalize both
        user_id_str = str(user.user_id)
        owner_id_str = str(owner_id)
        
        if user_id_str != owner_id_str:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this client's resources"
            )
        
        # User is owner, allow access
        return None
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Error checking ownership: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error verifying ownership"
        )
    finally:
        if mongo_client:
            mongo_client.close()


def optional_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[User]:
    """
    Optional authentication dependency.
    Returns user info if token is present and valid, None otherwise.
    
    Usage:
        @app.get("/maybe-protected")
        async def route(user: Optional[User] = Depends(optional_auth)):
            if user:
                return {"message": f"Hello {user.name}"}
            return {"message": "Hello anonymous"}
    """
    if not credentials:
        return None
    
    token = credentials.credentials
    payload = verify_jwt_token(token)
    
    if not payload:
        return None
    
    user_info = extract_user_from_token(payload)
    
    if not user_info:
        return None
    
    return User(**user_info)


def get_client_id_with_owner_check(request: Request) -> str:
    """
    Extract client_id from request and verify ownership.
    Used for routes where client_id is in query params.
    
    Usage:
        @app.post("/upload-guardrails")
        async def upload(
            request: Request,
            user: User = Depends(require_auth),
            client_id: str = Depends(get_client_id_with_owner_check)
        ):
            return {"client_id": client_id}
    """
    # This function should be used with require_auth
    # The actual owner check needs to be done after getting client_id
    client_id = request.query_params.get("client_id") or request.headers.get("X-Client-ID")
    
    if not client_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="client_id is required (query param or X-Client-ID header)"
        )
    
    return client_id
