"""
JWT Token Validator
Validates JWT tokens issued by the auth service
"""
import os
from typing import Optional, Dict
from jose import JWTError, jwt
from dotenv import load_dotenv

load_dotenv()

# JWT Configuration (must match auth service)
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "dev-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")


def verify_jwt_token(token: str) -> Optional[Dict]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded payload if valid, None if invalid
    """
    try:
        # Decode and verify token
        payload = jwt.decode(
            token, 
            JWT_SECRET_KEY, 
            algorithms=[JWT_ALGORITHM]
        )
        
        # Check if token type is correct
        if payload.get("type") != "access_token":
            print(f"❌ Invalid token type: {payload.get('type')}")
            return None
        
        return payload
        
    except JWTError as e:
        print(f"❌ JWT verification failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Error verifying JWT: {e}")
        return None


def extract_user_from_token(payload: Dict) -> Optional[Dict]:
    """
    Extract user information from JWT payload.
    
    Args:
        payload: Decoded JWT payload
        
    Returns:
        User info dictionary
    """
    if not payload:
        return None
    
    return {
        "user_id": payload.get("sub"),  # MongoDB ObjectId (google_id for now)
        "email": payload.get("email"),
        "name": payload.get("name"),
        "picture": payload.get("picture"),
        "google_id": payload.get("sub")
    }
