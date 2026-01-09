"""
Pydantic models for authentication
"""
from typing import Optional
from pydantic import BaseModel


class User(BaseModel):
    """Authenticated user model"""
    user_id: str  # MongoDB ObjectId as string
    email: str
    name: str
    picture: Optional[str] = None
    google_id: Optional[str] = None


class AuthenticatedRequest(BaseModel):
    """Request with authenticated user context"""
    user: User
    client_id: Optional[str] = None
