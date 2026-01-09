"""
Shared Authentication Module for Multi-Tenant Agent System
Provides JWT validation and owner authorization for all services
"""
from .jwt_validator import verify_jwt_token, extract_user_from_token
from .middleware import require_auth, check_client_ownership, optional_auth, get_client_id_with_owner_check
from .models import User, AuthenticatedRequest

__all__ = [
    "verify_jwt_token",
    "extract_user_from_token",
    "require_auth",
    "check_client_ownership",
    "optional_auth",
    "get_client_id_with_owner_check",
    "User",
    "AuthenticatedRequest",
]
