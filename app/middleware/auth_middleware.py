#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Authentication middleware for Mercurio Edge.

This middleware handles JWT token validation and user authentication.
"""

import logging
from typing import Optional, Callable
from fastapi import Request, Response, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.auth_service import AuthService, get_auth_service

# Configure logging
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()


class AuthMiddleware:
    """Middleware for handling authentication and authorization."""

    @staticmethod
    async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        auth_service: AuthService = Depends(get_auth_service)
    ):
        """
        Get the current authenticated user from JWT token.
        
        Args:
            credentials: Bearer token credentials
            auth_service: Authentication service

        Returns:
            Current user object

        Raises:
            HTTPException: If authentication fails
        """
        try:
            token = credentials.credentials
            payload = auth_service.decode_token(token)
            
            # Check token type
            if payload.get("type") != "access":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            user_id = int(payload.get("sub"))
            user = auth_service.get_user_by_id(user_id)
            
            if not user:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
            if not user.is_active:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Inactive user",
                    headers={"WWW-Authenticate": "Bearer"},
                )
                
            return user
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    @staticmethod
    async def get_optional_user(
        request: Request,
        db: Session = Depends(get_db)
    ):
        """
        Get the current user if authenticated, or None if not.
        
        Args:
            request: FastAPI request object
            db: Database session

        Returns:
            Current user object or None
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
            
        try:
            token = auth_header.replace("Bearer ", "")
            auth_service = AuthService(db)
            payload = auth_service.decode_token(token)
            
            if payload.get("type") != "access":
                return None
                
            user_id = int(payload.get("sub"))
            user = auth_service.get_user_by_id(user_id)
            
            if not user or not user.is_active:
                return None
                
            return user
        except Exception as e:
            logger.warning(f"Optional authentication failed: {str(e)}")
            return None
            
    @staticmethod
    def require_verified_email():
        """
        Dependency to require a verified email.
        
        Returns:
            Dependency function
        """
        async def verified_email_dependency(
            current_user = Depends(AuthMiddleware.get_current_user)
        ):
            if not current_user.is_verified:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Email verification required"
                )
            return current_user
        return verified_email_dependency
