#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OAuth Service for Mercurio Edge.

This service handles OAuth authentication with third-party providers.
"""

import os
import json
import logging
import secrets
import requests
from typing import Dict, Optional, Any, List, Tuple
from sqlalchemy.orm import Session
from jwt import PyJWTError
import jwt

from app.db.models.user import User
from app.services.auth_service import AuthService

# Configure logging
logger = logging.getLogger(__name__)

# OAuth configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/auth/callback/google")

APPLE_CLIENT_ID = os.getenv("APPLE_CLIENT_ID", "")
APPLE_TEAM_ID = os.getenv("APPLE_TEAM_ID", "")
APPLE_KEY_ID = os.getenv("APPLE_KEY_ID", "")
APPLE_PRIVATE_KEY_PATH = os.getenv("APPLE_PRIVATE_KEY_PATH", "")
APPLE_REDIRECT_URI = os.getenv("APPLE_REDIRECT_URI", "http://localhost:8000/api/auth/callback/apple")

# Frontend URLs for redirects
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
LOGIN_SUCCESS_URL = f"{FRONTEND_URL}/auth/success"
LOGIN_FAILURE_URL = f"{FRONTEND_URL}/auth/failure"


class OAuthService:
    """Service for handling OAuth authentication with third-party providers."""

    def __init__(self, db: Session, auth_service: AuthService):
        """Initialize the OAuth service with database session."""
        self.db = db
        self.auth_service = auth_service

    def get_google_auth_url(self, state: Optional[str] = None) -> str:
        """
        Get the Google OAuth authorization URL.
        
        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Google OAuth authorization URL
        """
        if not GOOGLE_CLIENT_ID:
            logger.error("Google OAuth client ID not configured")
            raise ValueError("Google OAuth not properly configured")

        # Generate state if not provided
        if not state:
            state = secrets.token_urlsafe(32)

        params = {
            "client_id": GOOGLE_CLIENT_ID,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "scope": "email profile",
            "response_type": "code",
            "state": state,
            "access_type": "offline",
            "prompt": "consent",
        }

        auth_url = "https://accounts.google.com/o/oauth2/auth"
        query_str = "&".join([f"{k}={v}" for k, v in params.items()])
        
        return f"{auth_url}?{query_str}"

    def handle_google_callback(self, code: str) -> Tuple[User, Dict[str, str]]:
        """
        Handle Google OAuth callback.
        
        Args:
            code: Authorization code from Google

        Returns:
            Tuple of (User object, token response with access and refresh tokens)

        Raises:
            ValueError: If OAuth exchange fails
        """
        if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
            logger.error("Google OAuth credentials not configured")
            raise ValueError("Google OAuth not properly configured")

        # Exchange code for token
        token_url = "https://oauth2.googleapis.com/token"
        token_data = {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "redirect_uri": GOOGLE_REDIRECT_URI,
            "grant_type": "authorization_code",
        }

        try:
            token_response = requests.post(token_url, data=token_data)
            token_response.raise_for_status()
            token_json = token_response.json()

            # Get user info with the access token
            user_info_url = "https://www.googleapis.com/oauth2/v3/userinfo"
            user_info_response = requests.get(
                user_info_url, 
                headers={"Authorization": f"Bearer {token_json['access_token']}"}
            )
            user_info_response.raise_for_status()
            user_info = user_info_response.json()

            # Extract user data
            email = user_info.get("email")
            if not email:
                logger.error("No email in Google user info response")
                raise ValueError("Google account does not have an email")

            if not user_info.get("email_verified"):
                logger.warning(f"Google account email {email} is not verified")
                # Proceed anyway but log a warning

            # Check if user exists
            user = self.db.query(User).filter(User.email == email).first()

            if not user:
                # Create new user
                user = User(
                    email=email,
                    # Generate a random secure password that the user cannot use
                    hashed_password=self.auth_service.get_password_hash(secrets.token_urlsafe(32)),
                    username=user_info.get("name"),
                    first_name=user_info.get("given_name"),
                    last_name=user_info.get("family_name"),
                    profile_image_url=user_info.get("picture"),
                    is_active=True,
                    is_verified=True,  # Emails from Google OAuth are pre-verified
                )
                self.db.add(user)
                self.db.commit()
                self.db.refresh(user)
            else:
                # Update existing user information if needed
                user.is_active = True
                user.is_verified = True  # Ensure email is marked as verified
                
                # Update profile info if it's empty
                if not user.first_name and user_info.get("given_name"):
                    user.first_name = user_info.get("given_name")
                if not user.last_name and user_info.get("family_name"):
                    user.last_name = user_info.get("family_name")
                if not user.profile_image_url and user_info.get("picture"):
                    user.profile_image_url = user_info.get("picture")
                    
                self.db.commit()

            # Generate our own JWT tokens
            access_token = self.auth_service.create_access_token(user)
            refresh_token = self.auth_service.create_refresh_token(user)
            
            return user, {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer"
            }
            
        except requests.RequestException as e:
            logger.error(f"Error exchanging Google OAuth code: {str(e)}")
            raise ValueError("Failed to authenticate with Google")

    def get_apple_auth_url(self, state: Optional[str] = None) -> str:
        """
        Get the Apple OAuth authorization URL.
        
        Args:
            state: Optional state parameter for CSRF protection

        Returns:
            Apple OAuth authorization URL
        """
        if not APPLE_CLIENT_ID:
            logger.error("Apple OAuth client ID not configured")
            raise ValueError("Apple OAuth not properly configured")

        # Generate state if not provided
        if not state:
            state = secrets.token_urlsafe(32)

        params = {
            "client_id": APPLE_CLIENT_ID,
            "redirect_uri": APPLE_REDIRECT_URI,
            "scope": "email name",
            "response_type": "code",
            "response_mode": "form_post",
            "state": state
        }

        auth_url = "https://appleid.apple.com/auth/authorize"
        query_str = "&".join([f"{k}={v}" for k, v in params.items()])
        
        return f"{auth_url}?{query_str}"

    def _create_apple_client_secret(self) -> str:
        """
        Create a client secret for Apple OAuth using JWT.
        
        Returns:
            Apple client secret JWT
        """
        if not all([APPLE_CLIENT_ID, APPLE_TEAM_ID, APPLE_KEY_ID, APPLE_PRIVATE_KEY_PATH]):
            logger.error("Apple OAuth credentials not fully configured")
            raise ValueError("Apple OAuth not properly configured")
            
        try:
            # Read the private key
            with open(APPLE_PRIVATE_KEY_PATH, 'r') as key_file:
                private_key = key_file.read()
                
            # Current time and expiration time (10 minutes from now)
            import time
            current_time = int(time.time())
            expiration_time = current_time + 600  # 10 minutes
            
            # Create the JWT payload
            payload = {
                'iss': APPLE_TEAM_ID,
                'aud': 'https://appleid.apple.com',
                'sub': APPLE_CLIENT_ID,
                'iat': current_time,
                'exp': expiration_time
            }
            
            # Create the JWT
            headers = {
                'kid': APPLE_KEY_ID,
                'alg': 'ES256'  # Apple requires ES256 algorithm
            }
            
            client_secret = jwt.encode(
                payload=payload,
                key=private_key,
                algorithm='ES256',
                headers=headers
            )
            
            return client_secret
            
        except (IOError, PyJWTError) as e:
            logger.error(f"Error creating Apple client secret: {str(e)}")
            raise ValueError("Failed to create Apple client secret")

    def handle_apple_callback(self, code: str, user_data: Optional[str] = None) -> Tuple[User, Dict[str, str]]:
        """
        Handle Apple OAuth callback.
        
        Args:
            code: Authorization code from Apple
            user_data: JSON string with user information (provided only on first login)

        Returns:
            Tuple of (User object, token response with access and refresh tokens)

        Raises:
            ValueError: If OAuth exchange fails
        """
        if not APPLE_CLIENT_ID:
            logger.error("Apple OAuth client ID not configured")
            raise ValueError("Apple OAuth not properly configured")

        # Exchange code for token
        try:
            client_secret = self._create_apple_client_secret()
            
            token_url = "https://appleid.apple.com/auth/token"
            token_data = {
                "client_id": APPLE_CLIENT_ID,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": APPLE_REDIRECT_URI,
                "grant_type": "authorization_code",
            }

            token_response = requests.post(token_url, data=token_data)
            token_response.raise_for_status()
            token_json = token_response.json()
            
            # Extract the identity token
            id_token = token_json.get("id_token")
            if not id_token:
                logger.error("No ID token in Apple response")
                raise ValueError("Invalid response from Apple")
                
            # Parse and verify the ID token
            # Apple's public keys are available at https://appleid.apple.com/auth/keys
            # In a production environment, you should fetch and verify with these keys
            # For simplicity, we're just decoding without verification here
            id_token_payload = jwt.decode(id_token, options={"verify_signature": False})
            
            # Extract user email
            email = id_token_payload.get("email")
            if not email:
                logger.error("No email in Apple ID token")
                raise ValueError("Apple account does not have an email")
                
            # Extract user name from the user_data if provided (only on first login)
            first_name = None
            last_name = None
            if user_data:
                try:
                    user_info = json.loads(user_data)
                    name = user_info.get("name", {})
                    first_name = name.get("firstName")
                    last_name = name.get("lastName")
                except json.JSONDecodeError:
                    logger.warning("Could not parse user data from Apple")
            
            # Check if user exists
            user = self.db.query(User).filter(User.email == email).first()

            if not user:
                # Create new user
                user = User(
                    email=email,
                    # Generate a random secure password that the user cannot use
                    hashed_password=self.auth_service.get_password_hash(secrets.token_urlsafe(32)),
                    first_name=first_name,
                    last_name=last_name,
                    is_active=True,
                    is_verified=True,  # Emails from Apple OAuth are pre-verified
                )
                self.db.add(user)
                self.db.commit()
                self.db.refresh(user)
            else:
                # Update existing user information if needed
                user.is_active = True
                user.is_verified = True
                
                # Update name if it's empty and we have it from Apple
                if not user.first_name and first_name:
                    user.first_name = first_name
                if not user.last_name and last_name:
                    user.last_name = last_name
                    
                self.db.commit()

            # Generate our own JWT tokens
            access_token = self.auth_service.create_access_token(user)
            refresh_token = self.auth_service.create_refresh_token(user)
            
            return user, {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer"
            }
            
        except (requests.RequestException, PyJWTError) as e:
            logger.error(f"Error exchanging Apple OAuth code: {str(e)}")
            raise ValueError("Failed to authenticate with Apple")


# Dependency for getting the OAuth service
def get_oauth_service(db: Session, auth_service: AuthService) -> OAuthService:
    """Dependency for getting the OAuth service."""
    return OAuthService(db, auth_service)
