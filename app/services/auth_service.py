#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Authentication Service for Mercurio Edge.

This service handles user authentication, registration, and session management.
"""

import os
import jwt
import uuid
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Union
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

from app.db.models.user import User
from app.db.database import get_db
from app.services.email_service import EmailService

# Configure logging
logger = logging.getLogger(__name__)

# Configure password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# OAuth2 scheme for token handling
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/login")

# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "temporary_secret_key_change_in_production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
JWT_REFRESH_TOKEN_EXPIRE_DAYS = 7


class AuthService:
    """Service for handling authentication operations."""

    def __init__(self, db: Session):
        """Initialize the auth service with database session."""
        self.db = db

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify that a plain password matches a hashed password."""
        return pwd_context.verify(plain_password, hashed_password)

    def get_password_hash(self, password: str) -> str:
        """Hash a password for storing."""
        return pwd_context.hash(password)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get a user by email."""
        return self.db.query(User).filter(User.email == email).first()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get a user by id."""
        return self.db.query(User).get(user_id)

    def create_user(self, email: str, password: str, username: Optional[str] = None,
                   first_name: Optional[str] = None, last_name: Optional[str] = None,
                   phone_number: Optional[str] = None, risk_profile: Optional[str] = None,
                   investment_goals: Optional[Dict[str, Any]] = None) -> User:
        """
        Create a new user.
        
        Args:
            email: User email
            password: Plain text password (will be hashed)
            username: Optional username
            first_name: Optional first name
            last_name: Optional last name
            phone_number: Optional phone number
            risk_profile: Optional risk profile
            investment_goals: Optional investment goals

        Returns:
            Newly created user object

        Raises:
            ValueError: If email already exists
        """
        # Check if email already exists
        if self.get_user_by_email(email):
            raise ValueError(f"User with email {email} already exists")

        # Create new user with hashed password
        user = User(
            email=email,
            hashed_password=self.get_password_hash(password),
            username=username,
            first_name=first_name,
            last_name=last_name,
            phone_number=phone_number,
            risk_profile=risk_profile,
            investment_goals=investment_goals,
            is_active=True,
            is_verified=False,  # User needs to verify email
            created_at=datetime.utcnow()
        )

        try:
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            
            # Generate and store email verification token
            self._send_verification_email(user)
            
            return user
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Error creating user: {str(e)}")
            raise ValueError("Error creating user. Please try again.")

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """
        Authenticate a user with email and password.
        
        Args:
            email: User email
            password: Plain text password

        Returns:
            User object if authentication succeeds, None otherwise
        """
        user = self.get_user_by_email(email)
        
        # Check if user exists and password is correct
        if not user or not self.verify_password(password, user.hashed_password):
            return None
            
        # Check if user is active
        if not user.is_active:
            raise ValueError("Account is disabled. Please contact support.")
        
        # Update last login timestamp
        user.last_login_at = datetime.utcnow()
        self.db.commit()
        
        return user

    def create_access_token(self, user: User) -> str:
        """
        Create a JWT access token for a user.
        
        Args:
            user: User to create token for

        Returns:
            JWT token string
        """
        expires_delta = timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
        expire = datetime.utcnow() + expires_delta
        
        to_encode = {
            "sub": str(user.id),
            "email": user.email,
            "exp": expire,
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt

    def create_refresh_token(self, user: User) -> str:
        """
        Create a JWT refresh token for a user.
        
        Args:
            user: User to create token for

        Returns:
            JWT token string
        """
        expires_delta = timedelta(days=JWT_REFRESH_TOKEN_EXPIRE_DAYS)
        expire = datetime.utcnow() + expires_delta
        
        # Include a unique token ID for potential revocation
        token_id = str(uuid.uuid4())
        
        to_encode = {
            "sub": str(user.id),
            "jti": token_id,
            "exp": expire,
            "type": "refresh"
        }
        
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt

    def decode_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate a JWT token.
        
        Args:
            token: JWT token string

        Returns:
            Decoded token payload

        Raises:
            HTTPException: If token is invalid or expired
        """
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    async def get_current_user(self, token: str = Depends(oauth2_scheme)) -> User:
        """
        Get the current user from JWT token.
        
        Args:
            token: JWT token from Authorization header

        Returns:
            Current user object

        Raises:
            HTTPException: If token is invalid or user not found
        """
        payload = self.decode_token(token)
        
        # Check token type
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = int(payload.get("sub"))
        user = self.get_user_by_id(user_id)
        
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

    async def get_current_active_user(self, current_user: User = Depends(get_current_user)) -> User:
        """
        Get the current active user.
        
        Args:
            current_user: User from get_current_user dependency

        Returns:
            Current user object if active

        Raises:
            HTTPException: If user is not active
        """
        if not current_user.is_active:
            raise HTTPException(status_code=400, detail="Inactive user")
        return current_user

    def refresh_access_token(self, refresh_token: str) -> Dict[str, str]:
        """
        Refresh an access token using a refresh token.
        
        Args:
            refresh_token: JWT refresh token

        Returns:
            Dictionary with new access token

        Raises:
            HTTPException: If refresh token is invalid
        """
        try:
            payload = jwt.decode(refresh_token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # Check token type
            if payload.get("type") != "refresh":
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            user_id = int(payload.get("sub"))
            user = self.get_user_by_id(user_id)
            
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
            
            # Create new access token
            access_token = self.create_access_token(user)
            
            return {
                "access_token": access_token,
                "token_type": "bearer"
            }
        except jwt.ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token has expired, please login again",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def change_password(self, user: User, current_password: str, new_password: str) -> bool:
        """
        Change a user's password.
        
        Args:
            user: User object
            current_password: Current plain text password
            new_password: New plain text password

        Returns:
            True if password changed successfully, False otherwise
        """
        # Verify current password
        if not self.verify_password(current_password, user.hashed_password):
            return False
            
        # Update password
        user.hashed_password = self.get_password_hash(new_password)
        user.updated_at = datetime.utcnow()
        
        self.db.commit()
        return True

    def generate_password_reset_token(self, email: str) -> bool:
        """
        Generate a password reset token and send reset email.
        
        Args:
            email: User email

        Returns:
            True if reset email sent, False if user not found
        """
        user = self.get_user_by_email(email)
        if not user:
            return False
            
        # Generate token
        reset_token = self._generate_verification_token(user, token_type="password_reset")
        
        # Send email with reset link
        # In a real implementation, use a proper email service
        reset_url = f"https://yourapp.com/reset-password?token={reset_token}"
        logger.info(f"Password reset URL for {email}: {reset_url}")
        
        # TODO: Implement actual email sending
        # email_service = EmailService()
        # email_service.send_password_reset(user.email, reset_url)
        
        return True

    def reset_password(self, token: str, new_password: str) -> bool:
        """
        Reset a user's password using a reset token.
        
        Args:
            token: Password reset token
            new_password: New plain text password

        Returns:
            True if password reset successful, False otherwise
        """
        try:
            # Decode and validate token
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # Check token type
            if payload.get("type") != "password_reset":
                return False
                
            user_id = int(payload.get("sub"))
            user = self.get_user_by_id(user_id)
            
            if not user:
                return False
                
            # Update password
            user.hashed_password = self.get_password_hash(new_password)
            user.updated_at = datetime.utcnow()
            
            self.db.commit()
            return True
        except jwt.ExpiredSignatureError:
            logger.warning("Password reset token has expired")
            return False
        except jwt.InvalidTokenError:
            logger.warning("Invalid password reset token")
            return False

    def verify_email(self, token: str) -> bool:
        """
        Verify a user's email using a verification token.
        
        Args:
            token: Email verification token

        Returns:
            True if email verified successfully, False otherwise
        """
        try:
            # Decode and validate token
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            
            # Check token type
            if payload.get("type") != "email_verification":
                return False
                
            user_id = int(payload.get("sub"))
            user = self.get_user_by_id(user_id)
            
            if not user:
                return False
                
            # Mark email as verified
            user.is_verified = True
            user.email_verified_at = datetime.utcnow()
            
            self.db.commit()
            return True
        except jwt.ExpiredSignatureError:
            logger.warning("Email verification token has expired")
            return False
        except jwt.InvalidTokenError:
            logger.warning("Invalid email verification token")
            return False

    def resend_verification_email(self, email: str) -> bool:
        """
        Resend verification email to user.
        
        Args:
            email: User email

        Returns:
            True if verification email sent, False if user not found or already verified
        """
        user = self.get_user_by_email(email)
        
        if not user or user.is_verified:
            return False
            
        self._send_verification_email(user)
        return True

    def _send_verification_email(self, user: User) -> None:
        """
        Send verification email to user.
        
        Args:
            user: User object
        """
        # Generate verification token
        verification_token = self._generate_verification_token(user)
        
        # Send email with verification link
        # In a real implementation, use a proper email service
        verification_url = f"https://yourapp.com/verify-email?token={verification_token}"
        logger.info(f"Email verification URL for {user.email}: {verification_url}")
        
        # TODO: Implement actual email sending
        # email_service = EmailService()
        # email_service.send_verification(user.email, verification_url)

    def _generate_verification_token(self, user: User, token_type: str = "email_verification") -> str:
        """
        Generate a verification token for a user.
        
        Args:
            user: User to generate token for
            token_type: Type of token (email_verification or password_reset)

        Returns:
            JWT token string
        """
        # Different expiration based on token type
        if token_type == "password_reset":
            expires_delta = timedelta(hours=1)
        else:  # email_verification
            expires_delta = timedelta(days=3)
            
        expire = datetime.utcnow() + expires_delta
        
        to_encode = {
            "sub": str(user.id),
            "email": user.email,
            "exp": expire,
            "type": token_type
        }
        
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt


# Dependency for getting the auth service
def get_auth_service(db: Session = Depends(get_db)) -> AuthService:
    """Dependency for getting the auth service."""
    return AuthService(db)
