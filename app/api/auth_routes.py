#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Authentication API routes for Mercurio Edge.

These routes handle user registration, login, logout, token refresh,
password management, and email verification.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Body, Request, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field

from app.db.database import get_db
from app.services.auth_service import AuthService, get_auth_service
from app.db.models.user import User

# Configure logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/auth", tags=["authentication"])

# Request/Response models
class UserRegisterRequest(BaseModel):
    """User registration request model."""
    email: EmailStr
    password: str = Field(..., min_length=8)
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    risk_profile: Optional[str] = None
    investment_goals: Optional[Dict[str, Any]] = None

class UserResponse(BaseModel):
    """User response model."""
    id: int
    email: str
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool
    is_verified: bool
    created_at: Any  # Using Any because datetime serialization can be tricky

    class Config:
        """Pydantic config."""
        from_attributes = True  # Formerly orm_mode = True in earlier Pydantic versions

class TokenResponse(BaseModel):
    """Token response model."""
    access_token: str
    refresh_token: str
    token_type: str

class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""
    refresh_token: str

class PasswordResetRequest(BaseModel):
    """Password reset request model."""
    email: EmailStr

class PasswordResetConfirmRequest(BaseModel):
    """Password reset confirmation request model."""
    token: str
    new_password: str = Field(..., min_length=8)

class ChangePasswordRequest(BaseModel):
    """Change password request model."""
    current_password: str
    new_password: str = Field(..., min_length=8)


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    user_data: UserRegisterRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Register a new user.
    
    Returns:
        Newly created user information
    """
    try:
        user = auth_service.create_user(
            email=user_data.email,
            password=user_data.password,
            username=user_data.username,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            phone_number=user_data.phone_number,
            risk_profile=user_data.risk_profile,
            investment_goals=user_data.investment_goals
        )
        return user
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/login", response_model=TokenResponse)
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Authenticate a user and provide access and refresh tokens.
    
    Returns:
        JWT token pair (access token and refresh token)
    """
    user = auth_service.authenticate_user(form_data.username, form_data.password)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    access_token = auth_service.create_access_token(user)
    refresh_token = auth_service.create_refresh_token(user)
    
    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer"
    }


@router.post("/refresh", response_model=Dict[str, str])
async def refresh_token(
    request: RefreshTokenRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Refresh an access token using a refresh token.
    
    Returns:
        New access token
    """
    try:
        token_data = auth_service.refresh_access_token(request.refresh_token)
        return token_data
    except HTTPException as e:
        raise e


@router.post("/password-reset", status_code=status.HTTP_202_ACCEPTED)
async def request_password_reset(
    request: PasswordResetRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Request a password reset email.
    
    Returns:
        202 Accepted response regardless of whether the email exists
    """
    # We don't reveal whether the email exists to prevent enumeration attacks
    auth_service.generate_password_reset_token(request.email)
    return {"message": "If your email is registered, you will receive a password reset link"}


@router.post("/password-reset/confirm", status_code=status.HTTP_200_OK)
async def confirm_password_reset(
    request: PasswordResetConfirmRequest,
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Reset password using a reset token.
    
    Returns:
        200 OK if password was reset successfully
    """
    if not auth_service.reset_password(request.token, request.new_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired password reset token"
        )
    
    return {"message": "Password has been reset successfully"}


@router.post("/verify-email", status_code=status.HTTP_200_OK)
async def verify_email(
    token: str = Body(..., embed=True),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Verify a user's email using a verification token.
    
    Returns:
        200 OK if email was verified successfully
    """
    if not auth_service.verify_email(token):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    
    return {"message": "Email verified successfully"}


@router.post("/verify-email/resend", status_code=status.HTTP_202_ACCEPTED)
async def resend_verification_email(
    email: EmailStr = Body(..., embed=True),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Resend verification email.
    
    Returns:
        202 Accepted response regardless of whether the email exists or is already verified
    """
    # We don't reveal whether the email exists or is already verified to prevent enumeration attacks
    auth_service.resend_verification_email(email)
    return {"message": "If your email is registered and not verified, you will receive a verification email"}


@router.post("/change-password", status_code=status.HTTP_200_OK)
async def change_password(
    request: ChangePasswordRequest,
    current_user: User = Depends(AuthService.get_current_user),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Change a user's password.
    
    Returns:
        200 OK if password was changed successfully
    """
    if not auth_service.change_password(current_user, request.current_password, request.new_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Incorrect current password"
        )
    
    return {"message": "Password changed successfully"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(AuthService.get_current_user)
):
    """
    Get current user information.
    
    Returns:
        Current user information
    """
    return current_user


@router.post("/logout", status_code=status.HTTP_200_OK)
async def logout_user(
    response: Response,
    # We require authentication to logout to prevent CSRF attacks
    current_user: User = Depends(AuthService.get_current_user)
):
    """
    Log out the current user.
    
    In a stateless JWT authentication system, we can't actually invalidate tokens.
    We would need to implement a token blacklist or use short-lived tokens.
    Here we just clear the client-side cookies if they exist.
    
    Returns:
        200 OK response
    """
    # Clear cookies if using cookie-based auth
    response.delete_cookie("access_token")
    response.delete_cookie("refresh_token")
    
    # In a real-world implementation, we might also add the token to a blacklist
    return {"message": "Logged out successfully"}
