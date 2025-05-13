#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
OAuth API routes for Mercurio Edge.

These routes handle third-party authentication with providers like Google and Apple.
"""

import os
import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.auth_service import AuthService, get_auth_service
from app.services.oauth_service import OAuthService, get_oauth_service

# Configure logging
logger = logging.getLogger(__name__)

# Create API router
router = APIRouter(prefix="/auth/oauth", tags=["oauth"])

# Frontend URLs for redirects
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
LOGIN_SUCCESS_URL = f"{FRONTEND_URL}/auth/success"
LOGIN_FAILURE_URL = f"{FRONTEND_URL}/auth/failure"


@router.get("/google")
async def google_login(
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Initiate Google OAuth login flow.
    
    Returns:
        Redirect to Google authorization page
    """
    try:
        oauth_service = get_oauth_service(db, auth_service)
        auth_url = oauth_service.get_google_auth_url()
        return RedirectResponse(url=auth_url)
    except ValueError as e:
        logger.error(f"Error initiating Google login: {str(e)}")
        return RedirectResponse(url=f"{LOGIN_FAILURE_URL}?error=google_setup_error")


@router.get("/google/callback")
async def google_callback(
    code: str,
    state: Optional[str] = None,
    error: Optional[str] = None,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service),
    response: Response = None
):
    """
    Handle Google OAuth callback.
    
    Returns:
        Redirect to frontend with tokens in query parameters
    """
    if error:
        logger.error(f"Google OAuth error: {error}")
        return RedirectResponse(url=f"{LOGIN_FAILURE_URL}?error=google_auth_error")
        
    try:
        oauth_service = get_oauth_service(db, auth_service)
        user, tokens = oauth_service.handle_google_callback(code)
        
        # In a real implementation, set tokens in secure HTTP-only cookies
        # For now, we'll pass them as query parameters (not recommended for production)
        success_url = f"{LOGIN_SUCCESS_URL}?access_token={tokens['access_token']}&refresh_token={tokens['refresh_token']}"
        return RedirectResponse(url=success_url)
    except ValueError as e:
        logger.error(f"Error handling Google callback: {str(e)}")
        return RedirectResponse(url=f"{LOGIN_FAILURE_URL}?error=google_callback_error")


@router.get("/apple")
async def apple_login(
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Initiate Apple OAuth login flow.
    
    Returns:
        Redirect to Apple authorization page
    """
    try:
        oauth_service = get_oauth_service(db, auth_service)
        auth_url = oauth_service.get_apple_auth_url()
        return RedirectResponse(url=auth_url)
    except ValueError as e:
        logger.error(f"Error initiating Apple login: {str(e)}")
        return RedirectResponse(url=f"{LOGIN_FAILURE_URL}?error=apple_setup_error")


@router.post("/apple/callback")
async def apple_callback(
    request: Request,
    db: Session = Depends(get_db),
    auth_service: AuthService = Depends(get_auth_service)
):
    """
    Handle Apple OAuth callback.
    
    Note: Apple uses POST for callbacks with form data
    
    Returns:
        Redirect to frontend with tokens in query parameters
    """
    form_data = await request.form()
    
    code = form_data.get("code")
    state = form_data.get("state")
    error = form_data.get("error")
    user_data = form_data.get("user")  # Apple sends user data in the first login only
    
    if error:
        logger.error(f"Apple OAuth error: {error}")
        return RedirectResponse(url=f"{LOGIN_FAILURE_URL}?error=apple_auth_error")
        
    if not code:
        logger.error("No code in Apple callback")
        return RedirectResponse(url=f"{LOGIN_FAILURE_URL}?error=apple_missing_code")
        
    try:
        oauth_service = get_oauth_service(db, auth_service)
        user, tokens = oauth_service.handle_apple_callback(code, user_data)
        
        # In a real implementation, set tokens in secure HTTP-only cookies
        # For now, we'll pass them as query parameters (not recommended for production)
        success_url = f"{LOGIN_SUCCESS_URL}?access_token={tokens['access_token']}&refresh_token={tokens['refresh_token']}"
        return RedirectResponse(url=success_url)
    except ValueError as e:
        logger.error(f"Error handling Apple callback: {str(e)}")
        return RedirectResponse(url=f"{LOGIN_FAILURE_URL}?error=apple_callback_error")
