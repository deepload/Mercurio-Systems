#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Authentication configuration for Mercurio Edge.

This module provides constants and configuration options for the authentication system.
"""

import os
from datetime import timedelta
from typing import Dict, Any

# JWT settings
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "temporary_secret_key_change_in_production")
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
JWT_REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# Password settings
PASSWORD_MIN_LENGTH = 8
PASSWORD_REQUIRES_LETTERS = True
PASSWORD_REQUIRES_NUMBERS = True
PASSWORD_REQUIRES_SPECIAL_CHARS = False
PASSWORD_REQUIRES_UPPERCASE = False

# OAuth settings
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/api/auth/oauth/google/callback")

APPLE_CLIENT_ID = os.getenv("APPLE_CLIENT_ID", "")
APPLE_TEAM_ID = os.getenv("APPLE_TEAM_ID", "")
APPLE_KEY_ID = os.getenv("APPLE_KEY_ID", "")
APPLE_PRIVATE_KEY_PATH = os.getenv("APPLE_PRIVATE_KEY_PATH", "")
APPLE_REDIRECT_URI = os.getenv("APPLE_REDIRECT_URI", "http://localhost:8000/api/auth/oauth/apple/callback")

# Frontend URLs
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
LOGIN_SUCCESS_URL = f"{FRONTEND_URL}/auth/success"
LOGIN_FAILURE_URL = f"{FRONTEND_URL}/auth/failure"
VERIFY_EMAIL_URL = f"{FRONTEND_URL}/auth/verify-email"
RESET_PASSWORD_URL = f"{FRONTEND_URL}/auth/reset-password"

# Email verification settings
EMAIL_VERIFICATION_REQUIRED = os.getenv("EMAIL_VERIFICATION_REQUIRED", "true").lower() == "true"
EMAIL_VERIFICATION_EXPIRE_DAYS = 3  # Days before verification link expires

# Rate limiting settings (requests per minute)
RATE_LIMIT_LOGIN = 5  # Login attempts per minute per IP
RATE_LIMIT_REGISTER = 3  # Registration attempts per minute per IP
RATE_LIMIT_PASSWORD_RESET = 2  # Password reset requests per minute per IP

# Subscription integration settings
FREE_TIER_ON_REGISTRATION = True  # Automatically assign free tier on registration

# Security headers
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Content-Security-Policy": "default-src 'self'; frame-ancestors 'none'",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin"
}

def get_cors_origins() -> list:
    """Get allowed CORS origins from environment or return default."""
    origins_str = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8000")
    return [origin.strip() for origin in origins_str.split(",")]

def get_auth_settings() -> Dict[str, Any]:
    """Get all authentication settings as a dictionary."""
    return {
        "jwt": {
            "secret_key": JWT_SECRET_KEY,
            "algorithm": JWT_ALGORITHM,
            "access_token_expire_minutes": JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
            "refresh_token_expire_days": JWT_REFRESH_TOKEN_EXPIRE_DAYS,
        },
        "password": {
            "min_length": PASSWORD_MIN_LENGTH,
            "requires_letters": PASSWORD_REQUIRES_LETTERS,
            "requires_numbers": PASSWORD_REQUIRES_NUMBERS,
            "requires_special_chars": PASSWORD_REQUIRES_SPECIAL_CHARS,
            "requires_uppercase": PASSWORD_REQUIRES_UPPERCASE,
        },
        "oauth": {
            "google_enabled": bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET),
            "apple_enabled": bool(APPLE_CLIENT_ID and APPLE_TEAM_ID and APPLE_KEY_ID),
        },
        "email_verification": {
            "required": EMAIL_VERIFICATION_REQUIRED,
            "expire_days": EMAIL_VERIFICATION_EXPIRE_DAYS,
        },
        "subscription": {
            "free_tier_on_registration": FREE_TIER_ON_REGISTRATION,
        },
    }
