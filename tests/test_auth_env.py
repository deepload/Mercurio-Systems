import os
import importlib

def test_auth_env_vars(monkeypatch):
    # Set some environment variables
    monkeypatch.setenv("JWT_SECRET_KEY", "testsecret")
    monkeypatch.setenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "99")
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-google-client")
    monkeypatch.setenv("APPLE_CLIENT_ID", "test-apple-client")
    monkeypatch.setenv("FRONTEND_URL", "http://test.local")

    # Reload auth_config to pick up env vars
    auth_config = importlib.reload(importlib.import_module("app.core.auth_config"))

    assert auth_config.JWT_SECRET_KEY == "testsecret"
    assert auth_config.JWT_ACCESS_TOKEN_EXPIRE_MINUTES == 99
    assert auth_config.GOOGLE_CLIENT_ID == "test-google-client"
    assert auth_config.APPLE_CLIENT_ID == "test-apple-client"
    assert auth_config.FRONTEND_URL == "http://test.local"
    # Check default for EMAIL_VERIFICATION_REQUIRED
    assert isinstance(auth_config.EMAIL_VERIFICATION_REQUIRED, bool)
