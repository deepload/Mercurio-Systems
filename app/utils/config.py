"""
Configuration Utilities

Functions and classes to handle configuration settings from environment variables.
"""
import os
from typing import Any, Dict, Optional
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Keys
    IEX_API_KEY: str = Field(default="", env="IEX_API_KEY")
    ALPACA_KEY: str = Field(default="", env="ALPACA_KEY")
    ALPACA_SECRET: str = Field(default="", env="ALPACA_SECRET")
    
    # Database
    DATABASE_URL: str = Field(default="postgresql+asyncpg://postgres:postgres@postgres:5432/mercurio", env="DATABASE_URL")
    
    # Redis
    REDIS_URL: str = Field(default="redis://redis:6379/0", env="REDIS_URL")
    
    # Application settings
    MODEL_DIR: str = Field(default="./models", env="MODEL_DIR")
    DATA_DIR: str = Field(default="./data", env="DATA_DIR")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    
    # Alpaca API settings
    ALPACA_PAPER: bool = Field(default=True, env="ALPACA_PAPER")
    ALPACA_BASE_URL: str = Field(default="https://paper-api.alpaca.markets", env="ALPACA_BASE_URL")
    ALPACA_DATA_URL: str = Field(default="https://data.alpaca.markets", env="ALPACA_DATA_URL")
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create a global settings instance
settings = Settings()

def get_setting(key: str, default: Any = None) -> Any:
    """
    Get a setting value from environment variables.
    
    Args:
        key: Name of the setting
        default: Default value if setting not found
        
    Returns:
        Setting value or default
    """
    return getattr(settings, key, default)

def get_settings() -> Settings:
    """
    Get all settings.
    
    Returns:
        Settings object
    """
    return settings
