"""
Environment Variable Loader for Mercurio AI

This module ensures environment variables are properly loaded from .env file
"""
import os
import logging
from pathlib import Path
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

def load_environment():
    """
    Load environment variables from .env file
    
    This function should be called at the beginning of the application startup
    to ensure all environment variables are properly loaded.
    """
    # Find the project root (where the .env file should be located)
    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent  # From /app/utils to project root
    
    # Full path to .env file
    env_path = project_root / '.env'
    
    if env_path.exists():
        logger.info(f"Loading environment variables from {env_path}")
        # Load environment variables from .env file
        load_dotenv(dotenv_path=env_path)
        return True
    else:
        logger.warning(f".env file not found at {env_path}")
        return False

# Auto-load when imported
loaded = load_environment()
if loaded:
    # Verify key environment variables were loaded
    alpaca_key = os.getenv("ALPACA_KEY")
    alpaca_secret = os.getenv("ALPACA_SECRET")
    polygon_key = os.getenv("POLYGON_API_KEY")
    
    if alpaca_key and alpaca_secret:
        logger.info("Alpaca API credentials loaded successfully")
    else:
        logger.warning("Alpaca API credentials not found in environment variables")
        
    if polygon_key:
        logger.info("Polygon API key loaded successfully")
    else:
        logger.warning("Polygon API key not found in environment variables")
