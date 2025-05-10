"""
Test script for paper trading with default settings
Based on Method 1 from the Quick Start Trading Guide
"""
import os
import sys
import logging

# Add the project root to Python path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This script simply runs the default paper trading configuration
if __name__ == "__main__":
    logger.info("Starting basic paper trading with default settings")
    logger.info("This will use the default moving average strategy")
    logger.info("Trading AAPL, MSFT, GOOGL, and BTC-USD symbols")
    logger.info("Checking for signals every 60 seconds with 2% risk limit per position")
    
    # Import and run the script
    import subprocess
    import sys
    
    # Use subprocess to run the original script with proper stdout/stderr capture
    result = subprocess.run(
        [sys.executable, "run_paper_trading.py"],
        cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
        capture_output=True, 
        text=True
    )
    
    # Log the output
    if result.stdout:
        for line in result.stdout.splitlines():
            logger.info(f"Output: {line}")
    
    # Log any errors
    if result.stderr:
        for line in result.stderr.splitlines():
            logger.error(f"Error: {line}")
            
    # Exit with the same code
    sys.exit(result.returncode)
