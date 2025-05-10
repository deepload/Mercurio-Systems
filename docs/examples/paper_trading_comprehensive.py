"""
Test script for comprehensive strategy testing
Based on Method 2 from the Quick Start Trading Guide
"""
import os
import sys
import logging

# Add the project root to Python path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# This script runs comprehensive strategy testing
if __name__ == "__main__":
    logger.info("Starting comprehensive strategy testing")
    logger.info("This will automatically discover and test all available strategies")
    logger.info("Running with default settings for quick testing")
    
    import subprocess
    import sys
    
    # Use subprocess to run the original script
    result = subprocess.run(
        [sys.executable, "paper_trading_test.py"],
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
