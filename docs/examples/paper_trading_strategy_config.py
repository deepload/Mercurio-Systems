"""
Test script for paper trading with advanced strategy configuration
Based on Method 4 from the Quick Start Trading Guide
"""
import os
import sys
import logging
import json
import argparse

# Add the project root to Python path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Paper Trading with Strategy Configuration')
    
    parser.add_argument('--strategy', type=str, default='MovingAverageStrategy',
                        help='Strategy to use')
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT',
                        help='Comma-separated list of symbols to trade')
    parser.add_argument('--params', type=str, default='{"short_window": 20, "long_window": 50, "use_ml": true}',
                        help='JSON string with strategy parameters')
    
    return parser.parse_args()

# This script demonstrates paper trading with advanced strategy configuration
if __name__ == "__main__":
    args = parse_arguments()
    
    # Parse the JSON params
    try:
        # Make sure quotes are properly formatted
        cleaned_params = args.params.replace("'", '"')
        params_dict = json.loads(cleaned_params)
        params_str = json.dumps(params_dict, indent=2)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON parameters: {args.params}")
        logger.error(f"Error details: {str(e)}")
        sys.exit(1)
    
    logger.info(f"Starting paper trading with strategy configuration:")
    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Parameters: \n{params_str}")
    
    import subprocess
    import sys
    
    # Build the command with all arguments
    cmd = [
        sys.executable, 
        "run_paper_trading.py",
        f"--strategy={args.strategy}",
        f"--symbols={args.symbols}",
        f"--params={json.dumps(params_dict)}"
    ]
    
    # Use subprocess to run the command
    result = subprocess.run(
        cmd,
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
