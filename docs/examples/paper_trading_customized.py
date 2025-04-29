"""
Test script for customized paper trading
Based on Method 3 from the Quick Start Trading Guide
"""
import os
import sys
import logging
import argparse

# Add the project root to Python path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Customized Paper Trading')
    
    parser.add_argument('--strategy', type=str, default='MovingAverageStrategy',
                        help='Strategy to use')
    parser.add_argument('--symbols', type=str, default='AAPL,TSLA,AMZN',
                        help='Comma-separated list of symbols to trade')
    parser.add_argument('--risk_limit', type=float, default=0.01,
                        help='Maximum portfolio percentage per position (0.01 = 1%%)')
    parser.add_argument('--interval', type=int, default=300,
                        help='Check frequency in seconds (300 = 5 minutes)')
    parser.add_argument('--fee_percentage', type=float, default=0.001,
                        help='Simulated transaction fee percentage')
    
    return parser.parse_args()

# This script demonstrates customized paper trading
if __name__ == "__main__":
    args = parse_arguments()
    
    logger.info(f"Starting customized paper trading with:")
    logger.info(f"  Strategy: {args.strategy}")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Risk limit: {args.risk_limit * 100}%")
    logger.info(f"  Check interval: {args.interval} seconds")
    logger.info(f"  Transaction fee: {args.fee_percentage * 100}%")
    
    import subprocess
    import sys
    
    # Build the command with all arguments
    cmd = [
        sys.executable, 
        "run_paper_trading.py",
        f"--strategy={args.strategy}",
        f"--symbols={args.symbols}",
        f"--risk_limit={args.risk_limit}",
        f"--interval={args.interval}",
        f"--fee_percentage={args.fee_percentage}"
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
