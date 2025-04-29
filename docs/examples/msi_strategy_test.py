"""
Test script for Multi-Source Intelligence (MSI) trading strategy
Based on the Advanced Strategy Testing section of the Quick Start Trading Guide
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
    parser = argparse.ArgumentParser(description='Multi-Source Intelligence Strategy Testing')
    
    parser.add_argument('--duration', type=int, default=24,
                        help='Test duration in hours')
    parser.add_argument('--symbols', type=str, default='BTC/USDT,ETH/USDT',
                        help='Comma-separated list of symbols to trade')
    parser.add_argument('--risk', type=str, default='moderate', 
                        choices=['conservative', 'moderate', 'aggressive'],
                        help='Risk profile (conservative, moderate, aggressive)')
    
    return parser.parse_args()

# This script tests the MSI strategy
if __name__ == "__main__":
    args = parse_arguments()
    
    # Parse symbols into the format expected by paper_trading_test.py
    symbols = args.symbols.replace(',', ' ')
    
    logger.info(f"Testing Multi-Source Intelligence strategy with:")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Duration: {args.duration} hours")
    logger.info(f"  Risk profile: {args.risk}")
    
    import subprocess
    import sys
    
    # Build the command with all arguments
    cmd = [
        sys.executable, 
        "paper_trading_test.py",
        "--strategies", "MultiSourceIntelligenceStrategy",
        "--duration", str(args.duration),
        "--symbols", symbols,
        "--risk", args.risk
    ]
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
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
