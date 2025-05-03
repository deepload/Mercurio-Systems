"""
Test script for Transformer-based trading strategies
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
    parser = argparse.ArgumentParser(description='Transformer Strategy Testing')
    
    parser.add_argument('--duration', type=int, default=24,
                        help='Test duration in hours')
    parser.add_argument('--symbols', type=str, default='BTC/USDT',
                        help='Comma-separated list of symbols to trade')
    parser.add_argument('--sequence_length', type=int, default=30,
                        help='Sequence length for transformer model')
    parser.add_argument('--d_model', type=int, default=64,
                        help='Model dimension for transformer')
    parser.add_argument('--nhead', type=int, default=4,
                        help='Number of heads in attention layers')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of transformer layers')
    
    return parser.parse_args()

# This script tests Transformer-based trading strategies
if __name__ == "__main__":
    args = parse_arguments()
    
    # Parse symbols into the format expected by run_paper_trading.py or paper_trading_test.py
    symbols = args.symbols.replace(',', ' ')
    
    logger.info(f"Testing Transformer strategy with:")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Duration: {args.duration} hours")
    logger.info(f"  Sequence length: {args.sequence_length}")
    logger.info(f"  Model dimension: {args.d_model}")
    logger.info(f"  Number of heads: {args.nhead}")
    logger.info(f"  Number of layers: {args.num_layers}")
    
    # Create strategy parameters JSON
    import json
    params = {
        "sequence_length": args.sequence_length,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers
    }
    params_json = json.dumps(params)
    
    # We can use either run_paper_trading.py or paper_trading_test.py
    # Here we'll use run_paper_trading.py for the specific configuration example
    
    import subprocess
    import sys
    
    # Build the command with all arguments
    cmd = [
        sys.executable, 
        "run_paper_trading.py",
        "--strategy=TransformerStrategy",
        f"--symbols={args.symbols}",
        f"--params={params_json}"
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
