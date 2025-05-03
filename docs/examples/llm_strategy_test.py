"""
Test script for LLM-based trading strategies
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
    parser = argparse.ArgumentParser(description='LLM Strategy Testing')
    
    parser.add_argument('--duration', type=int, default=24,
                        help='Test duration in hours')
    parser.add_argument('--symbols', type=str, default='BTC/USDT,ETH/USDT',
                        help='Comma-separated list of symbols to trade')
    parser.add_argument('--model_path', type=str, default='models/llama-2-7b-chat.gguf',
                        help='Path to the LLM model')
    parser.add_argument('--context_window', type=int, default=72,
                        help='Context window for LLM')
    parser.add_argument('--temperature', type=float, default=0.7,
                        help='LLM temperature parameter')
    
    return parser.parse_args()

# This script tests LLM-based trading strategies
if __name__ == "__main__":
    args = parse_arguments()
    
    # Parse symbols into the format expected by paper_trading_test.py
    symbols = args.symbols.replace(',', ' ')
    
    logger.info(f"Testing LLM strategy with:")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Duration: {args.duration} hours")
    logger.info(f"  Model path: {args.model_path}")
    logger.info(f"  Context window: {args.context_window}")
    logger.info(f"  Temperature: {args.temperature}")
    
    # Create strategy parameters JSON
    import json
    strategy_params = {
        "LLMStrategy": {
            "model_path": args.model_path,
            "context_window": args.context_window,
            "temperature": args.temperature,
            "max_tokens": 512
        }
    }
    params_json = json.dumps(strategy_params)
    
    import subprocess
    import sys
    
    # Build the command with all arguments
    cmd = [
        sys.executable, 
        "paper_trading_test.py",
        "--strategies", "LLMStrategy",
        "--duration", str(args.duration),
        "--symbols", symbols,
        "--config", "config/paper_test_config.json",
        "--strategy_params", params_json
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
