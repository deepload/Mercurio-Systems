"""
Test script for LLMStrategyV2 - Enhanced LLM-based trading with sentiment analysis
"""
import os
import sys
import logging
import argparse
import json

# Add the project root to Python path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='LLMStrategyV2 Testing')
    
    parser.add_argument('--duration', type=int, default=24,
                        help='Test duration in hours')
    parser.add_argument('--symbols', type=str, default='BTC-USD,ETH-USD',
                        help='Comma-separated list of symbols to trade (format: BTC-USD,ETH-USD)')
    parser.add_argument('--model_name', type=str, default='mistralai/Mixtral-8x7B-Instruct-v0.1',
                        help='Name of the LLM model to use')
    parser.add_argument('--sentiment_weight', type=float, default=0.5,
                        help='Weight of sentiment analysis in decision making (0-1)')
    parser.add_argument('--use_local_model', action='store_true',
                        help='Use a local model instead of API')
    parser.add_argument('--local_model_path', type=str,
                        help='Path to local model (if using local model)')
    parser.add_argument('--api_key', type=str, default='demo_mode',
                        help='API key for LLM service (or "demo_mode" for testing)')
    parser.add_argument('--min_confidence', type=float, default=0.65,
                        help='Minimum confidence threshold for trade signals')
    parser.add_argument('--news_lookback', type=int, default=24,
                        help='Hours of news data to consider for sentiment analysis')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Parse symbols into the format expected by paper_trading_test.py
    symbols = args.symbols.replace(',', ' ')
    
    logger.info(f"Testing LLMStrategyV2 with:")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Duration: {args.duration} hours")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Sentiment weight: {args.sentiment_weight}")
    logger.info(f"  API key mode: {'demo' if args.api_key == 'demo_mode' else 'custom'}")
    
    # Create strategy parameters JSON
    strategy_params = {
        "LLMStrategyV2": {
            "model_name": args.model_name,
            "use_local_model": args.use_local_model,
            "local_model_path": args.local_model_path,
            "api_key": args.api_key,
            "sentiment_weight": args.sentiment_weight,
            "min_confidence": args.min_confidence,
            "news_lookback_hours": args.news_lookback,
            "use_web_sentiment": True
        }
    }
    params_json = json.dumps(strategy_params)
    
    import subprocess
    
    # Build the command with all arguments
    cmd = [
        sys.executable, 
        "paper_trading_test.py",
        "--strategies", "LLM_V2",
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
