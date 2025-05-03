#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Mercurio AI Trading System
-------------------------
Main script to run the Mercurio AI trading system, with support for both stock and crypto markets.

Usage:
    python run.py [stock|crypto|all] [--duration HOURS] [--debug]

Examples:
    python run.py stock --duration 4      # Run stock trader for 4 hours
    python run.py crypto --duration 8     # Run crypto trader for 8 hours
    python run.py all --duration 1        # Run both stock and crypto trading for 1 hour
"""

import os
import sys
import json
import logging
import argparse
import asyncio
import traceback
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the individual scripts
from scripts.run_stock_daytrader import StockDayTrader
from scripts.run_crypto_daytrader import CryptoDayTrader

# Set up logging
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)

# Add file handler
os.makedirs('logs', exist_ok=True)
log_file = f'logs/mercurio_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(log_formatter)
root_logger.addHandler(file_handler)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(log_formatter)
root_logger.addHandler(console_handler)

# Define logger for this module
logger = logging.getLogger('mercurio')

def validate_api_keys():
    """
    Validate that API keys are properly configured or use fallback mechanism
    """
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'daytrader_config.json')
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check stock API keys
        stock_provider = config.get('stock', {}).get('market_data_provider')
        if stock_provider == 'alpaca':
            api_key = os.environ.get('ALPACA_API_KEY')
            api_secret = os.environ.get('ALPACA_API_SECRET')
            
            if not api_key or not api_secret:
                logger.warning("Alpaca API keys not found in environment variables.")
                logger.warning("Stock trading will use paper trading mode with sample data.")
                
                # Set provider to sample in config
                config['stock']['market_data_provider'] = 'sample'
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                    
                logger.info("Updated config to use sample data provider for stocks.")
        
        # Check crypto API keys
        crypto_exchange = config.get('crypto', {}).get('exchange')
        crypto_keys = config.get('crypto', {}).get('api_keys', {})
        
        if crypto_exchange and crypto_exchange.lower() != 'sample':
            exchange_keys = crypto_keys.get(crypto_exchange.lower(), {})
            
            if 'api_key' not in exchange_keys or exchange_keys['api_key'] == 'YOUR_API_KEY':
                logger.warning(f"{crypto_exchange} API keys not properly configured.")
                logger.warning("Crypto trading will use paper trading mode with sample data.")
                
                # Set exchange to sample in config
                config['crypto']['exchange'] = 'sample'
                
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)
                    
                logger.info("Updated config to use sample data for crypto trading.")
                
    except Exception as e:
        logger.error(f"Error validating API keys: {e}")
        logger.warning("Continuing with default configuration.")

async def run_trading_system(market_type, duration_hours, debug=False):
    """
    Run the trading system for the specified market type and duration
    
    Args:
        market_type: 'stock', 'crypto' or 'all'
        duration_hours: Number of hours to run the system for
        debug: Whether to enable debug logging
    """
    try:
        # Convert to seconds
        duration_seconds = int(duration_hours * 3600)
        
        # Get the path to the configuration file
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config', 'daytrader_config.json')
        
        if market_type in ['stock', 'all']:
            logger.info(f"Starting stock trading system for {duration_hours} hours")
            stock_trader = StockDayTrader(config_path=config_path, session_duration=duration_seconds)
            if market_type == 'all':
                # Start in background
                asyncio.create_task(stock_trader.start())
            else:
                # Run directly
                await stock_trader.start()
        
        if market_type in ['crypto', 'all']:
            logger.info(f"Starting crypto trading system for {duration_hours} hours")
            crypto_trader = CryptoDayTrader(config_path=config_path, session_duration=duration_seconds)
            await crypto_trader.start()
            
    except Exception as e:
        logger.error(f"Error running trading system: {e}")
        logger.debug(traceback.format_exc())

async def main_async():
    """
    Async main entry point for the Mercurio AI trading system
    """
    parser = argparse.ArgumentParser(description="Mercurio AI Trading System")
    parser.add_argument("market", choices=["stock", "crypto", "all"], 
                      help="Market type to trade (stock, crypto, or all)")
    parser.add_argument("--duration", type=float, default=4.0,
                      help="Trading session duration in hours (default: 4.0)")
    parser.add_argument("--debug", action="store_true", 
                      help="Enable debug logging")
    
    args = parser.parse_args()
    
    try:
        # Set log level
        if args.debug:
            root_logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Print banner
        logger.info("===================================================")
        logger.info("       MERCURIO AI TRADING SYSTEM STARTING        ")
        logger.info("===================================================")
        logger.info(f"Market: {args.market.upper()}")
        logger.info(f"Duration: {args.duration} hours")
        logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"End Time: {(datetime.now() + timedelta(hours=args.duration)).strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Log File: {log_file}")
        logger.info("===================================================")
        
        # Validate API keys and set fallback if needed
        validate_api_keys()
        
        # Run the trading system
        await run_trading_system(args.market, args.duration, args.debug)
        
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Error running Mercurio trading system: {e}")
        logger.debug(traceback.format_exc())
        return 1
    
    return 0

def main():
    """Synchronous wrapper for the async main function"""
    return asyncio.run(main_async())

if __name__ == "__main__":
    sys.exit(main())
