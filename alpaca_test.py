#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Alpaca API Access Test Script

This script tests access to Alpaca API to verify your credentials
and subscription are working correctly.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST
    logger.info("Successfully imported alpaca-trade-api")
except ImportError:
    logger.error("Failed to import alpaca-trade-api. Trying to install it...")
    
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "alpaca-trade-api"])
    
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import REST
    logger.info("Successfully installed and imported alpaca-trade-api")

def test_alpaca_access():
    """Test access to Alpaca API with current credentials"""
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if we're using paper or live mode
    alpaca_mode = os.environ.get("ALPACA_MODE", "paper").lower()
    
    if alpaca_mode == "paper":
        api_key = os.environ.get("ALPACA_PAPER_KEY")
        api_secret = os.environ.get("ALPACA_PAPER_SECRET")
        base_url = os.environ.get("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
        logger.info("Testing Alpaca PAPER trading mode")
    else:
        api_key = os.environ.get("ALPACA_LIVE_KEY")
        api_secret = os.environ.get("ALPACA_LIVE_SECRET")
        base_url = os.environ.get("ALPACA_LIVE_URL", "https://api.alpaca.markets")
        logger.info("Testing Alpaca LIVE trading mode")
    
    if not api_key or not api_secret:
        logger.error(f"Alpaca API credentials for {alpaca_mode.upper()} mode not found in .env file")
        logger.info("Please check your .env file configuration")
        return False
    
    logger.info(f"Testing Alpaca API with key: {api_key[:4]}...{api_key[-4:]}")
    
    # Initialize Alpaca API
    try:
        # Connect to Alpaca API with specified credentials
        api = REST(
            key_id=api_key,
            secret_key=api_secret,
            base_url=base_url
        )
        logger.info(f"Successfully connected to Alpaca {alpaca_mode.upper()} Trading API")
        
        # Get account information
        account = api.get_account()
        logger.info(f"Account ID: {account.id}")
        logger.info(f"Account Status: {account.status}")
        logger.info(f"Portfolio Value: ${float(account.portfolio_value):.2f}")
        logger.info(f"Cash Balance: ${float(account.cash):.2f}")
        logger.info(f"Buying Power: ${float(account.buying_power):.2f}")
        
        # Check market data access
        try:
            # Test market data access with a well-known symbol
            symbol = 'AAPL'
            end = datetime.now()
            start = end - timedelta(days=5)
            
            # Format dates as ISO format strings
            start_str = start.date().isoformat()
            end_str = end.date().isoformat()
            
            # Get daily bar data
            logger.info(f"Attempting to get historical data for {symbol} from {start_str} to {end_str}")
            bars = api.get_bars(symbol, '1Day', start_str, end_str)
            
            if len(bars) > 0:
                logger.info(f"Successfully retrieved {len(bars)} bars of data")
                logger.info(f"Latest close price for {symbol}: ${bars[-1].c:.2f}")
                logger.info("Market data access is working correctly!")
            else:
                logger.warning(f"No data returned for {symbol}. This could be due to market holidays or data limitations.")
        
        except Exception as e:
            logger.error(f"Error accessing market data: {e}")
            logger.warning("Your Alpaca subscription might not include market data access")
            
        # Check subscription details
        try:
            # This will only work if the account has access to this endpoint
            account_configurations = api.get_account_configurations()
            logger.info("Successfully retrieved account configurations")
            logger.info(f"Account Configurations: {json.dumps(account_configurations.__dict__, indent=2)}")
        except Exception as e:
            logger.warning(f"Could not retrieve account configurations: {e}")
        
        # Check for active subscriptions
        logger.info("Your Alpaca account is active and working correctly!")
        return True
        
    except Exception as e:
        logger.error(f"Error connecting to Alpaca API: {e}")
        logger.error("Please check your API credentials and subscription status")
        return False

def test_data_subscription(api, alpaca_mode):
    """Test specific data subscription features"""
    logger.info("\n=== Testing Alpaca Data Subscription ===\n")
    
    try:
        # Testing real-time data by requesting market data
        # This is a specific feature of paid subscriptions
        symbol = 'AAPL'
        end = datetime.now()
        start = end - timedelta(days=60)  # Try for longer period to test subscription
        
        # Format dates as ISO format strings
        start_str = start.date().isoformat()
        end_str = end.date().isoformat()
        
        # Get minute data as this might be restricted to paid plans
        logger.info(f"Attempting to get 1-minute resolution data for {symbol} (last 7 days)")
        week_start = end - timedelta(days=7)
        week_start_str = week_start.date().isoformat()
        
        minute_bars = api.get_bars(symbol, '1Min', week_start_str, end_str)
        if len(minute_bars) > 0:
            logger.info(f"✅ Successfully retrieved {len(minute_bars)} minute bars")
            logger.info(f"First bar: {minute_bars[0].__dict__}")
            logger.info(f"Last bar: {minute_bars[-1].__dict__}")
        else:
            logger.warning("❌ No minute data retrieved, this might indicate a subscription limitation")
        
        # Test for fundamental data (if available in the subscription)
        try:
            logger.info("\nAttempting to access fundamental data...")
            
            # Try to access news API (often restricted to paid subscriptions)
            news = api.get_news(symbol)
            if news and len(news) > 0:
                logger.info(f"✅ Successfully retrieved {len(news)} news items for {symbol}")
                logger.info(f"Latest news: {news[0].headline}")
            else:
                logger.warning("❌ No news data retrieved, this might indicate a subscription limitation")
        except Exception as e:
            logger.warning(f"❌ Could not access fundamental data: {e}")
            logger.warning("This might be restricted in your current subscription plan")
        
        # Test data for multiple symbols (batch request)
        try:
            symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
            logger.info(f"\nTesting batch data for {len(symbols)} symbols...")
            
            batch_bars = api.get_bars(symbols, '1Day', start_str, end_str)
            if batch_bars and sum(len(bars) for symbol, bars in batch_bars.items()) > 0:
                logger.info(f"✅ Successfully retrieved batch data for multiple symbols")
                for symbol, bars in batch_bars.items():
                    logger.info(f"  - {symbol}: {len(bars)} bars")
            else:
                logger.warning("❌ Batch data retrieval failed or returned no data")
        except Exception as e:
            logger.warning(f"❌ Batch data retrieval error: {e}")
            logger.warning("This might be restricted in your current subscription plan")
            
        return True
    except Exception as e:
        logger.error(f"Error testing data subscription: {e}")
        return False

if __name__ == "__main__":
    # Add command line arguments to choose between paper and live mode
    import argparse
    parser = argparse.ArgumentParser(description='Test Alpaca API access and subscription features')
    parser.add_argument('--mode', choices=['paper', 'live', 'both'], default='paper',
                       help='Trading mode to test: paper, live, or both (default: paper)')
    parser.add_argument('--data', action='store_true', help='Run additional tests for data subscription features')
    
    args = parser.parse_args()
    
    # Modify .env variable temporarily if testing both or specific mode
    if args.mode == 'both':
        modes = ['paper', 'live']
    else:
        modes = [args.mode]
    
    all_successful = True
    
    for mode in modes:
        logger.info(f"\n=== Alpaca API Access Test: {mode.upper()} MODE ===\n")
        
        # Override mode in environment
        os.environ['ALPACA_MODE'] = mode
        
        if test_alpaca_access():
            logger.info(f"✅ Basic connectivity test for {mode.upper()} mode completed successfully!")
            
            # Test data subscription features if requested
            if args.data:
                # Get the current mode's API credentials
                load_dotenv()
                if mode == "paper":
                    api_key = os.environ.get("ALPACA_PAPER_KEY")
                    api_secret = os.environ.get("ALPACA_PAPER_SECRET")
                    base_url = os.environ.get("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
                else:
                    api_key = os.environ.get("ALPACA_LIVE_KEY")
                    api_secret = os.environ.get("ALPACA_LIVE_SECRET")
                    base_url = os.environ.get("ALPACA_LIVE_URL", "https://api.alpaca.markets")
                
                api = REST(key_id=api_key, secret_key=api_secret, base_url=base_url)
                test_data_subscription(api, mode)
        else:
            logger.error(f"❌ Tests failed for {mode.upper()} mode. Please check the error messages above.")
            all_successful = False
    
    if all_successful:
        logger.info("\n=== SUMMARY ===\n")
        logger.info("✅ All Alpaca API tests completed successfully!")
        logger.info("Your Alpaca subscription is working correctly.")
    else:
        logger.error("\n=== SUMMARY ===\n")
        logger.error("❌ Some Alpaca API tests failed. Please check the error messages above.")
        logger.error("Your subscription might have limitations or configuration issues.")
        
    logger.info("\nTo test data subscription features more thoroughly, run:")
    logger.info("python alpaca_test.py --mode both --data")
    
    if 'live' in modes:
        logger.warning("\nNOTE: Tests included LIVE mode which connects to your real trading account.")
        logger.warning("      No trades were executed, but please verify the connection was successful.")

