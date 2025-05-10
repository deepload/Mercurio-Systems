"""
Test Alpaca API Connection

This script tests the connection to Alpaca API with your credentials.
"""
import os
import sys
from dotenv import load_dotenv
import alpaca_trade_api as tradeapi

# Load environment variables from .env file
print("Loading environment variables...")
load_dotenv()

# Get API keys based on ALPACA_MODE
alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
print(f"Using Alpaca mode: {alpaca_mode}")

if alpaca_mode == "paper":
    alpaca_key = os.getenv("ALPACA_PAPER_KEY")
    alpaca_secret = os.getenv("ALPACA_PAPER_SECRET")
    base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
    print("Using Alpaca PAPER trading environment")
else:  # live mode
    alpaca_key = os.getenv("ALPACA_LIVE_KEY")
    alpaca_secret = os.getenv("ALPACA_LIVE_SECRET")
    base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
    print("Using Alpaca LIVE trading environment")

# Check if keys are loaded
print(f"Alpaca API key found: {bool(alpaca_key)}")
print(f"Alpaca API secret found: {bool(alpaca_secret)}")
print(f"Using base URL: {base_url}")

if not alpaca_key or not alpaca_secret:
    print("ERROR: API keys not found in environment variables!")
    sys.exit(1)

# Initialize Alpaca client
print("\nInitializing Alpaca client...")
try:
    api = tradeapi.REST(
        key_id=alpaca_key,
        secret_key=alpaca_secret,
        base_url=base_url
    )
    print("Alpaca client initialized successfully!")
    
    # Test account info
    print("\nFetching account information...")
    account = api.get_account()
    print(f"Account ID: {account.id}")
    print(f"Account Status: {account.status}")
    print(f"Cash: ${float(account.cash):.2f}")
    print(f"Portfolio Value: ${float(account.portfolio_value):.2f}")
    
    # Test market status
    print("\nChecking market status...")
    clock = api.get_clock()
    print(f"Market is {'open' if clock.is_open else 'closed'}")
    print(f"Next market open: {clock.next_open}")
    print(f"Next market close: {clock.next_close}")
    
    print("\nConnection test successful! Your Alpaca API keys are working correctly.")
    
except Exception as e:
    print(f"ERROR: Failed to initialize Alpaca client: {e}")
    sys.exit(1)
