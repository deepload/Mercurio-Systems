"""
Mercurio AI Trading Service Example
Based on Chapter 3: Understanding the Platform

This script demonstrates paper trading using the Trading Service.
"""
import os
import sys
import asyncio
import pandas as pd
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import required services
from app.services.trading import TradingService
from app.services.market_data import MarketDataService
from app.db.models import TradeAction

async def paper_trading_example():
    """Demonstrates paper trading functionality with Mercurio AI"""
    
    print("Initializing Trading Service in paper trading mode...")
    trading = TradingService(is_paper=True)  # Initialize in paper trading mode
    
    # Initialize market data service for price information
    market_data = MarketDataService()
    
    # Check account information
    account_info = await trading.get_account_info()
    print("\nAccount Information:")
    if "error" in account_info:
        print(f"Warning: {account_info['error']}")
        print("Continuing with demo functionality...")
    else:
        print(f"Status: {account_info.get('status', 'Unknown')}")
        print(f"Cash: ${float(account_info.get('cash', 0)):.2f}")
        print(f"Portfolio Value: ${float(account_info.get('portfolio_value', 0)):.2f}")
        print(f"Buying Power: ${float(account_info.get('buying_power', 0)):.2f}")
    
    # Check market status
    market_status = await trading.check_market_status()
    print("\nMarket Status:")
    if "error" in market_status:
        print(f"Warning: {market_status['error']}")
        print("Continuing with demo functionality...")
    else:
        print(f"Market is {'open' if market_status.get('is_open', False) else 'closed'}")
        if 'next_open' in market_status:
            print(f"Next market open: {market_status['next_open']}")
        if 'next_close' in market_status:
            print(f"Next market close: {market_status['next_close']}")
    
    # Symbol to trade
    symbol = "AAPL"
    
    # Get the latest price
    try:
        price = await market_data.get_latest_price(symbol)
        print(f"\nCurrent {symbol} price: ${price:.2f}")
        
        # Calculate quantity for a $1000 position (or 10% of available capital)
        quantity = await trading.calculate_order_quantity(
            symbol=symbol,
            action=TradeAction.BUY,
            capital_pct=0.1  # Use 10% of available capital
        )
        
        print(f"Calculated order quantity: {quantity} shares")
        
        # Place a buy order
        print(f"\nPlacing paper trade order for {symbol}...")
        order_result = await trading.execute_trade(
            symbol=symbol,
            action=TradeAction.BUY,
            quantity=quantity,
            strategy_name="DemoStrategy"
        )
        
        # Check the order result
        if "error" in order_result or order_result.get("status") == "error":
            print(f"Order error: {order_result.get('message', 'Unknown error')}")
        else:
            print("Order placed successfully!")
            print(f"Order details: {order_result}")
            
            # If the order was successful, place a sell order
            if quantity > 0:
                print(f"\nPlacing sell order for {symbol} to close the position...")
                sell_result = await trading.execute_trade(
                    symbol=symbol,
                    action=TradeAction.SELL,
                    quantity=quantity,
                    strategy_name="DemoStrategy"
                )
                if "error" in sell_result or sell_result.get("status") == "error":
                    print(f"Sell order error: {sell_result.get('message', 'Unknown error')}")
                else:
                    print("Sell order placed successfully!")
                    print(f"Order details: {sell_result}")
    
    except Exception as e:
        print(f"Error during trading demonstration: {e}")
        print("This could be due to API limitations in demo mode.")
    
    print("\nPaper trading demonstration completed.")
    print("In a full implementation, you would:")
    print("1. Set up a loop to monitor price movements")
    print("2. Generate signals using a trading strategy")
    print("3. Execute trades based on those signals")
    print("4. Track performance and manage positions")

if __name__ == "__main__":
    print("Trading Service Example - Paper Trading")
    print("=" * 50)
    asyncio.run(paper_trading_example())
