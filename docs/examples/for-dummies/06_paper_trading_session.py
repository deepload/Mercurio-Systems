"""
Paper Trading Example
Based on Chapter 4: Paper Trading Basics

This script runs a paper trading session with the Moving Average strategy.
"""
import os
import sys
import asyncio
import pandas as pd
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import required components
from app.strategies.moving_average import MovingAverageStrategy
from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.db.models import TradeAction

async def paper_trading_session():
    """Run a complete paper trading session"""
    
    # 1. Initialize services
    print("Initializing services...")
    market_data = MarketDataService()
    trading = TradingService(is_paper=True)  # Paper trading mode
    
    # 2. Create strategy
    print("Creating trading strategy...")
    strategy = MovingAverageStrategy(short_window=10, long_window=30)
    
    # 3. Set trading parameters
    symbol = "AAPL"
    position_size = 0.2  # Use 20% of capital per position
    
    # 4. Get account information
    account_info = await trading.get_account_info()
    if "error" in account_info:
        print(f"Warning: {account_info['error']}")
        print("Continuing with simulated account...")
        initial_capital = 10000.0  # Simulated starting capital
    else:
        initial_capital = float(account_info.get('cash', 10000.0))
    
    # 5. Main trading loop
    print(f"\nStarting paper trading session for {symbol}")
    print(f"Initial capital: ${initial_capital:.2f}")
    
    try:
        for i in range(10):  # Run for 10 iterations (in real use, this would run continuously)
            print(f"\n--- Iteration {i+1} ---")
            
            # Get latest data (60 days lookback for analysis)
            end_date = datetime.now() - timedelta(days=i)  # Simulate different days
            start_date = end_date - timedelta(days=60)
            
            print(f"Fetching data for {symbol} from {start_date.date()} to {end_date.date()}...")
            data = await market_data.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(data) == 0:
                print(f"No data available for {symbol}. Skipping this iteration.")
                continue
                
            current_price = data['close'].iloc[-1]
            print(f"Current price: ${current_price:.2f}")
            
            # Preprocess data for strategy
            print("Analyzing market data...")
            processed_data = await strategy.preprocess_data(data)
            
            # Get trading signal
            signal, confidence = await strategy.predict(processed_data)
            print(f"Signal: {signal}, Confidence: {confidence:.2f}")
            
            # Check current positions
            positions = await trading.get_positions()
            has_position = False
            
            for position in positions:
                if position.get('symbol') == symbol:
                    has_position = True
                    position_quantity = float(position.get('qty', 0))
                    break
            
            # Execute trade based on signal
            if signal == TradeAction.BUY and not has_position:
                # Calculate quantity based on position size
                available_capital = initial_capital * position_size
                quantity = available_capital / current_price
                
                # Round to appropriate number of shares (whole shares for stocks)
                quantity = round(quantity, 2)
                
                if quantity > 0:
                    print(f"BUY SIGNAL: Attempting to buy {quantity} shares at ${current_price:.2f}")
                    
                    try:
                        order = await trading.execute_trade(
                            symbol=symbol,
                            action=TradeAction.BUY,
                            quantity=quantity,
                            strategy_name="MovingAverageStrategy"
                        )
                        
                        if "error" in order or order.get("status") == "error":
                            print(f"Order error: {order.get('message', 'Unknown error')}")
                        else:
                            print(f"Bought {quantity} shares at ${current_price:.2f}")
                    except Exception as e:
                        print(f"Error executing buy order: {e}")
                else:
                    print("Insufficient capital for buy order")
            
            elif signal == TradeAction.SELL and has_position:
                print(f"SELL SIGNAL: Attempting to sell {position_quantity} shares at ${current_price:.2f}")
                
                try:
                    order = await trading.execute_trade(
                        symbol=symbol,
                        action=TradeAction.SELL,
                        quantity=position_quantity,
                        strategy_name="MovingAverageStrategy"
                    )
                    
                    if "error" in order or order.get("status") == "error":
                        print(f"Order error: {order.get('message', 'Unknown error')}")
                    else:
                        print(f"Sold {position_quantity} shares at ${current_price:.2f}")
                except Exception as e:
                    print(f"Error executing sell order: {e}")
            
            else:
                print("No action taken")
                if signal == TradeAction.HOLD:
                    print("HOLD signal received")
                elif has_position and signal == TradeAction.BUY:
                    print("Already have a position, no additional buying")
                elif not has_position and signal == TradeAction.SELL:
                    print("No position to sell")
            
            # Print current portfolio status
            updated_account = await trading.get_account_info()
            if "error" in updated_account:
                print("Account information not available")
            else:
                print(f"\nCurrent balance: ${float(updated_account.get('cash', 0)):.2f}")
                print(f"Portfolio value: ${float(updated_account.get('portfolio_value', 0)):.2f}")
            
            # In a real scenario, we would wait for market updates
            # Here we'll just wait a second to simulate time passing
            await asyncio.sleep(1)
        
        # Final portfolio summary
        final_account = await trading.get_account_info()
        
        print("\n--- Final Portfolio Summary ---")
        print(f"Starting capital: ${initial_capital:.2f}")
        
        if "error" not in final_account:
            final_value = float(final_account.get('portfolio_value', initial_capital))
            print(f"Final portfolio value: ${final_value:.2f}")
            print(f"Total return: {(final_value / initial_capital - 1) * 100:.2f}%")
        else:
            print("Final portfolio information not available")
        
        # Get final positions
        final_positions = await trading.get_positions()
        if final_positions and not isinstance(final_positions, dict):
            print("\nFinal positions:")
            for position in final_positions:
                print(f"- {position.get('symbol')}: {position.get('qty')} shares at ${float(position.get('avg_entry_price', 0)):.2f}")
        else:
            print("\nNo open positions")
        
    except Exception as e:
        print(f"Error during paper trading session: {e}")

if __name__ == "__main__":
    print("Paper Trading Session Example")
    print("=" * 50)
    asyncio.run(paper_trading_session())
