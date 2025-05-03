"""
Position Sizing Example
Based on Chapter 4: Paper Trading Basics (Best Practices section)

This script demonstrates realistic position sizing techniques for paper trading.
"""
import os
import sys
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

# Import required components
from app.strategies.moving_average import MovingAverageStrategy
from app.services.market_data import MarketDataService
from app.services.trading import TradingService
from app.db.models import TradeAction

def calculate_position_size(capital, risk_percentage, entry_price, stop_loss_price):
    """
    Calculate position size based on risk percentage
    
    Args:
        capital: Available trading capital
        risk_percentage: Risk percentage (1-100)
        entry_price: Entry price
        stop_loss_price: Stop loss price
        
    Returns:
        Number of shares to trade
    """
    risk_amount = capital * (risk_percentage / 100)
    risk_per_share = abs(entry_price - stop_loss_price)
    
    # Avoid division by zero
    if risk_per_share == 0:
        return 0
        
    shares = risk_amount / risk_per_share
    
    # Round to 2 decimal places for fractional shares or whole number for stocks
    shares = round(shares, 2)
    
    return shares

async def position_sizing_example():
    """Demonstrate different position sizing techniques"""
    
    print("Initializing services...")
    market_data = MarketDataService()
    
    # Define trading parameters
    symbols = ["AAPL", "MSFT", "AMZN", "GOOGL"]
    initial_capital = 100000.0  # $100,000 starting capital
    
    print(f"Initial capital: ${initial_capital:.2f}")
    
    # Get current market data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Get 30 days of data
    
    print("\n=== Fixed Dollar Amount Position Sizing ===")
    fixed_amount = 5000.0  # $5,000 per position
    
    for symbol in symbols:
        try:
            # Get latest data
            data = await market_data.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(data) == 0:
                print(f"No data available for {symbol}. Skipping.")
                continue
                
            current_price = data['close'].iloc[-1]
            
            # Calculate shares using fixed dollar amount
            shares = fixed_amount / current_price
            shares = round(shares, 2)
            
            position_value = shares * current_price
            
            print(f"{symbol}: ${current_price:.2f} per share")
            print(f"  Shares: {shares} (${position_value:.2f} position, {(position_value/initial_capital)*100:.2f}% of portfolio)")
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    print("\n=== Equal Weight Position Sizing ===")
    num_positions = len(symbols)
    equal_weight_amount = initial_capital / num_positions
    
    for symbol in symbols:
        try:
            # Get latest data
            data = await market_data.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(data) == 0:
                print(f"No data available for {symbol}. Skipping.")
                continue
                
            current_price = data['close'].iloc[-1]
            
            # Calculate shares using equal weight
            shares = equal_weight_amount / current_price
            shares = round(shares, 2)
            
            position_value = shares * current_price
            
            print(f"{symbol}: ${current_price:.2f} per share")
            print(f"  Shares: {shares} (${position_value:.2f} position, {(position_value/initial_capital)*100:.2f}% of portfolio)")
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    print("\n=== Risk-Based Position Sizing ===")
    risk_percentage = 1.0  # Risk 1% of capital per trade
    
    for symbol in symbols:
        try:
            # Get latest data
            data = await market_data.get_historical_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if len(data) == 0:
                print(f"No data available for {symbol}. Skipping.")
                continue
                
            current_price = data['close'].iloc[-1]
            
            # Calculate ATR (Average True Range) for stop loss placement
            # Simplified ATR calculation for demonstration
            high = data['high']
            low = data['low']
            close = data['close']
            
            tr1 = np.abs(high - low)
            tr2 = np.abs(high - close.shift())
            tr3 = np.abs(low - close.shift())
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr_period = 14
            atr = true_range.rolling(window=atr_period).mean().iloc[-1]
            
            # Set stop loss using ATR
            stop_loss_distance = 2 * atr  # 2 x ATR for stop loss
            stop_loss_price = current_price - stop_loss_distance
            
            # Calculate position size based on risk
            shares = calculate_position_size(
                capital=initial_capital,
                risk_percentage=risk_percentage,
                entry_price=current_price,
                stop_loss_price=stop_loss_price
            )
            
            position_value = shares * current_price
            
            print(f"{symbol}: ${current_price:.2f} per share")
            print(f"  ATR: ${atr:.2f}, Stop Loss: ${stop_loss_price:.2f} (${stop_loss_distance:.2f} below current price)")
            print(f"  Risk Amount: ${initial_capital * (risk_percentage/100):.2f} ({risk_percentage}% of capital)")
            print(f"  Shares: {shares} (${position_value:.2f} position, {(position_value/initial_capital)*100:.2f}% of portfolio)")
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    print("\n=== Volatility-Adjusted Position Sizing ===")
    target_portfolio_volatility = 0.15  # 15% annualized target volatility
    
    total_allocated = 0
    allocations = {}
    
    for symbol in symbols:
        try:
            # Get latest data
            data = await market_data.get_historical_data(
                symbol=symbol,
                start_date=start_date - timedelta(days=365),  # Get 1 year of data for volatility calculation
                end_date=end_date
            )
            
            if len(data) < 252:  # Need at least 252 trading days (1 year)
                print(f"Insufficient data for {symbol}. Skipping.")
                continue
                
            current_price = data['close'].iloc[-1]
            
            # Calculate historical volatility
            returns = data['close'].pct_change().dropna()
            annual_volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Calculate inverse volatility weight
            if annual_volatility > 0:
                inverse_vol_weight = 1 / annual_volatility
                allocations[symbol] = {
                    'price': current_price,
                    'volatility': annual_volatility,
                    'inverse_weight': inverse_vol_weight
                }
                total_allocated += inverse_vol_weight
            else:
                print(f"{symbol}: Volatility is zero or negative. Skipping.")
        
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
    
    # Normalize weights and calculate positions
    if total_allocated > 0:
        for symbol, allocation in allocations.items():
            normalized_weight = allocation['inverse_weight'] / total_allocated
            position_value = initial_capital * normalized_weight
            shares = position_value / allocation['price']
            shares = round(shares, 2)
            
            print(f"{symbol}: ${allocation['price']:.2f} per share")
            print(f"  Annual Volatility: {allocation['volatility']*100:.2f}%")
            print(f"  Normalized Weight: {normalized_weight*100:.2f}%")
            print(f"  Shares: {shares} (${position_value:.2f} position)")
    
    print("\n=== Summary ===")
    print("Position sizing is a critical aspect of risk management in trading.")
    print("Different methods to consider:")
    print("1. Fixed Dollar Amount: Simple but doesn't account for asset volatility")
    print("2. Equal Weight: Balanced exposure across assets")
    print("3. Risk-Based: Sizes positions based on specific risk amount and stop loss")
    print("4. Volatility-Adjusted: Allocates more to less volatile assets")
    print("\nBest practice: Choose a position sizing method appropriate for your strategy and risk tolerance.")

if __name__ == "__main__":
    print("Position Sizing Example")
    print("=" * 50)
    asyncio.run(position_sizing_example())
