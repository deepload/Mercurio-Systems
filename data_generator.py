"""
Mercurio AI - Data Generator

This module generates realistic market data for January 2025
to be used with the trading strategy simulations.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_market_data(symbol, start_date, end_date, initial_price=None, volatility=None):
    """
    Generate realistic OHLCV market data for a symbol.
    
    Args:
        symbol: The market symbol (e.g., 'AAPL', 'BTC-USD')
        start_date: Start date for generated data
        end_date: End date for generated data
        initial_price: Starting price (if None, uses a realistic default)
        volatility: Price volatility (if None, uses realistic default based on symbol)
    
    Returns:
        DataFrame with columns: date, open, high, low, close, volume
    """
    # Set realistic initial prices and volatility if not provided
    if initial_price is None:
        price_defaults = {
            'AAPL': 190.0, 'MSFT': 410.0, 'GOOGL': 160.0, 'AMZN': 180.0, 'TSLA': 175.0,
            'BTC-USD': 42000.0, 'ETH-USD': 2200.0, 'SOL-USD': 95.0, 'ADA-USD': 0.45, 'DOT-USD': 7.0
        }
        initial_price = price_defaults.get(symbol, 100.0)
    
    if volatility is None:
        # Higher volatility for crypto
        if '-USD' in symbol:
            volatility = 0.025  # 2.5% daily volatility for crypto
        else:
            volatility = 0.015  # 1.5% daily volatility for stocks
    
    # Generate date range (exclude weekends for stocks)
    date_range = []
    current_date = start_date
    is_crypto = '-USD' in symbol
    
    while current_date <= end_date:
        # For stocks, skip weekends
        if is_crypto or current_date.weekday() < 5:  # 0-4 are Monday to Friday
            date_range.append(current_date)
        current_date += timedelta(days=1)
    
    # Ensure at least 3 rows of data for simulation
    min_rows = 3
    if len(date_range) < min_rows:
        # Extend backwards in time if needed
        print(f"[WARNING] Not enough data points for {symbol} between {start_date} and {end_date}. Auto-extending date range to ensure at least {min_rows} data points.")
        needed = min_rows - len(date_range)
        ext_date = (date_range[0] if date_range else end_date)
        ext_dates = []
        while len(ext_dates) < needed:
            ext_date = ext_date - timedelta(days=1)
            if is_crypto or ext_date.weekday() < 5:
                ext_dates.insert(0, ext_date)
        date_range = ext_dates + date_range
    
    # Generate price data using geometric Brownian motion
    n_days = len(date_range)
    returns = np.random.normal(0, volatility, n_days)
    
    # Add a slight drift (upward bias for January 2025)
    drift = 0.001  # 0.1% daily drift
    returns = returns + drift
    
    # Calculate price series
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Generate realistic OHLCV data
    data = []
    for i, date in enumerate(date_range):
        close_price = prices[i]
        high_low_range = close_price * volatility * 1.5
        
        # Ensure first day price matches initial_price for the close
        if i == 0:
            close_price = initial_price
            
        open_price = prices[i-1] if i > 0 else close_price * (1 - volatility/2)
        high_price = max(open_price, close_price) + np.random.uniform(0, high_low_range)
        low_price = min(open_price, close_price) - np.random.uniform(0, high_low_range)
        
        # Generate volume (higher for more volatile days)
        price_change = abs(close_price - open_price)
        base_volume = close_price * 1000  # Base volume proportional to price
        volume = int(base_volume * (1 + 5 * price_change / close_price))
        
        data.append({
            'date': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('date', inplace=True)
    
    return df

def save_market_data(symbol, data, directory='data'):
    """Save market data to CSV file."""
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, f"{symbol.replace('-', '_')}_data.csv")
    data.to_csv(filename)
    print(f"Saved data for {symbol} to {filename}")
    return filename

def generate_all_market_data(symbols, start_date, end_date, directory='data'):
    """Generate and save market data for all symbols."""
    files = {}
    
    for symbol in symbols:
        print(f"Generating data for {symbol}...")
        data = generate_market_data(symbol, start_date, end_date)
        filename = save_market_data(symbol, data, directory)
        files[symbol] = filename
    
    return files

def load_market_data(symbol, directory='data'):
    """Load market data from CSV file."""
    filename = os.path.join(directory, f"{symbol.replace('-', '_')}_data.csv")
    if os.path.exists(filename):
        data = pd.read_csv(filename)
        data['date'] = pd.to_datetime(data['date'])
        data.set_index('date', inplace=True)
        return data
    else:
        return None

if __name__ == "__main__":
    # Test the data generator
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD', 'ETH-USD']
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 1, 31)
    
    files = generate_all_market_data(symbols, start_date, end_date)
    print(f"Generated data files: {files}")
    
    # Load and display sample data
    aapl_data = load_market_data('AAPL')
    if aapl_data is not None:
        print("\nSample AAPL data:")
        print(aapl_data.head())
