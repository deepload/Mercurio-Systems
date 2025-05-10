"""
Mock Market Data Service for demo purposes
"""
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class MockMarketDataService:
    """
    A mock market data service that generates synthetic data for demo purposes.
    """
    
    def __init__(self):
        logger.info("Initializing mock market data service")
    
    async def get_historical_data(self, symbol, start_date, end_date):
        """Generate mock historical price data for a symbol"""
        logger.info(f"Generating mock data for {symbol} from {start_date} to {end_date}")
        
        # Create date range
        date_range = pd.date_range(start=start_date, end=end_date, freq="B")
        
        # Generate price data
        base_price = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0,
            "AMZN": 3000.0,
        }.get(symbol, random.uniform(50, 500))
        
        # Add some randomness with a slight upward trend
        prices = []
        current_price = base_price
        for _ in range(len(date_range)):
            # Daily return between -2% and +3%
            daily_return = random.uniform(-0.02, 0.03)
            current_price *= (1 + daily_return)
            prices.append(current_price)
        
        # Create DataFrame
        df = pd.DataFrame({
            "date": date_range,
            "open": prices,
            "high": [p * random.uniform(1.0, 1.05) for p in prices],
            "low": [p * random.uniform(0.95, 1.0) for p in prices],
            "close": [p * random.uniform(0.98, 1.02) for p in prices],
            "volume": [random.randint(100000, 10000000) for _ in range(len(date_range))]
        })
        
        logger.info(f"Generated {len(df)} data points for {symbol}")
        return df
    
    async def get_latest_price(self, symbol):
        """Get the latest price for a symbol"""
        # Return a random price based on typical values for the symbol
        base_price = {
            "AAPL": 150.0,
            "MSFT": 300.0,
            "GOOGL": 2500.0,
            "AMZN": 3000.0,
        }.get(symbol, 100.0)
        
        # Add some randomness
        latest_price = base_price * random.uniform(0.95, 1.05)
        logger.info(f"Mock latest price for {symbol}: ${latest_price:.2f}")
        return latest_price
    
    async def get_current_quotes(self, symbols):
        """Get current quotes for a list of symbols"""
        quotes = {}
        for symbol in symbols:
            base_price = {
                "AAPL": 150.0,
                "MSFT": 300.0,
                "GOOGL": 2500.0,
                "AMZN": 3000.0,
            }.get(symbol, 100.0)
            
            price = base_price * random.uniform(0.95, 1.05)
            quotes[symbol] = {
                "symbol": symbol,
                "last_price": price,
                "bid": price * 0.999,
                "ask": price * 1.001,
                "volume": random.randint(100000, 10000000)
            }
        
        return quotes
