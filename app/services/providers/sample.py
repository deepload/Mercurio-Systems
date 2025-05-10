"""
Sample Data Provider

This module provides an implementation of the MarketDataProvider interface
that generates synthetic market data. This is useful for testing, demos,
and as a last-resort fallback when no other data sources are available.
"""
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import base provider
from app.services.providers.base import MarketDataProvider

logger = logging.getLogger(__name__)

class SampleDataProvider(MarketDataProvider):
    """
    Sample data provider implementation.
    
    Generates realistic-looking synthetic market data for testing and demo purposes.
    Always available as a fallback option when other providers fail.
    """
    
    def __init__(self):
        """Initialize the sample data provider."""
        # Preset values for popular stocks to make the sample data more realistic
        self.popular_stocks = {
            'AAPL': 180, 'MSFT': 420, 'GOOGL': 170, 'AMZN': 180, 'META': 480,
            'TSLA': 175, 'NVDA': 880, 'JPM': 190, 'V': 275, 'WMT': 60,
            'JNJ': 150, 'PG': 165, 'XOM': 115, 'BAC': 38, 'DIS': 110
        }
        
        # Stocks with typically higher volatility
        self.high_volatility_stocks = ['TSLA', 'NVDA', 'COIN', 'GME', 'AMC']
    
    @property
    def name(self) -> str:
        """Get provider name"""
        return "Sample Data"
    
    @property
    def requires_api_key(self) -> bool:
        """Check if API key is required"""
        return False
    
    @property
    def is_available(self) -> bool:
        """Check if this provider is available to use"""
        return True  # Always available
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        Generate realistic sample market data.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe for data (e.g., '1d', '1h')
            
        Returns:
            DataFrame with synthetic data
        """
        # Generate date range based on timeframe
        if timeframe == "1d":
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
        elif timeframe == "1h":
            dates = pd.date_range(start=start_date, end=end_date, freq='H')
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Set base price based on known symbols or hash for unknown ones
        if symbol in self.popular_stocks:
            base_price = self.popular_stocks[symbol]
        else:
            symbol_hash = sum(ord(c) for c in symbol) % 100
            base_price = 50 + symbol_hash
        
        # Generate more realistic price movement
        n = len(dates)
        
        # Parameters based on symbol to simulate different volatilities
        volatility = 0.015  # Default volatility
        if symbol in self.high_volatility_stocks:
            volatility = 0.035  # Higher volatility for these stocks
        
        # Create more realistic price movements with random walk + trend
        trend = np.linspace(0, base_price * 0.1, n)  # 10% trend over the period
        cumulative_returns = np.random.normal(0, volatility, n).cumsum()
        prices = base_price * (1 + cumulative_returns + trend/base_price)
        
        # Add some mean reversion
        for i in range(3, n):
            # Mean reversion factor
            mean_reversion = 0.05 * (prices[i-1] - np.mean(prices[max(0, i-10):i-1]))
            prices[i] = prices[i] - mean_reversion
        
        # Ensure prices are positive
        prices = np.maximum(prices, 1)
        
        # Generate more realistic OHLCV data
        data = {
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, volatility * 0.7))) for p in prices],
            'low': [min(o, c) * (1 - abs(np.random.normal(0, volatility * 0.7))) 
                    for o, c in zip(prices, [p * (1 + np.random.normal(0, volatility * 0.3)) for p in prices])],
            'close': [p * (1 + np.random.normal(0, volatility * 0.3)) for p in prices],
            'volume': [int(base_price * 10000 * (1 + abs(np.random.normal(0, 1.0)))) for _ in range(n)]
        }
        
        # Ensure high is always the highest and low is always the lowest
        for i in range(n):
            data['high'][i] = max(data['open'][i], data['close'][i], data['high'][i])
            data['low'][i] = min(data['open'][i], data['close'][i], data['low'][i])
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        logger.info(f"Generated sample data for {symbol} with {len(df)} data points")
        return df
    
    async def get_latest_price(self, symbol: str) -> float:
        """
        Generate a sample latest price for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            
        Returns:
            Synthetic latest price
        """
        if symbol in self.popular_stocks:
            base_price = self.popular_stocks[symbol]
        else:
            symbol_hash = sum(ord(c) for c in symbol) % 100
            base_price = 50 + symbol_hash
        
        # Add some random noise to the price
        volatility = 0.02
        if symbol in self.high_volatility_stocks:
            volatility = 0.04
            
        price = base_price * (1 + np.random.normal(0, volatility))
        logger.info(f"Generated sample price for {symbol}: ${price:.2f}")
        return price
    
    async def get_market_symbols(self, market_type: str = "stock") -> List[str]:
        """
        Get a list of sample market symbols.
        
        Args:
            market_type: Type of market ('stock', 'crypto', etc.)
            
        Returns:
            List of sample symbols
        """
        if market_type == "stock":
            return list(self.popular_stocks.keys())
        elif market_type == "crypto":
            return ["BTC", "ETH", "XRP", "LTC", "DOGE"]
        else:
            return ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
