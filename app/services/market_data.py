"""
Market Data Service

Provides access to historical and real-time market data through
external data providers (IEX Cloud, Alpaca, etc.).
"""
import os
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# For IEX Cloud API
import pyEX as px

# For Alpaca API
import alpaca_trade_api as tradeapi

logger = logging.getLogger(__name__)

class MarketDataService:
    """
    Service for retrieving market data from various providers.
    
    Supports:
    - Historical data from IEX Cloud
    - Real-time and historical data from Alpaca
    - Fallback to sample data if no API keys are provided
    """
    
    def __init__(self):
        """Initialize the market data service with API clients"""
        # Initialize IEX Cloud client if API key is available
        self.iex_api_key = os.getenv("IEX_API_KEY")
        self.iex_client = None
        if self.iex_api_key:
            try:
                self.iex_client = px.Client(api_token=self.iex_api_key)
                logger.info("IEX Cloud client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize IEX Cloud client: {e}")
        
        # Initialize Alpaca client if API key is available
        self.alpaca_key = os.getenv("ALPACA_KEY")
        self.alpaca_secret = os.getenv("ALPACA_SECRET")
        self.alpaca_client = None
        if self.alpaca_key and self.alpaca_secret:
            try:
                self.alpaca_client = tradeapi.REST(
                    key_id=self.alpaca_key,
                    secret_key=self.alpaca_secret,
                    base_url=os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
                )
                logger.info("Alpaca client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca client: {e}")
    
    async def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe for data (e.g., '1d', '1h')
            
        Returns:
            DataFrame with historical data
        """
        # Try to get data from IEX Cloud first
        if self.iex_client:
            try:
                logger.info(f"Fetching historical data for {symbol} from IEX Cloud")
                
                # Format dates for IEX API
                start_str = start_date.strftime("%Y%m%d")
                
                # Get historical data
                data = self.iex_client.chartDF(symbol, timeframe=timeframe, from_=start_str)
                
                # Filter by date range
                if not data.empty:
                    data = data[(data.index >= pd.Timestamp(start_date)) & 
                                (data.index <= pd.Timestamp(end_date))]
                    
                    # Rename columns to lowercase
                    data.columns = [col.lower() for col in data.columns]
                    
                    # Ensure we have the required columns
                    required_columns = ['open', 'high', 'low', 'close', 'volume']
                    if all(col in data.columns for col in required_columns):
                        return data
            
            except Exception as e:
                logger.error(f"Error fetching data from IEX Cloud: {e}")
        
        # Try to get data from Alpaca as a fallback
        if self.alpaca_client:
            try:
                logger.info(f"Fetching historical data for {symbol} from Alpaca")
                
                # Convert timeframe to Alpaca format
                alpaca_timeframe = timeframe
                if timeframe == "1d":
                    alpaca_timeframe = "1Day"
                elif timeframe == "1h":
                    alpaca_timeframe = "1Hour"
                
                # Get historical data
                data = self.alpaca_client.get_bars(
                    symbol, 
                    alpaca_timeframe,
                    start=start_date.isoformat(),
                    end=end_date.isoformat()
                ).df
                
                if not data.empty:
                    # Rename columns to lowercase
                    data.columns = [col.lower() for col in data.columns]
                    return data
                
            except Exception as e:
                logger.error(f"Error fetching data from Alpaca: {e}")
        
        # Fallback to sample data if API calls fail
        logger.warning(f"Using sample data for {symbol} as fallback")
        return self._generate_sample_data(symbol, start_date, end_date, timeframe)
    
    async def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            
        Returns:
            Latest price
        """
        # Try IEX Cloud first
        if self.iex_client:
            try:
                quote = self.iex_client.quote(symbol)
                if quote and 'latestPrice' in quote:
                    return quote['latestPrice']
            except Exception as e:
                logger.error(f"Error fetching latest price from IEX Cloud: {e}")
        
        # Try Alpaca as fallback
        if self.alpaca_client:
            try:
                last_trade = self.alpaca_client.get_latest_trade(symbol)
                if last_trade:
                    return last_trade.price
            except Exception as e:
                logger.error(f"Error fetching latest price from Alpaca: {e}")
        
        # If all else fails, get the last price from historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        df = await self.get_historical_data(symbol, start_date, end_date)
        if not df.empty:
            return df['close'].iloc[-1]
        
        # If we still can't get a price, raise an exception
        raise ValueError(f"Could not get latest price for {symbol}")
    
    async def get_market_symbols(self, market_type: str = "stock") -> List[str]:
        """
        Get a list of available market symbols.
        
        Args:
            market_type: Type of market ('stock', 'crypto', etc.)
            
        Returns:
            List of available symbols
        """
        symbols = []
        
        # Try IEX Cloud first
        if self.iex_client:
            try:
                if market_type == "stock":
                    symbols_data = self.iex_client.symbols()
                    symbols = [s['symbol'] for s in symbols_data if s['isEnabled']]
                return symbols[:100]  # Limit to 100 symbols
            except Exception as e:
                logger.error(f"Error fetching symbols from IEX Cloud: {e}")
        
        # Try Alpaca as fallback
        if self.alpaca_client:
            try:
                assets = self.alpaca_client.list_assets(status='active')
                symbols = [asset.symbol for asset in assets]
                return symbols[:100]  # Limit to 100 symbols
            except Exception as e:
                logger.error(f"Error fetching symbols from Alpaca: {e}")
        
        # Return a default list of common symbols as fallback
        return ["AAPL", "MSFT", "AMZN", "GOOGL", "FB", "TSLA", "NVDA", "JPM", "JNJ", "V"]
    
    def _generate_sample_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        Generate sample market data for testing.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe for data (e.g., '1d', '1h')
            
        Returns:
            DataFrame with sample data
        """
        # Generate date range
        if timeframe == "1d":
            dates = pd.date_range(start=start_date, end=end_date, freq='B')
        elif timeframe == "1h":
            dates = pd.date_range(start=start_date, end=end_date, freq='H')
        else:
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Base price depends on the symbol (just for variety in the sample data)
        symbol_hash = sum(ord(c) for c in symbol) % 100
        base_price = 50 + symbol_hash
        
        # Generate random price data with a slight upward trend
        n = len(dates)
        trend = np.linspace(0, 15, n)  # Upward trend
        noise = np.random.normal(0, 1, n)  # Random noise
        prices = base_price + trend + noise * 3
        
        # Ensure prices are positive
        prices = np.maximum(prices, 1)
        
        # Generate OHLCV data
        data = {
            'date': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'volume': [int(1000000 * abs(np.random.normal(1, 0.5))) for _ in range(n)]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        
        logger.info(f"Generated sample data for {symbol} with {len(df)} data points")
        return df
