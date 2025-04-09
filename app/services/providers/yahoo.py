"""
Yahoo Finance Market Data Provider

This module provides an implementation of the MarketDataProvider interface
for Yahoo Finance using the yfinance package. This provider doesn't require
an API key and can be used as a fallback.
"""
import os
import logging
import pandas as pd
import numpy as np
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import sys
import importlib

# Import base provider
from app.services.providers.base import MarketDataProvider

# Setup logger first
logger = logging.getLogger(__name__)

# Try to directly import yfinance to make sure it's available
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    logger.warning("yfinance package not available. If you just installed it, you may need to restart Python.")

class YahooFinanceProvider(MarketDataProvider):
    """
    Market data provider implementation for Yahoo Finance.
    
    Provides access to historical and some real-time market data through
    the Yahoo Finance API (via yfinance package). This provider is free
    and doesn't require an API key, making it a good fallback option.
    """
    
    def __init__(self):
        """Initialize the Yahoo Finance provider."""
        # No API key needed for Yahoo Finance
        pass
    
    @property
    def name(self) -> str:
        """Get provider name"""
        return "Yahoo Finance"
    
    @property
    def requires_api_key(self) -> bool:
        """Check if API key is required"""
        return False
    
    @property
    def is_available(self) -> bool:
        """Check if this provider is available to use"""
        # Use the global flag we set during import
        return YFINANCE_AVAILABLE
    
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
        # Map timeframe to Yahoo format
        tf_mapping = {"1d": "1d", "1h": "1h", "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m"}
        yf_interval = tf_mapping.get(timeframe, "1d")
        
        # Handle case for non-daily data which has more limited history in Yahoo Finance
        if yf_interval != "1d" and (end_date - start_date).days > 60:
            logger.warning(f"Yahoo Finance has limited history for intraday data. Limiting to last 60 days.")
            start_date = end_date - timedelta(days=60)
        
        try:
            # If yfinance isn't available, we wouldn't get here, but just to be safe
            if not YFINANCE_AVAILABLE:
                logger.error("yfinance package is required but not available")
                raise ValueError("Could not get latest price - yfinance package not available")
                
            # No need to import again as we've already imported it globally if available
            
            # Run download in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            data = await loop.run_in_executor(
                None,
                lambda: yf.download(
                    symbol,
                    start=start_date.strftime("%Y-%m-%d"),
                    end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),  # Add one day to include end_date
                    interval=yf_interval,
                    progress=False,
                    show_errors=False
                )
            )
            
            # Process data
            if data.empty:
                logger.warning(f"No data returned from Yahoo Finance for {symbol}")
                return pd.DataFrame()
                
            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in data.columns:
                    if col == 'volume' and 'adj close' in data.columns:
                        # Some symbols might not have volume data
                        data['volume'] = 0
                    else:
                        logger.warning(f"Missing column {col} in Yahoo Finance data")
                        return pd.DataFrame()
            
            logger.info(f"Got {len(data)} data points for {symbol} from Yahoo Finance")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {e}")
            return pd.DataFrame()
    
    async def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            
        Returns:
            Latest price
        """
        try:
            # If yfinance isn't available, we wouldn't get here, but just to be safe
            if not YFINANCE_AVAILABLE:
                logger.error("yfinance package is required but not available")
                raise ValueError("Could not get latest price - yfinance package not available")
                
            # No need to import again as we've already imported it globally if available
            
            # Run ticker info in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            ticker = yf.Ticker(symbol)
            info = await loop.run_in_executor(None, lambda: ticker.info)
            
            # Try to get current price
            if 'regularMarketPrice' in info and info['regularMarketPrice'] is not None:
                return float(info['regularMarketPrice'])
            
            # Fallback to previous close
            if 'previousClose' in info and info['previousClose'] is not None:
                return float(info['previousClose'])
                
            raise ValueError("No price data available")
            
        except Exception as e:
            logger.error(f"Error fetching latest price from Yahoo Finance: {e}")
            
            # Try getting it from recent historical data as fallback
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=5)
                df = await self.get_historical_data(symbol, start_date, end_date)
                if not df.empty:
                    return df['close'].iloc[-1]
            except Exception as nested_e:
                logger.error(f"Fallback also failed: {nested_e}")
            
            raise ValueError(f"Could not get latest price for {symbol} from Yahoo Finance")
    
    async def get_market_symbols(self, market_type: str = "stock") -> List[str]:
        """
        Get a list of available market symbols.
        
        Args:
            market_type: Type of market ('stock', 'crypto', etc.)
            
        Returns:
            List of available symbols
        """
        # Yahoo Finance doesn't have a good API for listing all symbols
        # So we'll return a default list of common symbols based on market type
        
        if market_type == "stock":
            return [
                "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "NVDA", "JPM", "V",
                "JNJ", "WMT", "PG", "MA", "UNH", "HD", "BAC", "XOM", "ADBE", "CRM",
                "PFE", "NFLX", "CMCSA", "KO", "PEP", "AVGO", "CSCO", "ABT", "TMO", "ACN",
                "COST", "DIS", "MRK", "VZ", "INTC", "QCOM", "NKE", "T", "WFC", "AMD"
            ]
        elif market_type == "crypto":
            return [
                "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "XRP-USD",
                "SOL-USD", "DOGE-USD", "ADA-USD", "AVAX-USD", "DOT-USD"
            ]
        elif market_type == "forex":
            return [
                "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
                "USDCHF=X", "NZDUSD=X", "EURJPY=X", "GBPJPY=X", "EURGBP=X"
            ]
        else:
            # Default to common stocks
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "V", "JNJ"]
