"""
Polygon.io Market Data Provider

This module provides an implementation of the MarketDataProvider interface
for the Polygon.io API.
"""
import os
import logging
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

# Import base provider
from app.services.providers.base import MarketDataProvider

logger = logging.getLogger(__name__)

class PolygonProvider(MarketDataProvider):
    """
    Market data provider implementation for Polygon.io.
    
    Provides access to historical and real-time market data through
    the Polygon.io API.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Polygon.io provider.
        
        Args:
            api_key: Polygon.io API key (optional, will check env var if not provided)
        """
        self.api_key = api_key or os.getenv("POLYGON_API_KEY")
        self._client = None
        self._init_client()
    
    def _init_client(self) -> None:
        """Initialize the Polygon.io client"""
        if not self.api_key:
            logger.warning("No Polygon.io API key found")
            return
            
        try:
            # Lazy import to avoid dependency issues if Polygon is not used
            from polygon import RESTClient
            self._client = RESTClient(self.api_key)
            logger.info("Polygon.io client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Polygon.io client: {e}")
            self._client = None
    
    @property
    def name(self) -> str:
        """Get provider name"""
        return "Polygon.io"
    
    @property
    def requires_api_key(self) -> bool:
        """Check if API key is required"""
        return True
    
    @property
    def is_available(self) -> bool:
        """Check if this provider is available to use"""
        return self._client is not None
    
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
        if not self.is_available:
            raise ValueError("Polygon.io client is not available")
            
        # Map timeframe to Polygon format
        tf_mapping = {"1d": "day", "1h": "hour", "1m": "minute"}
        polygon_tf = tf_mapping.get(timeframe, "day")
        
        try:
            # Get data from Polygon
            aggs = self._client.get_aggs(
                ticker=symbol,
                multiplier=1,
                timespan=polygon_tf,
                from_=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime("%Y-%m-%d")
            )
            
            # Convert to DataFrame
            data = []
            for agg in aggs:
                data.append({
                    'date': pd.Timestamp(agg.timestamp, unit='ms'),
                    'open': agg.open,
                    'high': agg.high,
                    'low': agg.low,
                    'close': agg.close,
                    'volume': agg.volume
                })
            
            df = pd.DataFrame(data)
            
            # Set date as index
            if not df.empty:
                df.set_index('date', inplace=True)
                
            logger.info(f"Got {len(df)} data points for {symbol} from Polygon.io")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from Polygon.io: {e}")
            raise
    
    async def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            
        Returns:
            Latest price
        """
        if not self.is_available:
            raise ValueError("Polygon.io client is not available")
            
        try:
            # Get last trade
            last_trade = self._client.get_last_trade(symbol)
            return last_trade.price
        except Exception as e:
            logger.error(f"Error fetching latest price from Polygon.io: {e}")
            
            # Try getting it from recent aggs as fallback
            try:
                end_date = datetime.now()
                start_date = end_date - timedelta(days=1)
                df = await self.get_historical_data(symbol, start_date, end_date)
                if not df.empty:
                    return df['close'].iloc[-1]
            except Exception as nested_e:
                logger.error(f"Fallback also failed: {nested_e}")
                
            raise ValueError(f"Could not get latest price for {symbol} from Polygon.io")
    
    async def get_market_symbols(self, market_type: str = "stock") -> List[str]:
        """
        Get a list of available market symbols.
        
        Args:
            market_type: Type of market ('stock', 'crypto', etc.)
            
        Returns:
            List of available symbols
        """
        if not self.is_available:
            raise ValueError("Polygon.io client is not available")
            
        try:
            # Map market type to Polygon market
            market_mapping = {
                "stock": "stocks", 
                "crypto": "crypto",
                "forex": "fx"
            }
            polygon_market = market_mapping.get(market_type, "stocks")
            
            # Get tickers
            tickers = self._client.get_tickers(market=polygon_market, active=True)
            symbols = [ticker.ticker for ticker in tickers]
            return symbols[:100]  # Limit to 100 symbols
        except Exception as e:
            logger.error(f"Error fetching symbols from Polygon.io: {e}")
            raise
