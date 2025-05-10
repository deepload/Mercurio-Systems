"""
MercurioAI Enhanced Data Pipeline

This module implements an advanced data pipeline for market data processing,
with support for real-time streams, caching, and transformations.
"""
import asyncio
import logging
import time
from typing import Dict, List, Any, Callable, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import aiohttp
import json
import os
from pathlib import Path
import pickle

from .event_bus import EventBus, EventType

logger = logging.getLogger(__name__)

# Create cache directory if it doesn't exist
CACHE_DIR = Path("./data/cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class DataTransformer(ABC):
    """Abstract base class for data transformers"""
    
    @abstractmethod
    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform input data
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        pass


class DataCleaner(DataTransformer):
    """Cleans data by handling missing values, outliers, etc."""
    
    def __init__(self, fill_method: str = 'ffill', drop_na: bool = False, 
                 remove_outliers: bool = False, zscore_threshold: float = 3.0):
        """
        Initialize data cleaner
        
        Args:
            fill_method: Method to fill missing values ('ffill', 'bfill', 'interpolate')
            drop_na: Whether to drop rows with NaN values
            remove_outliers: Whether to remove outliers
            zscore_threshold: Z-score threshold for outlier detection
        """
        self.fill_method = fill_method
        self.drop_na = drop_na
        self.remove_outliers = remove_outliers
        self.zscore_threshold = zscore_threshold
    
    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean the data"""
        df = data.copy()
        
        # Handle missing values
        if self.fill_method == 'ffill':
            df = df.ffill()
        elif self.fill_method == 'bfill':
            df = df.bfill()
        elif self.fill_method == 'interpolate':
            df = df.interpolate()
            
        if self.drop_na:
            df = df.dropna()
            
        # Handle outliers if needed
        if self.remove_outliers:
            for col in df.select_dtypes(include=[np.number]).columns:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    continue  # Don't remove outliers from price data
                    
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                df = df[z_scores < self.zscore_threshold]
                
        return df


class FeatureEngineer(DataTransformer):
    """Adds derived features to market data"""
    
    def __init__(self, add_ta: bool = True, add_time: bool = True):
        """
        Initialize feature engineer
        
        Args:
            add_ta: Whether to add technical indicators
            add_time: Whether to add time-based features
        """
        self.add_ta = add_ta
        self.add_time = add_time
        
    async def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add features to the data"""
        df = data.copy()
        
        # Add technical indicators
        if self.add_ta:
            # Only import if needed
            import talib as ta
            
            # Check if we have OHLCV data
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            has_ohlcv = all(col in df.columns for col in required_cols)
            
            if has_ohlcv:
                # Add basic technical indicators
                df['sma_20'] = ta.SMA(df['close'], timeperiod=20)
                df['sma_50'] = ta.SMA(df['close'], timeperiod=50)
                df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)
                df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(
                    df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
                df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
                
                # Add some more advanced indicators
                df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = ta.BBANDS(
                    df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
                df['adx_14'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        
        # Add time-based features
        if self.add_time and 'timestamp' in df.columns:
            if isinstance(df['timestamp'].iloc[0], str):
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['hour_of_day'] = df['timestamp'].dt.hour
            df['month'] = df['timestamp'].dt.month
            df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
            df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
            
        return df


class DataCache:
    """Cache for market data to reduce API calls"""
    
    def __init__(self, max_age_days: int = 1, max_items: int = 1000):
        """
        Initialize data cache
        
        Args:
            max_age_days: Maximum age of cached data in days
            max_items: Maximum number of items in cache
        """
        self.max_age_days = max_age_days
        self.max_items = max_items
        self.cache: Dict[str, Tuple[pd.DataFrame, datetime]] = {}
        
    async def get(self, key: str) -> Optional[pd.DataFrame]:
        """
        Get data from cache
        
        Args:
            key: Cache key
            
        Returns:
            DataFrame if cache hit, None otherwise
        """
        if key not in self.cache:
            return None
            
        df, timestamp = self.cache[key]
        age = datetime.now() - timestamp
        
        if age.days > self.max_age_days:
            # Cache expired
            return None
            
        return df
        
    async def set(self, key: str, data: pd.DataFrame) -> None:
        """
        Store data in cache
        
        Args:
            key: Cache key
            data: DataFrame to cache
        """
        # Check if we need to evict items
        if len(self.cache) >= self.max_items:
            # Remove oldest item
            oldest_key = min(self.cache.items(), key=lambda x: x[1][1])[0]
            del self.cache[oldest_key]
            
        self.cache[key] = (data, datetime.now())
        
        # Also save to disk
        cache_path = CACHE_DIR / f"{key.replace(':', '_')}.pkl"
        data.to_pickle(cache_path)
        
    async def clear(self, older_than_days: Optional[int] = None) -> int:
        """
        Clear cache items
        
        Args:
            older_than_days: Only clear items older than this many days (None for all)
            
        Returns:
            Number of items cleared
        """
        if older_than_days is None:
            count = len(self.cache)
            self.cache.clear()
            return count
            
        keys_to_remove = []
        now = datetime.now()
        
        for key, (_, timestamp) in self.cache.items():
            age = now - timestamp
            if age.days > older_than_days:
                keys_to_remove.append(key)
                
        for key in keys_to_remove:
            del self.cache[key]
            
        return len(keys_to_remove)


class EnhancedDataPipeline:
    """
    Enhanced data pipeline with caching, transformations and event publishing
    """
    
    def __init__(self):
        """Initialize the enhanced data pipeline"""
        self.event_bus = EventBus()
        self.cache = DataCache()
        self.transformers: List[DataTransformer] = []
        self.market_data_service = None  # Will be set later
        
    def add_transformer(self, transformer: DataTransformer) -> None:
        """
        Add a data transformer to the pipeline
        
        Args:
            transformer: DataTransformer instance
        """
        self.transformers.append(transformer)
        logger.info(f"Added transformer: {transformer.__class__.__name__}")
        
    def set_market_data_service(self, service: Any) -> None:
        """
        Set the market data service
        
        Args:
            service: MarketDataService instance
        """
        self.market_data_service = service
        
    async def get_data(self, 
                       symbol: str, 
                       start_date: Union[datetime, str], 
                       end_date: Union[datetime, str],
                       interval: str = "1d",
                       use_cache: bool = True,
                       apply_transformations: bool = True) -> pd.DataFrame:
        """
        Get market data with caching and transformations
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            interval: Data interval ('1m', '5m', '1h', '1d', etc.)
            use_cache: Whether to use cache
            apply_transformations: Whether to apply transformations
            
        Returns:
            DataFrame with market data
        """
        # Convert dates to strings if they are datetime objects
        start_str = start_date.strftime("%Y-%m-%d") if isinstance(start_date, datetime) else start_date
        end_str = end_date.strftime("%Y-%m-%d") if isinstance(end_date, datetime) else end_date
        
        # Create cache key
        cache_key = f"{symbol}:{start_str}:{end_str}:{interval}"
        
        # Try to get from cache
        if use_cache:
            cached_data = await self.cache.get(cache_key)
            if cached_data is not None:
                logger.debug(f"Cache hit for {cache_key}")
                
                if apply_transformations:
                    return await self._apply_transformations(cached_data)
                return cached_data
        
        # Get data from market data service
        if self.market_data_service is None:
            raise ValueError("Market data service not set")
            
        logger.debug(f"Getting data for {symbol} from {start_str} to {end_str}")
        data = await self.market_data_service.get_historical_data(symbol, start_str, end_str)
        
        # Cache the data
        if use_cache and data is not None and not data.empty:
            await self.cache.set(cache_key, data)
            
        # Publish event
        await self.event_bus.publish(
            EventType.MARKET_DATA_UPDATED,
            {
                "symbol": symbol,
                "start_date": start_str,
                "end_date": end_str,
                "interval": interval,
                "data_points": len(data) if data is not None else 0
            }
        )
        
        # Apply transformations if needed
        if apply_transformations and data is not None and not data.empty:
            data = await self._apply_transformations(data)
            
        return data
    
    async def _apply_transformations(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all transformers to the data
        
        Args:
            data: Input DataFrame
            
        Returns:
            Transformed DataFrame
        """
        result = data.copy()
        for transformer in self.transformers:
            try:
                result = await transformer.transform(result)
            except Exception as e:
                logger.error(f"Error in transformer {transformer.__class__.__name__}: {e}")
                
        return result
    
    async def stream_data(self, 
                         symbols: List[str], 
                         callback: Callable[[Dict[str, Any]], None],
                         interval: str = "1m") -> asyncio.Task:
        """
        Stream real-time market data
        
        Args:
            symbols: List of symbols to stream
            callback: Callback function for new data
            interval: Data interval
            
        Returns:
            Task that can be cancelled to stop streaming
        """
        if self.market_data_service is None:
            raise ValueError("Market data service not set")
            
        # Create and return a streaming task
        task = asyncio.create_task(self._stream_worker(symbols, callback, interval))
        return task
    
    async def _stream_worker(self, 
                            symbols: List[str], 
                            callback: Callable[[Dict[str, Any]], None],
                            interval: str) -> None:
        """
        Worker for streaming data
        
        Args:
            symbols: List of symbols to stream
            callback: Callback function for new data
            interval: Data interval
        """
        # Determine polling interval based on the requested data interval
        if interval == "1m":
            poll_seconds = 60
        elif interval == "5m":
            poll_seconds = 300
        elif interval == "1h":
            poll_seconds = 3600
        else:
            poll_seconds = 60  # Default
            
        while True:
            try:
                for symbol in symbols:
                    # Get latest data
                    end = datetime.now()
                    start = end - timedelta(minutes=5)  # Last 5 minutes
                    
                    data = await self.get_data(
                        symbol=symbol,
                        start_date=start,
                        end_date=end,
                        interval=interval,
                        use_cache=False  # Always get fresh data for streaming
                    )
                    
                    if data is not None and not data.empty:
                        # Get latest bar
                        latest = data.iloc[-1].to_dict()
                        latest['symbol'] = symbol
                        latest['interval'] = interval
                        
                        # Call the callback
                        callback(latest)
                        
                        # Publish event
                        await self.event_bus.publish(
                            EventType.MARKET_DATA_UPDATED,
                            {
                                "symbol": symbol,
                                "interval": interval,
                                "timestamp": datetime.now().isoformat(),
                                "data": latest
                            }
                        )
            except Exception as e:
                logger.error(f"Error in stream worker: {e}")
                
            # Wait for next poll
            await asyncio.sleep(poll_seconds)
    
    async def initialize_with_defaults(self) -> None:
        """Initialize pipeline with default transformers"""
        # Add default transformers
        self.add_transformer(DataCleaner())
        self.add_transformer(FeatureEngineer())
        
        logger.info("Data pipeline initialized with default transformers")
            

# Example usage
"""
# Create pipeline and initialize
pipeline = EnhancedDataPipeline()
await pipeline.initialize_with_defaults()

# Set market data service
from app.services.market_data import MarketDataService
pipeline.set_market_data_service(MarketDataService())

# Get data with caching and transformations
data = await pipeline.get_data("AAPL", "2023-01-01", "2023-01-31")

# Stream real-time data
def on_data(bar_data):
    print(f"New data: {bar_data['symbol']} at {bar_data['timestamp']}: {bar_data['close']}")

stream_task = await pipeline.stream_data(["AAPL", "MSFT"], on_data)

# To stop streaming
# stream_task.cancel()
"""
