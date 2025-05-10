"""
Base Market Data Provider Interface

This module defines the abstract base class for market data providers.
All concrete provider implementations should inherit from this class.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

class MarketDataProvider(ABC):
    """
    Abstract base class for market data providers.
    
    Defines the interface that all market data providers must implement.
    """
    
    @abstractmethod
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
        pass
        
    @abstractmethod
    async def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            
        Returns:
            Latest price
        """
        pass
        
    @abstractmethod
    async def get_market_symbols(self, market_type: str = "stock") -> List[str]:
        """
        Get a list of available market symbols.
        
        Args:
            market_type: Type of market ('stock', 'crypto', etc.)
            
        Returns:
            List of available symbols
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Get the name of the provider.
        
        Returns:
            Provider name
        """
        pass
    
    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """
        Whether this provider requires an API key.
        
        Returns:
            True if API key is required, False otherwise
        """
        pass
