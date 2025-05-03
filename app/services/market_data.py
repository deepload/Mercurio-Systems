"""
Market Data Service

Provides access to historical and real-time market data through
external data providers with a pluggable provider system.
"""
import os
import logging
from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio

# For Alpaca API (legacy support)
import alpaca_trade_api as tradeapi

# Import provider system
from app.services.providers.factory import MarketDataProviderFactory
from app.services.providers.base import MarketDataProvider

# Import environment loader to ensure .env variables are loaded
from app.utils import env_loader

logger = logging.getLogger(__name__)

class MarketDataService:
    """
    Service for retrieving market data from various providers.
    
    Supports:
    - Multiple pluggable data providers (Polygon.io, Yahoo Finance, etc.)
    - Legacy support for direct Alpaca API access
    - Automatic fallback to free providers when paid APIs are unavailable
    - Sample data generation as a last resort fallback
    """
    
    def __init__(self, provider_name: Optional[str] = None):
        """
        Initialize the market data service.
        
        Args:
            provider_name: Optional name of the preferred provider to use
        """
        # Initialize provider factory
        self.provider_factory = MarketDataProviderFactory()
        
        # Set active provider based on preference or availability
        self.active_provider_name = provider_name
        self._active_provider = None
        
        # Initialize legacy Alpaca client for backward compatibility
        self.alpaca_key = os.getenv("ALPACA_KEY")
        self.alpaca_secret = os.getenv("ALPACA_SECRET")
        self.alpaca_client = None
        if self.alpaca_key and self.alpaca_secret:
            try:
                # Fix the base_url if it contains /v2 at the end
                base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
                if base_url.endswith("/v2"):
                    base_url = base_url.rstrip("/v2")
                    logger.info(f"Removed '/v2' from Alpaca base URL: {base_url}")
                
                self.alpaca_client = tradeapi.REST(
                    key_id=self.alpaca_key,
                    secret_key=self.alpaca_secret,
                    base_url=base_url
                )
                logger.info("Legacy Alpaca client initialized successfully")
                
                # Register Alpaca as a provider option
                class AlpacaProvider(MarketDataProvider):
                    def __init__(self, client):
                        self.client = client
                    
                    @property
                    def name(self):
                        return "Alpaca"
                        
                    @property
                    def requires_api_key(self):
                        return True
                        
                    async def get_historical_data(self, symbol, start_date, end_date, timeframe="1d"):
                        # Implementation using self.client
                        pass
                        
                    async def get_latest_price(self, symbol):
                        # Implementation using self.client
                        pass
                        
                    async def get_market_symbols(self, market_type="stock"):
                        # Implementation using self.client
                        pass
                
                # Don't actually register this as it's just for compatibility
                # self.provider_factory.register_provider("alpaca_legacy", AlpacaProvider)
                
            except Exception as e:
                logger.error(f"Failed to initialize Alpaca client: {e}")
    
    @property
    def active_provider(self) -> MarketDataProvider:
        """
        Get the currently active provider.
        
        Returns:
            The active provider instance
        """
        if self._active_provider:
            return self._active_provider
            
        # If user specified a provider, try to use it
        if self.active_provider_name:
            provider = self.provider_factory.get_provider(self.active_provider_name)
            if provider:
                self._active_provider = provider
                return provider
        
        # Otherwise get the default provider based on availability
        self._active_provider = self.provider_factory.get_default_provider()
        return self._active_provider
        
    def set_provider(self, provider_name: str) -> bool:
        """
        Set the active provider by name.
        
        Args:
            provider_name: Name of the provider to use
            
        Returns:
            True if provider was set successfully, False otherwise
        """
        provider = self.provider_factory.get_provider(provider_name)
        if provider:
            self._active_provider = provider
            self.active_provider_name = provider_name
            logger.info(f"Switched to {provider_name} provider")
            return True
        return False
        
    def get_available_providers(self) -> List[str]:
        """
        Get a list of all available provider names.
        
        Returns:
            List of provider names
        """
        return self.provider_factory.get_available_providers()
    
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
        # Get data from Alpaca (primary source)
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
                # Format dates per Alpaca's requirements
                if alpaca_timeframe in ["1Day", "1d"]:
                    start_str = start_date.strftime("%Y-%m-%d")
                    end_str = end_date.strftime("%Y-%m-%d")
                else:
                    # Use RFC3339 for intraday bars
                    start_str = start_date.isoformat() + "Z"
                    end_str = end_date.isoformat() + "Z"
                # Convertir le format du symbole pour les cryptos si nécessaire
                alpaca_symbol = symbol
                if "-USD" in symbol:
                    alpaca_symbol = symbol.replace("-USD", "USD")
                    logging.info(f"Converting crypto symbol for Alpaca: {symbol} -> {alpaca_symbol}")
                
                data = self.alpaca_client.get_bars(
                    alpaca_symbol, 
                    alpaca_timeframe,
                    start=start_str,
                    end=end_str
                ).df
                
                if not data.empty:
                    # Rename columns to lowercase
                    data.columns = [col.lower() for col in data.columns]
                    return data
                
            except Exception as e:
                import traceback
                logger.error(f"Error fetching data from Alpaca: {e}")
                # Print full HTTP response if available
                if hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Alpaca response status: {e.response.status_code}")
                    logger.error(f"Alpaca response content: {e.response.text}")
                traceback.print_exc()
        
        # Fallback to sample data if API calls fail
        logger.warning(f"Using sample data for {symbol} as fallback")
        return await self._generate_sample_data(symbol, start_date, end_date, timeframe)
    
    async def get_latest_price(self, symbol: str, provider_name: Optional[str] = None) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            provider_name: Optional specific provider to use for this request
            
        Returns:
            Latest price
        """
        # If a specific provider is requested for this call
        if provider_name:
            provider = self.provider_factory.get_provider(provider_name)
            if provider:
                try:
                    return await provider.get_latest_price(symbol)
                except Exception as e:
                    logger.error(f"Error fetching latest price from {provider_name}: {e}")
        
        # Try with the active provider
        try:
            logger.info(f"Fetching latest price for {symbol} using {self.active_provider.name}")
            return await self.active_provider.get_latest_price(symbol)
        except Exception as e:
            logger.error(f"Error fetching latest price from {self.active_provider.name}: {e}")
            
        # Legacy fallback for backward compatibility
        if self.alpaca_client:
            try:
                # Convertir le format du symbole pour les cryptos si nécessaire
                alpaca_symbol = symbol
                if "-USD" in symbol:
                    alpaca_symbol = symbol.replace("-USD", "USD")
                    logger.info(f"Converting crypto symbol for Alpaca: {symbol} -> {alpaca_symbol}")
                
                logger.info(f"Falling back to legacy Alpaca client for {symbol} price")
                last_trade = self.alpaca_client.get_latest_trade(alpaca_symbol)
                if last_trade:
                    return last_trade.price
            except Exception as e:
                logger.error(f"Error fetching latest price from legacy Alpaca client: {e}")
        
        # Fallback to historical data
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)
            df = await self.get_historical_data(symbol, start_date, end_date)
            if not df.empty:
                return df['close'].iloc[-1]
        except Exception as e:
            logger.error(f"Historical data fallback failed: {e}")
        
        # Final fallback - try sample data provider
        sample_provider = self.provider_factory.get_provider("sample")
        if sample_provider:
            try:
                return await sample_provider.get_latest_price(symbol)
            except Exception as e:
                logger.error(f"Sample data provider failed: {e}")
        
        # If we still can't get a price, raise an exception
        raise ValueError(f"Could not get latest price for {symbol} from any source")
    
    async def get_market_symbols(self, market_type: str = "stock", provider_name: Optional[str] = None) -> List[str]:
        """
        Get a list of available market symbols.
        
        Args:
            market_type: Type of market ('stock', 'crypto', etc.)
            provider_name: Optional specific provider to use for this request
            
        Returns:
            List of available symbols
        """
        # If a specific provider is requested for this call
        if provider_name:
            provider = self.provider_factory.get_provider(provider_name)
            if provider:
                try:
                    return await provider.get_market_symbols(market_type)
                except Exception as e:
                    logger.error(f"Error fetching symbols from {provider_name}: {e}")
        
        # Try with the active provider
        try:
            logger.info(f"Fetching {market_type} symbols using {self.active_provider.name}")
            return await self.active_provider.get_market_symbols(market_type)
        except Exception as e:
            logger.error(f"Error fetching symbols from {self.active_provider.name}: {e}")
            
        # Legacy fallback for backward compatibility
        if self.alpaca_client:
            try:
                logger.info(f"Falling back to legacy Alpaca client for symbols")
                assets = self.alpaca_client.list_assets(status='active')
                symbols = [asset.symbol for asset in assets]
                return symbols[:100]  # Limit to 100 symbols
            except Exception as e:
                logger.error(f"Error fetching symbols from legacy Alpaca client: {e}")
        
        # Final fallback - try sample data provider for default symbols
        sample_provider = self.provider_factory.get_provider("sample")
        if sample_provider:
            try:
                return await sample_provider.get_market_symbols(market_type)
            except Exception as e:
                logger.error(f"Sample data provider failed for symbols: {e}")
                
        # Return a hardcoded default list if everything fails
        if market_type == "stock":
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA", "NVDA", "JPM", "JNJ", "V"]
        elif market_type == "crypto":
            return ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "DOGE-USD"]
        else:
            return ["AAPL", "MSFT", "AMZN", "GOOGL", "META"]
    
    # Legacy method for backward compatibility - delegates to sample provider
    async def _generate_sample_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d"
    ) -> pd.DataFrame:
        """
        Generate realistic sample market data (legacy method).
        This method is maintained for backward compatibility.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            start_date: Start date for data
            end_date: End date for data
            timeframe: Timeframe for data (e.g., '1d', '1h')
            
        Returns:
            DataFrame with sample data
        """
        sample_provider = self.provider_factory.get_provider("sample")
        if sample_provider:
            return await sample_provider.get_historical_data(symbol, start_date, end_date, timeframe)
            
        # If provider isn't available for some reason, create an empty dataframe
        logger.error("Sample data provider not available for fallback")
        return pd.DataFrame()
