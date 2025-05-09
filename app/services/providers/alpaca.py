"""
Alpaca Market Data Provider

Provides market data through Alpaca's API.
"""
import os
import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import alpaca_trade_api as tradeapi
import requests

from app.services.providers.base import MarketDataProvider

logger = logging.getLogger(__name__)

class AlpacaProvider(MarketDataProvider):
    """
    Provider for Alpaca market data.
    
    Uses Alpaca's API directly to retrieve historical and real-time market data.
    This provider adapts to different Alpaca subscription levels:
    - Level 1 (Basic/Starter): Limited market data access
    - Level 2 (Pro): Extended data access and faster rates
    - Level 3 (AlgoTrader Plus): Premium data with options and full market depth
    """
    
    def __init__(self):
        """Initialize the Alpaca provider with API credentials and determine subscription level."""
        # Determine Alpaca mode (paper or live)
        alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
        
        # Configuration based on mode
        if alpaca_mode == "live":
            self.alpaca_key = os.getenv("ALPACA_LIVE_KEY")
            self.alpaca_secret = os.getenv("ALPACA_LIVE_SECRET")
            self.base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
            logger.info("AlpacaProvider: Configured for LIVE trading mode")
        else:  # paper mode by default
            self.alpaca_key = os.getenv("ALPACA_PAPER_KEY")
            self.alpaca_secret = os.getenv("ALPACA_PAPER_SECRET")
            self.base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
            logger.info("AlpacaProvider: Configured for PAPER trading mode")
        
        # Data URL is the same for both modes
        self.data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
        
        # Subscription level (default to 1 if not specified)
        self.subscription_level = int(os.getenv("ALPACA_SUBSCRIPTION_LEVEL", "1"))
        logger.info(f"AlpacaProvider: Using Alpaca subscription level {self.subscription_level}")
        
        # Initialize Alpaca client
        self.client = None
        if self.alpaca_key and self.alpaca_secret:
            try:
                # Remove /v2 from URL if present
                if self.base_url.endswith("/v2"):
                    self.base_url = self.base_url.rstrip("/v2")
                
                # Initialize the client with compatible parameters
                self.client = tradeapi.REST(
                    key_id=self.alpaca_key,
                    secret_key=self.alpaca_secret,
                    base_url=self.base_url
                )
                
                # Try to set data_url attribute if the client supports it
                try:
                    if hasattr(self.client, 'data_url'):
                        self.client.data_url = self.data_url
                        logger.info(f"AlpacaProvider: Set data_url attribute to {self.data_url}")
                except Exception as e:
                    logger.warning(f"AlpacaProvider: Could not set data_url attribute: {e}")
                logger.info(f"AlpacaProvider: Initialized Alpaca client with base_url: {self.base_url} and data_url: {self.data_url}")
            except Exception as e:
                logger.error(f"AlpacaProvider: Failed to initialize Alpaca client: {e}")
                self.client = None
    
    @property
    def name(self) -> str:
        """Return the provider name."""
        return "Alpaca"
    
    @property
    def requires_api_key(self) -> bool:
        """Return whether this provider requires API keys."""
        return True
    
    @property
    def is_available(self) -> bool:
        """Check if the provider is available."""
        return self.client is not None
        
    @property
    def has_options_data(self) -> bool:
        """Check if options data is available (subscription level 3)."""
        return self.subscription_level >= 3
        
    @property
    def has_extended_data(self) -> bool:
        """Check if extended market data is available (subscription level 2+)."""
        return self.subscription_level >= 2
    
    async def get_historical_data(self, symbol: str, start_date: datetime, end_date: datetime, timeframe: str = "1d") -> pd.DataFrame:
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
        if not self.client:
            logger.warning("AlpacaProvider: Client not initialized, cannot fetch historical data")
            return pd.DataFrame()
        
        try:
            # Format dates for API
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            # Map timeframe to Alpaca format
            alpaca_timeframe = timeframe
            if timeframe == "1d":
                alpaca_timeframe = "1Day"
            elif timeframe == "1h":
                alpaca_timeframe = "1Hour"
            
            # Check if it's a crypto symbol (contains '/')
            if '/' in symbol:
                logger.info(f"AlpacaProvider: Detected crypto symbol {symbol}, using crypto data API")
                return await self._get_crypto_data(symbol, start_str, end_str, alpaca_timeframe)
            
            # Default path for stocks
            logger.info(f"AlpacaProvider: Fetching historical data for {symbol} from {start_str} to {end_str} with timeframe {alpaca_timeframe}")
            
            # Ensure API call is compatible with installed version
            try:
                # Try the newer API first
                bars = self.client.get_bars(
                    symbol,
                    alpaca_timeframe,
                    start=start_str,
                    end=end_str,
                    limit=10000
                ).df
            except (TypeError, AttributeError):
                # Fall back to older API if needed
                logger.info(f"AlpacaProvider: Falling back to older Alpaca API for {symbol}")
                bars = self.client.get_barset(
                    symbols=symbol,
                    timeframe=alpaca_timeframe,
                    start=start_str,
                    end=end_str,
                    limit=10000
                ).df[symbol]
            
            # Process the data if we got any
            if not bars.empty:
                # Make sure the index is a datetime
                if not isinstance(bars.index, pd.DatetimeIndex):
                    bars.index = pd.to_datetime(bars.index)
                    
                logger.info(f"AlpacaProvider: Successfully retrieved {len(bars)} bars for {symbol}")
                return bars
            else:
                logger.warning(f"AlpacaProvider: No data returned for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"AlpacaProvider: Error fetching historical data: {str(e)}")
            # Try to be helpful with specific error messages
            if "not found" in str(e).lower():
                logger.warning(f"AlpacaProvider: Symbol {symbol} not found in Alpaca - make sure you have the right subscription level")
            return pd.DataFrame()
    
    async def _get_crypto_data(self, symbol: str, start_str: str, end_str: str, timeframe: str) -> pd.DataFrame:
        """
        Get crypto data directly from Alpaca API.
        
        Args:
            symbol: The crypto symbol (e.g., 'BTC/USD')
            start_str: Start date string in YYYY-MM-DD format
            end_str: End date string in YYYY-MM-DD format
            timeframe: Timeframe for data (e.g., '1Day', '1Hour')
            
        Returns:
            DataFrame with crypto data
        """
        try:
            # Use the crypto API endpoint format
            base_url = f"{self.data_url}/v1beta3/crypto/bars"
            
            # Map timeframe to v1beta3 format
            v1beta3_timeframe = timeframe
            if timeframe == "1Day":
                v1beta3_timeframe = "1D"
            elif timeframe == "1Hour":
                v1beta3_timeframe = "1H"
            
            # Request parameters
            params = {
                "symbols": symbol,
                "timeframe": v1beta3_timeframe,
                "start": start_str,
                "end": end_str,
                "limit": 1000
            }
            
            # Authentication headers
            headers = {
                "APCA-API-KEY-ID": self.alpaca_key,
                "APCA-API-SECRET-KEY": self.alpaca_secret
            }
            
            # Execute request
            logger.info(f"AlpacaProvider: Making direct API call to Alpaca crypto endpoint for {symbol}")
            response = requests.get(base_url, params=params, headers=headers)
            
            # Check response status
            if response.status_code == 200:
                data = response.json()
                
                # Verify we have data for this symbol
                if data and "bars" in data and symbol in data["bars"] and len(data["bars"][symbol]) > 0:
                    # Convert data to DataFrame
                    bars = data["bars"][symbol]
                    df = pd.DataFrame(bars)
                    
                    # Rename and format columns to match expected format
                    df.rename(columns={
                        "t": "timestamp",
                        "o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume"
                    }, inplace=True)
                    
                    # Convert timestamp column to datetime
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                    
                    logger.info(f"AlpacaProvider: Successfully retrieved {len(df)} bars for {symbol} from Alpaca v1beta3 API")
                    return df
                else:
                    logger.warning(f"AlpacaProvider: No data returned for {symbol} from Alpaca v1beta3 API")
                    return pd.DataFrame()
            else:
                error_msg = f"API error: {response.status_code} {response.text[:100]}"
                logger.warning(f"AlpacaProvider: {error_msg}")
                # Specifically check for authorization errors
                if response.status_code == 403:
                    logger.warning("AlpacaProvider: Received 403 Forbidden. Your Alpaca plan likely does not include crypto data access.")
                return pd.DataFrame()
        except Exception as e:
            logger.warning(f"AlpacaProvider: Error in direct API call to Alpaca: {str(e)[:200]}")
            return pd.DataFrame()

    async def get_latest_price(self, symbol: str) -> float:
        """
        Get the latest price for a symbol.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            
        Returns:
            Latest price
        """
        if not self.client:
            logger.warning("AlpacaProvider: Client not initialized, cannot fetch latest price")
            return 0.0

        try:
            # Check if it's a crypto symbol
            if '/' in symbol:
                # For crypto, use the crypto bars endpoint to get the latest price
                # Get the most recent bar
                timeframe = '1Min'
                try:
                    # Try using get_historical_data which is already implemented for crypto
                    end = datetime.now()
                    start = end - timedelta(minutes=10)
                    # Use the existing method that works with crypto
                    bars = await self.get_historical_data(symbol, start, end, timeframe)
                    if bars is not None and not bars.empty:
                        return float(bars['close'].iloc[-1])

                    # If we couldn't get historical data, try alternative methods
                    if hasattr(self, 'crypto_client') and self.crypto_client:
                        try:
                            crypto_bars = self.crypto_client.get_crypto_bars(symbol, timeframe, limit=1).df
                            if not crypto_bars.empty:
                                return float(crypto_bars['close'].iloc[-1])
                        except Exception as crypto_client_e:
                            logger.debug(f"Error using crypto_client for {symbol}: {str(crypto_client_e)}")
                except Exception as hist_e:
                    logger.error(f"AlpacaProvider: Error fetching crypto historical data for {symbol}: {str(hist_e)}")

                # Final fallback - try to get price from trade data if available
                try:
                    if hasattr(self.client, 'get_latest_crypto_trade'):
                        last_trade = self.client.get_latest_crypto_trade(symbol)
                        if last_trade and hasattr(last_trade, 'p'):
                            return float(last_trade.p)
                except Exception as trade_e:
                    logger.debug(f"Error getting latest crypto trade for {symbol}: {str(trade_e)}")

                # If we got here, we couldn't get a price
                logger.warning(f"Could not fetch latest price for crypto {symbol}, returning 0")
                return 0.0

            # For stocks, use the last trade
            last_trade = self.client.get_latest_trade(symbol)
            if last_trade and hasattr(last_trade, 'p'):
                return float(last_trade.p)  # price
            return 0.0

        except Exception as e:
            logger.error(f"AlpacaProvider: Error fetching latest price for {symbol}: {str(e)}")
            return 0.0

    async def get_market_symbols(self, market_type: str = "stock") -> List[str]:
        """
        Get a list of available market symbols.
        
        Args:
            market_type: Type of market ('stock', 'crypto', 'option', etc.)
            
        Returns:
            List of available symbols
        """
        if not self.client:
            logger.warning("AlpacaProvider: Client not initialized, cannot fetch market symbols")
            return []
        
        try:
            if market_type.lower() == "option":
                # Options data requires subscription level 3
                if not self.has_options_data:
                    logger.warning("AlpacaProvider: Options data requires Alpaca subscription level 3")
                    return []
                    
                # This would need to access Alpaca's options API
                # Implementation depends on the exact Alpaca SDK version
                logger.info("AlpacaProvider: Fetching available options symbols")
                try:
                    # This is an example - the actual implementation will depend on Alpaca's API
                    # For most recent Alpaca API versions
                    options = self.client.get_option_chain("SPY")
                    return [option.symbol for option in options]
                except AttributeError:
                    logger.warning("AlpacaProvider: Options API not available in this version of Alpaca SDK")
                    return []
                    
            elif market_type.lower() == "crypto":
                # Crypto data may require subscription level 2+
                logger.info("AlpacaProvider: Fetching available crypto symbols")
                assets = self.client.list_assets(asset_class='crypto')
                return [asset.symbol for asset in assets if asset.tradable]
                
            else:  # stocks and other standard assets
                logger.info("AlpacaProvider: Fetching available stock symbols")
                assets = self.client.list_assets(asset_class='us_equity')
                return [asset.symbol for asset in assets if asset.tradable]
                
        except Exception as e:
            logger.error(f"AlpacaProvider: Error fetching market symbols for {market_type}: {str(e)}")
            if "rate limit" in str(e).lower():
                logger.warning("AlpacaProvider: Rate limit reached. Consider upgrading your subscription level.")
            return []
