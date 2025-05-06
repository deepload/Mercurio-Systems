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
        
        # Vérifier le niveau d'abonnement Alpaca (1=Basic, 2=Pro, 3=AlgoTrader Plus)
        self.subscription_level = int(os.getenv("ALPACA_SUBSCRIPTION_LEVEL", "1"))
        logger.info(f"Alpaca subscription level: {self.subscription_level}")
        
        # Initialize Alpaca client with support for both paper and live trading
        # Déterminer le mode Alpaca (paper ou live)
        alpaca_mode = os.getenv("ALPACA_MODE", "paper").lower()
        
        # Configuration selon le mode
        if alpaca_mode == "live":
            self.alpaca_key = os.getenv("ALPACA_LIVE_KEY")
            self.alpaca_secret = os.getenv("ALPACA_LIVE_SECRET")
            base_url = os.getenv("ALPACA_LIVE_URL", "https://api.alpaca.markets")
            logger.info("Alpaca configured for LIVE trading mode")
        else:  # paper mode par défaut
            self.alpaca_key = os.getenv("ALPACA_PAPER_KEY")
            self.alpaca_secret = os.getenv("ALPACA_PAPER_SECRET")
            base_url = os.getenv("ALPACA_PAPER_URL", "https://paper-api.alpaca.markets")
            logger.info("Alpaca configured for PAPER trading mode")
            
        # Initialiser le client si les clés sont disponibles
        self.alpaca_client = None
        if self.alpaca_key and self.alpaca_secret:
            try:
                # Supprimer /v2 de l'URL si présent
                if base_url.endswith("/v2"):
                    base_url = base_url.rstrip("/v2")
                    logger.info(f"Removed '/v2' from Alpaca base URL: {base_url}")
                
                # URL des données de marché (identique pour paper et live)
                data_url = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
                # Commentons cette ligne pour éviter l'affichage qui peut créer des confusions
                # logger.info(f"Using Alpaca base_url: {base_url} and data_url: {data_url}")
                
                # Initialisation sans 'data_url' (compatible avec toutes les versions)
                # La version récente de l'API Alpaca a changé la façon dont le client est initialisé
                # Optons pour la méthode la plus compatible
                self.alpaca_client = tradeapi.REST(
                    key_id=self.alpaca_key,
                    secret_key=self.alpaca_secret,
                    base_url=base_url
                )
                logger.info(f"Initialized Alpaca client with base_url: {base_url}")
                
                # Pour les versions qui supportent data_url comme attribut
                # Note: Cela n'affectera pas les versions qui ne le supportent pas
                try:
                    if hasattr(self.alpaca_client, 'data_url'):
                        self.alpaca_client.data_url = data_url
                        logger.info(f"Set data_url attribute to {data_url}")
                except Exception as e:
                    logger.warning(f"Could not set data_url attribute: {e}")
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
                is_crypto = "-USD" in symbol
                
                if is_crypto:
                    # Format correct pour l'API crypto d'Alpaca: BTC/USD (avec slash)
                    alpaca_symbol = symbol.replace("-USD", "/USD")
                    logging.info(f"Converting crypto symbol format for Alpaca: {symbol} -> {alpaca_symbol}")
                
                # Utiliser l'endpoint approprié selon qu'il s'agit d'une crypto ou d'une action
                # Pour les cryptos, utiliser l'API spécifique d'Alpaca selon la documentation officielle
                if is_crypto:
                    logger.info(f"Cryptocurrency detected: {symbol}. Using dedicated Alpaca crypto API.")
                    
                    # Utiliser exclusivement l'API crypto v1beta3 pour les cryptomonnaies
                    # https://docs.alpaca.markets/docs/crypto-trading
                    try:
                        data = self._get_crypto_data_v1beta3(alpaca_symbol, start_str, end_str, alpaca_timeframe)
                        
                        if data is None or data.empty:
                            logger.error(f"Failed to get crypto data for {symbol} through the v1beta3 crypto API.")
                            raise ValueError(f"Failed to get crypto data for {symbol} through the v1beta3 crypto API.")
                    except Exception as e:
                        logger.error(f"Direct crypto API call failed for {symbol}: {str(e)[:200]}")
                        raise ValueError(f"Failed to get crypto data for {symbol}. Error: {str(e)[:200]}")
                else:
                    # Pour les actions, utiliser l'API stock standard (plus simple)
                    logger.info(f"Using standard stock API for {symbol}")
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
        
        # Si on a un abonnement premium, ne pas utiliser les données de repli
        if self.subscription_level >= 3:
            logger.error(f"Failed to get data for {symbol} despite premium subscription level {self.subscription_level}. Check API access and symbol validity.")
            return pd.DataFrame()  # Renvoie un DataFrame vide au lieu de données de repli
        else:
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
            
        # Legacy fallback for backward compatibility with Alpaca
        if self.alpaca_client:
            try:
                # Détecter si c'est une crypto
                is_crypto = "-USD" in symbol
                alpaca_symbol = symbol
                
                if is_crypto:
                    # Utiliser l'API dédiée pour les cryptos
                    # Format correct avec slash : BTC/USD
                    alpaca_symbol = symbol.replace("-USD", "/USD")
                    logger.info(f"Converting crypto symbol for Alpaca: {symbol} -> {alpaca_symbol}")
                    
                    # Essayer différentes méthodes pour obtenir le prix des cryptos
                    logger.info(f"Trying dedicated crypto price API for {alpaca_symbol}")
                    
                    # 1. Méthode directe via l'API v1beta3 (plus fiable)
                    try:
                        import requests
                        # Endpoint pour prix réel-time de crypto
                        url = "https://data.alpaca.markets/v1beta3/crypto/us/latest/quotes"
                        params = {"symbols": alpaca_symbol}
                        headers = {
                            "APCA-API-KEY-ID": self.alpaca_key,
                            "APCA-API-SECRET-KEY": self.alpaca_secret
                        }
                        
                        logger.info(f"Making API request to: {url} for {alpaca_symbol}")
                        response = requests.get(url, params=params, headers=headers)
                        if response.status_code == 200:
                            data = response.json()
                            if "quotes" in data and alpaca_symbol in data["quotes"]:
                                # Utiliser le prix moyen entre bid/ask
                                quote = data["quotes"][alpaca_symbol]
                                if "ap" in quote and "bp" in quote:
                                    price = (quote["ap"] + quote["bp"]) / 2
                                    logger.info(f"Got crypto price ${price:.2f} for {alpaca_symbol} via v1beta3 API")
                                    return price
                    except Exception as e:
                        logger.warning(f"Direct crypto API failed: {str(e)[:100]}")
                    
                    # 2. Méthode historique (dernier prix de la journée)
                    try:
                        end_date = datetime.now()
                        start_date = end_date - timedelta(minutes=15)
                        # Utiliser la méthode _get_crypto_data_v1beta3 déjà implémentée
                        df = self._get_crypto_data_v1beta3(
                            alpaca_symbol,
                            start_date.isoformat() + "Z",
                            end_date.isoformat() + "Z",
                            "1Min"
                        )
                        if not df.empty:
                            price = df['close'].iloc[-1]
                            logger.info(f"Got crypto price ${price:.2f} for {alpaca_symbol} via historical data")
                            return price
                    except Exception as e:
                        logger.warning(f"Historical crypto data failed: {str(e)[:100]}")
                else:
                    # Pour les actions, utiliser la méthode standard
                    logger.info(f"Falling back to legacy Alpaca client for {symbol} stock price")
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
    
    # Méthode pour appeler directement l'API crypto d'Alpaca selon la documentation officielle
    def _get_crypto_data_v1beta3(self, symbol: str, start_str: str, end_str: str, timeframe: str) -> pd.DataFrame:
        """
        Appelle directement l'API Crypto d'Alpaca en v1beta3 selon la documentation officielle
        https://docs.alpaca.markets/docs/crypto-trading
        
        Args:
            symbol: Le symbole crypto au format BTC/USD
            start_str: Date de début au format string
            end_str: Date de fin au format string
            timeframe: Intervalle de temps (1Day, 1Hour, etc.)
            
        Returns:
            DataFrame avec les données crypto ou None en cas d'échec
        """
        import requests
        
        try:
            # Format de l'URL selon la documentation
            # https://data.alpaca.markets/v1beta3/crypto/us/bars
            base_url = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
            
            # Adapter le timeframe au format de l'API v1beta3
            v1beta3_timeframe = timeframe
            if timeframe == "1Day":
                v1beta3_timeframe = "1D"
            elif timeframe == "1Hour":
                v1beta3_timeframe = "1H"
            
            # Paramètres de la requête
            params = {
                "symbols": symbol,
                "timeframe": v1beta3_timeframe,
                "start": start_str,
                "end": end_str,
                "limit": 1000
            }
            
            # En-têtes avec authentification
            headers = {
                "APCA-API-KEY-ID": self.alpaca_key,
                "APCA-API-SECRET-KEY": self.alpaca_secret
            }
            
            # Exécuter la requête
            logger.info(f"Making direct API call to Alpaca crypto endpoint for {symbol}")
            response = requests.get(base_url, params=params, headers=headers)
            
            # Vérifier le statut de la réponse
            if response.status_code == 200:
                data = response.json()
                
                # Vérifier que nous avons des données pour ce symbole
                if data and "bars" in data and symbol in data["bars"] and len(data["bars"][symbol]) > 0:
                    # Convertir les données en DataFrame
                    bars = data["bars"][symbol]
                    df = pd.DataFrame(bars)
                    
                    # Renommer et formater les colonnes pour correspondre au format attendu
                    df.rename(columns={
                        "t": "timestamp",
                        "o": "open",
                        "h": "high",
                        "l": "low",
                        "c": "close",
                        "v": "volume"
                    }, inplace=True)
                    
                    # Convertir la colonne timestamp en datetime
                    df["timestamp"] = pd.to_datetime(df["timestamp"])
                    df.set_index("timestamp", inplace=True)
                    
                    logger.info(f"Successfully retrieved {len(df)} bars for {symbol} from Alpaca v1beta3 API")
                    return df
                else:
                    logger.warning(f"No data returned for {symbol} from Alpaca v1beta3 API")
                    return pd.DataFrame()
            else:
                error_msg = f"API error: {response.status_code} {response.text[:100]}"
                logger.warning(error_msg)
                # Vérifier spécifiquement les erreurs d'autorisation
                if response.status_code == 403:
                    logger.warning("Received 403 Forbidden. Your Alpaca plan likely does not include crypto data access.")
                return pd.DataFrame()
                
        except Exception as e:
            logger.warning(f"Error in direct API call to Alpaca: {str(e)[:200]}")
            return pd.DataFrame()
    
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
