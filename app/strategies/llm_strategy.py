"""
MercurioAI LLM-Based Strategy

This module implements a trading strategy that uses Large Language Models (LLMs)
like Llama, GPT, or other AI models to analyze market sentiment, news, and patterns
for generating trading signals.
"""
import os
import logging
import json
from typing import Dict, List, Any, Tuple, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import asyncio
import aiohttp
import re
from abc import abstractmethod
from pathlib import Path
import requests
import time

from app.strategies.base import BaseStrategy
from app.db.models import TradeAction
from app.core.event_bus import EventBus, EventType

logger = logging.getLogger(__name__)

class LLMStrategy(BaseStrategy):
    """
    Trading strategy based on Large Language Models (LLMs).
    
    This strategy leverages natural language processing and reasoning capabilities
    of modern LLMs to analyze market data, news, and sentiment to generate
    trading signals.
    """
    
    def __init__(self, 
                 model_name: str = "llama2-7b",
                 use_local_model: bool = False,
                 local_model_path: Optional[str] = None,
                 api_key: Optional[str] = None,
                 endpoint_url: Optional[str] = None,
                 sentiment_threshold: float = 0.6,
                 news_lookback_hours: int = 24,
                 technical_indicators: List[str] = None,
                 max_tokens: int = 256,
                 temperature: float = 0.1,
                 cache_ttl: int = 3600,
                 **kwargs):
        """
        Initialize the LLM strategy
        
        Args:
            model_name: Name of the LLM to use
            use_local_model: Whether to use a locally deployed model
            local_model_path: Path to local model (if use_local_model is True)
            api_key: API key for remote LLM service
            endpoint_url: API endpoint for remote LLM service
            sentiment_threshold: Threshold for sentiment score to generate signals
            news_lookback_hours: Hours of news data to analyze
            technical_indicators: List of technical indicators to include
            max_tokens: Maximum tokens for LLM generation
            temperature: Temperature for LLM generation (randomness)
            cache_ttl: Time-to-live for cached responses in seconds
        """
        super().__init__(**kwargs)
        
        self.model_name = model_name
        self.use_local_model = use_local_model
        self.local_model_path = local_model_path
        self.api_key = api_key
        self.endpoint_url = endpoint_url
        self.sentiment_threshold = sentiment_threshold
        self.news_lookback_hours = news_lookback_hours
        self.technical_indicators = technical_indicators or [
            'sma_20', 'sma_50', 'rsi_14', 'macd', 'adx_14'
        ]
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Create cache for LLM responses
        self.cache = {}
        self.cache_ttl = cache_ttl
        
        # Event bus for receiving news updates
        self.event_bus = EventBus()
        
        # Create model directory
        self.model_dir = Path('./models/llm')
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up news data store
        self.news_data = []
        self.last_news_update = None
        
        # Initialize LLM client
        self._initialize_llm_client()
        
        # Register event handlers
        self._register_events()
        
        logger.info(f"Initialized LLMStrategy with {model_name}")
        
    def _initialize_llm_client(self):
        """Initialize the LLM client based on configuration"""
        if self.use_local_model:
            self._initialize_local_model()
        else:
            self._initialize_remote_client()
            
    def _initialize_local_model(self):
        """Initialize a local LLM model if available"""
        try:
            # Try to import necessary libraries
            import torch
            
            # Check for specific model types and initialize appropriately
            if 'llama' in self.model_name.lower():
                try:
                    from llama_cpp import Llama
                    
                    model_path = self.local_model_path or f"./models/llm/{self.model_name}.gguf"
                    if os.path.exists(model_path):
                        logger.info(f"Loading local Llama model from {model_path}")
                        self.llm = Llama(
                            model_path=model_path,
                            n_ctx=2048,
                            n_threads=4
                        )
                        logger.info("Local Llama model loaded successfully")
                    else:
                        logger.warning(f"Local model not found at {model_path}. Will use fallback methods.")
                        self.llm = None
                except ImportError:
                    logger.warning("llama-cpp-python not installed. Using fallback methods.")
                    self.llm = None
            
            # Add support for other local models as needed
            elif 'gpt' in self.model_name.lower():
                logger.warning("Local GPT model not supported yet. Using fallback methods.")
                self.llm = None
            else:
                logger.warning(f"Unsupported local model type: {self.model_name}. Using fallback methods.")
                self.llm = None
                
        except ImportError:
            logger.warning("Required libraries for local LLM not installed. Using fallback methods.")
            self.llm = None
            
    def _initialize_remote_client(self):
        """Initialize client for remote LLM API"""
        # No specific initialization needed for requests/aiohttp
        # Will use self.api_key and self.endpoint_url when making requests
        if not self.api_key and 'OPENAI_API_KEY' in os.environ:
            self.api_key = os.environ['OPENAI_API_KEY']
            
        if not self.endpoint_url:
            # Default to OpenAI's API if using 'gpt' model
            if 'gpt' in self.model_name.lower():
                self.endpoint_url = "https://api.openai.com/v1/chat/completions"
            else:
                logger.warning(f"No endpoint URL specified for model {self.model_name}")
                
    def _register_events(self):
        """Register for relevant events on the event bus"""
        # Note: NEWS_UPDATE event type is not defined in the current version of EventType
        # Using an existing event type that could be useful for news-like updates
        try:
            # Market data updates could contain relevant information for the LLM
            asyncio.create_task(self.event_bus.subscribe(
                EventType.MARKET_DATA_UPDATED,
                self._handle_news_update
            ))
            logger.info("LLM strategy registered for market data events")
        except Exception as e:
            logger.warning(f"Could not register for events: {e}")
            logger.info("Event subscription skipped - LLM will use polling instead")
        
    async def _handle_news_update(self, data: Dict[str, Any]):
        """Handle news update events"""
        if 'news_items' in data:
            self.news_data.extend(data['news_items'])
            self.last_news_update = datetime.now()
            
            # Keep only recent news
            self._prune_old_news()
            
    def _prune_old_news(self):
        """Remove old news from the news data store"""
        if not self.news_data:
            return
            
        now = datetime.now()
        cutoff = now - timedelta(hours=self.news_lookback_hours)
        
        self.news_data = [
            item for item in self.news_data
            if 'timestamp' in item and item['timestamp'] >= cutoff
        ]
        
    async def query_llm(self, prompt: str, use_cache: bool = True) -> str:
        """
        Query the LLM with a prompt
        
        Args:
            prompt: The prompt to send to the LLM
            use_cache: Whether to use cached responses
            
        Returns:
            LLM response text
        """
        # Check cache first if enabled
        if use_cache:
            cache_key = prompt
            if cache_key in self.cache:
                entry = self.cache[cache_key]
                if entry['timestamp'] + self.cache_ttl > time.time():
                    return entry['response']
        
        # Local model query
        if self.use_local_model and self.llm is not None:
            response = await self._query_local_model(prompt)
        # Remote API query
        elif self.api_key and self.endpoint_url:
            response = await self._query_remote_api(prompt)
        # Fallback method
        else:
            response = await self._query_fallback(prompt)
            
        # Cache the response if enabled
        if use_cache:
            self.cache[cache_key] = {
                'response': response,
                'timestamp': time.time()
            }
            
        return response
        
    async def _query_local_model(self, prompt: str) -> str:
        """Query a local LLM model"""
        try:
            result = self.llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop=["</s>", "Human:", "User:"],
                echo=False
            )
            
            # Extract text from result based on model type
            if isinstance(result, dict) and 'choices' in result:
                return result['choices'][0]['text'].strip()
            elif isinstance(result, dict) and 'generation' in result:
                return result['generation'].strip()
            elif isinstance(result, str):
                return result.strip()
            else:
                logger.warning(f"Unexpected response format from local model: {type(result)}")
                return ""
        except Exception as e:
            logger.error(f"Error querying local model: {e}")
            return await self._query_fallback(prompt)
            
    async def _query_remote_api(self, prompt: str) -> str:
        """Query a remote LLM API"""
        try:
            headers = {
                "Content-Type": "application/json"
            }
            
            if self.api_key:
                if 'openai' in self.endpoint_url:
                    headers["Authorization"] = f"Bearer {self.api_key}"
                else:
                    headers["x-api-key"] = self.api_key
            
            if 'openai' in self.endpoint_url or 'gpt' in self.model_name.lower():
                # Format for OpenAI API
                payload = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "You are an AI trading assistant analyzing financial data and news."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
            else:
                # Generic API format
                payload = {
                    "prompt": prompt,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
                
            async with aiohttp.ClientSession() as session:
                async with session.post(self.endpoint_url, headers=headers, json=payload) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"API error ({response.status}): {error_text}")
                        return await self._query_fallback(prompt)
                        
                    result = await response.json()
                    
                    # Extract text based on API format
                    if 'choices' in result and len(result['choices']) > 0:
                        if 'message' in result['choices'][0]:
                            return result['choices'][0]['message']['content'].strip()
                        elif 'text' in result['choices'][0]:
                            return result['choices'][0]['text'].strip()
                    
                    logger.warning(f"Unexpected response format: {result}")
                    return ""
                    
        except Exception as e:
            logger.error(f"Error querying remote API: {e}")
            return await self._query_fallback(prompt)
            
    async def _query_fallback(self, prompt: str) -> str:
        """Fallback method when LLM is not available"""
        logger.info("Using fallback logic for LLM query")
        
        # Simple keyword-based analysis
        lower_prompt = prompt.lower()
        
        if "bullish" in lower_prompt or "positive" in lower_prompt:
            return "Based on the analysis, the market sentiment appears bullish. Technical indicators are showing potential upward momentum. Consider a buy signal with confidence level of 0.7."
            
        elif "bearish" in lower_prompt or "negative" in lower_prompt:
            return "Based on the analysis, the market sentiment appears bearish. Technical indicators are showing potential downward momentum. Consider a sell signal with confidence level of 0.6."
            
        elif "trend" in lower_prompt:
            return "The current market is showing a neutral trend with mixed signals. Technical indicators are not giving clear direction. Consider holding current positions with a neutral outlook."
            
        else:
            return "Analysis is inconclusive. The data shows mixed signals with no clear directional bias. Recommend maintaining current positions and monitoring for clearer signals."
            
    async def load_data(self, symbol: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Load historical data for the strategy
        
        Args:
            symbol: Trading symbol
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with historical data
        """
        # Add extra days to account for technical indicator calculation
        extended_start = start_date - timedelta(days=50)
        
        # Load data using the market data service
        from app.services.market_data import MarketDataService
        market_data = MarketDataService()
        
        # Format dates to strings
        start_str = extended_start.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        data = await market_data.get_historical_data(symbol, start_str, end_str)
        
        return data
        
    async def load_news_data(self, symbol: str, lookback_hours: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Load news data for a symbol
        
        Args:
            symbol: Trading symbol
            lookback_hours: Hours to look back for news
            
        Returns:
            List of news items
        """
        hours = lookback_hours or self.news_lookback_hours
        
        # Check if we already have recent news data
        if self.news_data and self.last_news_update:
            if datetime.now() - self.last_news_update < timedelta(hours=1):
                # Filter for the specific symbol
                return [item for item in self.news_data if symbol.lower() in item.get('symbols', [])]
        
        # Fetch news data if needed
        try:
            # Try to use a news provider if available
            try:
                from app.services.news import NewsService
                news_service = NewsService()
                news_data = await news_service.get_news(symbol, hours=hours)
                return news_data
            except ImportError:
                # Fall back to sample news data
                return self._generate_sample_news(symbol, hours)
        except Exception as e:
            logger.error(f"Error loading news data: {e}")
            return []
            
    def _generate_sample_news(self, symbol: str, hours: int) -> List[Dict[str, Any]]:
        """Generate sample news data for testing"""
        base_asset = symbol.split('/')[0] if '/' in symbol else symbol
        
        current_time = datetime.now()
        news_items = []
        
        # Generic sample news
        headlines = [
            f"{base_asset} Rallies as Market Sentiment Improves",
            f"{base_asset} Sees Technical Breakout, Analysts Predict Further Gains",
            f"Market Volatility Affects {base_asset} Trading Volume",
            f"New Developments Could Impact {base_asset} Price Action",
            f"{base_asset} Trading at Key Support Levels, Technical Analysis"
        ]
        
        for i, headline in enumerate(headlines):
            timestamp = current_time - timedelta(hours=i*3)
            if timestamp >= current_time - timedelta(hours=hours):
                news_items.append({
                    'id': f"sample-{i}",
                    'headline': headline,
                    'summary': f"Sample news summary for {headline}. This is generated for testing purposes.",
                    'source': 'Sample News Provider',
                    'url': f"https://example.com/news/{i}",
                    'timestamp': timestamp,
                    'sentiment': (0.6 - (i * 0.1)) if i < 3 else (-0.2 - (i-3) * 0.1),
                    'symbols': [symbol]
                })
                
        return news_items
        
    async def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess data for analysis
        
        Args:
            data: Raw market data
            
        Returns:
            Preprocessed data
        """
        if data is None or len(data) == 0:
            logger.error("No data to preprocess")
            return None
            
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Make sure timestamp is a datetime
        if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Add technical indicators if not already present
        if 'sma_20' not in df.columns:
            logger.info("Adding technical indicators")
            df = await self._add_technical_indicators(df)
        
        return df
        
    async def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to the data without relying on TA-Lib
        
        Args:
            data: Price data
            
        Returns:
            Data with technical indicators
        """
        df = data.copy()
        
        # Simple Moving Averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        
        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macdsignal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macdhist'] = df['macd'] - df['macdsignal']
        
        # Bollinger Bands
        df['bb_middle'] = df['sma_20']
        rolling_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (rolling_std * 2)
        df['bb_lower'] = df['bb_middle'] - (rolling_std * 2)
        
        return df
        
    async def create_market_analysis_prompt(self, data: pd.DataFrame, news_data: List[Dict[str, Any]], symbol: str) -> str:
        """
        Create a prompt for the LLM to analyze market data
        
        Args:
            data: Preprocessed market data
            news_data: News data for the symbol
            symbol: Trading symbol
            
        Returns:
            Prompt for LLM analysis
        """
        # Get the latest data point and previous ones for comparison
        latest = data.iloc[-1]
        prev_1d = data.iloc[-2] if len(data) > 1 else None
        prev_1w = data.iloc[-5] if len(data) > 5 else None
        
        # Format market data
        # Calculate price changes with proper conditional handling
        price_change_24h = f"{((latest['close'] / prev_1d['close']) - 1) * 100:.2f}%" if prev_1d is not None else "N/A"
        price_change_7d = f"{((latest['close'] / prev_1w['close']) - 1) * 100:.2f}%" if prev_1w is not None else "N/A"
        
        market_data_section = f"""
MARKET DATA FOR {symbol} as of {latest.get('timestamp', 'latest')}:
- Current Price: {latest['close']}
- 24h Change: {price_change_24h}
- 7d Change: {price_change_7d}
- Volume: {latest['volume']}

TECHNICAL INDICATORS:
- SMA 20: {latest.get('sma_20', 'N/A')}
- SMA 50: {latest.get('sma_50', 'N/A')}
- RSI 14: {latest.get('rsi_14', 'N/A')}
- MACD: {latest.get('macd', 'N/A')}
- MACD Signal: {latest.get('macdsignal', 'N/A')}
- MACD Histogram: {latest.get('macdhist', 'N/A')}
"""
        
        # Format news data
        news_section = "RECENT NEWS:\n"
        if news_data:
            for i, news in enumerate(news_data[:5]):  # Limit to 5 most recent news items
                news_section += f"""
{i+1}. {news.get('headline', 'No headline')}
   Source: {news.get('source', 'Unknown')}
   Date: {news.get('timestamp', 'Unknown')}
   Summary: {news.get('summary', 'No summary available')}
"""
        else:
            news_section += "No recent news available for this asset.\n"
        
        # Create the full prompt
        prompt = f"""
Analyze the following market data and news for {symbol} and provide a trading recommendation.

{market_data_section}

{news_section}

Based on the above information, please provide:
1. A brief analysis of the current market situation for {symbol}
2. Your assessment of market sentiment (bullish, bearish, or neutral)
3. A specific trading recommendation (BUY, SELL, or HOLD)
4. A confidence level for your recommendation (0.0 to 1.0)
5. Key reasons supporting your recommendation

Format your response as a concise analysis with clear sections.
"""
        return prompt
        
    async def extract_trading_signal(self, llm_response: str) -> Tuple[TradeAction, float]:
        """
        Extract trading signal from LLM response
        
        Args:
            llm_response: Response from the LLM
            
        Returns:
            Trading action and confidence
        """
        # Default values
        action = TradeAction.HOLD
        confidence = 0.5
        
        # Extract recommendation (case insensitive)
        buy_pattern = r'(buy|long|bullish|purchase|acquire)'
        sell_pattern = r'(sell|short|bearish|exit|reduce)'
        hold_pattern = r'(hold|neutral|wait|maintain|stay)'
        
        # Look for explicit recommendation
        if re.search(buy_pattern, llm_response, re.IGNORECASE):
            action = TradeAction.BUY
        elif re.search(sell_pattern, llm_response, re.IGNORECASE):
            action = TradeAction.SELL
        elif re.search(hold_pattern, llm_response, re.IGNORECASE):
            action = TradeAction.HOLD
            
        # Look for confidence level
        confidence_pattern = r'confidence.*?(0\.\d+|\d+%|0\.\d+/1\.0|\d+/100)'
        confidence_match = re.search(confidence_pattern, llm_response, re.IGNORECASE)
        
        if confidence_match:
            confidence_str = confidence_match.group(1)
            try:
                if '%' in confidence_str:
                    # Convert percentage to decimal
                    confidence = float(confidence_str.strip('%')) / 100
                elif '/' in confidence_str:
                    # Handle fractions like 0.7/1.0 or 70/100
                    num, denom = confidence_str.split('/')
                    confidence = float(num) / float(denom)
                else:
                    confidence = float(confidence_str)
                    
                # Ensure confidence is between 0 and 1
                confidence = min(max(confidence, 0.0), 1.0)
            except (ValueError, ZeroDivisionError):
                logger.warning(f"Could not parse confidence value: {confidence_str}")
                
        return action, confidence
        
    async def predict(self, data: pd.DataFrame) -> Tuple[TradeAction, float]:
        """
        Generate trading signals
        
        Args:
            data: Market data
            
        Returns:
            Trading action and confidence
        """
        # Make sure we have data
        if data is None or len(data) < 20:  # Need at least 20 points for indicators
            logger.warning("Not enough data for prediction")
            return TradeAction.HOLD, 0.5
            
        # Get symbol from data if available
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        # Preprocess data if necessary
        if 'sma_20' not in data.columns:
            data = await self.preprocess_data(data)
            
        # Load news data
        news_data = await self.load_news_data(symbol)
        
        # Create analysis prompt
        prompt = await self.create_market_analysis_prompt(data, news_data, symbol)
        
        # Query LLM
        llm_response = await self.query_llm(prompt)
        logger.debug(f"LLM Response: {llm_response}")
        
        # Extract trading signal
        action, confidence = await self.extract_trading_signal(llm_response)
        
        logger.info(f"LLM prediction for {symbol}: {action.name} with confidence {confidence:.4f}")
        
        return action, confidence
        
    async def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train the strategy model on historical data.
        
        LLM strategies don't require traditional training like ML models,
        but we implement this method to satisfy the BaseStrategy interface.
        
        Args:
            data: Preprocessed market data
            
        Returns:
            Dictionary containing training metrics (mostly placeholders for LLM)
        """
        logger.info(f"LLM strategy doesn't require traditional training")
        
        # For LLM strategies, we don't actually train in the traditional sense,
        # but we can use this to initialize or validate the LLM setup
        
        # Check if the LLM client is properly initialized
        is_ready = False
        try:
            # Test the LLM with a simple prompt
            test_prompt = "You are a trading assistant. Reply with 'READY' if you can help analyze market data."
            response = await self.query_llm(test_prompt, use_cache=False)
            
            # Check if we got any response
            if response and len(response) > 0:
                is_ready = True
                logger.info(f"LLM is ready for inference: {self.model_name}")
            else:
                logger.warning(f"LLM test returned empty response")
                
        except Exception as e:
            logger.error(f"Error testing LLM: {e}")
            
        # Mark as trained regardless of the test outcome to allow the strategy to run
        self.is_trained = True
        
        return {
            "success": is_ready,
            "model": self.model_name,
            "message": "LLM strategy is ready" if is_ready else "LLM may not be available, will use fallbacks"
        }
        
    async def backtest(self, data: pd.DataFrame, initial_capital: float = 10000.0) -> Dict[str, Any]:
        """
        Backtest the strategy
        
        Args:
            data: Market data
            initial_capital: Initial capital
            
        Returns:
            Backtest results
        """
        # Preprocess data if necessary
        if 'sma_20' not in data.columns:
            data = await self.preprocess_data(data)
            
        # Initialize backtest variables
        capital = initial_capital
        position = 0
        equity_curve = []
        trades = []
        
        # Get symbol from data if available
        symbol = data['symbol'].iloc[0] if 'symbol' in data.columns else "UNKNOWN"
        
        # Split data into chunks for analysis (to simulate periodic analysis)
        chunk_size = 20  # Analyze every 20 data points
        
        for i in range(chunk_size, len(data), chunk_size):
            # Get data chunk
            chunk_end = min(i + chunk_size, len(data))
            analysis_data = data.iloc[:chunk_end]
            
            # Current data point for trading
            current_price = analysis_data['close'].iloc[-1]
            current_date = analysis_data['timestamp'].iloc[-1] if 'timestamp' in analysis_data.columns else i
            
            # Generate news data for this period
            # In a real backtest, we would need historical news data
            news_data = self._generate_sample_news(symbol, 24)
            
            # Create analysis prompt
            prompt = await self.create_market_analysis_prompt(analysis_data, news_data, symbol)
            
            # Query LLM
            llm_response = await self.query_llm(prompt, use_cache=False)  # Don't cache for backtest
            
            # Extract trading signal
            action, confidence = await self.extract_trading_signal(llm_response)
            
            # Calculate position and equity
            previous_position = position
            
            # Update position based on action
            if action == TradeAction.BUY and position <= 0:
                # Close any short position
                if position < 0:
                    capital += position * current_price * -1  # Close short
                    trades.append({
                        'timestamp': current_date,
                        'action': 'CLOSE_SHORT',
                        'price': current_price,
                        'quantity': position * -1,
                        'capital': capital
                    })
                    position = 0
                
                # Open long position - invest 95% of capital
                quantity = (capital * 0.95) / current_price
                capital -= quantity * current_price
                position += quantity
                trades.append({
                    'timestamp': current_date,
                    'action': 'BUY',
                    'price': current_price,
                    'quantity': quantity,
                    'capital': capital
                })
                
            elif action == TradeAction.SELL and position >= 0:
                # Close any long position
                if position > 0:
                    capital += position * current_price
                    trades.append({
                        'timestamp': current_date,
                        'action': 'CLOSE_LONG',
                        'price': current_price,
                        'quantity': position,
                        'capital': capital
                    })
                    position = 0
                
                # Open short position - 95% of capital
                quantity = (capital * 0.95) / current_price
                capital += quantity * current_price  # Short sell proceeds
                position -= quantity
                trades.append({
                    'timestamp': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'quantity': quantity,
                    'capital': capital
                })
            
            # Calculate equity (capital + position value)
            equity = capital + (position * current_price)
            
            equity_curve.append({
                'timestamp': current_date,
                'price': current_price,
                'action': action.name,
                'confidence': confidence,
                'position': position,
                'capital': capital,
                'equity': equity
            })
        
        # Calculate final equity
        final_equity = capital
        if position != 0:
            final_price = data['close'].iloc[-1]
            final_equity += position * final_price
            
        # Calculate performance metrics
        total_return = (final_equity / initial_capital) - 1
        equity_df = pd.DataFrame(equity_curve)
        
        # Prepare results
        results = {
            'initial_capital': initial_capital,
            'final_equity': final_equity,
            'total_return': total_return,
            'equity_curve': equity_df,
            'trades': trades,
            'position': position,
            'strategy': 'LLMStrategy'
        }
        
        return results
