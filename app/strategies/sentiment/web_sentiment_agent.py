"""
Web Sentiment Analysis Agent for LLM Trading Strategies

This module implements a web crawling and sentiment analysis agent
that extracts insights from financial websites and social media.
"""

import os
import json
import logging
import requests
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from app.utils.llm_utils import load_llm_model, call_llm

logger = logging.getLogger(__name__)

class LLMWebSentimentAgent:
    """
    Agent that crawls financial web and social sources (X, LinkedIn, Reddit, Coindesk),
    extracts sentiment-relevant data, and generates trading recommendations.
    
    The agent can operate in synchronous or async mode and supports caching
    to prevent excessive API calls.
    """

    def __init__(self, 
                 model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 use_local_model: bool = False,
                 local_model_path: Optional[str] = None,
                 api_key: Optional[str] = None,
                 cache_ttl_minutes: int = 60):
        """
        Initialize the web sentiment agent.
        
        Args:
            model_name: Name of the LLM to use for sentiment analysis
            use_local_model: Whether to use a local model
            local_model_path: Path to local model (if applicable)
            api_key: API key for remote model access
            cache_ttl_minutes: How long to cache sentiment data
        """
        self.model_name = model_name
        self.use_local_model = use_local_model
        self.local_model_path = local_model_path
        self.api_key = api_key or os.environ.get("LLM_API_KEY")
        
        # Initialize model
        self.model = load_llm_model(
            model_name=model_name,
            use_local=use_local_model,
            local_path=local_model_path,
            api_key=self.api_key
        )
        
        # Configure source URLs
        self.sources = {
            "x.com": {
                "base_url": "https://x.com/search?q=",
                "weight": 0.35
            },
            "reddit": {
                "base_url": "https://www.reddit.com/r/wallstreetbets/search/?q=",
                "weight": 0.25
            },
            "coindesk": {
                "base_url": "https://www.coindesk.com/search?q=",
                "weight": 0.2
            },
            "linkedin": {
                "base_url": "https://www.linkedin.com/search/results/content/?keywords=",
                "weight": 0.2
            }
        }
        
        # Initialize cache
        self.cache = {}
        self.cache_ttl = timedelta(minutes=cache_ttl_minutes)
        self.cache_timestamps = {}
        
        logger.info(f"Initialized LLMWebSentimentAgent with model {model_name}")

    def extract_text_from_url(self, url: str) -> str:
        """
        Extract text content from a URL.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Extracted text content
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Extract text from paragraphs, headings, and other text elements
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div'])
            text = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # Clean up the text
            text = ' '.join(text.split())
            
            # Limit size to avoid LLM context limits
            return text[:4000]
            
        except Exception as e:
            logger.error(f"Failed to extract text from {url}: {str(e)}")
            return f"Error extracting from {url}: {str(e)}"

    async def extract_text_from_url_async(self, url: str) -> str:
        """
        Asynchronously extract text content from a URL.
        
        Args:
            url: URL to extract content from
            
        Returns:
            Extracted text content
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status != 200:
                        return f"Error: HTTP {response.status}"
                    
                    html = await response.text()
                    
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Extract text from paragraphs, headings, and other text elements
            paragraphs = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'span', 'div'])
            text = ' '.join([p.get_text().strip() for p in paragraphs])
            
            # Clean up the text
            text = ' '.join(text.split())
            
            # Limit size to avoid LLM context limits
            return text[:4000]
            
        except Exception as e:
            logger.error(f"Failed to extract text from {url}: {str(e)}")
            return f"Error extracting from {url}: {str(e)}"

    def gather_sentiment_inputs(self, symbol: str) -> Dict[str, str]:
        """
        Gather text content from various sources for a symbol.
        
        Args:
            symbol: Symbol to gather sentiment for
            
        Returns:
            Dictionary mapping source names to text content
        """
        search_term = symbol.replace('USD', '') if 'USD' in symbol else symbol
        extracted = {}
        
        for name, source_info in self.sources.items():
            url = f"{source_info['base_url']}{search_term}"
            logger.info(f"Extracting sentiment data from {name} for {symbol}")
            extracted[name] = self.extract_text_from_url(url)
            
        return extracted

    async def gather_sentiment_inputs_async(self, symbol: str) -> Dict[str, str]:
        """
        Asynchronously gather text content from various sources for a symbol.
        
        Args:
            symbol: Symbol to gather sentiment for
            
        Returns:
            Dictionary mapping source names to text content
        """
        search_term = symbol.replace('USD', '') if 'USD' in symbol else symbol
        extracted = {}
        tasks = []
        
        # Create tasks for each source
        for name, source_info in self.sources.items():
            url = f"{source_info['base_url']}{search_term}"
            task = asyncio.create_task(self.extract_text_from_url_async(url))
            tasks.append((name, task))
        
        # Await all tasks
        for name, task in tasks:
            try:
                extracted[name] = await task
                logger.info(f"Extracted sentiment data from {name} for {symbol}")
            except Exception as e:
                logger.error(f"Error extracting from {name}: {str(e)}")
                extracted[name] = f"Error: {str(e)}"
        
        return extracted

    def build_prompt(self, content_by_source: Dict[str, str], symbol: str) -> str:
        """
        Build a prompt for the LLM to analyze sentiment.
        
        Args:
            content_by_source: Dictionary mapping source names to text content
            symbol: Symbol to analyze sentiment for
            
        Returns:
            Formatted prompt for LLM
        """
        prompt = f"""
You are a financial market sentiment analyst AI. You have crawled the following sources about {symbol}:

"""
        for src, text in content_by_source.items():
            prompt += f"\n[Source: {src}]\n{text[:500]}...\n"

        prompt += f"""

From this content, extract key financial sentiment insights for {symbol}.
Output a JSON block with this structure:
{{
  "symbol": "{symbol}",
  "action": "BUY | SELL | HOLD",
  "confidence": 0.0 - 1.0,
  "justification": "Short explanation combining insights from different sources",
  "sources": {{
    "x.com": "bullish | bearish | neutral",
    "reddit": "bullish | bearish | neutral",
    "coindesk": "bullish | bearish | neutral",
    "linkedin": "bullish | bearish | neutral"
  }},
  "stop_loss": float value as percentage (e.g., 0.05 for 5%),
  "take_profit": float value as percentage (e.g., 0.1 for 10%)
}}

Only respond with the JSON. Be sure to include the stop_loss and take_profit values.
"""
        return prompt

    def run_analysis(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Run sentiment analysis for a symbol.
        
        Args:
            symbol: Symbol to analyze
            use_cache: Whether to use cached results
            
        Returns:
            Analysis results as a dictionary
        """
        # Check cache first if enabled
        if use_cache and symbol in self.cache:
            cache_time = self.cache_timestamps.get(symbol)
            if cache_time and datetime.now() - cache_time < self.cache_ttl:
                logger.info(f"Using cached sentiment data for {symbol}")
                return self.cache[symbol]
        
        # Gather fresh data - ALWAYS using real web sentiment data regardless of demo mode
        content = self.gather_sentiment_inputs(symbol)
        prompt = self.build_prompt(content, symbol)
        
        # Get LLM response
        try:
            response = call_llm(self.model, prompt)
            
            # Extract JSON from response (in case LLM adds additional text)
            json_pattern = r'(\{.*\})'
            import re
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                try:
                    result = json.loads(json_str)
                    # Add timestamp
                    result['timestamp'] = datetime.now().isoformat()
                    
                    # Cache the result
                    if use_cache:
                        self.cache[symbol] = result
                        self.cache_timestamps[symbol] = datetime.now()
                        
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from LLM response: {json_str}")
            
            logger.error(f"LLM response didn't contain valid JSON: {response}")
            return self._generate_fallback_result(symbol)
            
        except Exception as e:
            logger.error(f"Error running sentiment analysis: {str(e)}")
            return self._generate_fallback_result(symbol)

    async def run_analysis_async(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Run sentiment analysis for a symbol asynchronously.
        
        Args:
            symbol: Symbol to analyze
            use_cache: Whether to use cached results
            
        Returns:
            Analysis results as a dictionary
        """
        # Check cache first if enabled
        if use_cache and symbol in self.cache:
            cache_time = self.cache_timestamps.get(symbol)
            if cache_time and datetime.now() - cache_time < self.cache_ttl:
                logger.info(f"Using cached sentiment data for {symbol}")
                return self.cache[symbol]
        
        # Gather fresh data asynchronously
        content = await self.gather_sentiment_inputs_async(symbol)
        prompt = self.build_prompt(content, symbol)
        
        # Get LLM response
        try:
            response = call_llm(self.model, prompt)
            
            # Extract JSON from response (in case LLM adds additional text)
            json_pattern = r'(\{.*\})'
            import re
            json_match = re.search(json_pattern, response, re.DOTALL)
            
            if json_match:
                json_str = json_match.group(1)
                try:
                    result = json.loads(json_str)
                    # Add timestamp
                    result['timestamp'] = datetime.now().isoformat()
                    
                    # Cache the result
                    if use_cache:
                        self.cache[symbol] = result
                        self.cache_timestamps[symbol] = datetime.now()
                        
                    return result
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse JSON from LLM response: {json_str}")
            
            logger.error(f"LLM response didn't contain valid JSON: {response}")
            return self._generate_fallback_result(symbol)
            
        except Exception as e:
            logger.error(f"Error running sentiment analysis: {str(e)}")
            return self._generate_fallback_result(symbol)

    def _generate_fallback_result(self, symbol: str) -> Dict[str, Any]:
        """
        Generate a fallback result when sentiment analysis fails.
        
        Args:
            symbol: Symbol to generate fallback for
            
        Returns:
            Fallback result
        """
        return {
            "symbol": symbol,
            "action": "HOLD",
            "confidence": 0.5,
            "justification": "Generated as fallback due to sentiment analysis failure.",
            "sources": {
                "x.com": "neutral",
                "reddit": "neutral",
                "coindesk": "neutral",
                "linkedin": "neutral"
            },
            "stop_loss": 0.05,
            "take_profit": 0.1,
            "timestamp": datetime.now().isoformat(),
            "is_fallback": True
        }

    def clear_cache(self, symbol: Optional[str] = None):
        """
        Clear the sentiment cache.
        
        Args:
            symbol: Symbol to clear cache for, or None to clear all
        """
        if symbol:
            if symbol in self.cache:
                del self.cache[symbol]
                if symbol in self.cache_timestamps:
                    del self.cache_timestamps[symbol]
                logger.info(f"Cleared cache for {symbol}")
        else:
            self.cache.clear()
            self.cache_timestamps.clear()
            logger.info("Cleared entire sentiment cache")


if __name__ == "__main__":
    # Simple test code
    logging.basicConfig(level=logging.INFO)
    agent = LLMWebSentimentAgent()
    
    symbol = "BTCUSD"
    result = agent.run_analysis(symbol)
    print(json.dumps(result, indent=2))
