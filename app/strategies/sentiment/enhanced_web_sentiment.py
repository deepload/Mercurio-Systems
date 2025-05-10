"""
Enhanced Web Sentiment Analysis Agent for LLM Trading Strategies

This module extends the standard web sentiment agent to ensure real web data 
is used even in demo mode, providing richer market sentiment information.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

from app.utils.llm_utils import call_llm
from app.strategies.sentiment.web_sentiment_agent import LLMWebSentimentAgent

logger = logging.getLogger(__name__)

class EnhancedWebSentimentAgent(LLMWebSentimentAgent):
    """
    Enhanced version of LLMWebSentimentAgent that forces real web data collection
    even when running in demo mode (LLM_API_KEY=demo_mode).
    
    This class overrides the run_analysis and run_analysis_async methods to force
    real web data crawling regardless of demo mode settings.
    """
    
    def __init__(self, 
                 model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
                 use_local_model: bool = False,
                 local_model_path: Optional[str] = None,
                 api_key: Optional[str] = None,
                 cache_ttl_minutes: int = 60):
        """
        Initialize the enhanced web sentiment agent.
        
        Args:
            model_name: Name of the LLM to use for sentiment analysis
            use_local_model: Whether to use a local model
            local_model_path: Path to local model (if applicable)
            api_key: API key for remote model access
            cache_ttl_minutes: How long to cache sentiment data
        """
        super().__init__(
            model_name=model_name,
            use_local_model=use_local_model,
            local_model_path=local_model_path,
            api_key=api_key,
            cache_ttl_minutes=cache_ttl_minutes
        )
        logger.info("Initialized EnhancedWebSentimentAgent with forced real web data")
    
    def run_analysis(self, symbol: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Run sentiment analysis for a symbol with real web data.
        
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
        
        # Get LLM response - force using real LLM for sentiment analysis even in demo mode
        try:
            response = call_llm(self.model, prompt, force_real_llm=True)
            
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
        Run sentiment analysis for a symbol asynchronously with real web data.
        
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
        
        # Gather fresh data asynchronously - ALWAYS using real web data
        content = await self.gather_sentiment_inputs_async(symbol)
        prompt = self.build_prompt(content, symbol)
        
        # Get LLM response - force using real LLM regardless of demo mode
        try:
            response = call_llm(self.model, prompt, force_real_llm=True)
            
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
