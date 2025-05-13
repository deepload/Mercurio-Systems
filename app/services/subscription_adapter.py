#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Adapter service that enforces subscription tier limits for external API services.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Union

from app.db.models import User
from app.services.subscription_service import SubscriptionService
from app.utils.subscription_config import SubscriptionTier

logger = logging.getLogger(__name__)


class SubscriptionServiceAdapter:
    """
    Adapter that enforces subscription tier limitations when accessing external services.
    """
    
    def __init__(self, subscription_service: SubscriptionService, user_id: int):
        """
        Initialize the adapter with the subscription service and user ID.
        
        Args:
            subscription_service: The subscription service
            user_id: ID of the user
        """
        self.subscription_service = subscription_service
        self.user_id = user_id
        self._subscription_status = None
        
    @property
    def subscription_status(self) -> Dict[str, Any]:
        """
        Get the current subscription status with lazy loading.
        
        Returns:
            Dictionary with subscription status information
        """
        if not self._subscription_status:
            self._subscription_status = self.subscription_service.check_subscription_status(self.user_id)
        return self._subscription_status
        
    def refresh_status(self) -> None:
        """Refresh the cached subscription status."""
        self._subscription_status = None
    
    def get_market_data_config(self) -> Dict[str, Any]:
        """
        Get market data configuration based on the user's subscription tier.
        
        Returns:
            Dictionary with market data configuration
        """
        status = self.subscription_status
        tier_name = status.get('tier', 'FREE')
        tier = getattr(SubscriptionTier, tier_name)
        
        # Get base configuration from subscription features
        subscription = self.subscription_service.get_user_subscription(self.user_id)
        features = subscription.get_features() if subscription else {}
        market_data = features.get('market_data', {})
        
        # Default to polygon.io API
        api_key = os.getenv("POLYGON_API_KEY", "")
        
        config = {
            'api_key': api_key,
            'max_symbols': market_data.get('max_symbols', 1),
            'delay_minutes': market_data.get('delay_minutes', 0),
            'real_time': market_data.get('delay_minutes', 1440) == 0
        }
        
        return config
    
    def get_trading_config(self) -> Dict[str, Any]:
        """
        Get trading configuration based on the user's subscription tier.
        
        Returns:
            Dictionary with trading configuration
        """
        status = self.subscription_status
        tier_name = status.get('tier', 'FREE')
        tier = getattr(SubscriptionTier, tier_name)
        
        # Get trading modes from subscription features
        subscription = self.subscription_service.get_user_subscription(self.user_id)
        features = subscription.get_features() if subscription else {}
        trading_modes = features.get('trading_modes', ['paper'])
        
        # Determine which Alpaca configuration to use
        if 'live' in trading_modes:
            # User can access live trading
            api_key = os.getenv("ALPACA_LIVE_KEY", "")
            api_secret = os.getenv("ALPACA_LIVE_SECRET", "")
            api_url = os.getenv("ALPACA_LIVE_URL", "")
            allow_live = True
        else:
            # User can only access paper trading
            api_key = os.getenv("ALPACA_PAPER_KEY", "")
            api_secret = os.getenv("ALPACA_PAPER_SECRET", "")
            api_url = os.getenv("ALPACA_PAPER_URL", "")
            allow_live = False
        
        config = {
            'api_key': api_key,
            'api_secret': api_secret,
            'api_url': api_url,
            'data_url': os.getenv("ALPACA_DATA_URL", ""),
            'allow_paper': True,  # All tiers have paper trading
            'allow_live': allow_live,
            'default_mode': 'paper'  # Default to paper trading
        }
        
        return config
    
    def get_strategy_config(self, strategy_type: str) -> Dict[str, Any]:
        """
        Get strategy configuration based on the user's subscription tier.
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Dictionary with strategy configuration or None if not accessible
        """
        # Check if user has access to this strategy
        if not self.subscription_service.check_strategy_access(self.user_id, strategy_type):
            logger.warning(f"User {self.user_id} attempted to access unauthorized strategy: {strategy_type}")
            return None
        
        # Get strategy-specific configuration
        config = {
            'strategy_type': strategy_type,
            'allowed': True
        }
        
        # Add LLM API key for LLM-based strategies
        if strategy_type in ['llm', 'transformer']:
            config['llm_api_key'] = os.getenv("LLM_API_KEY", "")
        
        return config
    
    def filter_symbols(self, symbols: List[str]) -> List[str]:
        """
        Filter symbols based on the user's subscription limit.
        
        Args:
            symbols: List of symbols to filter
            
        Returns:
            Filtered list of symbols based on subscription tier
        """
        symbol_limit = self.subscription_service.get_symbol_limit(self.user_id)
        
        # If limit is 0, it means unlimited symbols
        if symbol_limit == 0:
            return symbols
            
        # Otherwise, limit to the maximum allowed
        return symbols[:symbol_limit]
    
    def check_access(self, feature_path: str) -> bool:
        """
        Check if the user has access to a specific feature.
        
        Args:
            feature_path: Dot-notation path to the feature
            
        Returns:
            True if the user has access, False otherwise
        """
        return self.subscription_service.check_feature_access(self.user_id, feature_path)
    
    def get_accessible_strategies(self) -> List[str]:
        """
        Get a list of strategy types the user can access.
        
        Returns:
            List of accessible strategy types
        """
        return self.subscription_service.get_user_accessible_strategies(self.user_id)
    
    def record_api_usage(self, service: str, endpoint: str, count: int = 1) -> None:
        """
        Record API usage for rate limiting and analytics.
        
        Args:
            service: Service name (e.g., 'polygon', 'alpaca')
            endpoint: API endpoint
            count: Number of API calls made
        """
        # In a production system, you would implement rate limiting and usage tracking here
        # For now, we'll just log the usage
        logger.info(f"User {self.user_id} made {count} API calls to {service}/{endpoint}")
        
        # Check if we need to warn about approaching limits
        # This would integrate with a rate limiting system in production
        pass
