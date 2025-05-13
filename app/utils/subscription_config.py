#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Subscription Tier Configuration for Mercurio Edge

This module defines the subscription tiers and feature access for the
Mercurio Edge retail trader product.
"""

from enum import Enum
from typing import Dict, List, Set, Any

class SubscriptionTier(str, Enum):
    """Defines the available subscription tiers for Mercurio Edge."""
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ELITE = "elite"

# Define monthly pricing for each tier
TIER_PRICING = {
    SubscriptionTier.FREE: 0,
    SubscriptionTier.STARTER: 29,
    SubscriptionTier.PRO: 79,
    SubscriptionTier.ELITE: 199
}

# Define features for each tier
TIER_FEATURES = {
    SubscriptionTier.FREE: {
        "strategies": {
            "count": 1,
            "allowed_types": ["moving_average"]
        },
        "trading_modes": ["paper"],
        "market_data": {
            "max_symbols": 1,
            "delay_minutes": 1440  # 24 hours delay (1 day)
        },
        "api_access": False,
        "portfolio_analytics": False,
        "sentiment_analysis": False,
        "educational_content": {
            "basic": True,
            "advanced": False,
            "webinars": False
        }
    },
    SubscriptionTier.STARTER: {
        "strategies": {
            "count": 3,
            "allowed_types": ["moving_average", "mean_reversion", "momentum"]
        },
        "trading_modes": ["paper", "live"],
        "market_data": {
            "max_symbols": 5,
            "delay_minutes": 0  # Real-time
        },
        "api_access": False,
        "portfolio_analytics": False,
        "sentiment_analysis": False,
        "educational_content": {
            "basic": True,
            "advanced": False,
            "webinars": True
        }
    },
    SubscriptionTier.PRO: {
        "strategies": {
            "count": 10,
            "allowed_types": ["moving_average", "mean_reversion", "momentum", 
                             "lstm_predictor", "msi_strategy", "breakout",
                             "statistical_arbitrage", "volatility_strategy",
                             "options_covered_call", "options_cash_secured_put"]
        },
        "trading_modes": ["paper", "live"],
        "market_data": {
            "max_symbols": 50,
            "delay_minutes": 0
        },
        "api_access": True,
        "api_rate_limit": 60,  # requests per minute
        "portfolio_analytics": True,
        "sentiment_analysis": False,
        "educational_content": {
            "basic": True,
            "advanced": True,
            "webinars": True
        }
    },
    SubscriptionTier.ELITE: {
        "strategies": {
            "count": 0,  # Unlimited
            "allowed_types": []  # All strategies allowed
        },
        "trading_modes": ["paper", "live"],
        "market_data": {
            "max_symbols": 0,  # Unlimited
            "delay_minutes": 0
        },
        "api_access": True,
        "api_rate_limit": 300,  # requests per minute
        "portfolio_analytics": True,
        "sentiment_analysis": True,
        "educational_content": {
            "basic": True,
            "advanced": True,
            "webinars": True
        }
    }
}

def get_tier_features(tier: SubscriptionTier) -> Dict[str, Any]:
    """
    Returns the feature dictionary for a specific tier.
    
    Args:
        tier: The subscription tier
        
    Returns:
        Dictionary of features for the specified tier
    """
    return TIER_FEATURES.get(tier, TIER_FEATURES[SubscriptionTier.FREE])

def get_allowed_strategies(tier: SubscriptionTier) -> List[str]:
    """
    Get the list of strategy types allowed for a specific tier.
    
    Args:
        tier: The subscription tier
        
    Returns:
        List of allowed strategy types or empty list for unlimited access
    """
    features = get_tier_features(tier)
    allowed_types = features['strategies']['allowed_types']
    
    # Empty list means all strategies are allowed (ELITE tier)
    if not allowed_types:
        return []
        
    return allowed_types

def can_access_feature(tier: SubscriptionTier, feature_path: str) -> bool:
    """
    Check if a specific tier can access a feature.
    
    Args:
        tier: The subscription tier
        feature_path: Dot-notation path to the feature (e.g., "sentiment_analysis" 
                     or "educational_content.advanced")
        
    Returns:
        True if the tier has access to the feature, False otherwise
    """
    features = get_tier_features(tier)
    
    # Handle dot notation for nested features
    if "." in feature_path:
        parts = feature_path.split(".")
        current = features
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        
        # Convert the final value to boolean
        return bool(current)
    
    # Handle top-level features
    return bool(features.get(feature_path, False))

def check_strategy_access(tier: SubscriptionTier, strategy_type: str) -> bool:
    """
    Check if a user with the given tier can access a specific strategy type.
    
    Args:
        tier: The subscription tier
        strategy_type: The type of strategy to check
        
    Returns:
        True if the tier has access to the strategy, False otherwise
    """
    allowed_strategies = get_allowed_strategies(tier)
    
    # Empty list means all strategies are allowed (ELITE tier)
    if not allowed_strategies:
        return True
        
    return strategy_type in allowed_strategies

def check_symbol_limit(tier: SubscriptionTier, symbol_count: int) -> bool:
    """
    Check if a user with the given tier can access the specified number of symbols.
    
    Args:
        tier: The subscription tier
        symbol_count: The number of symbols to check
        
    Returns:
        True if the tier allows access to the specified number of symbols, False otherwise
    """
    features = get_tier_features(tier)
    max_symbols = features['market_data']['max_symbols']
    
    # 0 means unlimited
    if max_symbols == 0:
        return True
        
    return symbol_count <= max_symbols
