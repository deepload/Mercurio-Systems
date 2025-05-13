#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Mercurio Monetization Models

Supports:
- Model 1 (EDGE): Tiered Subscription (Free, Starter, Pro, Elite)
- Model 3 (ALPHA): Performance-Based Profit Sharing (Starter, Growth, Professional)

Model is selected via the MERCURIO_MODEL environment variable (EDGE or ALPHA).

A universal SubscriptionTier enum is always available for DB and tests.
"""
import os
from enum import Enum
from typing import Dict, List, Any

MERCURIO_MODEL = os.getenv("MERCURIO_MODEL", "EDGE").strip().upper()

# ---------------- Universal Enum (for DB, tests, helpers) ----------------
class SubscriptionTier(str, Enum):
    # Edge tiers
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ELITE = "elite"
    # Alpha tiers
    GROWTH = "growth"
    PROFESSIONAL = "professional"

# ---------------- Model 1: Mercurio Edge (Tiered) ----------------
class EdgeSubscriptionTier(str, Enum):
    FREE = "free"
    STARTER = "starter"
    PRO = "pro"
    ELITE = "elite"

EDGE_TIER_CONFIG = {
    SubscriptionTier.FREE: {
        "base_fee": 0.0,
        "trial_days": 0,
        "max_strategies": 1,
        "max_symbols": 1,
        "allowed_types": ["moving_average"],
        "paper_trading": True,
        "live_trading": False,
        "real_time_data": False,
        "data_delay_minutes": 1440,  # 1 day
        "portfolio_analytics": False,
        "sentiment_analysis": False,
        "api_access": False,
        "educational_content": {"basic": True, "advanced": False, "webinars": False},
    },
    SubscriptionTier.STARTER: {
        "base_fee": 29.0,
        "trial_days": 7,
        "max_strategies": 3,
        "max_symbols": 5,
        "allowed_types": ["moving_average", "mean_reversion", "momentum"],
        "paper_trading": True,
        "live_trading": True,
        "real_time_data": True,
        "data_delay_minutes": 0,
        "portfolio_analytics": False,
        "sentiment_analysis": False,
        "api_access": False,
        "educational_content": {"basic": True, "advanced": False, "webinars": True},
    },
    SubscriptionTier.PRO: {
        "base_fee": 79.0,
        "trial_days": 7,
        "max_strategies": 10,
        "max_symbols": 50,
        "allowed_types": ["moving_average", "mean_reversion", "momentum", "lstm_predictor", "msi_strategy", "breakout", "statistical_arbitrage", "volatility_strategy", "options_covered_call", "options_cash_secured_put"],
        "paper_trading": True,
        "live_trading": True,
        "real_time_data": True,
        "data_delay_minutes": 0,
        "portfolio_analytics": True,
        "sentiment_analysis": False,
        "api_access": True,
        "educational_content": {"basic": True, "advanced": True, "webinars": True},
    },
    SubscriptionTier.ELITE: {
        "base_fee": 199.0,
        "trial_days": 7,
        "max_strategies": None,  # Unlimited
        "max_symbols": None,     # Unlimited
        "allowed_types": [],     # All strategies
        "paper_trading": True,
        "live_trading": True,
        "real_time_data": True,
        "data_delay_minutes": 0,
        "portfolio_analytics": True,
        "sentiment_analysis": True,
        "api_access": True,
        "educational_content": {"basic": True, "advanced": True, "webinars": True},
    },
    SubscriptionTier.GROWTH: {"inactive": True},
    SubscriptionTier.PROFESSIONAL: {"inactive": True},
}

ALPHA_TIER_CONFIG = {
    SubscriptionTier.FREE: {"inactive": True},
    SubscriptionTier.STARTER: {
        "base_fee": 0.0,
        "profit_share": 0.10,
        "max_strategies": 5,
        "max_portfolio": 10_000,
        "customization": False,
    },
    SubscriptionTier.PRO: {"inactive": True},
    SubscriptionTier.ELITE: {"inactive": True},
    SubscriptionTier.GROWTH: {
        "base_fee": 0.0,
        "profit_share": 0.075,
        "max_strategies": 15,
        "max_portfolio": 100_000,
        "customization": False,
    },
    SubscriptionTier.PROFESSIONAL: {
        "base_fee": 0.0,
        "profit_share": 0.05,
        "max_strategies": None,
        "max_portfolio": None,
        "customization": True,
    },
}


# ---------------- Model 3: Mercurio Alpha (Profit Sharing) ----------------
class AlphaSubscriptionTier(str, Enum):
    STARTER = "starter"
    GROWTH = "growth"
    PROFESSIONAL = "professional"

ALPHA_TIER_CONFIG = {
    AlphaSubscriptionTier.STARTER: {
        "base_fee": 0.0,
        "profit_share": 0.10,
        "max_strategies": 5,
        "max_portfolio": 10_000,
        "customization": False,
    },
    AlphaSubscriptionTier.GROWTH: {
        "base_fee": 0.0,
        "profit_share": 0.075,
        "max_strategies": 15,
        "max_portfolio": 100_000,
        "customization": False,
    },
    AlphaSubscriptionTier.PROFESSIONAL: {
        "base_fee": 0.0,
        "profit_share": 0.05,
        "max_strategies": None,
        "max_portfolio": None,
        "customization": True,
    },
}
ALPHA_HIGH_WATER_MARK_ENABLED = True
# ---------------- Unified Interface ----------------

if MERCURIO_MODEL == "ALPHA":
    SubscriptionTier = AlphaSubscriptionTier
    TIER_CONFIG = ALPHA_TIER_CONFIG
    HIGH_WATER_MARK_ENABLED = ALPHA_HIGH_WATER_MARK_ENABLED
else:
    SubscriptionTier = EdgeSubscriptionTier
    TIER_CONFIG = EDGE_TIER_CONFIG
    HIGH_WATER_MARK_ENABLED = False

def get_tier_info(tier: SubscriptionTier) -> dict:
    # Always return a dict, even for inactive tiers
    if tier in TIER_CONFIG:
        return TIER_CONFIG[tier]
    # Fallback for inactive tiers (e.g., Edge tier in Alpha mode)
    return {}

def get_allowed_strategies(tier: SubscriptionTier) -> list:
    cfg = get_tier_info(tier)
    return cfg.get("allowed_types", [])

def can_access_feature(tier: SubscriptionTier, feature_path: str) -> bool:
    cfg = get_tier_info(tier)
    if not cfg:
        return False
    # For ALPHA: all features are available, limits only on strategies/portfolio
    if MERCURIO_MODEL == "ALPHA":
        return True
    # For EDGE: nested feature support
    if feature_path in cfg:
        return bool(cfg.get(feature_path, False))
    # Support dot notation for nested features
    if "." in feature_path:
        parts = feature_path.split(".")
        current = cfg
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        return bool(current)
    return False

def check_strategy_access(tier: SubscriptionTier, strategy_type: str) -> bool:
    allowed = get_allowed_strategies(tier)
    if not allowed:
        return True
    return strategy_type in allowed

def check_symbol_limit(tier: SubscriptionTier, symbol_count: int) -> bool:
    cfg = get_tier_info(tier)
    if not cfg:
        return False
    max_symbols = cfg.get("max_symbols") if MERCURIO_MODEL == "EDGE" else cfg.get("max_portfolio")
    if max_symbols is None or max_symbols == 0:
        return True
    return symbol_count <= max_symbols

# For future provider expansion:
SUPPORTED_BROKER_PROVIDERS = [
    "alpaca",  # default
    # Add more providers here (e.g., "ibkr", "tradestation", "etrade", ...)
]
def is_supported_broker(provider: str) -> bool:
    return provider.lower() in SUPPORTED_BROKER_PROVIDERS

    return True

def check_strategy_access(tier: SubscriptionTier, strategy_type: str) -> bool:
    """All strategies are available up to the allowed count."""
    return True

def check_symbol_limit(tier: SubscriptionTier, symbol_count: int) -> bool:
    max_portfolio = TIER_CONFIG.get(tier, {}).get("max_portfolio")
    if max_portfolio is None:
        return True
    return symbol_count <= max_portfolio

# For future provider expansion:
SUPPORTED_BROKER_PROVIDERS = [
    "alpaca",  # default
    # Add more providers here (e.g., "ibkr", "tradestation", "etrade", ...)
]

def is_supported_broker(provider: str) -> bool:
    return provider.lower() in SUPPORTED_BROKER_PROVIDERS

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
