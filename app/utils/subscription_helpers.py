#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Helper functions for subscription-related operations in Mercurio Edge.
"""

from typing import Dict

from app.utils.subscription_config import SubscriptionTier


def get_tier_name(tier: SubscriptionTier) -> str:
    """
    Get user-friendly display name for a subscription tier.
    
    Args:
        tier: The subscription tier
        
    Returns:
        User-friendly display name
    """
    tier_map = {
        SubscriptionTier.FREE: "Free",
        SubscriptionTier.STARTER: "Starter",
        SubscriptionTier.PRO: "Pro",
        SubscriptionTier.ELITE: "Elite"
    }
    
    return tier_map.get(tier, "Unknown")


def get_tier_price(tier: SubscriptionTier) -> float:
    """
    Get monthly price for a subscription tier.
    
    Args:
        tier: The subscription tier
        
    Returns:
        Monthly price in USD
    """
    tier_map = {
        SubscriptionTier.FREE: 0.0,
        SubscriptionTier.STARTER: 29.0,
        SubscriptionTier.PRO: 79.0,
        SubscriptionTier.ELITE: 199.0
    }
    
    return tier_map.get(tier, 0.0)


def get_tier_description(tier: SubscriptionTier) -> str:
    """
    Get description for a subscription tier.
    
    Args:
        tier: The subscription tier
        
    Returns:
        Description text
    """
    tier_map = {
        SubscriptionTier.FREE: "Basic algorithmic trading capabilities with one strategy and paper trading only.",
        SubscriptionTier.STARTER: "Get started with algorithmic trading using 3 strategies and real-time data for 5 symbols.",
        SubscriptionTier.PRO: "Advanced trading with 10 strategies, real-time data for 50 symbols, and portfolio analytics.",
        SubscriptionTier.ELITE: "Professional trading with all strategies, unlimited symbols, priority API access, and sentiment analysis."
    }
    
    return tier_map.get(tier, "Unknown subscription tier")


def get_tier_comparison() -> Dict[str, Dict]:
    """
    Get comparison information for all tiers.
    
    Returns:
        Dictionary with tier comparison data
    """
    return {
        "free": {
            "name": "Free",
            "price": 0,
            "description": "Basic algorithmic trading capabilities",
            "features": [
                "1 basic strategy (Moving Average)",
                "Paper trading only",
                "1-day delayed market data",
                "Basic educational content"
            ]
        },
        "starter": {
            "name": "Starter",
            "price": 29,
            "description": "Get started with algorithmic trading",
            "features": [
                "3 basic strategies",
                "Paper + live trading",
                "Real-time data for 5 symbols",
                "Basic educational content"
            ]
        },
        "pro": {
            "name": "Pro",
            "price": 79,
            "description": "Advanced trading capabilities",
            "features": [
                "10 strategies (including LSTM and Momentum)",
                "Paper + live trading",
                "Real-time data for 50 symbols",
                "Portfolio analytics",
                "Advanced educational content"
            ]
        },
        "elite": {
            "name": "Elite",
            "price": 199,
            "description": "Professional trading platform",
            "features": [
                "All strategies (including LLM and Transformer models)",
                "Paper + live trading",
                "Unlimited symbols with real-time data",
                "Priority API access",
                "Sentiment analysis",
                "Portfolio analytics",
                "Webinars and training sessions"
            ]
        }
    }
