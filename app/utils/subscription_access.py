#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utilities for checking subscription-based feature access in Mercurio Edge.
"""

from typing import Optional, List

from app.db.models import User, Subscription
from app.utils.subscription_config import SubscriptionTier, can_access_feature, check_strategy_access, check_symbol_limit


async def check_user_feature_access(user: User, feature_path: str) -> bool:
    """
    Check if a user has access to a specific feature.
    
    Args:
        user: The user to check
        feature_path: Dot-notation path to the feature
        
    Returns:
        True if the user has access, False otherwise
    """
    if not user or not user.subscription:
        # Default to free tier if no subscription
        return can_access_feature(SubscriptionTier.FREE, feature_path)
        
    return can_access_feature(user.subscription.tier, feature_path)


async def check_user_strategy_access(user: User, strategy_type: str) -> bool:
    """
    Check if a user can access a specific strategy type.
    
    Args:
        user: The user to check
        strategy_type: Type of strategy to check access for
        
    Returns:
        True if the user has access to this strategy, False otherwise
    """
    if not user or not user.subscription:
        # Default to free tier if no subscription
        return check_strategy_access(SubscriptionTier.FREE, strategy_type)
        
    return check_strategy_access(user.subscription.tier, strategy_type)


async def check_user_symbol_limit(user: User, symbols: List[str]) -> bool:
    """
    Check if a user is within their symbol limit.
    
    Args:
        user: The user to check
        symbols: List of symbols to check
        
    Returns:
        True if the user is within their symbol limit, False otherwise
    """
    if not user or not user.subscription:
        # Default to free tier if no subscription
        return check_symbol_limit(SubscriptionTier.FREE, len(symbols))
        
    return check_symbol_limit(user.subscription.tier, len(symbols))


async def get_user_tier_name(user: User) -> str:
    """
    Get the display name of a user's subscription tier.
    
    Args:
        user: The user to check
        
    Returns:
        Display name of the subscription tier
    """
    if not user or not user.subscription:
        return "Free"
        
    tier_map = {
        SubscriptionTier.FREE: "Free",
        SubscriptionTier.STARTER: "Starter",
        SubscriptionTier.PRO: "Pro",
        SubscriptionTier.ELITE: "Elite"
    }
    
    return tier_map.get(user.subscription.tier, "Unknown")
