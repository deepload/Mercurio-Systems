#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Middleware for enforcing subscription tier permissions in request handlers.
"""

import logging
from functools import wraps
from typing import Callable, Dict, Any, Optional, Union

from fastapi import HTTPException, Depends, Request
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.services.subscription_service import SubscriptionService
from app.services.subscription_adapter import SubscriptionServiceAdapter
from app.utils.subscription_config import SubscriptionTier

logger = logging.getLogger(__name__)


class SubscriptionAuthMiddleware:
    """Middleware for enforcing subscription-based permissions."""
    
    @staticmethod
    async def get_subscription_adapter(
        request: Request, db: Session = Depends(get_db)
    ) -> SubscriptionServiceAdapter:
        """
        Dependency for getting a subscription adapter.
        
        Args:
            request: FastAPI request object
            db: Database session
            
        Returns:
            SubscriptionServiceAdapter instance
        """
        # Get user ID from request state (would be set by authentication middleware)
        # For development, we'll use a mock user ID
        user_id = getattr(request.state, 'user_id', 1)  # Default to mock user ID 1
        
        # Create subscription service and adapter
        subscription_service = SubscriptionService(db)
        return SubscriptionServiceAdapter(subscription_service, user_id)
    
    @staticmethod
    def require_feature(feature_path: str):
        """
        Decorator for requiring access to a specific feature.
        
        Args:
            feature_path: Dot-notation path to the required feature
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(
                *args, 
                subscription_adapter: SubscriptionServiceAdapter = Depends(
                    SubscriptionAuthMiddleware.get_subscription_adapter
                ),
                **kwargs
            ):
                # Check if user has access to the feature
                if not subscription_adapter.check_access(feature_path):
                    # Get status to identify user's current tier
                    status = subscription_adapter.subscription_status
                    current_tier = status.get('display_tier', 'Free')
                    
                    # Log the access attempt
                    logger.warning(
                        f"Access denied: User ID {subscription_adapter.user_id} with {current_tier} "
                        f"tier attempted to access {feature_path}"
                    )
                    
                    # Find the minimum tier that provides access
                    min_tier = None
                    for tier in SubscriptionTier:
                        from app.utils.subscription_config import can_access_feature
                        if can_access_feature(tier, feature_path):
                            min_tier = tier
                            break
                            
                    from app.utils.subscription_helpers import get_tier_name
                    upgrade_message = ""
                    if min_tier:
                        upgrade_message = f" Please upgrade to {get_tier_name(min_tier)} tier or higher."
                    
                    raise HTTPException(
                        status_code=403,
                        detail=f"Your subscription tier does not provide access to this feature.{upgrade_message}"
                    )
                
                # User has access, proceed with the function
                return await func(*args, subscription_adapter=subscription_adapter, **kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def require_strategy_access(strategy_type: str = None):
        """
        Decorator for requiring access to a specific strategy type.
        If strategy_type is None, it will be extracted from request parameters.
        
        Args:
            strategy_type: Type of strategy to check access for (optional)
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(
                *args, 
                subscription_adapter: SubscriptionServiceAdapter = Depends(
                    SubscriptionAuthMiddleware.get_subscription_adapter
                ),
                **kwargs
            ):
                # Get strategy type from parameters if not specified
                strategy = strategy_type
                if not strategy:
                    # Try to get it from the request
                    for name, value in kwargs.items():
                        if name in ['strategy', 'strategy_name', 'strategy_type']:
                            strategy = value
                            break
                
                if not strategy:
                    logger.warning("Strategy type not specified for subscription check")
                    # If we can't determine the strategy, allow the request but log warning
                    return await func(*args, subscription_adapter=subscription_adapter, **kwargs)
                
                # Check if user has access to the strategy
                if not subscription_adapter.subscription_service.check_strategy_access(
                    subscription_adapter.user_id, strategy
                ):
                    # Get status to identify user's current tier
                    status = subscription_adapter.subscription_status
                    current_tier = status.get('display_tier', 'Free')
                    
                    # Find the minimum tier that provides access
                    from app.utils.subscription_config import check_strategy_access
                    min_tier = None
                    for tier in SubscriptionTier:
                        if check_strategy_access(tier, strategy):
                            min_tier = tier
                            break
                            
                    from app.utils.subscription_helpers import get_tier_name
                    upgrade_message = ""
                    if min_tier:
                        upgrade_message = f" Please upgrade to {get_tier_name(min_tier)} tier or higher."
                    
                    raise HTTPException(
                        status_code=403,
                        detail=(
                            f"Your {current_tier} subscription tier does not provide access "
                            f"to the {strategy} strategy.{upgrade_message}"
                        )
                    )
                
                # User has access, proceed with the function
                return await func(*args, subscription_adapter=subscription_adapter, **kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def require_symbol_limit(field_name: str = 'symbols'):
        """
        Decorator for enforcing symbol limits based on subscription tier.
        
        Args:
            field_name: Name of the parameter containing symbols
            
        Returns:
            Decorator function
        """
        def decorator(func: Callable):
            @wraps(func)
            async def wrapper(
                *args, 
                subscription_adapter: SubscriptionServiceAdapter = Depends(
                    SubscriptionAuthMiddleware.get_subscription_adapter
                ),
                **kwargs
            ):
                # Get symbols from kwargs
                symbols = kwargs.get(field_name)
                if not symbols:
                    # If no symbols parameter, just proceed
                    return await func(*args, subscription_adapter=subscription_adapter, **kwargs)
                
                # Convert to list if it's a single string
                if isinstance(symbols, str):
                    symbols = [symbols]
                
                # Check if user is within their symbol limit
                if not subscription_adapter.subscription_service.check_symbol_limit(
                    subscription_adapter.user_id, len(symbols)
                ):
                    # Get status to identify user's current tier
                    status = subscription_adapter.subscription_status
                    current_tier = status.get('display_tier', 'Free')
                    symbol_limit = subscription_adapter.subscription_service.get_symbol_limit(
                        subscription_adapter.user_id
                    )
                    
                    # Find the minimum tier that provides sufficient symbols
                    from app.utils.subscription_config import check_symbol_limit
                    min_tier = None
                    for tier in SubscriptionTier:
                        if check_symbol_limit(tier, len(symbols)):
                            min_tier = tier
                            break
                            
                    from app.utils.subscription_helpers import get_tier_name
                    upgrade_message = ""
                    if min_tier:
                        upgrade_message = f" Please upgrade to {get_tier_name(min_tier)} tier or higher."
                    
                    raise HTTPException(
                        status_code=403,
                        detail=(
                            f"Your {current_tier} subscription tier has a limit of {symbol_limit} symbols. "
                            f"You requested {len(symbols)} symbols.{upgrade_message}"
                        )
                    )
                
                # Filter symbols if needed and update kwargs
                filtered_symbols = subscription_adapter.filter_symbols(symbols)
                kwargs[field_name] = filtered_symbols
                
                # User is within limit, proceed with the function
                return await func(*args, subscription_adapter=subscription_adapter, **kwargs)
            
            return wrapper
        return decorator
    
    @staticmethod
    def require_live_trading_access():
        """
        Decorator for requiring access to live trading.
        
        Returns:
            Decorator function
        """
        return SubscriptionAuthMiddleware.require_feature('trading_modes.live')
    
    @staticmethod
    def require_api_access():
        """
        Decorator for requiring API access.
        
        Returns:
            Decorator function
        """
        return SubscriptionAuthMiddleware.require_feature('api_access')
    
    @staticmethod
    def require_sentiment_analysis():
        """
        Decorator for requiring sentiment analysis access.
        
        Returns:
            Decorator function
        """
        return SubscriptionAuthMiddleware.require_feature('sentiment_analysis')
    
    @staticmethod
    def require_portfolio_analytics():
        """
        Decorator for requiring portfolio analytics access.
        
        Returns:
            Decorator function
        """
        return SubscriptionAuthMiddleware.require_feature('portfolio_analytics')
