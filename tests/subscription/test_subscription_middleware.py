#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for subscription middleware functionality.
"""

import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from fastapi import HTTPException, Request

from app.middleware.subscription_middleware import SubscriptionAuthMiddleware
from app.services.subscription_adapter import SubscriptionServiceAdapter
from app.utils.subscription_config import SubscriptionTier


class TestSubscriptionMiddleware(unittest.TestCase):
    """Tests for subscription middleware functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock adapter and request
        self.adapter = MagicMock(spec=SubscriptionServiceAdapter)
        self.request = MagicMock(spec=Request)
        self.request.state.user_id = 1
        
        # Create the middleware
        self.middleware = SubscriptionAuthMiddleware()
    
    @pytest.mark.asyncio
    async def test_get_subscription_adapter(self):
        """Test getting a subscription adapter."""
        # Mock dependencies
        db = MagicMock()
        
        # Patch the subscription service class
        with patch('app.middleware.subscription_middleware.SubscriptionService') as MockService:
            # Call the dependency
            result = await SubscriptionAuthMiddleware.get_subscription_adapter(self.request, db)
            
            # Check that service was created with DB
            MockService.assert_called_once_with(db)
            
            # Check that adapter was created with service and user ID
            service = MockService.return_value
            self.assertIsInstance(result, SubscriptionServiceAdapter)
            self.assertEqual(result.user_id, 1)
    
    @pytest.mark.asyncio
    async def test_require_feature_with_access(self):
        """Test requiring a feature when user has access."""
        # Mock endpoint function
        async def test_endpoint(subscription_adapter=None):
            return {"status": "success"}
        
        # Mock adapter to have access
        self.adapter.check_access.return_value = True
        
        # Create the decorator
        decorator = SubscriptionAuthMiddleware.require_feature('api_access')
        decorated_func = decorator(test_endpoint)
        
        # Call the decorated function
        result = await decorated_func(subscription_adapter=self.adapter)
        
        # Check that access was checked
        self.adapter.check_access.assert_called_once_with('api_access')
        
        # Check that endpoint was called and returned correctly
        self.assertEqual(result, {"status": "success"})
    
    @pytest.mark.asyncio
    async def test_require_feature_without_access(self):
        """Test requiring a feature when user doesn't have access."""
        # Mock endpoint function
        async def test_endpoint(subscription_adapter=None):
            return {"status": "success"}
        
        # Mock adapter to not have access
        self.adapter.check_access.return_value = False
        self.adapter.subscription_status = {
            'display_tier': 'Free',
        }
        self.adapter.user_id = 1
        
        # Create the decorator
        decorator = SubscriptionAuthMiddleware.require_feature('sentiment_analysis')
        decorated_func = decorator(test_endpoint)
        
        # Call the decorated function - should raise HTTPException
        with patch('app.utils.subscription_config.can_access_feature') as mock_access:
            mock_access.return_value = True  # Mock the min tier check
            
            with patch('app.utils.subscription_helpers.get_tier_name') as mock_tier_name:
                mock_tier_name.return_value = 'Pro'  # Mock tier name
                
                with self.assertRaises(HTTPException) as context:
                    await decorated_func(subscription_adapter=self.adapter)
                
                # Check that the exception has the right status code and detail
                exception = context.exception
                self.assertEqual(exception.status_code, 403)
                self.assertTrue("upgrade" in exception.detail.lower())
    
    @pytest.mark.asyncio
    async def test_require_strategy_access_with_access(self):
        """Test requiring strategy access when user has access."""
        # Mock endpoint function
        async def test_endpoint(strategy, subscription_adapter=None):
            return {"strategy": strategy}
        
        # Mock adapter to have access
        self.adapter.subscription_service.check_strategy_access.return_value = True
        
        # Create the decorator and decorate the function
        decorator = SubscriptionAuthMiddleware.require_strategy_access()
        decorated_func = decorator(test_endpoint)
        
        # Call the decorated function
        result = await decorated_func(strategy="moving_average", subscription_adapter=self.adapter)
        
        # Check that strategy access was checked
        self.adapter.subscription_service.check_strategy_access.assert_called_once_with(
            self.adapter.user_id, "moving_average"
        )
        
        # Check that endpoint was called and returned correctly
        self.assertEqual(result, {"strategy": "moving_average"})
    
    @pytest.mark.asyncio
    async def test_require_strategy_access_without_access(self):
        """Test requiring strategy access when user doesn't have access."""
        # Mock endpoint function
        async def test_endpoint(strategy, subscription_adapter=None):
            return {"strategy": strategy}
        
        # Mock adapter to not have access
        self.adapter.subscription_service.check_strategy_access.return_value = False
        self.adapter.subscription_status = {
            'display_tier': 'Starter',
        }
        self.adapter.user_id = 1
        
        # Create the decorator
        decorator = SubscriptionAuthMiddleware.require_strategy_access()
        decorated_func = decorator(test_endpoint)
        
        # Call the decorated function - should raise HTTPException
        with patch('app.utils.subscription_config.check_strategy_access') as mock_access:
            mock_access.return_value = True  # Mock the min tier check
            
            with patch('app.utils.subscription_helpers.get_tier_name') as mock_tier_name:
                mock_tier_name.return_value = 'Pro'  # Mock tier name
                
                with self.assertRaises(HTTPException) as context:
                    await decorated_func(strategy="lstm", subscription_adapter=self.adapter)
                
                # Check that the exception has the right status code and detail
                exception = context.exception
                self.assertEqual(exception.status_code, 403)
                self.assertTrue("lstm" in exception.detail.lower())
                self.assertTrue("upgrade" in exception.detail.lower())
    
    @pytest.mark.asyncio
    async def test_require_symbol_limit_within_limit(self):
        """Test requiring symbol limit when within limit."""
        # Mock endpoint function
        async def test_endpoint(symbols, subscription_adapter=None):
            return {"symbols": symbols}
        
        # Mock adapter to be within limit
        self.adapter.subscription_service.check_symbol_limit.return_value = True
        
        # Setup filtered symbols (same as input)
        test_symbols = ["AAPL", "MSFT", "GOOGL"]
        self.adapter.filter_symbols.return_value = test_symbols
        
        # Create the decorator
        decorator = SubscriptionAuthMiddleware.require_symbol_limit()
        decorated_func = decorator(test_endpoint)
        
        # Call the decorated function
        result = await decorated_func(symbols=test_symbols, subscription_adapter=self.adapter)
        
        # Check that symbol limit was checked
        self.adapter.subscription_service.check_symbol_limit.assert_called_once_with(
            self.adapter.user_id, len(test_symbols)
        )
        
        # Check that symbols were filtered
        self.adapter.filter_symbols.assert_called_once_with(test_symbols)
        
        # Check that endpoint was called and returned correctly
        self.assertEqual(result, {"symbols": test_symbols})
    
    @pytest.mark.asyncio
    async def test_require_symbol_limit_exceeding_limit(self):
        """Test requiring symbol limit when exceeding limit."""
        # Mock endpoint function
        async def test_endpoint(symbols, subscription_adapter=None):
            return {"symbols": symbols}
        
        # Mock adapter to exceed limit
        self.adapter.subscription_service.check_symbol_limit.return_value = False
        self.adapter.subscription_status = {
            'display_tier': 'Starter',
        }
        self.adapter.user_id = 1
        
        # Mock symbol limit
        self.adapter.subscription_service.get_symbol_limit.return_value = 5
        
        # Create the decorator
        decorator = SubscriptionAuthMiddleware.require_symbol_limit()
        decorated_func = decorator(test_endpoint)
        
        # Test symbols exceeding limit
        test_symbols = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NFLX", "META"]
        
        # Call the decorated function - should raise HTTPException
        with patch('app.utils.subscription_config.check_symbol_limit') as mock_check:
            mock_check.return_value = True  # Mock the min tier check
            
            with patch('app.utils.subscription_helpers.get_tier_name') as mock_tier_name:
                mock_tier_name.return_value = 'Pro'  # Mock tier name
                
                with self.assertRaises(HTTPException) as context:
                    await decorated_func(symbols=test_symbols, subscription_adapter=self.adapter)
                
                # Check that the exception has the right status code and detail
                exception = context.exception
                self.assertEqual(exception.status_code, 403)
                self.assertTrue("limit of 5 symbols" in exception.detail)
                self.assertTrue("requested 7 symbols" in exception.detail)
    
    @pytest.mark.asyncio
    async def test_require_live_trading_access(self):
        """Test requiring live trading access."""
        # Test that it uses the require_feature decorator with right path
        with patch('app.middleware.subscription_middleware.SubscriptionAuthMiddleware.require_feature') as mock_feature:
            # Call the method
            result = SubscriptionAuthMiddleware.require_live_trading_access()
            
            # Check that require_feature was called with trading_modes.live
            mock_feature.assert_called_once_with('trading_modes.live')


if __name__ == '__main__':
    unittest.main()
