#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for subscription adapter functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
import os

from app.services.subscription_adapter import SubscriptionServiceAdapter
from app.utils.subscription_config import SubscriptionTier


class TestSubscriptionAdapter(unittest.TestCase):
    """Tests for subscription adapter functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock subscription service
        self.subscription_service = MagicMock()
        
        # Create user ID for testing
        self.user_id = 1
        
        # Create the adapter with mock service
        self.adapter = SubscriptionServiceAdapter(self.subscription_service, self.user_id)
        
        # We'll use patch for environment variables in each test method directly
    
    def test_check_access_true(self):
        """Test checking access when user has access."""
        # Set up mock service to return True for feature access
        self.subscription_service.check_feature_access.return_value = True
        
        # Call the method
        result = self.adapter.check_access('trading_modes.live')
        
        # Assert the underlying service was called correctly
        self.subscription_service.check_feature_access.assert_called_once_with(
            self.user_id, 'trading_modes.live'
        )
        
        # Result should be True
        self.assertTrue(result)
    
    def test_check_access_false(self):
        """Test checking access when user doesn't have access."""
        # Set up mock service to return False for feature access
        self.subscription_service.check_feature_access.return_value = False
        
        # Call the method
        result = self.adapter.check_access('sentiment_analysis')
        
        # Assert the underlying service was called correctly
        self.subscription_service.check_feature_access.assert_called_once_with(
            self.user_id, 'sentiment_analysis'
        )
        
        # Result should be False
        self.assertFalse(result)
    
    def test_subscription_status(self):
        """Test getting subscription status."""
        # Set up mock status return
        status_data = {
            'has_subscription': True,
            'tier': 'PRO',
            'display_tier': 'Pro',
            'status': 'active',
            'is_active': True,
            'is_trial': False,
            'days_remaining': 25,
        }
        self.subscription_service.check_subscription_status.return_value = status_data
        
        # Call the property
        result = self.adapter.subscription_status
        
        # Assert the underlying service was called correctly
        self.subscription_service.check_subscription_status.assert_called_once_with(self.user_id)
        
        # Result should match the mock status
        self.assertEqual(result, status_data)
    
    def test_get_trading_config_free_tier(self):
        """Test getting trading config for free tier."""
        # Set up mock status for free tier
        self.subscription_service.check_subscription_status.return_value = {
            'has_subscription': True,
            'tier': 'FREE',
            'is_active': True,
        }
        
        # Mock subscription object for features
        mock_subscription = MagicMock()
        mock_subscription.get_features.return_value = {
            'trading_modes': ['paper']
        }
        self.subscription_service.get_user_subscription.return_value = mock_subscription
        
        # Directly patch os.getenv to return test values
        with patch('app.services.subscription_adapter.os.getenv') as mock_getenv:
            def getenv_side_effect(key, default=None):
                if key == 'ALPACA_PAPER_KEY':
                    return 'test_alpaca_paper_key'
                elif key == 'ALPACA_PAPER_SECRET':
                    return 'test_alpaca_paper_secret'
                elif key == 'ALPACA_PAPER_URL':
                    return 'https://paper-api.alpaca.markets'
                return default
            
            # Set up the side effect
            mock_getenv.side_effect = getenv_side_effect
            
            # Call the method
            result = self.adapter.get_trading_config()
            
            # Check that we got paper trading credentials
            self.assertEqual(result['api_key'], 'test_alpaca_paper_key')
            self.assertEqual(result['api_secret'], 'test_alpaca_paper_secret')
            self.assertEqual(result['api_url'], 'https://paper-api.alpaca.markets')
            self.assertFalse(result['allow_live'])
    
    def test_get_trading_config_paid_tier(self):
        """Test getting trading config for paid tier."""
        # Set up mock status for paid tier (PRO)
        self.subscription_service.check_subscription_status.return_value = {
            'has_subscription': True,
            'tier': 'PRO',
            'is_active': True,
        }
        
        # Mock subscription object with live trading mode
        mock_subscription = MagicMock()
        mock_subscription.get_features.return_value = {
            'trading_modes': ['paper', 'live']
        }
        self.subscription_service.get_user_subscription.return_value = mock_subscription
        
        # Patch os.getenv to return test values for live trading
        with patch('app.services.subscription_adapter.os.getenv') as mock_getenv:
            def getenv_side_effect(key, default=None):
                if key == 'ALPACA_LIVE_KEY':
                    return 'test_alpaca_key'
                elif key == 'ALPACA_LIVE_SECRET':
                    return 'test_alpaca_secret'
                elif key == 'ALPACA_LIVE_URL':
                    return 'https://api.alpaca.markets'
                return default
            
            # Set up the side effect
            mock_getenv.side_effect = getenv_side_effect
            
            # Call the method
            result = self.adapter.get_trading_config()
            
            # Should get live API keys
            self.assertEqual(result['api_key'], 'test_alpaca_key')
            self.assertEqual(result['api_secret'], 'test_alpaca_secret')
            self.assertEqual(result['api_url'], 'https://api.alpaca.markets')
            self.assertTrue(result['allow_live'])
    
    def test_get_market_data_config_free_tier(self):
        """Test getting market data config for free tier."""
        # Set up mock status for free tier
        self.subscription_service.check_subscription_status.return_value = {
            'has_subscription': True,
            'tier': 'FREE',
            'is_active': True,
        }
        
        # Mock subscription object with market data settings
        mock_subscription = MagicMock()
        mock_subscription.get_features.return_value = {
            'market_data': {
                'max_symbols': 1,
                'delay_minutes': 1440  # 24-hour delay
            }
        }
        self.subscription_service.get_user_subscription.return_value = mock_subscription
        
        # Patch os.getenv to return test values
        with patch('app.services.subscription_adapter.os.getenv') as mock_getenv:
            mock_getenv.return_value = 'test_polygon_key'
            
            # Call the method
            result = self.adapter.get_market_data_config()
            
            # Check that we got delayed data config
            self.assertEqual(result['api_key'], 'test_polygon_key')
            self.assertEqual(result['delay_minutes'], 1440)
            self.assertFalse(result['real_time'])
            self.assertEqual(result['max_symbols'], 1)
    
    def test_get_market_data_config_elite_tier(self):
        """Test getting market data config for elite tier."""
        # Set up mock status for elite tier
        self.subscription_service.check_subscription_status.return_value = {
            'has_subscription': True,
            'tier': 'ELITE',
            'is_active': True,
        }
        
        # Mock subscription object with market data settings for elite tier
        mock_subscription = MagicMock()
        mock_subscription.get_features.return_value = {
            'market_data': {
                'max_symbols': 0,  # Unlimited
                'delay_minutes': 0  # Real-time
            }
        }
        self.subscription_service.get_user_subscription.return_value = mock_subscription
        
        # Patch os.getenv to return test values
        with patch('app.services.subscription_adapter.os.getenv') as mock_getenv:
            mock_getenv.return_value = 'test_polygon_key'
            
            # Call the method
            result = self.adapter.get_market_data_config()
            
            # Check that we got real-time data config with unlimited symbols
            self.assertEqual(result['api_key'], 'test_polygon_key')
            self.assertEqual(result['delay_minutes'], 0)  # No delay
            self.assertTrue(result['real_time'])  # Real-time data
            self.assertEqual(result['max_symbols'], 0)  # Unlimited symbols
    
    def test_filter_symbols_within_limit(self):
        """Test filtering symbols when within limit."""
        # Set up a list of symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Set up mock to allow all symbols
        self.subscription_service.get_symbol_limit.return_value = 10  # limit of 10
        
        # Call the method
        result = self.adapter.filter_symbols(symbols)
        
        # All symbols should be included
        self.assertEqual(len(result), 5)
        self.assertEqual(set(result), set(symbols))
    
    def test_filter_symbols_exceeding_limit(self):
        """Test filtering symbols when exceeding limit."""
        # Set up a list of symbols
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Set up mock with a limit lower than requested
        self.subscription_service.get_symbol_limit.return_value = 3  # limit of 3
        
        # Call the method
        result = self.adapter.filter_symbols(symbols)
        
        # Only first 3 symbols should be included
        self.assertEqual(len(result), 3)
        self.assertTrue(all(s in symbols for s in result))
    
    def test_get_llm_api_config_with_access(self):
        """Test getting LLM API config with access."""
        # Set up mock status for elite tier
        self.subscription_service.check_subscription_status.return_value = {
            'has_subscription': True,
            'tier': 'ELITE',
            'is_active': True,
        }
        
        # Mock strategy access
        self.subscription_service.check_strategy_access.return_value = True
        
        # Patch os.getenv to return test values
        with patch('app.services.subscription_adapter.os.getenv') as mock_getenv:
            # Mock the LLM API key
            mock_getenv.return_value = 'test_llm_key'
            
            # Call the method
            result = self.adapter.get_strategy_config('llm')
            
            # Check that we got the LLM API key
            self.assertEqual(result['llm_api_key'], 'test_llm_key')
            self.assertTrue(result['allowed'])
    
    def test_get_llm_api_config_without_access(self):
        """Test getting LLM API config without access."""
        # Set up mock status for non-elite tier
        self.subscription_service.check_subscription_status.return_value = {
            'has_subscription': True,
            'tier': 'STARTER',
            'is_active': True,
        }
        
        # Mock strategy access - not allowed
        self.subscription_service.check_strategy_access.return_value = False
        
        # Call the method
        result = self.adapter.get_strategy_config('llm')
        
        # Check that no result is returned when access is denied
        self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
