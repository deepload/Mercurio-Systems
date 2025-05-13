#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for subscription configuration functionality.
"""

import unittest
from app.utils.subscription_config import (
    SubscriptionTier, get_tier_features, can_access_feature,
    check_strategy_access, check_symbol_limit
)


class TestSubscriptionConfig(unittest.TestCase):
    """Tests for subscription tier configuration."""
    
    def test_tier_enum(self):
        """Test that SubscriptionTier enum contains expected values."""
        tiers = list(SubscriptionTier)
        self.assertEqual(len(tiers), 4)
        self.assertIn(SubscriptionTier.FREE, tiers)
        self.assertIn(SubscriptionTier.STARTER, tiers)
        self.assertIn(SubscriptionTier.PRO, tiers)
        self.assertIn(SubscriptionTier.ELITE, tiers)
    
    def test_get_tier_features(self):
        """Test retrieval of features for different tiers."""
        # Free tier features
        free_features = get_tier_features(SubscriptionTier.FREE)
        self.assertIn('strategies', free_features)
        self.assertIn('trading_modes', free_features)
        self.assertIn('market_data', free_features)
        
        # Free tier should only have paper trading
        self.assertEqual(free_features['trading_modes'], ['paper'])
        
        # Free tier should have limited strategies
        self.assertEqual(free_features['strategies']['count'], 1)
        self.assertIn('moving_average', free_features['strategies']['allowed_types'])
        self.assertEqual(len(free_features['strategies']['allowed_types']), 1)
        
        # Elite tier features
        elite_features = get_tier_features(SubscriptionTier.ELITE)
        
        # Elite tier should have both paper and live trading
        self.assertIn('paper', elite_features['trading_modes'])
        self.assertIn('live', elite_features['trading_modes'])
        
        # Elite tier should have unlimited strategies (represented as 0 count)
        self.assertEqual(elite_features['strategies']['count'], 0)
        # Elite tier should have empty allowed_types list, which means all are allowed
        self.assertEqual(len(elite_features['strategies']['allowed_types']), 0)
        
        # Elite tier should have unlimited symbols
        self.assertEqual(elite_features['market_data']['max_symbols'], 0)  # 0 means unlimited
        
        # Elite tier should have all premium features
        self.assertTrue(elite_features['api_access'])
        self.assertTrue(elite_features['portfolio_analytics'])
        self.assertTrue(elite_features['sentiment_analysis'])
    
    def test_feature_access(self):
        """Test feature access based on tier."""
        # Free tier tests
        self.assertIn('paper', get_tier_features(SubscriptionTier.FREE)['trading_modes'])
        self.assertNotIn('live', get_tier_features(SubscriptionTier.FREE)['trading_modes'])
        self.assertFalse(can_access_feature(SubscriptionTier.FREE, 'api_access'))
        
        # Starter tier tests
        self.assertIn('paper', get_tier_features(SubscriptionTier.STARTER)['trading_modes'])
        self.assertIn('live', get_tier_features(SubscriptionTier.STARTER)['trading_modes'])
        self.assertFalse(can_access_feature(SubscriptionTier.STARTER, 'sentiment_analysis'))
        
        # Pro tier tests
        self.assertTrue(can_access_feature(SubscriptionTier.PRO, 'portfolio_analytics'))
        self.assertFalse(can_access_feature(SubscriptionTier.PRO, 'sentiment_analysis'))
        
        # Elite tier tests
        self.assertTrue(can_access_feature(SubscriptionTier.ELITE, 'portfolio_analytics'))
        self.assertTrue(can_access_feature(SubscriptionTier.ELITE, 'sentiment_analysis'))
        self.assertTrue(can_access_feature(SubscriptionTier.ELITE, 'api_access'))
    
    def test_strategy_access(self):
        """Test strategy access based on tier."""
        # Free tier can only access moving_average
        self.assertTrue(check_strategy_access(SubscriptionTier.FREE, 'moving_average'))
        self.assertFalse(check_strategy_access(SubscriptionTier.FREE, 'momentum'))
        self.assertFalse(check_strategy_access(SubscriptionTier.FREE, 'lstm'))
        self.assertFalse(check_strategy_access(SubscriptionTier.FREE, 'llm'))
        
        # Starter tier has limited strategies
        self.assertTrue(check_strategy_access(SubscriptionTier.STARTER, 'moving_average'))
        self.assertTrue(check_strategy_access(SubscriptionTier.STARTER, 'momentum'))
        self.assertTrue(check_strategy_access(SubscriptionTier.STARTER, 'mean_reversion'))
        self.assertFalse(check_strategy_access(SubscriptionTier.STARTER, 'lstm_predictor'))
        self.assertFalse(check_strategy_access(SubscriptionTier.STARTER, 'transformer'))
        
        # Pro tier has most strategies
        self.assertTrue(check_strategy_access(SubscriptionTier.PRO, 'moving_average'))
        self.assertTrue(check_strategy_access(SubscriptionTier.PRO, 'momentum'))
        self.assertTrue(check_strategy_access(SubscriptionTier.PRO, 'lstm_predictor'))
        self.assertFalse(check_strategy_access(SubscriptionTier.PRO, 'transformer'))
        self.assertFalse(check_strategy_access(SubscriptionTier.PRO, 'llm'))
        
        # Elite tier has all strategies
        self.assertTrue(check_strategy_access(SubscriptionTier.ELITE, 'moving_average'))
        self.assertTrue(check_strategy_access(SubscriptionTier.ELITE, 'lstm_predictor'))
        self.assertTrue(check_strategy_access(SubscriptionTier.ELITE, 'transformer'))
        self.assertTrue(check_strategy_access(SubscriptionTier.ELITE, 'llm'))
    
    def test_symbol_limit(self):
        """Test symbol limits based on tier."""
        # Free tier has only 1 symbol
        self.assertTrue(check_symbol_limit(SubscriptionTier.FREE, 1))
        self.assertFalse(check_symbol_limit(SubscriptionTier.FREE, 2))
        
        # Starter tier has 5 symbols
        self.assertTrue(check_symbol_limit(SubscriptionTier.STARTER, 1))
        self.assertTrue(check_symbol_limit(SubscriptionTier.STARTER, 5))
        self.assertFalse(check_symbol_limit(SubscriptionTier.STARTER, 6))
        
        # Pro tier has 50 symbols
        self.assertTrue(check_symbol_limit(SubscriptionTier.PRO, 1))
        self.assertTrue(check_symbol_limit(SubscriptionTier.PRO, 50))
        self.assertFalse(check_symbol_limit(SubscriptionTier.PRO, 51))
        
        # Elite tier has unlimited symbols (represented as 0)
        self.assertTrue(check_symbol_limit(SubscriptionTier.ELITE, 1))
        self.assertTrue(check_symbol_limit(SubscriptionTier.ELITE, 100))
        self.assertTrue(check_symbol_limit(SubscriptionTier.ELITE, 1000))


if __name__ == '__main__':
    unittest.main()
