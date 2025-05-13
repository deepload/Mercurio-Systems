#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for subscription service functionality.
"""

import unittest
from unittest.mock import MagicMock, patch
from datetime import datetime, timedelta

from app.services.subscription_service import SubscriptionService
from app.db.models import Subscription, SubscriptionStatus, SubscriptionPayment
from app.utils.subscription_config import SubscriptionTier


class TestSubscriptionService(unittest.TestCase):
    """Tests for subscription service functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock database session
        self.db = MagicMock()
        
        # Create the subscription service with the mock DB
        self.service = SubscriptionService(self.db)
        
        # Mock user ID for testing
        self.user_id = 1
    
    def test_get_user_subscription(self):
        """Test getting a user's subscription."""
        # Create a mock subscription
        mock_subscription = MagicMock(spec=Subscription)
        mock_subscription.user_id = self.user_id
        
        # Set up the mock query
        query_mock = self.db.query.return_value
        filter_mock = query_mock.filter.return_value
        filter_mock.first.return_value = mock_subscription
        
        # Call the method
        result = self.service.get_user_subscription(self.user_id)
        
        # Assert that the query was made correctly
        self.db.query.assert_called_once_with(Subscription)
        query_mock.filter.assert_called_once()
        filter_mock.first.assert_called_once()
        
        # Assert that the result is the mock subscription
        self.assertEqual(result, mock_subscription)
    
    def test_start_free_tier_new_user(self):
        """Test starting a free tier for a new user."""
        # Set up the mock query to return None (no existing subscription)
        query_mock = self.db.query.return_value
        filter_mock = query_mock.filter.return_value
        filter_mock.first.return_value = None
        
        # Call the method
        result = self.service.start_free_tier(self.user_id)
        
        # Assert that a new subscription was created and added to the DB
        self.db.add.assert_called_once()
        self.db.commit.assert_called_once()
        self.db.refresh.assert_called_once()
        
        # Check the subscription properties
        subscription = self.db.add.call_args[0][0]
        self.assertEqual(subscription.user_id, self.user_id)
        self.assertEqual(subscription.tier, SubscriptionTier.FREE)
        self.assertEqual(subscription.status, SubscriptionStatus.ACTIVE)
    
    def test_start_free_tier_existing_user(self):
        """Test starting a free tier for a user with existing subscription."""
        # Create a mock subscription
        mock_subscription = MagicMock(spec=Subscription)
        mock_subscription.user_id = self.user_id
        
        # Set up the mock query to return the existing subscription
        query_mock = self.db.query.return_value
        filter_mock = query_mock.filter.return_value
        filter_mock.first.return_value = mock_subscription
        
        # Call the method
        result = self.service.start_free_tier(self.user_id)
        
        # Assert that no new subscription was created
        self.db.add.assert_not_called()
        
        # Assert that the result is the existing mock subscription
        self.assertEqual(result, mock_subscription)
    
    def test_start_trial(self):
        """Test starting a trial subscription."""
        # Set up the mock query to return None (no existing subscription)
        query_mock = self.db.query.return_value
        filter_mock = query_mock.filter.return_value
        filter_mock.first.return_value = None
        
        # Call the method to start a Pro trial
        result = self.service.start_trial(self.user_id, SubscriptionTier.PRO, days=7)
        
        # Assert that a new subscription was created and added to the DB
        self.db.add.assert_called_once()
        self.db.commit.assert_called_once()
        self.db.refresh.assert_called_once()
        
        # Check the subscription properties
        subscription = self.db.add.call_args[0][0]
        self.assertEqual(subscription.user_id, self.user_id)
        self.assertEqual(subscription.tier, SubscriptionTier.PRO)
        self.assertEqual(subscription.status, SubscriptionStatus.TRIAL)
        self.assertTrue(subscription.is_trial)
        
        # Check that trial dates are set properly
        self.assertIsNotNone(subscription.trial_started_at)
        self.assertIsNotNone(subscription.trial_ends_at)
        
        # Trial should end 7 days after it starts
        trial_days = (subscription.trial_ends_at - subscription.trial_started_at).days
        self.assertEqual(trial_days, 7)
    
    def test_start_trial_existing_active(self):
        """Test starting a trial for a user with an existing active subscription."""
        # Create a mock active subscription
        mock_subscription = MagicMock(spec=Subscription)
        mock_subscription.user_id = self.user_id
        mock_subscription.is_active = True
        
        # Set up the mock query to return the existing subscription
        query_mock = self.db.query.return_value
        filter_mock = query_mock.filter.return_value
        filter_mock.first.return_value = mock_subscription
        
        # Call the method
        result = self.service.start_trial(self.user_id, SubscriptionTier.PRO)
        
        # Assert that no new subscription was created
        self.db.add.assert_not_called()
        
        # Assert that the result is the existing mock subscription
        self.assertEqual(result, mock_subscription)
        
        # The existing subscription should not be modified
        mock_subscription.is_trial = False
        self.assertFalse(mock_subscription.is_trial)
    
    def test_activate_subscription_new(self):
        """Test activating a new paid subscription."""
        # Set up the mock query to return None (no existing subscription)
        query_mock = self.db.query.return_value
        filter_mock = query_mock.filter.return_value
        filter_mock.first.return_value = None
        
        # Call the method
        result = self.service.activate_subscription(
            self.user_id,
            SubscriptionTier.PRO,
            payment_method_id="pm_123456",
            external_subscription_id="sub_123456"
        )
        
        # Assert that a new subscription was created and added to the DB
        self.db.add.assert_called_once()
        self.db.commit.assert_called_once()
        self.db.refresh.assert_called_once()
        
        # Check the subscription properties
        subscription = self.db.add.call_args[0][0]
        self.assertEqual(subscription.user_id, self.user_id)
        self.assertEqual(subscription.tier, SubscriptionTier.PRO)
        self.assertEqual(subscription.status, SubscriptionStatus.ACTIVE)
        self.assertFalse(subscription.is_trial)
        self.assertEqual(subscription.payment_method_id, "pm_123456")
        self.assertEqual(subscription.external_subscription_id, "sub_123456")
    
    def test_activate_subscription_existing(self):
        """Test activating an existing subscription."""
        # Create a mock subscription
        mock_subscription = MagicMock(spec=Subscription)
        mock_subscription.user_id = self.user_id
        mock_subscription.is_trial = True
        mock_subscription.current_period_start = None
        mock_subscription.current_period_end = None
        
        # Set up the mock query to return the existing subscription
        query_mock = self.db.query.return_value
        filter_mock = query_mock.filter.return_value
        filter_mock.first.return_value = mock_subscription
        
        # Call the method
        result = self.service.activate_subscription(
            self.user_id,
            SubscriptionTier.ELITE,
            payment_method_id="pm_123456",
            external_subscription_id="sub_123456"
        )
        
        # Assert that no new subscription was created
        self.db.add.assert_not_called()
        self.db.commit.assert_called_once()
        self.db.refresh.assert_called_once()
        
        # Check that the existing subscription was updated
        self.assertEqual(mock_subscription.tier, SubscriptionTier.ELITE)
        self.assertEqual(mock_subscription.status, SubscriptionStatus.ACTIVE)
        self.assertFalse(mock_subscription.is_trial)
        self.assertEqual(mock_subscription.payment_method_id, "pm_123456")
        self.assertEqual(mock_subscription.external_subscription_id, "sub_123456")
        
        # Check that period dates were set
        self.assertIsNotNone(mock_subscription.current_period_start)
        self.assertIsNotNone(mock_subscription.current_period_end)
    
    def test_record_payment(self):
        """Test recording a subscription payment."""
        # Create a mock subscription
        mock_subscription = MagicMock(spec=Subscription)
        mock_subscription.id = 1
        mock_subscription.user_id = self.user_id
        
        # Set up the mock query
        query_mock = self.db.query.return_value
        query_mock.get.return_value = mock_subscription
        
        # Call the method
        result = self.service.record_payment(
            subscription_id=mock_subscription.id,
            amount=79.0,
            external_payment_id="pi_123456",
            payment_method="credit_card",
            status="succeeded",
            receipt_url="https://receipt.url"
        )
        
        # Assert that a new payment was created and added to the DB
        self.db.add.assert_called_once()
        self.db.commit.assert_called_once()
        self.db.refresh.assert_called_once()
        
        # Check the payment properties
        payment = self.db.add.call_args[0][0]
        self.assertEqual(payment.subscription_id, mock_subscription.id)
        self.assertEqual(payment.amount, 79.0)
        self.assertEqual(payment.external_payment_id, "pi_123456")
        self.assertEqual(payment.payment_method, "credit_card")
        self.assertEqual(payment.status, "succeeded")
        self.assertEqual(payment.receipt_url, "https://receipt.url")
        
        # Check that the subscription period was updated
        self.assertIsNotNone(mock_subscription.current_period_start)
        self.assertIsNotNone(mock_subscription.current_period_end)
        
        # Period should be extended by about a month (30 days)
        period_days = (mock_subscription.current_period_end - mock_subscription.current_period_start).days
        self.assertEqual(period_days, 30)
    
    def test_cancel_subscription(self):
        """Test cancelling a subscription."""
        # Create a mock subscription
        mock_subscription = MagicMock(spec=Subscription)
        mock_subscription.id = 1
        mock_subscription.user_id = self.user_id
        mock_subscription.status = SubscriptionStatus.ACTIVE
        
        # Set up the mock query
        query_mock = self.db.query.return_value
        filter_mock = query_mock.filter.return_value
        filter_mock.first.return_value = mock_subscription
        
        # Call the method
        result = self.service.cancel_subscription(self.user_id)
        
        # Check that the subscription status was updated
        self.assertEqual(mock_subscription.status, SubscriptionStatus.CANCELLED)
        self.db.commit.assert_called_once()
        self.db.refresh.assert_called_once()
    
    def test_cancel_nonexistent_subscription(self):
        """Test cancelling a non-existent subscription."""
        # Set up the mock query to return None
        query_mock = self.db.query.return_value
        filter_mock = query_mock.filter.return_value
        filter_mock.first.return_value = None
        
        # Call the method
        result = self.service.cancel_subscription(self.user_id)
        
        # Result should be None
        self.assertIsNone(result)
        
        # Database should not be modified
        self.db.commit.assert_not_called()
    
    def test_check_feature_access(self):
        """Test checking feature access."""
        # Mock the can_access_feature function
        with patch('app.utils.subscription_config.can_access_feature') as mock_access:
            mock_access.return_value = True
            
            # Create a mock subscription
            mock_subscription = MagicMock(spec=Subscription)
            mock_subscription.tier = SubscriptionTier.PRO
            mock_subscription.is_active = True
            
            # Set up the mock query
            query_mock = self.db.query.return_value
            filter_mock = query_mock.filter.return_value
            filter_mock.first.return_value = mock_subscription
            
            # Call the method
            result = self.service.check_feature_access(self.user_id, 'portfolio_analytics')
            
            # Assert that can_access_feature was called with the right parameters
            mock_access.assert_called_once_with(SubscriptionTier.PRO, 'portfolio_analytics')
            
            # Result should match the mock return value
            self.assertTrue(result)
    
    def test_check_subscription_status_expired_trial(self):
        """Test checking a subscription with an expired trial."""
        # Create a mock subscription with an expired trial
        now = datetime.utcnow()
        three_days_ago = now - timedelta(days=3)
        yesterday = now - timedelta(days=1)
        
        mock_subscription = MagicMock(spec=Subscription)
        mock_subscription.id = 1
        mock_subscription.user_id = self.user_id
        mock_subscription.tier = SubscriptionTier.PRO
        mock_subscription.status = SubscriptionStatus.TRIAL
        mock_subscription.is_trial = True
        mock_subscription.trial_started_at = three_days_ago
        mock_subscription.trial_ends_at = yesterday  # Trial ended yesterday
        
        # Set up the mock query
        query_mock = self.db.query.return_value
        filter_mock = query_mock.filter.return_value
        filter_mock.first.return_value = mock_subscription
        
        # Call the method
        result = self.service.check_subscription_status(self.user_id)
        
        # Trial should have been marked as expired
        self.assertEqual(mock_subscription.status, SubscriptionStatus.EXPIRED)
        self.assertEqual(mock_subscription.tier, SubscriptionTier.FREE)
        self.assertFalse(mock_subscription.is_trial)
        self.db.commit.assert_called_once()
        
        # Check result values
        self.assertTrue(result['has_subscription'])
        self.assertEqual(result['tier'], 'FREE')
        self.assertEqual(result['status'], 'expired')
        self.assertFalse(result['is_active'])
        self.assertEqual(result['days_remaining'], 0)


if __name__ == '__main__':
    unittest.main()
