#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for subscription API routes.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import Depends

from app.api.routes import api_router
from app.services.subscription_service import SubscriptionService
from app.main import app
from app.utils.subscription_config import SubscriptionTier

@pytest.fixture
def test_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_subscription_service():
    """Create a mock subscription service."""
    service = MagicMock(spec=SubscriptionService)
    
    # Override the dependency
    def get_subscription_service_override():
        return service
    
    from app.api.routes import get_subscription_service
    app.dependency_overrides[get_subscription_service] = get_subscription_service_override
    
    yield service
    
    # Clean up
    app.dependency_overrides = {}


class TestSubscriptionRoutes:
    """Tests for subscription API routes."""
    
    def test_get_subscription_tiers(self, test_client, mock_subscription_service):
        """Test getting all subscription tiers."""
        # Mock the service method to match new API response format
        mock_subscription_service.get_all_tiers.return_value = [
            {
                "name": "Free",
                "display_name": "Free",
                "base_fee": 0.0,
                "profit_share": 0.0,
                "description": "Free plan",
                "max_strategies": 1,
                "max_portfolio": 5,
                "customization": False,
                "recommended": False
            },
            {
                "name": "Starter",
                "display_name": "Starter",
                "base_fee": 29.0,
                "profit_share": 0.0,
                "description": "Starter plan",
                "max_strategies": 3,
                "max_portfolio": 10,
                "customization": False,
                "recommended": False
            },
            {
                "name": "Pro",
                "display_name": "Pro",
                "base_fee": 99.0,
                "profit_share": 0.05,
                "description": "Pro plan",
                "max_strategies": 10,
                "max_portfolio": 30,
                "customization": True,
                "recommended": True
            },
            {
                "name": "Elite",
                "display_name": "Elite",
                "base_fee": 199.0,
                "profit_share": 0.02,
                "description": "Elite plan",
                "max_strategies": 100,
                "max_portfolio": 100,
                "customization": True,
                "recommended": False
            }
        ]
        # ... update assertions to expect 4 tiers
        response = test_client.get("/api/subscription/tiers")
        assert response.status_code == 200
        data = response.json()
        assert "tiers" in data
        assert isinstance(data["tiers"], list)
        assert len(data["tiers"]) == 4
        assert data["tiers"][0]["name"] == "Free"
        assert data["tiers"][1]["name"] == "Starter"
        assert data["tiers"][2]["name"] == "Pro"
        assert data["tiers"][3]["name"] == "Elite"
        # ... other field checks as needed


        # Check that the service method was called
        mock_subscription_service.get_all_tiers.assert_called_once()
    
    def test_get_tier_details(self, test_client, mock_subscription_service):
        """Test getting details for a specific tier."""
        # Mock the service method to match new API response format
        mock_subscription_service.get_tier_details.return_value = {
            "name": "Pro",
            "display_name": "Pro",
            "base_fee": 79.0,
            "profit_share": 0.1,
            "description": "Professional traders",
            "max_strategies": 10,
            "max_portfolio": 50,
            "customization": True,
            "recommended": True
        }

        # Make the request
        response = test_client.get("/api/subscription/tiers/PRO")

        # Check the response
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Pro"
        assert data["display_name"] == "Pro"
        assert data["base_fee"] == 79.0
        assert data["profit_share"] == 0.1
        assert data["description"] == "Professional traders"
        assert data["max_strategies"] == 10
        assert data["max_portfolio"] == 50
        assert data["customization"] is True
        assert data["recommended"] is True

        # Check that the service method was called with right tier
        mock_subscription_service.get_tier_details.assert_called_once_with(SubscriptionTier.PRO)
    
    def test_get_user_subscription(self, test_client, mock_subscription_service):
        """Test getting a user's subscription."""
        # Mock user ID from auth
        with patch('app.api.routes.get_current_user_id', return_value=1):
            # Mock the check_subscription_status method
            mock_subscription_service.check_subscription_status.return_value = {
                "has_subscription": True,
                "tier": "PRO",
                "display_tier": "Pro",
                "status": "active",
                "is_active": True,
                "is_trial": False,
                "days_remaining": 25
            }
            
            # Make the request
            response = test_client.get("/subscription/current")
            
            # Check the response
            assert response.status_code == 200
            data = response.json()
            assert data["has_subscription"] == True
            assert data["tier"] == "PRO"
            assert data["status"] == "active"
            assert data["days_remaining"] == 25
            
            # Check that the service method was called with right user ID
            mock_subscription_service.check_subscription_status.assert_called_once_with(1)
    
    def test_start_trial(self, test_client, mock_subscription_service):
        """Test starting a trial subscription."""
        # Mock user ID from auth
        with patch('app.api.routes.get_current_user_id', return_value=1):
            # Mock the start_trial method
            mock_subscription = MagicMock()
            mock_subscription.tier = SubscriptionTier.PRO
            mock_subscription.is_trial = True
            mock_subscription_service.start_trial.return_value = mock_subscription
            
            # Make the request
            response = test_client.post(
                "/subscription/trial",
                json={"tier": "PRO"}
            )
            
            # Check the response
            assert response.status_code == 200
            data = response.json()
            assert "subscription_id" in data
            assert data["tier"] == "PRO"
            assert data["is_trial"] == True
            
            # Check that the service method was called with right parameters
            mock_subscription_service.start_trial.assert_called_once_with(
                1, SubscriptionTier.PRO, days=14  # default trial days
            )
    
    def test_activate_subscription(self, test_client, mock_subscription_service):
        """Test activating a subscription."""
        # Mock user ID from auth
        with patch('app.api.routes.get_current_user_id', return_value=1):
            # Mock the activate_subscription method
            mock_subscription = MagicMock()
            mock_subscription.tier = SubscriptionTier.ELITE
            mock_subscription.status = "active"
            mock_subscription_service.activate_subscription.return_value = mock_subscription
            
            # Make the request
            response = test_client.post(
                "/subscription/activate",
                json={
                    "tier": "ELITE",
                    "payment_method_id": "pm_123456",
                    "external_subscription_id": "sub_123456"
                }
            )
            
            # Check the response
            assert response.status_code == 200
            data = response.json()
            assert "subscription_id" in data
            assert data["tier"] == "ELITE"
            assert data["status"] == "active"
            
            # Check that the service method was called with right parameters
            mock_subscription_service.activate_subscription.assert_called_once_with(
                1, 
                SubscriptionTier.ELITE,
                payment_method_id="pm_123456",
                external_subscription_id="sub_123456"
            )
    
    def test_cancel_subscription(self, test_client, mock_subscription_service):
        """Test cancelling a subscription."""
        # Mock user ID from auth
        with patch('app.api.routes.get_current_user_id', return_value=1):
            # Mock the cancel_subscription method
            mock_subscription = MagicMock()
            mock_subscription.status = "cancelled"
            mock_subscription_service.cancel_subscription.return_value = mock_subscription
            
            # Make the request
            response = test_client.post("/subscription/cancel")
            
            # Check the response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "cancelled"
            assert "message" in data
            
            # Check that the service method was called with right user ID
            mock_subscription_service.cancel_subscription.assert_called_once_with(1)
    
    def test_upgrade_subscription(self, test_client, mock_subscription_service):
        """Test upgrading a subscription."""
        # Mock user ID from auth
        with patch('app.api.routes.get_current_user_id', return_value=1):
            # Mock the upgrade_subscription method
            mock_subscription = MagicMock()
            mock_subscription.tier = SubscriptionTier.PRO
            mock_subscription.status = "active"
            mock_subscription.get_features.return_value = {"feature1": True, "feature2": False}
            mock_subscription_service.upgrade_subscription.return_value = mock_subscription
            
            # Make the request
            response = test_client.post(
                "/users/me/subscription/upgrade",
                json={
                    "tier": "PRO",
                    "payment_method_id": "pm_123456",
                    "prorate": True
                }
            )
            
            # Check the response
            assert response.status_code == 200
            data = response.json()
            assert data["tier"] == "PRO"
            assert data["status"] == "active"
            assert "features" in data
            
            # Check that the service method was called with right parameters
            mock_subscription_service.upgrade_subscription.assert_called_once_with(
                user_id=1, 
                new_tier=SubscriptionTier.PRO,
                payment_method_id="pm_123456",
                prorate=True
            )
    
    def test_get_payment_history(self, test_client, mock_subscription_service):
        """Test getting payment history."""
        # Mock user ID from auth
        with patch('app.api.routes.get_current_user_id', return_value=1):
            # Mock the get_payment_history and get_total_spent methods
            mock_payment1 = MagicMock()
            mock_payment1.id = 1
            mock_payment1.subscription_id = 100
            mock_payment1.amount = 49.99
            mock_payment1.payment_method = "credit_card"
            mock_payment1.status = "succeeded"
            mock_payment1.payment_date = "2025-04-01T00:00:00"
            mock_payment1.period_start = "2025-04-01T00:00:00"
            mock_payment1.period_end = "2025-05-01T00:00:00"
            
            mock_payment2 = MagicMock()
            mock_payment2.id = 2
            mock_payment2.subscription_id = 100
            mock_payment2.amount = 49.99
            mock_payment2.payment_method = "credit_card"
            mock_payment2.status = "succeeded"
            mock_payment2.payment_date = "2025-05-01T00:00:00"
            mock_payment2.period_start = "2025-05-01T00:00:00"
            mock_payment2.period_end = "2025-06-01T00:00:00"
            
            mock_subscription_service.get_payment_history.return_value = [mock_payment1, mock_payment2]
            mock_subscription_service.get_total_spent.return_value = 99.98
            
            # Make the request
            response = test_client.get("/users/me/subscription/payments")
            
            # Check the response
            assert response.status_code == 200
            data = response.json()
            assert "payments" in data
            assert len(data["payments"]) == 2
            assert data["total_spent"] == 99.98
            assert data["currency"] == "USD"
            
            # Verify first payment details
            payment1 = data["payments"][0]
            assert payment1["id"] == 1
            assert payment1["amount"] == 49.99
            assert payment1["status"] == "succeeded"
            
            # Check that the service methods were called with the right user ID
            mock_subscription_service.get_payment_history.assert_called_once_with(1)
            mock_subscription_service.get_total_spent.assert_called_once_with(1)
    
    def test_get_usage_metrics(self, test_client, mock_subscription_service):
        """Test getting usage metrics."""
        # Mock user ID from auth
        with patch('app.api.routes.get_current_user_id', return_value=1):
            # Mock the get_usage_metrics method
            mock_usage = {
                "metrics": [
                    {
                        "name": "strategies",
                        "display_name": "Strategy Slots",
                        "current_usage": 3,
                        "limit": 5,
                        "percentage_used": 60.0
                    },
                    {
                        "name": "symbols",
                        "display_name": "Watchlist Symbols",
                        "current_usage": 10,
                        "limit": 20,
                        "percentage_used": 50.0
                    }
                ],
                "billing_cycle_start": "2025-05-01T00:00:00",
                "billing_cycle_end": "2025-06-01T00:00:00",
                "days_left_in_cycle": 19
            }
            mock_subscription_service.get_usage_metrics.return_value = mock_usage
            
            # Make the request
            response = test_client.get("/users/me/subscription/usage")
            
            # Check the response
            assert response.status_code == 200
            data = response.json()
            assert "metrics" in data
            assert len(data["metrics"]) == 2
            assert data["days_left_in_cycle"] == 19
            
            # Verify usage metrics details
            strategies = data["metrics"][0]
            assert strategies["name"] == "strategies"
            assert strategies["current_usage"] == 3
            assert strategies["limit"] == 5
            assert strategies["percentage_used"] == 60.0
            
            # Check that the service method was called with the right user ID
            mock_subscription_service.get_usage_metrics.assert_called_once_with(1)
    
    def test_payment_webhook(self, test_client, mock_subscription_service):
        """Test payment webhook handler."""
        # Mock the handle_payment_webhook method
        mock_subscription_service.handle_payment_webhook.return_value = {
            "success": True,
            "payment_id": 123
        }
        
        # Make the request with a mock webhook payload
        webhook_payload = {
            "event_type": "payment_succeeded",
            "event_id": "evt_123456",
            "timestamp": "2025-05-13T14:30:00",
            "data": {
                "subscription_id": 100,
                "amount": 49.99,
                "payment_id": "pi_123456",
                "receipt_url": "https://example.com/receipt"
            }
        }
        
        response = test_client.post(
            "/webhooks/payment",
            json=webhook_payload
        )
        
        # Check the response
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["payment_id"] == 123
        
        # Check that the service method was called with the right parameters
        mock_subscription_service.handle_payment_webhook.assert_called_once_with(
            event_type="payment_succeeded",
            event_data=webhook_payload["data"]
        )
