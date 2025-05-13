#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for subscription API routes.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from fastapi import Depends

from app.api.routes import app, api_router
from app.services.subscription_service import SubscriptionService
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
    
    app.dependency_overrides[Depends] = get_subscription_service_override
    
    yield service
    
    # Clean up
    app.dependency_overrides = {}


class TestSubscriptionRoutes:
    """Tests for subscription API routes."""
    
    def test_get_subscription_tiers(self, test_client, mock_subscription_service):
        """Test getting all subscription tiers."""
        # Mock the service method
        mock_subscription_service.get_all_tiers.return_value = [
            {
                "tier": "FREE",
                "name": "Free",
                "description": "Basic access",
                "price_monthly": 0.0,
                "features": ["Paper trading", "1 strategy"]
            },
            {
                "tier": "STARTER",
                "name": "Starter",
                "description": "Getting started",
                "price_monthly": 19.99,
                "features": ["Live trading", "5 strategies"]
            }
        ]
        
        # Make the request
        response = test_client.get("/api/subscription/tiers")
        
        # Check the response
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["tier"] == "FREE"
        assert data[1]["tier"] == "STARTER"
        assert data[1]["price_monthly"] == 19.99
        
        # Check that the service method was called
        mock_subscription_service.get_all_tiers.assert_called_once()
    
    def test_get_tier_details(self, test_client, mock_subscription_service):
        """Test getting details for a specific tier."""
        # Mock the service method
        mock_subscription_service.get_tier_details.return_value = {
            "tier": "PRO",
            "name": "Pro",
            "description": "Professional traders",
            "price_monthly": 49.99,
            "features": ["Live trading", "Advanced strategies", "Portfolio analytics"]
        }
        
        # Make the request
        response = test_client.get("/api/subscription/tiers/PRO")
        
        # Check the response
        assert response.status_code == 200
        data = response.json()
        assert data["tier"] == "PRO"
        assert data["price_monthly"] == 49.99
        
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
            response = test_client.get("/api/subscription/current")
            
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
                "/api/subscription/trial",
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
                "/api/subscription/activate",
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
            response = test_client.post("/api/subscription/cancel")
            
            # Check the response
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "cancelled"
            assert "message" in data
            
            # Check that the service method was called with right user ID
            mock_subscription_service.cancel_subscription.assert_called_once_with(1)
