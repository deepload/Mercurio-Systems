#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Service for managing subscription operations in Mercurio Edge.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Union

from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from app.db.models import User, Subscription, SubscriptionPayment, SubscriptionStatus
from app.utils.subscription_config import SubscriptionTier, get_tier_features, check_strategy_access, check_symbol_limit
from app.utils.subscription_helpers import get_tier_price, get_tier_name, get_tier_description
from app.services.stripe_service import StripeService

logger = logging.getLogger(__name__)


class SubscriptionService:
    """Service for managing user subscriptions."""
    
    def __init__(self, db: Session):
        """
        Initialize the subscription service.
        
        Args:
            db: Database session
        """
        self.db = db
        self.stripe_service = StripeService(db)
    
    def get_user_subscription(self, user_id: int) -> Optional[Subscription]:
        """
        Get the subscription for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Subscription object or None if not found
        """
        return self.db.query(Subscription).filter(Subscription.user_id == user_id).first()
    
    async def activate_subscription(
        self, 
        user_id: int, 
        tier: SubscriptionTier,
        payment_method_id: str,
        external_subscription_id: Optional[str] = None
    ) -> Subscription:
        """
        Activate a paid subscription for a user.
        
        Args:
            user_id: ID of the user
            tier: Subscription tier
            payment_method_id: ID of the payment method
            external_subscription_id: Optional subscription ID from payment provider
                                     (if already created externally)
            
        Returns:
            The activated subscription
        """
        try:
            # Get user
            user = self.db.query(User).get(user_id)
            if not user:
                raise ValueError(f"User with ID {user_id} not found")
                
            # Check if user already has a subscription
            existing = self.get_user_subscription(user_id)
            now = datetime.utcnow()
            
            # Create/get Stripe customer
            stripe_customer_id = await self.stripe_service.get_or_create_customer(user)
            
            # If external_subscription_id is not provided, create a new subscription in Stripe
            if not external_subscription_id:
                # Create subscription in Stripe
                stripe_subscription = await self.stripe_service.create_subscription(
                    customer_id=stripe_customer_id,
                    tier=tier,
                    payment_method_id=payment_method_id
                )
                
                external_subscription_id = stripe_subscription.get("id")
                current_period_start = stripe_subscription.get("current_period_start", now)
                current_period_end = stripe_subscription.get("current_period_end", now + timedelta(days=30))
            else:
                # Use default billing period
                current_period_start = now
                current_period_end = now + timedelta(days=30)
            
            if existing:
                # Update existing subscription
                existing.tier = tier
                existing.status = SubscriptionStatus.ACTIVE
                existing.is_trial = False
                existing.payment_method_id = payment_method_id
                existing.external_subscription_id = external_subscription_id
                existing.current_period_start = current_period_start
                existing.current_period_end = current_period_end
                existing.updated_at = now
                
                subscription = existing
            else:
                # Create new subscription
                subscription = Subscription(
                    user_id=user_id,
                    tier=tier,
                    status=SubscriptionStatus.ACTIVE,
                    is_trial=False,
                    payment_method_id=payment_method_id,
                    external_subscription_id=external_subscription_id,
                    current_period_start=current_period_start,
                    current_period_end=current_period_end,
                    created_at=now,
                    updated_at=now
                )
                self.db.add(subscription)
                
            self.db.commit()
            self.db.refresh(subscription)
            
            return subscription
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error activating subscription: {e}")
            raise ValueError(f"Failed to activate subscription: {str(e)}")
    
    async def process_payment(
        self,
        user_id: int,
        amount: float,
        payment_method_id: str,
        external_payment_id: Optional[str] = None
    ) -> SubscriptionPayment:
        """
        Process a payment for a user's subscription.
        
        Args:
            user_id: ID of the user
            amount: Payment amount
            payment_method_id: ID of the payment method
            external_payment_id: Optional payment ID from external provider
            
        Returns:
            The created payment record
        """
        try:
            # Get user and subscription
            user = self.db.query(User).get(user_id)
            if not user:
                raise ValueError(f"User with ID {user_id} not found")
                
            subscription = self.get_user_subscription(user_id)
            if not subscription:
                raise ValueError(f"No subscription found for user {user_id}")
                
            # Create payment record
            payment = self.record_payment(
                subscription_id=subscription.id,
                amount=amount,
                payment_method=payment_method_id,
                external_payment_id=external_payment_id or f"payment_{subscription.id}_{datetime.utcnow().timestamp()}",
                status="succeeded"
            )
            
            return payment
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error processing payment: {e}")
            raise ValueError(f"Failed to process payment: {str(e)}")
    
    def record_payment(
        self, 
        subscription_id: int,
        amount: float,
        external_payment_id: str,
        payment_method: str = "credit_card",
        status: str = "succeeded",
        receipt_url: Optional[str] = None
    ) -> SubscriptionPayment:
        """
        Record a subscription payment.
        
        Args:
            subscription_id: ID of the subscription
            amount: Payment amount
            external_payment_id: Payment ID from provider
            payment_method: Method of payment
            status: Payment status
            receipt_url: URL to payment receipt
            
        Returns:
            The created payment record
        """
        subscription = self.db.query(Subscription).get(subscription_id)
        if not subscription:
            raise ValueError(f"Subscription with ID {subscription_id} not found")
            
        # Calculate period
        now = datetime.utcnow()
        period_start = subscription.current_period_end or now
        period_end = period_start + timedelta(days=30)
        
        # Update subscription periods
        subscription.current_period_start = period_start
        subscription.current_period_end = period_end
        subscription.updated_at = now
        
        # Record payment
        payment = SubscriptionPayment(
            subscription_id=subscription_id,
            amount=amount,
            external_payment_id=external_payment_id,
            payment_method=payment_method,
            status=status,
            payment_date=now,
            period_start=period_start,
            period_end=period_end,
            receipt_url=receipt_url
        )
        
        self.db.add(payment)
        self.db.commit()
        self.db.refresh(payment)
        
        return payment
    
    async def cancel_subscription(self, user_id: int) -> Subscription:
        """
        Cancel a user's subscription.
        
        Args:
            user_id: ID of the user
            
        Returns:
            The updated subscription
        """
        try:
            # Get user
            user = self.db.query(User).get(user_id)
            if not user:
                raise ValueError(f"User with ID {user_id} not found")
                
            subscription = self.get_user_subscription(user_id)
            if not subscription:
                raise ValueError(f"No active subscription found for user {user_id}")
                
            # Can only cancel non-free tiers
            if subscription.tier == SubscriptionTier.FREE:
                raise ValueError("Cannot cancel free tier subscription")
                
            # Cancel subscription in Stripe if exists
            if subscription.external_subscription_id:
                stripe_result = await self.stripe_service.cancel_subscription(
                    subscription.external_subscription_id
                )
                
                # Update end date if provided by Stripe
                if stripe_result.get("canceled_at"):
                    # Set the end of billing period
                    subscription.canceled_at = stripe_result.get("canceled_at")
            
            # Update subscription status
            subscription.status = SubscriptionStatus.CANCELED
            subscription.updated_at = datetime.utcnow()
            
            self.db.commit()
            self.db.refresh(subscription)
            
            return subscription
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error canceling subscription: {e}")
            raise ValueError(f"Failed to cancel subscription: {str(e)}")
    
    def change_subscription_tier(self, user_id: int, new_tier: SubscriptionTier) -> Subscription:
        """
        Change a user's subscription tier.
        
        Args:
            user_id: ID of the user
            new_tier: New subscription tier
            
        Returns:
            The updated subscription
        """
        subscription = self.get_user_subscription(user_id)
        if not subscription:
            # If no subscription exists, create one with the new tier
            return self.start_free_tier(user_id)
            
        # Check if this is an upgrade or downgrade
        current_tier_order = list(SubscriptionTier).index(subscription.tier)
        new_tier_order = list(SubscriptionTier).index(new_tier)
        
        is_upgrade = new_tier_order > current_tier_order
        
        # Apply tier change
        subscription.tier = new_tier
        subscription.updated_at = datetime.utcnow()
        
        # If changing from free to paid tier, update status
        if subscription.tier != SubscriptionTier.FREE and new_tier != SubscriptionTier.FREE:
            subscription.status = SubscriptionStatus.ACTIVE
        
        # For downgrades, keep the same period
        # For upgrades, you might implement proration here
        
        self.db.commit()
        self.db.refresh(subscription)
        
        return subscription
    
    async def upgrade_subscription(
        self, 
        user_id: int, 
        new_tier: SubscriptionTier, 
        payment_method_id: Optional[str] = None,
        prorate: bool = True
    ) -> Subscription:
        """
        Upgrade or downgrade a user's subscription.
        
        Args:
            user_id: ID of the user
            new_tier: New subscription tier
            payment_method_id: ID of the payment method (required for upgrades from free tier)
            prorate: Whether to prorate charges for the current billing period
            
        Returns:
            The updated subscription
        """
        try:
            # Get user
            user = self.db.query(User).get(user_id)
            if not user:
                raise ValueError(f"User with ID {user_id} not found")
            
            # Get current subscription
            subscription = self.get_user_subscription(user_id)
            if not subscription:
                # If no subscription exists, start a free tier
                subscription = self.start_free_tier(user_id)
                
            # Check if this is an upgrade or downgrade
            current_tier_order = list(SubscriptionTier).index(subscription.tier)
            new_tier_order = list(SubscriptionTier).index(new_tier)
            
            is_upgrade = new_tier_order > current_tier_order
            from_free = subscription.tier == SubscriptionTier.FREE
            to_free = new_tier == SubscriptionTier.FREE
            
            now = datetime.utcnow()
            
            # For upgrade from free tier, we need a payment method
            if is_upgrade and from_free and not to_free and not payment_method_id:
                raise ValueError("Payment method is required when upgrading from free tier")
            
            # Get Stripe customer ID
            stripe_customer_id = await self.stripe_service.get_or_create_customer(user)
            
            # Handle tier change in Stripe if necessary
            if subscription.external_subscription_id and not to_free:
                # Update existing Stripe subscription
                stripe_result = await self.stripe_service.update_subscription(
                    stripe_subscription_id=subscription.external_subscription_id,
                    new_tier=new_tier,
                    prorate=prorate
                )
                
                # Update subscription details from Stripe
                if stripe_result.get("current_period_start"):
                    subscription.current_period_start = stripe_result.get("current_period_start")
                if stripe_result.get("current_period_end"):
                    subscription.current_period_end = stripe_result.get("current_period_end")
                
            # If upgrading from free tier to paid tier, need to create a new Stripe subscription
            elif from_free and not to_free:
                # Create a new subscription in Stripe
                stripe_result = await self.stripe_service.create_subscription(
                    customer_id=stripe_customer_id,
                    tier=new_tier,
                    payment_method_id=payment_method_id
                )
                
                # Update subscription with Stripe details
                subscription.external_subscription_id = stripe_result.get("id")
                if stripe_result.get("current_period_start"):
                    subscription.current_period_start = stripe_result.get("current_period_start")
                if stripe_result.get("current_period_end"):
                    subscription.current_period_end = stripe_result.get("current_period_end")
                
            # If downgrading to free tier, cancel existing Stripe subscription
            elif to_free and not from_free and subscription.external_subscription_id:
                # Cancel subscription in Stripe
                await self.stripe_service.cancel_subscription(subscription.external_subscription_id)
                subscription.external_subscription_id = None
            
            # Update local subscription data
            old_tier = subscription.tier
            subscription.tier = new_tier
            subscription.updated_at = now
            
            # Update status based on tier change
            if from_free and not to_free:
                subscription.status = SubscriptionStatus.ACTIVE
                subscription.payment_method_id = payment_method_id
            elif to_free and not from_free:
                subscription.status = SubscriptionStatus.ACTIVE  # Free tier is still active
                subscription.payment_method_id = None
            
            self.db.commit()
            self.db.refresh(subscription)
            
            return subscription
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error upgrading subscription: {e}")
            raise ValueError(f"Failed to upgrade subscription: {str(e)}")

    async def get_payment_history(self, user_id: int) -> List[SubscriptionPayment]:
        """
        Get payment history for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of payment records
        """
        try:
            # Get user and subscription
            user = self.db.query(User).get(user_id)
            if not user:
                raise ValueError(f"User with ID {user_id} not found")
                
            subscription = self.get_user_subscription(user_id)
            if not subscription:
                return []
                
            # For Stripe subscriptions, we might want to refresh payment data from Stripe
            # before returning it (for example in case of webhook failures)
            if subscription.external_subscription_id and user.stripe_customer_id:
                try:
                    # This would sync any missing payments from Stripe
                    # Currently not implemented - would require listing invoices in stripe_service
                    pass
                except Exception as e:
                    # Log error but continue with local data
                    logger.warning(f"Error syncing payment data from Stripe: {e}")
                
            # Get payments from database
            payments = self.db.query(SubscriptionPayment)\
                .filter(SubscriptionPayment.subscription_id == subscription.id)\
                .order_by(SubscriptionPayment.created_at.desc())\
                .all()
                
            return payments
            
        except Exception as e:
            logger.error(f"Error getting payment history: {e}")
            raise ValueError(f"Failed to get payment history: {str(e)}")
        
    def get_total_spent(self, user_id: int) -> float:
        """
        Calculate the total amount spent by a user on subscriptions.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Total amount spent
        """
        payments = self.get_payment_history(user_id)
        return sum(payment.amount for payment in payments if payment.status == "succeeded")
    
    async def get_usage_metrics(self, user_id: int) -> Dict[str, Any]:
        """
        Get usage metrics for a user's subscription.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with usage metrics
        """
        try:
            # Get user and subscription
            user = self.db.query(User).get(user_id)
            if not user:
                raise ValueError(f"User with ID {user_id} not found")
                
            subscription = self.get_user_subscription(user_id)
            
            # Default metrics for all tiers
            metrics = {
                "active_strategies": 0,  # Would be calculated from actual data
                "symbols_used": 0,  # Would be calculated from actual data
                "current_tier": {
                    "name": "Free",
                    "limits": get_tier_features(SubscriptionTier.FREE)
                },
                "usage_percentage": {},
                "billing_period": {}
            }
            
            if not subscription:
                return metrics
                
            # Get tier details
            tier_name = get_tier_name(subscription.tier)
            tier_features = get_tier_features(subscription.tier)
            
            # Add subscription status
            metrics["subscription_status"] = subscription.status.value
            
            # Add payment method info if available and not free tier
            if subscription.tier != SubscriptionTier.FREE and user.stripe_customer_id and subscription.payment_method_id:
                try:
                    # Get payment methods from Stripe
                    payment_methods = await self.stripe_service.get_payment_methods(user.stripe_customer_id)
                    for pm in payment_methods:
                        if pm["id"] == subscription.payment_method_id:
                            metrics["payment_method"] = {
                                "type": "card",
                                "brand": pm["brand"],
                                "last4": pm["last4"],
                                "exp_month": pm["exp_month"],
                                "exp_year": pm["exp_year"]
                            }
                            break
                except Exception as e:
                    logger.warning(f"Error getting payment method details: {e}")
            
            # Update with actual subscription data
            metrics["current_tier"] = {
                "name": tier_name,
                "limits": tier_features
            }
            
            # Set billing period info if available
            if subscription.current_period_start and subscription.current_period_end:
                metrics["billing_period"] = {
                    "start": subscription.current_period_start.isoformat(),
                    "end": subscription.current_period_end.isoformat(),
                    "days_remaining": max(0, (subscription.current_period_end - datetime.utcnow()).days)
                }
                
            # Calculate usage percentages (would use actual counts from database)
            # For demonstration, using mock values
            mock_strategies_count = 2
            mock_symbols_count = 5
            
            metrics["active_strategies"] = mock_strategies_count
            metrics["symbols_used"] = mock_symbols_count
            
            # Calculate percentages based on tier limits
            if tier_features.get("max_strategies"):
                metrics["usage_percentage"]["strategies"] = min(
                    100,
                    round(mock_strategies_count / tier_features["max_strategies"] * 100)
                )
            
            if tier_features.get("max_symbols"):
                metrics["usage_percentage"]["symbols"] = min(
                    100,
                    round(mock_symbols_count / tier_features["max_symbols"] * 100)
                )
                
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting usage metrics: {e}")
            raise ValueError(f"Failed to get usage metrics: {str(e)}")

        billing_cycle_start = subscription.current_period_start or datetime.utcnow()
        billing_cycle_end = subscription.current_period_end or (billing_cycle_start + timedelta(days=30))
        days_left = (billing_cycle_end - datetime.utcnow()).days
        
        return {
            "metrics": metrics,
            "billing_cycle_start": billing_cycle_start,
            "billing_cycle_end": billing_cycle_end,
            "days_left_in_cycle": max(0, days_left)
        }
    
    async def handle_webhook_event(self, payload: bytes, signature: str) -> Dict[str, Any]:
        """
        Handle webhook events from payment processor.
        
        Args:
            payload: Raw webhook payload
            signature: Webhook signature for verification
            
        Returns:
            Dictionary with result of processing the webhook
        """
        try:
            # Process the webhook through the Stripe service
            result = await self.stripe_service.handle_webhook_event(payload, signature)
            return result
        except Exception as e:
            logger.error(f"Error handling webhook event: {e}")
            raise ValueError(f"Failed to process webhook: {str(e)}")

    def get_all_tiers(self) -> List[Dict[str, Any]]:
        """
        Get information about all available subscription tiers.
        
        Returns:
            List of tier information dictionaries
        """
        tiers = []
        
        for tier in SubscriptionTier:
            tier_info = {
                'name': tier.name,
                'display_name': get_tier_name(tier),
                'description': get_tier_description(tier),
                'price': get_tier_price(tier),
                'features': get_tier_features(tier)
            }
            tiers.append(tier_info)
            
        return tiers
    
    def get_tier_details(self, tier: SubscriptionTier) -> Dict[str, Any]:
        """
        Get detailed information about a specific tier.
        
        Args:
            tier: The subscription tier
            
        Returns:
            Dictionary with tier information
        """
        return {
            'name': tier.name,
            'display_name': get_tier_name(tier),
            'description': get_tier_description(tier),
            'price': get_tier_price(tier),
            'features': get_tier_features(tier)
        }
        
    def check_feature_access(self, user_id: int, feature_path: str) -> bool:
        """
        Check if a user has access to a specific feature.
        
        Args:
            user_id: ID of the user
            feature_path: Dot-notation path to the feature
            
        Returns:
            True if the user has access, False otherwise
        """
        from app.utils.subscription_config import can_access_feature
        
        subscription = self.get_user_subscription(user_id)
        if not subscription or not subscription.is_active:
            # Default to free tier for inactive or missing subscriptions
            return can_access_feature(SubscriptionTier.FREE, feature_path)
            
        return can_access_feature(subscription.tier, feature_path)
    
    def check_strategy_access(self, user_id: int, strategy_type: str) -> bool:
        """
        Check if a user can access a specific strategy type.
        
        Args:
            user_id: ID of the user
            strategy_type: Type of strategy
            
        Returns:
            True if the user has access, False otherwise
        """
        subscription = self.get_user_subscription(user_id)
        if not subscription or not subscription.is_active:
            # Default to free tier for inactive or missing subscriptions
            return check_strategy_access(SubscriptionTier.FREE, strategy_type)
            
        return check_strategy_access(subscription.tier, strategy_type)
    
    def check_symbol_limit(self, user_id: int, symbol_count: int) -> bool:
        """
        Check if a user is within their symbol limit.
        
        Args:
            user_id: ID of the user
            symbol_count: Number of symbols to check
            
        Returns:
            True if within limit, False otherwise
        """
        subscription = self.get_user_subscription(user_id)
        if not subscription or not subscription.is_active:
            # Default to free tier for inactive or missing subscriptions
            return check_symbol_limit(SubscriptionTier.FREE, symbol_count)
            
        return check_symbol_limit(subscription.tier, symbol_count)
    
    def get_user_accessible_strategies(self, user_id: int) -> List[str]:
        """
        Get a list of strategy types the user can access.
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of accessible strategy types
        """
        subscription = self.get_user_subscription(user_id)
        if not subscription or not subscription.is_active:
            return 0
        return subscription.get_max_strategies()
    
    def get_symbol_limit(self, user_id: int) -> int:
        """
        Get the maximum number of symbols a user can track.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Maximum number of symbols
        """
        subscription = self.get_user_subscription(user_id)
        if not subscription or not subscription.is_active:
            return 0
        return subscription.get_max_portfolio()
    
    def get_tier_comparison(self) -> Dict[str, Dict]:
        """
        Get a comparison of all tiers for display purposes.
        
        Returns:
            Dictionary with tier comparison data
        """
        from app.utils.subscription_helpers import get_tier_comparison
        return get_tier_comparison()
    
    def check_subscription_status(self, user_id: int) -> Dict[str, Any]:
        """
        Check the status of a user's subscription and update if expired.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Dictionary with subscription status information
        """
        subscription = self.get_user_subscription(user_id)
        if not subscription:
            return {
                'has_subscription': False,
                'tier': None,
                'status': 'none',
                'is_active': False,
                'days_remaining': 0
            }
            
        # Check if subscription period has expired
        now = datetime.utcnow()
        if subscription.current_period_end and subscription.current_period_end < now:
            subscription.status = SubscriptionStatus.PAST_DUE
            subscription.updated_at = now
            self.db.commit()
            self.db.refresh(subscription)

        days_remaining = 0
        if subscription.current_period_end:
            delta = subscription.current_period_end - now
            days_remaining = max(0, delta.days)

        return {
            'has_subscription': True,
            'tier': subscription.tier.name,
            'display_tier': get_tier_name(subscription.tier),
            'status': subscription.status.value,
            'is_active': subscription.is_active,
            'days_remaining': days_remaining,
            'renewal_date': subscription.current_period_end
        }
