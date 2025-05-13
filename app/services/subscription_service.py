#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Service for managing subscription operations in Mercurio Edge.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

from sqlalchemy.orm import Session

from app.db.models import User, Subscription, SubscriptionPayment, SubscriptionStatus
from app.utils.subscription_config import SubscriptionTier, get_tier_features, check_strategy_access, check_symbol_limit
from app.utils.subscription_helpers import get_tier_price, get_tier_name, get_tier_description


class SubscriptionService:
    """Service for managing user subscriptions."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_subscription(self, user_id: int) -> Optional[Subscription]:
        """
        Get the subscription for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            Subscription object or None if not found
        """
        return self.db.query(Subscription).filter(Subscription.user_id == user_id).first()
    
    def start_free_tier(self, user_id: int) -> Subscription:
        """
        Start a free tier subscription for a user.
        
        Args:
            user_id: ID of the user
            
        Returns:
            The created subscription
        """
        # Check if user already has a subscription
        existing = self.get_user_subscription(user_id)
        if existing:
            return existing
            
        # Create new free subscription
        subscription = Subscription(
            user_id=user_id,
            tier=SubscriptionTier.FREE,
            status=SubscriptionStatus.ACTIVE,
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
        
        self.db.add(subscription)
        self.db.commit()
        self.db.refresh(subscription)
        
        return subscription
    
    def start_trial(self, user_id: int, tier: SubscriptionTier, days: int = 7) -> Subscription:
        """
        Start a trial subscription for a user.
        
        Args:
            user_id: ID of the user
            tier: Subscription tier for the trial
            days: Number of days for the trial (default: 7)
            
        Returns:
            The created subscription
        """
        # Check if user already has a subscription
        existing = self.get_user_subscription(user_id)
        if existing and existing.is_active:
            # If existing subscription is active, don't start trial
            return existing
            
        # Set trial dates
        now = datetime.utcnow()
        trial_ends = now + timedelta(days=days)
        
        # Create or update subscription
        if existing:
            # Update existing subscription
            existing.tier = tier
            existing.status = SubscriptionStatus.TRIAL
            existing.is_trial = True
            existing.trial_started_at = now
            existing.trial_ends_at = trial_ends
            existing.current_period_start = now
            existing.current_period_end = trial_ends
            existing.updated_at = now
            
            subscription = existing
        else:
            # Create new subscription
            subscription = Subscription(
                user_id=user_id,
                tier=tier,
                status=SubscriptionStatus.TRIAL,
                is_trial=True,
                trial_started_at=now,
                trial_ends_at=trial_ends,
                current_period_start=now,
                current_period_end=trial_ends,
                created_at=now,
                updated_at=now
            )
            self.db.add(subscription)
            
        self.db.commit()
        self.db.refresh(subscription)
        
        return subscription
    
    def activate_subscription(
        self, 
        user_id: int, 
        tier: SubscriptionTier,
        payment_method_id: str,
        external_subscription_id: str
    ) -> Subscription:
        """
        Activate a paid subscription for a user.
        
        Args:
            user_id: ID of the user
            tier: Subscription tier
            payment_method_id: ID of the payment method
            external_subscription_id: Subscription ID from payment provider
            
        Returns:
            The activated subscription
        """
        # Check if user already has a subscription
        existing = self.get_user_subscription(user_id)
        now = datetime.utcnow()
        
        if existing:
            # Update existing subscription
            existing.tier = tier
            existing.status = SubscriptionStatus.ACTIVE
            existing.is_trial = False
            existing.payment_method_id = payment_method_id
            existing.external_subscription_id = external_subscription_id
            existing.updated_at = now
            
            # If subscription periods aren't set, set them now
            if not existing.current_period_start:
                existing.current_period_start = now
                existing.current_period_end = now + timedelta(days=30)
                
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
                current_period_start=now,
                current_period_end=now + timedelta(days=30),
                created_at=now,
                updated_at=now
            )
            self.db.add(subscription)
            
        self.db.commit()
        self.db.refresh(subscription)
        
        return subscription
    
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
    
    def cancel_subscription(self, user_id: int) -> Optional[Subscription]:
        """
        Cancel a user's subscription.
        
        Args:
            user_id: ID of the user
            
        Returns:
            The updated subscription or None if not found
        """
        subscription = self.get_user_subscription(user_id)
        if not subscription:
            return None
            
        subscription.status = SubscriptionStatus.CANCELLED
        subscription.updated_at = datetime.utcnow()
        
        self.db.commit()
        self.db.refresh(subscription)
        
        return subscription
        
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
            tier = SubscriptionTier.FREE
        else:
            tier = subscription.tier
            
        features = get_tier_features(tier)
        strategies = features.get('strategies', {})
        
        return strategies.get('allowed_types', [])
    
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
            tier = SubscriptionTier.FREE
        else:
            tier = subscription.tier
            
        features = get_tier_features(tier)
        market_data = features.get('market_data', {})
        
        return market_data.get('max_symbols', 0)
    
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
                'tier': 'FREE',
                'status': 'none',
                'is_active': False,
                'days_remaining': 0
            }
            
        # Check if trial has expired
        now = datetime.utcnow()
        if subscription.is_trial and subscription.trial_ends_at and subscription.trial_ends_at < now:
            # Trial expired, change to free tier
            subscription.is_trial = False
            subscription.tier = SubscriptionTier.FREE
            subscription.status = SubscriptionStatus.EXPIRED
            subscription.updated_at = now
            self.db.commit()
            self.db.refresh(subscription)
            
        # Check if subscription period has expired
        elif not subscription.is_trial and subscription.current_period_end and subscription.current_period_end < now:
            # Check if it's a paid subscription
            if subscription.tier != SubscriptionTier.FREE:
                # Mark as past due - would typically be handled by payment processor
                subscription.status = SubscriptionStatus.PAST_DUE
                subscription.updated_at = now
                self.db.commit()
                self.db.refresh(subscription)
            
        # Determine days remaining
        days_remaining = 0
        if subscription.is_trial and subscription.trial_ends_at:
            delta = subscription.trial_ends_at - now
            days_remaining = max(0, delta.days)
        elif subscription.current_period_end:
            delta = subscription.current_period_end - now
            days_remaining = max(0, delta.days)
            
        return {
            'has_subscription': True,
            'tier': subscription.tier.name,
            'display_tier': get_tier_name(subscription.tier),
            'status': subscription.status.value,
            'is_active': subscription.is_active,
            'is_trial': subscription.is_trial,
            'days_remaining': days_remaining,
            'renewal_date': subscription.current_period_end
        }
