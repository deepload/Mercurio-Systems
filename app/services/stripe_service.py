#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stripe payment service for Mercurio's subscription system.
Handles Stripe customers, subscriptions, and payment processing.
"""

import os
import stripe
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from app.utils.subscription_config import SubscriptionTier, get_tier_features
from app.utils.subscription_helpers import get_tier_price, get_tier_name
from app.db.models import Subscription, User, SubscriptionStatus, SubscriptionPayment


# Configure Stripe with API key from environment variables
stripe.api_key = os.getenv("STRIPE_API_KEY", "")
stripe.api_version = "2022-11-15"  # Lock to a specific API version for stability

logger = logging.getLogger(__name__)


class StripeService:
    """Service for handling Stripe payment integration."""
    
    # Stripe product ID mapping for subscription tiers
    # In a real app, these would be stored in the database or fetched from Stripe
    TIER_PRODUCT_MAPPING = {
        SubscriptionTier.STARTER: "prod_starter",
        SubscriptionTier.PRO: "prod_pro", 
        SubscriptionTier.ELITE: "prod_elite"
    }
    
    def __init__(self, db):
        """
        Initialize the Stripe service.
        
        Args:
            db: Database session
        """
        self.db = db
        
        # Ensure Stripe API key is set
        if not stripe.api_key:
            logger.warning("Stripe API key not set. Payment functionality will be limited.")
    
    async def get_or_create_customer(self, user: User) -> str:
        """
        Get an existing Stripe customer ID or create a new one.
        
        Args:
            user: User model from database
            
        Returns:
            Stripe customer ID
        """
        if user.stripe_customer_id:
            return user.stripe_customer_id
            
        try:
            # Create a new Stripe customer
            customer = stripe.Customer.create(
                email=user.email,
                name=f"{user.first_name or ''} {user.last_name or ''}".strip() or None,
                metadata={
                    "user_id": str(user.id),
                    "username": user.username
                }
            )
            
            # Update user with the new Stripe customer ID
            user.stripe_customer_id = customer.id
            self.db.commit()
            
            return customer.id
            
        except stripe.error.StripeError as e:
            logger.error(f"Error creating Stripe customer: {e}")
            raise ValueError(f"Failed to create Stripe customer: {str(e)}")
    
    async def create_payment_method(self, customer_id: str, payment_token: str) -> str:
        """
        Attach a payment method to a customer.
        
        Args:
            customer_id: Stripe customer ID
            payment_token: Payment method token from client
            
        Returns:
            Payment method ID
        """
        try:
            # Attach the payment method to the customer
            payment_method = stripe.PaymentMethod.attach(
                payment_token,
                customer=customer_id
            )
            
            # Set as default payment method
            stripe.Customer.modify(
                customer_id,
                invoice_settings={
                    "default_payment_method": payment_method.id
                }
            )
            
            return payment_method.id
            
        except stripe.error.StripeError as e:
            logger.error(f"Error attaching payment method: {e}")
            raise ValueError(f"Failed to process payment method: {str(e)}")
    
    async def create_subscription(
        self, 
        customer_id: str, 
        tier: SubscriptionTier,
        payment_method_id: str,
        trial_days: int = 0
    ) -> Dict[str, Any]:
        """
        Create a new Stripe subscription.
        
        Args:
            customer_id: Stripe customer ID
            tier: Subscription tier
            payment_method_id: Payment method ID
            trial_days: Number of trial days (0 for no trial)
            
        Returns:
            Dictionary with subscription details
        """
        if tier == SubscriptionTier.FREE:
            # Free tier doesn't need a Stripe subscription
            return {
                "id": None,
                "status": "active",
                "current_period_start": datetime.utcnow(),
                "current_period_end": datetime.utcnow() + timedelta(days=365),  # Arbitrary future date
                "trial_end": None
            }
            
        try:
            # Get the product ID for this tier
            product_id = self.TIER_PRODUCT_MAPPING.get(tier)
            if not product_id:
                raise ValueError(f"No Stripe product configured for tier: {tier.name}")
                
            # Create the subscription
            subscription_args = {
                "customer": customer_id,
                "items": [{"price": self._get_price_id_for_tier(tier)}],
                "default_payment_method": payment_method_id,
                "expand": ["latest_invoice.payment_intent"],
                "metadata": {
                    "tier": tier.name
                }
            }
            
            # Add trial if specified
            if trial_days > 0:
                trial_end = int((datetime.utcnow() + timedelta(days=trial_days)).timestamp())
                subscription_args["trial_end"] = trial_end
            
            subscription = stripe.Subscription.create(**subscription_args)
            
            # Return standardized response
            return {
                "id": subscription.id,
                "status": subscription.status,
                "current_period_start": datetime.fromtimestamp(subscription.current_period_start),
                "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                "trial_end": datetime.fromtimestamp(subscription.trial_end) if subscription.trial_end else None
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Error creating subscription: {e}")
            raise ValueError(f"Failed to create subscription: {str(e)}")
    
    async def update_subscription(
        self, 
        stripe_subscription_id: str, 
        new_tier: SubscriptionTier,
        prorate: bool = True
    ) -> Dict[str, Any]:
        """
        Update an existing subscription to a new tier.
        
        Args:
            stripe_subscription_id: Stripe subscription ID
            new_tier: New subscription tier
            prorate: Whether to prorate charges
            
        Returns:
            Dictionary with updated subscription details
        """
        if not stripe_subscription_id:
            raise ValueError("No Stripe subscription ID provided")
            
        # Handle upgrade to/from free tier
        if new_tier == SubscriptionTier.FREE:
            # Cancel subscription and return free tier details
            await self.cancel_subscription(stripe_subscription_id)
            return {
                "id": None,
                "status": "active",
                "current_period_start": datetime.utcnow(),
                "current_period_end": datetime.utcnow() + timedelta(days=365),
                "trial_end": None
            }
            
        try:
            # Get current subscription to find the item ID
            current_sub = stripe.Subscription.retrieve(stripe_subscription_id)
            if not current_sub.items.data:
                raise ValueError(f"No items found in subscription {stripe_subscription_id}")
                
            subscription_item_id = current_sub.items.data[0].id
            
            # Update the subscription
            update_args = {
                "proration_behavior": "create_prorations" if prorate else "none",
                "items": [{
                    "id": subscription_item_id,
                    "price": self._get_price_id_for_tier(new_tier)
                }],
                "metadata": {
                    "tier": new_tier.name
                }
            }
            
            subscription = stripe.Subscription.modify(
                stripe_subscription_id, 
                **update_args
            )
            
            # Return standardized response
            return {
                "id": subscription.id,
                "status": subscription.status,
                "current_period_start": datetime.fromtimestamp(subscription.current_period_start),
                "current_period_end": datetime.fromtimestamp(subscription.current_period_end),
                "trial_end": datetime.fromtimestamp(subscription.trial_end) if subscription.trial_end else None
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Error updating subscription: {e}")
            raise ValueError(f"Failed to update subscription: {str(e)}")
    
    async def cancel_subscription(self, stripe_subscription_id: str) -> Dict[str, Any]:
        """
        Cancel a Stripe subscription.
        
        Args:
            stripe_subscription_id: Stripe subscription ID
            
        Returns:
            Dictionary with canceled subscription details
        """
        if not stripe_subscription_id:
            return {
                "status": "canceled",
                "canceled_at": datetime.utcnow()
            }
            
        try:
            subscription = stripe.Subscription.delete(
                stripe_subscription_id,
                prorate=True  # Provide prorated credit for unused time
            )
            
            return {
                "status": subscription.status,
                "canceled_at": datetime.fromtimestamp(subscription.canceled_at) if subscription.canceled_at else datetime.utcnow()
            }
            
        except stripe.error.StripeError as e:
            logger.error(f"Error canceling subscription: {e}")
            raise ValueError(f"Failed to cancel subscription: {str(e)}")
    
    async def handle_webhook_event(self, payload: bytes, signature: str) -> Dict[str, Any]:
        """
        Process a webhook event from Stripe.
        
        Args:
            payload: Raw webhook payload
            signature: Stripe signature from headers
            
        Returns:
            Dictionary with processing result
        """
        stripe_webhook_secret = os.getenv("STRIPE_WEBHOOK_SECRET", "")
        if not stripe_webhook_secret:
            logger.warning("Stripe webhook secret not configured")
            
        try:
            # Verify and construct the event
            event = stripe.Webhook.construct_event(
                payload=payload,
                sig_header=signature,
                secret=stripe_webhook_secret
            )
            
            # Process different event types
            event_type = event["type"]
            event_data = event["data"]["object"]
            
            if event_type == "invoice.payment_succeeded":
                return await self._handle_payment_succeeded(event_data)
            
            elif event_type == "invoice.payment_failed":
                return await self._handle_payment_failed(event_data)
                
            elif event_type == "customer.subscription.created":
                return await self._handle_subscription_created(event_data)
                
            elif event_type == "customer.subscription.updated":
                return await self._handle_subscription_updated(event_data)
                
            elif event_type == "customer.subscription.deleted":
                return await self._handle_subscription_deleted(event_data)
                
            else:
                logger.info(f"Unhandled webhook event type: {event_type}")
                return {"status": "ignored", "event_type": event_type}
                
        except stripe.error.SignatureVerificationError:
            logger.error("Invalid webhook signature")
            raise ValueError("Invalid webhook signature")
            
        except Exception as e:
            logger.error(f"Error handling webhook: {e}")
            raise ValueError(f"Webhook processing error: {str(e)}")
    
    async def get_payment_methods(self, customer_id: str) -> List[Dict[str, Any]]:
        """
        Get available payment methods for a customer.
        
        Args:
            customer_id: Stripe customer ID
            
        Returns:
            List of payment methods
        """
        try:
            payment_methods = stripe.PaymentMethod.list(
                customer=customer_id,
                type="card"
            )
            
            result = []
            for pm in payment_methods.data:
                card = pm.card
                result.append({
                    "id": pm.id,
                    "type": pm.type,
                    "brand": card.brand,
                    "last4": card.last4,
                    "exp_month": card.exp_month,
                    "exp_year": card.exp_year,
                    "is_default": pm.id == await self._get_default_payment_method(customer_id)
                })
                
            return result
            
        except stripe.error.StripeError as e:
            logger.error(f"Error getting payment methods: {e}")
            raise ValueError(f"Failed to retrieve payment methods: {str(e)}")
    
    # Internal helper methods
    
    def _get_price_id_for_tier(self, tier: SubscriptionTier) -> str:
        """
        Get the Stripe price ID for a subscription tier.
        
        Args:
            tier: Subscription tier
            
        Returns:
            Stripe price ID
        """
        # In a real app, these would come from the database or environment variables
        tier_price_mapping = {
            SubscriptionTier.STARTER: "price_starter_monthly",
            SubscriptionTier.PRO: "price_pro_monthly",
            SubscriptionTier.ELITE: "price_elite_monthly"
        }
        
        price_id = tier_price_mapping.get(tier)
        if not price_id:
            raise ValueError(f"No price configured for tier {tier.name}")
            
        return price_id
    
    async def _get_default_payment_method(self, customer_id: str) -> Optional[str]:
        """
        Get the default payment method ID for a customer.
        
        Args:
            customer_id: Stripe customer ID
            
        Returns:
            Default payment method ID or None
        """
        try:
            customer = stripe.Customer.retrieve(customer_id)
            return customer.invoice_settings.default_payment_method
        except Exception:
            return None
    
    async def _handle_payment_succeeded(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a successful payment event.
        
        Args:
            event_data: Stripe event data
            
        Returns:
            Processing result
        """
        invoice = event_data
        subscription_id = invoice.get("subscription")
        customer_id = invoice.get("customer")
        
        if not subscription_id:
            return {"status": "ignored", "reason": "No subscription ID"}
            
        try:
            # Find the subscription in our database
            subscription = self.db.query(Subscription)\
                .filter(Subscription.external_subscription_id == subscription_id)\
                .first()
                
            if not subscription:
                logger.warning(f"Subscription {subscription_id} not found in database")
                return {"status": "error", "reason": "Subscription not found"}
                
            # Record the payment
            payment = SubscriptionPayment(
                subscription_id=subscription.id,
                amount=invoice.get("amount_paid", 0) / 100.0,  # Convert from cents
                external_payment_id=invoice.get("id"),
                payment_method="card",  # Default to card
                status="succeeded",
                payment_date=datetime.utcnow(),
                period_start=datetime.fromtimestamp(invoice.get("period_start", 0)),
                period_end=datetime.fromtimestamp(invoice.get("period_end", 0)),
                receipt_url=invoice.get("hosted_invoice_url"),
                extra_data={
                    "invoice_id": invoice.get("id"),
                    "invoice_number": invoice.get("number"),
                    "invoice_pdf": invoice.get("invoice_pdf")
                }
            )
            
            self.db.add(payment)
            
            # Update subscription status if needed
            if subscription.status != SubscriptionStatus.ACTIVE:
                subscription.status = SubscriptionStatus.ACTIVE
                
            # Update subscription dates
            if invoice.get("period_start") and invoice.get("period_end"):
                subscription.current_period_start = datetime.fromtimestamp(invoice.get("period_start"))
                subscription.current_period_end = datetime.fromtimestamp(invoice.get("period_end"))
                
            self.db.commit()
            
            return {
                "status": "success",
                "payment_id": payment.id,
                "subscription_id": subscription.id
            }
            
        except Exception as e:
            logger.error(f"Error processing payment success: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_payment_failed(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a failed payment event.
        
        Args:
            event_data: Stripe event data
            
        Returns:
            Processing result
        """
        invoice = event_data
        subscription_id = invoice.get("subscription")
        customer_id = invoice.get("customer")
        
        if not subscription_id:
            return {"status": "ignored", "reason": "No subscription ID"}
            
        try:
            # Find the subscription in our database
            subscription = self.db.query(Subscription)\
                .filter(Subscription.external_subscription_id == subscription_id)\
                .first()
                
            if not subscription:
                logger.warning(f"Subscription {subscription_id} not found in database")
                return {"status": "error", "reason": "Subscription not found"}
                
            # Update subscription status
            subscription.status = SubscriptionStatus.PAST_DUE
            subscription.updated_at = datetime.utcnow()
            
            # Record the failed payment
            payment = SubscriptionPayment(
                subscription_id=subscription.id,
                amount=invoice.get("amount_due", 0) / 100.0,  # Convert from cents
                external_payment_id=invoice.get("id"),
                payment_method="card",  # Default to card
                status="failed",
                payment_date=datetime.utcnow(),
                period_start=datetime.fromtimestamp(invoice.get("period_start", 0)) if invoice.get("period_start") else None,
                period_end=datetime.fromtimestamp(invoice.get("period_end", 0)) if invoice.get("period_end") else None,
                receipt_url=invoice.get("hosted_invoice_url"),
                extra_data={
                    "invoice_id": invoice.get("id"),
                    "failure_reason": invoice.get("last_payment_error", {}).get("message", "Unknown")
                }
            )
            
            self.db.add(payment)
            self.db.commit()
            
            return {
                "status": "handled",
                "payment_id": payment.id,
                "subscription_id": subscription.id
            }
            
        except Exception as e:
            logger.error(f"Error processing payment failure: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_subscription_created(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a subscription created event.
        
        Args:
            event_data: Stripe event data
            
        Returns:
            Processing result
        """
        # This is usually handled by our own create_subscription method
        # But we'll implement it for completeness and to handle external creations
        stripe_subscription = event_data
        customer_id = stripe_subscription.get("customer")
        
        try:
            # Find the user by Stripe customer ID
            user = self.db.query(User).filter(User.stripe_customer_id == customer_id).first()
            if not user:
                logger.warning(f"User with Stripe customer ID {customer_id} not found")
                return {"status": "error", "reason": "User not found"}
                
            # Check if we already have this subscription in our database
            existing = self.db.query(Subscription)\
                .filter(Subscription.external_subscription_id == stripe_subscription.get("id"))\
                .first()
                
            if existing:
                # Already processed
                return {"status": "already_processed", "subscription_id": existing.id}
                
            # Determine tier from metadata or product
            tier_name = stripe_subscription.get("metadata", {}).get("tier")
            if tier_name:
                try:
                    tier = SubscriptionTier[tier_name]
                except KeyError:
                    tier = SubscriptionTier.STARTER  # Default
            else:
                # Try to determine from the price/product
                tier = SubscriptionTier.STARTER  # Default
            
            # Create subscription in our database
            subscription = Subscription(
                user_id=user.id,
                tier=tier,
                status=SubscriptionStatus(stripe_subscription.get("status", "active")),
                is_trial=stripe_subscription.get("trial_end") is not None,
                trial_started_at=datetime.fromtimestamp(stripe_subscription.get("trial_start")) if stripe_subscription.get("trial_start") else None,
                trial_ends_at=datetime.fromtimestamp(stripe_subscription.get("trial_end")) if stripe_subscription.get("trial_end") else None,
                payment_method_id=stripe_subscription.get("default_payment_method"),
                external_subscription_id=stripe_subscription.get("id"),
                current_period_start=datetime.fromtimestamp(stripe_subscription.get("current_period_start")),
                current_period_end=datetime.fromtimestamp(stripe_subscription.get("current_period_end")),
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            self.db.add(subscription)
            self.db.commit()
            
            return {
                "status": "created",
                "subscription_id": subscription.id
            }
            
        except Exception as e:
            logger.error(f"Error processing subscription creation: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_subscription_updated(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a subscription updated event.
        
        Args:
            event_data: Stripe event data
            
        Returns:
            Processing result
        """
        stripe_subscription = event_data
        subscription_id = stripe_subscription.get("id")
        
        try:
            # Find the subscription in our database
            subscription = self.db.query(Subscription)\
                .filter(Subscription.external_subscription_id == subscription_id)\
                .first()
                
            if not subscription:
                logger.warning(f"Subscription {subscription_id} not found in database")
                return {"status": "error", "reason": "Subscription not found"}
                
            # Update subscription status
            new_status = SubscriptionStatus.ACTIVE
            if stripe_subscription.get("status") == "canceled":
                new_status = SubscriptionStatus.CANCELLED
            elif stripe_subscription.get("status") == "unpaid":
                new_status = SubscriptionStatus.PAST_DUE
            elif stripe_subscription.get("status") == "trialing":
                new_status = SubscriptionStatus.TRIAL
                
            subscription.status = new_status
            
            # Update tier if it changed
            tier_name = stripe_subscription.get("metadata", {}).get("tier")
            if tier_name:
                try:
                    subscription.tier = SubscriptionTier[tier_name]
                except KeyError:
                    pass  # Keep existing tier
                    
            # Update billing period
            if stripe_subscription.get("current_period_start"):
                subscription.current_period_start = datetime.fromtimestamp(stripe_subscription.get("current_period_start"))
            if stripe_subscription.get("current_period_end"):
                subscription.current_period_end = datetime.fromtimestamp(stripe_subscription.get("current_period_end"))
                
            # Update trial information
            subscription.is_trial = stripe_subscription.get("trial_end") is not None
            if stripe_subscription.get("trial_start"):
                subscription.trial_started_at = datetime.fromtimestamp(stripe_subscription.get("trial_start"))
            if stripe_subscription.get("trial_end"):
                subscription.trial_ends_at = datetime.fromtimestamp(stripe_subscription.get("trial_end"))
                
            subscription.updated_at = datetime.utcnow()
            self.db.commit()
            
            return {
                "status": "updated",
                "subscription_id": subscription.id
            }
            
        except Exception as e:
            logger.error(f"Error processing subscription update: {e}")
            return {"status": "error", "message": str(e)}
    
    async def _handle_subscription_deleted(self, event_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a subscription deleted event.
        
        Args:
            event_data: Stripe event data
            
        Returns:
            Processing result
        """
        stripe_subscription = event_data
        subscription_id = stripe_subscription.get("id")
        
        try:
            # Find the subscription in our database
            subscription = self.db.query(Subscription)\
                .filter(Subscription.external_subscription_id == subscription_id)\
                .first()
                
            if not subscription:
                logger.warning(f"Subscription {subscription_id} not found in database")
                return {"status": "error", "reason": "Subscription not found"}
                
            # Update subscription status
            subscription.status = SubscriptionStatus.CANCELLED
            subscription.updated_at = datetime.utcnow()
            
            self.db.commit()
            
            return {
                "status": "cancelled",
                "subscription_id": subscription.id
            }
            
        except Exception as e:
            logger.error(f"Error processing subscription deletion: {e}")
            return {"status": "error", "message": str(e)}
