"""
Database models for Mercurio AI platform
"""
import enum
import json
from datetime import datetime
from typing import Optional, List, Dict, Any

from sqlalchemy import Column, Integer, String, Float, DateTime, Enum, ForeignKey, Text, JSON
from sqlalchemy import Boolean, Table
from sqlalchemy.orm import relationship

from app.db.database import Base
from app.utils.subscription_config import SubscriptionTier

class TradeAction(enum.Enum):
    """Enum for trade actions"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"

class Trade(Base):
    """Model for storing trade records"""
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String(10), index=True, nullable=False)
    strategy = Column(String(50), index=True, nullable=False)
    action = Column(Enum(TradeAction), nullable=False)
    price = Column(Float, nullable=False)
    quantity = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    
    # Additional metadata
    confidence = Column(Float)
    model_id = Column(Integer, ForeignKey("ai_models.id"), nullable=True)
    trade_metadata = Column(JSON, nullable=True)
    
    # Relationships
    model = relationship("AIModel", back_populates="trades")
    
    def __repr__(self):
        return f"<Trade(id={self.id}, symbol={self.symbol}, action={self.action})>"

class BacktestResult(Base):
    """Model for storing backtest results"""
    __tablename__ = "backtest_results"
    
    id = Column(Integer, primary_key=True, index=True)
    strategy = Column(String(50), index=True, nullable=False)
    symbol = Column(String(10), index=True, nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Performance metrics
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    total_return = Column(Float, nullable=False)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    
    # Run metadata
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    parameters = Column(JSON, nullable=True)
    
    # Relationships
    model_id = Column(Integer, ForeignKey("ai_models.id"), nullable=True)
    model = relationship("AIModel", back_populates="backtest_results")
    
    def __repr__(self):
        return f"<BacktestResult(id={self.id}, strategy={self.strategy}, return={self.total_return})>"

class AIModel(Base):
    """Model for storing trained AI models metadata"""
    __tablename__ = "ai_models"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    strategy = Column(String(50), index=True, nullable=False)
    model_type = Column(String(50), nullable=False)  # e.g., "RandomForest", "LSTM"
    symbols = Column(JSON, nullable=False)  # List of symbols this model was trained on
    
    # Training metadata
    train_start_date = Column(DateTime, nullable=False)
    train_end_date = Column(DateTime, nullable=False)
    
    # File paths and parameters
    model_path = Column(String(255), nullable=False)
    parameters = Column(JSON, nullable=True)
    metrics = Column(JSON, nullable=True)  # Training and validation metrics
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    
    # Relationships
    trades = relationship("Trade", back_populates="model")
    backtest_results = relationship("BacktestResult", back_populates="model")
    
    def __repr__(self):
        return f"<AIModel(id={self.id}, name={self.name}, strategy={self.strategy})>"


class User(Base):
    """User model for authentication and profile information"""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    
    # Authentication
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(50), unique=True, index=True, nullable=True)
    hashed_password = Column(String(255), nullable=False)
    
    # Account status
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    email_verified_at = Column(DateTime, nullable=True)
    
    # Profile
    first_name = Column(String(50), nullable=True)
    last_name = Column(String(50), nullable=True)
    phone_number = Column(String(20), nullable=True)
    profile_image_url = Column(String(255), nullable=True)
    
    # User preferences
    preferences = Column(JSON, nullable=True)
    
    # Risk profile from onboarding questionnaire
    risk_profile = Column(String(20), nullable=True)  # e.g., "conservative", "moderate", "aggressive"
    investment_goals = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login_at = Column(DateTime, nullable=True)
    
    # Relationships
    subscription = relationship("Subscription", back_populates="user", uselist=False)
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email})>"


class SubscriptionStatus(str, enum.Enum):
    """Defines possible states for a subscription."""
    ACTIVE = "active"
    TRIAL = "trial"
    EXPIRED = "expired"
    CANCELLED = "cancelled"
    PAST_DUE = "past_due"


class Subscription(Base):
    """Subscription model representing a user's subscription information."""
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    tier = Column(Enum(SubscriptionTier), index=True, default=SubscriptionTier.FREE)
    status = Column(Enum(SubscriptionStatus), index=True, default=SubscriptionStatus.ACTIVE)
    
    # Trial information
    is_trial = Column(Boolean, default=False)
    trial_started_at = Column(DateTime, nullable=True)
    trial_ends_at = Column(DateTime, nullable=True)
    
    # Subscription dates
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    current_period_start = Column(DateTime, nullable=True)
    current_period_end = Column(DateTime, nullable=True)
    
    # Payment information
    external_subscription_id = Column(String(255), nullable=True)  # ID from payment processor (e.g., Stripe)
    payment_method_id = Column(String(255), nullable=True)
    
    # Additional data
    extra_data = Column(Text, nullable=True)  # JSON storage for additional info
    
    # Relationships
    user = relationship("User", back_populates="subscription")
    payment_history = relationship("SubscriptionPayment", back_populates="subscription")
    
    @property
    def is_active(self) -> bool:
        """
        Check if the subscription is active.
        
        Returns:
            bool: True if the subscription is active or in trial, False otherwise
        """
        return self.status in [SubscriptionStatus.ACTIVE, SubscriptionStatus.TRIAL]
    
    @property
    def days_left_in_period(self) -> Optional[int]:
        """
        Calculate days left in the current period.
        
        Returns:
            Optional[int]: Number of days left or None if period end is not set
        """
        if not self.current_period_end:
            return None
            
        delta = self.current_period_end - datetime.utcnow()
        return max(0, delta.days)
    
    @property
    def days_left_in_trial(self) -> Optional[int]:
        """
        Calculate days left in trial period.
        
        Returns:
            Optional[int]: Number of days left in trial or None if not in trial
        """
        if not self.is_trial or not self.trial_ends_at:
            return None
            
        delta = self.trial_ends_at - datetime.utcnow()
        return max(0, delta.days)
    
    def get_features(self) -> Dict[str, Any]:
        """
        Get the features available for this subscription.
        
        Returns:
            Dict[str, Any]: Dictionary of features for the subscription tier
        """
        from app.utils.subscription_config import get_tier_features
        return get_tier_features(self.tier)


class SubscriptionPayment(Base):
    """Model for tracking subscription payment history."""
    __tablename__ = "subscription_payments"

    id = Column(Integer, primary_key=True, index=True)
    subscription_id = Column(Integer, ForeignKey("subscriptions.id"), index=True)
    
    amount = Column(Float, nullable=False)
    currency = Column(String(3), default="USD")
    
    # Payment tracking
    external_payment_id = Column(String(255), nullable=True)  # Payment ID from processor
    payment_method = Column(String(50), nullable=True)  # e.g., "credit_card", "paypal"
    
    status = Column(String(50), index=True)  # e.g., "succeeded", "failed", "refunded"
    
    # Dates
    payment_date = Column(DateTime, default=datetime.utcnow)
    period_start = Column(DateTime, nullable=True)
    period_end = Column(DateTime, nullable=True)
    
    # Metadata
    receipt_url = Column(String(255), nullable=True)
    invoice_id = Column(String(255), nullable=True)
    extra_data = Column(Text, nullable=True)  # JSON storage for additional info
    
    # Relationships
    subscription = relationship("Subscription", back_populates="payment_history")
