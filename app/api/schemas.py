"""
API Schemas

Defines Pydantic models for request/response validation in the REST API.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

from app.db.models import TradeAction
from app.utils.subscription_config import SubscriptionTier
from app.db.models import SubscriptionStatus

class StrategyInfo(BaseModel):
    """Information about a trading strategy"""
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    requires_training: bool = False

class PredictionResponse(BaseModel):
    """Response for a trading prediction"""
    symbol: str
    strategy: str
    action: str  # "buy", "sell", or "hold"
    confidence: float
    price: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    explanation: Optional[str] = None

class BacktestRequest(BaseModel):
    """Request to run a backtest"""
    strategy: str
    symbol: str
    start_date: str  # ISO format date
    end_date: str  # ISO format date
    initial_capital: float = 10000.0
    parameters: Optional[Dict[str, Any]] = None

class BacktestResponse(BaseModel):
    """Response with backtest results"""
    id: int
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    initial_capital: float
    final_capital: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    parameters: Optional[Dict[str, Any]] = None
    charts: Dict[str, str] = Field(default_factory=dict)  # base64 encoded images

class TrainRequest(BaseModel):
    """Request to train a model"""
    strategy: str
    symbols: List[str]
    start_date: str  # ISO format date
    end_date: str  # ISO format date
    parameters: Optional[Dict[str, Any]] = None

class TrainResponse(BaseModel):
    """Response with training results"""
    id: int
    strategy: str
    symbols: List[str]
    start_date: str
    end_date: str
    model_path: str
    metrics: Dict[str, Any] = Field(default_factory=dict)
    parameters: Optional[Dict[str, Any]] = None

class TradeRequest(BaseModel):
    """Request to execute a trade"""
    strategy: str
    symbol: str
    action: TradeAction
    quantity: Optional[float] = None  # If None, use capital_percentage
    capital_percentage: Optional[float] = 0.1  # Percentage of available capital to use
    order_type: Optional[str] = "market"  # "market", "limit", etc.
    limit_price: Optional[float] = None  # For limit orders
    paper_trading: bool = True

class MarketStatus(BaseModel):
    """Current market status"""
    is_open: bool
    next_open: Optional[str] = None  # ISO format datetime
    next_close: Optional[str] = None  # ISO format datetime
    timestamp: Optional[str] = None  # ISO format datetime
    error: Optional[str] = None

class AccountInfo(BaseModel):
    """Account information"""
    id: Optional[str] = None
    cash: Optional[float] = None
    portfolio_value: Optional[float] = None
    equity: Optional[float] = None
    buying_power: Optional[float] = None
    initial_margin: Optional[float] = None
    daytrade_count: Optional[int] = None
    status: Optional[str] = None
    error: Optional[str] = None


class UserCreate(BaseModel):
    """Request to create a new user"""
    email: str
    username: Optional[str] = None
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    phone_number: Optional[str] = None
    risk_profile: Optional[str] = None
    investment_goals: Optional[Dict[str, Any]] = None


class UserResponse(BaseModel):
    """Response with user information"""
    id: int
    email: str
    username: Optional[str] = None
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    is_active: bool
    is_verified: bool
    created_at: datetime
    subscription_tier: Optional[str] = None
    subscription_status: Optional[str] = None


class SubscriptionInfo(BaseModel):
    """Information about a subscription"""
    id: int
    user_id: int
    tier: str
    status: str
    is_trial: bool
    trial_started_at: Optional[datetime] = None
    trial_ends_at: Optional[datetime] = None
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    days_left_in_period: Optional[int] = None
    days_left_in_trial: Optional[int] = None
    features: Dict[str, Any] = Field(default_factory=dict)


class SubscriptionResponse(BaseModel):
    """Response with subscription information"""
    subscription: SubscriptionInfo
    user: UserResponse


class TrialRequest(BaseModel):
    """Request to start a trial subscription"""
    tier: SubscriptionTier


class ActivateSubscriptionRequest(BaseModel):
    """Request to activate a paid subscription"""
    tier: SubscriptionTier
    payment_method_id: str


class SubscriptionTierInfo(BaseModel):
    """Information about a subscription tier"""
    name: str
    display_name: str
    price: float
    description: str
    features: List[str]
    recommended: bool = False


class SubscriptionTiersResponse(BaseModel):
    """Response with all subscription tiers"""
    tiers: List[SubscriptionTierInfo]
