"""
API Routes

Defines all REST API endpoints for the Mercurio AI platform.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import logging

from app.db.database import get_db
from app.api.schemas import (
    TradeRequest, BacktestRequest, TrainRequest, 
    PredictionResponse, BacktestResponse, TrainResponse,
    AccountInfo, MarketStatus, StrategyInfo,
    UserCreate, UserResponse, SubscriptionInfo, SubscriptionResponse,
    TrialRequest, ActivateSubscriptionRequest, SubscriptionTierInfo, SubscriptionTiersResponse
)
from app.services.strategy_manager import StrategyManager
from app.services.trading import TradingService
from app.services.backtesting import BacktestingService
from app.services.market_data import MarketDataService
from app.services.subscription_service import SubscriptionService
from app.utils.subscription_config import SubscriptionTier, get_tier_features
from app.utils.subscription_helpers import get_tier_name, get_tier_price, get_tier_description

logger = logging.getLogger(__name__)

# Create router
api_router = APIRouter()

# Import our authentication routes
from app.api.auth_routes import router as auth_router
from app.api.schemas import UpgradeSubscriptionRequest, PaymentHistoryResponse, UsageMetricsResponse, PaymentWebhookRequest
from app.api.oauth_routes import router as oauth_router
from app.middleware.auth_middleware import AuthMiddleware

# Include authentication routers
api_router.include_router(auth_router)
api_router.include_router(oauth_router)

# Authentication dependency - uses our real authentication middleware now
async def get_current_user_id(current_user = Depends(AuthMiddleware.get_current_user)):
    """
    Get the ID of the currently authenticated user using our auth middleware.
    
    Args:
        current_user: User object from authentication middleware
        
    Returns:
        User ID of the authenticated user
    """
    return current_user.id

# Strategy API endpoints
@api_router.get("/strategies", response_model=List[StrategyInfo], tags=["Strategies"])
async def list_strategies():
    """
    List all available trading strategies.
    """
    try:
        strategy_manager = StrategyManager()
        strategies = await strategy_manager.list_strategies()
        return strategies
    except Exception as e:
        logger.error(f"Error listing strategies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/strategies/{strategy_name}", response_model=StrategyInfo, tags=["Strategies"])
async def get_strategy(strategy_name: str):
    """
    Get information about a specific strategy.
    """
    try:
        strategy_manager = StrategyManager()
        strategy_info = await strategy_manager.get_strategy_info(strategy_name)
        if not strategy_info:
            raise HTTPException(status_code=404, detail=f"Strategy {strategy_name} not found")
        return strategy_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting strategy info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prediction API endpoints
@api_router.get("/predict", response_model=PredictionResponse, tags=["Trading"])
async def predict(
    symbol: str = Query(..., description="Stock symbol (e.g., AAPL)"),
    strategy: str = Query(..., description="Strategy name to use for prediction"),
    model_id: Optional[int] = Query(None, description="Optional model ID to use for prediction")
):
    """
    Get a trading prediction (buy/sell/hold) for a symbol using the specified strategy.
    """
    try:
        strategy_manager = StrategyManager()
        prediction = await strategy_manager.get_prediction(symbol, strategy, model_id)
        return prediction
    except Exception as e:
        logger.error(f"Error getting prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Backtest API endpoints
@api_router.post("/backtest", response_model=BacktestResponse, tags=["Backtesting"])
async def run_backtest(request: BacktestRequest, db: AsyncSession = Depends(get_db)):
    """
    Run a backtest for a given strategy.
    """
    try:
        strategy_manager = StrategyManager()
        backtesting_service = BacktestingService()
        
        # Parse dates
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
        
        # Get the strategy
        strategy = await strategy_manager.get_strategy(
            request.strategy,
            request.model_id
        )
        if not strategy:
            raise HTTPException(status_code=404, detail="Strategy not found")
        
        # Run the backtest
        result = await backtesting_service.run_backtest(
            strategy=strategy,
            symbol=request.symbol,
            start_date=start_date,
            end_date=end_date,
            initial_capital=request.initial_capital,
            parameters=request.parameters
        )
        
        return BacktestResponse(
            strategy=request.strategy,
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            initial_capital=request.initial_capital,
            final_capital=result.get("final_capital", 0),
            total_return=result.get("total_return", 0),
            sharpe_ratio=result.get("sharpe_ratio", 0),
            max_drawdown=result.get("max_drawdown", 0),
            parameters=request.parameters,
            charts=result.get("charts", {})
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/backtests", tags=["Backtesting"])
async def run_backtests_alias(request: BacktestRequest, db: AsyncSession = Depends(get_db)):
    """
    Alias for /backtest endpoint to run a backtest (for test compatibility).
    """
    return await run_backtest(request, db)

@api_router.post("/api/backtests", tags=["Backtesting"])
async def run_backtests_alias_api(request: BacktestRequest, db: AsyncSession = Depends(get_db)):
    """
    Alias for /backtest endpoint to run a backtest (for test compatibility).
    """
    return await run_backtest(request, db)

# Trade API endpoints
@api_router.post("/trade", tags=["Trading"])
async def execute_trade(request: TradeRequest, db: AsyncSession = Depends(get_db)):
    """
    Execute a trade using a strategy.
    """
    try:
        strategy_manager = StrategyManager()
        trading_service = TradingService(is_paper=request.paper_trading)
        
        # Check if we need to calculate quantity
        quantity = request.quantity
        if quantity is None or quantity <= 0:
            quantity = await trading_service.calculate_order_quantity(
                request.symbol,
                request.action,
                request.strategy,
                request.model_id
            )
        
        # Execute the trade
        result = await trading_service.execute_trade(
            symbol=request.symbol,
            action=request.action,
            quantity=quantity,
            paper_trading=request.paper_trading,
            limit_price=request.limit_price,
            strategy_name=request.strategy
        )
        
        if result.get("status") == "error":
            raise HTTPException(status_code=500, detail=result.get("message", "Unknown error"))
        
        # Save trade to database
        if result.get("status") == "success":
            await strategy_manager.save_trade(result.get("order"), db)
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing trade: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.post("/trades", tags=["Trading"])
async def execute_trades_alias(request: TradeRequest, db: AsyncSession = Depends(get_db)):
    """
    Alias for /trade endpoint to execute a trade (for test compatibility).
    """
    return await execute_trade(request, db)

@api_router.post("/api/trades", tags=["Trading"])
async def execute_trades_alias_api(request: TradeRequest, db: AsyncSession = Depends(get_db)):
    """
    Alias for /trade endpoint to execute a trade (for test compatibility).
    """
    return await execute_trade(request, db)

# Market data API endpoints
@api_router.get("/market/status", response_model=MarketStatus, tags=["Market"])
async def get_market_status():
    """
    Get current market status.
    """
    try:
        trading_service = TradingService()
        status = await trading_service.check_market_status()
        return status
    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/market/symbols", response_model=List[str], tags=["Market"])
async def get_market_symbols(market_type: str = Query("stock", description="Market type (stock, crypto, etc.)")):
    """
    Get available market symbols.
    """
    try:
        market_data = MarketDataService()
        symbols = await market_data.get_market_symbols(market_type)
        return symbols
    except Exception as e:
        logger.error(f"Error getting market symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Account API endpoints
@api_router.get("/account", response_model=AccountInfo, tags=["Account"])
async def get_account_info(paper_trading: bool = Query(True, description="Whether to use paper trading")):
    """
    Get account information.
    """
    try:
        trading_service = TradingService(is_paper=paper_trading)
        account_info = await trading_service.get_account_info()
        return account_info
    except Exception as e:
        logger.error(f"Error getting account info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@api_router.get("/account/positions", tags=["Account"])
async def get_positions(paper_trading: bool = Query(True, description="Whether to use paper trading")):
    """
    Get current positions.
    """
    try:
        trading_service = TradingService(is_paper=paper_trading)
        positions = await trading_service.get_positions()
        return positions
    except Exception as e:
        logger.error(f"Error getting positions: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Subscription API endpoints
def get_subscription_service(db: AsyncSession = Depends(get_db)):
    return SubscriptionService(db)

@api_router.get("/subscription/tiers", response_model=SubscriptionTiersResponse, tags=["Subscription"])
async def get_subscription_tiers_alias(service: SubscriptionService = Depends(get_subscription_service)):
    tiers = await service.get_all_tiers()
    return {"tiers": tiers}

# Alias for test compatibility
@api_router.get("/subscription/tiers", response_model=SubscriptionTiersResponse, tags=["Subscription"])
async def get_subscription_tiers_alias_api(service: SubscriptionService = Depends(get_subscription_service)):
    tiers = await service.get_all_tiers()
    return {"tiers": tiers}

@api_router.get("/subscription/tiers/{tier}", response_model=SubscriptionTierInfo, tags=["Subscription"])
async def get_tier_details(tier: str, service: SubscriptionService = Depends(get_subscription_service)):
    """
    Get details for a specific subscription tier.
    """
    try:
        tier_enum = SubscriptionTier[tier.upper()]
    except KeyError:
        raise HTTPException(status_code=404, detail="Tier not found")
    return service.get_tier_details(tier_enum)

# Alias for test compatibility
@api_router.get("/subscription/tiers/{tier}", response_model=SubscriptionTierInfo, tags=["Subscription"])
async def get_tier_details_alias_api(tier: str, service: SubscriptionService = Depends(get_subscription_service)):
    try:
        tier_enum = SubscriptionTier[tier.upper()]
    except KeyError:
        raise HTTPException(status_code=404, detail="Tier not found")
    return service.get_tier_details(tier_enum)

@api_router.get("/subscription/current", tags=["Subscription"])
async def get_user_subscription_alias(db: AsyncSession = Depends(get_db), current_user_id: int = Depends(get_current_user_id)):
    return await get_current_subscription(db, current_user_id)

# Alias for test compatibility
@api_router.get("/subscription/current", tags=["Subscription"])
async def get_user_subscription_alias_api(db: AsyncSession = Depends(get_db), current_user_id: int = Depends(get_current_user_id)):
    subscription_service = SubscriptionService(db)
    subscription = await subscription_service.get_user_subscription(current_user_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="No active subscription found")
    # Map to the structure expected by the test
    return {
        "has_subscription": True,
        "tier": subscription.tier.value if hasattr(subscription.tier, 'value') else subscription.tier,
        "status": subscription.status.value if hasattr(subscription.status, 'value') else subscription.status,
        "days_remaining": getattr(subscription, 'days_left_in_period', None),
        "display_tier": getattr(subscription, 'display_tier', subscription.tier.value if hasattr(subscription.tier, 'value') else subscription.tier),
        "is_active": getattr(subscription, 'is_active', subscription.status.value == "active" if hasattr(subscription.status, 'value') else subscription.status == "active"),
        "is_trial": getattr(subscription, 'is_trial', False)
    }

@api_router.post("/subscription/trial", tags=["Subscription"])
async def start_trial_alias(
    request: TrialRequest,
    service: SubscriptionService = Depends(get_subscription_service),
    current_user_id: int = Depends(get_current_user_id)
):
    return await start_trial(request, service, current_user_id)

# Alias for test compatibility
@api_router.post("/subscription/trial", tags=["Subscription"])
async def start_trial_alias_api(
    request: TrialRequest,
    service: SubscriptionService = Depends(get_subscription_service),
    current_user_id: int = Depends(get_current_user_id)
):
    subscription = await service.start_trial(current_user_id, request.tier)
    return SubscriptionResponse(subscription=subscription)

@api_router.post("/subscription/activate", tags=["Subscription"])
async def activate_subscription_alias(
    request: ActivateSubscriptionRequest,
    service: SubscriptionService = Depends(get_subscription_service),
    current_user_id: int = Depends(get_current_user_id)
):
    subscription = await service.activate_subscription(
        current_user_id,
        request.tier,
        payment_method_id=request.payment_method_id,
        external_subscription_id=request.external_subscription_id
    )
    return SubscriptionResponse(subscription=subscription)

# Alias for test compatibility
@api_router.post("/subscription/activate", tags=["Subscription"])
async def activate_subscription_alias_api(
    request: ActivateSubscriptionRequest,
    service: SubscriptionService = Depends(get_subscription_service),
    current_user_id: int = Depends(get_current_user_id)
):
    subscription = await service.activate_subscription(
        current_user_id,
        request.tier,
        payment_method_id=request.payment_method_id,
        external_subscription_id=request.external_subscription_id
    )
    return SubscriptionResponse(subscription=subscription)

@api_router.post("/subscription/cancel", tags=["Subscription"])
async def cancel_subscription_alias(
    service: SubscriptionService = Depends(get_subscription_service),
    current_user_id: int = Depends(get_current_user_id)
):
    subscription = await service.cancel_subscription(current_user_id)
    return SubscriptionResponse(subscription=subscription)

# Alias for test compatibility
@api_router.post("/subscription/cancel", tags=["Subscription"])
async def cancel_subscription_alias_api(
    service: SubscriptionService = Depends(get_subscription_service),
    current_user_id: int = Depends(get_current_user_id)
):
    subscription = await service.cancel_subscription(current_user_id)
    return SubscriptionResponse(subscription=subscription)

@api_router.post("/users/me/subscription/upgrade", tags=["Subscription"])
async def upgrade_subscription_alias(
    request: UpgradeSubscriptionRequest,
    service: SubscriptionService = Depends(get_subscription_service),
    current_user_id: int = Depends(get_current_user_id)
):
    subscription = await service.upgrade_subscription(
        current_user_id,
        request.tier,
        payment_method_id=request.payment_method_id,
        prorate=request.prorate
    )
    return SubscriptionResponse(subscription=subscription)

@api_router.get("/users/me/subscription/payments", tags=["Subscription"])
async def get_payment_history_alias(
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    subscription_service = SubscriptionService(db)
    payments = await subscription_service.get_payment_history(current_user_id)
    total_spent = await subscription_service.get_total_spent(current_user_id)
    payment_infos = []
    for payment in payments:
        payment_infos.append(SubscriptionPaymentInfo(
            id=payment.id,
            subscription_id=payment.subscription_id,
            amount=payment.amount,
            currency="USD",
            payment_method=payment.payment_method,
            status=payment.status,
            payment_date=payment.payment_date,
            billing_period_start=payment.period_start,
            billing_period_end=payment.period_end,
            receipt_url=payment.receipt_url,
            extra_data=payment.extra_data
        ))
    return PaymentHistoryResponse(
        payments=payment_infos,
        total_spent=total_spent,
        currency="USD"
    )

@api_router.get("/users/me/subscription/usage", tags=["Subscription"])
async def get_usage_metrics_alias(
    db: AsyncSession = Depends(get_db),
    current_user_id: int = Depends(get_current_user_id)
):
    subscription_service = SubscriptionService(db)
    usage_data = await subscription_service.get_usage_metrics(current_user_id)
    if not usage_data:
        raise HTTPException(status_code=404, detail="No active subscription found")
    metrics = []
    for metric in usage_data.get("metrics", []):
        metrics.append(UsageMetric(
            name=metric["name"],
            display_name=metric["display_name"],
            current_usage=metric["current_usage"],
            limit=metric["limit"],
            percentage_used=metric["percentage_used"]
        ))
    return UsageMetricsResponse(
        metrics=metrics,
        billing_cycle_start=usage_data["billing_cycle_start"],
        billing_cycle_end=usage_data["billing_cycle_end"],
        days_left_in_cycle=usage_data["days_left_in_cycle"]
    )

@api_router.post("/webhooks/payment", tags=["Subscription"])
async def payment_webhook_alias(
    request: PaymentWebhookRequest,
    db: AsyncSession = Depends(get_db)
):
    try:
        subscription_service = SubscriptionService(db)
        result = await subscription_service.handle_payment_webhook(
            event_type=request.event_type.value if hasattr(request.event_type, 'value') else request.event_type,
            event_data=request.data
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return {"status": "success", **result}
    except HTTPException:
        raise
    except Exception as e:
        return {"status": "received", "processing_error": str(e)}

# Alias for test compatibility
@api_router.post("/subscription/trial", tags=["Subscription"])
async def start_trial_alias_api(
    request: TrialRequest,
    service: SubscriptionService = Depends(get_subscription_service),
    current_user_id: int = Depends(get_current_user_id)
):
    subscription = await service.start_trial(current_user_id, request.tier)
    return SubscriptionResponse(subscription=subscription)

@api_router.post("/subscription/activate", tags=["Subscription"])
async def activate_subscription_alias(
    request: ActivateSubscriptionRequest,
    service: SubscriptionService = Depends(get_subscription_service),
    current_user_id: int = Depends(get_current_user_id)
):
    subscription = await service.activate_subscription(
        current_user_id,
        request.tier,
        payment_method_id=request.payment_method_id,
        external_subscription_id=request.external_subscription_id
    )
    return SubscriptionResponse(subscription=subscription)

# Alias for test compatibility
@api_router.post("/subscription/activate", tags=["Subscription"])
async def activate_subscription_alias_api(
    request: ActivateSubscriptionRequest,
    service: SubscriptionService = Depends(get_subscription_service),
    current_user_id: int = Depends(get_current_user_id)
):
    return await activate_subscription(request, service, current_user_id)

@api_router.post("/subscription/cancel", tags=["Subscription"])
async def cancel_subscription_alias(
    service: SubscriptionService = Depends(get_subscription_service),
    current_user_id: int = Depends(get_current_user_id)
):
    subscription = await service.cancel_subscription(current_user_id)
    return SubscriptionResponse(subscription=subscription)

# Alias for test compatibility
@api_router.post("/subscription/cancel", tags=["Subscription"])
async def cancel_subscription_alias_api(
    service: SubscriptionService = Depends(get_subscription_service),
    current_user_id: int = Depends(get_current_user_id)
):
    return await cancel_subscription(service, current_user_id)

@api_router.get("/subscription/tiers", response_model=SubscriptionTiersResponse, tags=["Subscription"])
async def get_subscription_tiers():
    """
    Get all available Mercurio Alpha Model 3 tiers (performance-based, $0 base fee, profit share).
    """
    from app.utils.subscription_config import get_tier_info, SubscriptionTier
    tiers = []
    for tier in SubscriptionTier:
        info = get_tier_info(tier)
        tiers.append(SubscriptionTierInfo(
            name=tier.value,
            display_name=tier.name.title(),
            base_fee=info.get("base_fee", 0.0),
            profit_share=info.get("profit_share", 0.10),
            description=f"{tier.name.title()} plan: ${info.get('base_fee', 0.0):.2f}/mo + {int(info.get('profit_share', 0.10)*100)}% of profits. Max strategies: {info.get('max_strategies', 'Unlimited')}, Max portfolio: {info.get('max_portfolio', 'Unlimited')}",
            max_strategies=info.get("max_strategies"),
            max_portfolio=info.get("max_portfolio"),
            customization=info.get("customization", False),
            recommended=(tier == SubscriptionTier.GROWTH)
        ))
    return SubscriptionTiersResponse(tiers=tiers)

@api_router.get("/users/me/subscription", response_model=SubscriptionInfo, tags=["Subscription"])
async def get_current_subscription(db: AsyncSession = Depends(get_db), current_user_id: int = Depends(get_current_user_id)):
    """
    Get the current user's subscription information (Mercurio Alpha Model 3).
    """
    subscription_service = SubscriptionService(db)
    subscription = await subscription_service.get_user_subscription(current_user_id)
    if not subscription:
        raise HTTPException(status_code=404, detail="No active subscription found")
    features = get_tier_info(subscription.tier)
    return SubscriptionInfo(
        id=subscription.id,
        user_id=subscription.user_id,
        tier=subscription.tier.value,
        status=subscription.status.value,
        current_period_start=subscription.current_period_start,
        current_period_end=subscription.current_period_end,
        days_left_in_period=subscription.days_left_in_period,
        profit_share=subscription.profit_share,
        base_fee=subscription.base_fee,
        high_water_mark=subscription.high_water_mark,
        net_profit_this_period=subscription.net_profit_this_period,
        features=features
    )

@api_router.post("/users/me/subscription/activate", response_model=SubscriptionInfo, tags=["Subscription"])
async def activate_subscription(request: ActivateSubscriptionRequest, db: AsyncSession = Depends(get_db), current_user_id: int = Depends(get_current_user_id)):
    """
    Activate a paid subscription for the current user (Mercurio Alpha Model 3).
    """
    try:
        # Here you would integrate with your payment processor (e.g., Stripe)
        # to create a subscription and get the external_subscription_id
        # For this example, we'll just create a mock subscription
        
        subscription_service = SubscriptionService(db)
        subscription = await subscription_service.activate_subscription(
            user_id=current_user_id,
            tier=request.tier,
            payment_method_id=request.payment_method_id,
            external_subscription_id="mock_subscription_id"  # In production, this would come from your payment processor
        )
        
        # Get subscription features
        features = get_tier_info(subscription.tier)
        
        # Create response
        return SubscriptionInfo(
            id=subscription.id,
            user_id=subscription.user_id,
            tier=subscription.tier.value,
            status=subscription.status.value,
            current_period_start=subscription.current_period_start,
            current_period_end=subscription.current_period_end,
            days_left_in_period=subscription.days_left_in_period,
            profit_share=subscription.profit_share,
            base_fee=subscription.base_fee,
            high_water_mark=subscription.high_water_mark,
            net_profit_this_period=subscription.net_profit_this_period,
            features=features
        )
    except Exception as e:
        logger.error(f"Error activating subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/users/me/subscription/upgrade", response_model=SubscriptionInfo, tags=["Subscription"])
async def upgrade_subscription(request: UpgradeSubscriptionRequest, db: AsyncSession = Depends(get_db), current_user_id: int = Depends(get_current_user_id)):
    """
    Upgrade or downgrade the current user's subscription tier.
    """
    try:
        subscription_service = SubscriptionService(db)
        subscription = await subscription_service.upgrade_subscription(
            user_id=current_user_id,
            new_tier=request.tier,
            payment_method_id=request.payment_method_id,
            prorate=request.prorate
        )
        
        # Get subscription features
        features = subscription.get_features()
        
        # Create response
        return SubscriptionInfo(
            id=subscription.id,
            user_id=subscription.user_id,
            tier=subscription.tier.name,
            status=subscription.status.value,
            is_trial=subscription.is_trial,
            trial_started_at=subscription.trial_started_at,
            trial_ends_at=subscription.trial_ends_at,
            current_period_start=subscription.current_period_start,
            current_period_end=subscription.current_period_end,
            days_left_in_period=subscription.days_left_in_period,
            days_left_in_trial=subscription.days_left_in_trial,
            features=features
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error upgrading subscription: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/users/me/subscription/payments", response_model=PaymentHistoryResponse, tags=["Subscription"])
async def get_payment_history(db: AsyncSession = Depends(get_db), current_user_id: int = Depends(get_current_user_id)):
    """
    Get payment history for the current user's subscription.
    """
    try:
        subscription_service = SubscriptionService(db)
        payments = await subscription_service.get_payment_history(current_user_id)
        total_spent = await subscription_service.get_total_spent(current_user_id)
        
        # Convert to response model
        payment_infos = []
        for payment in payments:
            payment_infos.append(SubscriptionPaymentInfo(
                id=payment.id,
                subscription_id=payment.subscription_id,
                amount=payment.amount,
                currency="USD",  # Default currency
                payment_method=payment.payment_method,
                status=payment.status,
                payment_date=payment.payment_date,
                billing_period_start=payment.period_start,
                billing_period_end=payment.period_end,
                receipt_url=payment.receipt_url,
                extra_data=payment.extra_data
            ))
        
        return PaymentHistoryResponse(
            payments=payment_infos,
            total_spent=total_spent,
            currency="USD"
        )
    except Exception as e:
        logger.error(f"Error getting payment history: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.get("/users/me/subscription/usage", response_model=UsageMetricsResponse, tags=["Subscription"])
async def get_usage_metrics(db: AsyncSession = Depends(get_db), current_user_id: int = Depends(get_current_user_id)):
    """
    Get usage metrics for the current user's subscription.
    Shows current usage vs limits for various features.
    """
    try:
        subscription_service = SubscriptionService(db)
        usage_data = await subscription_service.get_usage_metrics(current_user_id)
        
        if not usage_data:
            raise HTTPException(status_code=404, detail="No active subscription found")
        
        # Convert to response model
        metrics = []
        for metric in usage_data.get("metrics", []):
            metrics.append(UsageMetric(
                name=metric["name"],
                display_name=metric["display_name"],
                current_usage=metric["current_usage"],
                limit=metric["limit"],
                percentage_used=metric["percentage_used"]
            ))
        
        return UsageMetricsResponse(
            metrics=metrics,
            billing_cycle_start=usage_data["billing_cycle_start"],
            billing_cycle_end=usage_data["billing_cycle_end"],
            days_left_in_cycle=usage_data["days_left_in_cycle"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting usage metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api_router.post("/webhooks/payment", tags=["Subscription"])
async def payment_webhook(request: PaymentWebhookRequest, db: AsyncSession = Depends(get_db)):
    """
    Handle webhook callbacks from payment provider.
    This endpoint would typically be secured with a shared secret or signature verification.
    """
    try:
        # Validate the webhook signature - would be implemented with your payment provider's SDK
        # For example, stripe.Webhook.construct_event(payload, signature, webhook_secret)
        
        subscription_service = SubscriptionService(db)
        result = await subscription_service.handle_payment_webhook(
            event_type=request.event_type.value,
            event_data=request.data
        )
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return {"status": "success", **result}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing payment webhook: {e}")
        # Always return 200 to the payment provider to avoid retries
        # but log the error for investigation
        return {"status": "received", "processing_error": str(e)}
