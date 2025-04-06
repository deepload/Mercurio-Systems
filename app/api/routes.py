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
    AccountInfo, MarketStatus, StrategyInfo
)
from app.services.strategy_manager import StrategyManager
from app.services.trading import TradingService
from app.services.backtesting import BacktestingService
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

# Create router
api_router = APIRouter()

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
    Run a backtest for a trading strategy.
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
            request.parameters or {}
        )
        
        if not strategy:
            raise HTTPException(status_code=404, detail=f"Strategy {request.strategy} not found")
        
        # Run the backtest
        result = await backtesting_service.run_backtest(
            strategy,
            request.symbol,
            start_date,
            end_date,
            request.initial_capital
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Save backtest result to database
        await strategy_manager.save_backtest_result(result, db)
        
        return BacktestResponse(
            id=result.get("id", 0),
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

# Training API endpoints
@api_router.post("/train", response_model=TrainResponse, tags=["Training"])
async def train_model(request: TrainRequest, db: AsyncSession = Depends(get_db)):
    """
    Train a model for a trading strategy.
    """
    try:
        strategy_manager = StrategyManager()
        
        # Parse dates
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
        
        # Train the model
        result = await strategy_manager.train_strategy(
            request.strategy,
            request.symbols,
            start_date,
            end_date,
            request.parameters or {}
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Save model metadata to database
        model_id = await strategy_manager.save_model_metadata(result, db)
        
        return TrainResponse(
            id=model_id,
            strategy=request.strategy,
            symbols=request.symbols,
            start_date=request.start_date,
            end_date=request.end_date,
            model_path=result.get("model_path", ""),
            metrics=result.get("metrics", {}),
            parameters=request.parameters
        )
    except Exception as e:
        logger.error(f"Error training model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Trading API endpoints
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
                request.capital_percentage or 0.1
            )
        
        # Execute the trade
        result = await trading_service.execute_trade(
            symbol=request.symbol,
            action=request.action,
            quantity=quantity,
            order_type=request.order_type or "market",
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
