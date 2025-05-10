"""
Trading Tasks

Celery tasks for automated trading operations.
"""
import os
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import asyncio

from app.tasks.celery_app import celery_app
from app.services.strategy_manager import StrategyManager
from app.services.trading import TradingService
from app.db.models import TradeAction

logger = logging.getLogger(__name__)

@celery_app.task(name="execute_strategy")
def execute_strategy(
    strategy_name: str,
    symbols: List[str],
    model_id: Optional[int] = None,
    paper_trading: bool = True,
    capital_percentage: float = 0.1
) -> Dict[str, Any]:
    """
    Celery task to execute a trading strategy on a set of symbols.
    
    Args:
        strategy_name: Name of the strategy
        symbols: List of symbols to trade
        model_id: Optional ID of a specific trained model to use
        paper_trading: Whether to use paper trading
        capital_percentage: Percentage of available capital to use per trade
        
    Returns:
        Dictionary with execution results
    """
    logger.info(f"Starting strategy execution task: {strategy_name} on {symbols}")
    
    # Run the async trading function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Initialize services
        strategy_manager = StrategyManager()
        trading_service = TradingService(is_paper=paper_trading)
        
        results = []
        
        for symbol in symbols:
            try:
                # Get prediction
                prediction = loop.run_until_complete(
                    strategy_manager.get_prediction(symbol, strategy_name, model_id)
                )
                
                # Check for error
                if "error" in prediction:
                    results.append({
                        "symbol": symbol,
                        "status": "error",
                        "message": prediction["error"]
                    })
                    continue
                
                # Extract trade action
                action = TradeAction(prediction["action"])
                
                # Skip if HOLD
                if action == TradeAction.HOLD:
                    results.append({
                        "symbol": symbol,
                        "status": "skipped",
                        "message": "HOLD signal, no trade executed",
                        "action": action.value,
                        "confidence": prediction["confidence"]
                    })
                    continue
                
                # Calculate quantity
                quantity = loop.run_until_complete(
                    trading_service.calculate_order_quantity(
                        symbol, action, capital_percentage
                    )
                )
                
                # Skip if quantity is zero
                if quantity <= 0:
                    results.append({
                        "symbol": symbol,
                        "status": "skipped",
                        "message": "Zero quantity calculated",
                        "action": action.value,
                        "confidence": prediction["confidence"]
                    })
                    continue
                
                # Execute trade
                trade_result = loop.run_until_complete(
                    trading_service.execute_trade(
                        symbol=symbol,
                        action=action,
                        quantity=quantity,
                        strategy_name=strategy_name
                    )
                )
                
                # Add to results
                results.append({
                    "symbol": symbol,
                    "status": trade_result["status"],
                    "action": action.value,
                    "quantity": quantity,
                    "confidence": prediction["confidence"],
                    "order": trade_result.get("order", {}),
                    "message": trade_result.get("message", "")
                })
                
            except Exception as e:
                logger.error(f"Error executing trade for {symbol}: {e}")
                results.append({
                    "symbol": symbol,
                    "status": "error",
                    "message": str(e)
                })
        
        return {
            "strategy": strategy_name,
            "timestamp": datetime.now().isoformat(),
            "paper_trading": paper_trading,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error in strategy execution task: {e}")
        return {"error": str(e)}
    finally:
        loop.close()

@celery_app.task(name="run_trading_session")
def run_trading_session(
    strategy_name: str,
    symbols: List[str],
    paper_trading: bool = True,
    check_market_hours: bool = True
) -> Dict[str, Any]:
    """
    Celery task to run a complete trading session.
    Checks market hours, executes strategy, and logs results.
    
    Args:
        strategy_name: Name of the strategy
        symbols: List of symbols to trade
        paper_trading: Whether to use paper trading
        check_market_hours: Whether to check if market is open
        
    Returns:
        Dictionary with session results
    """
    logger.info(f"Starting trading session: {strategy_name} on {symbols}")
    
    # Run the async trading function in event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Initialize trading service
        trading_service = TradingService(is_paper=paper_trading)
        
        # Check market hours if required
        if check_market_hours:
            market_status = loop.run_until_complete(
                trading_service.check_market_status()
            )
            
            if not market_status.get("is_open", False):
                return {
                    "status": "skipped",
                    "message": "Market is closed",
                    "market_status": market_status
                }
        
        # Execute strategy
        execution_result = execute_strategy.delay(
            strategy_name=strategy_name,
            symbols=symbols,
            paper_trading=paper_trading
        )
        
        # Get account info
        account_info = loop.run_until_complete(
            trading_service.get_account_info()
        )
        
        return {
            "status": "submitted",
            "task_id": execution_result.id,
            "strategy": strategy_name,
            "symbols": symbols,
            "paper_trading": paper_trading,
            "timestamp": datetime.now().isoformat(),
            "account_info": account_info
        }
        
    except Exception as e:
        logger.error(f"Error in trading session task: {e}")
        return {"error": str(e)}
    finally:
        loop.close()
