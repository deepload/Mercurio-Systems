"""
Strategy Manager

Manages the loading, initialization, and execution of trading strategies.
Acts as a central coordinator for strategy-related operations.
"""
import os
import importlib
import logging
import inspect
from typing import Dict, Any, List, Optional, Type, Tuple
import pandas as pd
from datetime import datetime
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from app.db.models import AIModel, Trade, BacktestResult, TradeAction
from app.strategies.base import BaseStrategy
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Service for managing trading strategies.
    
    Responsibilities:
    - Discover and load strategy plugins
    - Initialize strategies with parameters
    - Coordinate training, prediction, and backtesting
    - Manage strategy models and metadata
    """
    
    def __init__(self):
        """Initialize the strategy manager"""
        self.market_data = MarketDataService()
        self.strategies_cache = {}  # Cache of loaded strategy classes
        
    async def list_strategies(self) -> List[Dict[str, Any]]:
        """
        List all available trading strategies.
        
        Returns:
            List of strategy information dictionaries
        """
        strategies = []
        
        # Get the strategies directory path
        strategies_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "strategies")
        
        # Load all Python files in the strategies directory
        for filename in os.listdir(strategies_dir):
            if filename.endswith(".py") and filename != "__init__.py" and filename != "base.py":
                module_name = filename[:-3]  # Remove .py extension
                
                try:
                    # Import the module
                    module = importlib.import_module(f"app.strategies.{module_name}")
                    
                    # Find strategy classes in the module
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BaseStrategy) and 
                            obj.__module__ == module.__name__ and
                            obj != BaseStrategy):
                            
                            # Get strategy parameters from constructor
                            params = {}
                            signature = inspect.signature(obj.__init__)
                            for param_name, param in signature.parameters.items():
                                if param_name not in ['self', 'args', 'kwargs']:
                                    if param.default != inspect.Parameter.empty:
                                        params[param_name] = param.default
                                    else:
                                        params[param_name] = None
                            
                            # Get strategy description from docstring
                            description = obj.__doc__.strip().split('\n')[0] if obj.__doc__ else ""
                            
                            # Check if it requires training
                            requires_training = hasattr(obj, 'train') and callable(getattr(obj, 'train'))
                            
                            # Create strategy info
                            strategy_info = {
                                "name": obj.__name__,
                                "description": description,
                                "parameters": params,
                                "requires_training": requires_training
                            }
                            
                            strategies.append(strategy_info)
                            
                            # Cache the strategy class
                            self.strategies_cache[obj.__name__] = obj
                
                except Exception as e:
                    logger.error(f"Error loading strategy module {module_name}: {e}")
        
        return strategies
    
    async def get_strategy_info(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific strategy.
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy information dictionary or None if not found
        """
        # Ensure strategies are loaded
        if not self.strategies_cache:
            await self.list_strategies()
        
        # Find the strategy class
        strategy_class = self.strategies_cache.get(strategy_name)
        if not strategy_class:
            return None
        
        # Get strategy parameters from constructor
        params = {}
        signature = inspect.signature(strategy_class.__init__)
        for param_name, param in signature.parameters.items():
            if param_name not in ['self', 'args', 'kwargs']:
                if param.default != inspect.Parameter.empty:
                    params[param_name] = param.default
                else:
                    params[param_name] = None
        
        # Get strategy description from docstring
        description = strategy_class.__doc__.strip().split('\n')[0] if strategy_class.__doc__ else ""
        
        # Check if it requires training
        requires_training = hasattr(strategy_class, 'train') and callable(getattr(strategy_class, 'train'))
        
        return {
            "name": strategy_name,
            "description": description,
            "parameters": params,
            "requires_training": requires_training
        }
    
    async def get_strategy(
        self,
        strategy_name: str,
        parameters: Dict[str, Any] = None
    ) -> Optional[BaseStrategy]:
        """
        Get an instance of a strategy.
        
        Args:
            strategy_name: Name of the strategy
            parameters: Parameters for the strategy
            
        Returns:
            Strategy instance or None if not found
        """
        if parameters is None:
            parameters = {}
        
        # Ensure strategies are loaded
        if not self.strategies_cache:
            await self.list_strategies()
        
        # Find the strategy class
        strategy_class = self.strategies_cache.get(strategy_name)
        if not strategy_class:
            return None
        
        # Initialize the strategy with parameters
        try:
            strategy = strategy_class(**parameters)
            return strategy
        except Exception as e:
            logger.error(f"Error initializing strategy {strategy_name}: {e}")
            return None
    
    async def get_prediction(
        self,
        symbol: str,
        strategy_name: str,
        model_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Get a trading prediction for a symbol using a strategy.
        
        Args:
            symbol: The market symbol (e.g., 'AAPL')
            strategy_name: Name of the strategy to use
            model_id: Optional ID of a specific trained model to use
            
        Returns:
            Prediction dictionary
        """
        # Get the strategy
        strategy = await self.get_strategy(strategy_name)
        if not strategy:
            return {"error": f"Strategy {strategy_name} not found"}
        
        # If model_id is provided, load the model
        if model_id:
            # TODO: Load model from database and set it to the strategy
            pass
        
        try:
            # Get recent data for the symbol
            end_date = datetime.now()
            start_date = end_date - pd.Timedelta(days=60)  # Get 60 days of data
            
            # Load data
            data = await strategy.load_data(symbol, start_date, end_date)
            
            # Preprocess data
            processed_data = await strategy.preprocess_data(data)
            
            # Make prediction
            action, confidence = await strategy.predict(processed_data)
            
            # Get latest price
            price = await self.market_data.get_latest_price(symbol)
            
            # Format the prediction
            prediction = {
                "symbol": symbol,
                "strategy": strategy_name,
                "action": action.value,
                "confidence": confidence,
                "price": price,
                "timestamp": datetime.now().isoformat(),
                "explanation": f"Based on the {strategy_name} strategy with {confidence:.2f} confidence"
            }
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return {"error": str(e)}
    
    async def train_strategy(
        self,
        strategy_name: str,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        parameters: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Train a strategy model.
        
        Args:
            strategy_name: Name of the strategy
            symbols: List of symbols to train on
            start_date: Start date for training data
            end_date: End date for training data
            parameters: Parameters for the strategy
            
        Returns:
            Dictionary with training results
        """
        if parameters is None:
            parameters = {}
        
        # Get the strategy
        strategy = await self.get_strategy(strategy_name, parameters)
        if not strategy:
            return {"error": f"Strategy {strategy_name} not found"}
        
        try:
            # Train on each symbol
            results = {}
            combined_data = None
            
            for symbol in symbols:
                # Load data
                data = await strategy.load_data(symbol, start_date, end_date)
                
                # Preprocess data
                processed_data = await strategy.preprocess_data(data)
                
                # Add symbol column for multi-symbol training
                processed_data['symbol'] = symbol
                
                # Combine data
                if combined_data is None:
                    combined_data = processed_data
                else:
                    combined_data = pd.concat([combined_data, processed_data])
            
            # Train the model on combined data
            training_metrics = await strategy.train(combined_data)
            
            # Save the model
            model_dir = os.getenv("MODEL_DIR", "./models")
            os.makedirs(model_dir, exist_ok=True)
            
            model_path = await strategy.save_model(model_dir)
            
            # Format the results
            results = {
                "strategy": strategy_name,
                "symbols": symbols,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "model_path": model_path,
                "metrics": training_metrics,
                "parameters": parameters
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Error training strategy: {e}")
            return {"error": str(e)}
    
    async def save_model_metadata(
        self,
        training_result: Dict[str, Any],
        db: AsyncSession
    ) -> int:
        """
        Save model metadata to the database.
        
        Args:
            training_result: Results from training
            db: Database session
            
        Returns:
            Model ID
        """
        # Create model metadata
        model = AIModel(
            name=f"{training_result['strategy']}_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            strategy=training_result['strategy'],
            model_type=training_result.get('model_type', 'unknown'),
            symbols=training_result['symbols'],
            train_start_date=datetime.fromisoformat(training_result['start_date']),
            train_end_date=datetime.fromisoformat(training_result['end_date']),
            model_path=training_result['model_path'],
            parameters=training_result.get('parameters', {}),
            metrics=training_result.get('metrics', {}),
            created_at=datetime.now()
        )
        
        # Save to database
        db.add(model)
        await db.commit()
        await db.refresh(model)
        
        return model.id
    
    async def save_backtest_result(
        self,
        backtest_result: Dict[str, Any],
        db: AsyncSession
    ) -> int:
        """
        Save backtest result to the database.
        
        Args:
            backtest_result: Results from backtesting
            db: Database session
            
        Returns:
            Backtest result ID
        """
        # Create backtest result
        result = BacktestResult(
            strategy=backtest_result['strategy'],
            symbol=backtest_result['symbol'],
            start_date=datetime.fromisoformat(backtest_result['start_date']),
            end_date=datetime.fromisoformat(backtest_result['end_date']),
            initial_capital=backtest_result['initial_capital'],
            final_capital=backtest_result['final_capital'],
            total_return=backtest_result['total_return'],
            sharpe_ratio=backtest_result.get('sharpe_ratio', 0),
            max_drawdown=backtest_result.get('max_drawdown', 0),
            parameters=backtest_result.get('parameters', {}),
            created_at=datetime.now()
        )
        
        # Save to database
        db.add(result)
        await db.commit()
        await db.refresh(result)
        
        # Update the result with the ID
        backtest_result['id'] = result.id
        
        return result.id
    
    async def save_trade(
        self,
        trade_data: Dict[str, Any],
        db: AsyncSession
    ) -> int:
        """
        Save trade to the database.
        
        Args:
            trade_data: Trade data
            db: Database session
            
        Returns:
            Trade ID
        """
        # Determine trade action
        action = TradeAction.BUY if trade_data['side'] == 'buy' else TradeAction.SELL
        
        # Create trade record
        trade = Trade(
            symbol=trade_data['symbol'],
            strategy=trade_data.get('strategy', 'unknown'),
            action=action,
            price=float(trade_data.get('filled_avg_price', 0)),
            quantity=float(trade_data.get('qty', 0)),
            timestamp=datetime.now(),
            trade_metadata=trade_data
        )
        
        # Save to database
        db.add(trade)
        await db.commit()
        await db.refresh(trade)
        
        return trade.id
