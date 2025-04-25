"""
MercurioAI Enhanced Backtesting Engine

This module implements an improved backtesting engine with event-driven architecture,
better performance, and more realistic trading simulation.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import uuid
import json
from pathlib import Path

from .event_bus import EventBus, EventType
from app.db.models import TradeAction

logger = logging.getLogger(__name__)

class BacktestResult:
    """Container for backtest results with rich metadata"""
    
    def __init__(self, 
                 strategy_name: str,
                 symbol: str,
                 start_date: Union[str, datetime],
                 end_date: Union[str, datetime],
                 initial_capital: float):
        """
        Initialize backtest result container
        
        Args:
            strategy_name: Name of the strategy used
            symbol: Symbol that was traded
            start_date: Backtest start date
            end_date: Backtest end date
            initial_capital: Initial capital
        """
        self.id = str(uuid.uuid4())
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.start_date = start_date if isinstance(start_date, str) else start_date.isoformat()
        self.end_date = end_date if isinstance(end_date, str) else end_date.isoformat()
        self.initial_capital = initial_capital
        
        # Performance metrics
        self.final_capital = initial_capital
        self.total_return = 0.0
        self.annual_return = 0.0
        self.sharpe_ratio = 0.0
        self.max_drawdown = 0.0
        self.win_rate = 0.0
        self.profit_factor = 0.0
        self.trade_count = 0
        self.transaction_costs = 0.0
        
        # Trade data
        self.trades = []
        self.equity_curve = None
        self.positions = None
        self.daily_returns = None
        
        # Execution details
        self.execution_time = 0.0
        self.created_at = datetime.now().isoformat()
        self.metadata = {}
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        result = {
            "id": self.id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "trade_count": self.trade_count,
            "transaction_costs": self.transaction_costs,
            "execution_time": self.execution_time,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
        
        # Convert DataFrames to lists for JSON serialization
        if self.equity_curve is not None:
            result["equity_curve"] = self.equity_curve.to_dict(orient="records")
            
        if self.positions is not None:
            result["positions"] = self.positions.to_dict(orient="records")
            
        if self.daily_returns is not None:
            result["daily_returns"] = self.daily_returns.to_dict()
            
        result["trades"] = self.trades
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BacktestResult':
        """Create instance from dictionary"""
        result = cls(
            strategy_name=data["strategy_name"],
            symbol=data["symbol"],
            start_date=data["start_date"],
            end_date=data["end_date"],
            initial_capital=data["initial_capital"]
        )
        
        result.id = data["id"]
        result.final_capital = data["final_capital"]
        result.total_return = data["total_return"]
        result.annual_return = data["annual_return"]
        result.sharpe_ratio = data["sharpe_ratio"]
        result.max_drawdown = data["max_drawdown"]
        result.win_rate = data["win_rate"]
        result.profit_factor = data["profit_factor"]
        result.trade_count = data["trade_count"]
        result.transaction_costs = data.get("transaction_costs", 0.0)
        result.execution_time = data["execution_time"]
        result.created_at = data["created_at"]
        result.metadata = data.get("metadata", {})
        result.trades = data["trades"]
        
        # Convert JSON to DataFrames
        if "equity_curve" in data:
            result.equity_curve = pd.DataFrame(data["equity_curve"])
            
        if "positions" in data:
            result.positions = pd.DataFrame(data["positions"])
            
        if "daily_returns" in data:
            result.daily_returns = pd.Series(data["daily_returns"])
            
        return result


class TransactionCostModel:
    """Model for realistic transaction costs"""
    
    def __init__(self, 
                 percentage_fee: float = 0.001,
                 fixed_fee: float = 0.0,
                 minimum_fee: float = 0.0,
                 slippage_model: str = "fixed",
                 slippage_value: float = 0.0001):
        """
        Initialize transaction cost model
        
        Args:
            percentage_fee: Percentage fee (e.g., 0.001 for 0.1%)
            fixed_fee: Fixed fee per trade in currency units
            minimum_fee: Minimum fee per trade
            slippage_model: Slippage model ('fixed', 'percentage', 'volatility')
            slippage_value: Slippage parameter value
        """
        self.percentage_fee = percentage_fee
        self.fixed_fee = fixed_fee
        self.minimum_fee = minimum_fee
        self.slippage_model = slippage_model
        self.slippage_value = slippage_value
        
    def calculate_transaction_cost(self, price: float, quantity: float, volatility: Optional[float] = None) -> float:
        """
        Calculate transaction cost for a trade
        
        Args:
            price: Execution price
            quantity: Trade quantity
            volatility: Optional volatility for volatility-based slippage
            
        Returns:
            Total transaction cost
        """
        # Calculate trade value
        trade_value = price * abs(quantity)
        
        # Calculate fee
        fee = trade_value * self.percentage_fee + self.fixed_fee
        
        # Apply minimum fee if needed
        fee = max(fee, self.minimum_fee)
        
        # Add slippage cost
        slippage_cost = self.calculate_slippage(price, quantity, volatility)
        total_cost = fee + slippage_cost
        
        return total_cost
    
    def calculate_slippage(self, price: float, quantity: float, volatility: Optional[float] = None) -> float:
        """
        Calculate slippage cost
        
        Args:
            price: Execution price
            quantity: Trade quantity
            volatility: Optional volatility (e.g., ATR)
            
        Returns:
            Slippage cost
        """
        if self.slippage_model == "fixed":
            # Fixed pip value
            slippage_price = price * self.slippage_value
            
        elif self.slippage_model == "percentage":
            # Percentage of price
            slippage_price = price * self.slippage_value
            
        elif self.slippage_model == "volatility" and volatility is not None:
            # Volatility-based slippage
            slippage_price = volatility * self.slippage_value
            
        else:
            # Default to fixed
            slippage_price = price * 0.0001
            
        # Calculate slippage cost (price impact * quantity)
        slippage_cost = slippage_price * abs(quantity)
        return slippage_cost


class EnhancedBacktester:
    """
    Enhanced backtesting engine with realistic simulation and performance analysis
    """
    
    def __init__(self):
        """Initialize enhanced backtester"""
        self.event_bus = EventBus()
        self.transaction_cost_model = TransactionCostModel()
        self.results_dir = Path("./results/backtests")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def set_transaction_cost_model(self, model: TransactionCostModel) -> None:
        """
        Set the transaction cost model
        
        Args:
            model: Transaction cost model
        """
        self.transaction_cost_model = model
        
    async def run_backtest(self,
                          strategy: Any,
                          data: pd.DataFrame,
                          symbol: str,
                          initial_capital: float = 10000.0,
                          start_date: Optional[Union[str, datetime]] = None,
                          end_date: Optional[Union[str, datetime]] = None,
                          risk_limit: float = 0.02) -> BacktestResult:
        """
        Run a backtest with enhanced features
        
        Args:
            strategy: Strategy instance
            data: Market data DataFrame
            symbol: Trading symbol
            initial_capital: Initial capital
            start_date: Optional start date (filters data if provided)
            end_date: Optional end date (filters data if provided)
            risk_limit: Maximum percentage of capital per position
            
        Returns:
            BacktestResult with detailed performance metrics
        """
        start_time = datetime.now()
        
        # Publish event for backtest start
        await self.event_bus.publish(EventType.BACKTEST_STARTED, {
            "strategy": strategy.__class__.__name__,
            "symbol": symbol,
            "initial_capital": initial_capital,
            "data_points": len(data)
        })
        
        # Filter data if dates provided
        if start_date is not None or end_date is not None:
            if 'timestamp' in data.columns:
                timestamp_col = 'timestamp'
            elif 'date' in data.columns:
                timestamp_col = 'date'
            else:
                # Try to use index
                if isinstance(data.index, pd.DatetimeIndex):
                    data = data.copy()
                    data['timestamp'] = data.index
                    timestamp_col = 'timestamp'
                else:
                    raise ValueError("No timestamp column found in data")
            
            # Convert to datetime if string
            if isinstance(data[timestamp_col].iloc[0], str):
                data[timestamp_col] = pd.to_datetime(data[timestamp_col])
            
            # Apply filters
            if start_date is not None:
                start_dt = pd.to_datetime(start_date)
                data = data[data[timestamp_col] >= start_dt]
                
            if end_date is not None:
                end_dt = pd.to_datetime(end_date)
                data = data[data[timestamp_col] <= end_dt]
            
        # Create result container
        result = BacktestResult(
            strategy_name=strategy.__class__.__name__,
            symbol=symbol,
            start_date=start_date or data['timestamp'].min() if 'timestamp' in data.columns else "unknown",
            end_date=end_date or data['timestamp'].max() if 'timestamp' in data.columns else "unknown",
            initial_capital=initial_capital
        )
        
        # Execute the backtest
        try:
            # Run the strategy's backtest method
            if hasattr(strategy, 'backtest'):
                strategy_result = await strategy.backtest(data, initial_capital)
                
                # Process strategy results
                result.equity_curve = strategy_result.get('equity_curve')
                result.positions = strategy_result.get('positions')
                result.trades = strategy_result.get('trades', [])
                
                # Apply transaction costs if not already included
                if not strategy_result.get('includes_transaction_costs', False):
                    result = await self._apply_transaction_costs(result)
                    
                # Calculate performance metrics
                result = await self._calculate_performance_metrics(result)
                
            else:
                # If strategy doesn't implement backtest, use our implementation
                result = await self._run_standard_backtest(strategy, data, symbol, initial_capital, risk_limit)
                
            # Set execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time
            
            # Publish completion event
            await self.event_bus.publish(EventType.BACKTEST_COMPLETED, {
                "backtest_id": result.id,
                "strategy": result.strategy_name,
                "symbol": result.symbol,
                "initial_capital": result.initial_capital,
                "final_capital": result.final_capital,
                "total_return": result.total_return,
                "sharpe_ratio": result.sharpe_ratio,
                "max_drawdown": result.max_drawdown,
                "execution_time": result.execution_time
            })
            
            # Save result to disk
            self._save_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            # Publish error event
            await self.event_bus.publish(EventType.ERROR_OCCURRED, {
                "component": "EnhancedBacktester",
                "operation": "run_backtest",
                "error": str(e),
                "strategy": strategy.__class__.__name__,
                "symbol": symbol
            })
            raise
    
    async def _run_standard_backtest(self,
                                    strategy: Any,
                                    data: pd.DataFrame,
                                    symbol: str,
                                    initial_capital: float,
                                    risk_limit: float) -> BacktestResult:
        """
        Run a standard backtest if strategy doesn't implement backtest
        
        Args:
            strategy: Strategy instance
            data: Market data DataFrame
            symbol: Trading symbol
            initial_capital: Initial capital
            risk_limit: Maximum percentage of capital per position
            
        Returns:
            BacktestResult
        """
        # Create result container
        result = BacktestResult(
            strategy_name=strategy.__class__.__name__,
            symbol=symbol,
            start_date=data['timestamp'].min() if 'timestamp' in data.columns else "unknown",
            end_date=data['timestamp'].max() if 'timestamp' in data.columns else "unknown",
            initial_capital=initial_capital
        )
        
        # Preprocess data if strategy implements it
        if hasattr(strategy, 'preprocess_data'):
            data = await strategy.preprocess_data(data)
            
        # Generate signals if strategy implements predict
        if hasattr(strategy, 'predict'):
            predictions = []
            for i in range(len(data)):
                # Simulate a rolling window prediction
                window = data.iloc[:i+1]
                if i > 0:  # Skip first row as we need some history
                    try:
                        action, confidence = await strategy.predict(window)
                        predictions.append({
                            'timestamp': data.iloc[i]['timestamp'] if 'timestamp' in data.columns else i,
                            'action': action,
                            'confidence': confidence
                        })
                    except Exception as e:
                        logger.error(f"Error in strategy prediction: {e}")
                        predictions.append({
                            'timestamp': data.iloc[i]['timestamp'] if 'timestamp' in data.columns else i,
                            'action': TradeAction.HOLD,
                            'confidence': 0.0
                        })
            
            # Convert predictions to DataFrame
            predictions_df = pd.DataFrame(predictions)
            
            # Merge with data
            if not predictions_df.empty:
                if 'timestamp' in data.columns:
                    merged_data = pd.merge(data, predictions_df, on='timestamp', how='left')
                else:
                    # Use index as merge key
                    data_copy = data.copy()
                    data_copy['index'] = data_copy.index
                    predictions_df['index'] = predictions_df.index
                    merged_data = pd.merge(data_copy, predictions_df, on='index', how='left')
                    
                # Fill NaN with HOLD
                merged_data['action'] = merged_data['action'].fillna(TradeAction.HOLD)
                merged_data['confidence'] = merged_data['confidence'].fillna(0.0)
                
                # Simulate trades
                result = await self._simulate_trades(merged_data, result, risk_limit)
            
        # Apply transaction costs
        result = await self._apply_transaction_costs(result)
        
        # Calculate performance metrics
        result = await self._calculate_performance_metrics(result)
        
        return result
    
    async def _simulate_trades(self, 
                              data: pd.DataFrame, 
                              result: BacktestResult,
                              risk_limit: float) -> BacktestResult:
        """
        Simulate trades based on signals
        
        Args:
            data: DataFrame with signals
            result: BacktestResult to update
            risk_limit: Maximum percentage of capital per position
            
        Returns:
            Updated BacktestResult
        """
        equity = result.initial_capital
        position = 0
        trades = []
        equity_curve = []
        
        for i in range(len(data)):
            row = data.iloc[i]
            timestamp = row['timestamp'] if 'timestamp' in data.columns else i
            price = row['close'] if 'close' in row else row.get('price', 0)
            action = row.get('action', TradeAction.HOLD)
            confidence = row.get('confidence', 0.0)
            
            # Skip if price is 0 or NaN
            if price <= 0 or pd.isna(price):
                continue
                
            # Calculate position value
            position_value = position * price
            
            # Calculate unrealized P&L
            unrealized_pnl = 0
            if i > 0 and position != 0:
                prev_price = data.iloc[i-1]['close'] if 'close' in data.columns else data.iloc[i-1].get('price', 0)
                if prev_price > 0 and not pd.isna(prev_price):
                    unrealized_pnl = position * (price - prev_price)
                    
            # Update equity with unrealized P&L
            equity += unrealized_pnl
            
            # Process trading action
            if action in [TradeAction.BUY, TradeAction.SELL] and confidence > 0.5:
                # Calculate max position size based on risk limit
                max_position_size = (equity * risk_limit) / price
                
                # Scale by confidence
                target_position = max_position_size if action == TradeAction.BUY else -max_position_size
                target_position *= confidence
                
                # Calculate quantity to trade (difference from current position)
                quantity = target_position - position
                
                if abs(quantity) > 0:
                    # Record trade
                    trade = {
                        'timestamp': timestamp,
                        'price': price,
                        'action': action.name,
                        'quantity': quantity,
                        'value': quantity * price,
                        'confidence': confidence,
                        'equity_before': equity
                    }
                    trades.append(trade)
                    
                    # Update position
                    position += quantity
            
            # Record equity curve point
            equity_point = {
                'timestamp': timestamp,
                'equity': equity,
                'position': position,
                'position_value': position_value,
                'price': price
            }
            equity_curve.append(equity_point)
            
        # Convert lists to DataFrames
        result.equity_curve = pd.DataFrame(equity_curve)
        result.trades = trades
        result.final_capital = equity
        
        return result
    
    async def _apply_transaction_costs(self, result: BacktestResult) -> BacktestResult:
        """
        Apply transaction costs to backtest result
        
        Args:
            result: BacktestResult to update
            
        Returns:
            Updated BacktestResult with transaction costs
        """
        total_costs = 0.0
        
        # Apply costs to each trade
        for i, trade in enumerate(result.trades):
            price = trade.get('price', 0)
            quantity = trade.get('quantity', 0)
            
            if price > 0 and quantity != 0:
                # Get volatility if available (for volatility-based slippage)
                volatility = None
                if result.equity_curve is not None and 'atr_14' in result.equity_curve.columns:
                    timestamp = trade.get('timestamp')
                    if timestamp is not None:
                        # Find matching row in equity curve
                        if 'timestamp' in result.equity_curve.columns:
                            matching_rows = result.equity_curve[result.equity_curve['timestamp'] == timestamp]
                            if not matching_rows.empty:
                                volatility = matching_rows['atr_14'].iloc[0]
                
                # Calculate cost
                cost = self.transaction_cost_model.calculate_transaction_cost(price, quantity, volatility)
                
                # Update trade
                result.trades[i]['transaction_cost'] = cost
                
                # Deduct from final capital
                result.final_capital -= cost
                
                # Add to total costs
                total_costs += cost
        
        # Update equity curve if available
        if result.equity_curve is not None:
            # Create a series of cumulative costs
            cumulative_costs = [0.0] * len(result.equity_curve)
            
            for trade in result.trades:
                timestamp = trade.get('timestamp')
                cost = trade.get('transaction_cost', 0.0)
                
                if timestamp is not None and cost > 0:
                    # Find index in equity curve
                    if 'timestamp' in result.equity_curve.columns:
                        indices = result.equity_curve.index[result.equity_curve['timestamp'] == timestamp].tolist()
                        if indices:
                            idx = indices[0]
                            # Update all subsequent points
                            for i in range(idx, len(cumulative_costs)):
                                cumulative_costs[i] += cost
            
            # Add cumulative costs to equity curve
            result.equity_curve['cumulative_costs'] = cumulative_costs
            
            # Adjust equity for costs
            result.equity_curve['adjusted_equity'] = result.equity_curve['equity'] - result.equity_curve['cumulative_costs']
        
        # Update result
        result.transaction_costs = total_costs
        
        return result
    
    async def _calculate_performance_metrics(self, result: BacktestResult) -> BacktestResult:
        """
        Calculate performance metrics for backtest result
        
        Args:
            result: BacktestResult to update
            
        Returns:
            Updated BacktestResult with performance metrics
        """
        # Skip if no equity curve
        if result.equity_curve is None or len(result.equity_curve) == 0:
            return result
            
        # Use adjusted equity if available
        if 'adjusted_equity' in result.equity_curve.columns:
            equity_col = 'adjusted_equity'
        else:
            equity_col = 'equity'
            
        # Calculate returns
        equity_series = result.equity_curve[equity_col]
        result.daily_returns = equity_series.pct_change().dropna()
        
        # Total return
        result.total_return = (result.final_capital / result.initial_capital) - 1
        
        # Annualized return (assuming 252 trading days per year)
        n_days = len(result.equity_curve)
        if n_days > 1:
            result.annual_return = (1 + result.total_return) ** (252 / n_days) - 1
        
        # Sharpe ratio (assuming 0% risk-free rate)
        if len(result.daily_returns) > 0:
            sharpe = result.daily_returns.mean() / result.daily_returns.std() if result.daily_returns.std() > 0 else 0
            result.sharpe_ratio = sharpe * (252 ** 0.5)  # Annualized
        
        # Maximum drawdown
        peak = equity_series.expanding(min_periods=1).max()
        drawdown = (equity_series / peak) - 1
        result.max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
        
        # Trade statistics
        if result.trades:
            # Count trades
            result.trade_count = len(result.trades)
            
            # Win rate
            wins = sum(1 for t in result.trades if 
                      (t.get('action') == 'BUY' and t.get('price', 0) < result.equity_curve['price'].iloc[-1]) or
                      (t.get('action') == 'SELL' and t.get('price', 0) > result.equity_curve['price'].iloc[-1]))
            result.win_rate = wins / result.trade_count if result.trade_count > 0 else 0
            
            # Profit factor
            gross_profit = sum(t.get('value', 0) for t in result.trades if t.get('value', 0) > 0)
            gross_loss = sum(abs(t.get('value', 0)) for t in result.trades if t.get('value', 0) < 0)
            result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        return result
    
    def _save_result(self, result: BacktestResult) -> None:
        """
        Save backtest result to disk
        
        Args:
            result: BacktestResult to save
        """
        # Convert to dict
        result_dict = result.to_dict()
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{result.strategy_name}_{result.symbol}_{timestamp}.json"
        filepath = self.results_dir / filename
        
        # Save to disk
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2)
            
        logger.info(f"Saved backtest result to {filepath}")


# Example usage
"""
# Create backtester
backtester = EnhancedBacktester()

# Set custom transaction cost model
cost_model = TransactionCostModel(
    percentage_fee=0.001,  # 0.1%
    fixed_fee=1.0,        # $1.00 per trade
    minimum_fee=1.0,      # Minimum $1.00
    slippage_model="volatility",
    slippage_value=0.1    # 10% of volatility
)
backtester.set_transaction_cost_model(cost_model)

# Get strategy and data
from app.services.strategy_manager import StrategyManager
from app.services.market_data import MarketDataService

strategy_manager = StrategyManager()
market_data = MarketDataService()

# Get strategy
strategy = await strategy_manager.get_strategy("MovingAverageStrategy")

# Get data
data = await market_data.get_historical_data("AAPL", "2023-01-01", "2023-06-30")

# Run backtest
result = await backtester.run_backtest(
    strategy=strategy,
    data=data,
    symbol="AAPL",
    initial_capital=10000.0,
    risk_limit=0.02
)

# Print summary
print(f"Total Return: {result.total_return:.2%}")
print(f"Sharpe Ratio: {result.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.max_drawdown:.2%}")
print(f"Win Rate: {result.win_rate:.2%}")
print(f"Transaction Costs: ${result.transaction_costs:.2f}")
"""
