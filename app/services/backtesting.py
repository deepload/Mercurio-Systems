"""
Backtesting Service

Provides functionality for backtesting trading strategies on historical data.
"""
import os
import logging
import tempfile
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
import backtrader as bt
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from app.strategies.base import BaseStrategy
from app.db.models import BacktestResult
from app.services.market_data import MarketDataService

logger = logging.getLogger(__name__)

class BacktestingService:
    """
    Service for running backtests on trading strategies.
    
    Supports:
    - Backtesting using strategy implementations
    - Backtesting with Backtrader for more detailed analysis
    - Performance metrics calculation
    """
    
    def __init__(self):
        """Initialize the backtesting service"""
        self.market_data = MarketDataService()
    
    async def run_backtest(
        self,
        strategy: BaseStrategy,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Run a backtest for a strategy on historical data.
        
        Args:
            strategy: Strategy instance
            symbol: The market symbol (e.g., 'AAPL')
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital for the backtest
            
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest for {symbol} with {strategy.__class__.__name__}")
        
        try:
            # Load historical data
            data = await strategy.load_data(symbol, start_date, end_date)
            
            # Preprocess the data
            processed_data = await strategy.preprocess_data(data)
            
            # Run backtest using strategy's implementation
            results = await strategy.backtest(processed_data, initial_capital)
            
            # Generate performance charts
            charts = await self._generate_backtest_charts(
                results.get("backtest_data", processed_data),
                strategy.__class__.__name__,
                symbol
            )
            
            # Combine results with charts
            results["charts"] = charts
            
            # Add metadata
            results["strategy"] = strategy.__class__.__name__
            results["symbol"] = symbol
            results["start_date"] = start_date.isoformat()
            results["end_date"] = end_date.isoformat()
            results["initial_capital"] = initial_capital
            
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return {"error": str(e)}
    
    async def run_backtest_with_backtrader(
        self,
        strategy_class: Any,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        initial_capital: float = 10000.0,
        strategy_params: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run a backtest using Backtrader framework for detailed analysis.
        
        Args:
            strategy_class: Backtrader strategy class
            symbol: The market symbol (e.g., 'AAPL')
            start_date: Start date for backtest
            end_date: End date for backtest
            initial_capital: Initial capital for the backtest
            strategy_params: Strategy parameters
            
        Returns:
            Dictionary with backtest results
        """
        if strategy_params is None:
            strategy_params = {}
        
        logger.info(f"Running Backtrader backtest for {symbol}")
        
        try:
            # Create a Backtrader cerebro engine
            cerebro = bt.Cerebro()
            
            # Add the strategy
            cerebro.addstrategy(strategy_class, **strategy_params)
            
            # Set initial cash
            cerebro.broker.setcash(initial_capital)
            
            # Set realistic commission
            cerebro.broker.setcommission(commission=0.001)  # 0.1% commission
            
            # Load historical data
            data = await self.market_data.get_historical_data(symbol, start_date, end_date)
            
            # Create a Backtrader data feed
            feed = self._create_backtrader_feed(data, symbol)
            
            # Add the data feed to cerebro
            cerebro.adddata(feed)
            
            # Add analyzers
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            
            # Run the backtest
            results = cerebro.run()
            strategy_result = results[0]
            
            # Extract metrics from analyzers
            sharpe_ratio = strategy_result.analyzers.sharpe.get_analysis().get('sharperatio', 0)
            drawdown = strategy_result.analyzers.drawdown.get_analysis()
            returns = strategy_result.analyzers.returns.get_analysis()
            trades = strategy_result.analyzers.trades.get_analysis()
            
            # Calculate performance metrics
            max_drawdown = drawdown.get('max', {}).get('drawdown', 0)
            total_return = returns.get('rtot', 0)
            final_capital = cerebro.broker.getvalue()
            
            # Generate performance chart
            chart = self._generate_backtrader_chart(cerebro)
            
            # Format the results
            backtest_results = {
                "strategy": strategy_class.__name__,
                "symbol": symbol,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "initial_capital": initial_capital,
                "final_capital": final_capital,
                "total_return": total_return,
                "annualized_return": returns.get('ravg', 0) * 252,
                "sharpe_ratio": sharpe_ratio,
                "max_drawdown": max_drawdown,
                "trades": trades.get('total', {}).get('total', 0),
                "win_rate": trades.get('won', {}).get('total', 0) / trades.get('total', {}).get('total', 1),
                "charts": {"backtrader_chart": chart}
            }
            
            return backtest_results
            
        except Exception as e:
            logger.error(f"Error running Backtrader backtest: {e}")
            return {"error": str(e)}
    
    def _create_backtrader_feed(self, data: pd.DataFrame, symbol: str) -> bt.feeds.PandasData:
        """
        Create a Backtrader data feed from a pandas DataFrame.
        
        Args:
            data: Historical price data
            symbol: Symbol name
            
        Returns:
            Backtrader data feed
        """
        # Ensure column names are lowercase
        data.columns = [col.lower() for col in data.columns]
        
        # Create a data feed
        feed = bt.feeds.PandasData(
            dataname=data,
            name=symbol,
            datetime=None,  # Use index for date
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=-1  # Not used
        )
        
        return feed
    
    def _generate_backtrader_chart(self, cerebro: bt.Cerebro) -> str:
        """
        Generate a chart from Backtrader cerebro.
        
        Args:
            cerebro: Backtrader cerebro instance
            
        Returns:
            Base64-encoded chart image
        """
        # Create a temporary file for the plot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmpfile:
            # Plot and save to the temporary file
            cerebro.plot(style='candlestick', barup='green', bardown='red', 
                        grid=True, volume=True, savefig=dict(fname=tmpfile.name, dpi=300))
            
            # Read the file and encode to base64
            with open(tmpfile.name, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
            
            # Delete the temporary file
            tmpfile.close()
            os.unlink(tmpfile.name)
        
        return img_data
    
    async def _generate_backtest_charts(
        self,
        data: pd.DataFrame,
        strategy_name: str,
        symbol: str
    ) -> Dict[str, str]:
        """
        Generate performance charts for backtest results.
        
        Args:
            data: Backtest data with returns
            strategy_name: Name of the strategy
            symbol: Symbol name
            
        Returns:
            Dictionary with base64-encoded chart images
        """
        charts = {}
        
        try:
            # Ensure we have the required columns
            if not all(col in data.columns for col in ['cumulative_returns', 'cumulative_strategy_returns']):
                return charts
            
            # Create equity curve chart
            plt.figure(figsize=(12, 6))
            plt.plot(data.index, data['cumulative_returns'], label=f'{symbol} Buy & Hold')
            plt.plot(data.index, data['cumulative_strategy_returns'], label=f'{strategy_name}')
            plt.title(f'Equity Curve: {symbol} - {strategy_name}')
            plt.xlabel('Date')
            plt.ylabel('Growth of $1')
            plt.grid(True)
            plt.legend()
            
            # Save the chart to a buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            
            # Encode the chart as base64
            equity_curve = base64.b64encode(buffer.getvalue()).decode('utf-8')
            charts['equity_curve'] = equity_curve
            
            plt.close()
            
            # Create drawdown chart if we have drawdown data
            if 'drawdown' in data.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(data.index, data['drawdown'] * 100)
                plt.title(f'Drawdown: {symbol} - {strategy_name}')
                plt.xlabel('Date')
                plt.ylabel('Drawdown (%)')
                plt.grid(True)
                plt.fill_between(data.index, data['drawdown'] * 100, 0, alpha=0.3, color='red')
                
                # Save the chart to a buffer
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=100)
                buffer.seek(0)
                
                # Encode the chart as base64
                drawdown_chart = base64.b64encode(buffer.getvalue()).decode('utf-8')
                charts['drawdown'] = drawdown_chart
                
                plt.close()
            
        except Exception as e:
            logger.error(f"Error generating backtest charts: {e}")
        
        return charts
