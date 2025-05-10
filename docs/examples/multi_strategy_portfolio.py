"""
Test script for the Multi-Strategy Portfolio example from README.md
"""
import os
import sys
import asyncio
import logging
from datetime import datetime, timedelta

# Add the project root to Python path so we can import app modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required services and strategies
from app.services.market_data import MarketDataService
from app.strategies.moving_average import MovingAverageStrategy
from app.strategies.lstm_predictor import LSTMPredictorStrategy
from app.strategies.transformer_strategy import TransformerStrategy
from app.services.backtesting import BacktestingService

class PortfolioManager:
    """
    A simplified portfolio manager for testing the example from README.md
    """
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.strategies = []
        self.market_data = MarketDataService()
        self.backtesting_service = BacktestingService()
        
    def add_strategy(self, strategy, symbol, allocation=0.3):
        self.strategies.append({
            "strategy": strategy,
            "symbol": symbol,
            "allocation": allocation
        })
        
    async def backtest(self, start_date, end_date):
        """Run backtest for all strategies in the portfolio"""
        total_return = 0
        results_by_strategy = {}
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        for strategy_info in self.strategies:
            strategy = strategy_info["strategy"]
            symbol = strategy_info["symbol"]
            allocation = strategy_info["allocation"]
            capital = self.initial_capital * allocation
            
            logger.info(f"Running backtest for {symbol} with {strategy.__class__.__name__}")
            
            # Get historical data
            data = await self.market_data.get_historical_data(symbol, start, end)
            
            # Run backtest
            backtest_result = await self.backtesting_service.run_backtest(
                strategy=strategy,
                symbol=symbol,
                start_date=start,
                end_date=end,
                initial_capital=capital
            )
            
            # Store results
            strategy_return = backtest_result.get("total_return", 0)
            strategy_contribution = strategy_return * allocation
            total_return += strategy_contribution
            
            results_by_strategy[strategy.__class__.__name__] = {
                "symbol": symbol,
                "allocation": allocation,
                "return": strategy_return,
                "contribution": strategy_contribution
            }
            
            logger.info(f"Strategy {strategy.__class__.__name__} on {symbol}: Return={strategy_return:.2f}%, Contribution={strategy_contribution:.2f}%")
        
        return {
            "total_return": total_return,
            "strategies": results_by_strategy
        }

async def create_diversified_portfolio():
    """
    This is the example from README.md
    """
    portfolio = PortfolioManager(initial_capital=10000)
    
    # Add different strategies
    portfolio.add_strategy(MovingAverageStrategy(10, 30), "AAPL", allocation=0.3)
    portfolio.add_strategy(LSTMPredictorStrategy(), "MSFT", allocation=0.3)
    portfolio.add_strategy(TransformerStrategy(), "GOOGL", allocation=0.4)
    
    # Backtest the portfolio
    results = await portfolio.backtest("2024-01-01", "2024-03-01")
    print(f"\nPortfolio Return: {results['total_return']:.2f}%")
    
    # Print individual strategy contributions
    print("\nStrategy Contributions:")
    for strategy_name, strategy_result in results['strategies'].items():
        print(f"- {strategy_name} ({strategy_result['symbol']}): {strategy_result['contribution']:.2f}%")

if __name__ == "__main__":
    # Run the example
    asyncio.run(create_diversified_portfolio())
