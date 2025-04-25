"""
MercurioAI Paper Trading Test Suite

This script runs all available strategies in paper trading mode,
allowing you to evaluate and compare their performance before using real funds.

Features:
- Tests all strategies in MercurioAI
- Configurable test duration and parameters
- Generates performance reports
- Logs all trade activities
"""
import os
import sys
import asyncio
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import json
import argparse
import importlib
import inspect
from typing import Dict, List, Any, Tuple, Optional, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("paper_trading_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Ensure app directory is in path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class PaperTradingTest:
    """
    Paper trading test runner for MercurioAI strategies
    """
    
    def __init__(self, 
                config_path: str = "config/paper_test_config.json",
                initial_capital: float = 10000.0,
                test_duration_hours: float = 24.0,
                symbols: List[str] = None,
                strategies: List[str] = None,
                risk_profile: str = "conservative"):
        """
        Initialize paper trading test
        
        Args:
            config_path: Path to configuration file
            initial_capital: Initial capital for paper trading
            test_duration_hours: Test duration in hours
            symbols: List of symbols to trade
            strategies: List of strategies to test (None = all)
            risk_profile: Risk profile to use
        """
        self.config_path = config_path
        self.initial_capital = initial_capital
        self.test_duration_hours = test_duration_hours
        self.symbols = symbols or ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
        self.strategy_names = strategies
        self.risk_profile = risk_profile
        
        # Load configuration
        self.config = self._load_config()
        
        # Strategy instances
        self.strategies = {}
        
        # Trading services
        self.market_data_service = None
        self.trading_service = None
        
        # Risk manager
        self.risk_manager = None
        
        # Test results
        self.results = {}
        
    def _load_config(self) -> Dict[str, Any]:
        """Load test configuration"""
        # Default configuration
        default_config = {
            "initial_capital": self.initial_capital,
            "test_duration_hours": self.test_duration_hours,
            "symbols": self.symbols,
            "update_interval_seconds": 60,
            "risk_profiles": {
                "conservative": {
                    "max_position_size": 0.02,
                    "max_drawdown": 0.10,
                    "max_daily_loss": 0.03,
                    "position_scaling": "volatility",
                    "stop_loss_pct": 0.03,
                    "take_profit_pct": 0.09
                },
                "moderate": {
                    "max_position_size": 0.05,
                    "max_drawdown": 0.15,
                    "max_daily_loss": 0.05,
                    "position_scaling": "volatility",
                    "stop_loss_pct": 0.05,
                    "take_profit_pct": 0.15
                },
                "aggressive": {
                    "max_position_size": 0.10,
                    "max_drawdown": 0.25,
                    "max_daily_loss": 0.08,
                    "position_scaling": "fixed",
                    "stop_loss_pct": 0.08,
                    "take_profit_pct": 0.24
                }
            },
            "strategy_params": {
                "MovingAverageStrategy": {
                    "short_window": 20,
                    "long_window": 50
                },
                "RSIStrategy": {
                    "rsi_period": 14,
                    "oversold_threshold": 30,
                    "overbought_threshold": 70
                },
                "LSTMPredictorStrategy": {
                    "sequence_length": 20,
                    "prediction_horizon": 5,
                    "epochs": 50
                },
                "TransformerStrategy": {
                    "sequence_length": 30,
                    "d_model": 32,
                    "nhead": 4,
                    "num_layers": 2,
                    "epochs": 20
                },
                "LLMStrategy": {
                    "model_name": "llama2-7b",
                    "use_local_model": False,
                    "news_lookback_hours": 24
                }
            }
        }
        
        # Try to load configuration file
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    
                # Merge with default config
                for key, value in config.items():
                    if isinstance(value, dict) and key in default_config and isinstance(default_config[key], dict):
                        default_config[key].update(value)
                    else:
                        default_config[key] = value
                        
                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Configuration file {self.config_path} not found, using defaults")
                
                # Save default config for future use
                os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
                with open(self.config_path, 'w') as f:
                    json.dump(default_config, f, indent=4)
                    
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            
        return default_config
    
    async def initialize(self):
        """Initialize services and strategies"""
        logger.info("Initializing paper trading test")
        
        # Initialize market data service
        from app.services.market_data import MarketDataService
        self.market_data_service = MarketDataService()
        
        # Initialize trading service (paper mode)
        from app.services.trading import TradingService
        self.trading_service = TradingService(is_paper=True)
        
        # Initialize risk manager
        from app.core.risk_manager import RiskProfile, PortfolioRiskManager
        
        risk_profile_config = self.config["risk_profiles"].get(
            self.risk_profile, self.config["risk_profiles"]["conservative"]
        )
        
        risk_profile = RiskProfile(
            name=self.risk_profile,
            **risk_profile_config
        )
        
        self.risk_manager = PortfolioRiskManager(risk_profile)
        
        # Initialize strategies
        await self._initialize_strategies()
        
    async def _initialize_strategies(self):
        """Initialize all available strategies or specified ones"""
        # Get available strategy classes
        strategy_classes = await self._discover_strategy_classes()
        
        # Filter by requested strategies if specified
        if self.strategy_names:
            strategy_classes = {name: cls for name, cls in strategy_classes.items() 
                               if name in self.strategy_names}
        
        # Initialize each strategy
        for name, cls in strategy_classes.items():
            try:
                # Get strategy parameters
                params = self.config["strategy_params"].get(name, {})
                
                # Create strategy instance
                logger.info(f"Initializing strategy: {name}")
                strategy = cls(**params)
                
                # Add to strategies dict
                self.strategies[name] = strategy
                
            except Exception as e:
                logger.error(f"Error initializing strategy {name}: {e}")
                
        logger.info(f"Initialized {len(self.strategies)} strategies")
    
    async def _discover_strategy_classes(self) -> Dict[str, Any]:
        """Discover all strategy classes in MercurioAI"""
        from app.strategies.base import BaseStrategy
        
        strategy_classes = {}
        
        # Import strategy modules
        strategy_dir = Path("app/strategies")
        if not strategy_dir.exists():
            logger.error(f"Strategy directory {strategy_dir} not found")
            return strategy_classes
            
        # Find all Python files in strategy directory
        for file_path in strategy_dir.glob("*.py"):
            if file_path.name == "__init__.py" or file_path.name == "base.py":
                continue
                
            try:
                # Construct module name and import
                module_name = f"app.strategies.{file_path.stem}"
                module = importlib.import_module(module_name)
                
                # Find all classes in the module
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    # Only include classes defined in this module and derived from BaseStrategy
                    if (obj.__module__ == module_name and 
                        issubclass(obj, BaseStrategy) and 
                        obj != BaseStrategy):
                        strategy_classes[name] = obj
                        
            except Exception as e:
                logger.error(f"Error importing strategy from {file_path}: {e}")
                
        return strategy_classes
        
    async def run_test(self):
        """Run the paper trading test"""
        if not self.strategies:
            logger.error("No strategies initialized, cannot run test")
            return
            
        logger.info(f"Starting paper trading test with {len(self.strategies)} strategies")
        logger.info(f"Test duration: {self.test_duration_hours} hours")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        
        # Initialize test start time
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=self.test_duration_hours)
        
        # Initialize performance tracking for each strategy
        performance = {name: {
            'initial_capital': self.initial_capital,
            'current_capital': self.initial_capital,
            'positions': {},
            'trades': [],
            'equity_curve': []
        } for name in self.strategies}
        
        # Initialize trade tracking
        active_trades = {name: {} for name in self.strategies}
        
        # Main test loop
        current_time = start_time
        update_interval = self.config.get("update_interval_seconds", 60)
        
        while current_time < end_time:
            # Update current time
            current_time = datetime.now()
            
            logger.info(f"Test time: {current_time}, {(end_time - current_time).total_seconds() / 3600:.2f} hours remaining")
            
            # Process each symbol
            for symbol in self.symbols:
                try:
                    # Get historical data for recent period (last 7 days)
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=7)  # Get a week of data
                    
                    # Pass datetime objects directly, not strings
                    data = await self.market_data_service.get_historical_data(symbol, start_date, end_date)
                    
                    if data is None or data.empty or len(data) < 20:
                        logger.warning(f"Insufficient data for {symbol}, skipping")
                        continue
                        
                    # Current price (most recent data point)
                    current_price = data['close'].iloc[-1]
                    
                    # Process each strategy
                    for name, strategy in self.strategies.items():
                        try:
                            # Get prediction
                            action, confidence = await strategy.predict(data)
                            
                            # Get current portfolio state
                            portfolio = performance[name]
                            positions = portfolio['positions']
                            current_position = positions.get(symbol, 0)
                            
                            # Apply risk management
                            adjusted_action, position_size = self._apply_risk_management(
                                name, symbol, action, confidence, current_price, 
                                portfolio, current_position
                            )
                            
                            # Execute trade
                            if adjusted_action != 'HOLD' and position_size > 0:
                                # Log trade
                                trade = {
                                    'timestamp': current_time,
                                    'symbol': symbol,
                                    'action': adjusted_action,
                                    'price': current_price,
                                    'size': position_size,
                                    'confidence': confidence
                                }
                                
                                portfolio['trades'].append(trade)
                                
                                # Update positions and capital
                                if adjusted_action == 'BUY':
                                    # Calculate cost
                                    cost = position_size * current_price
                                    
                                    # Update position
                                    positions[symbol] = positions.get(symbol, 0) + position_size
                                    portfolio['current_capital'] -= cost
                                    
                                    logger.info(f"Strategy {name}: BUY {position_size} {symbol} at {current_price}")
                                    
                                elif adjusted_action == 'SELL':
                                    # Calculate revenue
                                    revenue = position_size * current_price
                                    
                                    # Update position
                                    positions[symbol] = positions.get(symbol, 0) - position_size
                                    portfolio['current_capital'] += revenue
                                    
                                    logger.info(f"Strategy {name}: SELL {position_size} {symbol} at {current_price}")
                            
                            # Update equity curve
                            equity = portfolio['current_capital']
                            for sym, pos in positions.items():
                                # Get price for this symbol
                                if sym == symbol:
                                    sym_price = current_price
                                else:
                                    # Use last known price
                                    sym_data = await self.market_data_service.get_latest_data(sym)
                                    sym_price = sym_data['close'].iloc[-1] if sym_data is not None and len(sym_data) > 0 else 0
                                    
                                equity += pos * sym_price
                                
                            portfolio['equity_curve'].append({
                                'timestamp': current_time,
                                'equity': equity
                            })
                            
                        except Exception as e:
                            logger.error(f"Error processing strategy {name} for {symbol}: {e}")
                    
                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {e}")
                
            # Sleep until next update
            await asyncio.sleep(update_interval)
        
        # Test completed
        logger.info("Paper trading test completed")
        
        # Calculate final results
        await self._calculate_results(performance)
        
        return self.results
    
    def _apply_risk_management(self, 
                               strategy_name: str, 
                               symbol: str,
                               action: str, 
                               confidence: float, 
                               price: float,
                               portfolio: Dict[str, Any],
                               current_position: float) -> Tuple[str, float]:
        """
        Apply risk management rules to determine position size
        
        Args:
            strategy_name: Strategy name
            symbol: Trading symbol
            action: Trading action
            confidence: Signal confidence
            price: Current price
            portfolio: Strategy portfolio
            current_position: Current position for this symbol
            
        Returns:
            Adjusted action and position size
        """
        # Convert action to string if it's an enum
        action_str = action.name if hasattr(action, 'name') else str(action)
        
        # Default position size (no trade)
        position_size = 0
        
        # Adjust confidence to confidence * 2 - 1 to get -1 to 1 range
        # This makes SELL signals have negative confidence
        adjusted_confidence = confidence * 2 - 1 if action_str == 'BUY' else -(confidence * 2 - 1) if action_str == 'SELL' else 0
        
        # Only trade if confidence is above threshold
        min_confidence = 0.6
        if abs(adjusted_confidence) < min_confidence:
            return 'HOLD', 0
            
        # Get available capital
        available_capital = portfolio['current_capital']
        
        # Don't trade if not enough capital
        if available_capital <= 0 and action_str == 'BUY':
            return 'HOLD', 0
            
        # Get risk profile
        risk_profile_config = self.config["risk_profiles"].get(
            self.risk_profile, self.config["risk_profiles"]["conservative"]
        )
        
        # Calculate position size based on risk profile
        max_position_size = risk_profile_config.get("max_position_size", 0.02)
        
        # For simplicity, use a fixed percentage of capital
        if action_str == 'BUY':
            # Buy using percentage of available capital
            trade_amount = available_capital * max_position_size
            position_size = trade_amount / price
            
        elif action_str == 'SELL':
            # If we have a position, sell a percentage of it
            if current_position > 0:
                position_size = current_position * 0.5  # Sell half the position
            else:
                # Short selling - use same logic as buying
                trade_amount = available_capital * max_position_size
                position_size = trade_amount / price
        
        # Ensure position size is valid
        position_size = max(0, position_size)
        
        return action_str, position_size
    
    async def _calculate_results(self, performance: Dict[str, Any]):
        """
        Calculate final results for all strategies
        
        Args:
            performance: Performance data for all strategies
        """
        results = {}
        
        for name, data in performance.items():
            initial_capital = data['initial_capital']
            
            # Convert equity curve to DataFrame
            if data['equity_curve']:
                equity_df = pd.DataFrame(data['equity_curve'])
                
                # Calculate metrics
                final_equity = equity_df['equity'].iloc[-1] if len(equity_df) > 0 else initial_capital
                
                # Return metrics
                total_return = (final_equity / initial_capital) - 1
                annualized_return = self._calculate_annualized_return(
                    initial_capital, final_equity, self.test_duration_hours / 24
                )
                
                # Risk metrics
                max_drawdown = self._calculate_max_drawdown(equity_df['equity'])
                sharpe_ratio = self._calculate_sharpe_ratio(equity_df['equity'])
                
                # Trade metrics
                num_trades = len(data['trades'])
                win_rate = 0
                
                if num_trades > 0:
                    # Calculate win rate by analyzing trades
                    winning_trades = 0
                    for i, trade in enumerate(data['trades']):
                        # Skip last trade if no next trade to compare
                        if i == len(data['trades']) - 1:
                            continue
                            
                        current_trade = trade
                        next_trade = data['trades'][i + 1]
                        
                        # Only count if same symbol and opposite action
                        if (current_trade['symbol'] == next_trade['symbol'] and
                            ((current_trade['action'] == 'BUY' and next_trade['action'] == 'SELL') or
                             (current_trade['action'] == 'SELL' and next_trade['action'] == 'BUY'))):
                            
                            # Check if profitable
                            if ((current_trade['action'] == 'BUY' and next_trade['price'] > current_trade['price']) or
                                (current_trade['action'] == 'SELL' and next_trade['price'] < current_trade['price'])):
                                winning_trades += 1
                                
                    win_rate = winning_trades / (num_trades / 2) if num_trades > 1 else 0
                
                # Store results
                results[name] = {
                    'initial_capital': initial_capital,
                    'final_equity': final_equity,
                    'total_return': total_return,
                    'annualized_return': annualized_return,
                    'max_drawdown': max_drawdown,
                    'sharpe_ratio': sharpe_ratio,
                    'num_trades': num_trades,
                    'win_rate': win_rate,
                    'positions': data['positions'],
                    'test_duration_hours': self.test_duration_hours
                }
                
                logger.info(f"Strategy {name} results:")
                logger.info(f"  Initial capital: ${initial_capital:.2f}")
                logger.info(f"  Final equity: ${final_equity:.2f}")
                logger.info(f"  Total return: {total_return:.2%}")
                logger.info(f"  Annualized return: {annualized_return:.2%}")
                logger.info(f"  Max drawdown: {max_drawdown:.2%}")
                logger.info(f"  Sharpe ratio: {sharpe_ratio:.4f}")
                logger.info(f"  Number of trades: {num_trades}")
                logger.info(f"  Win rate: {win_rate:.2%}")
            else:
                logger.warning(f"No equity data for strategy {name}")
                results[name] = {
                    'initial_capital': initial_capital,
                    'final_equity': initial_capital,
                    'total_return': 0,
                    'error': 'No trading activity'
                }
                
        self.results = results
                
    def _calculate_annualized_return(self, initial_value: float, final_value: float, days: float) -> float:
        """Calculate annualized return"""
        if days <= 0 or initial_value <= 0:
            return 0
            
        total_return = (final_value / initial_value) - 1
        years = days / 365
        
        if years < 0.01:  # Avoid very short periods that could lead to extreme numbers
            # For very short periods, simply annualize linearly
            return total_return * (1 / years)
            
        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return
        
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if len(equity_series) <= 1:
            return 0
            
        # Calculate running maximum
        running_max = equity_series.cummax()
        
        # Calculate drawdown
        drawdown = (equity_series / running_max) - 1
        
        # Get maximum drawdown (will be negative)
        max_drawdown = drawdown.min()
        
        return abs(max_drawdown)
        
    def _calculate_sharpe_ratio(self, equity_series: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if len(equity_series) <= 1:
            return 0
            
        # Calculate returns
        returns = equity_series.pct_change().dropna()
        
        if len(returns) <= 1:
            return 0
            
        # Calculate annualized Sharpe ratio
        # Assuming values are daily returns
        risk_free_rate = 0.02 / 365  # 2% annual risk-free rate
        
        excess_returns = returns - risk_free_rate
        
        if excess_returns.std() == 0:
            return 0
            
        sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)  # Annualized
        
        return sharpe
        
    async def generate_report(self, output_file: str = "paper_trading_test_results.json"):
        """
        Generate a detailed report of test results
        
        Args:
            output_file: Output file for the report
        """
        if not self.results:
            logger.error("No results to report")
            return
            
        try:
            # Save results to JSON file
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=4)
                
            logger.info(f"Results saved to {output_file}")
            
            # Create a comparison table
            comparison = []
            
            for name, results in self.results.items():
                comparison.append({
                    'Strategy': name,
                    'Return (%)': f"{results.get('total_return', 0) * 100:.2f}%",
                    'Ann. Return (%)': f"{results.get('annualized_return', 0) * 100:.2f}%",
                    'Max Drawdown (%)': f"{results.get('max_drawdown', 0) * 100:.2f}%",
                    'Sharpe Ratio': f"{results.get('sharpe_ratio', 0):.4f}",
                    'Trades': results.get('num_trades', 0),
                    'Win Rate (%)': f"{results.get('win_rate', 0) * 100:.2f}%"
                })
                
            comparison_df = pd.DataFrame(comparison)
            
            # Sort by annualized return
            comparison_df = comparison_df.sort_values(by='Ann. Return (%)', ascending=False)
            
            # Print comparison table
            logger.info("\nStrategy Comparison:")
            logger.info(comparison_df.to_string(index=False))
            
            # Show best strategy
            best_strategy = comparison_df.iloc[0]['Strategy']
            logger.info(f"\nBest performing strategy: {best_strategy}")
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")


async def main():
    """Run the paper trading test"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="MercurioAI Paper Trading Test")
    
    parser.add_argument("--config", type=str, default="config/paper_test_config.json",
                        help="Path to configuration file")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Initial capital for paper trading")
    parser.add_argument("--duration", type=float, default=24.0,
                        help="Test duration in hours")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                        help="Symbols to trade")
    parser.add_argument("--strategies", type=str, nargs="+", default=None,
                        help="Strategies to test (None = all)")
    parser.add_argument("--risk", type=str, default="conservative",
                        choices=["conservative", "moderate", "aggressive"],
                        help="Risk profile to use")
    parser.add_argument("--output", type=str, default="paper_trading_test_results.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    try:
        # Initialize test
        test = PaperTradingTest(
            config_path=args.config,
            initial_capital=args.capital,
            test_duration_hours=args.duration,
            symbols=args.symbols,
            strategies=args.strategies,
            risk_profile=args.risk
        )
        
        # Initialize services and strategies
        await test.initialize()
        
        # Run test
        results = await test.run_test()
        
        # Generate report
        await test.generate_report(args.output)
        
        logger.info("Paper trading test completed successfully")
        
    except Exception as e:
        logger.error(f"Error running paper trading test: {e}", exc_info=True)


if __name__ == "__main__":
    # Create config directory if it doesn't exist
    os.makedirs("config", exist_ok=True)
    
    asyncio.run(main())
