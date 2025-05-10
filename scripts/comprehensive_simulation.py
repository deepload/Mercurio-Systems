"""
Mercurio AI - Year-Long Strategy Simulation

This script performs a full-year simulation of all available trading strategies
across multiple timeframes (daily, weekly, monthly) for the period
from March 3, 2024 to April 25, 2025.

It leverages Mercurio AI's fallback mechanisms for testing without API keys.
"""
import os
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import random

# Setup basic styling for matplotlib
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['axes.grid'] = True

# Create a simple tabulate function if not available
def simple_tabulate(data, headers=None, floatfmt=".2f"):
    """Simple tabulate implementation for when the package is not available."""
    if not data:
        return ""
    
    result = []
    
    # Add headers
    if headers:
        result.append(" | ".join(str(h) for h in headers))
        result.append("-" * (sum(len(str(h)) for h in headers) + 3 * (len(headers) - 1)))
    
    # Add rows
    for row in data:
        formatted_row = []
        for item in row:
            if isinstance(item, float):
                formatted_row.append(f"{item:{floatfmt}}")
            else:
                formatted_row.append(str(item))
        result.append(" | ".join(formatted_row))
    
    return "\n".join(result)

# Use tabulate if available, otherwise use our simple implementation
try:
    from tabulate import tabulate
except ImportError:
    tabulate = simple_tabulate
    print("Using simple table formatting (tabulate not available)")

# Import simulation utilities
from simulation_utils import generate_simulation_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure output directories exist
os.makedirs('reports/comprehensive', exist_ok=True)

# Define simulation parameters - March 2024 to April 2025 timespan
SIMULATION_CONFIG = {
    "start_date": datetime(2024, 3, 3),
    "end_date": datetime(2025, 4, 25),
    "initial_capital": 10000,  # $10,000 per strategy
    "timeframes": ["day", "week", "month"],
    "assets": {
        "stocks": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        "crypto": ["BTC-USD", "ETH-USD", "SOL-USD", "ADA-USD", "DOT-USD"]
    }
}

# Helper functions for timeframes and strategy setup
def setup_timeframes(timeframe):
    """Configure settings for different timeframes."""
    if timeframe == "day":
        return {
            "data_freq": "1d",
            "lookback_days": 90,
            "trade_interval": "day"
        }
    elif timeframe == "week":
        return {
            "data_freq": "1w",
            "lookback_days": 180,
            "trade_interval": "week"
        }
    elif timeframe == "month":
        return {
            "data_freq": "1mo",
            "lookback_days": 365,
            "trade_interval": "month"
        }
    else:
        return {"data_freq": "1d", "lookback_days": 90, "trade_interval": "day"}

def prepare_strategy(strategy_class, params=None):
    """Prepare a strategy instance with fallback error handling."""
    if params is None:
        params = {}
    
    try:
        return strategy_class(**params)
    except Exception as e:
        logger.error(f"Error initializing {strategy_class.__name__}: {e}")
        return None

class ComprehensiveSimulation:
    """
    Runs comprehensive simulations of all trading strategies
    across multiple timeframes and assets for a full year.
    """
    
    def __init__(self, config=None):
        """Initialize the simulation with configuration."""
        self.config = config or SIMULATION_CONFIG
        self.results = []
        self.strategies = []
        self.market_data = None
        
    async def initialize(self):
        """Initialize market data service and load strategies."""
        try:
            print("Initializing market data service...")
            # Import market data service with fallback to sample data
            try:
                from app.services.market_data import MarketDataService
                self.market_data = MarketDataService()
                logger.info("Market data service initialized successfully")
            except ImportError as e:
                logger.warning(f"Could not import MarketDataService: {e}")
                logger.warning("Using synthetic data generation as fallback")
                self.market_data = None
            
            # Load all available strategies
            await self._load_strategies()
            
            if not self.strategies:
                logger.error("No strategies could be loaded. Simulation cannot proceed.")
                return False
                
            logger.info(f"Successfully loaded {len(self.strategies)} strategies")
            return True
        except Exception as e:
            logger.error(f"Error initializing simulation: {e}")
            return False
    
    async def _load_strategies(self):
        """Load all available strategies from the Mercurio AI platform."""
        self.strategies = []
        
        # Try to load each strategy type with robust error handling
        try:
            from app.strategies.moving_average import MovingAverageStrategy
            print("Loading Moving Average strategies...")
            
            # Standard Moving Average strategy
            self.strategies.append({
                "name": "MovingAverage",
                "class": MovingAverageStrategy,
                "params": {"short_window": 10, "long_window": 30, "use_ml": False}
            })
            
            # ML-enhanced Moving Average strategy
            self.strategies.append({
                "name": "MovingAverage_ML",
                "class": MovingAverageStrategy,
                "params": {"short_window": 10, "long_window": 30, "use_ml": True}
            })
            
            logger.info("Added MovingAverage strategies")
        except Exception as e:
            logger.warning(f"Could not load MovingAverage strategy: {e}")
        
        try:
            from app.strategies.lstm_predictor import LSTMPredictorStrategy
            print("Loading LSTM Predictor strategy...")
            
            self.strategies.append({
                "name": "LSTM",
                "class": LSTMPredictorStrategy,
                "params": {"sequence_length": 20, "prediction_horizon": 1}
            })
            
            logger.info("Added LSTM Predictor strategy")
        except Exception as e:
            logger.warning(f"Could not load LSTM Predictor strategy: {e}")
        
        try:
            from app.strategies.llm_strategy import LLMStrategy
            print("Loading LLM strategy...")
            
            self.strategies.append({
                "name": "LLM",
                "class": LLMStrategy,
                "params": {}
            })
            
            logger.info("Added LLM strategy")
        except Exception as e:
            logger.warning(f"Could not load LLM strategy: {e}")
        
        # Try to load Transformer strategy if available
        try:
            from app.strategies.transformer_strategy import TransformerStrategy
            print("Loading Transformer strategy...")
            
            self.strategies.append({
                "name": "Transformer",
                "class": TransformerStrategy,
                "params": {}
            })
            
            logger.info("Added Transformer strategy")
        except Exception as e:
            logger.warning(f"Could not load Transformer strategy: {e}")
            
        # Try to load MSI strategy if available
        try:
            from app.strategies.msi_strategy import MSIStrategy
            print("Loading MSI strategy...")
            
            self.strategies.append({
                "name": "MSI",
                "class": MSIStrategy,
                "params": {}
            })
            
            logger.info("Added MSI strategy")
        except Exception as e:
            logger.warning(f"Could not load MSI strategy: {e}")
        
        if not self.strategies:
            # Create fallback moving average strategy directly if all imports failed
            logger.warning("No strategies could be loaded from app.strategies. Creating fallback strategy.")
            
            # Simple Moving Average strategy implementation as fallback
            class FallbackMovingAverageStrategy:
                def __init__(self, short_window=10, long_window=30):
                    self.short_window = short_window
                    self.long_window = long_window
                    self.name = "FallbackMovingAverage"
                
                async def preprocess_data(self, data):
                    # Calculate moving averages
                    data = data.copy()
                    data['short_ma'] = data['close'].rolling(window=self.short_window).mean()
                    data['long_ma'] = data['close'].rolling(window=self.long_window).mean()
                    return data
                
                async def predict(self, data):
                    # Generate trading signals
                    if len(data) < self.long_window:
                        return 'HOLD', 0.0
                    
                    last_row = data.iloc[-1]
                    if last_row['short_ma'] > last_row['long_ma']:
                        return 'BUY', 0.6
                    elif last_row['short_ma'] < last_row['long_ma']:
                        return 'SELL', 0.6
                    else:
                        return 'HOLD', 0.5
                
                async def backtest(self, data, initial_capital=10000, label=None):
                    # Simple backtest implementation
                    equity_curve = [initial_capital]
                    position = None
                    trades = []
                    
                    for i in range(len(data)-1):
                        if i < self.long_window:
                            continue
                        
                        # Get signal
                        signal, _ = await self.predict(data.iloc[:i+1])
                        
                        if signal == 'BUY' and position is None:
                            # Buy
                            entry_price = data.iloc[i+1]['open']
                            shares = equity_curve[-1] / entry_price
                            position = {'price': entry_price, 'shares': shares}
                        elif signal == 'SELL' and position is not None:
                            # Sell
                            exit_price = data.iloc[i+1]['open']
                            value = position['shares'] * exit_price
                            trades.append({
                                'entry_price': position['price'],
                                'exit_price': exit_price,
                                'pnl': (exit_price/position['price'] - 1) * 100
                            })
                            equity_curve.append(value)
                            position = None
                        else:
                            # Hold
                            if position is not None:
                                equity_curve.append(position['shares'] * data.iloc[i+1]['close'])
                            else:
                                equity_curve.append(equity_curve[-1])
                    
                    return {
                        'equity_curve': equity_curve,
                        'trades': trades,
                        'final_equity': equity_curve[-1] if equity_curve else initial_capital
                    }
            
            # Add fallback strategy
            self.strategies.append({
                "name": "MovingAverage_Fallback",
                "class": FallbackMovingAverageStrategy,
                "params": {"short_window": 10, "long_window": 30}
            })
            
            logger.info("Added fallback moving average strategy")
        
        logger.info(f"Loaded {len(self.strategies)} strategies for simulation")
        
    async def run_simulations(self):
        """Run simulations for all strategies across timeframes and assets."""
        start_date = self.config["start_date"]
        end_date = self.config["end_date"]
        initial_capital = self.config["initial_capital"]
        timeframes = self.config["timeframes"]
        
        # Combine all assets for processing
        all_assets = self.config["assets"]["stocks"] + self.config["assets"]["crypto"]
        
        print(f"\n===== RUNNING COMPREHENSIVE SIMULATION =====")
        print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        print(f"Assets: {len(all_assets)} (Stocks: {len(self.config['assets']['stocks'])}, Crypto: {len(self.config['assets']['crypto'])})")
        print(f"Strategies: {len(self.strategies)}")
        print(f"Timeframes: {', '.join(timeframes)}")
        print(f"Initial Capital: ${initial_capital:,.2f} per strategy")
        print("=======================================\n")
        
        # Track all results
        all_results = []
        
        # Process each timeframe
        for tf_idx, timeframe in enumerate(timeframes):
            print(f"Processing {timeframe} timeframe ({tf_idx+1}/{len(timeframes)})")
            
            # Configure timeframe settings
            tf_settings = setup_timeframes(timeframe)
            
            # Process each asset
            for asset_idx, asset in enumerate(all_assets):
                print(f"  Asset: {asset} ({asset_idx+1}/{len(all_assets)})")
                
                try:
                    # Get or generate data for this asset
                    if self.market_data:
                        try:
                            # Try to get data from market data service
                            data = await self.market_data.get_historical_data(
                                asset, 
                                start_date=start_date - timedelta(days=tf_settings["lookback_days"]),
                                end_date=end_date,
                                timeframe=tf_settings["data_freq"]
                            )
                            print(f"    Using market data service for {asset}")
                        except Exception as e:
                            print(f"    Error getting data from market data service: {e}")
                            print(f"    Falling back to synthetic data generation")
                            data = generate_simulation_data(asset, start_date, end_date, tf_settings["data_freq"])
                    else:
                        # Generate synthetic data
                        data = generate_simulation_data(asset, start_date, end_date, tf_settings["data_freq"])
                    
                    if data.empty:
                        print(f"    No data available for {asset}, skipping")
                        continue
                    
                    # Process each strategy
                    for strat_idx, strategy_config in enumerate(self.strategies):
                        strategy_name = strategy_config["name"]
                        print(f"    Running {strategy_name} ({strat_idx+1}/{len(self.strategies)})")
                        
                        try:
                            # Create strategy instance
                            strategy_class = strategy_config["class"]
                            strategy_params = strategy_config["params"]
                            
                            # Initialize strategy
                            strategy = prepare_strategy(strategy_class, strategy_params)
                            if not strategy:
                                print(f"      Failed to initialize {strategy_name}, skipping")
                                continue
                            
                            # Preprocess data
                            try:
                                processed_data = await strategy.preprocess_data(data)
                                print(f"      Data preprocessing successful")
                            except Exception as e:
                                print(f"      Error preprocessing data: {e}")
                                continue
                            
                            # Run backtest with robust parameter handling
                            try:
                                print(f"      Running backtest...")
                                
                                # Check backtest method signature to handle different parameter requirements
                                import inspect
                                backtest_params = {}
                                
                                # Add initial_capital parameter (common to all strategies)
                                backtest_params["initial_capital"] = initial_capital
                                
                                # Check if the strategy's backtest method accepts a 'label' parameter
                                # and only add it if supported
                                try:
                                    sig = inspect.signature(strategy.backtest)
                                    if 'label' in sig.parameters:
                                        backtest_params["label"] = f"{strategy_name}_{asset}_{timeframe}"
                                except Exception:
                                    # If we can't inspect the signature, try without the label
                                    pass
                                
                                # Different strategies might have different parameter requirements
                                # Call with appropriate parameters
                                backtest_result = await strategy.backtest(processed_data, **backtest_params)
                                print(f"      Backtest complete")
                            except Exception as e:
                                print(f"      Error during backtest: {e}")
                                # Fallback: implement a simple backtest simulation if the strategy's backtest fails
                                print(f"      Using fallback backtest simulation")
                                
                                # Simple price-based simulation
                                equity = [initial_capital]
                                position = None
                                trades = []
                                
                                try:
                                    # Process each day to generate a basic equity curve
                                    for i in range(len(processed_data) - 1):
                                        # Skip early data points until we have enough history
                                        if i < 20:  # Minimum data points for most strategies
                                            continue
                                            
                                        # Get signal using current data slice
                                        try:
                                            signal, confidence = await strategy.predict(processed_data.iloc[:i+1])
                                        except Exception:
                                            # If predict fails, assume HOLD
                                            signal, confidence = "HOLD", 0
                                            
                                        # Get next day's price for simulation
                                        next_price = processed_data.iloc[i+1]["close"]
                                        current_equity = equity[-1]
                                        
                                        if signal == "BUY" and position is None:
                                            # Buy signal
                                            position = {"entry_price": next_price, "shares": current_equity / next_price}
                                            equity.append(current_equity)  # Value stays the same on entry
                                        elif signal == "SELL" and position is not None:
                                            # Sell signal
                                            exit_value = position["shares"] * next_price
                                            pnl = exit_value - position["shares"] * position["entry_price"]
                                            trades.append({"pnl": pnl, "return": pnl / (position["shares"] * position["entry_price"])})                                     
                                            equity.append(exit_value)
                                            position = None
                                        else:
                                            # Hold
                                            if position is not None:
                                                # Update position value
                                                equity.append(position["shares"] * next_price)
                                            else:
                                                # Cash position stays the same
                                                equity.append(current_equity)
                                    
                                    # Create fallback backtest result
                                    backtest_result = {
                                        "equity_curve": equity,
                                        "trades": trades,
                                        "final_equity": equity[-1] if equity else initial_capital
                                    }
                                except Exception as sim_error:
                                    print(f"      Error in fallback simulation: {sim_error}")
                                    # Ultimate fallback with minimal data
                                    backtest_result = {
                                        "equity_curve": [initial_capital],
                                        "trades": [],
                                        "final_equity": initial_capital,
                                        "error": str(e)
                                    }
                            
                            # Calculate metrics
                            try:
                                # Basic metrics calculation
                                final_equity = backtest_result.get("final_equity", initial_capital)
                                total_return = (final_equity / initial_capital) - 1 if initial_capital > 0 else 0
                                trades_count = len(backtest_result.get("trades", []))
                                
                                # Advanced metrics if equity curve is available
                                equity_curve = backtest_result.get("equity_curve", [initial_capital])
                                if len(equity_curve) > 1:
                                    # Calculate returns for Sharpe ratio
                                    returns = []
                                    for i in range(1, len(equity_curve)):
                                        if equity_curve[i-1] > 0:
                                            returns.append(equity_curve[i] / equity_curve[i-1] - 1)
                                        else:
                                            returns.append(0)
                                    
                                    # Calculate Sharpe ratio
                                    sharpe_ratio = 0
                                    if len(returns) > 0 and np.std(returns) > 0:
                                        sharpe_ratio = np.sqrt(252) * np.mean(returns) / np.std(returns)
                                    
                                    # Calculate max drawdown
                                    peak = np.maximum.accumulate(equity_curve)
                                    drawdown = (equity_curve - peak) / peak
                                    max_drawdown = abs(min(drawdown)) if drawdown.size > 0 else 0
                                else:
                                    sharpe_ratio = 0
                                    max_drawdown = 0
                                
                                # Calculate annualized return
                                days_elapsed = (end_date - start_date).days
                                if days_elapsed > 0:
                                    annualized_return = (1 + total_return) ** (365 / days_elapsed) - 1
                                else:
                                    annualized_return = 0
                                
                                metrics = {
                                    "initial_capital": initial_capital,
                                    "final_capital": final_equity,
                                    "total_return": total_return,
                                    "annualized_return": annualized_return,
                                    "sharpe_ratio": sharpe_ratio,
                                    "max_drawdown": max_drawdown,
                                    "trades_count": trades_count
                                }
                            except Exception as e:
                                print(f"      Error calculating metrics: {e}")
                                metrics = {
                                    "initial_capital": initial_capital,
                                    "final_capital": initial_capital,
                                    "total_return": 0,
                                    "annualized_return": 0,
                                    "sharpe_ratio": 0,
                                    "max_drawdown": 0,
                                    "trades_count": 0,
                                    "error": str(e)
                                }
                            
                            # Store result
                            result = {
                                "timeframe": timeframe,
                                "asset": asset,
                                "strategy": strategy_name,
                                "metrics": metrics,
                                "asset_type": "crypto" if "-USD" in asset else "stock"
                            }
                            
                            all_results.append(result)
                            
                            # Print basic results
                            print(f"      Results: {metrics['total_return']*100:.2f}% return, "
                                  f"${metrics['final_capital']:.2f} final capital, "
                                  f"{metrics['trades_count']} trades")
                            
                        except Exception as e:
                            print(f"      Error running {strategy_name} on {asset}: {e}")
                
                except Exception as e:
                    print(f"    Error processing {asset}: {e}")
        
        # Store all results
        self.results = all_results
        
        # Generate summary report
        self._generate_summary_report()
        
        return all_results
    
    def _generate_summary_report(self):
        """Generate a summary report from the simulation results."""
        if not self.results:
            print("No results to report")
            return
        
        print("\n===== SIMULATION RESULTS SUMMARY =====\n")
        
        # Convert results to pandas DataFrame for easier analysis
        results_data = []
        for result in self.results:
            metrics = result["metrics"]
            
            entry = {
                "Strategy": result["strategy"],
                "Asset": result["asset"],
                "Asset Type": result["asset_type"],
                "Timeframe": result["timeframe"],
                "Initial Capital": metrics["initial_capital"],
                "Final Capital": metrics["final_capital"],
                "Total Return (%)": metrics["total_return"] * 100,
                "Annualized Return (%)": metrics["annualized_return"] * 100,
                "Sharpe Ratio": metrics["sharpe_ratio"],
                "Max Drawdown (%)": metrics["max_drawdown"] * 100,
                "Trades": metrics["trades_count"]
            }
            
            results_data.append(entry)
        
        # Create DataFrame
        try:
            results_df = pd.DataFrame(results_data)
            
            # Save full results to CSV
            os.makedirs('reports/comprehensive', exist_ok=True)
            results_df.to_csv("reports/comprehensive/full_simulation_results.csv", index=False)
            print("Saved full simulation results to reports/comprehensive/full_simulation_results.csv")
            
            # Generate basic statistics
            print("\nOverall Statistics:")
            print(f"Total Simulations: {len(results_df)}")
            print(f"Average Return: {results_df['Total Return (%)'].mean():.2f}%")
            print(f"Best Return: {results_df['Total Return (%)'].max():.2f}%")
            print(f"Worst Return: {results_df['Total Return (%)'].min():.2f}%")
            
            # Best performers
            print("\nTop 5 Best Performers:")
            top_performers = results_df.nlargest(5, "Total Return (%)")
            
            # Format top performers as a table
            top_table = []
            for _, row in top_performers.iterrows():
                top_table.append([
                    row["Strategy"],
                    row["Asset"],
                    row["Timeframe"],
                    f"{row['Total Return (%)']:.2f}%",
                    f"{row['Sharpe Ratio']:.2f}",
                    f"{row['Max Drawdown (%)']:.2f}%",
                    row["Trades"]
                ])
            
            headers = ["Strategy", "Asset", "Timeframe", "Return (%)", "Sharpe", "Max DD (%)", "Trades"]
            print(tabulate(top_table, headers=headers))
            
            # Strategy comparison
            print("\nAverage Returns by Strategy:")
            strategy_returns = results_df.groupby("Strategy")["Total Return (%)"].mean().reset_index()
            strategy_returns = strategy_returns.sort_values("Total Return (%)", ascending=False)
            
            strategy_table = []
            for _, row in strategy_returns.iterrows():
                strategy_table.append([
                    row["Strategy"],
                    f"{row['Total Return (%)']:.2f}%"
                ])
            
            print(tabulate(strategy_table, headers=["Strategy", "Avg Return (%)"]))
            
            # Timeframe comparison
            print("\nAverage Returns by Timeframe:")
            timeframe_returns = results_df.groupby("Timeframe")["Total Return (%)"].mean().reset_index()
            timeframe_returns = timeframe_returns.sort_values("Total Return (%)", ascending=False)
            
            timeframe_table = []
            for _, row in timeframe_returns.iterrows():
                timeframe_table.append([
                    row["Timeframe"],
                    f"{row['Total Return (%)']:.2f}%"
                ])
            
            print(tabulate(timeframe_table, headers=["Timeframe", "Avg Return (%)"]))
            
            # Asset type comparison
            print("\nAverage Returns by Asset Type:")
            asset_type_returns = results_df.groupby("Asset Type")["Total Return (%)"].mean().reset_index()
            
            asset_type_table = []
            for _, row in asset_type_returns.iterrows():
                asset_type_table.append([
                    row["Asset Type"],
                    f"{row['Total Return (%)']:.2f}%"
                ])
            
            print(tabulate(asset_type_table, headers=["Asset Type", "Avg Return (%)"]))
            
            # Generate plots if matplotlib is available
            try:
                # Create comparison chart for strategies
                plt.figure(figsize=(12, 6))
                
                bars = plt.bar(
                    strategy_returns["Strategy"],
                    strategy_returns["Total Return (%)"]
                )
                
                # Add values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.3,
                        f'{height:.1f}%',
                        ha='center', va='bottom'
                    )
                
                plt.title("Average Returns by Strategy")
                plt.ylabel("Return (%)")
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                # Save chart
                plt.savefig("reports/comprehensive/strategy_returns.png", dpi=300)
                print("Generated strategy comparison chart")
                
                # Create timeframe comparison chart
                plt.figure(figsize=(10, 5))
                
                bars = plt.bar(
                    timeframe_returns["Timeframe"],
                    timeframe_returns["Total Return (%)"]
                )
                
                # Add values on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width()/2.,
                        height + 0.3,
                        f'{height:.1f}%',
                        ha='center', va='bottom'
                    )
                
                plt.title("Average Returns by Timeframe")
                plt.ylabel("Return (%)")
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                
                # Save chart
                plt.savefig("reports/comprehensive/timeframe_returns.png", dpi=300)
                print("Generated timeframe comparison chart")
                
                # Create strategy performance by asset type chart
                asset_strategy_returns = results_df.groupby(["Strategy", "Asset Type"])["Total Return (%)"].mean().unstack()
                
                if not asset_strategy_returns.empty:
                    plt.figure(figsize=(14, 7))
                    
                    ax = asset_strategy_returns.plot(kind="bar", figsize=(14, 7))
                    
                    plt.title("Strategy Performance by Asset Type")
                    plt.ylabel("Return (%)")
                    plt.grid(axis='y', alpha=0.3)
                    plt.legend(title="Asset Type")
                    plt.tight_layout()
                    
                    # Save chart
                    plt.savefig("reports/comprehensive/strategy_asset_returns.png", dpi=300)
                    print("Generated strategy by asset type chart")
            except Exception as e:
                print(f"Error generating charts: {e}")
                
        except Exception as e:
            print(f"Error generating summary report: {e}")
            
            # Basic text report if DataFrame creation failed
            with open("reports/comprehensive/simulation_report.txt", "w") as f:
                f.write("SIMULATION REPORT\n")
                f.write("=================\n\n")
                f.write(f"Total Simulations: {len(self.results)}\n\n")
                
                f.write("RESULTS:\n")
                for result in self.results:
                    metrics = result["metrics"]
                    f.write(f"Strategy: {result['strategy']}\n")
                    f.write(f"Asset: {result['asset']}\n")
                    f.write(f"Timeframe: {result['timeframe']}\n")
                    f.write(f"Return: {metrics['total_return']*100:.2f}%\n")
                    f.write(f"Final Capital: ${metrics['final_capital']:.2f}\n")
                    f.write(f"Trades: {metrics['trades_count']}\n\n")
            
async def main():
    """Main entry point for the simulation."""
    print("\n===== MERCURIO AI YEAR-LONG STRATEGY SIMULATION =====\n")
    print(f"Simulating from {SIMULATION_CONFIG['start_date'].strftime('%Y-%m-%d')} to {SIMULATION_CONFIG['end_date'].strftime('%Y-%m-%d')}")
    print(f"Timeframes: {', '.join(SIMULATION_CONFIG['timeframes'])}")
    print(f"Assets: {len(SIMULATION_CONFIG['assets']['stocks'] + SIMULATION_CONFIG['assets']['crypto'])} total")
    print("==================================================\n")
    
    try:
        # Create and initialize simulator
        simulator = ComprehensiveSimulation()
        
        # Initialize
        success = await simulator.initialize()
        if not success:
            print("Failed to initialize simulation. Check logs for details.")
            return
        
        # Run simulations (this also generates reports)
        print("\nRunning year-long strategy simulations...")
        print("This may take some time depending on the number of strategies and assets.")
        
        results = await simulator.run_simulations()
        
        print("\nSimulation complete! Results are available in the 'reports/comprehensive/' directory.")
        print("You can visualize the results with the comprehensive_dashboard.py script.")
    except Exception as e:
        print(f"\nError during simulation: {e}")
        print("Check the logs for more details.")

if __name__ == "__main__":
    asyncio.run(main())
            

    
    async def run_simulations(self):
        """Run all simulations across timeframes, assets, and strategies."""
        start_date = self.config["start_date"]
        end_date = self.config["end_date"]
        initial_capital = self.config["initial_capital"]
        timeframes = self.config["timeframes"]
        
        # Combine all assets
        all_assets = self.config["assets"]["stocks"] + self.config["assets"]["crypto"]
        
        # Track all results
        all_results = []
        
        # Setup progress tracking
        total_simulations = len(timeframes) * len(all_assets) * len(self.strategies)
        logger.info(f"Running {total_simulations} simulations...")
        
        # Process each timeframe
        for timeframe in timeframes:
            logger.info(f"Processing {timeframe} timeframe")
            
            # Configure timeframe settings
            tf_settings = setup_timeframes(timeframe)
            
            # Process each asset
            for asset in all_assets:
                logger.info(f"Processing {asset} for {timeframe} timeframe")
                
                try:
                    # Get data for this asset
                    data = await self._get_asset_data(asset, start_date, end_date, tf_settings["data_freq"])
                    
                    if data.empty:
                        logger.warning(f"No data available for {asset}, skipping")
                        continue
                    
                    # Run each strategy
                    for strategy_config in self.strategies:
                        strategy_name = strategy_config["name"]
                        logger.info(f"Running {strategy_name} on {asset} ({timeframe})")
                        
                        try:
                            # Prepare strategy instance
                            strategy = prepare_strategy_instance(
                                strategy_config["class"],
                                strategy_config["params"],
                                timeframe
                            )
                            
                            # Run backtest
                            backtest_result = await run_backtest(
                                strategy,
                                data,
                                initial_capital,
                                f"{strategy_name}_{asset}_{timeframe}"
                            )
                            
                            # Calculate metrics
                            metrics = calculate_performance_metrics(backtest_result)
                            
                            # Store results
                            result = {
                                "timeframe": timeframe,
                                "asset": asset,
                                "strategy": strategy_name,
                                "backtest_result": backtest_result,
                                "metrics": metrics,
                                "asset_type": "crypto" if "-USD" in asset else "stock"
                            }
                            
                            all_results.append(result)
                            
                            # Log basic results
                            logger.info(f"  {strategy_name} on {asset} ({timeframe}): {metrics['total_return']*100:.2f}% return, Sharpe: {metrics['sharpe_ratio']:.2f}")
                            
                        except Exception as e:
                            logger.error(f"Error running {strategy_name} on {asset} ({timeframe}): {e}")
                
                except Exception as e:
                    logger.error(f"Error processing {asset}: {e}")
        
        # Store all results
        self.results = all_results
        
        return all_results
    
    async def _get_asset_data(self, asset, start_date, end_date, freq="1d"):
        """Get asset data for the specified time range and frequency."""
        try:
            # Try to get data from market data service
            if self.market_data:
                data = await self.market_data.get_historical_data(
                    asset, 
                    start_date=start_date,
                    end_date=end_date,
                    timeframe=freq
                )
                
                if not data.empty:
                    return data
            
            # Fallback to generated data if needed
            logger.info(f"Generating synthetic data for {asset} ({freq})")
            return generate_simulation_data(asset, start_date, end_date, freq)
            
        except Exception as e:
            logger.error(f"Error getting data for {asset}: {e}")
            # Return empty dataframe as last resort
            return pd.DataFrame()
    
    def generate_reports(self):
        """Generate comprehensive reports from simulation results."""
        if not self.results:
            logger.warning("No results to report")
            return
        
        # Convert results to DataFrame for analysis
        results_data = []
        
        for result in self.results:
            metrics = result["metrics"]
            
            entry = {
                "Strategy": result["strategy"],
                "Asset": result["asset"],
                "Asset Type": result["asset_type"],
                "Timeframe": result["timeframe"],
                "Initial Capital": self.config["initial_capital"],
                "Final Capital": metrics["final_value"],
                "Total Return (%)": metrics["total_return"] * 100,
                "Annualized Return (%)": metrics["annualized_return"] * 100,
                "Sharpe Ratio": metrics["sharpe_ratio"],
                "Max Drawdown (%)": metrics["max_drawdown"] * 100,
                "Trades": metrics["trades_count"]
            }
            
            results_data.append(entry)
        
        # Create DataFrame
        results_df = pd.DataFrame(results_data)
        
        # Save full results
        results_df.to_csv("reports/comprehensive/full_simulation_results.csv", index=False)
        logger.info("Saved full simulation results to reports/comprehensive/full_simulation_results.csv")
        
        # Generate summary report
        generate_performance_report(results_df, "reports/comprehensive")
        
        # Generate strategy comparisons by timeframe
        self._generate_strategy_comparisons(results_df)
        
        # Print summary of best performers
        self._print_best_performers(results_df)
    
    def _generate_strategy_comparisons(self, results_df):
        """Generate comparison charts for strategies across timeframes."""
        try:
            # Create directory for charts
            os.makedirs("reports/comprehensive/charts", exist_ok=True)
            
            # 1. Average returns by strategy and timeframe
            plt.figure(figsize=(14, 8))
            
            # Pivot data for heatmap
            pivot_data = results_df.pivot_table(
                index="Strategy", 
                columns="Timeframe",
                values="Total Return (%)",
                aggfunc="mean"
            )
            
            # Create heatmap
            sns.heatmap(
                pivot_data,
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                linewidths=0.5,
                cbar_kws={"label": "Average Return (%)"}
            )
            
            plt.title("Average Returns by Strategy and Timeframe")
            plt.tight_layout()
            plt.savefig("reports/comprehensive/charts/strategy_timeframe_returns.png", dpi=300)
            plt.close()
            
            # 2. Strategy performance by asset type
            plt.figure(figsize=(12, 8))
            
            # Pivot data
            asset_pivot = results_df.pivot_table(
                index="Strategy",
                columns="Asset Type",
                values="Total Return (%)",
                aggfunc="mean"
            )
            
            # Create bar chart
            asset_pivot.plot(kind="bar", figsize=(12, 8))
            plt.title("Average Returns by Strategy and Asset Type")
            plt.ylabel("Average Return (%)")
            plt.grid(axis="y", alpha=0.3)
            plt.tight_layout()
            plt.savefig("reports/comprehensive/charts/strategy_asset_returns.png", dpi=300)
            plt.close()
            
            # 3. Risk-return scatter plot
            plt.figure(figsize=(12, 8))
            
            scatter = plt.scatter(
                results_df["Max Drawdown (%)"],
                results_df["Sharpe Ratio"],
                c=pd.factorize(results_df["Strategy"])[0],
                s=100,
                alpha=0.7
            )
            
            # Add legend
            strategies = results_df["Strategy"].unique()
            plt.legend(
                scatter.legend_elements()[0],
                strategies,
                title="Strategy",
                loc="upper left"
            )
            
            plt.xlabel("Maximum Drawdown (%)")
            plt.ylabel("Sharpe Ratio")
            plt.title("Risk-Return Profile by Strategy")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("reports/comprehensive/charts/risk_return_scatter.png", dpi=300)
            plt.close()
            
            logger.info("Generated strategy comparison charts")
            
        except Exception as e:
            logger.error(f"Error generating strategy comparisons: {e}")
    
    def generate_reports(self):
        """Generate all reports and visualizations."""
        try:
            if not hasattr(self, 'results_df') or self.results_df is None or len(self.results_df) == 0:
                logger.error("No results available to generate reports")
                return
                
            # Create strategy comparisons
            self._generate_strategy_comparisons(self.results_df)
            
            # Print best performers
            self._print_best_performers(self.results_df)
            
            logger.info("All reports generated successfully")
            
        except Exception as e:
            logger.error(f"Error generating reports: {e}")
    
    def _print_best_performers(self, results_df):
        """Print summary of best performing strategies."""
        print("\n===== COMPREHENSIVE SIMULATION RESULTS =====\n")
        
        # Best overall
        best_overall_idx = results_df["Total Return (%)"].idxmax()
        best_overall = results_df.loc[best_overall_idx]
        
        print(f"Best Overall: {best_overall['Strategy']} on {best_overall['Asset']} ({best_overall['Timeframe']})")
        print(f"  Return: {best_overall['Total Return (%)']:.2f}%")
        print(f"  Sharpe: {best_overall['Sharpe Ratio']:.2f}")
        
        # Best by timeframe
        print("\nBest by Timeframe:")
        for timeframe in results_df["Timeframe"].unique():
            timeframe_df = results_df[results_df["Timeframe"] == timeframe]
            best_idx = timeframe_df["Total Return (%)"].idxmax()
            best = timeframe_df.loc[best_idx]
            
            print(f"  {timeframe.capitalize()}: {best['Strategy']} on {best['Asset']} ({best['Total Return (%)']:.2f}%)")
        
        # Best by asset type
        print("\nBest by Asset Type:")
        for asset_type in results_df["Asset Type"].unique():
            asset_df = results_df[results_df["Asset Type"] == asset_type]
            best_idx = asset_df["Total Return (%)"].idxmax()
            best = asset_df.loc[best_idx]
            
            print(f"  {asset_type.capitalize()}: {best['Strategy']} on {best['Asset']} ({best['Timeframe']}) - {best['Total Return (%)']:.2f}%")
        
        # Best strategy overall
        print("\nAverage Returns by Strategy:")
        strategy_returns = results_df.groupby("Strategy")["Total Return (%)"].mean()
        for strategy, avg_return in strategy_returns.sort_values(ascending=False).items():
            print(f"  {strategy}: {avg_return:.2f}%")
        
        print("\nDetailed reports saved to the 'reports/comprehensive/' directory")

async def main():
    """Main entry point."""
    print("\n===== MERCURIO AI COMPREHENSIVE STRATEGY SIMULATION =====\n")
    print(f"Simulating all strategies from {SIMULATION_CONFIG['start_date'].strftime('%Y-%m-%d')} to {SIMULATION_CONFIG['end_date'].strftime('%Y-%m-%d')}")
    print(f"Timeframes: {', '.join(SIMULATION_CONFIG['timeframes'])}")
    print("=" * 60)
    
    # Create simulator
    simulator = ComprehensiveSimulation()
    
    # Initialize
    success = await simulator.initialize()
    if not success:
        print("Failed to initialize simulation. Check logs for details.")
        return
    
    # Run simulations
    print("\nRunning comprehensive simulations (this may take some time)...")
    results = await simulator.run_simulations()
    
    # Generate reports
    simulator.generate_reports()
    
    print("\nComprehensive simulation complete!")
    print("View detailed reports in the 'reports/comprehensive/' directory")
    print("Run 'streamlit run comprehensive_dashboard.py' for an interactive dashboard")

if __name__ == "__main__":
    asyncio.run(main())
