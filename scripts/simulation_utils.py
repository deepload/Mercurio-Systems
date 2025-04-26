"""
Mercurio AI - Simulation Utilities

Helper functions for the comprehensive strategy simulation.
These utilities handle data generation, timeframe configuration,
strategy preparation, backtesting, and performance reporting.
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Seaborn import is optional
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    print("Seaborn not available, using matplotlib for visualizations")

# Tabulate import is optional
try:
    from tabulate import tabulate
except ImportError:
    def tabulate(data, **kwargs):
        """Simple tabulate fallback"""
        result = ""
        if "headers" in kwargs:
            result += "\t".join(kwargs["headers"]) + "\n"
        for row in data:
            result += "\t".join(str(x) for x in row) + "\n"
        return result

import logging

# Setup logging
logger = logging.getLogger(__name__)

def generate_simulation_data(symbol, start_date, end_date, freq="1d"):
    """
    Generate realistic market data for simulations when real data is unavailable.
    This function creates synthetic price data that mimics realistic market behavior
    including trends, cycles, and market events for the specified period.
    
    Args:
        symbol: Asset symbol (e.g., 'AAPL', 'BTC-USD')
        start_date: Start date for the simulation (datetime)
        end_date: End date for the simulation (datetime)
        freq: Data frequency ('1d', '1w', '1mo')
        
    Returns:
        DataFrame with OHLCV data (timestamp, open, high, low, close, volume)
    """
    print(f"Generating synthetic data for {symbol} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Determine if stock or crypto
    is_crypto = "-USD" in symbol
    
    # Asset parameters dictionary with realistic starting values for March 2024
    ASSET_PARAMS = {
        # Cryptocurrencies
        "BTC-USD": {"price": 61000, "volatility": 0.035, "trend": 0.0008, "cycle_strength": 0.03},
        "ETH-USD": {"price": 3400, "volatility": 0.042, "trend": 0.0007, "cycle_strength": 0.04},
        "SOL-USD": {"price": 125, "volatility": 0.055, "trend": 0.0009, "cycle_strength": 0.05},
        "ADA-USD": {"price": 0.55, "volatility": 0.05, "trend": 0.0005, "cycle_strength": 0.045},
        "DOT-USD": {"price": 7.5, "volatility": 0.048, "trend": 0.0006, "cycle_strength": 0.042},
        
        # Stocks
        "AAPL": {"price": 173, "volatility": 0.018, "trend": 0.0004, "cycle_strength": 0.015},
        "MSFT": {"price": 415, "volatility": 0.02, "trend": 0.0006, "cycle_strength": 0.012},
        "GOOGL": {"price": 148, "volatility": 0.022, "trend": 0.0005, "cycle_strength": 0.018},
        "AMZN": {"price": 178, "volatility": 0.026, "trend": 0.0006, "cycle_strength": 0.02},
        "TSLA": {"price": 180, "volatility": 0.04, "trend": 0.0002, "cycle_strength": 0.03},
    }
    
    # Get asset parameters or use defaults
    params = ASSET_PARAMS.get(symbol, {
        "price": 100,  # Default starting price 
        "volatility": 0.025 if not is_crypto else 0.04,  # Higher volatility for crypto
        "trend": 0.0004 if not is_crypto else 0.0008,   # Stronger trend for crypto
        "cycle_strength": 0.015 if not is_crypto else 0.035  # Stronger cycles for crypto
    })
    
    initial_price = params["price"]
    volatility = params["volatility"]
    trend_factor = params["trend"]
    cycle_strength = params["cycle_strength"]
    
    # Calculate date range based on frequency with proper business day handling
    if freq == "1d":
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
    elif freq == "1w":
        date_range = pd.date_range(start=start_date, end=end_date, freq='W-FRI')  # Weekly (Fridays)
    elif freq == "1mo":
        date_range = pd.date_range(start=start_date, end=end_date, freq='BMS')  # Business month start
    else:
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Default to business days
    
    # Handle empty date ranges (can happen with certain date combinations)
    if len(date_range) == 0:
        print(f"Warning: No dates in range for {symbol} with freq={freq}. Generating minimal data.")
        # Generate at least a two-point dataset
        date_range = [start_date, min(start_date + timedelta(days=1), end_date)]
    
    # Generate price series with enhanced realism
    np.random.seed(hash(symbol) % 10000)  # Different seed per symbol for variety
    
    # Add trend and seasonal components
    days = np.array([(d - date_range[0]).days for d in date_range])
    
    # Base trend with random direction shifts
    trend_shifts = np.random.normal(trend_factor, trend_factor/2, len(date_range))
    trend = np.cumsum(trend_shifts)
    
    # Seasonal components with varying frequencies
    # Daily cycle (intraday patterns simulated at daily close)
    daily_cycle = 0.005 * np.sin(2 * np.pi * days / 5) * cycle_strength  # Weekly pattern
    
    # Monthly cycle (approximately 21 trading days)
    monthly_cycle = 0.015 * np.sin(2 * np.pi * days / 21) * cycle_strength
    
    # Quarterly cycle (approximately 63 trading days)
    quarterly_cycle = 0.03 * np.sin(2 * np.pi * days / 63) * cycle_strength
    
    # Combine components with controlled random walk
    random_walk = np.random.normal(0, volatility, len(date_range))
    returns = trend + daily_cycle + monthly_cycle + quarterly_cycle + random_walk
    
    # Create data with OHLCV structure
    data = pd.DataFrame(index=date_range)
    data['open'] = price_series
    
    # Generate realistic OHLC values
    for i in range(len(data)):
        if i > 0:
            data.loc[data.index[i], 'open'] = data.loc[data.index[i-1], 'close'] * \
                                          (1 + np.random.normal(0, 0.002))
        
        high_factor = 1 + abs(np.random.normal(0, 0.004))
        low_factor = 1 - abs(np.random.normal(0, 0.004))
        
        data.loc[data.index[i], 'high'] = data.loc[data.index[i], 'open'] * high_factor
        data.loc[data.index[i], 'low'] = data.loc[data.index[i], 'open'] * low_factor
        data.loc[data.index[i], 'close'] = np.random.uniform(
            data.loc[data.index[i], 'low'],
            data.loc[data.index[i], 'high']
        )
    
    # Generate volume
    avg_volume = 1000000 if not is_crypto else 10000
    data['volume'] = np.random.normal(avg_volume, avg_volume * 0.2, len(data_range))
    data['volume'] = data['volume'].clip(min=0).astype(int)
    
    # Add important events to make price action more realistic
    # Market corrections
    for i in range(2, len(data), 90):  # Every ~3 months
        if i < len(data):
            correction_factor = np.random.uniform(0.90, 0.97)  # 3-10% correction
            correction_length = np.random.randint(3, 10)
            
            for j in range(correction_length):
                if i+j < len(data):
                    data.loc[data.index[i+j], 'open'] *= correction_factor**(j+1)/correction_length
                    data.loc[data.index[i+j], 'high'] *= correction_factor**(j+1)/correction_length
                    data.loc[data.index[i+j], 'low'] *= correction_factor**(j+1)/correction_length
                    data.loc[data.index[i+j], 'close'] *= correction_factor**(j+1)/correction_length
                    data.loc[data.index[i+j], 'volume'] *= 1.5  # Increased volume during corrections
    
    # Market rallies
    for i in range(45, len(data), 90):  # Every ~3 months, offset from corrections
        if i < len(data):
            rally_factor = np.random.uniform(1.05, 1.15)  # 5-15% rally
            rally_length = np.random.randint(5, 15)
            
            for j in range(rally_length):
                if i+j < len(data):
                    data.loc[data.index[i+j], 'open'] *= rally_factor**(j+1)/rally_length
                    data.loc[data.index[i+j], 'high'] *= rally_factor**(j+1)/rally_length
                    data.loc[data.index[i+j], 'low'] *= rally_factor**(j+1)/rally_length
                    data.loc[data.index[i+j], 'close'] *= rally_factor**(j+1)/rally_length
                    data.loc[data.index[i+j], 'volume'] *= 1.3  # Increased volume during rallies
    
    # Ensure the data is cleaned up
    data = data.reset_index().rename(columns={"index": "timestamp"})
    data = data[["timestamp", "open", "high", "low", "close", "volume"]]
    
    # Add datetime index (but keep timestamp column for compatibility)
    data.set_index("timestamp", inplace=True, drop=False)
    
    return data

def setup_timeframes(timeframe):
    """
    Configure settings for different timeframes.
    
    Args:
        timeframe: "day", "week", or "month"
        
    Returns:
        Dictionary of timeframe settings
    """
    if timeframe == "day":
        return {
            "data_freq": "1d",
            "lookback_days": 90,
            "trade_interval": "day",
            "holding_period": 1,
            "rebalance_freq": 1
        }
    elif timeframe == "week":
        return {
            "data_freq": "1w",
            "lookback_days": 180,
            "trade_interval": "week",
            "holding_period": 1,
            "rebalance_freq": 1
        }
    elif timeframe == "month":
        return {
            "data_freq": "1mo",
            "lookback_days": 365,
            "trade_interval": "month",
            "holding_period": 1,
            "rebalance_freq": 1
        }
    else:
        # Default to daily
        return {
            "data_freq": "1d",
            "lookback_days": 90,
            "trade_interval": "day",
            "holding_period": 1,
            "rebalance_freq": 1
        }

def prepare_strategy_instance(strategy_class, params, timeframe):
    """
    Prepare a strategy instance with the appropriate parameters for a timeframe.
    
    Args:
        strategy_class: Strategy class to instantiate
        params: Base parameters for the strategy
        timeframe: Trading timeframe
        
    Returns:
        Initialized strategy instance
    """
    # Adjust parameters based on timeframe
    adjusted_params = params.copy()
    
    # For example, adjust window sizes for different timeframes
    if hasattr(adjusted_params, "short_window") and hasattr(adjusted_params, "long_window"):
        if timeframe == "week":
            adjusted_params["short_window"] = max(2, params["short_window"] // 2)
            adjusted_params["long_window"] = max(5, params["long_window"] // 2)
        elif timeframe == "month":
            adjusted_params["short_window"] = max(2, params["short_window"] // 3)
            adjusted_params["long_window"] = max(3, params["long_window"] // 3)
    
    # Create instance
    return strategy_class(**adjusted_params)

async def run_backtest(strategy, data, initial_capital, label):
    """
    Run a backtest for a strategy on the provided data.
    
    Args:
        strategy: Strategy instance
        data: Market data DataFrame
        initial_capital: Starting capital
        label: Label for the backtest
        
    Returns:
        Dictionary with backtest results
    """
    try:
        # Preprocess data
        processed_data = await strategy.preprocess_data(data)
        
        # Run backtest (if supported by the strategy)
        if hasattr(strategy, "backtest") and callable(getattr(strategy, "backtest")):
            backtest_result = await strategy.backtest(
                processed_data, 
                initial_capital=initial_capital,
                label=label
            )
            return backtest_result
        
        # Alternative approach if backtest method is not available
        positions = []
        equity_curve = [initial_capital]
        trades = []
        cash = initial_capital
        position = None
        
        # Generate signals
        for i in range(len(processed_data)):
            current_data = processed_data.iloc[:i+1]
            if len(current_data) < 2:
                continue
                
            # Get signal
            signal, confidence = await strategy.predict(current_data)
            current_price = current_data.iloc[-1]["close"]
            timestamp = current_data.index[-1]
            
            # Process signal
            if signal.name == "BUY" and position is None:
                # Calculate position size
                position_size = cash / current_price
                
                # Open position
                position = {
                    "entry_time": timestamp,
                    "entry_price": current_price,
                    "size": position_size,
                    "value": cash
                }
                
                # Record trade
                trades.append({
                    "entry_time": timestamp,
                    "entry_price": current_price,
                    "exit_time": None,
                    "exit_price": None,
                    "profit_loss": 0,
                    "profit_loss_pct": 0
                })
                
                # Update cash
                cash = 0
                
            elif signal.name == "SELL" and position is not None:
                # Calculate profit/loss
                exit_value = position["size"] * current_price
                profit_loss = exit_value - position["value"]
                profit_loss_pct = profit_loss / position["value"]
                
                # Update last trade
                last_trade = trades[-1]
                last_trade["exit_time"] = timestamp
                last_trade["exit_price"] = current_price
                last_trade["profit_loss"] = profit_loss
                last_trade["profit_loss_pct"] = profit_loss_pct
                
                # Close position
                cash = exit_value
                position = None
            
            # Calculate equity
            current_equity = cash
            if position is not None:
                current_equity += position["size"] * current_price
                
            equity_curve.append(current_equity)
            positions.append(position)
        
        # Create basic backtest result
        backtest_result = {
            "equity_curve": equity_curve,
            "trades": trades,
            "final_equity": equity_curve[-1] if equity_curve else initial_capital,
            "processed_data": processed_data
        }
        
        return backtest_result
        
    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        # Return minimal result in case of error
        return {
            "equity_curve": [initial_capital],
            "trades": [],
            "final_equity": initial_capital,
            "error": str(e)
        }

def calculate_performance_metrics(backtest_result):
    """
    Calculate performance metrics from backtest results.
    
    Args:
        backtest_result: Dictionary with backtest results
        
    Returns:
        Dictionary with performance metrics
    """
    try:
        # Extract basic data
        initial_capital = backtest_result["equity_curve"][0] if backtest_result["equity_curve"] else 0
        final_capital = backtest_result["final_equity"]
        trades = backtest_result["trades"]
        
        # Calculate returns
        total_return = (final_capital / initial_capital) - 1 if initial_capital > 0 else 0
        
        # Try to extract equity curve for further calculations
        if "equity_curve" in backtest_result and len(backtest_result["equity_curve"]) > 1:
            equity_curve = pd.Series(backtest_result["equity_curve"])
            
            # Calculate daily returns
            daily_returns = equity_curve.pct_change().dropna()
            
            # Sharpe ratio (annualized)
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 0 and daily_returns.std() > 0 else 0
            
            # Maximum drawdown
            peak = equity_curve.expanding().max()
            drawdown = (equity_curve / peak - 1)
            max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
            
            # Annualized return
            if len(equity_curve) > 1:
                days = len(equity_curve)
                annualized_return = (1 + total_return) ** (252 / days) - 1
            else:
                annualized_return = 0
        else:
            # Defaults if equity curve is not available
            sharpe_ratio = 0
            max_drawdown = 0
            annualized_return = 0
        
        # Win rate and other trade metrics
        if trades:
            winning_trades = [t for t in trades if t.get("profit_loss", 0) > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            # Average profit/loss
            if winning_trades:
                avg_profit = sum(t.get("profit_loss", 0) for t in winning_trades) / len(winning_trades)
            else:
                avg_profit = 0
                
            losing_trades = [t for t in trades if t.get("profit_loss", 0) <= 0]
            if losing_trades:
                avg_loss = sum(t.get("profit_loss", 0) for t in losing_trades) / len(losing_trades)
            else:
                avg_loss = 0
                
            # Profit factor
            total_profit = sum(t.get("profit_loss", 0) for t in winning_trades)
            total_loss = abs(sum(t.get("profit_loss", 0) for t in losing_trades))
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        else:
            win_rate = 0
            avg_profit = 0
            avg_loss = 0
            profit_factor = 0
        
        # Return metrics
        return {
            "initial_value": initial_capital,
            "final_value": final_capital,
            "total_return": total_return,
            "annualized_return": annualized_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "trades_count": len(trades),
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "avg_profit": avg_profit,
            "avg_loss": avg_loss
        }
        
    except Exception as e:
        logger.error(f"Error calculating performance metrics: {e}")
        # Return default metrics in case of error
        return {
            "initial_value": initial_capital if 'initial_capital' in locals() else 0,
            "final_value": final_capital if 'final_capital' in locals() else 0,
            "total_return": 0,
            "annualized_return": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "trades_count": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "avg_profit": 0,
            "avg_loss": 0
        }

def generate_performance_report(results_df, output_dir):
    """
    Generate a comprehensive performance report from simulation results.
    
    Args:
        results_df: DataFrame with simulation results
        output_dir: Directory to save the report
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Generate summary report
        with open(f"{output_dir}/performance_summary.txt", "w") as f:
            f.write("MERCURIO AI STRATEGY SIMULATION - PERFORMANCE SUMMARY\n")
            f.write("====================================================\n\n")
            
            # Overall statistics
            f.write("OVERALL STATISTICS\n")
            f.write("-----------------\n")
            f.write(f"Simulated Strategies: {results_df['Strategy'].nunique()}\n")
            f.write(f"Assets: {results_df['Asset'].nunique()}\n")
            f.write(f"Timeframes: {results_df['Timeframe'].nunique()}\n")
            f.write(f"Total Simulations: {len(results_df)}\n\n")
            
            f.write(f"Average Return: {results_df['Total Return (%)'].mean():.2f}%\n")
            f.write(f"Best Return: {results_df['Total Return (%)'].max():.2f}%\n")
            f.write(f"Worst Return: {results_df['Total Return (%)'].min():.2f}%\n\n")
            
            # Best performers
            f.write("TOP PERFORMERS\n")
            f.write("-------------\n")
            top_performers = results_df.nlargest(10, "Total Return (%)")
            
            # Format for table
            top_table = []
            for _, row in top_performers.iterrows():
                top_table.append([
                    row["Strategy"],
                    row["Asset"],
                    row["Timeframe"],
                    f"{row['Total Return (%)']:.2f}%",
                    f"{row['Sharpe Ratio']:.2f}",
                    f"{row['Max Drawdown (%)']:.2f}%"
                ])
            
            headers = ["Strategy", "Asset", "Timeframe", "Return", "Sharpe", "Max DD"]
            f.write(tabulate(top_table, headers=headers))
            f.write("\n\n")
            
            # Average returns by strategy
            f.write("AVERAGE RETURNS BY STRATEGY\n")
            f.write("--------------------------\n")
            strategy_returns = results_df.groupby("Strategy")["Total Return (%)"].mean().reset_index()
            strategy_returns = strategy_returns.sort_values("Total Return (%)", ascending=False)
            
            strategy_table = []
            for _, row in strategy_returns.iterrows():
                strategy_table.append([
                    row["Strategy"],
                    f"{row['Total Return (%)']:.2f}%"
                ])
            
            headers = ["Strategy", "Avg Return"]
            f.write(tabulate(strategy_table, headers=headers))
            f.write("\n\n")
            
            # Average returns by timeframe
            f.write("AVERAGE RETURNS BY TIMEFRAME\n")
            f.write("---------------------------\n")
            timeframe_returns = results_df.groupby("Timeframe")["Total Return (%)"].mean().reset_index()
            timeframe_returns = timeframe_returns.sort_values("Total Return (%)", ascending=False)
            
            timeframe_table = []
            for _, row in timeframe_returns.iterrows():
                timeframe_table.append([
                    row["Timeframe"],
                    f"{row['Total Return (%)']:.2f}%"
                ])
            
            headers = ["Timeframe", "Avg Return"]
            f.write(tabulate(timeframe_table, headers=headers))
            f.write("\n\n")
            
            # Average returns by asset
            f.write("AVERAGE RETURNS BY ASSET\n")
            f.write("-----------------------\n")
            asset_returns = results_df.groupby("Asset")["Total Return (%)"].mean().reset_index()
            asset_returns = asset_returns.sort_values("Total Return (%)", ascending=False)
            
            asset_table = []
            for _, row in asset_returns.iterrows():
                asset_table.append([
                    row["Asset"],
                    f"{row['Total Return (%)']:.2f}%"
                ])
            
            headers = ["Asset", "Avg Return"]
            f.write(tabulate(asset_table, headers=headers))
            f.write("\n\n")
            
            # Footer
            f.write("NOTE: Full results available in the CSV files in this directory.\n")
            f.write("Generated charts can be found in the 'charts/' subdirectory.\n")
        
        # 2. Generate strategy reports
        for strategy in results_df["Strategy"].unique():
            strategy_df = results_df[results_df["Strategy"] == strategy]
            
            with open(f"{output_dir}/{strategy}_report.txt", "w") as f:
                f.write(f"STRATEGY REPORT: {strategy}\n")
                f.write("=" * (16 + len(strategy)) + "\n\n")
                
                # Overall performance
                avg_return = strategy_df["Total Return (%)"].mean()
                avg_sharpe = strategy_df["Sharpe Ratio"].mean()
                avg_drawdown = strategy_df["Max Drawdown (%)"].mean()
                avg_trades = strategy_df["Trades"].mean()
                
                f.write(f"Average Return: {avg_return:.2f}%\n")
                f.write(f"Average Sharpe Ratio: {avg_sharpe:.2f}\n")
                f.write(f"Average Max Drawdown: {avg_drawdown:.2f}%\n")
                f.write(f"Average Trades per Simulation: {avg_trades:.1f}\n\n")
                
                # Performance by timeframe
                f.write("PERFORMANCE BY TIMEFRAME\n")
                f.write("----------------------\n")
                timeframe_perf = strategy_df.groupby("Timeframe")["Total Return (%)"].mean().reset_index()
                timeframe_perf = timeframe_perf.sort_values("Total Return (%)", ascending=False)
                
                timeframe_table = []
                for _, row in timeframe_perf.iterrows():
                    timeframe_table.append([
                        row["Timeframe"],
                        f"{row['Total Return (%)']:.2f}%"
                    ])
                
                headers = ["Timeframe", "Avg Return"]
                f.write(tabulate(timeframe_table, headers=headers))
                f.write("\n\n")
                
                # Best asset combinations
                f.write("TOP ASSET COMBINATIONS\n")
                f.write("--------------------\n")
                top_assets = strategy_df.nlargest(5, "Total Return (%)")
                
                asset_table = []
                for _, row in top_assets.iterrows():
                    asset_table.append([
                        row["Asset"],
                        row["Timeframe"],
                        f"{row['Total Return (%)']:.2f}%",
                        f"{row['Sharpe Ratio']:.2f}",
                        f"{row['Max Drawdown (%)']:.2f}%"
                    ])
                
                headers = ["Asset", "Timeframe", "Return", "Sharpe", "Max DD"]
                f.write(tabulate(asset_table, headers=headers))
                f.write("\n\n")
        
        logger.info(f"Generated performance reports in {output_dir}")
        
    except Exception as e:
        logger.error(f"Error generating performance report: {e}")
        # Create minimal report in case of error
        with open(f"{output_dir}/performance_summary.txt", "w") as f:
            f.write("SIMULATION REPORT\n")
            f.write("=================\n\n")
            f.write("Error generating detailed report. See logs for more information.\n")
            f.write(f"Error: {str(e)}\n")
