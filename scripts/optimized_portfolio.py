"""
Mercurio AI - Optimized Portfolio Strategy

This script implements the optimized trading strategy portfolio
using the parameters determined from our January 2025 simulations.
It utilizes Mercurio AI's fallback mechanisms for testing without API keys.
"""
import os
import asyncio
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure output directories exist
os.makedirs('reports/portfolio', exist_ok=True)

# Optimized strategy parameters from January 2025 simulations
STRATEGY_PARAMS = {
    "ETH-USD": {"short_window": 10, "long_window": 15, "use_ml": False},
    "GOOGL": {"short_window": 7, "long_window": 10, "use_ml": False},
    "BTC-USD": {"short_window": 10, "long_window": 15, "use_ml": False},
    "MSFT": {"short_window": 7, "long_window": 20, "use_ml": False},
    "AAPL": {"short_window": 10, "long_window": 15, "use_ml": False}
}

# Recommended portfolio allocation
PORTFOLIO_ALLOCATION = {
    "ETH-USD": 0.40,  # 40%
    "GOOGL": 0.25,    # 25%
    "BTC-USD": 0.20,  # 20%
    "MSFT": 0.10,     # 10%
    "AAPL": 0.05      # 5%
}

# Trading configuration
INITIAL_CAPITAL = 10000  # $10,000 total portfolio
TRADING_MODE = "paper"   # "paper" or "live"

class OptimizedPortfolio:
    """
    Implements the optimized portfolio strategy using Mercurio AI's trading platform.
    """
    
    def __init__(self, initial_capital=10000, trading_mode="paper"):
        """Initialize the portfolio."""
        self.initial_capital = initial_capital
        self.trading_mode = trading_mode
        self.portfolio = {}
        self.strategies = {}
        self.market_data = None
        self.trading_service = None
        
        # Portfolio positions and performance tracking
        self.positions = {}
        self.performance = {
            "initial_capital": initial_capital,
            "current_value": initial_capital,
            "returns": [],
            "trades": []
        }
    
    async def initialize(self):
        """Initialize market data and trading services."""
        try:
            # Import services with fallback mechanisms
            from app.services.market_data import MarketDataService
            from app.services.trading import TradingService
            
            # Initialize market data service with fallback to sample data
            self.market_data = MarketDataService()
            logger.info("Market data service initialized")
            
            # Initialize trading service in paper mode (with fallback mechanisms)
            self.trading_service = TradingService(mode=self.trading_mode)
            logger.info(f"Trading service initialized in {self.trading_mode} mode")
            
            # Initialize each strategy for each asset
            for symbol, allocation in PORTFOLIO_ALLOCATION.items():
                # Calculate allocated capital
                allocated_capital = self.initial_capital * allocation
                
                # Get strategy parameters
                params = STRATEGY_PARAMS.get(symbol, {
                    "short_window": 10, 
                    "long_window": 15, 
                    "use_ml": False
                })
                
                # Initialize strategy
                from app.strategies.moving_average import MovingAverageStrategy
                strategy = MovingAverageStrategy(
                    short_window=params["short_window"],
                    long_window=params["long_window"],
                    use_ml=params["use_ml"]
                )
                
                # Store strategy and initial portfolio allocation
                self.strategies[symbol] = strategy
                self.portfolio[symbol] = {
                    "allocation": allocation,
                    "capital": allocated_capital,
                    "strategy": strategy,
                    "position": None
                }
                
                logger.info(f"Initialized {symbol} strategy: MA({params['short_window']}/{params['long_window']}) with ${allocated_capital:.2f} allocated")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing portfolio: {e}")
            return False
    
    async def run_portfolio(self, days=30):
        """
        Run the portfolio strategy for a specified number of days.
        
        Args:
            days: Number of days to run the simulation
        """
        logger.info(f"Running portfolio strategy for {days} days")
        
        # Define date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Daily performance tracking
        daily_performance = []
        
        # Process each day
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            logger.info(f"Processing date: {date_str}")
            
            # Track daily portfolio value
            daily_value = 0
            
            # Process each asset in portfolio
            for symbol, details in self.portfolio.items():
                strategy = details["strategy"]
                allocated_capital = details["capital"]
                
                try:
                    # Get latest data
                    data = await self.market_data.get_historical_data(
                        symbol, 
                        start_date=current_date - timedelta(days=60),  # Get enough history for indicators
                        end_date=current_date
                    )
                    
                    if data.empty:
                        logger.warning(f"No data available for {symbol} on {date_str}")
                        continue
                    
                    # Preprocess data
                    processed_data = await strategy.preprocess_data(data)
                    
                    # Get trading signal
                    signal, confidence = await strategy.predict(processed_data)
                    
                    # Get current position
                    position = self.positions.get(symbol)
                    
                    # Get latest price
                    latest_price = await self.market_data.get_latest_price(symbol)
                    
                    # Process signal
                    if signal.name == "BUY" and position is None:
                        # Calculate quantity to buy
                        quantity = allocated_capital / latest_price
                        
                        # Execute trade
                        if self.trading_service:
                            trade_result = await self.trading_service.place_order(
                                symbol=symbol,
                                quantity=quantity,
                                side="buy",
                                price=latest_price
                            )
                            
                            # Record trade
                            trade = {
                                "date": date_str,
                                "symbol": symbol,
                                "action": "BUY",
                                "price": latest_price,
                                "quantity": quantity,
                                "value": quantity * latest_price
                            }
                            self.performance["trades"].append(trade)
                            
                            # Update position
                            self.positions[symbol] = {
                                "entry_date": date_str,
                                "entry_price": latest_price,
                                "quantity": quantity,
                                "current_price": latest_price,
                                "current_value": quantity * latest_price
                            }
                            
                            logger.info(f"BUY {quantity:.4f} {symbol} at ${latest_price:.2f}")
                    
                    elif signal.name == "SELL" and position is not None:
                        # Execute trade
                        if self.trading_service:
                            trade_result = await self.trading_service.place_order(
                                symbol=symbol,
                                quantity=position["quantity"],
                                side="sell",
                                price=latest_price
                            )
                            
                            # Calculate profit/loss
                            entry_value = position["entry_price"] * position["quantity"]
                            exit_value = latest_price * position["quantity"]
                            pnl = exit_value - entry_value
                            pnl_pct = (exit_value / entry_value - 1) * 100
                            
                            # Record trade
                            trade = {
                                "date": date_str,
                                "symbol": symbol,
                                "action": "SELL",
                                "price": latest_price,
                                "quantity": position["quantity"],
                                "value": exit_value,
                                "pnl": pnl,
                                "pnl_pct": pnl_pct
                            }
                            self.performance["trades"].append(trade)
                            
                            logger.info(f"SELL {position['quantity']:.4f} {symbol} at ${latest_price:.2f} (P&L: ${pnl:.2f}, {pnl_pct:.2f}%)")
                            
                            # Reset position
                            self.positions[symbol] = None
                    
                    # Update position value if exists
                    if self.positions.get(symbol):
                        self.positions[symbol]["current_price"] = latest_price
                        self.positions[symbol]["current_value"] = self.positions[symbol]["quantity"] * latest_price
                        daily_value += self.positions[symbol]["current_value"]
                    else:
                        # Add allocated capital to daily value if not invested
                        daily_value += allocated_capital
                
                except Exception as e:
                    logger.error(f"Error processing {symbol} on {date_str}: {e}")
                    # Add allocated capital to daily value if error
                    daily_value += allocated_capital
            
            # Record daily performance
            daily_performance.append({
                "date": date_str,
                "portfolio_value": daily_value
            })
            
            # Update overall performance
            self.performance["current_value"] = daily_value
            self.performance["returns"].append({
                "date": date_str,
                "value": daily_value,
                "return_pct": (daily_value / self.initial_capital - 1) * 100
            })
            
            # Move to next day
            current_date += timedelta(days=1)
        
        # Calculate final performance metrics
        self._calculate_performance_metrics()
        
        # Save portfolio results
        self._save_portfolio_results()
        
        logger.info("Portfolio strategy execution completed")
        
        return self.performance
    
    def _calculate_performance_metrics(self):
        """Calculate portfolio performance metrics."""
        if not self.performance["returns"]:
            return
        
        # Get returns data
        returns_data = pd.DataFrame(self.performance["returns"])
        
        # Calculate metrics
        initial_value = self.initial_capital
        final_value = self.performance["current_value"]
        total_return = (final_value / initial_value - 1) * 100
        
        # Calculate daily returns
        returns_data["daily_return"] = returns_data["value"].pct_change()
        
        # Calculate metrics
        sharpe_ratio = np.sqrt(252) * returns_data["daily_return"].mean() / returns_data["daily_return"].std() if len(returns_data) > 1 else 0
        max_drawdown = self._calculate_max_drawdown(returns_data["value"])
        
        # Store metrics
        self.performance["metrics"] = {
            "initial_value": initial_value,
            "final_value": final_value,
            "total_return": total_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "trades_count": len(self.performance["trades"])
        }
    
    def _calculate_max_drawdown(self, values):
        """Calculate maximum drawdown from a series of values."""
        if len(values) <= 1:
            return 0
            
        # Calculate drawdown series
        values_series = pd.Series(values)
        peak = values_series.expanding().max()
        drawdown = (values_series / peak - 1) * 100
        
        return drawdown.min()
    
    def _save_portfolio_results(self):
        """Save portfolio results to file."""
        # Create output directory
        os.makedirs("reports/portfolio", exist_ok=True)
        
        # Save performance metrics
        if "metrics" in self.performance:
            metrics = self.performance["metrics"]
            metrics_file = "reports/portfolio/performance_metrics.txt"
            
            with open(metrics_file, "w") as f:
                f.write("PORTFOLIO PERFORMANCE METRICS\n")
                f.write("============================\n\n")
                f.write(f"Initial Capital: ${metrics['initial_value']:.2f}\n")
                f.write(f"Final Value: ${metrics['final_value']:.2f}\n")
                f.write(f"Total Return: {metrics['total_return']:.2f}%\n")
                f.write(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n")
                f.write(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%\n")
                f.write(f"Number of Trades: {metrics['trades_count']}\n")
            
            logger.info(f"Performance metrics saved to {metrics_file}")
        
        # Save trade history
        trades_df = pd.DataFrame(self.performance["trades"])
        if not trades_df.empty:
            trades_file = "reports/portfolio/trade_history.csv"
            trades_df.to_csv(trades_file, index=False)
            logger.info(f"Trade history saved to {trades_file}")
        
        # Save returns data
        returns_df = pd.DataFrame(self.performance["returns"])
        if not returns_df.empty:
            returns_file = "reports/portfolio/daily_returns.csv"
            returns_df.to_csv(returns_file, index=False)
            logger.info(f"Daily returns saved to {returns_file}")
            
            # Generate performance chart
            self._generate_performance_chart(returns_df)
    
    def _generate_performance_chart(self, returns_df):
        """Generate portfolio performance chart."""
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(pd.to_datetime(returns_df["date"]), returns_df["return_pct"])
            plt.title("Portfolio Performance")
            plt.xlabel("Date")
            plt.ylabel("Return (%)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save chart
            chart_file = "reports/portfolio/performance_chart.png"
            plt.savefig(chart_file, dpi=300)
            plt.close()
            
            logger.info(f"Performance chart saved to {chart_file}")
            
        except Exception as e:
            logger.error(f"Error generating performance chart: {e}")

async def main():
    """Main entry point."""
    print("\n===== MERCURIO AI OPTIMIZED PORTFOLIO =====\n")
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Trading Mode: {TRADING_MODE}")
    print("=" * 50)
    
    # Print allocation
    print("\nPortfolio Allocation:")
    for symbol, allocation in PORTFOLIO_ALLOCATION.items():
        params = STRATEGY_PARAMS.get(symbol, {})
        print(f"  {symbol}: {allocation*100:.1f}% (${INITIAL_CAPITAL*allocation:.2f}) - MA({params.get('short_window', 10)}/{params.get('long_window', 15)})")
    
    # Initialize portfolio
    portfolio = OptimizedPortfolio(
        initial_capital=INITIAL_CAPITAL,
        trading_mode=TRADING_MODE
    )
    
    # Initialize services and strategies
    success = await portfolio.initialize()
    if not success:
        print("Failed to initialize portfolio. Check logs for details.")
        return
    
    # Ask for confirmation
    print("\nPortfolio initialized. Ready to execute trading strategy.")
    confirmation = input("Proceed with trading strategy? (y/n): ")
    
    if confirmation.lower() != 'y':
        print("Trading strategy cancelled.")
        return
    
    # Run portfolio strategy
    print("\nExecuting trading strategy...")
    performance = await portfolio.run_portfolio(days=30)
    
    # Print summary
    if "metrics" in performance:
        metrics = performance["metrics"]
        print("\n===== PORTFOLIO PERFORMANCE SUMMARY =====\n")
        print(f"Initial Capital: ${metrics['initial_value']:.2f}")
        print(f"Final Value: ${metrics['final_value']:.2f}")
        print(f"Total Return: {metrics['total_return']:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.2f}%")
        print(f"Number of Trades: {metrics['trades_count']}")
        
        print("\nView detailed reports in the 'reports/portfolio/' directory")
        print("\nRun 'streamlit run strategy_dashboard.py' for an interactive dashboard")

if __name__ == "__main__":
    asyncio.run(main())
