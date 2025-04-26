"""
Mercurio AI - Strategy Simulator (V2)

This script runs simulations for all available trading strategies on various assets
(stocks and cryptocurrencies) over a specified period and generates a comparison report.
"""
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging
from tabulate import tabulate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import strategies
from app.strategies.moving_average import MovingAverageStrategy
from app.strategies.lstm_predictor import LSTMPredictorStrategy
from app.services.market_data import MarketDataService

# Try to import optional strategies
try:
    from app.strategies.llm_strategy import LLMStrategy
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    logger.warning("LLM Strategy not available")

try:
    from app.strategies.msi_strategy import MSIStrategy
    MSI_AVAILABLE = True
except ImportError:
    MSI_AVAILABLE = False
    logger.warning("MSI Strategy not available")

try:
    from app.strategies.transformer_strategy import TransformerStrategy
    TRANSFORMER_AVAILABLE = True
except ImportError:
    TRANSFORMER_AVAILABLE = False
    logger.warning("Transformer Strategy not available")

class StrategySimulator:
    """
    Simulates multiple trading strategies on various assets and generates a comparison report.
    """
    
    def __init__(self, initial_capital=2000):
        """
        Initialize the strategy simulator.
        
        Args:
            initial_capital: Initial capital for each strategy backtest ($)
        """
        self.initial_capital = initial_capital
        self.strategies = {}
        self.stock_symbols = ['AAPL', 'MSFT', 'AMZN', 'GOOGL', 'TSLA']
        self.crypto_symbols = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD']
        self.results = {}
        
    def initialize_strategies(self):
        """Initialize all available trading strategies."""
        self.strategies = {
            "MovingAverage": MovingAverageStrategy(short_window=20, long_window=50),
            "MovingAverage_ML": MovingAverageStrategy(short_window=20, long_window=50, use_ml=True),
            "LSTM": LSTMPredictorStrategy(sequence_length=60, prediction_horizon=1)
        }
        
        # Add optional strategies if available
        if LLM_AVAILABLE:
            self.strategies["LLM"] = LLMStrategy()
        
        if MSI_AVAILABLE:
            self.strategies["MSI"] = MSIStrategy()
        
        if TRANSFORMER_AVAILABLE:
            self.strategies["Transformer"] = TransformerStrategy()
            
        logger.info(f"Initialized {len(self.strategies)} strategies")
        
    async def run_simulation(self, months=1):
        """
        Run backtests for all strategies on all symbols.
        
        Args:
            months: Number of months of historical data to use
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30*months)
        
        logger.info(f"Running simulations from {start_date} to {end_date}")
        
        all_symbols = self.stock_symbols + self.crypto_symbols
        
        for symbol in all_symbols:
            logger.info(f"Processing {symbol}")
            symbol_results = {}
            
            for strategy_name, strategy in self.strategies.items():
                logger.info(f"Running {strategy_name} on {symbol}")
                
                try:
                    # Load data
                    data = await strategy.load_data(symbol, start_date, end_date)
                    
                    # Preprocess data
                    processed_data = await strategy.preprocess_data(data)
                    
                    # If strategy requires training, train it
                    if hasattr(strategy, 'train') and not strategy.is_trained:
                        await strategy.train(processed_data)
                    
                    # Run backtest
                    backtest_results = await strategy.backtest(processed_data, initial_capital=self.initial_capital)
                    
                    # Store results
                    symbol_results[strategy_name] = backtest_results
                    
                except Exception as e:
                    logger.error(f"Error running {strategy_name} on {symbol}: {str(e)}")
                    symbol_results[strategy_name] = {"error": str(e)}
            
            self.results[symbol] = symbol_results
            
        logger.info("All simulations completed")
        
    def generate_report(self, output_dir="reports"):
        """
        Generate a comparative report of all strategy performances.
        
        Args:
            output_dir: Directory to save the report files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for the comparison table
        comparison_data = []
        
        for symbol in self.results:
            for strategy_name, results in self.results[symbol].items():
                if "error" in results:
                    continue
                
                row = {
                    "Symbol": symbol,
                    "Strategy": strategy_name,
                    "Initial Capital": f"${self.initial_capital:.2f}",
                    "Final Capital": f"${results.get('final_capital', 0):.2f}",
                    "Total Return": f"{results.get('total_return', 0) * 100:.2f}%",
                    "Annualized Return": f"{results.get('annualized_return', 0) * 100:.2f}%",
                    "Sharpe Ratio": f"{results.get('sharpe_ratio', 0):.2f}",
                    "Max Drawdown": f"{results.get('max_drawdown', 0) * 100:.2f}%",
                    "Trades": results.get('trades', 0)
                }
                comparison_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "strategy_comparison.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison data to {csv_path}")
        
        # Generate performance charts
        self._generate_performance_charts(output_dir)
        
        # Print summary to console
        print("\n" + "="*80)
        print("STRATEGY SIMULATION RESULTS")
        print("="*80)
        print(tabulate(df, headers='keys', tablefmt='fancy_grid'))
        print("\n" + "="*80)
        print(f"Full reports saved to: {output_dir}")
        print("="*80 + "\n")

    def _generate_performance_charts(self, output_dir):
        """Generate performance comparison charts."""
        # Create directory for charts
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data for plotting
        symbols = []
        strategy_names = []
        returns = []
        
        for symbol in self.results:
            for strategy_name, results in self.results[symbol].items():
                if "error" in results:
                    continue
                symbols.append(symbol)
                strategy_names.append(strategy_name)
                returns.append(results.get('total_return', 0) * 100)  # Convert to percentage
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Symbol': symbols,
            'Strategy': strategy_names,
            'Return (%)': returns
        })
        
        # Plot returns by strategy and symbol
        plt.figure(figsize=(14, 10))
        
        # Group by symbol
        grouped = plot_df.groupby('Symbol')
        
        # Create a bar chart for each symbol
        pos = 0
        bar_width = 0.15
        colors = plt.cm.viridis(np.linspace(0, 1, len(self.strategies)))
        
        for i, (symbol, group) in enumerate(grouped):
            for j, strategy in enumerate(self.strategies.keys()):
                strategy_data = group[group['Strategy'] == strategy]
                if not strategy_data.empty:
                    plt.bar(
                        pos + j * bar_width, 
                        strategy_data['Return (%)'].values[0], 
                        width=bar_width, 
                        color=colors[j], 
                        label=strategy if i == 0 else ""
                    )
            pos += len(self.strategies) * bar_width + 0.2
        
        # Add labels and legend
        plt.xlabel('Symbol', fontsize=14)
        plt.ylabel('Return (%)', fontsize=14)
        plt.title('Strategy Returns by Symbol', fontsize=16)
        plt.xticks([pos * (i + 0.5) / len(grouped) for i in range(len(grouped))], grouped.groups.keys())
        plt.legend(title='Strategy')
        plt.grid(axis='y', alpha=0.3)
        
        # Save chart
        plt.savefig(os.path.join(output_dir, 'returns_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create risk-return chart
        sharpe_ratios = []
        max_drawdowns = []
        
        for symbol in self.results:
            for strategy_name, results in self.results[symbol].items():
                if "error" in results:
                    continue
                sharpe_ratios.append(results.get('sharpe_ratio', 0))
                max_drawdowns.append(results.get('max_drawdown', 0) * 100)  # Convert to percentage
        
        # Create scatter plot
        plt.figure(figsize=(12, 8))
        
        # Use different markers for different symbols
        for i, symbol in enumerate(set(symbols)):
            symbol_idx = [j for j, s in enumerate(symbols) if s == symbol]
            plt.scatter(
                [max_drawdowns[j] for j in symbol_idx],
                [sharpe_ratios[j] for j in symbol_idx],
                s=100,
                label=symbol,
                marker=f"${i}$"
            )
        
        # Add labels
        for i in range(len(sharpe_ratios)):
            plt.annotate(
                strategy_names[i],
                (max_drawdowns[i], sharpe_ratios[i]),
                textcoords="offset points",
                xytext=(0, 5),
                ha='center'
            )
        
        # Add axes labels and title
        plt.xlabel('Maximum Drawdown (%)', fontsize=14)
        plt.ylabel('Sharpe Ratio', fontsize=14)
        plt.title('Risk-Return Profile by Strategy and Symbol', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.legend(title='Symbol')
        
        # Save chart
        plt.savefig(os.path.join(output_dir, 'risk_return_profile.png'), dpi=300, bbox_inches='tight')
        plt.close()

async def main():
    """Main entry point for the strategy simulator."""
    # Create simulator instance
    simulator = StrategySimulator(initial_capital=2000)
    
    # Initialize strategies
    simulator.initialize_strategies()
    
    # Run simulations
    await simulator.run_simulation(months=1)
    
    # Generate comparison report
    simulator.generate_report()

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
