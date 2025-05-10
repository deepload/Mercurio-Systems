"""
Mercurio AI - Strategy Simulator (V3)

This script runs simulations for ALL available trading strategies on various assets
(stocks and cryptocurrencies) with detailed logging to identify and resolve any issues.
"""
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import logging
from tabulate import tabulate
import traceback
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add a file handler to keep logs
os.makedirs('logs', exist_ok=True)
file_handler = logging.FileHandler('logs/strategy_simulator.log')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

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
        
    async def initialize_strategies(self):
        """Initialize all available trading strategies."""
        logger.info("Initializing strategies...")
        
        # Import strategies
        try:
            from app.strategies.moving_average import MovingAverageStrategy
            logger.info("✅ Successfully imported Moving Average Strategy")
            
            # Initialize both versions of MovingAverage
            self.strategies["MovingAverage"] = MovingAverageStrategy(
                short_window=20, 
                long_window=50
            )
            logger.info("✅ Initialized MovingAverage strategy")
            
            self.strategies["MovingAverage_ML"] = MovingAverageStrategy(
                short_window=20, 
                long_window=50, 
                use_ml=True
            )
            logger.info("✅ Initialized MovingAverage_ML strategy")
        except Exception as e:
            logger.error(f"❌ Failed to import or initialize Moving Average Strategy: {e}")
            logger.error(traceback.format_exc())
        
        try:
            from app.strategies.lstm_predictor import LSTMPredictorStrategy
            logger.info("✅ Successfully imported LSTM Predictor Strategy")
            
            self.strategies["LSTM"] = LSTMPredictorStrategy(
                sequence_length=60, 
                prediction_horizon=1
            )
            logger.info("✅ Initialized LSTM strategy")
        except Exception as e:
            logger.error(f"❌ Failed to import or initialize LSTM Predictor Strategy: {e}")
            logger.error(traceback.format_exc())
        
        try:
            from app.strategies.llm_strategy import LLMStrategy
            logger.info("✅ Successfully imported LLM Strategy")
            
            self.strategies["LLM"] = LLMStrategy()
            logger.info("✅ Initialized LLM strategy")
        except Exception as e:
            logger.error(f"❌ Failed to import or initialize LLM Strategy: {e}")
            logger.error(traceback.format_exc())
        
        try:
            from app.strategies.msi_strategy import MSIStrategy
            logger.info("✅ Successfully imported MSI Strategy")
            
            self.strategies["MSI"] = MSIStrategy()
            logger.info("✅ Initialized MSI strategy")
        except Exception as e:
            logger.error(f"❌ Failed to import or initialize MSI Strategy: {e}")
            logger.error(traceback.format_exc())
        
        try:
            from app.strategies.transformer_strategy import TransformerStrategy
            logger.info("✅ Successfully imported Transformer Strategy")
            
            self.strategies["Transformer"] = TransformerStrategy()
            logger.info("✅ Initialized Transformer strategy")
        except Exception as e:
            logger.error(f"❌ Failed to import or initialize Transformer Strategy: {e}")
            logger.error(traceback.format_exc())
        
        logger.info(f"Strategy initialization complete. Total strategies: {len(self.strategies)}")
        logger.info(f"Available strategies: {list(self.strategies.keys())}")
        
        if not self.strategies:
            logger.error("No strategies were successfully initialized!")
            print("ERROR: No strategies were successfully initialized! Check logs/strategy_simulator.log for details.")
            sys.exit(1)
            
    async def run_simulation(self, months=1):
        """
        Run backtests for all strategies on all symbols.
        
        Args:
            months: Number of months of historical data to use
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30*months)
        
        logger.info(f"Running simulations from {start_date} to {end_date}")
        logger.info(f"Testing {len(self.strategies)} strategies on {len(self.stock_symbols)} stocks and {len(self.crypto_symbols)} cryptocurrencies")
        
        all_symbols = self.stock_symbols + self.crypto_symbols
        
        # Create a market data service to use across all strategies
        try:
            from app.services.market_data import MarketDataService
            market_data = MarketDataService()
            logger.info("✅ Successfully initialized Market Data Service")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Market Data Service: {e}")
            logger.error(traceback.format_exc())
            print("ERROR: Could not initialize Market Data Service. See logs for details.")
            return
        
        for symbol in all_symbols:
            logger.info(f"Processing {symbol}")
            symbol_results = {}
            
            # Load data once per symbol to ensure consistency across strategies
            try:
                logger.info(f"Loading data for {symbol}...")
                data = await market_data.get_historical_data(symbol, start_date, end_date)
                if data.empty:
                    logger.warning(f"⚠️ No data returned for {symbol}, skipping...")
                    continue
                logger.info(f"✅ Successfully loaded data for {symbol} with {len(data)} rows")
            except Exception as e:
                logger.error(f"❌ Failed to load data for {symbol}: {e}")
                logger.error(traceback.format_exc())
                continue
            
            for strategy_name, strategy in self.strategies.items():
                logger.info(f"Running {strategy_name} on {symbol}")
                
                try:
                    # Preprocess data
                    logger.info(f"Preprocessing data for {strategy_name} on {symbol}")
                    processed_data = await strategy.preprocess_data(data.copy())
                    
                    # If strategy requires training, train it
                    if hasattr(strategy, 'train') and not strategy.is_trained:
                        logger.info(f"Training {strategy_name}...")
                        await strategy.train(processed_data)
                    
                    # Run backtest
                    logger.info(f"Backtesting {strategy_name} on {symbol}")
                    backtest_results = await strategy.backtest(processed_data, initial_capital=self.initial_capital)
                    
                    logger.info(f"✅ {strategy_name} backtest on {symbol} completed. Final capital: ${backtest_results.get('final_capital', 0):.2f}")
                    
                    # Store results
                    symbol_results[strategy_name] = backtest_results
                    
                except Exception as e:
                    logger.error(f"❌ Error running {strategy_name} on {symbol}: {str(e)}")
                    logger.error(traceback.format_exc())
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
                    # Include error information
                    row = {
                        "Symbol": symbol,
                        "Strategy": strategy_name,
                        "Initial Capital": f"${self.initial_capital:.2f}",
                        "Final Capital": "$0.00",
                        "Total Return": "0.00%",
                        "Annualized Return": "0.00%",
                        "Sharpe Ratio": "0.00",
                        "Max Drawdown": "0.00%",
                        "Trades": 0,
                        "Error": results.get("error", "Unknown error")
                    }
                else:
                    row = {
                        "Symbol": symbol,
                        "Strategy": strategy_name,
                        "Initial Capital": f"${self.initial_capital:.2f}",
                        "Final Capital": f"${results.get('final_capital', 0):.2f}",
                        "Total Return": f"{results.get('total_return', 0) * 100:.2f}%",
                        "Annualized Return": f"{results.get('annualized_return', 0) * 100:.2f}%",
                        "Sharpe Ratio": f"{results.get('sharpe_ratio', 0):.2f}",
                        "Max Drawdown": f"{results.get('max_drawdown', 0) * 100:.2f}%",
                        "Trades": results.get('trades', 0),
                        "Error": ""
                    }
                comparison_data.append(row)
        
        # Create DataFrame
        df = pd.DataFrame(comparison_data)
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "strategy_comparison.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison data to {csv_path}")
        
        # Save a clean version without error column for dashboard
        if 'Error' in df.columns:
            df_clean = df.drop(columns=['Error'])
            clean_csv_path = os.path.join(output_dir, "strategy_comparison_clean.csv")
            df_clean.to_csv(clean_csv_path, index=False)
            logger.info(f"Saved clean comparison data to {clean_csv_path}")
        
        # Generate performance charts if we have enough data
        try:
            self._generate_performance_charts(output_dir)
        except Exception as e:
            logger.error(f"Failed to generate performance charts: {e}")
            logger.error(traceback.format_exc())
        
        # Count strategies with errors
        errors_df = df[df['Error'] != ""]
        error_count = len(errors_df)
        
        # Print summary to console
        print("\n" + "="*80)
        print("STRATEGY SIMULATION RESULTS")
        print("="*80)
        
        if error_count > 0:
            print(f"⚠️ {error_count} strategy-symbol combinations had errors. See logs for details.")
            
        print(tabulate(df.drop(columns=['Error']) if 'Error' in df.columns else df, 
               headers='keys', tablefmt='fancy_grid'))
        
        print("\n" + "="*80)
        print(f"Full reports saved to: {output_dir}")
        print(f"Logs saved to: logs/strategy_simulator.log")
        print("="*80 + "\n")
        
        # Report on strategy-symbol combinations with highest returns
        try:
            df_numeric = df.copy()
            df_numeric['Total Return'] = df_numeric['Total Return'].str.rstrip('%').astype('float') / 100
            top_performers = df_numeric.sort_values('Total Return', ascending=False).head(5)
            
            print("\nTOP 5 PERFORMING STRATEGY-SYMBOL COMBINATIONS:")
            print("-" * 50)
            for i, row in top_performers.iterrows():
                print(f"{row['Strategy']} on {row['Symbol']}: {row['Total Return']*100:.2f}% return")
            print("-" * 50)
        except Exception as e:
            logger.error(f"Error generating top performers report: {e}")

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
        
        if not returns:  # No valid results to plot
            logger.warning("No valid results to generate charts")
            return
        
        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'Symbol': symbols,
            'Strategy': strategy_names,
            'Return (%)': returns
        })
        
        # Plot returns by strategy and symbol
        plt.figure(figsize=(14, 10))
        
        # Determine all unique strategies
        unique_strategies = sorted(plot_df['Strategy'].unique())
        unique_symbols = sorted(plot_df['Symbol'].unique())
        
        if len(unique_strategies) == 0 or len(unique_symbols) == 0:
            logger.warning("Not enough data to generate charts")
            return
        
        # Create grouped bar chart
        width = 0.15  # width of bars
        x = np.arange(len(unique_symbols))  # the x positions for the symbols
        
        # Plot each strategy as a group of bars
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, strategy in enumerate(unique_strategies):
            strategy_data = plot_df[plot_df['Strategy'] == strategy]
            strategy_returns = []
            
            for symbol in unique_symbols:
                symbol_data = strategy_data[strategy_data['Symbol'] == symbol]
                if not symbol_data.empty:
                    strategy_returns.append(symbol_data['Return (%)'].values[0])
                else:
                    strategy_returns.append(0)  # No data for this combination
                    
            ax.bar(x + i*width - (len(unique_strategies)-1)*width/2, 
                   strategy_returns, 
                   width, 
                   label=strategy)
            
        # Add labels and legend
        ax.set_ylabel('Return (%)', fontsize=14)
        ax.set_title('Strategy Returns by Symbol', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(unique_symbols, rotation=45)
        ax.legend(title='Strategy', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save chart
        plt.savefig(os.path.join(output_dir, 'returns_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create risk-return scatter plot if we have data with Sharpe ratios and drawdowns
        sharpe_ratios = []
        max_drawdowns = []
        
        for symbol in self.results:
            for strategy_name, results in self.results[symbol].items():
                if "error" in results and (
                   'sharpe_ratio' not in results or 
                   'max_drawdown' not in results):
                    continue
                
                sharpe_ratios.append(results.get('sharpe_ratio', 0))
                max_drawdowns.append(results.get('max_drawdown', 0) * 100)  # Convert to percentage
        
        if sharpe_ratios and max_drawdowns:  # If we have risk-return data
            plt.figure(figsize=(12, 8))
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot points
            scatter = ax.scatter(
                max_drawdowns, 
                sharpe_ratios, 
                c=pd.Categorical(strategy_names).codes,  # Color by strategy
                s=100,  # Point size
                alpha=0.7,
                cmap='viridis'
            )
            
            # Add labels for each point
            for i, symbol in enumerate(symbols):
                ax.annotate(
                    symbol,
                    (max_drawdowns[i], sharpe_ratios[i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center'
                )
                
            # Add legend
            legend1 = ax.legend(
                scatter.legend_elements()[0], 
                unique_strategies,
                title="Strategy",
                loc="upper right"
            )
            ax.add_artist(legend1)
            
            # Add labels and title
            ax.set_xlabel('Maximum Drawdown (%)', fontsize=14)
            ax.set_ylabel('Sharpe Ratio', fontsize=14)
            ax.set_title('Risk-Return Profile by Strategy and Symbol', fontsize=16)
            ax.grid(True, alpha=0.3)
            
            # Save chart
            plt.savefig(os.path.join(output_dir, 'risk_return_profile.png'), dpi=300, bbox_inches='tight')
            plt.close()

async def main():
    """Main entry point for the strategy simulator."""
    print("Starting Mercurio AI Strategy Simulator v3")
    print("=" * 80)
    
    # Create simulator instance
    simulator = StrategySimulator(initial_capital=2000)
    
    # Initialize strategies
    await simulator.initialize_strategies()
    
    # Run simulations
    await simulator.run_simulation(months=1)
    
    # Generate comparison report
    simulator.generate_report()
    
    print("Simulation complete! Check the reports directory for results.")
    print("Run 'streamlit run strategy_dashboard.py' to view the interactive dashboard.")

if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
