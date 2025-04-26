"""
Mercurio AI - Strategy Simulator Final

This script runs simulations for all available trading strategies using
synthetic but realistic market data for January 2025.
"""
import os
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from tabulate import tabulate
import traceback
import matplotlib.pyplot as plt
import seaborn as sns

# Import our data generator
from data_generator import generate_market_data, generate_all_market_data, load_market_data

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure reports directory exists
os.makedirs('reports', exist_ok=True)
os.makedirs('data', exist_ok=True)

class StrategySimulator:
    """
    Runs trading strategy simulations using synthetic market data.
    """
    
    def __init__(self, initial_capital=2000):
        """Initialize the simulator."""
        self.initial_capital = initial_capital
        self.results = []
        self.strategies = {}
        
        # Define specific date range for the last 10 days (to ensure enough data for strategies)
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=10)
        print(f"Simulation period: {self.start_date.strftime('%Y-%m-%d %H:%M:%S')} to {self.end_date.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Define stocks and cryptos to test
        self.stocks = ['AAPL', 'MSFT', 'GOOGL']
        self.cryptos = ['BTC-USD', 'ETH-USD']
        self.all_symbols = self.stocks + self.cryptos
        
    def initialize_strategies(self):
        """Initialize all available strategies with appropriate parameters."""
        print("Initializing strategies...")
        
        # Import and initialize strategies
        try:
            # MovingAverage with smaller windows for January data
            from app.strategies.moving_average import MovingAverageStrategy
            self.strategies["MovingAverage"] = MovingAverageStrategy(
                short_window=2, 
                long_window=3
            )
            print("✓ Added MovingAverage strategy")
        except Exception as e:
            print(f"Failed to initialize MovingAverage strategy: {e}")
            traceback.print_exc()
        
        try:
            # MovingAverage with ML using smaller windows
            from app.strategies.moving_average import MovingAverageStrategy
            self.strategies["MovingAverage_ML"] = MovingAverageStrategy(
                short_window=2, 
                long_window=3, 
                use_ml=True
            )
            print("✓ Added MovingAverage_ML strategy")
        except Exception as e:
            print(f"Failed to initialize MovingAverage_ML strategy: {e}")
        
        try:
            # LSTM with smaller sequence length for January data
            from app.strategies.lstm_predictor import LSTMPredictorStrategy
            self.strategies["LSTM"] = LSTMPredictorStrategy(
                sequence_length=2,  # Further reduced for short synthetic data
                prediction_horizon=1,
                epochs=20,
                batch_size=4
            )
            print("✓ Added LSTM strategy")
        except Exception as e:
            print(f"Failed to initialize LSTM strategy: {e}")
        
        try:
            # LLM strategy
            from app.strategies.llm_strategy import LLMStrategy
            self.strategies["LLM"] = LLMStrategy()
            print("✓ Added LLM strategy")
        except Exception as e:
            print(f"Failed to initialize LLM strategy: {e}")
        
        try:
            # Transformer strategy
            from app.strategies.transformer_strategy import TransformerStrategy
            self.strategies["Transformer"] = TransformerStrategy()
            print("✓ Added Transformer strategy")
        except Exception as e:
            print(f"Failed to initialize Transformer strategy: {e}")
        
        print(f"Initialized {len(self.strategies)} strategies")
    
    def generate_market_data(self):
        """Generate market data for all symbols."""
        print(f"Generating market data for {len(self.all_symbols)} symbols...")
        generate_all_market_data(self.all_symbols, self.start_date, self.end_date, 'data')
        print("Market data generation complete")
    
    async def get_data_for_strategy(self, strategy, symbol):
        """Get appropriate data for a strategy, handling format conversions."""
        try:
            # Load the generated data
            data = load_market_data(symbol)
            
            if data is None or data.empty:
                print(f"No data available for {symbol}")
                return None
            
            # Ensure data is in the format expected by strategies
            if 'timestamp' not in data.columns and isinstance(data.index, pd.DatetimeIndex):
                # Add a timestamp column if needed
                data['timestamp'] = range(len(data))
            
            # Reset index if it's a DatetimeIndex to avoid issues with some strategies
            if isinstance(data.index, pd.DatetimeIndex):
                data = data.reset_index()
            
            # Ensure column names are lowercase
            data.columns = [col.lower() for col in data.columns]
            
            # Handle date column if needed
            if 'date' in data.columns and pd.api.types.is_datetime64_any_dtype(data['date']):
                # Some strategies expect a string date
                data['date_str'] = data['date'].dt.strftime('%Y-%m-%d')
            
            # Fill any NaN values
            data = data.ffill().bfill()
            
            print(f"Prepared data for {symbol}: {len(data)} rows, columns: {data.columns.tolist()}")
            return data
            
        except Exception as e:
            print(f"Error getting data for {symbol}: {e}")
            traceback.print_exc()
            return None
    
    async def run_strategy_backtest(self, strategy_name, strategy, symbol):
        """Run backtest for a single strategy on a single symbol."""
        print(f"Testing {strategy_name} on {symbol}")
        result = {
            "symbol": symbol,
            "strategy": strategy_name,
            "initial_capital": self.initial_capital,
            "final_capital": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "trades": 0,
            "error": None
        }
        
        try:
            # Get data for this strategy/symbol
            data = await self.get_data_for_strategy(strategy, symbol)
            if data is None or data.empty:
                result["error"] = "No data available"
                return result
            
            # Preprocess data
            try:
                print(f"  Preprocessing data for {strategy_name} on {symbol}...")
                processed_data = await strategy.preprocess_data(data)
                if processed_data.empty:
                    result["error"] = "Preprocessing resulted in empty dataset"
                    return result
                print(f"  Preprocessed data shape: {processed_data.shape}")
            except Exception as e:
                print(f"  Error preprocessing data: {e}")
                traceback.print_exc()
                result["error"] = f"Preprocessing error: {str(e)}"
                return result
            
            # Train if needed
            if hasattr(strategy, 'train') and not strategy.is_trained:
                try:
                    print(f"  Training {strategy_name}...")
                    await strategy.train(processed_data)
                    print(f"  Training complete")
                except Exception as e:
                    print(f"  Training error: {e}")
                    traceback.print_exc()
                    # For strategies that must be trained, this is a critical error
                    if strategy_name in ["LSTM", "Transformer"]:
                        result["error"] = f"Training error: {str(e)}"
                        return result
            
            # Run backtest
            try:
                print(f"  Running backtest...")
                backtest_results = await strategy.backtest(processed_data, initial_capital=self.initial_capital)
                
                # Extract key metrics
                result["final_capital"] = backtest_results.get('final_capital', 0)
                result["total_return"] = backtest_results.get('total_return', 0)
                result["annualized_return"] = backtest_results.get('annualized_return', 0)
                result["sharpe_ratio"] = backtest_results.get('sharpe_ratio', 0)
                result["max_drawdown"] = backtest_results.get('max_drawdown', 0)
                result["trades"] = backtest_results.get('trades', 0)
                
                # Save backtest data for visualization
                if 'backtest_data' in backtest_results:
                    result["backtest_data"] = backtest_results['backtest_data']
                
                print(f"  Backtest complete: ${result['final_capital']:.2f} ({result['total_return']*100:.2f}% return)")
                
            except Exception as e:
                print(f"  Backtest error: {e}")
                traceback.print_exc()
                result["error"] = f"Backtest error: {str(e)}"
            
        except Exception as e:
            print(f"Error testing {strategy_name} on {symbol}: {e}")
            traceback.print_exc()
            result["error"] = str(e)
        
        return result
    
    async def run_simulations(self):
        """Run backtests for all strategies on all symbols."""
        print(f"Starting simulation for {len(self.strategies)} strategies on {len(self.all_symbols)} symbols")
        
        for symbol in self.all_symbols:
            for strategy_name, strategy in self.strategies.items():
                result = await self.run_strategy_backtest(strategy_name, strategy, symbol)
                self.results.append(result)
                
        print(f"Completed {len(self.results)} strategy-symbol combinations")
    
    def generate_reports(self):
        """Generate reports and visualizations from the simulation results."""
        print("Generating reports...")
        
        if not self.results:
            print("No results to report")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        # Clean up data for display
        display_df = df.copy()
        
        # Process columns for display
        display_df['initial_capital'] = display_df['initial_capital'].apply(lambda x: f"${x:.2f}")
        display_df['final_capital'] = display_df['final_capital'].apply(lambda x: f"${x:.2f}")
        display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x*100:.2f}%")
        display_df['annualized_return'] = display_df['annualized_return'].apply(lambda x: f"{x*100:.2f}%")
        display_df['sharpe_ratio'] = display_df['sharpe_ratio'].apply(lambda x: f"{x:.2f}")
        display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x*100:.2f}%")
        
        # Remove backtest_data for display
        if 'backtest_data' in display_df.columns:
            display_df = display_df.drop(columns=['backtest_data'])
        
        # Save results to CSV
        display_df.to_csv('reports/strategy_comparison.csv', index=False)
        print("Saved results to reports/strategy_comparison.csv")
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Print summary table
        print("\n===== STRATEGY COMPARISON RESULTS =====\n")
        
        # Create separate tables for successful runs and errors
        successful_df = display_df[display_df['error'].isna()].drop(columns=['error'])
        error_df = display_df[~display_df['error'].isna()][['symbol', 'strategy', 'error']]
        
        if not successful_df.empty:
            print(tabulate(successful_df, headers='keys', tablefmt='grid'))
            
            # Find best strategy overall
            numeric_df = df[df['error'].isna()]
            if not numeric_df.empty:
                best_idx = numeric_df['total_return'].idxmax()
                best_strategy = numeric_df.loc[best_idx]
                
                print("\n===== BEST PERFORMING STRATEGIES =====\n")
                print(f"Best Overall: {best_strategy['strategy']} on {best_strategy['symbol']}")
                print(f"  Return: {best_strategy['total_return']*100:.2f}%")
                print(f"  Initial: ${best_strategy['initial_capital']:.2f}, Final: ${best_strategy['final_capital']:.2f}")
                
                # Best by asset class
                stocks_df = numeric_df[~numeric_df['symbol'].str.contains('-USD')]
                crypto_df = numeric_df[numeric_df['symbol'].str.contains('-USD')]
                
                if not stocks_df.empty:
                    best_stock_idx = stocks_df['total_return'].idxmax()
                    best_stock = stocks_df.loc[best_stock_idx]
                    print(f"\nBest Stock Strategy: {best_stock['strategy']} on {best_stock['symbol']}")
                    print(f"  Return: {best_stock['total_return']*100:.2f}%")
                
                if not crypto_df.empty:
                    best_crypto_idx = crypto_df['total_return'].idxmax()
                    best_crypto = crypto_df.loc[best_crypto_idx]
                    print(f"\nBest Crypto Strategy: {best_crypto['strategy']} on {best_crypto['symbol']}")
                    print(f"  Return: {best_crypto['total_return']*100:.2f}%")
                
                # Best by strategy type
                print("\nAverage Returns by Strategy:")
                avg_returns = numeric_df.groupby('strategy')['total_return'].mean()
                for strategy_name, avg_return in avg_returns.items():
                    print(f"  {strategy_name}: {avg_return*100:.2f}%")
        
        if not error_df.empty:
            print("\n===== STRATEGIES WITH ERRORS =====\n")
            print(tabulate(error_df, headers=['Symbol', 'Strategy', 'Error'], tablefmt='grid'))
        
        print("\nResults saved to reports/strategy_comparison.csv")
        print("Run 'streamlit run strategy_dashboard.py' to view the interactive dashboard.")
    
    def _generate_visualizations(self):
        """Generate visualizations of the simulation results."""
        try:
            # Only include successful runs
            df = pd.DataFrame(self.results)
            numeric_df = df[df['error'].isna()].copy()
            
            if numeric_df.empty:
                print("No successful runs to visualize")
                return
            
            # Create directory for visualizations
            os.makedirs('reports/visualizations', exist_ok=True)
            
            # 1. Returns by Strategy and Symbol
            plt.figure(figsize=(12, 8))
            
            # Prepare data for plotting
            pivot_df = numeric_df.pivot_table(
                index='symbol', 
                columns='strategy', 
                values='total_return',
                aggfunc='mean'
            )
            
            # Create heatmap
            sns.heatmap(
                pivot_df * 100,  # Convert to percentage
                annot=True, 
                fmt=".2f", 
                cmap="YlGnBu",
                linewidths=0.5,
                cbar_kws={'label': 'Return (%)'}
            )
            
            plt.title('Strategy Returns by Symbol (January 2025)')
            plt.tight_layout()
            plt.savefig('reports/visualizations/returns_heatmap.png', dpi=300)
            plt.close()
            
            # 2. Risk-Return Scatterplot
            plt.figure(figsize=(10, 8))
            
            scatter = plt.scatter(
                numeric_df['max_drawdown'] * 100,  # Convert to percentage
                numeric_df['sharpe_ratio'],
                c=pd.factorize(numeric_df['strategy'])[0],
                s=100,
                alpha=0.7
            )
            
            # Add strategy names as legend
            strategies = numeric_df['strategy'].unique()
            plt.legend(
                scatter.legend_elements()[0], 
                strategies,
                title="Strategy",
                loc="upper left"
            )
            
            # Add annotations for each point
            for i, row in numeric_df.iterrows():
                plt.annotate(
                    row['symbol'],
                    (row['max_drawdown'] * 100, row['sharpe_ratio']),
                    xytext=(5, 5),
                    textcoords='offset points'
                )
            
            plt.xlabel('Maximum Drawdown (%)')
            plt.ylabel('Sharpe Ratio')
            plt.title('Risk-Return Profile by Strategy and Symbol')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig('reports/visualizations/risk_return_scatter.png', dpi=300)
            plt.close()
            
            # 3. Bar chart of returns by strategy
            plt.figure(figsize=(10, 6))
            
            strategy_returns = numeric_df.groupby('strategy')['total_return'].mean() * 100
            strategy_returns.sort_values(ascending=False).plot(kind='bar')
            
            plt.xlabel('Strategy')
            plt.ylabel('Average Return (%)')
            plt.title('Average Returns by Strategy (January 2025)')
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig('reports/visualizations/strategy_returns_bar.png', dpi=300)
            plt.close()
            
            print("Visualizations saved to reports/visualizations/")
            
        except Exception as e:
            print(f"Error generating visualizations: {e}")
            traceback.print_exc()

async def main():
    """Main entry point."""
    print("\n===== MERCURIO AI STRATEGY SIMULATION =====\n")
    # Create simulator
    simulator = StrategySimulator(initial_capital=2000)

    print(f"Simulating trading strategies for the last 10 days: {simulator.start_date.strftime('%Y-%m-%d %H:%M:%S')} to {simulator.end_date.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
    # Create simulator
    simulator = StrategySimulator(initial_capital=2000)
    
    # Initialize strategies
    simulator.initialize_strategies()
    
    # Generate market data
    simulator.generate_market_data()
    
    # Run simulations
    await simulator.run_simulations()
    
    # Generate reports
    simulator.generate_reports()
    
    print("\nSimulation complete!")

if __name__ == "__main__":
    asyncio.run(main())
