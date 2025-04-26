"""
Mercurio AI - Fixed Strategy Comparison

This script tests all available strategies with proper error handling and
uses Mercurio's sample data provider to ensure consistent test conditions.
"""
import asyncio
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from tabulate import tabulate
import traceback
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure reports directory exists
os.makedirs('reports', exist_ok=True)

class StrategyTester:
    """Class to handle testing of trading strategies."""
    
    def __init__(self, initial_capital=2000, test_period_days=30):
        self.initial_capital = initial_capital
        self.test_period_days = test_period_days
        self.results = []
        
        # Define specific date range for January 2025
        self.start_date = datetime(2025, 1, 1)
        self.end_date = datetime(2025, 1, 31)
        
    async def run_strategy_test(self, strategy_instance, strategy_name, symbol):
        """Test a single strategy on a symbol and return results."""
        print(f"Testing {strategy_name} on {symbol}...")
        
        try:
            # Use January 2025 date range
            start_date = self.start_date
            end_date = self.end_date
            print(f"  Using date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
            
            # Use the MarketDataService directly
            from app.services.market_data import MarketDataService
            market_data = MarketDataService(provider_name="sample")  # Force sample data
            
            # Get data
            print(f"  Loading data for {symbol}...")
            data = await market_data.get_historical_data(symbol, start_date, end_date)
            
            if data.empty:
                print(f"  No data available for {symbol}")
                return self._create_error_result(strategy_name, symbol, "No data available")
            
            print(f"  Loaded {len(data)} rows of data")
            
            # Preprocess the data with detailed diagnostics
            try:
                print(f"  Preprocessing data...")
                
                # Add diagnostics for input data
                print(f"  Input data shape: {data.shape}, columns: {data.columns.tolist()}")
                if len(data) < 30:
                    print(f"  WARNING: Input data has only {len(data)} rows, which may be insufficient")
                
                # Handle missing columns that strategies might need
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in required_columns:
                    if col not in data.columns and col.upper() not in data.columns:
                        # Add missing columns with reasonable defaults
                        if col == 'volume' and 'close' in data.columns:
                            print(f"  Adding synthetic '{col}' column")
                            data[col] = data['close'] * 1000  # Synthetic volume
                        elif col.upper() in data.columns:
                            # Copy from uppercase version
                            print(f"  Copying {col.upper()} to {col}")
                            data[col] = data[col.upper()]
                
                # Ensure datetime index
                if not isinstance(data.index, pd.DatetimeIndex):
                    print("  Converting index to datetime")
                    data.index = pd.to_datetime(data.index)
                
                processed_data = await strategy_instance.preprocess_data(data)
                
                # Check processed data
                if processed_data.empty:
                    print("  Error: Preprocessing resulted in empty dataset")
                    return self._create_error_result(strategy_name, symbol, "Preprocessing resulted in empty dataset")
                    
                print(f"  Processed data shape: {processed_data.shape}")
            except Exception as e:
                print(f"  Error preprocessing data: {str(e)}")
                traceback.print_exc()
                return self._create_error_result(strategy_name, symbol, f"Preprocessing error: {str(e)}")
            
            # Train if necessary with enhanced handling
            if hasattr(strategy_instance, 'train') and not strategy_instance.is_trained:
                try:
                    print(f"  Training {strategy_name}...")
                    
                    # Make sure we have enough data for training
                    min_required_rows = 30  # Minimum rows needed for most algorithms
                    if len(processed_data) < min_required_rows:
                        print(f"  WARNING: Only {len(processed_data)} rows available for training, minimum {min_required_rows} recommended")
                    
                    # Special handling for LSTM which needs more data
                    if strategy_name == "LSTM" and len(processed_data) < 60:
                        print("  Adding synthetic data points to meet LSTM requirements")
                        # Add some synthetic data points to allow training
                        last_row = processed_data.iloc[-1]
                        for i in range(60 - len(processed_data)):
                            new_row = last_row.copy()
                            # Add slight variations to avoid exact duplicates
                            for col in processed_data.columns:
                                if col != 'date' and pd.api.types.is_numeric_dtype(processed_data[col]):
                                    new_row[col] *= (1 + np.random.normal(0, 0.001))
                            processed_data = pd.concat([processed_data, pd.DataFrame([new_row])], ignore_index=True)
                        print(f"  Expanded data to {len(processed_data)} rows for training")
                    
                    await strategy_instance.train(processed_data)
                    print(f"  Training completed successfully")
                except Exception as e:
                    print(f"  Training error: {str(e)}")
                    traceback.print_exc()
                    # For LSTM and complex strategies, training failure is critical
                    if strategy_name in ["LSTM", "Transformer"]:
                        return self._create_error_result(strategy_name, symbol, f"Training error: {str(e)}")
                    # Other strategies may run with defaults
            
            # Run backtest
            try:
                print(f"  Running backtest...")
                backtest_results = await strategy_instance.backtest(processed_data, initial_capital=self.initial_capital)
                
                # Extract results
                final_capital = backtest_results.get('final_capital', 0)
                total_return = backtest_results.get('total_return', 0)
                trades = backtest_results.get('trades', 0)
                
                print(f"  Backtest complete: ${final_capital:.2f} ({total_return*100:.2f}% return)")
                
                return {
                    "symbol": symbol,
                    "strategy": strategy_name,
                    "initial_capital": self.initial_capital,
                    "final_capital": final_capital,
                    "total_return": total_return,
                    "trades": trades,
                    "sharpe_ratio": backtest_results.get('sharpe_ratio', 0),
                    "max_drawdown": backtest_results.get('max_drawdown', 0),
                    "backtest_data": backtest_results.get('backtest_data', None)
                }
            except Exception as e:
                print(f"  Backtest error: {str(e)}")
                return self._create_error_result(strategy_name, symbol, f"Backtest error: {str(e)}")
                
        except Exception as e:
            print(f"Error testing {strategy_name} on {symbol}: {str(e)}")
            return self._create_error_result(strategy_name, symbol, str(e))
    
    def _create_error_result(self, strategy_name, symbol, error_msg):
        """Create a result entry for an error case."""
        return {
            "symbol": symbol,
            "strategy": strategy_name,
            "initial_capital": self.initial_capital,
            "final_capital": 0,
            "total_return": 0,
            "trades": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "error": error_msg
        }
    
    async def run_all_tests(self):
        """Run tests for all strategies on all symbols."""
        # Define symbols to test
        stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        cryptos = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'ADA-USD', 'DOT-USD']
        
        # Start with fewer symbols for quicker testing
        test_stocks = stocks[:3]  # First 3 stocks
        test_cryptos = cryptos[:2]  # First 2 cryptos
        
        all_symbols = test_stocks + test_cryptos
        print(f"Testing on symbols: {', '.join(all_symbols)}")
        
        # Initialize strategies to test
        strategies = []
        
        try:
            # Always try MovingAverage with smaller windows for limited data
            from app.strategies.moving_average import MovingAverageStrategy
            strategies.append(
                (MovingAverageStrategy(short_window=5, long_window=15), "MovingAverage")
            )
            strategies.append(
                (MovingAverageStrategy(short_window=5, long_window=15, use_ml=True), "MovingAverage_ML")
            )
            print("✓ Added MovingAverage strategies with reduced window sizes (5/15)")
        except Exception as e:
            print(f"Could not initialize MovingAverage strategy: {e}")
            traceback.print_exc()
        
        try:
            # Add LSTM with modified parameters for smaller datasets
            from app.strategies.lstm_predictor import LSTMPredictorStrategy
            strategies.append(
                (LSTMPredictorStrategy(
                    sequence_length=20,  # Reduced from 60 to work with less data
                    prediction_horizon=1,
                    epochs=10,  # Reduced training epochs
                    batch_size=8  # Smaller batch size for smaller datasets
                ), "LSTM")
            )
            print("✓ Added LSTM strategy with parameters optimized for smaller datasets")
        except Exception as e:
            print(f"Could not initialize LSTM strategy: {e}")
            traceback.print_exc()
        
        try:
            # Add LLM
            from app.strategies.llm_strategy import LLMStrategy
            strategies.append(
                (LLMStrategy(), "LLM")
            )
            print("✓ Added LLM strategy")
        except Exception as e:
            print(f"Could not initialize LLM strategy: {e}")
        
        try:
            # Add MSI if available
            from app.strategies.msi_strategy import MSIStrategy
            strategies.append(
                (MSIStrategy(), "MSI")
            )
            print("✓ Added MSI strategy")
        except Exception as e:
            print(f"MSI strategy not available: {e}")
        
        try:
            # Add Transformer if available
            from app.strategies.transformer_strategy import TransformerStrategy
            strategies.append(
                (TransformerStrategy(), "Transformer")
            )
            print("✓ Added Transformer strategy")
        except Exception as e:
            print(f"Transformer strategy not available: {e}")
        
        print(f"\nBeginning tests with {len(strategies)} strategies on {len(all_symbols)} symbols")
        print("This may take a few minutes...\n")
        
        # Run all tests - one at a time to prevent resource contention
        for symbol in all_symbols:
            for strategy_instance, strategy_name in strategies:
                result = await self.run_strategy_test(strategy_instance, strategy_name, symbol)
                self.results.append(result)
        
        return self.results
    
    def generate_report(self):
        """Generate reports from test results."""
        if not self.results:
            print("No results to report!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.results)
        
        # Clean up for display
        display_df = df.copy()
        if 'backtest_data' in display_df.columns:
            display_df = display_df.drop(columns=['backtest_data'])
        
        display_df['initial_capital'] = display_df['initial_capital'].apply(lambda x: f"${x:.2f}")
        display_df['final_capital'] = display_df['final_capital'].apply(lambda x: f"${x:.2f}")
        display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x*100:.2f}%")
        display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x*100:.2f}%")
        
        # Save raw results
        if 'backtest_data' in df.columns:
            # Convert pandas DataFrames to JSON
            for i, row in df.iterrows():
                if isinstance(row['backtest_data'], pd.DataFrame):
                    df.at[i, 'backtest_data'] = row['backtest_data'].to_dict()
        
        # Save to CSV
        df.to_csv('reports/all_strategies_comparison.csv', index=False)
        display_df.to_csv('reports/all_strategies_display.csv', index=False)
        
        # Print summary table
        print("\n===== STRATEGY COMPARISON RESULTS =====\n")
        
        # Sort by return (highest first)
        display_df = display_df.sort_values('total_return', ascending=False)
        
        # Format for display
        display_columns = ['symbol', 'strategy', 'initial_capital', 'final_capital', 
                         'total_return', 'sharpe_ratio', 'max_drawdown', 'trades']
        
        if 'error' in display_df.columns:
            display_df_clean = display_df[~display_df['error'].notna()]
            display_df_errors = display_df[display_df['error'].notna()]
            
            if not display_df_clean.empty:
                print(tabulate(display_df_clean[display_columns], 
                             headers=[col.capitalize().replace('_', ' ') for col in display_columns], 
                             tablefmt='grid'))
            
            if not display_df_errors.empty:
                print("\n===== STRATEGIES WITH ERRORS =====\n")
                print(tabulate(display_df_errors[['symbol', 'strategy', 'error']], 
                             headers=['Symbol', 'Strategy', 'Error'], 
                             tablefmt='grid'))
        else:
            print(tabulate(display_df[display_columns], 
                         headers=[col.capitalize().replace('_', ' ') for col in display_columns], 
                         tablefmt='grid'))
        
        # Find and display the best performers
        if 'error' in df.columns:
            success_df = df[~df['error'].notna()]
        else:
            success_df = df
            
        if not success_df.empty:
            # Find best strategy overall
            best_idx = success_df['total_return'].idxmax()
            best_strategy = success_df.loc[best_idx]
            
            print("\n===== BEST PERFORMING STRATEGIES =====\n")
            print(f"Best Overall: {best_strategy['strategy']} on {best_strategy['symbol']}")
            print(f"  Return: {best_strategy['total_return']*100:.2f}%")
            print(f"  Initial: ${best_strategy['initial_capital']:.2f}, Final: ${best_strategy['final_capital']:.2f}")
            
            # Best by asset class
            stocks = success_df[~success_df['symbol'].str.contains('-USD')]
            cryptos = success_df[success_df['symbol'].str.contains('-USD')]
            
            if not stocks.empty:
                best_stock_idx = stocks['total_return'].idxmax()
                best_stock = stocks.loc[best_stock_idx]
                print(f"\nBest Stock Strategy: {best_stock['strategy']} on {best_stock['symbol']}")
                print(f"  Return: {best_stock['total_return']*100:.2f}%")
            
            if not cryptos.empty:
                best_crypto_idx = cryptos['total_return'].idxmax()
                best_crypto = cryptos.loc[best_crypto_idx]
                print(f"\nBest Crypto Strategy: {best_crypto['strategy']} on {best_crypto['symbol']}")
                print(f"  Return: {best_crypto['total_return']*100:.2f}%")
            
            # Best by strategy type
            print("\nAverage Returns by Strategy:")
            avg_returns = success_df.groupby('strategy')['total_return'].mean()
            for strategy_name, avg_return in avg_returns.items():
                print(f"  {strategy_name}: {avg_return*100:.2f}%")
        
        print("\nResults saved to reports/all_strategies_comparison.csv")
        print("Run 'streamlit run strategy_dashboard.py' to view the interactive dashboard.")

async def main():
    """Main entry point."""
    print("\n===== MERCURIO AI STRATEGY COMPARISON =====\n")
    
    # Create tester instance
    tester = StrategyTester(initial_capital=2000, test_period_days=30)
    
    # Run all tests
    await tester.run_all_tests()
    
    # Generate report
    tester.generate_report()

if __name__ == "__main__":
    asyncio.run(main())
