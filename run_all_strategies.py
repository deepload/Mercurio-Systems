"""
Mercurio AI - Run All Strategies

This script tests all available strategies on sample data and generates a comparison report.
It uses Mercurio's built-in fallback mechanisms to work without requiring API keys.
"""
import asyncio
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from tabulate import tabulate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ensure reports directory exists
os.makedirs('reports', exist_ok=True)

async def test_strategy(strategy_class, strategy_name, symbol, initial_capital=2000):
    """Test a single strategy on a symbol and return results."""
    try:
        print(f"Testing {strategy_name} on {symbol}...")
        
        # Initialize the strategy
        strategy = strategy_class()
        
        # Get dates for last month
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Load data
        print(f"  Loading data for {symbol}...")
        data = await strategy.load_data(symbol, start_date, end_date)
        rows = len(data)
        print(f"  Loaded {rows} rows of data")
        
        if data.empty:
            return {
                "error": "No data returned from data service",
                "symbol": symbol,
                "strategy": strategy_name,
                "initial_capital": initial_capital,
                "final_capital": 0,
                "total_return": 0,
                "trades": 0
            }
        
        # Preprocess data
        print(f"  Preprocessing data...")
        processed_data = await strategy.preprocess_data(data)
        
        # Train if needed
        if hasattr(strategy, 'train') and not strategy.is_trained:
            print(f"  Training {strategy_name}...")
            try:
                await strategy.train(processed_data)
                print(f"  Training complete")
            except Exception as e:
                print(f"  Training error: {str(e)}")
                # Continue without training
        
        # Run backtest
        print(f"  Running backtest...")
        backtest_results = await strategy.backtest(processed_data, initial_capital=initial_capital)
        
        # Extract key metrics
        final_capital = backtest_results.get('final_capital', 0)
        total_return = backtest_results.get('total_return', 0)
        trades = backtest_results.get('trades', 0)
        
        print(f"  Backtest complete. Return: {total_return*100:.2f}%, Final: ${final_capital:.2f}")
        
        return {
            "symbol": symbol,
            "strategy": strategy_name,
            "initial_capital": initial_capital,
            "final_capital": final_capital,
            "total_return": total_return,
            "trades": trades
        }
        
    except Exception as e:
        print(f"Error testing {strategy_name} on {symbol}: {str(e)}")
        return {
            "error": str(e),
            "symbol": symbol,
            "strategy": strategy_name,
            "initial_capital": initial_capital,
            "final_capital": 0,
            "total_return": 0,
            "trades": 0
        }

async def main():
    """Run all available strategies and compare results."""
    print("\n===== MERCURIO AI STRATEGY COMPARISON =====\n")
    
    # Define stocks and cryptos to test
    stocks = ['AAPL', 'MSFT', 'GOOGL']
    cryptos = ['BTC-USD', 'ETH-USD'] 
    
    all_symbols = stocks + cryptos
    results = []
    
    # Import all strategies
    try:
        from app.strategies.moving_average import MovingAverageStrategy
        from app.strategies.lstm_predictor import LSTMPredictorStrategy
        from app.strategies.llm_strategy import LLMStrategy
    except Exception as e:
        print(f"Error importing strategies: {e}")
        return
    
    # Try to import optional strategies
    try:
        from app.strategies.msi_strategy import MSIStrategy
        has_msi = True
    except ImportError:
        has_msi = False
        print("MSI Strategy not available")
        
    try:
        from app.strategies.transformer_strategy import TransformerStrategy
        has_transformer = True
    except ImportError:
        has_transformer = False
        print("Transformer Strategy not available")
    
    # Define strategies to test
    strategies = [
        (MovingAverageStrategy, "MovingAverage"),
        (lambda: MovingAverageStrategy(use_ml=True), "MovingAverage_ML"),
        (LSTMPredictorStrategy, "LSTM"),
        (LLMStrategy, "LLM")
    ]
    
    # Add optional strategies if available
    if has_msi:
        strategies.append((MSIStrategy, "MSI"))
        
    if has_transformer:
        strategies.append((TransformerStrategy, "Transformer"))
    
    print(f"Testing {len(strategies)} strategies on {len(all_symbols)} symbols...\n")
    
    # Test each strategy on each symbol
    for symbol in all_symbols:
        for strategy_class, strategy_name in strategies:
            result = await test_strategy(strategy_class, strategy_name, symbol)
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Format for display
    display_df = df.copy()
    display_df['initial_capital'] = display_df['initial_capital'].apply(lambda x: f"${x:.2f}")
    display_df['final_capital'] = display_df['final_capital'].apply(lambda x: f"${x:.2f}")
    display_df['total_return'] = display_df['total_return'].apply(lambda x: f"{x*100:.2f}%")
    
    # Sort by return (descending)
    display_df = display_df.sort_values('total_return', ascending=False)
    
    # Save raw results to CSV
    df.to_csv('reports/all_strategies_raw.csv', index=False)
    display_df.to_csv('reports/all_strategies_comparison.csv', index=False)
    
    # Print results table
    print("\n===== STRATEGY COMPARISON RESULTS =====\n")
    print(tabulate(display_df, headers='keys', tablefmt='grid'))
    
    # Find best strategy overall
    if 'error' in df.columns:
        success_df = df[df['error'].isna()]
    else:
        success_df = df
        
    if not success_df.empty:
        best_strategy = success_df.loc[success_df['total_return'].idxmax()]
        print(f"\nBEST STRATEGY: {best_strategy['strategy']} on {best_strategy['symbol']}")
        print(f"Return: {best_strategy['total_return']*100:.2f}%")
        print(f"Initial: ${best_strategy['initial_capital']:.2f}, Final: ${best_strategy['final_capital']:.2f}")
    
    print("\nResults saved to reports/all_strategies_comparison.csv")
    print("\nRun 'streamlit run strategy_dashboard.py' to view the interactive dashboard.")

if __name__ == "__main__":
    asyncio.run(main())
